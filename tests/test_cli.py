"""Tests for the command line interface (``synth2 ...``) and the bundled demos."""
from __future__ import annotations

import json
import os
import shutil
import struct
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import surface_fakes as fakes
from synth2 import cli
from synth2.ai.tools import Session, call_tool
from synth2.project import Project

ROOT = Path(__file__).resolve().parents[1]


def _env(pythonpath):
    return dict(os.environ, PYTHONPATH=str(pythonpath) + os.pathsep + os.environ.get("PYTHONPATH", ""))


# ================================================================== director

def _chord_midi() -> bytes:
    """A type-0 MIDI file holding one C major chord (C4 E4 G4, one beat)."""
    trk = bytes([0x00, 0x90, 60, 100, 0x00, 0x90, 64, 100, 0x00, 0x90, 67, 100,
                 0x83, 0x60, 0x80, 60, 0, 0x00, 0x80, 64, 0, 0x00, 0x80, 67, 0, 0x00, 0xFF, 0x2F, 0x00])
    return b"MThd" + struct.pack(">IHHH", 6, 0, 1, 480) + b"MTrk" + struct.pack(">I", len(trk)) + trk


def _fake_director(script):
    """A Director stand-in: each chat() runs the next scripted step (a callable on the session, or an exception)."""

    class FakeDirector:
        def __init__(self, session, model=None):
            self.session = session

        def chat(self, msg):
            step = script.pop(0)
            if isinstance(step, BaseException):
                raise step
            if step is not None:
                step(self.session)
            return {"reply": f"re: {msg}", "actions": []}

    return FakeDirector


def _write3(session):
    call_tool(session, "write_melody", {"score": "@ 4 C4 q a D4 q b E4 q c"})


def _director_args(project, message=None):
    return SimpleNamespace(project=str(project), message=message, model=None)


def test_director_never_overwrites_an_imported_file(tmp_path, capsys):
    mid = tmp_path / "chords.mid"
    mid.write_bytes(_chord_midi())
    # a question that edits nothing writes nothing
    assert cli._director_main(_director_args(mid, "what key is this in?"), _fake_director([None]), Session) == 0
    assert mid.read_bytes() == _chord_midi()
    assert not (tmp_path / "chords.s2p.json").exists()
    # edits go to <stem>.s2p.json next to the import; the source file keeps every chord note
    assert cli._director_main(_director_args(mid, "add a melody"), _fake_director([_write3]), Session) == 0
    assert mid.read_bytes() == _chord_midi()
    assert "changes are saved to" in capsys.readouterr().err
    saved = Project.load(tmp_path / "chords.s2p.json")
    assert [n.lyric for t in saved.tracks for n in t.notes][-3:] == ["a", "b", "c"]


def test_director_native_and_new_project_files_are_saved_in_place(tmp_path):
    new = tmp_path / "new.s2p.json"
    assert cli._director_main(_director_args(new, "hi"), _fake_director([None]), Session) == 0
    assert new.exists()  # a project file named on the command line is created
    assert cli._director_main(_director_args(new, "write"), _fake_director([_write3]), Session) == 0
    assert len(Project.load(new).tracks[0].notes) == 3


def test_director_keeps_the_session_after_api_errors_and_ctrl_c(tmp_path, monkeypatch, capsys):
    proj = tmp_path / "work.s2p.json"
    inputs = iter(["write", "boom", "slow", "more"])

    def fake_input(prompt=""):
        try:
            return next(inputs)
        except StopIteration:
            raise KeyboardInterrupt from None  # Ctrl-C at the prompt ends the session

    monkeypatch.setattr("builtins.input", fake_input)
    script = [_write3, RuntimeError("overloaded_error"), KeyboardInterrupt(),
              lambda s: call_tool(s, "set_tempo", {"bpm": 90})]
    assert cli._director_main(_director_args(proj), _fake_director(script), Session) == 0
    err = capsys.readouterr().err
    assert "error: RuntimeError: overloaded_error" in err and "(interrupted)" in err
    saved = Project.load(proj)  # work from before and after the failures was saved
    assert len(saved.tracks[0].notes) == 3 and saved.bpm == 90
    # one-shot mode: the failure is reported with a non-zero exit code, earlier work is still saved
    proj2 = tmp_path / "one.s2p.json"
    Session(path=proj2).call("save_project", {})
    assert cli._director_main(_director_args(proj2, "go"), _fake_director([RuntimeError("down")]), Session) == 1
    # an exception that is not handled per message still saves before it propagates
    monkeypatch.setattr("builtins.input", lambda prompt="": "x")

    class Boom(BaseException):
        pass

    def write_then_fail(session):
        _write3(session)
        raise Boom

    with pytest.raises(Boom):
        cli._director_main(_director_args(proj2), _fake_director([write_then_fail]), Session)
    assert len(Project.load(proj2).tracks[0].notes) == 3


# ================================================================== mcp

def test_mcp_command_exit_code_and_autosave(tmp_path):
    bad = tmp_path / "bad.s2p.json"
    bad.write_text("{not json")
    r = subprocess.run([sys.executable, "-m", "synth2", "mcp", "--project", str(bad)], input=b"",
                       capture_output=True, cwd=str(tmp_path), env=_env(ROOT), timeout=60)
    assert r.returncode == 2 and b"could not load" in r.stderr
    proj = tmp_path / "song.s2p.json"
    call = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
            "params": {"name": "write_melody", "arguments": {"score": "@ 4 C4 q a D4 q b"}}}
    r = subprocess.run([sys.executable, "-m", "synth2", "mcp", "--project", str(proj), "--autosave"],
                       input=(json.dumps(call) + "\n").encode(), capture_output=True, cwd=str(tmp_path),
                       env=_env(ROOT), timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["result"]["isError"] is False
    assert len(Project.load(proj).tracks[0].notes) == 2


# ================================================================== render

TWO_TRACKS = """\
title: T
tempo: 120
time: 4/4
track: Lead voice=aurora style=flat
r w | C4 q la D4 q la E4 q la F4 q la | G4 h la E4 h la | D4 q la F4 q la E4 h la
track: Low voice=atlas style=flat gain=-4
r w | C3 w la
"""


@pytest.fixture
def score(tmp_path):
    p = tmp_path / "t.s2s"
    p.write_text(TWO_TRACKS, encoding="utf-8")
    return p


def test_render_rejects_unknown_tracks_and_empty_ranges(score, tmp_path, capsys):
    out = tmp_path / "n.wav"
    for extra in (["--track", "Nope"], ["--track", "7"], ["--start", "4", "--end", "2"],
                  ["--start", "4", "--end", "4"]):
        with pytest.raises(SystemExit) as exc:
            cli.main(["render", str(score), "-o", str(out), "-q", *extra])
        assert exc.value.code == 2
        assert not out.exists()
    err = capsys.readouterr().err
    assert "no track 'Nope'" in err and "out of range" in err and "--end must be greater than --start" in err


def test_render_tracks_by_name_and_index_and_check_only_judges_the_rendered_range(score, tmp_path, capsys):
    out = tmp_path / "r.wav"
    assert cli.main(["render", str(score), "-o", str(out), "-q", "--track", "low", "--track", "0",
                     "--start", "4", "--end", "8", "--check"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert out.exists() and sorted(t["name"] for t in report["tracks"]) == ["Lead", "Low"]
    for tr in report["pitch_accuracy"]["tracks"]:
        for w in tr["worst"] + tr["unvoiced_notes"]:
            assert 4 <= w["beat"] < 8, w


def test_check_filter_matches_the_analyze_render_tool():
    pa = {"tracks": [{"worst": [{"beat": 3.5}, {"beat": 4}, {"beat": 7.9}, {"beat": 8}, {"beat": 10}],
                      "unvoiced_notes": [{"beat": 10}, {"beat": 5}]}]}
    out = cli._filter_to_range(json.loads(json.dumps(pa)), 4, 8)
    assert [w["beat"] for w in out["tracks"][0]["worst"]] == [4, 7.9]
    assert [w["beat"] for w in out["tracks"][0]["unvoiced_notes"]] == [5]
    assert cli._filter_to_range(json.loads(json.dumps(pa)), None, None) == pa
    assert [w["beat"] for w in cli._filter_to_range(json.loads(json.dumps(pa)), None, 4)["tracks"][0]["worst"]] \
        == [3.5]


# ================================================================== demo

def test_demo_scores_match_the_examples():
    from synth2 import demo

    assert set(demo.SCORES) == set(demo.DEMOS)
    for name, text in demo.SCORES.items():
        assert (ROOT / "examples" / f"{name}.s2s").read_text(encoding="utf-8") == text, name


def test_demo_works_from_an_installed_package_without_the_repository(tmp_path):
    site = tmp_path / "site"
    shutil.copytree(ROOT / "synth2", site / "synth2", ignore=shutil.ignore_patterns("__pycache__"))
    code = ("from synth2.demo import DEMOS, build_demo\n"
            "print({n: [len(t.notes) for t in build_demo(n).project.tracks] for n in DEMOS})")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, cwd=str(tmp_path), env=_env(site),
                       timeout=120)
    assert r.returncode == 0, r.stderr.decode()
    counts = eval(r.stdout)  # noqa: S307 - our own subprocess output
    assert len(counts["neon_hearts"]) == 3 and len(counts["amazing_grace"]) == 2
    assert all(n > 0 for tracks in counts.values() for n in tracks)


# ================================================================== engines, quality, credits

def _run(argv) -> int:
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


@pytest.fixture
def auto_engine(monkeypatch):
    """No engine or quality pinned by the environment (the suite pins the formant engine)."""
    monkeypatch.delenv("SYNTH2_ENGINE", raising=False)
    monkeypatch.delenv("SYNTH2_QUALITY", raising=False)


def test_engines_without_models_says_how_to_install(auto_engine, monkeypatch, tmp_path, capsys):
    base = fakes.empty_models(monkeypatch, tmp_path)
    assert _run(["engines"]) == 0
    out = capsys.readouterr().out
    assert "formant  available" in out and "neural   not available" in out
    assert "synth2 models install" in out and "default (auto) on this machine: formant" in out
    assert "voicebanks: none installed" in out
    assert _run(["engines", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert [e["name"] for e in info["engines"]] == ["formant", "neural"]
    assert info["engines"][1]["available"] is False and "synth2 models install" in info["engines"][1]["install_hint"]
    assert info["neural"]["models_dir"] == str(base) and info["default_engine"] == "formant"
    assert info["qualities"] == ["draft", "normal", "high"] and info["default_quality"] == "normal"


def test_engines_lists_voicebanks_credits_and_licence_notes(auto_engine, monkeypatch, tmp_path, capsys):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    assert _run(["engines"]) == 0
    out = capsys.readouterr().out
    assert "neural   available" in out and "default (auto) on this machine: neural" in out
    assert "fakebank" in out and "languages: en, ja" in out and "terms: Credit the voicebank." in out
    assert f"credit: {fakes.BANK_CREDIT}" in out and f"- {fakes.VOCODER_CREDIT}" in out
    assert "commercial use stated: no" in out and "commercial use is not allowed" in out
    monkeypatch.setenv("SYNTH2_ENGINE", "formant")
    monkeypatch.setenv("SYNTH2_QUALITY", "draft")
    assert _run(["engines"]) == 0
    out = capsys.readouterr().out
    assert "default (auto) on this machine: formant  [SYNTH2_ENGINE=formant; SYNTH2_QUALITY=draft]" in out
    assert "draft (default)" in out


def test_render_with_an_unavailable_or_misspelt_engine_exits_2_without_a_traceback(
        auto_engine, score, monkeypatch, tmp_path, capsys):
    fakes.empty_models(monkeypatch, tmp_path)
    out = tmp_path / "n.wav"
    assert _run(["render", str(score), "-o", str(out), "-q", "--engine", "neural"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("error: the neural engine was requested but is not available")
    assert "synth2 models install" in err and "Traceback" not in err and len(err.strip().splitlines()) == 1
    for extra, text in ((["--engine", "nueral"], "unknown engine 'nueral'"),
                        (["--engine", "formant:x"], "unknown engine"),
                        (["--quality", "best"], "invalid choice: 'best'")):
        assert _run(["render", str(score), "-o", str(out), "-q", *extra]) == 2
        err = capsys.readouterr().err
        assert text in err and "Traceback" not in err
    assert not out.exists()
    # a project that asks for the neural engine itself, and a file that is not there
    wants = tmp_path / "wants.s2s"
    wants.write_text("engine: neural\n" + TWO_TRACKS, encoding="utf-8")
    assert _run(["render", str(wants), "-o", str(out), "-q"]) == 2
    assert "synth2 models install" in capsys.readouterr().err
    assert _run(["render", str(tmp_path / "missing.s2s"), "-o", str(out), "-q"]) == 2
    assert "could not load" in capsys.readouterr().err
    assert not out.exists()
    # the formant engine still renders, by name and by default
    assert _run(["render", str(score), "-o", str(out), "-q", "--engine", "formant", "--end", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["engines"] == ["formant"] and out.exists()
    assert _run(["render", str(wants), "-o", str(out), "-q", "--engine", "formant", "--end", "4"]) == 0
    assert json.loads(capsys.readouterr().out)["engines"] == ["formant"]


def test_render_prints_engines_credits_licence_notes_and_warnings(auto_engine, score, monkeypatch, tmp_path, capsys):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    out = tmp_path / "n.wav"
    assert _run(["render", str(score), "-o", str(out), "--end", "4", "--quality", "draft"]) == 0
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    assert report["engines"] == ["neural"] and report["quality"] == "draft"
    assert report["credits"] == [fakes.BANK_CREDIT, fakes.VOCODER_CREDIT]
    assert report["licence_notes"] == ["Fakevoc (vocoder): CC BY-NC 4.0 - commercial use is not allowed"]
    assert all("alignment" not in t["neural"] and "phonemes" not in t for t in report["tracks"])
    assert {q for _, q in fakes.CALLS} == {"draft"}                       # the quality reached the engine
    assert "engines: neural (voicebank fakebank)" in cap.err
    assert "credits (show these when you publish):" in cap.err and f"  - {fakes.VOCODER_CREDIT}" in cap.err
    assert "licence notes:" in cap.err and "commercial use is not allowed" in cap.err
    # --engine overrides the automatic choice; --quiet keeps the summary out of stderr
    assert _run(["render", str(score), "-o", str(out), "-q", "--end", "4", "--engine", "formant"]) == 0
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    assert report["engines"] == ["formant"] and report["credits"] == [] and report["licence_notes"] == []
    assert cap.err == ""
    assert _run(["render", str(score), "-o", str(out), "-q", "--end", "4", "--engine", "neural:fakebank"]) == 0
    assert json.loads(capsys.readouterr().out)["tracks"][0]["bank"] == "fakebank"
    assert _run(["render", str(score), "-o", str(out), "-q", "--engine", "neural:other"]) == 2
    assert "voicebank 'other' is not installed" in capsys.readouterr().err


def test_render_reports_sound_credits_and_fallbacks(score, monkeypatch, tmp_path, capsys):
    fakes.install_fake_sound(monkeypatch, tmp_path)
    proj = tmp_path / "band.s2p.json"
    session = Session.open(score)
    session.project.backing = fakes.backing()
    call_tool(session, "save_project", {"path": str(proj)})
    out = tmp_path / "band.wav"
    assert _run(["render", str(proj), "-o", str(out), "--end", "4"]) == 0
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    assert report["credits"] == [fakes.SOUND_CREDIT] and f"  - {fakes.SOUND_CREDIT}" in cap.err
    assert not [w for w in report["warnings"] if "plays instead" in w]
    session.project.backing = fakes.backing("not-installed-set")
    call_tool(session, "save_project", {"path": str(proj)})
    assert _run(["render", str(proj), "-o", str(out), "--end", "4"]) == 0
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    fallback = [w for w in report["warnings"] if "plays instead" in w]
    assert len(fallback) == 1 and "synth2 sounds install not-installed-set" in fallback[0]
    assert report["credits"] == [] and "warnings:" in cap.err and "plays instead" in cap.err


def test_voices_and_info_show_the_neural_side(auto_engine, score, monkeypatch, tmp_path, capsys):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    assert _run(["voices"]) == 0
    out = capsys.readouterr().out
    assert "aurora" in out and "neural engine: voicebank fakebank, formant shift 0, level +2 dB" in out
    assert "tone +12" in out                                   # the low voices are sung an octave up in the model
    assert _run(["voices", "--json"]) == 0
    voices = {v["id"]: v for v in json.loads(capsys.readouterr().out)}
    assert voices["atlas"]["neural"]["tone"] == 12 and voices["atlas"]["neural"]["voicebank"] == "fakebank"
    wants = tmp_path / "wants.s2s"
    wants.write_text(TWO_TRACKS.replace("track: Low", "track: Low engine=formant role=harmony"), encoding="utf-8")
    assert _run(["info", str(wants)]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["engine"] == "auto"
    engines = [(e["track"], e["asks_for"], e["engine"]) for e in info["engines"]]
    assert engines == [("Lead", "auto", "neural"), ("Low", "formant", "formant")]
    assert info["tracks"][1]["role"] == "harmony"
    fakes.empty_models(monkeypatch, tmp_path)
    monkeypatch.setattr("synth2.neural.available", lambda bank=None: (False, "no voicebank is installed"))
    assert _run(["voices"]) == 0
    out = capsys.readouterr().out
    assert "no voicebank installed" in out and "the neural engine is not available on this machine" in out


def test_demo_takes_engine_and_quality(auto_engine, monkeypatch, tmp_path, capsys):
    from synth2 import demo

    fakes.install_fake_neural(monkeypatch, tmp_path)
    monkeypatch.setattr(demo, "DEMOS", {"sakura": []})
    monkeypatch.setitem(demo.SCORES, "sakura", "tempo: 120\ntrack: Lead voice=nova lang=ja\nr q A4 q さ A4 q く")
    out = tmp_path / "demo"
    assert _run(["demo", "-o", str(out), "--engine", "neural", "--quality", "draft"]) == 0
    cap = capsys.readouterr()
    assert cap.out.strip() == str(out / "sakura.wav") and (out / "sakura.s2p.json").exists()
    assert "engines: neural (voicebank fakebank)" in cap.err and fakes.BANK_CREDIT in cap.err
    assert fakes.CALLS == [("Lead", "draft")]
    assert _run(["demo", "-o", str(out), "--engine", "neural:nope"]) == 2
    assert "voicebank 'nope' is not installed" in capsys.readouterr().err


def test_studio_mcp_and_director_take_a_default_engine(auto_engine, monkeypatch, tmp_path, capsys):
    import synth2.ai.mcp_server as mcp_server
    import synth2.server.app as app

    fakes.empty_models(monkeypatch, tmp_path)
    for argv in (["studio", "--engine", "neural"], ["mcp", "--engine", "neural"],
                 ["director", "hi", "--engine", "neural"]):
        assert _run(argv) == 2
        assert "synth2 models install" in capsys.readouterr().err
    seen = {}
    monkeypatch.setattr(app, "run", lambda **kw: seen.update(studio=(os.environ.get("SYNTH2_ENGINE"),
                                                                     os.environ.get("SYNTH2_QUALITY"))))
    monkeypatch.setattr(mcp_server, "main", lambda argv: seen.update(mcp=os.environ.get("SYNTH2_ENGINE")) or 0)
    monkeypatch.setenv("SYNTH2_ENGINE", "")
    monkeypatch.setenv("SYNTH2_QUALITY", "")                    # restored after the test
    assert _run(["studio", "--engine", "formant", "--quality", "draft"]) == 0
    assert _run(["mcp", "--engine", "formant"]) == 0
    assert seen == {"studio": ("formant", "draft"), "mcp": "formant"}


# ================================================================== with installed models
# Real renders: skipped unless onnxruntime and a voicebank are installed (SYNTH2_MODELS names the folder).

def _neural_installed() -> tuple[bool, str]:
    from synth2 import neural

    return neural.available()


needs_models = pytest.mark.skipif(not _neural_installed()[0],
                                  reason=f"neural engine unavailable: {_neural_installed()[1]}")
REAL_SCORE = "tempo: 120\ntrack: Lead voice=aurora style=pop lang=en\nr q  C4 q hello  E4 q world  r q\n"


@needs_models
def test_real_neural_render_from_the_command_line(auto_engine, monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "0")
    score = tmp_path / "real.s2s"
    score.write_text(REAL_SCORE, encoding="utf-8")
    out = tmp_path / "real.wav"
    assert _run(["engines", "--json"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["neural"]["available"] and info["neural"]["banks"] and info["credits"]
    assert _run(["render", str(score), "-o", str(out), "--engine", "neural", "--quality", "draft"]) == 0
    cap = capsys.readouterr()
    report = json.loads(cap.out)
    assert report["engines"] == ["neural"] and report["quality"] == "draft" and out.stat().st_size > 10000
    assert report["credits"] == info["credits"] and report["tracks"][0]["bank"] == info["neural"]["default_bank"]
    assert "credits (show these when you publish):" in cap.err
    for line in report["credits"]:
        assert f"  - {line}" in cap.err
