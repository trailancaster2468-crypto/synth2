"""Real renders with the neural engine. Skipped unless onnxruntime and a voicebank are installed
(point ``SYNTH2_MODELS`` at the models folder). ``SYNTH2_TEST_BANK2`` may name a second models
folder holding another voicebank: it is then loaded and sung with as well.

The phrase cache is off except in the tests that are about it, and those keep it in a folder of
the test run."""
import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from synth2 import neural
from synth2.engine.backend import render_options
from synth2.engine.renderer import render_project
from synth2.score import parse_score

pytestmark = pytest.mark.skipif(not neural.available()[0], reason=f"neural engine unavailable: {neural.available()[1]}")

REPO = str(Path(__file__).resolve().parent.parent)
SR = 44100
TWINKLE = """title: Twinkle
tempo: 120
track: Lead voice=aurora style=pop lang=en
C4 q twinkle  C4 q +  G4 q twinkle  G4 q + | A4 q little  A4 q +  G4 q star  r q |
F4 q how  F4 q I  E4 q wonder  E4 q + | D4 q what  D4 q you  C4 h are
"""
WINDOW = (3.9, 8.4)  # seconds: the second phrase
FIRST = (0.0, 3.7)  # seconds: the first phrase


def twinkle():
    p = parse_score(TWINKLE)
    return p, p.tracks[0]


def backend():
    from synth2.neural import renderer

    return renderer


def sing(sr=SR, t0=0.0, t1=None, note_range=None, edit=None, quality="draft"):
    """Render the song (after ``edit(project, track)`` changed it) through the neural backend."""
    p, tr = twinkle()
    if edit is not None:
        edit(p, tr)
    t1 = p.tempo.beat_to_sec(tr.end) + 0.6 if t1 is None else t1
    with render_options(quality=quality):
        return backend().get_backend().render_track(p, tr, sr, t0, t1, note_range)


def digest(x) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest()


@pytest.fixture(scope="module", autouse=True)
def no_cache():
    """Every render here runs the models: a cache hit would prove nothing about determinism."""
    old = os.environ.get("SYNTH2_NEURAL_CACHE")
    os.environ["SYNTH2_NEURAL_CACHE"] = "0"
    yield
    if old is None:
        os.environ.pop("SYNTH2_NEURAL_CACHE", None)
    else:
        os.environ["SYNTH2_NEURAL_CACHE"] = old


@pytest.fixture(scope="module")
def full(no_cache):
    return sing()


@pytest.fixture(scope="module")
def warm(full, tmp_path_factory):
    """``(folder, audio, report)``: a phrase cache folder and the render of the song that filled it."""
    root = tmp_path_factory.mktemp("cache")
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SYNTH2_CACHE", str(root))
        mp.setenv("SYNTH2_NEURAL_CACHE", "1")
        audio, rep = sing()
    return root, audio, rep


@pytest.fixture
def cached(warm, monkeypatch):
    """Switches the phrase cache on, in the folder that holds the song; returns the cache module."""
    from synth2.neural import cache

    monkeypatch.setenv("SYNTH2_CACHE", str(warm[0]))
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "1")
    return cache


def in_another_process(cache_folder=None) -> tuple[str, int]:
    """``(sha256 of the audio, phrases served from the cache)`` of ``WINDOW`` rendered by a new
    interpreter, with the phrase cache in ``cache_folder`` or off."""
    t0, t1 = WINDOW
    code = (
        "import sys, hashlib\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "import numpy as np\n"
        "from synth2.score import parse_score\n"
        "from synth2.engine.backend import render_options\n"
        "from synth2.neural import renderer\n"
        f"p = parse_score({TWINKLE!r}); tr = p.tracks[0]\n"
        "with render_options(quality='draft'):\n"
        f"    x, rep = renderer.get_backend().render_track(p, tr, {SR}, {t0}, {t1})\n"
        "print(hashlib.sha256(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest(), rep['neural']['cached'])\n")
    env = {**os.environ, "SYNTH2_NEURAL_CACHE": "1" if cache_folder else "0"}
    if cache_folder:
        env["SYNTH2_CACHE"] = str(cache_folder)
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=240, cwd=REPO,
                         env=env)
    assert out.returncode == 0, out.stderr[-2000:]
    sha, hits = out.stdout.strip().splitlines()[-1].split()
    return sha, int(hits)


def test_full_render_length_report_and_credits(full):
    audio, rep = full
    p, tr = twinkle()
    assert audio.dtype == np.float32 and len(audio) == int(round((p.tempo.beat_to_sec(tr.end) + 0.6) * SR))
    assert np.isfinite(audio).all() and 0.02 < np.abs(audio).max() <= 1.0
    assert rep["engine"] == "neural" and rep["bank"] and rep["notes"] == len(tr.notes)
    assert rep["credits"] and rep["credits"] == neural.credits(rep["bank"])
    assert rep["neural"]["phrases"] == 2 and rep["neural"]["steps"] == 5 and rep["neural"]["failed"] == 0
    # the rest between the phrases is silent and the notes are not
    assert np.abs(audio[int(3.66 * SR):int(3.72 * SR)]).max() < 1e-4
    assert np.sqrt(np.mean(audio[int(0.2 * SR):int(3.0 * SR)] ** 2)) > 0.005


def test_partial_render_equals_the_full_render(full):
    audio, _ = full
    t0, t1 = WINDOW
    p, tr = twinkle()
    part, rep = sing(t0=t0, t1=t1, note_range=(p.tempo.sec_to_beat(t0), p.tempo.sec_to_beat(t1)))
    a = int(round(t0 * SR))
    assert len(part) == int(round((t1 - t0) * SR)) and rep["neural"]["phrases"] == 1
    assert np.array_equal(part, audio[a:a + len(part)])  # this is also a second render: same noise, same audio
    assert rep["notes"] < len(tr.notes)


def test_another_process_renders_the_same_audio(full):
    audio, _ = full
    t0, t1 = WINDOW
    a = int(round(t0 * SR))
    assert in_another_process() == (digest(audio[a:a + int(round((t1 - t0) * SR))]), 0)


@pytest.mark.parametrize("sr", [22050, 48000])
def test_other_sample_rates(full, sr):
    audio, _ = full
    t0, t1 = WINDOW
    part, _ = sing(sr=sr, t0=t0, t1=t1)
    assert part.dtype == np.float32 and len(part) == int(round((t1 - t0) * sr))
    ref = audio[int(round(t0 * SR)):int(round(t1 * SR))]
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    assert rms(part) == pytest.approx(rms(ref), rel=0.03)
    # the same waveform on the other grid: compare at a few shared instants (every 0.1 s is on both grids)
    k = np.arange(1, int((t1 - t0) * 10) - 1)
    assert np.abs(part[k * sr // 10] - ref[k * SR // 10]).max() < (0.05 if sr < SR else 0.01)


def test_render_project_reports_engine_and_credits():
    p, _ = twinkle()
    res = render_project(p, engine="neural", quality="draft", end_beat=4.0)
    assert res.report["engines"] == ["neural"] and res.report["credits"]
    assert res.report["tracks"][0]["engine"] == "neural" and res.report["tracks"][0]["bank"]
    assert np.abs(res.mono()).max() > 0.01


# ---- the phrase cache ------------------------------------------------------------------------------
MODELS = {"pitch", "variance", "acoustic", "vocoder"}  # report["neural"]["seconds"] keys of the models that sing


def test_a_second_render_is_served_from_the_phrase_cache(full, warm, cached, monkeypatch):
    audio, _ = full
    root, first, rep = warm
    assert rep["neural"]["cached"] == 0 and MODELS <= set(rep["neural"]["seconds"])  # the render that filled it
    assert np.array_equal(first, audio)  # the cache does not change what is sung
    again, rep = sing()
    assert rep["neural"]["cached"] == rep["neural"]["phrases"] == 2 and np.array_equal(again, audio)
    assert not MODELS & set(rep["neural"]["seconds"])  # only the phrases' layout was computed
    monkeypatch.setattr(cached, "_default", None)  # nothing in memory: the entries are read from the folder
    cold, rep = sing()
    assert rep["neural"]["cached"] == 2 and np.array_equal(cold, audio)
    assert len(list((root / "phrases").glob("*.npy"))) >= 4  # per phrase: the audio and a note about its pitch
    # another process finds the same entries under the same keys
    t0, t1 = WINDOW
    a = int(round(t0 * SR))
    assert in_another_process(root) == (digest(audio[a:a + int(round((t1 - t0) * SR))]), 1)


@pytest.mark.parametrize("params,sr,tolerance", [
    ({}, SR, 0.0),
    ({"neural.fx.alien": 1.0, "neural.fx.double": 0.6, "neural.fx.crush": 0.5}, SR, 0.0),
    ({"neural.fx.robot": 1.0, "neural.fx.radio": 1.0}, SR, 1e-4),  # a comb with feedback and a filter
    ({}, 48000, 1e-6),
    ({"neural.fx.alien": 0.6, "neural.fx.robot": 0.5}, 48000, 1e-4),
], ids=["dry", "alien+double+crush", "robot+radio", "dry-48k", "alien+robot-48k"])
def test_cached_partial_renders_equal_the_full_render(warm, cached, params, sr, tolerance):
    edit = lambda p, tr: tr.voice_params.update(params)
    whole, rep = sing(sr=sr, edit=edit)
    assert rep["neural"]["cached"] == 2 and not rep["warnings"]  # effects and the sample rate come after the cache
    if sr == SR:
        assert np.array_equal(whole, warm[1]) == (not params)
    for t0, t1 in (WINDOW, (1.0, 1.25), (3.0, 4.6), (7.9, 8.6)):
        part, rep = sing(sr=sr, t0=t0, t1=t1, edit=edit)
        a = int(round(t0 * sr))
        assert len(part) == int(round((t1 - t0) * sr)) and rep["neural"]["cached"] == rep["neural"]["phrases"]
        assert np.abs(part - whole[a:a + len(part)]).max() <= tolerance, (t0, t1)


def breathy(amount):
    def edit(p, tr):
        for beat, value in ((0.0, 0.0), (2.0, amount), (6.0, 0.0)):
            tr.curve("breathiness").add(beat, value)
    return edit


@pytest.mark.parametrize("edit,quality", [
    (lambda p, tr: setattr(tr.notes[1], "pitch", tr.notes[1].pitch + 2), "draft"),
    (lambda p, tr: setattr(tr.notes[2], "lyric", "wrinkle"), "draft"),
    (breathy(0.9), "draft"),
    (lambda p, tr: tr.voice_params.update({"neural.gender": 0.4}), "draft"),
    (None, "normal"),
], ids=["note", "lyric", "curve-point", "voice-parameter", "quality"])
def test_the_cache_serves_nothing_stale_after_an_edit(warm, cached, monkeypatch, edit, quality):
    t0, t1 = FIRST
    base = warm[1][:int(round(t1 * SR))]
    if edit is not None and quality == "draft" and edit.__name__ == "edit":
        # the curve exists already, with another value at one point, and that render is in the cache
        before, rep = sing(t0=t0, t1=t1, edit=breathy(0.5))
        assert not np.array_equal(before, base)
        base = before
    new, rep = sing(t0=t0, t1=t1, edit=edit, quality=quality)
    assert rep["neural"]["cached"] == 0 and rep["neural"]["phrases"] == 1  # the phrase is sung again
    assert not np.array_equal(new, base)
    again, rep = sing(t0=t0, t1=t1, edit=edit, quality=quality)
    assert rep["neural"]["cached"] == 1 and np.array_equal(again, new)
    old, rep = sing(t0=t0, t1=t1)  # the entry of the song as it was is still there, for the song as it was
    assert rep["neural"]["cached"] == 1 and np.array_equal(old, warm[1][:len(old)])
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "0")
    fresh, rep = sing(t0=t0, t1=t1, edit=edit, quality=quality)
    assert rep["neural"]["cached"] == 0 and np.array_equal(new, fresh)  # exactly what a render without a cache gives


def test_level_and_loudness_changes_reuse_the_cached_phrases(warm, cached):
    t0, t1 = FIRST
    base = warm[1][:int(round(t1 * SR))]
    quiet, rep = sing(t0=t0, t1=t1, edit=lambda p, tr: tr.voice_params.update({"neural.level_db": -6.0}))
    assert rep["neural"]["cached"] == 1 and np.allclose(quiet, base * 10 ** (-6 / 20), atol=1e-6)

    def ride(p, tr):
        tr.curve("loudness").add(0.0, 0.0)
        tr.curve("loudness").add(4.0, -12.0)
    faded, rep = sing(t0=t0, t1=t1, edit=ride)
    assert rep["neural"]["cached"] == 1 and not np.array_equal(faded, base)
    rms = lambda x: float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))
    late = slice(int(2.2 * SR), int(3.0 * SR))
    assert rms(faded[late]) < 0.5 * rms(base[late]) and rms(faded[:SR // 2]) > 0.8 * rms(base[:SR // 2])


@pytest.mark.skipif(not os.environ.get("SYNTH2_TEST_BANK2"), reason="SYNTH2_TEST_BANK2 names no second models folder")
def test_a_second_voicebank_loads_and_sings(monkeypatch):
    nb = backend()
    monkeypatch.setenv("SYNTH2_MODELS", os.environ["SYNTH2_TEST_BANK2"])
    neural.refresh()
    try:
        ok, why = neural.available()
        assert ok, why
        bank = nb.load_bank(None)
        assert bank.summary()["sample_rate"] > 0 and bank.has("SP")
        audio, rep = sing(t0=0.0, t1=3.3)
        assert rep["bank"] == bank.id and rep["neural"]["failed"] == 0 and rep["neural"]["phrases"] == 1
        assert len(audio) == int(round(3.3 * SR)) and np.sqrt(np.mean(audio ** 2)) > 0.005
        again, _ = sing(t0=0.0, t1=3.3)
        assert np.array_equal(audio, again)
        modes = sorted(bank.modes)
        if modes:
            p, tr = twinkle()
            tr.voice_params[f"neural.mode.{modes[-1]}"] = 1.0
            with render_options(quality="draft"):
                other, _ = nb.get_backend().render_track(p, tr, SR, 0.0, 3.3)
            assert np.isfinite(other).all() and np.sqrt(np.mean(other ** 2)) > 0.005
    finally:
        monkeypatch.undo()
        neural.refresh()
