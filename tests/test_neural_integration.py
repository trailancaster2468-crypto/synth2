"""Real renders with the neural engine. Skipped unless onnxruntime and a voicebank are installed
(point ``SYNTH2_MODELS`` at the models folder). ``SYNTH2_TEST_BANK2`` may name a second models
folder holding another voicebank: it is then loaded and sung with as well."""
import hashlib
import importlib
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


def twinkle():
    p = parse_score(TWINKLE)
    return p, p.tracks[0]


def backend():
    return importlib.import_module("synth2.neural.backend")


def sing(sr=SR, t0=0.0, t1=None, note_range=None):
    p, tr = twinkle()
    t1 = p.tempo.beat_to_sec(tr.end) + 0.6 if t1 is None else t1
    with render_options(quality="draft"):
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


def test_full_render_length_report_and_credits(full):
    audio, rep = full
    p, tr = twinkle()
    assert audio.dtype == np.float32 and len(audio) == int(round((p.tempo.beat_to_sec(tr.end) + 0.6) * SR))
    assert np.isfinite(audio).all() and 0.02 < np.abs(audio).max() <= 1.0
    assert rep["engine"] == "neural" and rep["bank"] and rep["notes"] == len(tr.notes)
    assert rep["credits"] and rep["credits"] == neural.credits(rep["bank"])[:len(rep["credits"])]
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
    code = (
        "import sys, hashlib, importlib\n"
        f"sys.path.insert(0, {REPO!r})\n"
        "import numpy as np\n"
        "from synth2.score import parse_score\n"
        "from synth2.engine.backend import render_options\n"
        f"p = parse_score({TWINKLE!r}); tr = p.tracks[0]\n"
        "nb = importlib.import_module('synth2.neural.backend')\n"
        "with render_options(quality='draft'):\n"
        f"    x, _ = nb.get_backend().render_track(p, tr, {SR}, {t0}, {t1})\n"
        "print(hashlib.sha256(np.ascontiguousarray(x, np.float32).tobytes()).hexdigest())\n")
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=240, cwd=REPO,
                         env={**os.environ, "SYNTH2_NEURAL_CACHE": "0"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == digest(audio[a:a + int(round((t1 - t0) * SR))])


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
