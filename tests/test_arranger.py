"""Tests for the backing-track arranger (synth2.arranger) and its project/tool integration."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from synth2 import arranger
from synth2.ai.tools import Session, ToolError, call_tool
from synth2.arranger import (BackingStem, kick_times, render_backing, validate_arrangement)
from synth2.arranger.arrangement import parse_riff
from synth2.arranger.fx import FX_SOUNDS, PROCESS_EFFECTS, process, render_fx
from synth2.arranger.synth import PRESETS, Hit, preset_params, render_hit
from synth2.arranger.theory import degree_semitones, parse_chord, scale_for, voice_chord
from synth2.project import Project

SR = 22050
BPM = 168.0
BAR = 4 * 60 / BPM
ROLES = {"drums", "bass", "keys", "lead", "pad", "fx", "perc"}


def make_project(arr=None, bpm=BPM):
    p = Project.new("Arranger test", bpm=bpm)
    p.key = "A minor"
    p.backing = copy.deepcopy(arr if arr is not None else arranger.example_arrangement())
    return p


@pytest.fixture(scope="module")
def example_render():
    p = make_project()
    t1 = 4 * BAR + 0.5
    return p, t1, render_backing(p, SR, 0.0, t1)


# ------------------------------------------------------------------------------------------- rendering

def test_example_is_valid():
    assert validate_arrangement(arranger.example_arrangement()) == []


def test_render_example_stems(example_render):
    p, t1, stems = example_render
    n = int(round(t1 * SR))
    assert stems and all(isinstance(s, BackingStem) for s in stems)
    names = [s.name for s in stems]
    assert len(set(names)) == len(names)
    assert {"drums kick", "drums snare", "drums hats", "bass", "gtr", "lead", "arp", "fx"} <= set(names)
    for s in stems:
        assert s.audio.dtype == np.float32 and s.audio.shape == (n, 2), s.name
        assert np.isfinite(s.audio).all(), s.name
        peak = float(np.abs(s.audio).max())
        rms = float(np.sqrt(np.mean(s.audio.astype(np.float64) ** 2)))
        assert 1e-4 < peak < 1.5, (s.name, peak)  # calibrated: well inside full scale at gain_db 0
        assert rms > 1e-5, s.name
        assert s.role in ROLES
    by = {s.name: s for s in stems}
    assert by["bass"].sidechain and not by["drums kick"].sidechain
    assert by["gtr"].pan == pytest.approx(-0.3) and by["gtr"].gain_db == pytest.approx(-1)


def test_window_is_identical_to_full_render(example_render):
    p, t1, full = example_render
    t0, t2 = 1.37, 3.05
    part = render_backing(p, SR, t0, t2)
    a = int(round(t0 * SR))
    n = int(round((t2 - t0) * SR))
    full_by = {s.name: s.audio for s in full}
    assert part
    for s in part:
        assert s.audio.shape == (n, 2)
        ref = full_by[s.name][a:a + n]
        assert np.abs(ref - s.audio).max() <= 1e-4 * max(1.0, float(np.abs(ref).max())), s.name


def test_kick_times_match_pattern_and_audio(example_render):
    p, t1, stems = example_render
    kt = kick_times(p)
    # "x.....x...x....." per bar (beats 0, 1.5, 2.5); in bar 4 the fill replaces the groove from beat 3
    expected = [b + 4 * k for k in range(3) for b in (0.0, 1.5, 2.5)] + [12.0, 13.5, 14.0]
    expected_s = np.array(expected) * 60 / BPM
    assert len(kt) == len(expected) and np.all(np.diff(kt) > 0)
    assert np.allclose(kt, expected_s, atol=0.002)
    kick = next(s for s in stems if s.name == "drums kick").audio.mean(axis=1)
    peak = np.abs(kick).max()
    for t in kt:
        i = int(t * SR)
        assert np.abs(kick[i:i + int(0.03 * SR)]).max() > 0.3 * peak


def test_kick_has_weight_and_snare_has_crack(example_render):
    _, _, stems = example_render
    by = {s.name: s.audio.mean(axis=1) for s in stems}

    def peak_hz(x, lo, hi):
        spec = np.abs(np.fft.rfft(x, 1 << 17))
        f = np.fft.rfftfreq(1 << 17, 1 / SR)
        sel = (f > lo) & (f < hi)
        return f[sel][np.argmax(spec[sel])], spec, f

    k_hz, _, _ = peak_hz(by["drums kick"], 20, 400)
    assert 45 <= k_hz <= 65
    s_hz, spec, f = peak_hz(by["drums snare"], 120, 400)
    assert 150 <= s_hz <= 260
    noise_band = (spec[(f > 2000) & (f < 9000)] ** 2).sum()
    assert noise_band > 0.2 * (spec ** 2).sum()  # the snare is mostly noise crack above 2 kHz


def test_stop_silences_parts(example_render):
    p, t1, stems = example_render
    a = int((3 * BAR + 3.5 * 60 / BPM + 0.01) * SR)  # bar 4 beat 4.5 (+10 ms: past the 4 ms fade)
    b = int(4 * BAR * SR) - 2
    by = {s.name: s.audio for s in stems}
    for name in ("bass", "gtr", "lead", "arp"):
        assert np.abs(by[name][a:b]).max() < 1e-6, name
    assert np.abs(by["drums snare"][a:b]).max() > 1e-3  # drums are excluded from the stop


def test_no_backing():
    p = Project.new("plain", bpm=120)
    assert render_backing(p, SR, 0.0, 1.0) == []
    assert len(kick_times(p)) == 0


def test_every_preset_renders_cleanly():
    for name in PRESETS:
        prm = preset_params(name)
        y = render_hit(prm, Hit(0.0, 0.25, (57, 60, 64) if PRESETS[name].get("role") not in ("bass", "lead")
                                else (45 if PRESETS[name]["role"] == "bass" else 69,), 0.9, seed=1), SR)
        assert y.dtype == np.float32 and y.ndim == 2 and y.shape[1] == 2, name
        assert np.isfinite(y).all(), name
        assert 1e-3 < np.abs(y).max() < 2.0, name
        # click-free note boundaries: both ends fade to (near) zero
        assert np.abs(y[-1]).max() < 1e-3 and np.abs(y[0]).max() < 0.05, name


def test_fx_sounds_and_processing_effects():
    rng = np.random.default_rng(3)
    for sound in FX_SOUNDS:
        y = render_fx(sound, 0.4, None, SR, rng, {}, 0.36)
        assert y.ndim == 2 and y.shape[1] == 2 and np.isfinite(y).all() and np.abs(y).max() > 1e-3, sound
    base = (np.random.default_rng(1).standard_normal((SR, 2)) * 0.1).astype(np.float32)
    for eff in PROCESS_EFFECTS:
        buf = base.copy()
        process(eff, buf, 2000, 15000, SR, 0.36, {})
        assert buf.shape == base.shape and np.isfinite(buf).all(), eff
        assert np.array_equal(buf[:2000], base[:2000]) and np.array_equal(buf[15000:], base[15000:]), eff


# -------------------------------------------------------------------------------------------- format

def test_theory_and_degrees():
    key = scale_for("A minor")
    amg = parse_chord("Am/G")
    assert (amg.root, amg.bass, amg.intervals) == (9, 7, (0, 3, 7))
    assert degree_semitones("R", amg, key) == -2  # the slash bass G
    e = parse_chord("E")
    assert degree_semitones("7", e, key) == 10  # D: the key's 7th over E in A minor
    assert degree_semitones("3", e, key) == 4  # G#: the chord's own third
    assert degree_semitones("5'", parse_chord("F#dim"), key) == 18
    assert parse_chord("N.C.") is None
    with pytest.raises(ValueError):
        parse_chord("Hm")
    assert voice_chord(parse_chord("G"), "power8", 60, 40) == [43, 50, 55]
    tri = voice_chord(parse_chord("Am"), "triad", 64, 48)
    assert sorted(p % 12 for p in tri) == [0, 4, 9] and abs(sum(tri) / 3 - 64) < 6
    notes = parse_riff("A4 e C5+E5 q r e G4! s")
    assert notes[0] == (0.0, (69,), 0.5, 0.85) and notes[1][1] == (72, 76) and notes[2][0] == 2.0
    assert notes[2][3] == 1.0


def test_validation_messages_are_helpful():
    arr = arranger.example_arrangement()
    arr["instruments"]["bass"]["preset"] = "bass_slapp"
    arr["sections"][0]["parts"]["gtr"] = "chugg"
    arr["sections"][0]["chords"][1] = "Fxyz"
    arr["patterns"]["beat"]["lanes"]["snair"] = "x..."
    arr["sections"][0]["parts"]["lead"] = "beat"  # a drums pattern on a pitched instrument
    msgs = validate_arrangement(arr)
    text = "\n".join(msgs)
    assert all(m.startswith(("error:", "warning:")) for m in msgs)
    assert "bass_slap" in text and "did you mean" in text  # preset suggestion
    assert "'chugg'" in text and "chug" in text
    assert "Fxyz" in text
    assert "snair" in text and "snare" in text
    assert "drums pattern" in text or "drum kits play" in text
    assert validate_arrangement({"sections": []})[0].startswith("error:")
    assert validate_arrangement("nope")[0].startswith("error:")
    with pytest.raises(arranger.ArrangementError):
        render_backing(make_project(arr), SR, 0.0, 1.0)


def test_sections_like_segments_fills_and_events():
    arr = {
        "instruments": {"kit": {"type": "drums"}, "keys": {"preset": "pluck"}},
        "patterns": {"b": {"kind": "drums", "lanes": {"kick": "x...x...x...x..."}},
                     "f": {"kind": "drums", "lanes": {"snare": "xxxxxxxxxxxxxxxx"}},
                     "st": {"kind": "chords", "rhythm": "x---------------"}},
        "sections": [{"name": "A", "start_bar": 1, "bars": 2, "chords": ["Am F"], "parts": {"kit": "b", "keys": "st"}},
                     {"name": "B", "like": "A", "start_bar": 3, "bars": 2,
                      "parts": {"kit": [{"pattern": "b", "bars": [3, 3]}, {"pattern": None, "bars": [4, 4]}],
                                "keys": "st"}}],
        "fills": [{"part": "kit", "bar": 2, "beat": 3, "pattern": "f"}],
        "events": [{"type": "hit", "part": "kit", "voice": "crash", "bar": 4, "beat": 2}],
    }
    assert validate_arrangement(arr) == []
    comp = arranger.compile_arrangement(arr, "A minor")
    kicks = [e.beat for e in comp.drum_events["kit"] if e.voice == "kick"]
    assert kicks == [0, 1, 2, 3, 4, 5, 8, 9, 10, 11]  # bar 2 beats 3-4 are the fill, bar 4 is silent
    snares = [e.beat for e in comp.drum_events["kit"] if e.voice == "snare"]
    assert snares[0] == 6.0 and len(snares) == 8
    assert [e.beat for e in comp.drum_events["kit"] if e.voice == "crash"] == [13.0]
    # held chord re-struck at the mid-bar chord change ("Am F" splits the bar)
    starts = [e.beat for e in comp.note_events["keys"]]
    assert starts[:2] == [0.0, 2.0]


def test_project_serialisation_round_trip(tmp_path):
    p = make_project()
    d = p.to_dict()
    assert d["backing"] == arranger.example_arrangement()
    q = Project.from_dict(json.loads(json.dumps(d)))
    assert q.backing == p.backing and q.backing is not p.backing
    assert p.copy().backing == p.backing
    path = tmp_path / "song.s2p.json"
    p.save(path)
    assert Project.load(path).backing == p.backing
    plain = Project.new("plain")
    assert "backing" not in plain.to_dict() and Project.from_dict(plain.to_dict()).backing is None
    assert "backing" in p.summary()


# --------------------------------------------------------------------------------------------- tools

def test_backing_tools():
    s = Session()
    call_tool(s, "new_project", {"title": "Band", "bpm": 168, "key": "A minor"})
    assert call_tool(s, "get_backing", {})["backing"] is None
    ref = call_tool(s, "get_backing", {"detail": "reference"})
    assert set(PRESETS) <= set(ref["presets"]) and "kick" in ref["drum_voices"] and "laser" in ref["fx_sounds"]
    ex = call_tool(s, "get_backing", {"detail": "example"})["arrangement"]
    r = call_tool(s, "set_backing", {"arrangement": ex})
    assert r["ok"] and r["warnings"] == [] and r["summary"]["sections"][0]["name"] == "Demo"
    assert s.project.backing == ex
    undo_depth = len(s.undo_stack)

    # invalid -> ToolError, nothing changes
    bad = copy.deepcopy(ex)
    bad["instruments"]["lead"]["preset"] = "kazoo"
    with pytest.raises(ToolError, match="kazoo"):
        call_tool(s, "set_backing", {"arrangement": bad})
    assert s.project.backing == ex and len(s.undo_stack) == undo_depth

    # merge: change one pattern, keep the rest
    r = call_tool(s, "set_backing", {"merge": True, "arrangement": {
        "patterns": {"bounce": {"kind": "line", "steps": "R.......R......."}}}})
    assert s.project.backing["patterns"]["bounce"]["steps"] == "R.......R......."
    assert s.project.backing["patterns"]["riff"] == ex["patterns"]["riff"]
    full = call_tool(s, "get_backing", {"detail": "full"})["arrangement"]
    assert full == s.project.backing
    summ = call_tool(s, "get_backing", {})["backing"]
    assert summ["kick_hits"] == 12 and summ["length_bars"] == 4

    call_tool(s, "undo", {})
    assert s.project.backing["patterns"]["bounce"] == ex["patterns"]["bounce"]
    assert call_tool(s, "clear_backing", {})["cleared"] is True
    assert s.project.backing is None
    call_tool(s, "undo", {})
    assert s.project.backing == ex
