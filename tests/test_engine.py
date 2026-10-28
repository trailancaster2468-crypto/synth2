import numpy as np
import pytest

from synth2 import Project
from synth2.analysis import analyze_audio, estimate_f0, pitch_accuracy
from synth2.engine import render_project, render_track
from synth2.engine.expression import NoteTiming, build_pitch, get_style, STYLES
from synth2.engine.planner import plan_track
from synth2.phonemes import resolve_notes, split_lyrics
from synth2.voices import VOICES, get_voice

SR = 22050


def make_project(lyrics="hello world sing", pitches=(64, 67, 69, 67), voice="aurora", style="pop", language="en"):
    p = Project.new("t", bpm=120, voice=voice, language=language)
    p.sample_rate = SR
    t = p.tracks[0]
    t.style = style
    toks = split_lyrics(lyrics, language)
    for i, (tok, pitch) in enumerate(zip(toks, pitches)):
        t.add_note(1 + i, 1, pitch, tok)
    return p


def test_lyric_resolution_continuations_and_melisma():
    p = make_project("hello", (60, 62, 64))
    t = p.tracks[0]
    t.add_note(4, 1, 65, "-")
    sylls = resolve_notes(t.notes)
    assert sylls[0].phones[:2] == ["hh", "ax"] or sylls[0].phones[0] == "hh"
    assert sylls[1].phones[-1] == "ow"
    assert sylls[2].melisma  # "+" without remaining syllables


def test_planner_vowel_on_note_onset_and_consonants_before():
    p = make_project("star", (67,))
    t = p.tracks[0]
    plan = plan_track(t.notes, p.tempo, "en", get_style("pop"), np.random.default_rng(0))
    onset = p.tempo.beat_to_sec(1)
    vowel = next(s for s in plan.segments if s.ph.name == "aa")
    assert abs(vowel.start - onset) < 1e-6
    s_seg = next(s for s in plan.segments if s.ph.name == "s")
    assert s_seg.start < onset
    assert plan.timings[0].lead > 0


def test_pitch_curve_hits_notes_flat_style():
    voice = get_voice("aurora")
    t = np.arange(0, 2.0, 0.003)
    notes = [NoteTiming(0, 0.1, 1.0, 64, {}, True), NoteTiming(1, 1.0, 1.9, 69, {}, False, True)]
    pitch = build_pitch(t, notes, get_style("flat"), voice, np.random.default_rng(0))
    assert abs(pitch[(t > 0.5) & (t < 0.8)].mean() - 64) < 0.05
    assert abs(pitch[(t > 1.4) & (t < 1.6)].mean() - 69) < 0.05


def test_render_pitch_accuracy_and_levels():
    p = make_project()
    res = render_project(p)
    assert res.audio.ndim == 2 and res.audio.shape[1] == 2
    assert res.sample_rate == SR
    assert np.isfinite(res.audio).all()
    assert res.report["peak_db"] <= 0.0
    pa = pitch_accuracy(p, res)["tracks"][0]
    assert pa["notes_checked"] >= 3
    assert pa["mean_abs_cents"] < 20
    a = analyze_audio(res.audio, SR)
    assert a["clipped_samples"] == 0
    assert a["voiced_ratio"] > 0.4


@pytest.mark.parametrize("voice", list(VOICES))
def test_every_voice_renders_in_range(voice):
    v = VOICES[voice]
    mid = (v.range_low + v.range_high) // 2
    p = make_project("la la la", (mid - 2, mid, mid + 2), voice=voice)
    mono, rep = render_track(p, 0, sample_rate=SR)
    assert np.isfinite(mono).all() and np.abs(mono).max() > 1e-3
    assert not any("outside" in w for w in rep["warnings"])


@pytest.mark.parametrize("style", list(STYLES))
def test_every_style_renders(style):
    p = make_project("oh oh", (62, 65), style=style)
    mono, _ = render_track(p, 0, sample_rate=SR)
    assert np.isfinite(mono).all()


def test_japanese_render():
    p = make_project("さくら", (69, 69, 71), voice="nova", language="ja")
    res = render_project(p)
    phon = list(res.report["tracks"][0]["phonemes"].values())
    assert phon == ["s a", "k u", "dx a"]


def test_partial_render_and_stems(tmp_path):
    p = make_project("one two three four", (60, 62, 64, 65))
    p.add_track("Harmony", voice="atlas").add_note(1, 4, 52, "oh")
    full = render_project(p)
    part = render_project(p, start_beat=2, end_beat=3)
    assert part.duration < full.duration
    assert len(full.stems) == 2
    paths = full.save_stems(tmp_path)
    assert len(paths) == 2 and all(x.exists() for x in paths)
    wav = full.save(tmp_path / "mix.wav", bits=24)
    from synth2.io.wav import read_wav

    a, sr = read_wav(wav)
    assert sr == SR and a.shape[1] == 2 and abs(len(a) - len(full.audio)) <= 1


def test_mute_solo():
    p = make_project("la", (60,))
    p.add_track("B").add_note(1, 1, 64, "la")
    p.tracks[1].solo = True
    res = render_project(p)
    assert list(res.stems) == [p.tracks[1].id]


def test_curves_affect_output():
    p = make_project("ah", (64,))
    p.tracks[0].notes[0].duration = 2
    p.master.normalize = False
    base = render_project(p).audio
    p.tracks[0].curve("loudness").add(0, -12)
    quieter = render_project(p).audio
    assert np.sqrt(np.mean(quieter ** 2)) < 0.5 * np.sqrt(np.mean(base ** 2))
    p.tracks[0].curves.clear()
    p.tracks[0].curve("pitchDelta").add(0, 200)
    res = render_project(p)
    t, f0 = estimate_f0(res.mono(), SR)
    sung = np.median(f0[(t > 0.7) & (t < 1.2) & (f0 > 0)])
    assert abs(12 * np.log2(sung / 440) + 69 - 66) < 0.3


def test_loudness_calibration():
    """A sustained vowel at default settings should land around -20..-10 dBFS RMS before normalisation."""
    p = make_project("ah", (64,))
    p.tracks[0].notes[0].duration = 2
    p.master.normalize = False
    p.tracks[0].reverb_send = 0
    res = render_project(p)
    rms = 20 * np.log10(np.sqrt(np.mean(res.mono()[int(0.6 * SR):int(1.4 * SR)] ** 2)))
    assert -24 < rms < -8


def test_overlapping_notes_warn():
    p = make_project("la la", (60, 62))
    p.tracks[0].notes[0].duration = 1.5
    res = render_project(p)
    assert any("overlap" in w for w in res.report["warnings"])
