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


# --------------------------------------------------------------------------- voice quality (DSP)
def _sustain(phones, pitch=64, voice="aurora", sr=SR, dur=1.5, style="flat"):
    """Render one sustained note and return its steady middle part (mono float64)."""
    p = Project.new("q", bpm=60, voice=voice)
    p.sample_rate = sr
    t = p.tracks[0]
    t.style = style
    t.add_note(1, dur, pitch, f"[{phones}]")
    mono, _ = render_track(p, 0, sample_rate=sr)
    a, b = p.tempo.beat_to_sec(1) + 0.3, p.tempo.beat_to_sec(1 + dur) - 0.3
    return mono[int(a * sr):int(b * sr)].astype(float)


def _band_db(x, sr, lo, hi):
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
    f = np.fft.rfftfreq(len(x), 1 / sr)
    return 10 * np.log10(spec[(f >= lo) & (f < hi)].sum() + 1e-30)


def test_glottal_source_follows_tension():
    """Pressed phonation weakens H1 and brightens the top; lax phonation does the opposite."""
    from synth2.engine.vocoder import glottal_spectrum, rd_glottal

    voice = get_voice("aurora")
    f0 = np.array([220.0, 220.0, 220.0])
    oq, fa = rd_glottal(f0, np.array([-0.8, 0.0, 0.8]), np.zeros(3), voice)
    assert oq[0] > oq[1] > oq[2] and fa[0] < fa[1] < fa[2]
    harm = 220.0 * np.arange(1, 15)
    g = 20 * np.log10(np.abs(glottal_spectrum(harm, f0, oq)) + 1e-12)
    g -= 10 * np.log10(1 + (harm[None, :] / fa[:, None]) ** 2)
    h1_h2 = g[:, 0] - g[:, 1]
    h1_hf = g[:, 0] - g[:, 13]  # ~3 kHz
    assert h1_h2[0] > h1_h2[1] > h1_h2[2]
    assert h1_hf[0] > h1_hf[1] + 5 > h1_hf[2] + 10


def test_voiced_signal_has_no_dc_and_no_hf_brickwall():
    sr = 44100
    x = _sustain("aa", sr=sr)
    rms = np.sqrt(np.mean(x ** 2))
    assert abs(x.mean()) < 1e-3 * rms
    # the vocal tract keeps resonating above F5: the 7-10 kHz band must not vanish
    assert _band_db(x, sr, 7000, 10000) > _band_db(x, sr, 300, 2000) - 55


def test_nasal_murmur_duller_than_vowel():
    sr = 44100
    vowel, nasal = _sustain("aa", sr=sr), _sustain("m", sr=sr)
    tilt = lambda x: _band_db(x, sr, 5000, 16000) - _band_db(x, sr, 100, 1000)  # noqa: E731
    assert tilt(nasal) < tilt(vowel) - 6


def test_vowel_loudness_is_consistent():
    """Vowels at the same pitch and level stay within a few dB of each other (loudness-weighted)."""
    from synth2.engine.vocoder import loudness_weight

    for pitch in (57, 64):
        levels = []
        for v in ("iy", "ih", "eh", "aa", "ao", "uw", "er"):
            x = _sustain(v, pitch=pitch, dur=1.2)
            S = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
            levels.append(10 * np.log10(np.sum(S * loudness_weight(np.fft.rfftfreq(len(x), 1 / SR)))))
        assert max(levels) - min(levels) < 5.0, (pitch, np.round(levels, 1))


def test_phrase_start_scoop_has_no_pitch_step():
    """The scoop must already be in place when the (voiced) onset consonants start."""
    voice = get_voice("aurora")
    t = np.arange(0, 2.0, 0.0029)
    notes = [NoteTiming(0, 0.2, 0.6, 60, {}, True, True),
             NoteTiming(1, 1.2, 1.8, 67, {}, True, True, lead=0.2)]
    pitch = build_pitch(t, notes, get_style("rock"), voice, np.random.default_rng(0))
    m = (t > 1.2 - 0.2) & (t < 1.3)  # onset consonants + start of the vowel
    assert np.max(np.abs(np.diff(pitch[m]))) < 0.15


# --------------------------------------------------------------------------- expression regressions
def test_overshoot_none_attr_uses_style_default():
    """``overshoot: null`` (its documented default) must not crash and acts like no attribute."""
    voice = get_voice("aurora")
    t = np.arange(0, 2.0, 0.0029)

    def pitch(attrs):
        notes = [NoteTiming(0, 0.2, 1.0, 60, {}), NoteTiming(1, 1.0, 1.8, 64, attrs)]
        return build_pitch(t, notes, get_style("pop"), voice, np.random.default_rng(3))

    assert np.allclose(pitch({"overshoot": None}), pitch({}))
    p = Project.from_dict({"sample_rate": SR, "tracks": [{"notes": [
        {"onset": 1, "duration": 0.5, "pitch": 60, "lyric": "la"},
        {"onset": 1.5, "duration": 0.5, "pitch": 64, "lyric": "la", "attrs": {"overshoot": None}}]}]})
    mono, _ = render_track(p, 0, sample_rate=SR)
    assert np.isfinite(mono).all()


def _flat_dynamics(notes, t):
    from synth2.engine.expression import build_dynamics

    voice = get_voice("aurora")
    style = get_style("flat")
    return build_dynamics(t, notes, style, voice, np.full(len(t), 60.0))


def test_velocity_zero_is_the_quietest_not_full_level():
    t = np.arange(0, 1.5, 0.0029)
    m = (t > 0.6) & (t < 0.9)
    levels = {}
    for vel in (None, 1.0, 0.05, 0.0):
        attrs = {} if vel is None else {"velocity": vel}
        levels[vel] = _flat_dynamics([NoteTiming(0, 0.5, 1.0, 60, attrs)], t)[0][m].mean()
    assert abs(levels[None]) < 1e-6 and abs(levels[1.0]) < 1e-6
    assert abs(levels[0.0] - 20 * np.log10(0.05)) < 1e-6
    assert abs(levels[0.0] - levels[0.05]) < 1e-6


def test_note_dynamics_cover_onset_consonants_and_humanized_timing():
    t = np.arange(0, 2.5, 0.0029)
    # a soft note's onset consonants (before its start) are soft too, and so are its tension and breath
    notes = [NoteTiming(0, 0.5, 1.0, 60, {}),
             NoteTiming(1, 1.0, 1.6, 60, {"velocity": 0.2, "tension": 0.5, "breathiness": 0.3}, lead=0.1)]
    loud, tension, breath = _flat_dynamics(notes, t)
    cons = (t > 0.93) & (t < 0.98)
    assert np.allclose(loud[cons], 20 * np.log10(0.2), atol=0.05)
    assert np.allclose(tension[cons], 0.5, atol=0.01) and np.allclose(breath[cons], 0.3, atol=0.01)
    assert np.allclose(loud[(t > 0.55) & (t < 0.87)], 0.0, atol=0.05)
    # humanized starts with unjittered ends: overlapping and gapped notes keep one flat level
    notes = [NoteTiming(0, 0.5, 1.0, 60, {"velocity": 0.5}), NoteTiming(1, 0.985, 1.5, 60, {"velocity": 0.5}),
             NoteTiming(2, 1.515, 2.0, 60, {"velocity": 0.5})]
    loud = _flat_dynamics(notes, t)[0]
    assert np.allclose(loud[(t > 0.55) & (t < 2.0)], 20 * np.log10(0.5), atol=1e-6)


def test_short_rest_with_borrowed_onset_keeps_previous_vowel_pitch():
    """Onset consonants that fill a short rest and borrow from the previous note must not drag the
    pitch switch into the previous, still-sounding vowel."""
    voice = get_voice("aurora")
    t = np.arange(0, 2.0, 0.0029)
    # 0.1 s rest, but 0.12 s of onset consonants: the C4 vowel is sung until 1.1 - 0.12 = 0.98 s
    notes = [NoteTiming(0, 0.5, 1.0, 60, {}, True), NoteTiming(1, 1.1, 1.6, 67, {}, False, True, lead=0.12)]
    for style in ("pop", "flat", "rock"):
        pitch = build_pitch(t, notes, get_style(style, {"drift_cents": 0}), voice, np.random.default_rng(0))
        assert np.max(np.abs(pitch[(t > 0.7) & (t < 0.98)] - 60)) < 0.5, style  # after the phrase-start scoop
        assert abs(pitch[(t > 1.3) & (t < 1.4)].mean() - 67) < 0.3, style


def test_curves_interpolate_in_beat_space_across_tempo_changes():
    from synth2.engine.expression import evaluate_curve
    from synth2.project import Curve
    from synth2.timing import TempoMap

    tm = TempoMap()
    tm.set(0, 60)
    tm.set(4, 240)
    beats = np.array([-1.0, 0.0, 1.0, 2.0, 4.0, 5.0, 6.0, 8.0, 9.0])
    secs = np.array([tm.beat_to_sec(b) for b in beats])
    expect = np.clip(-12 + 1.5 * beats, -12, 0)
    c = Curve([(0, -12.0), (8, 0.0)], "linear")
    for conv in (tm.beat_to_sec, tm):
        assert np.allclose(evaluate_curve(c, secs, conv, 0.0), expect)
    c.interpolation = "cubic"
    u = np.clip(beats / 8, 0, 1)
    assert np.allclose(evaluate_curve(c, secs, tm.beat_to_sec, 0.0), -12 + 12 * (0.5 - 0.5 * np.cos(np.pi * u)))
    c = Curve([(0, 0.0), (4, 1.0), (6, 2.0)], "step")
    assert list(evaluate_curve(c, secs, tm.beat_to_sec, 0.0)) == [0, 0, 0, 0, 1, 1, 2, 2, 2]
    # any monotonic beat -> seconds callable still works
    assert np.allclose(evaluate_curve(Curve([(0, 0.0), (8, 8.0)], "linear"), np.array([1.0, 2.0]),
                                      lambda b: 0.5 * b, 0.0), [2.0, 4.0])


def test_curve_points_need_not_be_sorted():
    from synth2.engine.expression import evaluate_curve
    from synth2.project import Curve
    from synth2.timing import TempoMap

    tm = TempoMap()
    tm.set(0, 60)
    t = np.array([0.5, 1.5, 2.0, 2.5, 3.5])
    c = Curve([(3, 0.0), (1, 0.0), (2, 10.0)])
    for kind, expect in (("linear", [0, 5, 10, 5, 0]), ("cubic", [0, 5, 10, 5, 0]), ("step", [0, 0, 10, 10, 0])):
        c.interpolation = kind
        assert np.allclose(evaluate_curve(c, t, tm.beat_to_sec, 0.0), expect), kind


def test_partial_window_sings_the_same_pitch_performance():
    """Drift and vibrato depend on absolute time and the seed only, not on the render window or on
    how much of the shared generator was used before."""
    voice = get_voice("aurora")
    style = get_style("pop")
    fp = 64 / SR
    notes = [NoteTiming(i, 0.5 + 0.8 * i, 1.3 + 0.8 * i, [60, 62, 64, 65, 67, 65][i], {}, i == 0, i == 5)
             for i in range(6)]
    tF = np.arange(int(5.5 / fp)) * fp
    pF = build_pitch(tF, notes, style, voice, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    rng.normal(size=7)  # e.g. the planner's timing jitter for a different number of notes
    tP = 2.5 + np.arange(int(2.8 / fp)) * fp
    pP = build_pitch(tP, notes[2:], style, voice, rng)
    m = (tP > 3.0) & (tP < 5.2)
    assert np.max(np.abs(pP[m] - np.interp(tP[m], tF, pF))) * 100 < 1.0
    # still random: another seed gives another performance
    pF2 = build_pitch(tF, notes, style, voice, np.random.default_rng(43))
    assert np.max(np.abs(pF2 - pF)) * 100 > 3.0


def test_vibrato_rate_zero_disables_vibrato():
    voice = get_voice("aurora")
    t = np.arange(0, 3.0, 0.0029)
    notes = [NoteTiming(0, 0.5, 2.5, 60, {"vibrato_rate": 0.0, "vibrato_depth": 2.0})]
    pitch = build_pitch(t, notes, get_style("flat"), voice, np.random.default_rng(17))
    assert np.allclose(pitch, 60.0)
    notes[0].attrs["vibrato_rate"] = 5.5
    pitch = build_pitch(t, notes, get_style("flat"), voice, np.random.default_rng(17))
    assert np.ptp(pitch[(t > 1.5) & (t < 2.3)]) > 2.0


# --------------------------------------------------------------------------- planner: breaths, closures, timing
def _plan(specs, bpm=120, language="en", style="pop", seed=0, attrs=None):
    """Plan (onset, duration, pitch, lyric[, phonemes]) note specs; returns (plan, notes)."""
    from synth2.project import Note
    from synth2.timing import TempoMap

    tm = TempoMap()
    tm.set(0, bpm)
    notes = [Note(o, d, p, ly, phonemes=(rest[0] if rest else None), attrs=dict(attrs or {}))
             for o, d, p, ly, *rest in specs]
    return plan_track(notes, tm, language, get_style(style), np.random.default_rng(seed)), notes


def _assert_no_overlaps(plan):
    segs = sorted(plan.segments, key=lambda s: s.start)
    for s in segs:
        assert s.end >= s.start, (s.ph.name, s.part, s.start, s.end)
    for a, b in zip(segs, segs[1:]):
        assert a.end <= b.start + 1e-9, ((a.ph.name, a.part, a.start, a.end, a.unit),
                                         (b.ph.name, b.part, b.start, b.end, b.unit))


def test_breath_note_is_unpitched_and_ends_the_phrase():
    """A 'br' note sings no pitch (same curve as an empty rest) and separates two phrases."""
    voice, style = get_voice("aurora"), get_style("pop")
    t = np.arange(0, 3.5, 0.0029)
    curves = {}
    for lyr in ("br", ""):
        plan, notes = _plan([(1, 2, 62, "day"), (3, 1, 50, lyr), (4, 2, 64, "love")])
        assert [nt.pitch for nt in plan.timings] == [62, 64]
        assert plan.timings[0].phrase_end and plan.timings[1].phrase_start
        curves[lyr] = build_pitch(t, plan.timings, style, voice, np.random.default_rng(3))
        if lyr == "br":
            day, br, love = plan.units
            assert br.breath and day.phrase_end and love.phrase_start
            assert plan.note_phonemes[notes[1].id] == "br"
            breath = [s for s in plan.segments if s.ph.name == "br" and s.unit == 1]
            assert len(breath) == 1 and breath[0].end <= love.start - love.lead + 1e-9
    assert np.allclose(curves["br"], curves[""])
    end_of_day = curves["br"][(t >= 1.40) & (t < 1.50)]
    assert end_of_day.min() > 60.5  # no glide down to the breath note's D3
    # a breath as the last note still lets the final note end its phrase
    plan, _ = _plan([(1, 2, 62, "day"), (3, 1, 50, "br")])
    assert plan.units[0].phrase_end and plan.timings[-1].phrase_end


def test_explicit_breath_phoneme_is_a_breath_not_a_schwa():
    for spec in [(2, 1, 60, "br", "br"), (2, 1, 60, "[br]")]:
        plan, notes = _plan([(1, 1, 60, "la"), spec, (3, 1, 60, "la")])
        assert plan.units[1].breath
        segs = [s for s in plan.segments if s.note == 1]
        assert segs and all(s.ph.name == "br" for s in segs)
        assert min(s.start for s in segs) >= 1.0 - 1e-9  # no 250 ms lead into the previous note
        assert len(plan.timings) == 2
        _assert_no_overlaps(plan)


def test_melisma_after_breath_is_sung():
    plan, notes = _plan([(1, 0.5, 64, "br"), (1.5, 2, 64, "-")], style="flat")
    assert [u.breath for u in plan.units] == [True, False]
    vowel = [s for s in plan.segments if s.unit == 1 and s.ph.is_vowel]
    assert vowel and vowel[0].start == pytest.approx(0.75) and vowel[-1].end > 1.6
    assert [nt.index for nt in plan.timings] == [1]


def test_small_tsu_and_silence_are_not_sung_on_a_schwa():
    plan, notes = _plan([(1, 0.5, 60, "か"), (1.5, 0.5, 60, "っ"), (2, 1, 60, "た")], language="ja")
    segs = [s for s in plan.segments if s.note == 1]
    assert segs and all(s.ph.name == "q" and s.part == "closure" for s in segs)
    assert plan.note_phonemes[notes[1].id] == "q"
    assert [nt.index for nt in plan.timings] == [0, 2]
    ka, cl, ta = plan.units
    assert cl.closure and not ka.phrase_end and not ta.phrase_start  # a geminate stays inside the phrase
    _assert_no_overlaps(plan)
    for lyr, phonemes, reported in [("[sil]", None, "sil"), ("x", "pau", "sil"), ("[s]", None, "s ax")]:
        plan, notes = _plan([(1, 1, 60, "la"), (2, 1, 60, lyr, phonemes), (3, 1, 60, "la")])
        assert plan.note_phonemes[notes[1].id] == reported
        sung = [s.ph.name for s in plan.segments if s.note == 1]
        assert sung == ([] if reported == "sil" else ["s", "ax"])


def test_onset_consonants_never_cover_the_previous_short_vowel():
    cases = [(180, 0.25, ["a", "spring", "a"]), (120, 0.125, ["time", "star", "la"]),
             (120, 0.125, ["strengths", "splashed", "strengths", "splashed", "la"]),
             (120, 0.0625, ["strengths", "splashed", "la"])]
    for bpm, dur, words in cases:
        for cl in (1.0, 2.0):
            specs = [(1 + i * dur, dur, 60, w) for i, w in enumerate(words)]
            plan, _ = _plan(specs, bpm=bpm, attrs={"consonant_length": cl})
            _assert_no_overlaps(plan)
            for k, u in enumerate(plan.units[:-1]):
                vowels = [s for s in plan.segments if s.unit == k and s.ph.is_vowel]
                assert vowels and vowels[0].start == pytest.approx(u.start)
                nxt = plan.units[k + 1]
                assert vowels[-1].end <= nxt.start - nxt.lead + 1e-9


def test_short_affricates_keep_closure_burst_order():
    for w in ("lunch", "much", "judged", "branched"):
        plan, _ = _plan([(1, 0.125, 60, w), (1.125, 1, 60, "la")])
        _assert_no_overlaps(plan)
        closures = [s for s in plan.segments if s.part == "closure" and s.ph.name in ("t", "d")]
        assert closures and all(s.end > s.start for s in closures)


def test_invalid_explicit_lyric_does_not_break_plus_words():
    specs = [(1, 1, 60, "hello"), (2, 1, 62, "+"), (3, 1, 64, "world")]
    good, good_notes = _plan(specs)
    plan, notes = _plan(specs + [(5, 1, 60, "...")])
    assert [plan.note_phonemes[n.id] for n in notes[:3]] == [good.note_phonemes[n.id] for n in good_notes]
    assert plan.note_phonemes[notes[1].id] != "-" and plan.note_phonemes[notes[3].id] == "l aa"
    assert sum("invalid" in w for w in plan.warnings) == 1


def test_near_duplicate_onsets_and_zero_length_notes():
    plan, notes = _plan([(1, 1, 60, "la"), (1.004, 1, 64, "lo"), (2.004, 1, 62, "la")])
    assert len(plan.units) == 2 and plan.units[0].notes == [0]
    assert plan.timings[0].pitch == 64 and any("dropped" in w for w in plan.warnings)
    _assert_no_overlaps(plan)
    plan, _ = _plan([(0, 2, 60, "la"), (1, 0, 60, "lo")])
    assert len(plan.units) == 1 and plan.units[0].end == pytest.approx(1.0) and not plan.warnings


def test_unpronounceable_lyric_warns():
    plan, notes = _plan([(1, 1, 60, "la"), (2, 1, 60, "2"), (3, 1, 60, "la")])
    assert len(plan.units) == 2
    assert any(notes[1].id in w and "no pronounceable" in w for w in plan.warnings)


def test_timing_humanize_never_moves_the_first_note_before_zero():
    for seed in range(12):
        plan, _ = _plan([(0, 1, 64, "ah"), (1, 1, 66, "oh")], style="jazz", seed=seed)
        assert plan.start >= 0.0


# --------------------------------------------------------------------------- renderer regressions
def _capture_curves(monkeypatch):
    """Record the frame times, pitch and loudness / breathiness curves of the last track render."""
    import synth2.engine.renderer as R

    cap = {}
    orig_pitch, orig_dyn = R.build_pitch, R.build_dynamics

    def pitch(t, *a, **k):
        out = orig_pitch(t, *a, **k)
        cap["t"], cap["pitch"] = t.copy(), out.copy()
        return out

    def dyn(t, *a, **k):
        out = orig_dyn(t, *a, **k)
        cap["loud"], cap["breath"] = out[0].copy(), out[2].copy()
        return out

    monkeypatch.setattr(R, "build_pitch", pitch)
    monkeypatch.setattr(R, "build_dynamics", dyn)
    return cap


def _region_diff(full, part, a, b):
    m = (part["t"] >= a) & (part["t"] < b)
    assert m.sum() > 50
    return {k: float(np.max(np.abs(part[k][m] - np.interp(part["t"][m], full["t"], full[k]))))
            for k in ("pitch", "loud", "breath")}


def test_partial_render_sings_the_region_like_the_full_render(monkeypatch):
    cap = _capture_curves(monkeypatch)
    # end side: a legato phrase continuing past end_beat is not planned as ending there
    p = Project.new("t", bpm=120)
    p.sample_rate = SR
    tr = p.tracks[0]
    tr.add_note(1, 4, 64, "aa"), tr.add_note(5, 1, 67, "aa"), tr.add_note(6, 2, 65, "aa")
    render_project(p)
    full = dict(cap)
    render_project(p, start_beat=0, end_beat=4)
    d = _region_diff(full, cap, 0.0, 2.0)
    assert d["pitch"] < 1e-6 and d["loud"] < 1e-6 and d["breath"] < 1e-6
    # start side: a mid-phrase region keeps its place in the phrase arc
    p = make_project("hello world sing a song of love tonight", (60, 63, 66, 61, 64, 67, 62, 65))
    render_project(p)
    full = dict(cap)
    render_project(p, start_beat=3, end_beat=6)
    d = _region_diff(full, cap, 1.5, 3.0)
    assert d["pitch"] < 1e-6 and d["loud"] < 1e-6 and d["breath"] < 1e-6


def test_partial_render_keeps_word_head_of_long_continuations():
    p = Project.new("x")
    p.sample_rate = SR
    t = p.tracks[0]
    t.style = "flat"
    t.add_note(1, 1, 60, "hello"), t.add_note(2, 6, 64, "+"), t.add_note(8, 1, 62, "-")
    full = render_project(p).report["tracks"][0]["phonemes"]
    part = render_project(p, start_beat=7, end_beat=9).report["tracks"][0]["phonemes"]
    held = t.notes[1].id
    assert full[held] == part[held] == "l ow"


def test_region_end_fades_out_instead_of_singing_on():
    p = Project.new("t", bpm=120)
    p.sample_rate = SR
    p.master.normalize = p.master.limiter = False
    t = p.tracks[0]
    t.reverb_send = 0
    t.add_note(1, 1, 60, "la"), t.add_note(2, 1, 62, "la"), t.add_note(3, 4, 64, "la")  # last note held 3..7
    a = render_project(p, start_beat=1, end_beat=3).mono()
    assert len(a) <= int(round((1.0 + 0.05) * SR)) + 1  # region (1 s) plus the short fade
    rms = lambda x: np.sqrt(np.mean(x ** 2))  # noqa: E731
    held = rms(a[int(0.9 * SR):int(1.0 * SR)])  # the held note, just before end_beat
    assert held > 0.03 and rms(a[-int(0.01 * SR):]) < 0.1 * held  # faded out, no hard cut at full level
    assert np.max(np.abs(a[-int(0.002 * SR):])) < 2e-3
    t.notes[-1].duration = 1
    t.add_note(4, 1, 65, "sing")
    b = render_project(p, start_beat=1, end_beat=4).mono()
    assert len(b) <= int(round((1.5 + 0.05) * SR)) + 1  # the note at end_beat is not sung on


def test_partial_renders_are_not_normalized_on_their_own():
    p = Project.new("t", bpm=120)
    p.sample_rate = SR
    t = p.tracks[0]
    t.add_note(1, 1, 60, "la"), t.add_note(2, 1, 62, "la"), t.add_note(8, 1, 64, "la")
    r = render_project(p, start_beat=3.0, end_beat=3.4)  # only the release / reverb of note 2
    p.master.normalize = False
    r2 = render_project(p, start_beat=3.0, end_beat=3.4)
    assert r.report["peak_db"] == pytest.approx(r2.report["peak_db"], abs=0.01) and r.report["peak_db"] < -15
    q = Project.new("b", bpm=120)
    q.sample_rate = SR
    q.tracks[0].add_note(1, 1, 60, "br")
    rq = render_project(q)
    assert rq.report["normalize_gain_db"] <= 20.0 and rq.report["rms_db"] < -28


def test_master_gain_applies_after_normalization():
    p = make_project("la la la la", (60, 62, 64, 66))
    p.master.limiter = False
    peaks = {}
    for g in (-12, 0):
        p.master.gain_db = g
        peaks[g] = render_project(p).report["peak_db"]
    assert peaks[0] == pytest.approx(-1.0, abs=0.05) and peaks[-12] == pytest.approx(-13.0, abs=0.05)


def test_stems_follow_master_gain_and_never_clip(tmp_path):
    from synth2.io.wav import read_wav

    p = make_project("la la la la", (60, 62, 64, 66))
    t = p.tracks[0]
    t.reverb_send = 0
    p.master.limiter = False
    r = render_project(p)  # normalised: the stems carry the same gain, so they sum to the dry master
    assert np.allclose(sum(r.stems.values()), r.audio, atol=1e-6)
    t.gain_db, t.pan, t.reverb_send = 12, 1.0, 0.25
    p.master.limiter, p.master.gain_db = True, 6  # the limiter keeps the master clean, not the stems
    r = render_project(p)
    assert r.report["peak_db"] <= 0
    s, _ = read_wav(r.save_stems(tmp_path)[0])
    assert int(np.sum(np.abs(s) >= 0.9999)) <= 1 and r.report["stems_gain_db"] < 0


def test_save_stems_keeps_unicode_and_duplicate_names(tmp_path):
    p = make_project("la", (60,))
    p.tracks[0].name = "ボーカル"
    for nm in ("コーラス", "Lead", "Lead", "Lead/Harmony?", "Lead_Harmony_", "???"):
        p.add_track(nm).add_note(1, 1, 64, "la")
    r = render_project(p, end_beat=2)
    out = r.save_stems(tmp_path)
    names = [x.name for x in out]
    assert len(set(names)) == len(out) == 7 and len(list(tmp_path.iterdir())) == 7
    assert names[:4] == ["ボーカル.wav", "コーラス.wav", "Lead.wav", "Lead_2.wav"]
    assert names[-1] == f"{p.tracks[-1].id}.wav"


def test_track_ids_resolve_like_project_track():
    p = make_project("la", (60,))
    p.add_track("Harmony").add_note(1, 1, 55, "la")
    for ids, want in ((["lead"], 0), (["-1"], 1), ([-1], 1), (["HARMONY"], 1), ([p.tracks[1].id], 1)):
        r = render_project(p, track_ids=ids, end_beat=2)
        assert [x["id"] for x in r.report["tracks"]] == [p.tracks[want].id]
    with pytest.raises(LookupError, match="nope"):
        render_project(p, track_ids=["nope"])
    with pytest.raises(LookupError):
        render_project(p, track_ids=["5"])


def test_tail_trim_is_relative_and_keeps_the_last_note():
    p = make_project("hello world sing a song", (60, 63, 66, 61, 64))
    p.master.normalize = False
    base = render_project(p).report["duration_s"]
    p.master.gain_db = -48
    assert render_project(p).report["duration_s"] == pytest.approx(base, abs=0.01)
    p.master.gain_db = 0
    p.tracks[0].gain_db = -60
    end = p.tempo.beat_to_sec(p.tracks[0].end)
    assert render_project(p).report["duration_s"] >= end + 0.1 - 1e-3


def test_partial_report_lists_only_region_notes():
    p = Project.new("t", bpm=120)
    p.sample_rate = SR
    for i in range(8):
        p.tracks[0].add_note(1 + i, 1, 60 + i, "la")
    rep = render_project(p, start_beat=5, end_beat=6).report["tracks"][0]
    assert rep["notes"] == 1 and list(rep["phonemes"]) == [p.tracks[0].notes[4].id]


def test_guessed_pronunciation_warning_uses_note_language():
    p = Project.new("w")
    p.sample_rate = SR
    t = p.tracks[0]
    t.add_note(1, 1, 64, "ryuu", language="ja"), t.add_note(2, 1, 64, "nya", language="ja")
    assert not any("guessed" in w for w in render_project(p).report["warnings"])
    p2 = Project.new("w", language="ja")
    p2.sample_rate = SR
    p2.tracks[0].add_note(1, 1, 64, "xyzzq", language="en")
    assert any("guessed" in w and "xyzzq" in w for w in render_project(p2).report["warnings"])


# --------------------------------------------------------------------------- params / vocoder regressions
def _frame_params(notes, t, voice="aurora", pitch=64.0):
    """build_params for (onset, duration, pitch, lyric) beat specs at 120 bpm on the frame times ``t``."""
    from synth2.engine.params import build_params
    from synth2.project import Note
    from synth2.timing import TempoMap

    tm = TempoMap()
    tm.set(0, 120)
    plan = plan_track([Note(o, d, p, ly) for o, d, p, ly in notes], tm, "en", get_style("flat"),
                      np.random.default_rng(0))
    z = np.zeros(len(t))
    return build_params(plan, t, get_voice(voice), z + pitch, z, z, z, z, z, z + 1.0)


def test_segments_before_a_partial_window_are_not_squashed_onto_frame_zero():
    """A segment that ends before the first frame (the notes a partial render keeps before its
    start) must not be written into frame 0, which made every preview start with a burst."""
    hop = 64 / SR
    notes = [(1, 1, 64, "sing"), (2, 1, 65, "this"), (3, 1, 66, "song"), (9, 1, 64, "la")]
    tP = 3.0 + np.arange(int(0.5 / hop)) * hop  # a window starting in the rest after "song"
    fp = _frame_params(notes, tP)
    assert fp.voiced.max() == 0.0 and fp.asp.max() == 0.0 and fp.fric.max() == 0.0
    # a window starting mid-vowel (and mid-attack) has the full render's values, no stray /s/
    tF = np.arange(int(2.5 / hop)) * hop
    full = _frame_params(notes, tF)
    for start in (0.5 + 0.02 * hop, 0.9 + 0.01 * hop, 1.53):
        k = int(round(start / hop))
        part = _frame_params(notes, tF[k:])
        m = slice(12, 200)  # past the edge of the smoothing windows
        assert np.allclose(part.voiced[m], full.voiced[k:][m], atol=1e-9)
        assert np.allclose(part.fric[m], full.fric[k:][m], atol=1e-9)
        assert part.fric[:3].max() <= full.fric[k:k + 3].max() + 1e-9  # no stray /s/ from earlier notes
    k = int(round(0.53 / hop))  # 30 ms into the vowel of "sing": the attack ramp is not restarted
    assert _frame_params(notes, tF[k:]).voiced[3] == pytest.approx(full.voiced[k + 3], rel=0.02)


def test_f0_near_or_above_nyquist_stays_bounded():
    """Normalisation gain is capped and the voiced source fades out before Nyquist: no +200 dB
    notes, and one absurd note does not push every other track to silence."""
    for sr, m, cents in ((16000, 108, 1200), (16000, 116, 0), (16000, 120, 0), (22050, 127, 0), (22050, 108, 1200)):
        p = Project.new("t", bpm=240)
        p.sample_rate = sr
        p.tracks[0].add_note(1, 2, m, "aa")
        if cents:
            p.tracks[0].curve("pitchDelta").add(0, cents)
        assert render_track(p, 0)[1]["peak_db"] < 3.0, (sr, m, cents)
    p = Project.new("t", bpm=240)
    p.sample_rate = 16000
    for i in range(4):
        p.tracks[0].add_note(1 + i, 1, 64, "la")
    t2 = p.add_track("hi")
    t2.add_note(3, 0.5, 108, "aa")
    t2.curve("pitchDelta").add(0, 1200)
    res = render_project(p)
    a = res.mono()[int(0.3 * 16000):int(1.2 * 16000)]
    assert 10 * np.log10(np.mean(a.astype(float) ** 2)) > -40
    assert max(r["peak_db"] for r in res.report["tracks"]) < 3.0


def test_breath_noise_follows_the_harmonics_at_high_pitch(monkeypatch):
    """Breath noise is scaled by the power the (few) harmonics carry, so it cannot swamp a high note."""
    import synth2.engine.vocoder as V

    orig = V._noise

    def render(m, noise):
        if not noise:
            monkeypatch.setattr(V, "_noise", lambda *a, **k: orig(*a, **k) * 0)
        p = Project.new("t", bpm=120)
        p.sample_rate = SR
        p.tracks[0].add_note(1, 1.6, m, "aa")
        a, rep = render_track(p, 0)
        monkeypatch.setattr(V, "_noise", orig)
        return np.mean(a[int(0.7 * SR):int(1.1 * SR)].astype(float) ** 2), rep["peak_db"]

    hnr = {}
    for m in (60, 102, 108):
        (tot, peak), (vo, _) = render(m, True), render(m, False)
        hnr[m] = 10 * np.log10(vo / (tot - vo))
        assert peak < -1.0, (m, peak)
    assert hnr[60] > 20 and hnr[102] > 10 and hnr[108] > 10, hnr


def test_formants_above_nyquist_do_not_fold_back():
    """A formant shifted above Nyquist (gender -1, toneShift +1200) fades out instead of showing
    up as a resonance at sr - F; the project sounds about the same at any export rate."""
    import synth2.engine.vocoder as V

    F = np.array([[2062.0, 4302.0, 7158.0, 11102.0, 13368.0]])
    B = np.array([[150.0, 220.0, 290.0, 200.0, 270.0]])
    f = np.arange(100, 7000, 10.0)
    ref = V.vocal_tract_db(f, F, B, np.zeros(1), np.full(1, 1500.0), 96000)[0]
    for sr in (16000, 22050):
        vt = V.vocal_tract_db(f, F, B, np.zeros(1), np.full(1, 1500.0), sr)[0]
        peaks = f[1:-1][(vt[1:-1] > vt[:-2]) & (vt[1:-1] > vt[2:])]
        assert all(np.min(np.abs(F[0, :3] - x)) < 150 for x in peaks), (sr, peaks)
        assert np.max(np.abs(vt - ref)[f < 0.4 * sr]) < 8.0
    peaks = []
    for sr in (16000, 44100):
        p = Project.new("t", bpm=240, voice="pixel")
        p.sample_rate = sr
        t = p.tracks[0]
        t.add_note(1, 2, 64, "aa")
        t.curve("gender").add(0, -1)
        t.curve("toneShift").add(0, 1200)
        peaks.append(render_track(p, 0)[1]["peak_db"])
    assert abs(peaks[0] - peaks[1]) < 3.0, peaks


def test_low_bass_pulses_keep_their_whole_open_phase(monkeypatch):
    """The pulse FFT grows for very low f0, so the anticausal open phase is not cut into a
    per-period step (buzz) at 22.05 or 48 kHz."""
    import synth2.engine.vocoder as V

    orig, first = V._ola, {}

    def spy(out, starts, blocks):
        first.setdefault("b", blocks[len(blocks) // 2].copy())  # the pulses are overlap-added first
        return orig(out, starts, blocks)

    monkeypatch.setattr(V, "_ola", spy)
    for sr in (22050, 48000):
        for tension, breath in ((0, 0), (-1, 1)):
            first.clear()
            p = Project.new("t", bpm=240, voice="basso")
            p.sample_rate = sr
            p.tracks[0].add_note(1, 2, 24, "aa")
            if tension:
                p.tracks[0].curve("tension").add(0, tension)
                p.tracks[0].curve("breathiness").add(0, breath)
            render_track(p, 0)
            b = first["b"]
            assert np.abs(b[4:8]).max() < 0.02 * np.abs(b).max(), (sr, tension)


def test_breath_noise_spectra_cover_low_f0(monkeypatch):
    """Below ~57 Hz the pulses are more than two envelope-grid rows apart; the breath-noise
    spectra must still be filled on every voiced row (no gating on and off at the grid rate)."""
    import synth2.engine.vocoder as V

    orig, cap = V._noise, {}

    def spy(fp, voice, sr, rng, gain, v_amp, phase, hop, n_samples, vmag, N):
        cap["vmag"], cap["v_amp"] = vmag.copy(), v_amp.copy()
        return orig(fp, voice, sr, rng, gain, v_amp, phase, hop, n_samples, vmag, N)

    monkeypatch.setattr(V, "_noise", spy)
    for m in (24, 28):
        p = Project.new("t", bpm=240, voice="basso")
        p.sample_rate = SR
        p.tracks[0].add_note(1, 2, m, "aa")
        p.tracks[0].curve("breathiness").add(0, 1.0)
        render_track(p, 0)
        vm, va = cap["vmag"], cap["v_amp"]
        active = va[np.minimum(np.arange(len(vm)) * V.GRID, len(va) - 1)] > 1e-3 * va.max()
        assert active.sum() > 20 and not np.any(active & (vm.max(axis=1) == 0)), m


def test_vocoder_memory_stays_small_per_sample(monkeypatch):
    """Per-sample bookkeeping is done in blocks and in single precision (was ~70 bytes/sample)."""
    import tracemalloc

    import synth2.engine.renderer as R

    orig, st = R.synthesize, {}

    def spy(fp, voice, sr, rng):
        tracemalloc.start()
        try:
            a = orig(fp, voice, sr, rng)
            st["peak"] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
        st["n"] = len(a)
        return a

    monkeypatch.setattr(R, "synthesize", spy)
    p = Project.new("t", bpm=120)
    p.sample_rate = SR
    for i in range(40):  # 20 s
        p.tracks[0].add_note(1 + i, 1, 60 + (i * 5) % 12, "la")
    render_track(p, 0)
    assert st["peak"] / st["n"] < 45, st["peak"] / st["n"]


def test_a_sound_that_is_not_installed_is_reported_once_in_the_arrangers_own_words(tmp_path, monkeypatch):
    from synth2 import arranger

    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path / "no-sounds"))
    p = Project.new("song", bpm=120, voice="aurora")
    p.sample_rate = 16000
    p.tracks[0].add_note(0, 2, 60, "la")
    p.backing = {"instruments": {"keys": {"preset": "piano", "sound": "no-such-set"},
                                 "drums": {"type": "drums", "kit": "rock", "sound": "no-such-kit"},
                                 "bass": {"preset": "bass_pick"}},
                 "patterns": {"hold": {"kind": "chords", "rhythm": "x---------------"},
                              "beat": {"kind": "drums", "lanes": {"kick": "x...x...x...x...", "snare": "....x.......x..."}},
                              "root": {"kind": "line", "steps": "R...R...R...R..."}},
                 "sections": [{"name": "A", "bars": 1, "chords": ["C"],
                               "parts": {"keys": "hold", "drums": "beat", "bass": "root"}}]}
    rep = render_project(p).report
    want = arranger.sound_report(p)["warnings"]
    assert len(want) == 2 and all("plays instead" in w for w in want)
    # the same sentences as arranger.sound_report, each once (a kit's stems all carry the kit's warning)
    assert sorted(w for w in rep["warnings"] if "plays instead" in w) == sorted(want)
    assert "credits" not in rep and all("sound" not in b for b in rep["backing"])
    assert {b["name"].split()[0] for b in rep["backing"]} == {"keys", "drums", "bass"}
