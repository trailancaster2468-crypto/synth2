"""The neural engine without models: alignment, phoneme mapping, presets, the voicebank loader,
and the backend's window arithmetic, noise handling and reports against a fake onnxruntime."""
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from synth2.engine.backend import render_options
from synth2.engine.expression import get_style
from synth2.engine.planner import plan_track
from synth2.neural import adlib
from synth2.neural import align as al
from synth2.neural import pitch as pt
from synth2.neural import renderer as nb
from synth2.neural.phonemes import PhonemeMap
from synth2.neural.presets import DEFAULT_DIAL, DIAL_NAMES, DIALS, PRESETS, VoicePreset, preset_for
from synth2.score import parse_score
from synth2.voices import VOICES, get_voice

SR, HOP, BINS = 44100, 512, 8
FRAME = HOP / SR
EN = ("aa ae ah ao aw ax ay b ch d dh dx eh er ey f g hh ih iy jh k l m n ng ow oy p r s sh t th uh uw v w y "
      "z zh").split()
EN_VOWELS = "aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw".split()
JA = "N a b ch cl d e f g h hy i j k m n o p r s sh t ts u w y z".split()
SYMBOLS = ["SP", "AP"] + [f"en/{p}" for p in EN] + [f"ja/{p}" for p in JA]

SONG = """tempo: 120
track: Lead voice=aurora style=pop lang=en
C4 q hello  D4 q world  r h | E4 e sing  E4 e a  G4 q song  r h | C4 h more
"""


def song(text=SONG):
    p = parse_score(text)
    return p, p.tracks[0]


def plan_of(track, project):
    voice = get_voice(track.voice).with_params(track.voice_params)
    style = get_style(track.style or voice.default_style, {})
    return plan_track(track.notes, project.tempo, track.language, style, np.random.default_rng(track.seed))


def identity(ph, ja=False, nxt=None):
    return {"br": "AP", "sil": "SP"}.get(ph, ("ja/" if ja else "en/") + ph)


# ---- alignment ------------------------------------------------------------------------------------
def test_phrases_split_at_rests_and_groups_start_on_note_onsets():
    p, tr = song()
    plan = plan_of(tr, p)
    phrases = al.build_phrases(plan)
    assert len(phrases) == 3
    groups, end = al.phrase_groups(phrases[0], plan, identity)
    assert groups[0]["phones"][0] == "SP" and groups[0]["pos"] == pytest.approx(groups[1]["pos"] - al.PAD)
    vowels = [g for g in groups if g.get("vowel")]
    assert [g["phones"] for g in groups] == [["SP", "en/hh"], ["en/ax", "en/l"], ["en/ow", "en/w"],
                                             ["en/er", "en/l", "en/d"]]
    assert len(vowels) == 3
    onsets = sorted(tm.start for tm in plan.timings)[:2]
    assert vowels[0]["pos"] == pytest.approx(onsets[0])
    assert all(a["pos"] <= b["pos"] for a, b in zip(groups, groups[1:]))
    assert end == pytest.approx(phrases[0][-1].end)


def test_align_keeps_vowels_on_their_positions_and_fills_every_group():
    p, tr = song()
    plan = plan_of(tr, p)
    groups, end = al.phrase_groups(al.build_phrases(plan)[1], plan, identity)
    phones, word_div, word_dur, ph_midi = al.dur_inputs(groups, end, FRAME)
    assert len(phones) == sum(word_div) == len(ph_midi) and len(word_dur) == len(groups)
    pred = 3.0 + np.arange(len(phones)) % 4
    starts = al.align(groups, end, pred, FRAME)
    assert len(starts) == len(phones) - 1
    assert all(b > a for a, b in zip(starts, starts[1:]))
    idx = np.cumsum(word_div)
    for g, i in zip(groups[1:], idx[:-1]):
        assert starts[i - 1] == pytest.approx(g["pos"])  # the first phoneme of every group
    seq, durs, f_start, is_vowel = al.phrase_frames(groups, end, pred, FRAME)
    assert seq[0] == seq[-1] == "SP" and durs[0] == al.HEAD and durs[-1] == al.TAIL
    assert len(seq) == len(durs) == len(is_vowel) and min(durs) >= 1
    assert f_start + sum(durs) == int(round(end / FRAME)) + al.TAIL
    lo, hi = al.phrase_bounds(al.build_phrases(plan)[1], FRAME)
    assert lo <= f_start * FRAME and (f_start + sum(durs)) * FRAME <= hi
    assert sum(is_vowel) == sum(1 for g in groups if g.get("vowel"))


def test_align_min_vowel_takes_time_from_the_consonants():
    groups = [dict(pos=0.5, phones=["SP", "en/s"], midi=60), dict(pos=1.0, phones=["en/aa", "en/s", "en/t"], midi=60,
                                                                  vowel=True)]
    pred = np.array([10.0, 5.0, 0.5, 20.0, 20.0])
    plain = al.align(groups, 1.2, pred, FRAME)
    kept = al.align(groups, 1.2, pred, FRAME, min_vowel=0.04)
    assert plain[2] - plain[1] < 0.01 and kept[2] - kept[1] == pytest.approx(0.04)
    assert kept[1] == pytest.approx(1.0)


def test_consonant_only_phrase_and_split_long():
    unit = types.SimpleNamespace(notes=[0], phones=["s"], start=1.0, end=1.2, breath=False, closure=False, lead=0.0,
                                 onset=[], nuclei=[], coda=[])
    plan = types.SimpleNamespace(timings=[], units=[unit])
    groups, end = al.phrase_groups([unit], plan, identity)
    assert len(groups) == 2 and groups[0]["phones"] == ["SP", "en/s"]
    mk = lambda a, b: types.SimpleNamespace(start=a, end=b, lead=0.0, phones=["a"])
    long = [mk(0, 5), mk(5.05, 12), mk(12.2, 19), mk(19.01, 30)]
    out, cuts = al.split_long([long, [mk(40, 41)]])
    assert cuts >= 1 and len(out) == cuts + 2
    assert [u.start for ph in out for u in ph] == [0, 5.05, 12.2, 19.01, 40]
    assert out[0][-1].end == 12  # cut at the longest gap (0.2 s)
    one = [mk(0, 30)]
    assert al.split_long([one]) == ([one], 0)  # a single unit cannot be cut


def test_slow_motion_grid_maps_both_ways():
    groups = [dict(pos=0.5, phones=["SP", "en/t"], midi=60), dict(pos=1.0, phones=["en/aa"], midi=60, vowel=True),
              dict(pos=1.1, phones=["en/iy", "en/t"], midi=60, vowel=True), dict(pos=1.6, phones=["SP"], midi=60)]
    k = al.slow_factors(groups, 1.9)
    assert k.tolist() == [1.0, 2.0, 2.0, 1.0, 1.0, 1.0]  # the 0.1 s note and the consonant before it
    assert al.slow_factors(groups, 1.9, slow_note=0.15, slow_max=1.2)[2] == pytest.approx(1.2)
    durs = [8, 4, 9, 30, 10, 20]
    slow, to_true, to_slow = al.slow_grid(durs, k)
    assert slow == [8, 8, 18, 30, 10, 20]
    assert len(to_true) == sum(slow) and len(to_slow) == sum(durs)
    assert np.all(np.diff(to_true) > 0) and np.all(np.diff(to_slow) > 0)
    x = np.arange(sum(slow), dtype=np.float32)[:, None] * np.ones((1, 3), np.float32)
    back = al.warp_rows(x, to_slow)
    assert back.shape == (sum(durs), 3) and back.dtype == np.float32
    assert np.allclose(back[:8, 0], np.arange(8))  # unstretched frames are untouched
    assert al.warp_rows(np.ones((1, 2), np.float32), np.zeros(4)).shape == (4, 2)


def test_voicing_floor_lifts_collapsed_vowels_only():
    seq = ["SP", "en/aa", "en/t", "en/iy", "en/s", "en/aa", "SP"]
    durs = [8, 20, 5, 20, 5, 20, 10]
    is_vowel = [False, True, False, True, False, True, False]
    v = np.full(sum(durs), -6.0, np.float32)
    v[33:53] = -40.0  # the second vowel collapsed
    v[28:33] = -50.0  # a consonant stays as predicted
    v[70:78] = -30.0  # the last vowel dies away into the rest
    out = al.floor_voicing(v, seq, durs, is_vowel, FRAME, 6.0)
    assert np.allclose(out[33:53], -12.0) and np.allclose(out[28:33], -50.0) and np.allclose(out[70:78], -30.0)
    assert al.floor_voicing(v, seq, durs, is_vowel, FRAME, None) is v
    assert out.dtype == np.float32


def test_consonant_classes():
    assert [al.cons_class(p) for p in ("en/t", "en/b", "en/ch", "ja/sh", "en/z", "ja/N", "en/ng", "en/_r", "en/aa",
                                       "SP", "AP", "k")] == ["stop", "vstop", "aff", "fric", "vfric", None, "nasal",
                                                             "glide", None, None, None, "stop"]
    assert set(al.CONS_CLASS.values()) == set(al.CONS_CLASSES) and set(al.VOICED_CONS) < set(al.CONS_CLASSES)


def _note_groups(span=0.3):
    """SP, one note "s aa t s" of ``span`` s at 1.0 s, and the note after it."""
    return [dict(pos=0.5, phones=["SP", "en/s"], midi=60),
            dict(pos=1.0, phones=["en/aa", "en/t", "en/s"], midi=60, vowel=True),
            dict(pos=1.0 + span, phones=["en/iy"], midi=62, vowel=True)], 1.5 + span


def test_consonant_caps_give_their_time_to_the_vowel():
    groups, end = _note_groups(0.3)
    pred = np.array([10.0, 5.0, 4.0, 12.0, 8.0, 10.0])  # the note's consonants want 20 of 24 frames
    lens = lambda starts: np.diff(starts[1:5])  # aa, t, s
    plain = lens(al.align(groups, end, pred, FRAME))
    assert plain.sum() == pytest.approx(0.3) and plain[1] == pytest.approx(0.15) and plain[2] == pytest.approx(0.1)
    cap = {"stop": 0.05, "fric": 0.08}
    capped = lens(al.align(groups, end, pred, FRAME, cons_cap=cap))
    assert capped.sum() == pytest.approx(0.3) and capped[1] == pytest.approx(0.05) and capped[2] == pytest.approx(0.08)
    # never under a share of what the duration model gave
    kept = lens(al.align(groups, end, pred, FRAME, cons_cap=cap, cons_cap_keep=0.5))
    assert kept[1] == pytest.approx(0.075) and kept[2] == pytest.approx(0.08)
    # the caps bind only as far as it takes to bring the consonants down to their share of the note
    shared = lens(al.align(groups, end, pred, FRAME, cons_cap=cap, cons_cap_share=0.6))
    assert shared[1] + shared[2] == pytest.approx(0.6 * 0.3) and plain[1] > shared[1] > capped[1]
    assert np.allclose(lens(al.align(groups, end, pred, FRAME, cons_cap=cap, cons_cap_share=0.9)), plain)
    # a class without a cap and the consonants before the first note are left alone
    only_s = al.align(groups, end, pred, FRAME, cons_cap={"fric": 0.02})
    assert np.diff(only_s[1:5])[1] == pytest.approx(0.15) and only_s[1] - only_s[0] == pytest.approx(5 * FRAME)
    seq, durs, _, _ = al.phrase_frames(groups, end, pred, FRAME, cons_cap=cap, cons_cap_keep=0.5)
    assert sum(durs) == sum(al.phrase_frames(groups, end, pred, FRAME)[1])


def test_onset_keep_gives_consonants_after_a_rest_their_predicted_length():
    groups = [dict(pos=0.5, phones=["SP"], midi=60), dict(pos=1.0, phones=["en/aa"], midi=60, vowel=True),
              dict(pos=1.2, phones=["AP", "en/s", "en/t"], midi=60), dict(pos=1.5, phones=["en/iy"], midi=60, vowel=True)]
    pred = np.array([10.0, 10.0, 2.0, 5.0, 4.0, 10.0])
    plain = al.align(groups, 2.0, pred, FRAME)
    kept = al.align(groups, 2.0, pred, FRAME, onset_keep=True)
    assert plain[2] - plain[1] == pytest.approx(0.3 * 2 / 11)  # the breath shares the gap in proportion
    assert kept[3] - kept[2] == pytest.approx(5 * FRAME) and kept[4] - kept[3] == pytest.approx(4 * FRAME)
    assert kept[4] == pytest.approx(1.5) and kept[1] == pytest.approx(1.2)
    crowded = al.align(groups, 2.0, np.array([10.0, 10.0, 2.0, 50.0, 50.0, 10.0]), FRAME, onset_keep=True)
    assert crowded[2] - crowded[1] == pytest.approx(2 * FRAME)  # the rest keeps two frames


BREATH = {"min_gap": 0.12, "fill": 0.45, "length": 0.30, "min_len": 0.10, "breathiness": -44.0, "first": True}
LINES = """tempo: 120
track: Lead voice=aurora style=flat lang=en
C4 q sing  D4 e it  r e  E4 q so  r h | C4 q again
"""


def test_breaths_go_into_the_gaps_between_lines():
    p, tr = song(LINES)
    plan = plan_of(tr, p)
    first, second = al.build_phrases(plan)
    plain, _ = al.phrase_groups(first, plan, identity, FRAME)
    assert [g["phones"][0] for g in plain] == ["SP", "en/ih", "en/ih", "SP", "en/ow"]
    groups, end = al.phrase_groups(first, plan, identity, FRAME, BREATH)
    starters = [g["phones"][0] for g in groups]
    assert starters == ["SP", "AP", "en/ih", "en/ih", "AP", "en/ow"]  # before the phrase, and filling the short gap
    on = first[0].start - first[0].lead
    assert groups[1]["pos"] == pytest.approx(on - 0.30) and groups[0]["pos"] == pytest.approx(groups[1]["pos"] - al.PAD)
    assert groups[4]["pos"] == pytest.approx(first[1].end)
    assert all(b["pos"] >= a["pos"] + FRAME - 1e-9 for a, b in zip(groups, groups[1:]))
    # a longer gap rests first and breathes over its last part
    long = [g["phones"][0] for g in al.phrase_groups(first, plan, identity, FRAME, dict(BREATH, fill=0.1))[0]]
    assert long == ["SP", "AP", "en/ih", "en/ih", "SP", "AP", "en/ow"]
    # the silence before a phrase: room after the phrase before it, none at the start when "first" is off
    assert al.phrase_groups(second, plan, identity, FRAME, BREATH, first[-1].end)[0][1]["phones"] == ["AP"]
    near = second[0].start - second[0].lead - 0.15
    assert al.phrase_groups(second, plan, identity, FRAME, BREATH, near)[0][1]["phones"][0] != "AP"
    short = al.phrase_groups(second, plan, identity, FRAME, BREATH, near - 0.1)[0]
    assert short[1]["phones"] == ["AP"] and short[2]["pos"] - short[1]["pos"] < 0.2  # as long as the room allows
    assert al.phrase_groups(first, plan, identity, FRAME, dict(BREATH, first=False))[0][1]["phones"][0] != "AP"
    # no breath where the track is told to stay quiet
    avoid = dict(BREATH, avoid=[(on - 0.2, on - 0.1)])
    assert [g["phones"][0] for g in al.phrase_groups(first, plan, identity, FRAME, avoid)[0]][1] == "en/ih"
    # a breath the consonants leave too little room for is a rest
    phones, _, _, _ = al.dur_inputs(groups, end, FRAME)
    pred = np.full(len(phones), 4.0)
    seq, durs, f_start, _ = al.phrase_frames(groups, end, pred, FRAME, min_breath=0.10, onset_keep=True)
    assert seq.count("AP") == 2
    seq2, durs2, _, _ = al.phrase_frames(groups, end, pred, FRAME, min_breath=0.5, onset_keep=True)
    assert "AP" not in seq2 and durs2 == durs
    lo, hi = al.phrase_bounds(first, FRAME, BREATH["length"])
    assert lo <= f_start * FRAME and (f_start + sum(durs)) * FRAME <= hi


def test_voiced_consonants_and_breaths_in_the_variance_curves():
    seq = ["SP", "AP", "en/m", "en/aa", "en/z", "en/iy", "en/t", "en/aa", "SP"]
    durs = [8, 20, 6, 30, 6, 30, 6, 30, 10]
    is_vowel = [p in ("en/aa", "en/iy") for p in seq]
    idx = np.concatenate([[0], np.cumsum(durs)])
    var = {"voicing": np.full(sum(durs), -8.0, np.float32), "breathiness": np.full(sum(durs), 3.0, np.float32)}
    for i in (2, 4, 6):
        var["voicing"][idx[i]:idx[i + 1]] = -40.0
    out = al.shape_consonants({k: v.copy() for k, v in var.items()}, seq, durs, is_vowel, FRAME,
                              {"nasal": 8.0, "vfric": 11.0, "vstop": 16.0}, -44.0)
    v, br = out["voicing"], out["breathiness"]
    assert np.allclose(v[idx[2]:idx[3]], -16.0) and np.allclose(v[idx[4]:idx[5]], -19.0)  # held under the vowels
    assert np.allclose(v[idx[6]:idx[7]], -40.0) and np.allclose(v[idx[3]:idx[4]], -8.0)  # an unvoiced stop is not
    assert np.allclose(v[idx[1]:idx[2]], -96.0)  # a breath has no voice
    b = br[idx[1]:idx[2]]
    assert b.max() == pytest.approx(-44.0, abs=1.0) and b[0] < b.max() - 3 and b[-1] < b.max() - 15
    assert np.allclose(br[idx[3]:], 0.0) and br.dtype == np.float32  # clipped to the range the model takes
    assert np.allclose(al.breath_shape(20, FRAME), b + 44.0, atol=1e-4)
    # without breath shaping the breath keeps the prediction; without floors the consonants do
    keep = al.shape_consonants({k: x.copy() for k, x in var.items()}, seq, durs, is_vowel, FRAME, None, None)
    assert np.allclose(keep["voicing"][idx[1]:idx[3]], [-8.0] * 20 + [-40.0] * 6)
    w = al.vowel_weight(durs, is_vowel)
    assert w.dtype == np.float32 and w.sum() == 3 * 29 and w[idx[3]] == 0.5 and w[idx[3] + 1] == 1 and w[idx[2]] == 0


def test_stops_are_sung_long_and_cut_out_around_their_release():
    seq, durs = ["SP", "en/t", "en/aa", "en/s", "en/k", "en/aa", "SP"], [8, 3, 20, 4, 12, 20, 10]
    k = al.slow_consonants(np.ones(len(seq)), seq, durs, FRAME, {"stop": 0.09})
    assert k[1] == pytest.approx(0.09 / (3 * FRAME)) and k[4] == 1.0 and k[3] == 1.0  # long enough, or not a stop
    assert al.slow_consonants(np.ones(len(seq)), seq, [8, 1] + durs[2:], FRAME, {"stop": 0.09})[1] == al.SLOW_CONS_MAX
    assert al.slow_consonants(np.full(len(seq), 5.0), seq, durs, FRAME, {"stop": 0.09})[1] == 5.0
    slow, to_true, to_slow = al.slow_grid(durs, k)
    assert slow[1] == 8 and slow[2:] == durs[2:]
    mel_slow = np.zeros((sum(slow), 8), np.float32)
    mel_slow[:, 0] = np.arange(sum(slow))  # row numbers
    mel_slow[8:16, 4:] = [[-10.0], [-12.0], [-12.0], [-11.0], [-12.0], [-2.0], [-1.0], [-6.0]]  # closure, release at 5
    mel = al.warp_rows(mel_slow, to_slow)
    out = al.squeeze_bursts(mel, mel_slow, seq, durs, slow, ["stop", "aff"])
    assert list(out[8:11, 0]) == [12.0, 13.0, 14.0]  # one frame of closure, the release, the frame after it
    assert np.array_equal(out[:8], mel[:8]) and np.array_equal(out[11:], mel[11:])
    assert not np.array_equal(out[8:11], mel[8:11]) and out.shape == mel.shape
    assert np.array_equal(al.squeeze_bursts(mel, mel_slow, seq, durs, slow, ["aff"]), mel)


def test_consonant_gain_after_the_vocoder():
    seq, durs = ["SP", "en/s", "en/aa", "en/t", "en/aa", "SP"], [8, 10, 20, 6, 20, 10]
    idx = np.concatenate([[0], np.cumsum(durs)]) * HOP
    wav = (0.1 * np.sin(2 * np.pi * 6000 * np.arange(idx[-1]) / SR)).astype(np.float32)
    out = al.consonant_gain(wav, seq, durs, {"fric": 6.0, "stop": -6.0}, HOP, SR)
    rms = lambda x, a, b: float(np.sqrt(np.mean(x[a:b].astype(np.float64) ** 2)))
    mid = lambda i: (idx[i] + 600, idx[i + 1] - 600)
    assert 20 * np.log10(rms(out, *mid(1)) / rms(wav, *mid(1))) == pytest.approx(6.0, abs=0.05)
    assert 20 * np.log10(rms(out, *mid(3)) / rms(wav, *mid(3))) == pytest.approx(-6.0, abs=0.05)
    assert np.array_equal(out[idx[2] + 600:idx[3] - 600], wav[idx[2] + 600:idx[3] - 600]) and out.dtype == np.float32
    assert out[idx[4] + 300] != wav[idx[4] + 300]  # a stop's gain runs 10 ms into the vowel after it
    assert al.consonant_gain(wav, seq, durs, {"nasal": 6.0}, HOP, SR) is wav
    low = (0.1 * np.sin(2 * np.pi * 200 * np.arange(idx[-1]) / SR)).astype(np.float32)
    high_only = al.consonant_gain(low, seq, durs, {"fric": 6.0}, HOP, SR, hf=2000.0)
    assert 20 * np.log10(rms(high_only, *mid(1)) / rms(low, *mid(1))) == pytest.approx(0.0, abs=0.1)


def test_phrase_seed_is_stable_and_depends_on_place_only():
    assert al.phrase_seed(1, 100, 500) == al.phrase_seed(1, 100, 500)
    assert 0 <= al.phrase_seed(17, -3, 9) < 2 ** 31
    assert len({al.phrase_seed(1, 100, 500), al.phrase_seed(2, 100, 500), al.phrase_seed(1, 101, 500),
                al.phrase_seed(1, 100, 501)}) == 4
    assert al.phrase_seed(1, 0, 870) == 787971629


# ---- pitch helpers --------------------------------------------------------------------------------
def test_build_notes_pads_rests_and_covers_the_phrase():
    seq = ["SP", "en/t", "en/aa", "en/iy", "SP"]
    durs = [8, 3, 20, 20, 10]
    vowel = lambda p: p in ("SP", "AP", "en/aa", "en/iy")
    notes = [pt.PhraseNote(11, 31, 60.0), pt.PhraseNote(31, 51, 64.0, slur=True)]
    ni = pt.build_notes(notes, seq, durs, vowel)
    assert int(ni.dur.sum()) == sum(durs) and ni.rest.tolist() == [True, False, False, True]
    assert ni.midi.tolist() == [60.0, 60.0, 64.0, 64.0]
    assert len(ni.frame_midi()) == sum(durs)
    base = pt.base_pitch(ni)
    assert base[0] == pytest.approx(60.0) and base[-1] == pytest.approx(64.0) and 60.5 < base[31] < 63.5
    pred = ni.frame_midi().copy()
    pred[20] -= 12.0
    excess = pt.range_excess(pred, ni)
    assert excess[20] == pytest.approx(12.0) and excess[40] == 0.0
    fixed = pt.bridge(pred, excess > pt.MAX_DEV)
    assert fixed[20] == pytest.approx(60.0)
    w = pt.soft_window(np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0], bool), 2)
    assert w[0] == 0 and w[5] == 1 and 0 <= w[2] < 1
    with pytest.raises(ValueError):
        pt.build_notes([], seq, durs, vowel)


def test_bank_pitch_falls_back_to_synth2_curve():
    p, tr = song()
    plan = plan_of(tr, p)
    t = np.arange(600) * FRAME
    s2 = np.full(600, 60.0)
    bp = pt.BankPitch(t, plan.timings, get_style("pop"), s2, s2, np.ones(600), 1.0, 10, "auto", lambda p_: True, FRAME)
    sl = slice(0, 100)

    def broken(*a, **k):
        raise RuntimeError("no model")

    seq, durs = ["SP", "en/aa", "SP"], [8, 82, 10]
    units = al.build_phrases(plan)[0]
    ni = bp.notes(units, seq, durs, 0)
    curve, why, repaired = bp.phrase(broken, ni, seq, durs, sl)
    assert np.array_equal(curve, s2[sl]) and "no model" in why and repaired == 0
    good = lambda seq, durs, ni, expr, steps, **kw: ni.frame_midi() + 0.1
    curve, why, repaired = bp.phrase(good, ni, seq, durs, sl)
    assert why == "" and curve.shape == (100,)
    far = lambda seq, durs, ni, expr, steps, **kw: ni.frame_midi() + 20.0
    assert "out of range" in bp.phrase(far, ni, seq, durs, sl)[1]
    assert bp.key(ni, sl)[0] == "bank" and bp.key(None, sl)[0] == "expression"
    assert pt.default_expr(get_style("robotic"), get_voice("nova")) == 0.5
    assert pt.default_expr(get_style("pop"), get_voice("aurora")) == 1.0
    assert pt.default_vibrato(get_style("robotic"), tr) == "synthetic"
    assert pt.default_vibrato(get_style("pop"), tr) == "auto"


# ---- phoneme mapping ------------------------------------------------------------------------------
def test_phoneme_map_english_japanese_and_specials():
    pm = PhonemeMap(set(SYMBOLS).__contains__)
    assert pm.convert("aa") == "en/aa" and pm.convert("br") == "AP" and pm.convert("sil") == "SP"
    assert pm.convert("q") == "SP" and pm.convert("q", ja=True) == "ja/cl"
    assert pm.convert("a", ja=True) == "ja/a" and pm.convert("N", ja=True) == "ja/N"
    assert [pm.convert(x, ja=True) for x in ("ph", "dz", "jh", "dx", "hh")] == ["ja/f", "ja/z", "ja/j", "ja/r", "ja/h"]
    assert pm.convert("c", ja=True, nxt="i") == "ja/h" and pm.convert("c", ja=True, nxt="a") == "ja/hy"
    assert pm.warnings() == []


def test_phoneme_map_falls_back_and_reports():
    have = set(SYMBOLS) - {"en/ax", "en/zh", "ja/hy", "ja/ts"}
    pm = PhonemeMap(have.__contains__)
    assert pm.convert("ax") == "en/ah" and pm.convert("zh") == "en/jh"
    assert pm.convert("c", ja=True, nxt="a") == "ja/h"
    assert pm.convert("ts", ja=True) == "ja/t"
    assert pm.convert("aa", ja=True) == "en/aa"  # borrowed from the other language
    assert pm.convert("v", ja=True) == "en/v"
    pm2 = PhonemeMap({"SP", "AP", "aa"}.__contains__)
    assert pm2.convert("aa") == "aa" and pm2.convert("k") == "SP"
    assert any("en/ax as en/ah" in w for w in pm.warnings())
    assert any("en/k" in w and "silent" in w for w in pm2.warnings())


# ---- presets --------------------------------------------------------------------------------------
def test_every_voice_has_a_preset_and_overrides_are_numeric():
    assert set(PRESETS) == set(VOICES)
    assert preset_for("aurora") == VoicePreset(level_db=2.0)
    # a level per voice: the low voices come out of the bank quieter
    assert [PRESETS[v].level_db for v in ("pixel", "nova", "atlas", "ember", "basso")] == [0.0, 1.5, 4.5, 6.0, 8.5]
    # the low voices sing with the acoustic model an octave above the rendered pitch
    assert [preset_for(v).tone for v in ("basso", "atlas", "ember")] == [12.0, 12.0, 12.0]
    assert preset_for("atlas").gender == pytest.approx(-0.3) and preset_for("nova").tone == 0.0
    assert preset_for("no-such-voice") == VoicePreset()
    p = preset_for("atlas", {"neural.gender": 0.2, "neural.tone": -3, "neural.breath_db": 4, "neural.voicing_db": -2,
                             "neural.velocity": 9, "neural.level_db": -5, "neural.tension": 1.5, "neural.expr": 0.75,
                             "neural.slow": 0, "neural.voicing_floor": 3, "neural.gender2": 1, "neural.tone ": "x",
                             "neural.level_db_bad": "loud"})
    assert (p.gender, p.tone, p.breath_db, p.voicing_db, p.level_db) == (0.2, -3.0, 4.0, -2.0, -5.0)
    assert p.velocity == 2.0 and p.tension == 1.5 and p.expr == 0.75 and p.slow == 0.0 and p.voicing_floor == 3.0
    assert preset_for("aurora", {"neural.gender": "high"}).gender == 0.0
    assert preset_for("aurora", {"neural.gender": 9}).gender == 1.2


def test_preset_effects_modes_and_formant_engine_parameters():
    p = preset_for("nova", {"neural.fx.radio": 0.5, "neural.fx.alien": 2, "neural.fx.double": 0.3, "neural.fx.zap": 1,
                            "neural.fx.crush": 0, "neural.mode.Power": 3, "neural.mode.soft": 1, "robotic": 0.9})
    assert [n for n, _ in p.fx] == ["double", "alien", "radio", "zap"]
    assert p.fx_spec() == ["double:0.3", "alien:1", "radio:0.5", "zap:1"]
    assert p.mode_weights() == {"power": 3.0, "soft": 1.0}
    assert preset_for("nova", {"robotic": 1.0}).fx == ()  # robotic lowers expr, it is not an effect
    up = preset_for("aurora", {"formant_scale": VOICES["aurora"].formant_scale * 2 ** (2.5 / 12)})
    assert up.gender == pytest.approx(0.5)
    assert preset_for("aurora", {"breathiness": 0.5}).breath_db == pytest.approx(6.0)
    s = p.summary()
    assert json.loads(json.dumps(s))["fx"][0] == {"preset": "double", "amount": 0.3} and s["modes"]["power"] == 3.0


def test_scores_accept_neural_voice_parameters():
    p = parse_score("tempo: 120\ntrack: Lead voice=atlas voice.neural.gender=0.2 voice.neural.fx.alien=0.6\nC4 q la\n")
    params = p.tracks[0].voice_params
    assert params["neural.gender"] == 0.2 and params["neural.fx.alien"] == 0.6
    assert preset_for("atlas", params).gender == 0.2 and preset_for("atlas", params).fx == (("alien", 0.6),)


def test_the_dial_is_one_named_setting_and_every_part_can_be_overridden(monkeypatch):
    monkeypatch.delenv("SYNTH2_NEURAL_DIAL", raising=False)
    assert DEFAULT_DIAL == "crisp" and VoicePreset().dial == "crisp" and tuple(DIALS) == DIAL_NAMES
    crisp = preset_for("nova")
    assert dict(crisp.cons_cap) == DIALS["crisp"]["cons_cap"] and crisp.cons_cap_share == 0.4
    assert dict(crisp.burst) == {"stop": 0.09, "aff": 0.12} and dict(crisp.cons_floor)["vstop"] == 16.0
    assert crisp.cons_cap_keep == 0.5 and crisp.onset_keep and crisp.vowel_breath_db == -3.0 and crisp.detune is None
    for value, name in ((0, "smooth"), (1, "balanced"), (2, "crisp"), ("balanced", "balanced"), (1.4, "balanced"),
                        (7, "crisp"), ("loud", "crisp")):
        assert preset_for("nova", {"neural.dial": value}).dial == name
    smooth, balanced = preset_for("nova", {"neural.dial": 0}), preset_for("nova", {"neural.dial": 1})
    assert smooth.cons_cap_share is None and smooth.burst == () and dict(smooth.cons_cap)["stop"] == 0.035
    assert balanced.cons_cap_share == 0.35 and balanced.burst == () and dict(balanced.cons_floor)["vstop"] == 14.0
    monkeypatch.setenv("SYNTH2_NEURAL_DIAL", "smooth")
    assert preset_for("nova").dial == "smooth" and preset_for("nova", {"neural.dial": 2}).dial == "crisp"
    monkeypatch.setenv("SYNTH2_NEURAL_DIAL", "1")
    assert preset_for("nova").dial == "balanced"
    monkeypatch.setenv("SYNTH2_NEURAL_DIAL", "sharp")
    assert preset_for("nova").dial == "crisp"
    monkeypatch.delenv("SYNTH2_NEURAL_DIAL")
    p = preset_for("nova", {"neural.dial": 1, "neural.cons_cap.stop": 0.1, "neural.cons_cap.glide": 0,
                            "neural.cons_cap_share": 0, "neural.cons_cap_keep": 0.25, "neural.onset_keep": 0,
                            "neural.cons_floor.vstop": 20, "neural.burst": 1, "neural.burst.aff": 0,
                            "neural.vowel_breath_db": -1, "neural.detune": 0, "neural.cons_gain.stop": -4,
                            "neural.cons_gain.hf": 2000, "neural.breaths": 0.5, "neural.cons_cap.vowel": 1})
    cap = dict(p.cons_cap)
    assert cap["stop"] == 0.1 and "glide" not in cap and cap["fric"] == 0.065 and "vowel" not in cap
    assert p.cons_cap_share is None and p.cons_cap_keep == 0.25 and p.onset_keep is False
    assert dict(p.cons_floor) == {"vfric": 8.0, "nasal": 8.0, "glide": 8.0, "vstop": 20.0}
    assert dict(p.burst) == {"stop": 0.09} and p.vowel_breath_db == -1.0 and p.detune == 0.0
    assert dict(p.cons_gain) == {"stop": -4.0} and p.cons_gain_hf == 2000.0
    assert p.breath()["breathiness"] == pytest.approx(-44.0 + 20 * np.log10(0.5)) and p.breath()["length"] == 0.30
    off = preset_for("nova", {"neural.cons_cap": 0, "neural.cons_floor": 0, "neural.burst": 0, "neural.breaths": 0})
    assert off.cons_cap == () and off.cons_floor == () and off.burst == () and off.breath() is None
    one = preset_for("nova", {"neural.cons_cap": 0.04, "neural.cons_floor": 10, "neural.burst": 0.1})
    assert set(dict(one.cons_cap).values()) == {0.04} and len(one.cons_cap) == 7
    assert dict(one.cons_floor) == {c: 10.0 for c in ("nasal", "glide", "vstop", "vfric")}
    assert dict(one.burst) == {"stop": 0.1, "aff": 0.1}
    s = json.loads(json.dumps(p.summary()))
    assert s["dial"] == "balanced" and s["cons_cap"]["stop"] == 0.1 and s["burst"] == {"stop": 0.09}
    assert preset_for("nova").breath()["breathiness"] == -44.0


# ---- a fake onnxruntime and a fake voicebank ------------------------------------------------------
class FakeOrt(types.ModuleType):
    """Stands in for onnxruntime. A model file is a JSON description of its inputs and outputs;
    a model marked noisy returns, on its first run, a function of the seed set before it was
    created, and something else on every later run."""

    def __init__(self):
        super().__init__("onnxruntime")
        self.__spec__ = importlib.machinery.ModuleSpec("onnxruntime", None)
        self.seed, self.created, self.calls = None, [], []
        mod = self

        class SessionOptions:
            intra_op_num_threads = 0
            log_severity_level = 0

        class InferenceSession:
            def __init__(self, path, so=None, providers=None):
                self.name = Path(path).parent.name + "/" + Path(path).name
                self.spec = json.loads(Path(path).read_text().split("\n")[0])
                self.seed, self.runs, self.threads = mod.seed, 0, so.intra_op_num_threads
                mod.created.append((self.name, mod.seed))

            def get_inputs(self):
                return [types.SimpleNamespace(name=n, shape=s) for n, s in self.spec["inputs"]]

            def get_outputs(self):
                return [types.SimpleNamespace(name=n) for n in self.spec["outputs"]]

            def run(self, names, feed):
                assert list(feed) == [n for n, _ in self.spec["inputs"]], (self.name, list(feed))
                mod.calls.append((self.name, dict(feed)))
                noise = 0.0
                if self.spec.get("noisy"):
                    noise = ((self.seed or 0) % 997) * 1e-5 + self.runs * 1e-2
                self.runs += 1
                out = []
                for name in self.spec["outputs"]:
                    if name == "encoder_out":
                        out.append(np.zeros((1, feed["tokens"].shape[1], 4), np.float32))
                    elif name == "x_masks":
                        out.append(np.zeros((1, feed["tokens"].shape[1]), bool))
                    elif name == "ph_dur_pred":
                        n = feed["encoder_out"].shape[1]
                        out.append((4.0 + np.arange(n) % 3)[None].astype(np.float32))
                    elif name == "pitch_pred":
                        steps = np.repeat(feed["note_midi"][0], feed["note_dur"][0])
                        out.append((steps + noise)[None].astype(np.float32))
                    elif name.endswith("_pred"):
                        out.append(np.full((1, feed["pitch"].shape[1]), -6.0 - noise, np.float32))
                    elif name == "mel":
                        mel = np.zeros((1, feed["f0"].shape[1], BINS), np.float32)
                        mel[0, :, 0] = feed["voicing"][0]
                        mel[0, :, 1] = noise
                        out.append(mel)
                    elif name == "waveform":
                        f0 = np.repeat(feed["f0"][0].astype(np.float64), HOP)
                        amp = 0.1 + np.repeat(feed["mel"][0, :, 1].astype(np.float64), HOP)
                        out.append((amp * np.sin(2 * np.pi * np.cumsum(f0) / SR))[None].astype(np.float32))
                return out

        self.SessionOptions, self.InferenceSession = SessionOptions, InferenceSession

    def set_seed(self, n):
        self.seed = n


def _model(path, inputs, outputs, noisy=False, tail=""):
    spec = {"inputs": [[n, [1, "n"] if n != "spk_embed" else [1, "n", 6]] for n in inputs], "outputs": outputs,
            "noisy": noisy}
    path.write_text(json.dumps(spec) + "\n" + tail)


def make_bank(root: Path, speakers=(), tension=False, pitch=True, variance=True, vocoder=True):
    """A voicebank folder in OpenUtau layout with fake models."""
    yaml_list = lambda key, items: f"{key}:\n" + "".join(f"  - {i}\n" for i in items) if items else ""
    spk = ["spk_embed"] if speakers else []
    table = json.dumps({s: i for i, s in enumerate(SYMBOLS)})
    langs = json.dumps({"en": 1, "ja": 2})
    var = ["breathiness", "voicing"] + (["tension"] if tension else [])
    for sub in ["", "dsdur"] + (["dsvariance"] if variance else []) + (["dspitch"] if pitch else []):
        d = root / sub
        d.mkdir(parents=True, exist_ok=True)
        (d / "phonemes.json").write_text(table, encoding="utf-8-sig")
        (d / "languages.json").write_text(langs)
        for s in speakers:
            (d / "embeds").mkdir(exist_ok=True)
            np.full(6, 1.0 + len(s), "<f4").tofile(str(d / "embeds" / f"{s}.emb"))
    common = ("phonemes: phonemes.json\nlanguages: languages.json\nuse_lang_id: true\n"
              f"sample_rate: {SR}\nhop_size: {HOP}\n")
    spk_yaml = yaml_list("speakers", [f"embeds/{s}" for s in speakers])
    (root / "dsconfig.yaml").write_text(
        "﻿" + common + f"acoustic: acoustic.onnx\nnum_mel_bins: {BINS}\nmax_depth: 0.4\nvocoder: fake_voc\n"
        "augmentation_args:\n  random_pitch_shifting:\n    range: [-5.0, 5.0]\n" + spk_yaml, encoding="utf-8")
    _model(root / "acoustic.onnx", ["tokens", "languages", "durations", "f0"] + (var if variance else [])
           + ["gender", "velocity"] + spk + ["depth", "steps"], ["mel"], noisy=True)
    (root / "dsdur" / "dsconfig.yaml").write_text(common + "linguistic: linguistic.onnx\ndur: dur.onnx\n" + spk_yaml)
    _model(root / "dsdur" / "linguistic.onnx", ["tokens", "languages", "word_div", "word_dur"],
           ["encoder_out", "x_masks"])
    _model(root / "dsdur" / "dur.onnx", ["encoder_out", "x_masks", "ph_midi"] + spk, ["ph_dur_pred"])
    if variance:
        (root / "dsvariance" / "dsconfig.yaml").write_text(
            common + "linguistic: linguistic.onnx\nvariance: variance.onnx\n" + spk_yaml)
        _model(root / "dsvariance" / "linguistic.onnx", ["tokens", "languages", "ph_dur"], ["encoder_out", "x_masks"])
        _model(root / "dsvariance" / "variance.onnx", ["encoder_out", "ph_dur", "pitch"] + var + ["retake"] + spk
               + ["steps"], [f"{v}_pred" for v in var], noisy=True)
    if pitch:
        (root / "dspitch" / "dsconfig.yaml").write_text(
            common + "linguistic: linguistic.onnx\npitch: pitch.onnx\nuse_expr: true\nuse_note_rest: true\n"
            "use_continuous_acceleration: true\n" + spk_yaml)
        (root / "dspitch" / "dsdict.yaml").write_text(
            "entries: []\nsymbols:\n- symbol: SP\n  type: vowel\n- {symbol: en/t, type: stop}\n"
            + "".join(f"- symbol: en/{v}\n  type: vowel\n" for v in EN_VOWELS))
        _model(root / "dspitch" / "linguistic.onnx", ["tokens", "languages", "ph_dur"], ["encoder_out", "x_masks"])
        _model(root / "dspitch" / "pitch.onnx", ["encoder_out", "ph_dur", "note_midi", "note_rest", "note_dur", "pitch",
                                                 "expr", "retake"] + spk + ["steps"], ["pitch_pred"], noisy=True)
    if vocoder:
        (root / "dsvocoder").mkdir(exist_ok=True)
        (root / "dsvocoder" / "vocoder.yaml").write_text(
            f"name: fake_voc\nmodel: voc.onnx\nsample_rate: {SR}\nhop_size: {HOP}\nnum_mel_bins: {BINS}\n")
        _model(root / "dsvocoder" / "voc.onnx", ["mel", "f0"], ["waveform"])
    if speakers:
        (root / "character.yaml").write_text(
            "subbanks:\n" + "".join(f'- {{color: "{"" if i == 0 else s.title()}", suffix: embeds/{s}}}\n'
                                    for i, s in enumerate(speakers)))
    return root


def install(models_dir: Path, bank_id="fake", **kw) -> Path:
    folder = models_dir / bank_id
    make_bank(folder / "files", **kw)
    (folder / "synth2-model.json").write_text(json.dumps({
        "format": "synth2-model", "version": 1, "id": bank_id, "kind": "voicebank", "name": "Fake", "root": "files",
        "credits": ["Fake singer (test)"], "default": bank_id == "fake"}))
    return folder / "files"


def install_vocoder(models_dir: Path, vid: str, name: str, **manifest) -> None:
    d = models_dir / vid / "files"
    d.mkdir(parents=True)
    (d / "vocoder.yaml").write_text(f"name: {name}\nmodel: v.onnx\nsample_rate: {SR}\nhop_size: {HOP}\n"
                                    f"num_mel_bins: {BINS}\n")
    _model(d / "v.onnx", ["mel", "f0"], ["waveform"])
    (models_dir / vid / "synth2-model.json").write_text(json.dumps({
        "format": "synth2-model", "id": vid, "kind": "vocoder", "root": "files", "credits": [f"Vocoder {vid}"],
        **manifest}))


def remove_bundled_vocoder(root: Path) -> None:
    for f in (root / "dsvocoder").iterdir():
        f.unlink()
    (root / "dsvocoder").rmdir()


@pytest.fixture
def fake(tmp_path, monkeypatch):
    """A models folder with one fake voicebank and a fake onnxruntime; returns the fake module."""
    pytest.importorskip("yaml")
    from synth2 import neural

    ort = FakeOrt()
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setenv("SYNTH2_MODELS", str(tmp_path / "models"))
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "0")
    monkeypatch.setenv("SYNTH2_CACHE", str(tmp_path / "cache"))  # a test that turns the cache on writes here
    monkeypatch.setenv("SYNTH2_NEURAL_THREADS", "3")
    for var in ("SYNTH2_NEURAL_STEPS", "SYNTH2_NEURAL_REPRODUCIBLE", "SYNTH2_NEURAL_PITCH", "SYNTH2_NEURAL_BANK",
                "SYNTH2_NEURAL_DIAL", "SYNTH2_QUALITY"):
        monkeypatch.delenv(var, raising=False)
    (tmp_path / "models").mkdir()
    ort.root = install(tmp_path / "models")
    ort.models = tmp_path / "models"
    neural.refresh()
    nb.reset()
    yield ort
    nb.reset()
    neural.refresh()


def sing(project, track, sr=SR, t0=0.0, t1=None, note_range=None, bank=None):
    t1 = project.tempo.beat_to_sec(track.end) + 0.6 if t1 is None else t1
    return nb.get_backend(bank).render_track(project, track, sr, t0, t1, note_range)


# ---- the voicebank loader -------------------------------------------------------------------------
def test_voicebank_reads_configs_tables_and_models(fake):
    from synth2.neural.voicebank import Voicebank

    vb = Voicebank(fake.root)
    assert (vb.sample_rate, vb.hop, vb.mel_bins, vb.max_depth) == (SR, HOP, BINS, 0.4)
    assert vb.key_shift_range == (-5.0, 5.0) and vb.frame == FRAME
    assert vb.variance is not None and vb.pitch is not None and vb.vocoder_id == "fake_voc"
    assert vb.has("en/aa") and not vb.has("en/xx") and vb.is_vowel("en/aa") and not vb.is_vowel("en/t")
    assert vb.acoustic.tokens(["SP", "en/aa"]).tolist() == [[0, SYMBOLS.index("en/aa")]]
    assert vb.acoustic.language_ids(["SP", "en/aa", "ja/a"]).tolist() == [[0, 1, 2]]
    assert not fake.created  # nothing is loaded before it is needed
    io = vb.io("acoustic")
    assert io.inputs[0] == "tokens" and io.takes("depth") and not io.takes("spk_embed") and io.outputs == ["mel"]
    assert vb.session("acoustic") is vb.session("acoustic") and len(fake.created) == 1
    assert vb.session("acoustic").threads == 3
    fresh = vb.fresh_session("acoustic", 1234)
    assert fresh is not vb.session("acoustic") and fake.created[-1] == ("files/acoustic.onnx", 1234)
    assert vb.draws_noise("acoustic") and vb.draws_noise("variance") and vb.draws_noise("pitch")
    assert not vb.draws_noise("vocoder") and not vb.draws_noise("dur") and not vb.draws_noise("dur.linguistic")
    ident = vb.identity()
    assert ident[0] == "files" and any(f[0] == "acoustic.onnx" for f in ident[2])
    assert json.dumps(vb.summary())


def test_voicebank_errors_name_the_file(fake, tmp_path):
    from synth2.neural.voicebank import Voicebank, VoicebankError, feed, speedup_for

    with pytest.raises(VoicebankError, match="dsconfig.yaml"):
        Voicebank(tmp_path / "nowhere")
    root = make_bank(tmp_path / "b1")
    (root / "acoustic.onnx").unlink()
    with pytest.raises(VoicebankError, match=r"dsconfig\.yaml.*acoustic"):
        Voicebank(root)
    root = make_bank(tmp_path / "b2")
    (root / "dsvocoder" / "vocoder.yaml").write_text(f"model: voc.onnx\nsample_rate: 48000\nhop_size: {HOP}\n")
    with pytest.raises(VoicebankError, match=r"vocoder\.yaml: sample_rate is 48000"):
        Voicebank(root)
    root = make_bank(tmp_path / "b3", vocoder=False)
    with pytest.raises(VoicebankError, match="no vocoder"):
        Voicebank(root)
    root = make_bank(tmp_path / "b4")
    (root / "dsdur" / "phonemes.json").write_text("{not json")
    with pytest.raises(VoicebankError, match=r"dsdur.*phonemes\.json"):
        Voicebank(root)
    root = make_bank(tmp_path / "b5")
    (root / "dspitch" / "dsconfig.yaml").write_text("phonemes: phonemes.json\npitch: pitch.onnx\nlinguistic: "
                                                    "linguistic.onnx\npredict_dur: true\n")
    vb = Voicebank(root)
    assert vb.pitch is None and "predict_dur" in vb.pitch_note
    with pytest.raises(VoicebankError, match="inputs this engine does not provide: speedup2"):
        feed(types.SimpleNamespace(inputs=["tokens", "speedup2"]), {"tokens": 1})
    assert [speedup_for(s) for s in (5, 10, 20, 3, 1000, 4000)] == [200, 100, 50, 250, 1, 1]
    with pytest.raises(VoicebankError, match="no phoneme 'en/xx'"):
        vb.acoustic.tokens(["en/xx"])


def test_speakers_modes_and_blends(fake, tmp_path):
    from synth2.neural.voicebank import Voicebank, VoicebankError, match_speaker

    vb = Voicebank(make_bank(tmp_path / "multi", speakers=("std", "power", "soft")))
    assert vb.default_mode == "embeds/std" and set(vb.modes) == {"std", "power", "soft"}
    assert match_speaker(["dsmain/embeds/A", "dsmain/embeds/B"], "embeds/B") == "dsmain/embeds/B"
    assert match_speaker(["a", "b"], "c") is None
    assert vb.embed(vb.acoustic, 6).tolist() == [4.0] * 6  # the default mode
    assert vb.blend({}) is None and vb.blend({"POWER": 2.0}) == (("embeds/power", 1.0),)
    mix = vb.blend({"power": 3, "Soft": 1})
    assert mix == (("embeds/power", 0.75), ("embeds/soft", 0.25))
    assert np.allclose(vb.embed(vb.dur, 6, mix), 0.75 * 6.0 + 0.25 * 5.0)
    with pytest.raises(VoicebankError, match="no voice mode 'loud'"):
        vb.blend({"loud": 1})
    with pytest.raises(VoicebankError, match=r"std\.emb: 6 values, the acoustic model takes 8"):
        vb.embed(vb.acoustic, 8)
    with pytest.raises(VoicebankError, match="lists no speakers"):
        Voicebank(fake.root).embed(Voicebank(fake.root).acoustic, 6)


def test_dictionary_scan_and_random_operator_scan(tmp_path):
    from synth2.neural.voicebank import draws_random, scan_symbols

    f = tmp_path / "dsdict.yaml"
    f.write_text("symbols:\n- symbol: en/aa\n  type: vowel\n- {symbol: 'en/t', type: stop}\n- symbol: SP\n"
                 "  type: vowel\nentries:\n- grapheme: type\n  phonemes:\n  - en/t\n", encoding="utf-8-sig")
    assert scan_symbols(f) == {"en/aa": "vowel", "en/t": "stop", "SP": "vowel"}
    assert scan_symbols(tmp_path / "missing.yaml") == {}
    quiet, noisy = tmp_path / "a.onnx", tmp_path / "b.onnx"
    quiet.write_bytes(b"\x00" * (1 << 20) + b"Conv" * 10)
    noisy.write_bytes(b"\x01" * ((1 << 20) - 5) + b"RandomNormalLike" + b"\x00" * 100)  # across a chunk edge
    assert not draws_random(quiet) and draws_random(noisy) and not draws_random(tmp_path / "none.onnx")


def test_vocoder_resolution_order(fake):
    from synth2.neural import models
    from synth2.neural.voicebank import VoicebankError, load_installed, resolve_vocoder

    bank = models.find("voicebank", "fake")
    assert resolve_vocoder(bank)[0] is None  # the bank's own dsvocoder/ comes first
    assert load_installed(bank).vocoder_id == "fake_voc"
    remove_bundled_vocoder(fake.root)
    with pytest.raises(VoicebankError, match="no vocoder is installed"):
        resolve_vocoder(bank)
    install_vocoder(fake.models, "voc-default", "other", default=True)
    install_vocoder(fake.models, "voc-named", "fake_voc")
    assert resolve_vocoder(bank)[1] == "voc-named"  # the dsconfig's vocoder name beats the default
    manifest = json.loads((fake.models / "fake" / "synth2-model.json").read_text())
    (fake.models / "fake" / "synth2-model.json").write_text(json.dumps({**manifest, "vocoder": "voc-default"}))
    vb = load_installed(models.find("voicebank", "fake"))
    assert vb.vocoder_id == "voc-default" and vb.credits == ["Fake singer (test)", "Vocoder voc-default"]


def test_credits_and_licence_are_those_of_the_vocoder_in_use(fake):
    from synth2 import neural

    install_vocoder(fake.models, "voc", "fake_voc", default=True, commercial_use=False)
    path = fake.models / "fake" / "synth2-model.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), "commercial_use": True}))
    neural.refresh()
    p, tr = song()
    # the bank sings through its own dsvocoder/: the installed vocoder is neither credited nor a licence limit
    assert neural.credits() == neural.credits("fake") == ["Fake singer (test)"] == sing(p, tr)[1]["credits"]
    st = neural.status()
    assert st["credits"] == ["Fake singer (test)"] and st["commercial_use"] is True and st["available"]
    remove_bundled_vocoder(fake.root)
    neural.refresh()
    assert neural.credits() == ["Fake singer (test)", "Vocoder voc"] == sing(p, tr)[1]["credits"]
    st = neural.status()
    assert st["credits"] == ["Fake singer (test)", "Vocoder voc"] and st["commercial_use"] is False
    assert neural.credits("missing") == []


# ---- the backend ----------------------------------------------------------------------------------
def test_render_length_alignment_and_report(fake):
    p, tr = song()
    audio, rep = sing(p, tr)
    t1 = p.tempo.beat_to_sec(tr.end) + 0.6
    assert audio.dtype == np.float32 and len(audio) == int(round(t1 * SR)) and np.abs(audio).max() > 0.05
    assert rep["engine"] == "neural" and rep["bank"] == "fake" and rep["credits"] == ["Fake singer (test)"]
    assert rep["notes"] == len(tr.notes) and set(rep["phonemes"]) == {n.id for n in tr.notes}
    info = rep["neural"]
    assert info["phrases"] == 3 and info["pitch"] == "bank" and info["seeded"] and info["steps"] == 10
    # silence between the phrases, sound inside the first note
    plan = plan_of(tr, p)
    first = plan.timings[0]
    mid = int((first.start + first.end) / 2 * SR)
    assert np.abs(audio[mid - 500:mid + 500]).max() > 0.05
    gap = int((plan.timings[1].end + 0.5) * SR)
    assert np.abs(audio[gap - 200:gap + 200]).max() == 0.0
    names = [c[0] for c in fake.calls]
    assert names[:8] == ["dsdur/linguistic.onnx", "dsdur/dur.onnx", "dspitch/linguistic.onnx", "dspitch/pitch.onnx",
                         "dsvariance/linguistic.onnx", "dsvariance/variance.onnx", "files/acoustic.onnx",
                         "dsvocoder/voc.onnx"]
    depth = [c[1]["depth"] for c in fake.calls if c[0] == "files/acoustic.onnx"][0]
    assert float(depth) == pytest.approx(0.4)  # clipped to the bank's max_depth


def test_partial_render_equals_the_full_render(fake):
    p, tr = song()
    full, _ = sing(p, tr)
    for t0, t1 in ((1.3, 3.1), (0.0, 0.9), (2.0, 2.05), (3.9, 6.0)):
        part, rep = sing(p, tr, t0=t0, t1=t1, note_range=(p.tempo.sec_to_beat(t0), p.tempo.sec_to_beat(t1)))
        a = int(round(t0 * SR))
        want = full[a:a + len(part)]
        assert len(part) == int(round((t1 - t0) * SR))
        assert np.array_equal(part[:len(want)], want) and not part[len(want):].any()
    part, rep = sing(p, tr, t0=2.0, t1=2.9, note_range=(4.0, 5.8))
    assert rep["neural"]["phrases"] == 1 and rep["notes"] < len(tr.notes)


@pytest.mark.parametrize("sr", [22050, 48000])
def test_other_sample_rates(fake, sr):
    p, tr = song()
    t1 = p.tempo.beat_to_sec(tr.end) + 0.6
    full, _ = sing(p, tr, sr=sr)
    assert len(full) == int(round(t1 * sr)) and full.dtype == np.float32
    ref, _ = sing(p, tr)
    assert np.sqrt(np.mean(full ** 2)) == pytest.approx(np.sqrt(np.mean(ref ** 2)), rel=0.02)
    part, _ = sing(p, tr, sr=sr, t0=1.3, t1=3.1)
    a = int(round(1.3 * sr))
    assert len(part) == int(round(1.8 * sr))
    assert np.abs(part - full[a:a + len(part)]).max() < 1e-5


def test_reproducible_mode_seeds_a_fresh_session_per_phrase(fake, monkeypatch):
    p, tr = song()
    a, _ = sing(p, tr)
    made = list(fake.created)
    b, _ = sing(p, tr)
    assert np.array_equal(a, b)
    noisy = [m for m in made if m[0] in ("files/acoustic.onnx", "dsvariance/variance.onnx", "dspitch/pitch.onnx")]
    assert len(noisy) == 9 and len({s for _, s in noisy}) == 3  # three phrases, one seed each
    assert all(s is not None for _, s in noisy)
    assert sum(1 for m in made if m[0] == "dsvocoder/voc.onnx") == 1  # a deterministic vocoder is shared
    tr.seed += 1
    c, _ = sing(p, tr)
    assert not np.array_equal(a, c)
    tr.seed -= 1
    monkeypatch.setenv("SYNTH2_NEURAL_REPRODUCIBLE", "0")
    n = len(fake.created)
    d, rep = sing(p, tr)
    e, _ = sing(p, tr)
    assert not rep["neural"]["seeded"] and not np.array_equal(d, e)  # shared sessions: every run is another take
    assert len(fake.created) - n == 3  # one shared session per diffusion model


def test_a_vocoder_that_draws_noise_gets_fresh_sessions(fake):

    _model(fake.root / "dsvocoder" / "voc.onnx", ["mel", "f0"], ["waveform"], tail="RandomNormalLike")
    nb.reset()
    p, tr = song()
    sing(p, tr)
    assert sum(1 for m in fake.created if m[0] == "dsvocoder/voc.onnx") == 3


def test_quality_steps_and_environment(fake, monkeypatch):
    from synth2.neural import engine

    steps = lambda: [int(c[1]["steps"]) for c in fake.calls if c[0] == "files/acoustic.onnx"][-1]
    p, tr = song()
    for q, n in (("draft", 5), ("normal", 10), ("high", 20)):
        with render_options(quality=q):
            sing(p, tr, t0=0.0, t1=1.0)
        assert steps() == n and engine.QUALITY_STEPS[q] == n
    monkeypatch.setenv("SYNTH2_NEURAL_STEPS", "7")
    sing(p, tr, t0=0.0, t1=1.0)
    assert steps() == 7
    monkeypatch.setenv("SYNTH2_NEURAL_PITCH", "expression")
    fake.calls.clear()
    _, rep = sing(p, tr, t0=0.0, t1=1.0)
    assert rep["neural"]["pitch"] == "expression" and not any("dspitch" in c[0] for c in fake.calls)
    monkeypatch.delenv("SYNTH2_NEURAL_PITCH")
    tr.voice_params["neural.expr"] = 0.0
    assert sing(p, tr, t0=0.0, t1=1.0)[1]["neural"]["pitch"] == "expression"
    tr.voice_params["neural.expr"] = 0.7
    fake.calls.clear()
    sing(p, tr, t0=0.0, t1=1.0)
    expr = [c[1]["expr"] for c in fake.calls if c[0] == "dspitch/pitch.onnx"][0]
    assert np.allclose(expr, 0.7)


def test_parameters_reach_the_models(fake):
    p, tr = song()
    tr.voice = "atlas"
    tr.voice_params.update({"neural.breath_db": 3.0, "neural.voicing_db": -2.0, "neural.velocity": 1.2,
                            "neural.tone": 2.0, "neural.breaths": 0, "neural.vowel_breath_db": 0})
    c = tr.curve("gender", create=True)
    c.add(0.0, 0.5)
    c.add(100.0, 0.5)
    fake.calls.clear()
    sing(p, tr, t0=0.0, t1=1.0)
    ac = [c[1] for c in fake.calls if c[0] == "files/acoustic.onnx"][0]
    var = [c[1] for c in fake.calls if c[0] == "dsvariance/variance.onnx"][0]
    assert np.allclose(ac["gender"], -0.3 - 0.5)  # Synth2's gender curve is negative = brighter
    assert np.allclose(ac["breathiness"], -6.0 + 3.0, atol=0.02) and np.allclose(ac["voicing"], -6.0 - 2.0, atol=0.02)
    assert np.allclose(ac["velocity"], 1.2)
    voc = [c[1] for c in fake.calls if c[0] == "dsvocoder/voc.onnx"][0]
    semis = 12 * np.log2(ac["f0"][0, 30] / voc["f0"][0, 30])
    assert semis == pytest.approx(2.0, abs=1e-3)  # the timbre is shifted, the sung pitch is not
    assert np.allclose(12 * np.log2(ac["f0"] / 440.0) + 69, var["pitch"], atol=1e-3)
    assert ac["tokens"].dtype == np.int64 and ac["f0"].dtype == np.float32 and var["retake"].shape[2] == 2


def test_slow_motion_stretches_fast_notes_for_the_models_only(fake):
    p, tr = song("tempo: 150\ntrack: Lead voice=aurora lang=en\n" + "C4 s la  D4 s la  E4 s la  F4 s la  " * 2 + "\n")
    fake.calls.clear()
    sing(p, tr)
    ac = [c[1] for c in fake.calls if c[0] == "files/acoustic.onnx"][0]
    voc = [c[1] for c in fake.calls if c[0] == "dsvocoder/voc.onnx"][0]
    assert ac["f0"].shape[1] > 1.5 * voc["f0"].shape[1] - 30
    assert voc["mel"].shape[1] == voc["f0"].shape[1] and int(ac["durations"].sum()) == ac["f0"].shape[1]
    tr.voice_params["neural.slow"] = 0.0
    fake.calls.clear()
    sing(p, tr)
    ac = [c[1] for c in fake.calls if c[0] == "files/acoustic.onnx"][0]
    assert ac["f0"].shape[1] == voc["f0"].shape[1]


def _feeds(fake, model):
    return [c[1] for c in fake.calls if c[0] == model]


def test_the_dial_sets_the_consonants_the_models_get(fake):
    text = "tempo: 100\ntrack: Lead voice=aurora lang=en\n" + "C4 e test  D4 e stick  E4 e catch  F4 e sits  " * 2 + "\n"
    durations, frames = {}, {}
    for dial in (0, 1, 2):
        p, tr = song(text)
        tr.voice_params["neural.dial"] = dial
        fake.calls.clear()
        _, rep = sing(p, tr)
        assert rep["neural"]["dial"] == ("smooth", "balanced", "crisp")[dial]
        assert rep["neural"]["preset"]["cons_cap_share"] == (None, 0.35, 0.4)[dial]
        ac, voc = _feeds(fake, "files/acoustic.onnx")[0], _feeds(fake, "dsvocoder/voc.onnx")[0]
        durations[dial] = [d * FRAME for ph in rep["neural"]["alignment"] for d in ph["frames"]]
        frames[dial] = (ac["f0"].shape[1], voc["f0"].shape[1])
        assert voc["mel"].shape[1] == voc["f0"].shape[1]
    assert durations[0] != durations[1] != durations[2] and len(set(map(len, durations.values()))) == 1
    assert frames[0][1] == frames[1][1] == frames[2][1]  # the same real frames
    # crisp sings its stops in slow motion (90 ms each) and cuts them back to their real length
    assert frames[0][0] == frames[0][1] and frames[1][0] == frames[1][1] and frames[2][0] > frames[2][1] + 20
    p, tr = song(text)
    tr.voice_params["neural.burst"] = 0
    fake.calls.clear()
    sing(p, tr)
    assert _feeds(fake, "files/acoustic.onnx")[0]["f0"].shape[1] == frames[2][1]


def test_breaths_between_lines_by_default(fake):
    ap = SYMBOLS.index("AP")
    p, tr = song(LINES)
    fake.calls.clear()
    audio, rep = sing(p, tr)
    lines = rep["neural"]["alignment"]
    assert [ph["phones"].count("AP") for ph in lines] == [2, 1]  # before each phrase and in the short gap
    ac = _feeds(fake, "files/acoustic.onnx")[0]
    on = np.repeat(ac["tokens"][0] == ap, ac["durations"][0])
    assert on.sum() > 30 and ac["breathiness"][0][on].max() == pytest.approx(-44.0, abs=1.0)
    assert np.allclose(ac["voicing"][0][on], -96.0) and np.allclose(ac["voicing"][0][~on], -6.0, atol=0.02)
    # the first breath starts 0.3 s before the first consonant: the render reaches back for it
    first = lines[0]
    start = (first["frame"] + first["frames"][0]) * FRAME
    assert start == pytest.approx(plan_of(tr, p).units[0].start - plan_of(tr, p).units[0].lead - 0.30, abs=2 * FRAME)
    tr.voice_params["neural.breaths"] = 0.5
    fake.calls.clear()
    sing(p, tr)
    ac = _feeds(fake, "files/acoustic.onnx")[0]
    assert ac["breathiness"][0][np.repeat(ac["tokens"][0] == ap, ac["durations"][0])].max() == pytest.approx(-50.0, abs=1.0)
    tr.voice_params["neural.breaths"] = 0
    fake.calls.clear()
    _, rep = sing(p, tr)
    assert not any("AP" in ph["phones"] for ph in rep["neural"]["alignment"])
    assert not (_feeds(fake, "files/acoustic.onnx")[0]["tokens"] == ap).any()
    assert rep["neural"]["preset"]["breaths"] == 0.0


def test_partial_render_equals_the_full_render_with_breaths_and_bursts(fake):
    p, tr = song(LINES)
    full, rep = sing(p, tr)
    assert sum(ph["phones"].count("AP") for ph in rep["neural"]["alignment"]) == 3
    for t0, t1 in ((0.0, 0.4), (0.2, 1.7), (1.5, 2.6), (3.2, 4.5)):
        part, _ = sing(p, tr, t0=t0, t1=t1)
        a = int(round(t0 * SR))
        want = full[a:a + len(part)]
        assert np.array_equal(part[:len(want)], want) and not part[len(want):].any()


def test_vowel_breathiness_and_detune(fake, monkeypatch):
    p, tr = song()
    tr.voice_params["neural.breaths"] = 0
    fake.calls.clear()
    _, rep = sing(p, tr, t0=0.0, t1=1.0)
    ac, voc = _feeds(fake, "files/acoustic.onnx")[0], _feeds(fake, "dsvocoder/voc.onnx")[0]
    lay = rep["neural"]["alignment"][0]
    vowel = np.repeat(lay["vowels"], lay["frames"])
    br = ac["breathiness"][0]
    assert sorted(set(np.round(br[vowel], 2))) == [-9.0, -7.5] and np.allclose(br[~vowel], -6.0, atol=0.02)
    assert rep["neural"]["detune"] == 7.0  # the bank's predictor sings: its flat tuning is made up for
    flat = voc["f0"][0].copy()
    tr.voice_params.update({"neural.detune": 0, "neural.vowel_breath_db": 0})
    fake.calls.clear()
    _, rep = sing(p, tr, t0=0.0, t1=1.0)
    ac, voc = _feeds(fake, "files/acoustic.onnx")[0], _feeds(fake, "dsvocoder/voc.onnx")[0]
    assert np.allclose(ac["breathiness"][0], -6.0, atol=0.02) and rep["neural"]["detune"] == 0.0
    assert np.allclose(1200 * np.log2(flat / voc["f0"][0]), 7.0, atol=0.01)
    del tr.voice_params["neural.detune"]
    monkeypatch.setenv("SYNTH2_NEURAL_PITCH", "expression")
    assert sing(p, tr, t0=0.0, t1=1.0)[1]["neural"]["detune"] == 0.0  # Synth2's own curve is in tune


# ---- ad-lib mode ----------------------------------------------------------------------------------
SHOUTS = """tempo: 120
track: Shouts voice=aurora style=flat lang=en voice.neural.adlib=1
C4 e hey  r e  C4 e hey  r e  E4 e click  E4 e click  r q | G4 q yeah  r q  C4 e hey  r e  r q
"""


def test_adlib_options_and_hits():
    ad = adlib.settings(preset_for("nova", {"neural.adlib": 1, "neural.adlib.split": 1, "neural.adlib.bend": -2,
                                            "neural.adlib.bend.ha": 3, "neural.adlib.salt.click": 2,
                                            "neural.adlib.accent.1": -2.5, "neural.adlib.accent.0": 0,
                                            "neural.adlib.onset_cap.stop": 0.03, "neural.adlib.onset_floor.glide": 0.046,
                                            "neural.adlib.cons_cap.vowel": 1, "neural.adlib.nonsense": 4}).adlib)
    assert ad["split"] == "syllable" and ad["bend"] == -2.0 and ad["bends"] == {"ha": 3.0} and ad["salts"] == {"click": 2.0}
    assert ad["accent"] == [0.0, -2.5] and ad["onset_cap"]["stop"] == 0.03 and ad["onset_cap"]["fric"] == 0.07
    assert ad["onset_floor"] == {"glide": 0.046} and ad["cons_cap"] == {} and "nonsense" not in ad
    assert ad["stamp"] == 1.0 and ad["tight"] == 1.0 and ad["hold"] == 0.07
    plain = adlib.settings(())
    assert plain["split"] == "word" and plain["accent"] is None and plain["bend"] == 0.0
    assert preset_for("nova").adlib is None and preset_for("nova", {"neural.adlib.bend": 1}).adlib is None
    assert preset_for("nova", {"neural.adlib": 0, "neural.adlib.bend": 1}).adlib is None
    on = preset_for("nova", {"neural.adlib": 1})
    assert on.adlib == () and on.vowel_breath_db == 0.0 and on.cons_floor == ()  # a hit keeps the bank's own curves
    assert preset_for("nova", {"neural.adlib": 1, "neural.cons_floor.glide": 8}).cons_floor == (("glide", 8.0),)
    p, tr = song("tempo: 120\ntrack: A voice=aurora style=flat lang=en\nC4 e hello  D4 e +  E4 e there  r e  G4 q you\n")
    plan = plan_of(tr, p)
    count = lambda split: [len(h) for h in adlib.hits_of(plan, tr.notes, split)]
    assert count("word") == [2, 1, 1] and count("syllable") == [1, 1, 1, 1] and count("run") == [3, 1]


def test_adlib_track_is_sung_hit_by_hit_with_stamped_takes(fake):
    p, tr = song(SHOUTS)
    fake.calls.clear()
    audio, rep = sing(p, tr)
    info = rep["neural"]
    assert info["pitch"] == "adlib" and info["detune"] == 0.0 and info["alignment"] == []
    hits = info["hits"]
    assert [h["lyric"] for h in hits] == ["hey", "hey", "click", "click", "yeah", "hey"]
    assert len({h["take"] for h in hits}) == 3 and info["phrases"] == 3  # one take per distinct sound
    assert [h["touches"] for h in hits] == [False, False, True, False, False, False]
    assert hits[2]["cut_s"] > 0 and hits[3]["cut_s"] == hits[2]["cut_s"] and hits[0]["cut_s"] == 0.0
    names = [c[0] for c in fake.calls]
    assert names.count("files/acoustic.onnx") == 3 and "dspitch/pitch.onnx" not in names
    ap = SYMBOLS.index("AP")
    for ac in _feeds(fake, "files/acoustic.onnx"):
        assert not (ac["tokens"] == ap).any()  # no breath
        semis = 12 * np.log2(ac["f0"][0] / 440.0) + 69
        assert np.ptp(semis) < 1e-3 and round(float(semis[0])) in (60, 64, 67)  # the written pitch, held flat
    # every repetition is a copy of the same samples
    first, second, last = (int(round(hits[i]["t"] * SR)) for i in (0, 1, 5))
    assert np.abs(audio[first:first + 6000]).max() > 0.05
    assert np.array_equal(audio[first:first + 8000], audio[second:second + 8000])
    assert np.array_equal(audio[second - 2000:second + 8000], audio[last - 2000:last + 8000])
    assert not audio[int((hits[0]["end"] + 0.08) * SR):int((hits[0]["end"] + 0.14) * SR)].any()  # silence between the hits
    # a part of the track equals the same span of the whole
    for t0, t1 in ((0.3, 1.2), (1.9, 2.4), (2.2, 4.4)):
        part, _ = sing(p, tr, t0=t0, t1=t1)
        a = int(round(t0 * SR))
        want = audio[a:a + len(part)]
        assert np.array_equal(part[:len(want)], want) and not part[len(want):].any()
    # without the stamp every hit is a take of its own
    tr.voice_params["neural.adlib.stamp"] = 0
    fake.calls.clear()
    fresh, rep = sing(p, tr)
    assert [c[0] for c in fake.calls].count("files/acoustic.onnx") == 6 and rep["neural"]["phrases"] == 6
    assert not np.array_equal(fresh[first:first + 8000], fresh[second:second + 8000])


def test_adlib_pitch_shapes_takes_and_cache(fake, monkeypatch):
    p, tr = song(SHOUTS)
    ref, rep = sing(p, tr)
    hits = rep["neural"]["hits"]
    hey, click, yeah = (int(round(hits[i]["t"] * SR)) for i in (0, 2, 4))
    # another take of one sound
    tr.voice_params["neural.adlib.salt.hey"] = 1
    salted, rep2 = sing(p, tr)
    assert not np.array_equal(salted[hey:hey + 8000], ref[hey:hey + 8000])
    assert np.array_equal(salted[click:click + 8000], ref[click:click + 8000])
    assert rep2["neural"]["hits"][0]["take"] == hits[0]["take"]  # the same sound, another seed
    del tr.voice_params["neural.adlib.salt.hey"]
    # a fall on one lyric
    tr.voice_params["neural.adlib.bend.yeah"] = -3
    fake.calls.clear()
    _, rep3 = sing(p, tr)
    shapes = {h["lyric"]: h["shape"] for h in rep3["neural"]["hits"]}
    assert shapes == {"hey": "flat", "click": "flat", "yeah": "fall"}
    falls = [12 * np.log2(ac["f0"][0] / 440.0) + 69 for ac in _feeds(fake, "files/acoustic.onnx")]
    yeah_f0 = [f for f in falls if round(float(f[0])) == 67][0]
    assert yeah_f0[0] == pytest.approx(67.0, abs=1e-3) and yeah_f0.min() < 67.0 - 3.0 and np.all(np.diff(yeah_f0) <= 1e-9)
    # takes go through the phrase cache
    store = {}
    stand_in = types.SimpleNamespace(
        enabled=lambda: True, default_cache=lambda: types.SimpleNamespace(get=store.get, put=store.__setitem__))
    monkeypatch.setattr(nb, "phrase_cache", stand_in)
    first, rep = sing(p, tr)
    assert len(store) == 3 and rep["neural"]["cached"] == 0
    fake.calls.clear()
    again, rep = sing(p, tr)
    assert np.array_equal(first, again) and rep["neural"]["cached"] == 3
    assert {c[0] for c in fake.calls} == {"dsdur/linguistic.onnx", "dsdur/dur.onnx"}
    tr.voice_params["neural.adlib.chop_hz"], tr.voice_params["neural.adlib.chop_db"] = 11, -9
    chopped, rep = sing(p, tr)  # shaped after the cache
    assert rep["neural"]["cached"] == 3 and not np.array_equal(chopped, first)
    tr.voice_params["neural.gender"] = 0.3
    assert sing(p, tr)[1]["neural"]["cached"] == 0


def test_a_line_takes_no_breath_where_an_adlib_of_its_voice_sounds(fake):
    text = LINES + "track: Shouts voice={v} style=flat lang=en voice.neural.adlib=1\nr w r e C4 e hey\n"
    p = parse_score(text.format(v="aurora"))
    lead, shouts = p.tracks
    spans = adlib.busy_spans(p, lead)
    assert len(spans) == 1 and spans[0][1] == pytest.approx(p.tempo.beat_to_sec(shouts.notes[0].end))
    assert adlib.busy_spans(p, shouts) == []
    _, rep = sing(p, lead)
    assert [ph["phones"].count("AP") for ph in rep["neural"]["alignment"]] == [2, 0]  # the second line starts dry
    other = parse_score(text.format(v="nova"))
    _, rep = sing(other, other.tracks[0])
    assert [ph["phones"].count("AP") for ph in rep["neural"]["alignment"]] == [2, 1]


def test_loudness_curve_and_level_are_gain(fake):
    p, tr = song()
    ref, _ = sing(p, tr)
    tr.voice_params["neural.level_db"] = -6.0
    quiet, _ = sing(p, tr)
    assert np.allclose(quiet, ref * 10 ** ((-6 - PRESETS["aurora"].level_db) / 20), atol=1e-6)
    del tr.voice_params["neural.level_db"]
    tr.notes[0].attrs["velocity"] = 0.5
    soft, _ = sing(p, tr)
    plan = plan_of(tr, p)
    mid = int((plan.timings[0].start + 0.1) * SR)
    assert np.abs(soft[mid:mid + 2000]).max() == pytest.approx(0.5 * np.abs(ref[mid:mid + 2000]).max(), rel=0.05)
    assert np.array_equal(soft[-SR:], ref[-SR:])
    part, _ = sing(p, tr, t0=0.4, t1=1.2)
    a = int(round(0.4 * SR))
    assert np.array_equal(part, soft[a:a + len(part)])


def test_odd_input_gives_silence_and_a_warning_not_an_error(fake):
    p, tr = song()
    n = int(round(2.0 * SR))
    tr.notes = []
    audio, rep = sing(p, tr, t1=2.0)
    assert len(audio) == n and not audio.any() and any("nothing to sing" in w for w in rep["warnings"])
    for text in ("r w | r w\n", "C4 0.01 la\n", "C4 w la\n"):
        p, tr = song("tempo: 120\ntrack: Lead voice=aurora lang=en\n" + text)
        audio, rep = sing(p, tr, t1=2.0)
        assert len(audio) == n and audio.dtype == np.float32
    p, tr = song()
    tr.notes[1].onset = tr.notes[0].onset + 0.25  # overlapping notes
    audio, rep = sing(p, tr, t1=2.0)
    assert len(audio) == n and np.isfinite(audio).all()
    assert len(sing(p, tr, t0=1.0, t1=1.0)[0]) == 0
    audio, _ = sing(p, tr, t0=50.0, t1=51.0)
    assert len(audio) == SR and not audio.any()


def test_a_phrase_that_fails_is_skipped_with_a_warning(fake, monkeypatch):

    real = nb.synth_phrase
    count = {"n": 0}

    def flaky(run, ph, opts):
        count["n"] += 1
        if count["n"] == 2:
            raise ValueError("model exploded")
        return real(run, ph, opts)

    monkeypatch.setattr(nb, "synth_phrase", flaky)
    p, tr = song()
    audio, rep = sing(p, tr)
    assert rep["neural"]["failed"] == 1 and rep["neural"]["phrases"] == 2
    assert any("could not be sung" in w and "model exploded" in w for w in rep["warnings"])


def test_report_warns_about_range_phoneme_fallbacks_and_unknown_modes(fake):
    p, tr = song("tempo: 120\ntrack: Lead voice=aurora lang=en\nC6 q vision  C6 q la\n")
    table = {s: i for i, s in enumerate(x for x in SYMBOLS if x != "en/zh")}
    for sub in ("", "dsdur", "dsvariance", "dspitch"):
        (fake.root / sub / "phonemes.json").write_text(json.dumps(table))
    audio, rep = sing(p, tr)
    assert any("above the voicebank's comfortable range" in w for w in rep["warnings"])
    assert any("en/zh as en/jh" in w for w in rep["warnings"])
    tr.voice_params["neural.mode.growl"] = 1.0
    audio2, rep = sing(p, tr)
    assert any("no voice mode growl" in w for w in rep["warnings"]) and np.array_equal(audio, audio2)


def test_voice_modes_feed_the_speaker_embedding(fake):

    install(fake.models, "multi", speakers=("std", "power"))
    nb.reset()
    p, tr = song()
    fake.calls.clear()
    sing(p, tr, t1=1.0, bank="multi")
    emb = [c[1]["spk_embed"] for c in fake.calls if c[0] == "files/acoustic.onnx"][0]
    assert emb.shape[0] == 1 and emb.shape[2] == 6 and np.allclose(emb, 4.0) and emb.dtype == np.float32
    dur = [c[1]["spk_embed"] for c in fake.calls if c[0] == "dsdur/dur.onnx"][0]
    assert dur.shape[1] == [c[1]["ph_midi"] for c in fake.calls if c[0] == "dsdur/dur.onnx"][0].shape[1]
    tr.voice_params.update({"neural.mode.power": 1.0, "neural.mode.std": 1.0})
    fake.calls.clear()
    sing(p, tr, t1=1.0, bank="multi")
    emb = [c[1]["spk_embed"] for c in fake.calls if c[0] == "files/acoustic.onnx"][0]
    assert np.allclose(emb, 0.5 * 4.0 + 0.5 * 6.0)


def test_backend_registry_and_engine_selection(fake):
    from synth2 import neural
    from synth2.engine.backend import EngineUnavailable, backend_for_track

    p, tr = song()
    assert neural.available() == (True, "")
    b = backend_for_track(p, tr, "neural")
    assert b.name == "neural" and b is backend_for_track(p, tr, "neural") and set(b.voices()) == set(VOICES)
    assert neural.backend("fake") is nb.get_backend("fake") and neural.backend() is nb.get_backend()
    # the package's backend() is a function and no module of the package takes its name
    assert isinstance(neural.backend, types.FunctionType) and neural.backend() is b is not neural.backend("fake")
    assert importlib.util.find_spec("synth2.neural.backend") is None
    install(fake.models, "b2")
    install(fake.models, "b3")
    for bank in ("fake", "b2", "b3"):
        assert nb.load_bank(bank).id == bank
    assert len(nb._banks) == nb.MAX_BANKS
    with pytest.raises(EngineUnavailable):
        nb.load_bank("missing")
    tr.voice_params["neural.gender"] = 0.1
    assert nb.get_backend().bank_for(VoicePreset(bank="b2"))[0].id == "b2"
    bank, notes = nb.get_backend().bank_for(VoicePreset(bank="gone"))
    assert bank.id == "fake" and "not installed" in notes[0]


def test_render_project_through_the_neural_engine(fake):
    from synth2.engine.renderer import render_project

    p, tr = song()
    res = render_project(p, engine="neural")
    assert res.report["engines"] == ["neural"] and res.report["credits"] == ["Fake singer (test)"]
    assert res.report["tracks"][0]["bank"] == "fake"


# ---- roles ----------------------------------------------------------------------------------------
DUET = """tempo: 120
track: Lead voice=aurora style=flat lang=en role=lead
C4 q sing  D4 e it  r e  E4 q so  r h | C4 q again  r q  E4 h more
track: Double voice=aurora style=flat lang=en role=double pan=-0.3
r w | C4 q again  r q  r h
"""


def test_a_role_sets_how_a_track_is_sung(fake):
    for role, breaths in (("", 3), ("lead", 3), ("crowd", 3), ("harmony", 0), ("answer", 0), ("double", 0)):
        p, tr = song(LINES)
        tr.role = role
        _, rep = sing(p, tr)
        assert sum(ph["phones"].count("AP") for ph in rep["neural"]["alignment"]) == breaths and rep["role"] == role
    p, tr = song(LINES)
    tr.role = "harmony"
    tr.voice_params["neural.breaths"] = 1  # the track's own parameter wins
    assert sum(ph["phones"].count("AP") for ph in sing(p, tr)[1]["neural"]["alignment"]) == 3
    p, tr = song(SHOUTS.replace(" voice.neural.adlib=1", " role=ad-lib"))
    _, rep = sing(p, tr)
    assert rep["role"] == "adlib" and rep["neural"]["pitch"] == "adlib" and len(rep["neural"]["hits"]) == 6
    tr.voice_params["neural.adlib"] = 0
    assert sing(p, tr)[1]["neural"]["pitch"] == "bank"
    p, tr = song(LINES)
    tr.role = "juggler"
    _, rep = sing(p, tr)
    assert rep["role"] == "" and any("unknown role 'juggler'" in w for w in rep["warnings"])


def test_a_double_is_the_leads_take_over_its_own_notes(fake):
    from synth2.engine import vocal_chain

    p = parse_score(DUET)
    lead, double = p.tracks
    t1 = p.tempo.beat_to_sec(lead.end) + 0.6
    take, _ = sing(p, lead, t1=t1)
    fake.calls.clear()
    copy, rep = sing(p, double, t1=t1)
    info = rep["neural"]
    assert info["pitch"] == "copy" and info["copy"] == ["Lead"] and info["copy_of"] == "lead" and rep["role"] == "double"
    # the lead was sung (again) for it, the double's own notes were not
    assert [c[0] for c in fake.calls].count("files/acoustic.onnx") == 3
    a, b = (int(p.tempo.beat_to_sec(x) * SR) for x in (4.0, 5.0))  # the double's one note: "again"
    assert np.array_equal(copy[a:b], take[a:b]) and np.abs(copy[a:b]).max() > 0.05
    assert not copy[: a - int(0.1 * SR)].any() and not copy[b + int(0.16 * SR):].any()
    assert 0 < np.abs(copy[a - int(0.07 * SR)]) < np.abs(take[a - int(0.07 * SR)]) or not take[a - int(0.07 * SR)]
    # the mixer is told: the double gets the copy's delay and detune, the lead its chain
    assert vocal_chain.noted_role(double.id, copy) == {**vocal_chain.noted_role(double.id, copy), "role": "double",
                                                       "copy_of": "lead"}
    assert vocal_chain.noted_role(lead.id, take)["role"] == "lead"
    # a part of the copy equals the same span of the whole
    part, _ = sing(p, double, t0=1.7, t1=2.6)
    assert np.array_equal(part, copy[int(round(1.7 * SR)):int(round(1.7 * SR)) + len(part)])
    # the double's loudness curve rides the copy
    double.curve("loudness").add(0.0, -6.0)
    double.curve("loudness").add(100.0, -6.0)
    quiet, _ = sing(p, double, t1=t1)
    assert np.allclose(quiet[a:b], copy[a:b] * 10 ** (-6 / 20), atol=1e-6)
    # a named track is doubled whatever its role; a double with nothing to double sings its own notes
    lead.role = ""
    alone, rep = sing(p, double, t1=t1)
    assert rep["neural"]["pitch"] == "bank" and rep["neural"]["phrases"] == 1
    assert vocal_chain.noted_role(double.id, alone)["copy_of"] is None
    double.role = "double:Lead"
    assert sing(p, double, t1=t1)[1]["neural"]["copy"] == ["Lead"]
    double.role = "double:Nobody"
    assert sing(p, double, t1=t1)[1]["neural"]["pitch"] == "bank"
    lead.role, double.role = "double:Double", "double:Lead"  # two tracks doubling each other: one of them sings
    _, rep = sing(p, double, t1=t1)
    assert rep["neural"]["pitch"] == "copy"


def test_render_project_gives_the_tracks_with_a_role_their_chain(fake):
    from synth2.engine.renderer import render_project

    # a lead, its double, and a crowd that only sings in the last bar
    p = parse_score(DUET + "track: Crowd voice=atlas style=flat lang=en role=crowd\nr w | r h  C3 h more\n")
    res = render_project(p, engine="neural")
    strips = res.report["mix"]["vocal_strips"]
    assert strips["Lead"]["role"] == "lead" and strips["Lead"]["dial"] == "crisp" and "lift" in strips["Lead"]["chain"]
    assert strips["Double"]["role"] == "double" and strips["Double"]["adt"]["delay"] == 0.018
    assert strips["Crowd"]["role"] == "crowd" and strips["Crowd"]["duck_db"] == 3.0
    assert [t["role"] for t in res.report["tracks"]] == ["lead", "double", "crowd"]
    # the same project sung by the formant engine, or without roles, is mixed as before
    formant = render_project(p, engine="formant")
    assert all("role" not in s for s in formant.report["mix"].get("vocal_strips", {}).values())
    for tr in p.tracks:
        tr.role = ""
    plain = render_project(p, engine="neural")
    assert all("role" not in s for s in plain.report["mix"].get("vocal_strips", {}).values())
    assert not np.array_equal(plain.audio, res.audio)


def test_window_span_and_place():
    from synth2.neural import resample

    assert nb.window_span(0.5, 1.5, SR, SR) == (22050, 44100, 22050, 66150)
    assert nb.window_span(0.5, 1.5, SR, SR, pre_roll=0.1) == (22050, 44100, 22050 - 4410, 66150)
    o, n, lo, hi = nb.window_span(1.0, 2.0, 48000, SR)
    m = resample.margin(SR, 48000)
    assert (o, n) == (48000, 48000) and lo <= SR - m and hi >= 2 * SR + m
    buf = np.zeros(3000, np.float32)
    nb.place(buf, 5000, np.ones(2000, np.float32), 5200, SR)
    fade = int(nb.FADE * SR)
    assert not buf[:201].any() and 0 < buf[210] < 1 and np.all(buf[200 + fade:2200 - fade] == 1.0)
    assert 0 < buf[2190] < 1 and not buf[2199:].any()
    nb.place(buf, 5000, np.ones(6000, np.float32), 3500, SR)  # reaches past both ends of the buffer
    assert buf[0] == 1.0 and buf[2999] == 1.0 and buf[1000] == 2.0
    nb.place(buf, 5000, np.ones(2000, np.float32), 9000, SR)  # outside: nothing is added
    assert buf[2999] == 1.0


def test_cache_keys_cover_arrays_types_and_order():
    stable_hash = nb.stable_hash
    a = np.arange(6, dtype=np.float32)
    k = stable_hash("1", ["SP", "en/aa"], [8, 20], a, {"voicing": a, "tension": 0.0}, None)
    assert k == stable_hash("1", ["SP", "en/aa"], [8, 20], a.copy(), {"tension": 0.0, "voicing": a}, None)
    others = [stable_hash("2", ["SP", "en/aa"], [8, 20], a, {"voicing": a, "tension": 0.0}, None),
              stable_hash("1", ["SP", "en/ae"], [8, 20], a, {"voicing": a, "tension": 0.0}, None),
              stable_hash("1", ["SP", "en/aa"], [8, 21], a, {"voicing": a, "tension": 0.0}, None),
              stable_hash("1", ["SP", "en/aa"], [8, 20], a + 1e-6, {"voicing": a, "tension": 0.0}, None),
              stable_hash("1", ["SP", "en/aa"], [8, 20], a.astype(np.float64), {"voicing": a, "tension": 0.0}, None),
              stable_hash("1", ["SP", "en/aa"], [8, 20], a.reshape(2, 3), {"voicing": a, "tension": 0.0}, None),
              stable_hash("1", ["SP", "en/aa"], [8, 20], a, {"voicing": a, "tension": 0.5}, None),
              stable_hash("1", ["SP", "en/aa"], [8, 20], a, {"voicing": a, "tension": 0.0}, 7)]
    assert len(set(others + [k])) == len(others) + 1 and len(k) == 64


def test_phrase_cache_key_changes_with_every_setting(fake, monkeypatch):

    keys = []
    real = nb.stable_hash
    monkeypatch.setattr(nb, "stable_hash", lambda *parts: keys.append(real(*parts)) or keys[-1])

    def first_key(mutate=None, env=None, quality=None):
        p, tr = song()
        if mutate:
            mutate(p, tr)
        for k, v in (env or {}).items():
            monkeypatch.setenv(k, v)
        keys.clear()
        with render_options(quality=quality):
            sing(p, tr, t0=0.0, t1=1.0)
        for k in env or {}:
            monkeypatch.delenv(k)
        return keys[0]

    base = first_key()
    assert base == first_key()
    variants = [
        first_key(lambda p, tr: tr.voice_params.update({"neural.gender": 0.3})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.tone": 1.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.breath_db": 1.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.voicing_db": -1.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.velocity": 1.1})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.expr": 0.6})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.slow": 0.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.voicing_floor": 0.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.dial": 0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.dial": 1})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.cons_cap.fric": 0.02})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.cons_cap_share": 0.1})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.cons_cap_keep": 0.9})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.onset_keep": 0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.cons_floor.glide": 3.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.burst": 0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.burst.stop": 0.2})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.vowel_breath_db": -1.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.detune": 3.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.breaths": 0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.breaths": 0.7})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.cons_gain.fric": 2.0})),
        first_key(lambda p, tr: tr.voice_params.update({"neural.cons_gain.fric": 2.0, "neural.cons_gain.hf": 3000})),
        first_key(lambda p, tr: setattr(tr, "seed", tr.seed + 1)),
        first_key(lambda p, tr: setattr(tr.notes[0], "pitch", tr.notes[0].pitch + 1)),
        first_key(lambda p, tr: setattr(tr.notes[0], "lyric", "yellow")),
        first_key(env={"SYNTH2_NEURAL_REPRODUCIBLE": "0"}),
        first_key(env={"SYNTH2_NEURAL_PITCH": "expression"}),
        first_key(env={"SYNTH2_NEURAL_STEPS": "9"}),
        first_key(quality="draft"),
    ]
    point = lambda name, value: lambda p, tr: tr.curve(name).add(0.5, value)
    variants += [first_key(point(name, value)) for name, value in (
        ("pitchDelta", 30.0), ("breathiness", 0.4), ("breathiness", 0.5), ("voicing", 0.6), ("gender", 0.3),
        ("toneShift", 50.0), ("vibratoEnv", 0.2))]
    assert base not in variants and len(set(variants)) == len(variants)
    # level, loudness, effects and (for a bank without a tension input) tension are applied after the cache:
    # they do not change the key
    for same in (lambda p, tr: tr.voice_params.update({"neural.level_db": -3.0, "neural.fx.alien": 1.0}),
                 point("loudness", -3.0), point("tension", 0.5)):
        assert base == first_key(same)
    # two ways to the same model input are one entry: the dial named by the track or by the environment,
    assert first_key(env={"SYNTH2_NEURAL_DIAL": "balanced"}) == variants[variants.index(base) if base in variants else 9]
    # a constant toneShift curve and the tone parameter
    assert first_key(point("toneShift", 100.0)) == variants[1]


def test_phrase_cache_serves_repeats_without_running_the_models(fake, monkeypatch):
    store = {}
    stand_in = types.SimpleNamespace(
        enabled=lambda: os.environ.get("SYNTH2_NEURAL_CACHE", "1") != "0",
        default_cache=lambda: types.SimpleNamespace(get=store.get, put=store.__setitem__))
    monkeypatch.setattr(nb, "phrase_cache", stand_in)  # a dict in place of the cache folder
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "1")
    p, tr = song()
    first, rep = sing(p, tr)
    assert rep["neural"]["cached"] == 0 and len(store) == 6  # three phrases: audio and a note about the pitch
    assert all(v.dtype == np.float32 for v in store.values())
    fake.calls.clear()
    again, rep = sing(p, tr)
    assert np.array_equal(first, again) and rep["neural"]["cached"] == 3
    assert {c[0] for c in fake.calls} == {"dsdur/linguistic.onnx", "dsdur/dur.onnx"}  # only the layout is computed
    tr.voice_params["neural.level_db"] = -6.0  # level is applied after the cache
    quiet, rep = sing(p, tr)
    assert rep["neural"]["cached"] == 3
    assert np.allclose(quiet, first * 10 ** ((-6 - PRESETS["aurora"].level_db) / 20), atol=1e-6)
    tr.notes[0].pitch += 2  # one phrase changes, the other two are served from the cache
    _, rep = sing(p, tr)
    assert rep["neural"]["cached"] == 2 and len(store) == 8
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "0")
    assert sing(p, tr)[1]["neural"]["cached"] == 0


def test_effects_run_on_the_bank_rate_buffer_with_absolute_time(fake, monkeypatch):
    seen = []

    def parse_fx(spec):
        items = [spec] if isinstance(spec, str) else list(spec)
        for item in items:
            if item.split(":")[0] not in ("alien", "radio"):
                raise ValueError(f"unknown effect {item!r}; valid: alien, radio")
        return [{"type": "gain"} for _ in items]

    def apply_fx(x, sr, spec, *, t0=0.0, seed=0):
        seen.append((len(x), sr, list(spec), t0, seed))
        return (np.asarray(x) * 0.5).astype(np.float32)

    stand_in = types.SimpleNamespace(parse_fx=parse_fx, apply_fx=apply_fx, fx_tail=lambda spec: 0.25)
    monkeypatch.setattr(nb, "fx", stand_in)  # records what the renderer hands the effects
    p, tr = song()
    ref, _ = sing(p, tr)
    assert not seen  # no effect unless a neural.fx parameter asks for one
    tr.voice_params.update({"neural.fx.alien": 0.6, "neural.fx.zap": 1.0})
    wet, rep = sing(p, tr, t0=1.0, t1=2.0)
    assert np.allclose(wet, 0.5 * ref[SR:2 * SR], atol=1e-7)
    n, sr, spec, t0, seed = seen[-1]
    assert sr == SR and spec == ["alien:0.6"] and seed == tr.seed
    assert t0 == pytest.approx(0.75) and n == SR + int(0.25 * SR)  # the effect's pre-roll comes before the window
    assert any("'zap' ignored" in w for w in rep["warnings"])
    tr.voice_params = {"neural.fx.zap": 1.0}  # nothing valid is left: no effect call and no pre-roll
    seen.clear()
    dry, rep = sing(p, tr, t0=1.0, t1=2.0)
    assert np.array_equal(dry, ref[SR:2 * SR]) and not seen and any("'zap' ignored" in w for w in rep["warnings"])
