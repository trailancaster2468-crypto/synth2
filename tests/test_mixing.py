"""Mixing / mastering DSP (synth2.engine.mixing) and its integration in render_project."""
import sys
import types
from dataclasses import dataclass

import numpy as np
import pytest

from synth2 import Project
from synth2.engine import mixing as M
from synth2.engine import render_project
from synth2.project import MasterSettings

SR = 22050


def _lfilter(b, a, x):
    """Direct-form biquad (reference for the FFT implementation)."""
    b = np.asarray(b) / a[0]
    a = np.asarray(a) / a[0]
    y = np.zeros(len(x))
    x1 = x2 = y1 = y2 = 0.0
    for i, v in enumerate(x):
        out = b[0] * v + b[1] * x1 + b[2] * x2 - a[1] * y1 - a[2] * y2
        x2, x1, y2, y1 = x1, v, y1, out
        y[i] = out
    return y


def _sine(freq, sec, sr, amp=1.0, phase=0.0):
    t = np.arange(int(sec * sr)) / sr
    return amp * np.sin(2 * np.pi * freq * t + phase)


# --------------------------------------------------------------------------------------- EQ
def test_eq_band_responses():
    f = np.array([20.0, 100.0, 1000.0, 3200.0, 10000.0, 20000.0])
    db = lambda bands: 20 * np.log10(np.abs(M.eq_response(bands, 48000, f)))  # noqa: E731
    assert db([("peak", 3200, 4.0, 1.0)])[3] == pytest.approx(4.0, abs=1e-6)
    assert abs(db([("peak", 3200, 4.0, 1.0)])[0]) < 0.05
    assert db([("hp", 100, 0, 0.7071)])[1] == pytest.approx(-3.01, abs=0.02)  # Butterworth -3 dB at fc
    assert db([("hp", 100, 0, 0.7071, 4)])[1] == pytest.approx(-3.01, abs=0.05)
    assert db([("hp", 100, 0, 0.7071, 4)])[0] < db([("hp", 100, 0, 0.7071, 2)])[0] - 20  # 24 vs 12 dB/oct
    hs = db([("highshelf", 3000, 6.0, 0.7071)])
    assert hs[5] == pytest.approx(6.0, abs=0.3) and abs(hs[1]) < 0.05
    ls = db([("lowshelf", 300, -4.0, 0.7071)])
    assert ls[0] == pytest.approx(-4.0, abs=0.1) and abs(ls[5]) < 0.05
    assert db([("bandpass", 1000, 0, 2.0)])[2] == pytest.approx(0.0, abs=1e-6)
    with pytest.raises(ValueError):
        M.as_band(("wobble", 100))


def test_eq_matches_the_iir_filter():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    bands = [("hp", 90, 0, 0.7071, 2), ("peak", 300, -2, 1.0), ("peak", 3000, 3, 0.9), ("highshelf", 8000, 2.5, 0.7),
             ("lp", 9000, 0, 0.7071, 4)]
    ref = x.copy()
    for b in bands:
        for bb, aa in M.biquads(M.as_band(b), SR):
            ref = _lfilter(bb, aa, ref)
    y = M.eq(x, bands, SR)
    assert np.max(np.abs(y - ref)) < 2e-4 * np.max(np.abs(ref))
    st = M.eq(np.stack([x, -x], axis=1), bands, SR)  # stereo: each channel the same filter
    assert st.shape == (4000, 2) and np.allclose(st[:, 0], -st[:, 1], atol=1e-6)


# --------------------------------------------------------------------------------------- dynamics
def test_compressor_static_curve_and_gain_reduction():
    lvl = np.array([-40.0, -20.0, -10.0, 0.0])
    gr = M.static_gr_db(lvl, -20.0, 4.0, knee_db=0.0)
    assert np.allclose(gr, [0.0, 0.0, 7.5, 15.0])
    soft = M.static_gr_db(np.array([-20.0]), -20.0, 4.0, knee_db=6.0)[0]
    assert 0 < soft < 1.0  # inside the knee
    # a -6 dBFS-peak sine (RMS -9.03 dB), threshold -20, 4:1 -> 8.2 dB of gain reduction
    x = _sine(1000, 1.0, SR, 10 ** (-6 / 20))
    y = M.compress(x, SR, threshold_db=-20.0, ratio=4.0, attack=0.005, release=0.05, knee_db=0.0)
    seg = y[int(0.5 * SR):int(0.9 * SR)]
    out_db = 10 * np.log10(np.mean(seg ** 2))
    assert out_db == pytest.approx(-9.03 - 0.75 * 10.97, abs=0.3)
    # attack: the reduction builds up with the attack time constant
    lv = np.r_[np.full(100, -60.0), np.full(400, 0.0)]
    g = M.compressor_db(lv, -20.0, 4.0, attack=0.01, release=0.1, knee_db=0.0)
    assert -g[100 + 10] == pytest.approx(15.0 * (1 - np.exp(-1)), rel=0.15)
    assert -g[-1] == pytest.approx(15.0, abs=0.05)


def test_ar_smooth_matches_reference_loop():
    rng = np.random.default_rng(3)
    x = np.maximum(0, rng.standard_normal(3000)) * 5
    y = M.ar_smooth(x, 3.0, 40.0, start=0.0)
    # reference: peak follower with exponential release, then a one-pole attack filter
    cr, ca = np.exp(-1 / 40.0), np.exp(-1 / 3.0)
    p = s = 0.0
    ref = []
    for v in x:
        p = max(v, cr * p)
        s = ca * s + (1 - ca) * p
        ref.append(s)
    assert np.allclose(y, ref, atol=1e-9)


def test_kick_sidechain_shape():
    sr = 44100
    hop = M.grid_hop(sr)
    blk = hop / sr
    env = M.kick_duck_db([0.5], int(1.2 / blk), sr, depth_db=6.0, attack=0.004, hold=0.012, release=0.16)
    t = (np.arange(len(env)) + 0.5) * blk
    at = lambda s: env[int(np.argmin(np.abs(t - s)))]  # noqa: E731
    assert at(0.49) == 0.0 and at(0.4995) < 0  # the duck starts just before the kick
    assert at(0.5005) == pytest.approx(-6.0, abs=0.01) and at(0.51) == pytest.approx(-6.0, abs=0.01)
    assert at(0.512 + 0.08) == pytest.approx(-3.0, abs=0.3)  # half way through the release
    assert at(0.512 + 0.17) == 0.0
    rec = env[(t > 0.513) & (t < 0.672)]
    assert np.all(np.diff(rec) >= -1e-12)  # monotonic recovery
    two = M.kick_duck_db([0.5, 0.6], int(1.2 / blk), sr, 6.0)
    assert two.min() >= -6.0 - 1e-9  # overlapping ducks do not stack


def test_vocal_activity_and_dynamic_band():
    sr = SR
    x = np.zeros(sr * 2)
    x[int(0.5 * sr):int(1.0 * sr)] = _sine(300, 0.5, sr, 0.1)
    act = M.activity(x, sr)
    hop = M.grid_hop(sr)
    blk = hop / sr
    assert act[int(0.3 / blk)] == 0.0 and act[int(0.9 / blk)] > 0.95
    assert 0.0 < act[int(1.25 / blk)] < 0.95 and act[int(1.95 / blk)] < 0.1  # held, then released
    full = np.full(100, -20.0)
    band = np.r_[np.full(50, -30.0), np.full(50, -18.0)]
    g = M.dynamic_band_db(full, band, rel_threshold_db=-6.0, ratio=3.0, max_cut_db=6.0, attack=0.001, release=0.001)
    assert np.all(g[:50] == 0) and g[-1] == pytest.approx(-8.0 * (1 - 1 / 3), abs=0.05)  # 8 dB over, 3:1


# --------------------------------------------------------------------------------------- loudness / peaks
def test_k_weighting_coefficients_match_bs1770():
    (b1, a1), (b2, a2) = M.k_weighting(48000)
    assert np.allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-8)
    assert np.allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-8)
    assert np.allclose(b2, [1.0, -2.0, 1.0]) and np.allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-8)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_lufs_of_997hz_sines(sr):
    """BS.1770 calibration: a 997 Hz sine whose peak is at X dBFS in *both* channels reads X LUFS
    (EBU Tech 3341 test 1: -23 dBFS stereo -> -23.0 LUFS). The same sine in one channel only (or as a
    mono signal) is 3 dB lower: a -20 dBFS sine reads -23.0 LUFS."""
    s23 = _sine(997, 6.0, sr, 10 ** (-23 / 20))
    s20 = _sine(997, 6.0, sr, 10 ** (-20 / 20))
    assert M.integrated_loudness(np.stack([s23, s23], 1), sr) == pytest.approx(-23.0, abs=0.05)
    assert M.integrated_loudness(np.stack([s20, s20], 1), sr) == pytest.approx(-20.0, abs=0.05)
    assert M.integrated_loudness(s20, sr) == pytest.approx(-23.0, abs=0.05)
    assert M.integrated_loudness(np.stack([s20, 0 * s20], 1), sr) == pytest.approx(-23.0, abs=0.05)
    st = M.loudness_stats(np.stack([s23, s23], 1), sr)
    assert st["momentary_max"] == pytest.approx(-23.0, abs=0.05) and st["short_term_max"] == pytest.approx(-23.0, abs=0.05)


def test_lufs_gating():
    """EBU Tech 3341 test 3: 10 s at -36, 60 s at -23, 10 s at -36 dBFS (stereo) -> -23.0 LUFS
    (the quiet parts fall under the relative gate); silence alone has no loudness."""
    sr = 8000
    sig = np.concatenate([_sine(997, 10, sr, 10 ** (-36 / 20)), _sine(997, 60, sr, 10 ** (-23 / 20)),
                          _sine(997, 10, sr, 10 ** (-36 / 20))])
    assert M.integrated_loudness(np.stack([sig, sig], 1), sr) == pytest.approx(-23.0, abs=0.1)
    assert M.integrated_loudness(np.zeros((sr * 2, 2)), sr) is None
    assert M.loudness_stats(np.zeros((100, 2)), sr)["integrated"] is None


def test_true_peak_finds_inter_sample_peaks():
    s = _sine(SR / 4, 0.5, SR, 1.0, np.pi / 4)  # samples at +-0.707, the waveform peaks at 1.0
    fade = np.minimum(1.0, np.minimum(np.arange(len(s)), np.arange(len(s))[::-1]) / (0.02 * SR))
    s = s * (0.5 - 0.5 * np.cos(np.pi * fade))  # no edge (an abrupt start overshoots for real)
    assert np.max(np.abs(s)) == pytest.approx(0.7071, abs=1e-3)
    assert M.true_peak_db(s) == pytest.approx(0.0, abs=0.2)
    env = M.true_peak_envelope(np.stack([s, s], 1))
    assert env.max() == pytest.approx(M.true_peak(s), rel=1e-6)


def test_limiter_ceiling():
    rng = np.random.default_rng(1)
    x = M.eq(rng.standard_normal((SR * 2, 2)) * 0.5, [("lp", 8000, 0, 0.7071)], SR).astype(np.float64)
    x[5000] *= 6.0  # a wild transient
    for ceiling in (-1.0, -3.0):
        y = M.limit(x, SR, ceiling)
        assert M.true_peak_db(y) <= ceiling + 0.01
        assert 20 * np.log10(np.max(np.abs(y))) <= ceiling + 0.01
    quiet = x * 0.05
    assert np.array_equal(M.limit(quiet, SR, -1.0), quiet)  # nothing to do below the ceiling
    g = M.limiter_gain(np.abs(x).max(axis=1), SR, 0.5)
    assert np.all(g * np.abs(x).max(axis=1) <= 0.5 + 1e-9) and g.max() <= 1.0


def test_master_to_loudness_hits_the_target():
    rng = np.random.default_rng(2)
    n = SR * 6
    beat = (np.arange(n) % (SR // 2) < SR // 20).astype(float)  # punchy bursts: needs limiting
    x = np.stack([rng.standard_normal(n), rng.standard_normal(n)], 1) * (0.1 + 0.15 * beat[:, None])
    x = M.eq(x, [("lp", 6000, 0, 0.7071)], SR).astype(np.float64)
    y, info = M.master_to_loudness(x, SR, -12.0, ceiling_db=-1.0)
    assert info["target_reached"] and M.integrated_loudness(y, SR) == pytest.approx(-12.0, abs=0.1)
    assert M.true_peak_db(y) <= -1.0 + 0.01 and info["limiter_max_db"] > 0.5
    assert info["true_peak"] == pytest.approx(M.true_peak(y), rel=1e-3)
    y2, info2 = M.master_to_loudness(x, SR, -12.0, ceiling_db=-1.0, limiter=False)
    assert M.true_peak_db(y2) <= -1.0 + 0.01  # without the limiter the gain stops at the ceiling
    y3, info3 = M.master_to_loudness(x, SR, -3.0, ceiling_db=-1.0, max_limiting_db=6.0)  # absurd target
    assert not info3["target_reached"] and info3["limiter_max_db"] <= 6.5 and M.true_peak_db(y3) <= -0.99


# --------------------------------------------------------------------------------------- stereo / time
def test_pingpong_delay_alternates_and_decays():
    x = np.zeros(SR * 2)
    x[100] = 1.0
    d = 0.1
    y = M.pingpong_delay(x, SR, d, feedback=0.5, hp_hz=50, lp_hz=10000, damp_hz=10000)
    assert y.shape[1] == 2 and len(y) > len(x)
    e = []
    for k in range(1, 5):
        c = 100 + int(round(k * d * SR))
        seg = y[c - 200:c + 400]
        e.append((np.sum(seg[:, 0] ** 2), np.sum(seg[:, 1] ** 2)))
    assert e[0][0] > 0 and e[0][1] < 1e-6 * e[0][0]  # first repeat left
    assert e[1][1] > 0 and e[1][0] < 1e-6 * e[1][1]  # second right
    assert e[1][1] < e[0][0] and e[2][0] < e[1][1]  # decaying
    assert np.sqrt(e[1][1] / e[0][0]) == pytest.approx(0.5, rel=0.25)


def test_stereo_width_and_mono_bass():
    lo, hi = _sine(60, 1, SR), _sine(3000, 1, SR)
    x = np.stack([lo + hi, -(lo + hi)], 1) * 0.3  # all side
    mono = M.stereo_width(x, SR, 0.0)
    assert np.max(np.abs(mono)) < 1e-4
    y = M.stereo_width(x, SR, 1.0, mono_below_hz=200)
    side = (y[:, 0] - y[:, 1]) / 2
    spec = np.abs(np.fft.rfft(side[SR // 4:] * np.hanning(len(side) - SR // 4)))
    f = np.fft.rfftfreq(len(side) - SR // 4, 1 / SR)
    assert spec[np.argmin(np.abs(f - 60))] < 0.15 * spec[np.argmin(np.abs(f - 3000))]


# --------------------------------------------------------------------------------------- settings
def test_master_settings_backward_compatible():
    old = {"gain_db": -2.0, "reverb_room": 0.3, "reverb_wet": 0.8, "normalize": False, "limiter": True}
    m = MasterSettings.from_dict(old)
    assert m.gain_db == -2.0 and m.loudness_target is None and m.vocal_chain and m.vocal_delay == 0.0
    assert MasterSettings.from_dict({**old, "bogus": 1}) == m
    m.mix_overrides["bass"] = {"gain_db": -3}
    m.loudness_target = -9.0
    m2 = MasterSettings.from_dict(m.to_dict())
    assert m2 == m
    m2.mix_overrides["bass"]["gain_db"] = 0
    assert m.mix_overrides["bass"]["gain_db"] == -3  # deep copy
    p = Project.new("x")
    p.master = m
    assert Project.from_dict(p.to_dict()).master == m


# --------------------------------------------------------------------------------------- song mix
@dataclass
class FakeStem:
    name: str
    role: str
    audio: np.ndarray
    gain_db: float = 0.0
    pan: float = 0.0
    sidechain: bool = False


BEAT = 0.5  # s at 120 bpm


def _fake_arranger(length_s: float, calls: list | None = None):
    """A tiny backing provider with the synth2.arranger contract: a kick on every beat, a bass
    (sidechained), a bright keys part and hats."""
    mod = types.ModuleType("synth2.arranger")

    def kick_times(project):
        return np.arange(0.0, length_s, BEAT)

    def song_length_seconds(project):
        return length_s

    def render_backing(project, sr, t0, t1):
        if calls is not None:
            calls.append((t0, t1))
        n = int(round((t1 - t0) * sr))
        t = t0 + np.arange(n) / sr
        on = (t < length_s).astype(float)
        ph = np.mod(t, BEAT)
        kick = np.sin(2 * np.pi * (50 * ph + 60 * (1 - np.exp(-ph / 0.03)) * 0.03)) * np.exp(-ph / 0.12) * on
        bass = 0.4 * np.sign(np.sin(2 * np.pi * 55 * t)) * on
        keys = sum(0.08 * np.sin(2 * np.pi * f * k * t) / k for f in (440, 554.4, 659.3) for k in (1, 2, 3, 4)) * on
        rng = np.random.default_rng(int(t0 * 1000))
        hats = 0.05 * rng.standard_normal(n) * np.exp(-np.mod(t, BEAT / 2) / 0.02) * on
        st = lambda m, w=0.0: np.stack([m * (1 - w), m * (1 + w)], 1).astype(np.float32)  # noqa: E731
        return [FakeStem("drums kick", "drums", st(kick), 0.0), FakeStem("drums hats", "drums", st(hats, 0.2), -4.0),
                FakeStem("bass", "bass", st(bass), -2.0, 0.0, True),
                FakeStem("keys", "keys", st(keys, 0.3), -3.0, -0.2, True)]

    mod.render_backing, mod.kick_times, mod.song_length_seconds = render_backing, kick_times, song_length_seconds
    return mod


def _song(sec=4.0):
    p = Project.new("song", bpm=120, voice="aurora")
    p.sample_rate = SR
    lead = p.tracks[0]
    for i, (pitch, ly) in enumerate([(64, "la"), (67, "la"), (69, "la"), (67, "la"), (64, "oh")]):
        lead.add_note(1 + i, 1, pitch, ly)
    p.add_track("Harmony", voice="seraph", pan=0.4, gain_db=-6).add_note(1, 4, 60, "ah")
    p.backing = {"fake": True}
    return p


def test_render_song_with_backing(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setitem(sys.modules, "synth2.arranger", _fake_arranger(4.5, calls))
    p = _song()
    p.master.vocal_delay = 0.2
    r = render_project(p)
    rep = r.report
    assert rep["mix"]["mode"] == "song" and calls and calls[0][0] == 0.0
    assert rep["loudness_lufs"] == pytest.approx(-11.0, abs=0.3)
    assert rep["true_peak_db"] <= -1.0 + 0.02 and M.true_peak_db(r.audio) <= -1.0 + 0.02
    assert rep["vocal_to_band_db"] == pytest.approx(4.0, abs=0.3)
    assert rep["duration_s"] >= 4.5  # the backing outlasts the vocals
    keys = set(r.stems)
    assert {"drums kick", "drums hats", "bass", "keys"} <= keys and {t.id for t in p.tracks} <= keys
    levels = {s["name"]: s for s in rep["stem_levels"]}
    assert levels["bass"]["group"] == "backing" and levels["Lead"]["group"] == "vocal"
    assert all(s["peak_db"] < 0 for s in rep["stem_levels"])
    assert [b["name"] for b in rep["backing"]] == ["drums kick", "drums hats", "bass", "keys"]
    assert rep["mix"]["backing"][2].get("sidechain_db", 0) > 0
    import json

    json.dumps(rep, allow_nan=False)  # the server sends it as JSON
    # stems carry the glue and loudness gain: they sum to the master minus the effect returns,
    # before the limiter (which only trims peaks)
    dry = sum(v.astype(np.float64) for v in r.stems.values())
    c = np.sum(dry * r.audio) / np.sqrt(np.sum(dry ** 2) * np.sum(r.audio.astype(np.float64) ** 2))
    assert c > 0.95
    paths = r.save_stems(tmp_path)
    assert sorted(x.name for x in paths) == sorted(["Lead.wav", "Harmony.wav", "drums_kick.wav", "drums_hats.wav",
                                                    "bass.wav", "keys.wav"])


def test_band_ducks_and_carves_under_the_vocals(monkeypatch):
    sr = SR
    n = sr * 3
    vocal = np.zeros(n)
    vocal[sr:2 * sr] = _sine(2600, 1.0, sr, 0.1)
    band = np.stack([_sine(2600, 3.0, sr, 0.1)] * 2, 1).astype(np.float32)
    srcs = [M.Source("v", "Vox", "vocal", vocal, lowest_hz=200.0),
            M.Source("k", "keys", "backing", band, role="keys")]
    ms = MasterSettings(vocal_balance_db=None, band_glue=0.0, band_duck_db=2.0, vocal_carve_db=4.0)
    mix = M.mix_sources(srcs, sr, band_mode=True, settings=ms, reverb_wet=0.0)
    k = mix.stems["k"].astype(np.float64)
    rms = lambda a, b: 10 * np.log10(np.mean(k[int(a * sr):int(b * sr)] ** 2))  # noqa: E731
    dip = rms(0.3, 0.9) - rms(1.4, 1.9)
    assert 4.5 < dip < 7.5  # 2 dB duck + 4 dB carve at the carve centre (keys role factor 1)
    assert abs(rms(0.3, 0.9) - rms(2.75, 2.95)) < 0.5  # back to normal after the hold and release


def test_mix_is_independent_of_thread_count():
    rng = np.random.default_rng(5)
    n = SR * 2
    srcs = [M.Source("v", "Vox", "vocal", _sine(330, 2, SR, 0.2) * (np.arange(n) % SR < SR // 2), lowest_hz=300.0)]
    for i, role in enumerate(("bass", "keys", "pad", "lead")):
        srcs.append(M.Source(f"b{i}", role, "backing", (rng.standard_normal((n, 2)) * 0.1).astype(np.float32),
                             role=role, sidechain=i % 2 == 0))
    kw = dict(band_mode=True, settings=MasterSettings(vocal_delay=0.2), kicks=np.arange(0, 2, 0.5), bpm=120)
    a = M.mix_sources(srcs, SR, threads=1, **kw)
    b = M.mix_sources(srcs, SR, threads=4, **kw)
    assert np.array_equal(a.dry, b.dry) and np.array_equal(a.wet, b.wet)
    assert all(np.array_equal(a.stems[k], b.stems[k]) for k in a.stems)


def test_partial_render_subset_and_solo_with_backing(monkeypatch):
    calls = []
    monkeypatch.setitem(sys.modules, "synth2.arranger", _fake_arranger(4.5, calls))
    p = _song()
    p.title = "partial test"  # (a project of its own for the level cache)
    part = render_project(p, start_beat=2, end_beat=4)
    assert part.report["mix"]["mode"] == "song" and "bass" in part.stems
    assert not part.report["mix"]["levels_from_full_render"]  # no full render yet: measured on the region
    full = render_project(p)
    part = render_project(p, start_beat=2, end_beat=4)
    mx, fx = part.report["mix"], full.report["mix"]
    assert mx["levels_from_full_render"] and mx["band_gain_db"] == fx["band_gain_db"]
    assert mx["master_gain_db"] == fx["master_gain_db"]
    a, b = int(1.2 * SR), int(1.8 * SR)  # inside the region (beats 2-4 = 1-2 s): same as the full render
    seg = lambda r, o: r.audio[a - o:b - o].astype(np.float64)  # noqa: E731
    o = int(round(part.start_seconds * SR))
    full_db = 10 * np.log10(np.mean(full.audio[a:b].astype(np.float64) ** 2))
    part_db = 10 * np.log10(np.mean(seg(part, o) ** 2))
    assert abs(full_db - part_db) < 0.5
    assert 1.0 < part.duration < 1.0 + 0.3 + 2.2 * p.master.reverb_room + 0.2 and np.isfinite(part.audio).all()
    assert calls[-1][0] < 1.0  # rendered from before the region (reverb pre-roll)
    n0 = len(calls)
    sub = render_project(p, track_ids=["Lead"])
    assert len(calls) == n0 and sub.report["mix"]["mode"] == "vocal" and "bass" not in sub.stems
    p.tracks[1].solo = True
    solo = render_project(p)
    assert len(calls) == n0 and list(solo.stems) == [p.tracks[1].id]


def test_backing_without_arranger_falls_back_to_vocals(monkeypatch):
    monkeypatch.setitem(sys.modules, "synth2.arranger", None)  # import raises ImportError
    p = _song()
    r = render_project(p)
    assert r.report["mix"]["mode"] == "vocal" and any("arranger" in w for w in r.report["warnings"])
    assert r.report["peak_db"] == pytest.approx(-1.0, abs=0.05)  # the classic peak normalisation


def test_vocal_only_loudness_target_and_overrides(monkeypatch):
    monkeypatch.setitem(sys.modules, "synth2.arranger", _fake_arranger(4.5))
    p = _song()
    p.backing = None
    base = render_project(p)
    assert base.report["mix"]["mode"] == "vocal" and base.report["loudness_lufs"] is not None
    p.master.loudness_target = -16.0
    r = render_project(p)
    assert r.report["mix"]["mode"] == "mastered" and r.report["loudness_lufs"] == pytest.approx(-16.0, abs=0.3)
    # per-stem override: the bass 6 dB down relative to the rest of the band
    p.backing = {"fake": True}
    p.master.loudness_target = None
    p.master.vocal_balance_db = None
    a = render_project(p)
    p.master.mix_overrides = {"bass": {"gain_db": -6.0}}
    b = render_project(p)
    rel = lambda r: r.report["stem_levels"]  # noqa: E731
    lv = lambda r, name: next(s["rms_db"] for s in rel(r) if s["name"] == name)  # noqa: E731
    assert (lv(a, "bass") - lv(a, "keys")) - (lv(b, "bass") - lv(b, "keys")) == pytest.approx(6.0, abs=0.7)


def test_low_end_balance_and_drum_transient_limiter():
    n = SR * 3
    t = np.arange(n) / SR
    mids = (0.05 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    boomy = (0.5 * np.sin(2 * np.pi * 45 * t)).astype(np.float32)
    S = M._Settings(MasterSettings())
    srcs = [M.Source("b", "bass", "backing", np.stack([boomy + mids] * 2, 1), role="bass")]
    trim, ratio = M._low_end_trim(srcs, SR, [], S)
    assert ratio > 10 and trim == pytest.approx(-4.0)  # capped at 4 dB of cut
    srcs = [M.Source("b", "bass", "backing", np.stack([0.1 * boomy + mids] * 2, 1), role="bass")]
    trim, ratio = M._low_end_trim(srcs, SR, [], S)
    assert trim == 0.0 and ratio < 4.0
    # clicky hats: peaks capped at the loud level + crest_db
    rng = np.random.default_rng(4)
    hats = 0.02 * rng.standard_normal(n) * np.exp(-np.mod(t, 0.125) / 0.03)
    hats[::SR // 8] = 0.9  # one-sample clicks
    src = M.Source("h", "drums hats", "backing", np.stack([hats, hats], 1).astype(np.float32), role="drums")
    mix = M.mix_sources([src], SR, band_mode=True, settings=MasterSettings(), reverb_wet=0.0)
    info = mix.info["backing"][0]
    assert info["transient_max_db"] > 6.0
    assert np.max(np.abs(mix.stems["h"])) < 0.5 * 0.9


def test_vocal_transient_limiter_is_opt_in():
    n = SR * 2
    t = np.arange(n) / SR
    # a buzzy low "voice": 110 Hz pulse train with a few much stronger pulses
    x = 0.05 * np.sin(2 * np.pi * 220 * t)
    x[:: SR // 110] += 0.1
    x[SR // 2:: SR // 4] += 0.8
    x = x.astype(np.float32)
    plain = M.Source("v", "voice", "vocal", x, lowest_hz=110.0)
    y0, info0 = M._processed_vocal(plain, SR, True, True, [])
    assert "transient_max_db" not in info0  # off by default
    capped = M.Source("v", "voice", "vocal", x, lowest_hz=110.0, overrides={"crest_db": 10.0})
    y1, info1 = M._processed_vocal(capped, SR, True, True, [])
    assert info1["transient_max_db"] > 3.0
    assert np.max(np.abs(y1)) < 0.25 * np.max(np.abs(y0))
    # the body of the voice is (nearly) untouched
    q = slice(SR // 2 + SR // 8 - 900, SR // 2 + SR // 8 + 900)  # between two strong pulses
    assert np.sqrt(np.mean(y1[q] ** 2)) == pytest.approx(np.sqrt(np.mean(y0[q] ** 2)), rel=0.08)
