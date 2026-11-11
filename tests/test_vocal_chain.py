"""The vocal chain and the roles of vocal tracks in the mix (numpy only, no models)."""
import json

import numpy as np
import pytest

from synth2.engine import mixing
from synth2.engine import vocal_chain as vc
from synth2.project import Project, Track
from synth2.score import parse_score, project_to_score

SR = 32000


def tone(dur, f0=220.0, level=0.2, sr=SR):
    """A sung note: harmonics of f0 falling by 6 dB per octave."""
    t = np.arange(int(dur * sr)) / sr
    x = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 12))
    return level * x / np.abs(x).max()


def hiss(dur, level=0.05, lo=4500.0, seed=0, sr=SR):
    """A sibilant: noise above ``lo`` Hz."""
    x = np.random.default_rng(seed).standard_normal(int(dur * sr))
    x = mixing.eq(x, [("hp", lo, 0, 0.7071, 4)], sr)
    return level * x / np.abs(x).max()


def gap(dur, sr=SR):
    return np.zeros(int(dur * sr))


def voice(levels=(0.25, 0.08, 0.2, 0.05), sib=0.05, seed=0):
    """Notes at different levels with a sibilant between each two and a rest after every pair."""
    parts, marks, at = [gap(0.2)], [], 0.2
    for i, lv in enumerate(levels * 2):
        for kind, seg in (("s", hiss(0.09, sib, seed=seed + i)), ("v", tone(0.35, 200.0 + 20 * (i % 4), lv))):
            marks.append((kind, at, at + len(seg) / SR))
            parts.append(seg)
            at += len(seg) / SR
        if i % 2:
            parts.append(gap(0.25))
            at += 0.25
    return np.concatenate(parts + [gap(0.3)]).astype(np.float32), marks


def voiced_s(level=0.05):
    """Two notes with a sibilant between them, the voice sounding on quietly under it."""
    return np.concatenate([tone(0.4), hiss(0.06, level) + tone(0.06, level=level / 6), tone(0.4)]).astype(np.float32)


def rms_db(x, a, b, sr=SR):
    seg = np.asarray(x[int(a * sr):int(b * sr)], np.float64)
    return 10 * np.log10(np.mean(seg ** 2) + 1e-20)


def test_chain_keeps_length_type_and_active_level():
    x, _ = voice()
    for cfg in (vc.lead_chain("smooth"), vc.lead_chain("balanced"), vc.lead_chain("crisp"), vc.HARMONY, None):
        y, info = vc.process(x, SR, cfg)
        assert y.dtype == np.float32 and y.shape == x.shape and np.isfinite(y).all()
        hop = mixing.grid_hop(SR)
        lvl = lambda s: vc.active_rms(np.asarray(s, np.float64), vc.level_db(s, SR, 0.02), hop)
        assert 20 * np.log10(lvl(y) / lvl(x)) == pytest.approx(0.0, abs=0.05)
        assert "makeup_db" in info and json.dumps(info)
    silent, info = vc.process(np.zeros(SR, np.float32), SR)
    assert not silent.any() and info == {}
    with pytest.raises(KeyError):
        vc.process(x, SR, {"wobble": {}})
    y, info, gain = vc.process(x, SR, {"rider": vc.LEAD["rider"]}, return_gain=True)
    assert len(gain) == len(vc.level_db(x, SR, 0.02)) and gain.max() > 2 and gain.min() < -2


def test_rider_and_leveller_even_out_the_sung_level_without_lifting_the_noise():
    x, marks = voice()
    spread = lambda y: np.std([rms_db(y, a + 0.1, b - 0.05) for k, a, b in marks if k == "v"])
    sib_re_sung = lambda y: (np.mean([rms_db(y, a + 0.02, b - 0.02) for k, a, b in marks if k == "s"])
                             - np.mean([rms_db(y, a + 0.1, b - 0.05) for k, a, b in marks if k == "v"]))
    rider, info = vc.process(x, SR, {"rider": vc.LEAD["rider"]})
    assert spread(rider) < 0.6 * spread(x) and info["rider"]["max_up_db"] > 3
    comp, info = vc.process(x, SR, {"comp1": vc.LEAD["comp1"]})
    assert spread(comp) < 0.75 * spread(x) and info["comp1"]["max_db"] > 3
    # a compressor that read the noise itself would lift the quiet sibilants against the notes
    plain, _ = vc.process(x, SR, {"comp1": dict(vc.LEAD["comp1"], follow_sung=False)})
    assert sib_re_sung(comp) < sib_re_sung(plain) - 1.0
    # less of both at the crisper settings
    assert spread(vc.process(x, SR, vc.lead_chain("smooth"))[0]) < spread(vc.process(x, SR, vc.lead_chain("crisp"))[0])


def test_detectors_tell_sibilants_from_sung_frames():
    x, marks = voice(sib=0.06)
    grid = lambda a, b: slice(int(a * 1000), int(b * 1000))
    tilt, noise = vc.tilt_share(x, SR), vc.noise_share(x, SR)
    for kind, a, b in marks:
        if kind == "s":
            assert tilt[grid(a + 0.035, b - 0.035)].min() > 0.9
        else:
            assert tilt[grid(a + 0.03, b - 0.03)].max() < 0.1 and noise[grid(a + 0.05, b - 0.05)].max() < 0.1
    # a sibilant with the voice still sounding under it: the periodicity detector, which listens under 1.5 kHz,
    # reads it as sung; the tilt detector finds it
    mid = slice(420, 440)
    assert vc.noise_share(voiced_s(), SR)[mid].mean() < 0.1 and vc.tilt_share(voiced_s(), SR, sustain=0.02)[mid].mean() > 0.9
    burst = np.concatenate([tone(0.3), hiss(0.012, 0.2), tone(0.3)]).astype(np.float32)
    assert vc.tilt_share(burst, SR, sustain=0.03)[300:312].max() < 0.5  # too short to be a fricative
    assert vc.tilt_share(burst, SR, sustain=0.0)[300:312].max() > 0.9


def test_lift_raises_sibilants_above_its_frequency_up_to_the_ceiling():
    x, marks = voice(levels=(0.2, 0.2, 0.2, 0.2), sib=0.012)
    lift = {"db": 3.0, "hf": 2500.0, "floor_db": -40.0, "width_db": 6.0, "sustain": 0.03, "ceiling_db": -5.0}
    y, info = vc.process(x, SR, {"lift": lift})
    sib = [(a, b) for k, a, b in marks if k == "s"]
    vow = [(a, b) for k, a, b in marks if k == "v"]
    up = np.mean([rms_db(y, a + 0.03, b - 0.03) - rms_db(x, a + 0.03, b - 0.03) for a, b in sib])
    same = np.mean([rms_db(y, a + 0.1, b - 0.05) - rms_db(x, a + 0.1, b - 0.05) for a, b in vow])
    assert up - same == pytest.approx(3.0, abs=0.4) and info["lift"]["max_db"] == pytest.approx(3.0, abs=0.1)
    loud, _ = voice(levels=(0.2, 0.2, 0.2, 0.2), sib=0.5)  # sibilants already over the ceiling: left alone
    y2, _ = vc.process(loud, SR, {"lift": lift})
    assert np.mean([rms_db(y2, a + 0.03, b - 0.03) - rms_db(loud, a + 0.03, b - 0.03) for a, b in sib]) == \
        pytest.approx(np.mean([rms_db(y2, a + 0.1, b - 0.05) - rms_db(loud, a + 0.1, b - 0.05) for a, b in vow]), abs=0.3)
    # a sibilant over a voiced tail: lifted with the tilt detector, missed with the periodicity detector
    rise = lambda det: (rms_db(vc.process(voiced_s(0.012), SR, {"lift": dict(lift, detector=det, sustain=0.02)})[0],
                               0.42, 0.44) - rms_db(voiced_s(0.012), 0.42, 0.44))
    assert rise("tilt") > rise("periodicity") + 1.0
    # the limiter form of the consonant leveller takes loud noise down and never lifts
    limit = {"cons": {"target_db": -10.0, "ratio": 2.0, "max_up_db": 0.0, "max_down_db": 6.0, "floor_db": -24.0,
                      "width_db": 6.0, "detector": "tilt", "sustain": 0.0}}
    y4, info = vc.process(loud, SR, limit)
    assert info["cons"]["max_up_db"] == 0.0 and info["cons"]["max_down_db"] > 1.0


def test_the_dial_sets_the_leads_chain_and_room():
    assert vc.lead_chain("smooth") == vc.LEAD and list(vc.DIAL_CHAINS) == list(vc.DIALS)
    b, c = vc.lead_chain("balanced"), vc.lead_chain("crisp")
    assert list(b) == ["eq", "deess", "rider", "comp1", "comp2", "cons", "lift", "breaths", "crest"]
    assert list(c) == ["eq", "deess", "rider", "comp1", "comp2", "cons", "trans", "lift", "cons2", "breaths", "crest"]
    assert c["comp1"] is None and c["cons"] is None and c["comp2"]["ratio"] == 2.0 and c["comp2"]["offset_db"] == -6.0
    assert b["comp1"]["ratio"] == 1.25 and b["rider"] == {"window": 0.12, "range_db": 5.0, "smooth": 0.04, "strength": 0.6}
    assert b["cons"]["max_up_db"] == 0.0 and "replace" not in b["cons"] and "replace" not in c["cons2"]
    assert b["lift"]["db"] == 2.0 and c["lift"]["db"] == 2.5 and c["lift"]["detector"] == "tilt"
    assert vc.lead_chain("no such setting") == c and vc.DEFAULT_DIAL == "crisp"
    assert [vc.REVERB_SCALE[d] * 0.25 for d in vc.DIALS] == pytest.approx([0.50, 0.25, 0.15])
    assert vc.dial_chain({"eq": None, "gate": {"x": 1}}, {"eq": [1], "crest": 9.0}) == {"eq": None, "crest": 9.0,
                                                                                        "gate": {"x": 1}}


def test_roles_and_their_presets():
    assert vc.ROLES == ("lead", "double", "harmony", "answer", "adlib", "crowd")
    assert vc.parse_role("Lead") == ("lead", None) and vc.parse_role(" ad-lib ") == ("adlib", None)
    assert vc.parse_role("double: Harmony L") == ("double", "Harmony L") and vc.parse_role("ad_lib") == ("adlib", None)
    assert vc.parse_role("") == ("", None) and vc.parse_role("shouting") == ("", None) and vc.parse_role(None) == ("", None)
    lead = vc.role_preset("lead", "smooth")
    assert lead["chain"] == vc.LEAD and lead["reverb_scale"] == 2.0 and lead["adt"] is None
    assert vc.role_preset("lead")["chain"] == vc.lead_chain("crisp") and vc.role_preset("lead")["reverb_scale"] == 0.6
    answer = vc.role_preset("answer", "balanced")
    assert answer["chain"] == vc.lead_chain("balanced") and answer["reverb_scale"] == 1.0 and answer["room"]
    assert vc.role_preset("harmony", "smooth")["chain"] == vc.role_preset("harmony", "crisp")["chain"] == vc.HARMONY
    assert vc.role_preset("double")["adt"] is None  # a double that sang its own take
    assert vc.role_preset("double", copy_of="lead", pan=-0.3)["adt"] == vc.DOUBLE
    assert vc.role_preset("double", copy_of="lead", pan=0.3)["adt"] == vc.DOUBLE_R
    assert vc.role_preset("double", copy_of="harmony")["adt"] == vc.HARMONY_COPY
    adlib = vc.role_preset("adlib")
    assert list(adlib["chain"]) == ["eq"] and adlib["duck_db"] == 2.0 and adlib["slap"]["time"] == 0.095
    crowd = vc.role_preset("crowd")
    assert crowd["chain"] is None and crowd["duck_db"] == 3.0 and crowd["room"]["decay"] == 0.38
    assert vc.role_preset("nothing") == {"role": "", "chain": None, "adt": None, "reverb_scale": 1.0}


def test_apply_role_for_every_role():
    x, marks = voice()
    lead_env = vc.sing_env(x, SR)
    assert lead_env.max() > 0.95 and lead_env[: int(0.1 * SR)].max() < 0.05 and len(lead_env) == len(x)
    for role in vc.ROLES:
        y, info = vc.apply_role(x, SR, role, "crisp", lead_env, copy_of="lead" if role == "double" else None)
        assert y.dtype == np.float32 and y.shape == x.shape and np.isfinite(y).all() and info["role"] == role
        assert ("chain" in info) == (role != "crowd") and ("adt" in info) == (role == "double")
        assert info["reverb_scale"] == (0.6 if role == "lead" else 1.0) and json.dumps(info)
    same, info = vc.apply_role(x, SR, "", "crisp")
    assert same is x or np.array_equal(same, x)
    assert vc.apply_role(np.zeros(100, np.float32), SR, "lead")[1] == {"role": "lead", "reverb_scale": 0.6}
    # an ad-lib is band-limited and steps back while the lead sings
    line = tone(1.0, 300.0, 0.2).astype(np.float32)
    quiet, _ = vc.apply_role(line, SR, "adlib", lead_env=np.ones(len(line), np.float32))
    free, _ = vc.apply_role(line, SR, "adlib", lead_env=np.zeros(len(line), np.float32))
    assert rms_db(free, 0.3, 0.6) - rms_db(quiet, 0.3, 0.6) == pytest.approx(2.0, abs=0.05)
    assert np.array_equal(free, vc.apply_role(line, SR, "adlib")[0])  # no lead in the mix: no duck
    # one repeat 95 ms after a hit, and a room after a crowd's shout
    hit = np.concatenate([tone(0.05, 500.0, 0.3), gap(0.6)]).astype(np.float32)
    rep, _ = vc.apply_role(hit, SR, "adlib")
    assert rms_db(rep, 0.1, 0.14) > rms_db(rep, 0.06, 0.09) + 20 and rms_db(rep, 0.25, 0.5) < -90
    shout, _ = vc.apply_role(hit, SR, "crowd")
    assert rms_db(shout, 0.12, 0.3) > -75


def test_a_double_made_from_the_take():
    x, marks = voice(levels=(0.2, 0.2, 0.2, 0.2))
    d = vc.adt(x, SR, **vc.DOUBLE)
    assert d.dtype == np.float32 and d.shape == x.shape
    # a few cents sharp: the delay shrinks while the voice sings
    a0 = int((marks[1][1] + 0.08) * SR)

    def peak(y, lo=2000.0, hi=2500.0, size=8192, pad=16):
        spec = np.abs(np.fft.rfft(np.asarray(y[a0:a0 + size], np.float64) * np.hanning(size), size * pad))
        f = np.fft.rfftfreq(size * pad, 1 / SR)
        band = (f >= lo) & (f <= hi)
        return f[band][np.argmax(spec[band])]

    sharp = vc.adt(x, SR, delay=0.018, depth=0.0, detune_cents=6.0)
    assert 1200 * np.log2(peak(sharp) / peak(x)) == pytest.approx(6.0, abs=1.5)
    plain = vc.adt(x, SR, delay=0.018, depth=0.0, detune_cents=0.0)
    assert np.allclose(plain[int(0.018 * SR) + 100:5000], x[100:5000 - int(0.018 * SR)], atol=1e-6)
    assert vc.adt(np.zeros(1000, np.float32), SR).shape == (1000,)
    dark = vc.adt(x, SR, delay=0.0, depth=0.0, detune_cents=0.0, dark=vc.DOUBLE["dark"])
    hf = lambda y: rms_db(mixing.eq(np.asarray(y, np.float64), [("hp", 6000, 0, 0.7071, 2)], SR), 0.3, 0.9)
    assert hf(dark) < hf(x) - 3
    ir = vc.room_ir(SR, decay=0.3, predelay=0.03)
    assert np.sum(ir ** 2) == pytest.approx(1.0, abs=1e-6) and np.abs(ir[: int(0.02 * SR)]).max() < 1e-6
    wet = vc.room(x, SR, decay=0.3, predelay=0.03, level_db=-12.0)
    assert wet.shape == x.shape and rms_db(wet, 0.5, 2.0) == pytest.approx(rms_db(x, 0.5, 2.0) - 12.0, abs=3.0)
    echo = vc.slap(x, SR, time=0.1, level_db=-6.0)
    assert echo.shape == x.shape and not np.array_equal(echo, x)


def test_a_noted_role_is_used_only_for_the_audio_it_was_noted_for():
    x, _ = voice()
    vc.note_role("t1", x, "lead", "balanced")
    hit = vc.noted_role("t1", x)
    assert hit["role"] == "lead" and hit["dial"] == "balanced" and hit["copy_of"] is None
    padded = np.zeros(len(x) + 5000)
    padded[: len(x)] = x  # what the mixer is handed: float64, longer
    assert vc.noted_role("t1", padded)["role"] == "lead"
    faded = padded.copy()
    faded[len(x) - 2000:len(x)] *= np.linspace(1, 0, 2000)  # a region's fade at the end leaves it valid
    assert vc.noted_role("t1", faded) is not None
    assert vc.noted_role("t1", padded * 0.999) is None and vc.noted_role("t1", padded[: len(x) // 4]) is None
    other, _ = voice(seed=5)
    assert vc.noted_role("t1", other) is None and vc.noted_role("t2", x) is None
    vc.note_role("t1", x, "")  # no role: forgotten
    assert vc.noted_role("t1", x) is None
    vc.note_role("t1", x, "juggler")
    assert vc.noted_role("t1", x) is None
    vc.note_role("t3", np.zeros(1000, np.float32), "lead")
    assert vc.noted_role("t3", np.zeros(1000)) is None  # silence proves nothing
    late = np.concatenate([np.zeros(3 * len(x), np.float32), x])  # a track that only sings at the end
    vc.note_role("t4", late, "crowd")
    assert vc.noted_role("t4", np.concatenate([late, np.zeros(100)]))["role"] == "crowd"
    assert vc.noted_role("t4", np.concatenate([np.zeros(3 * len(x)), other])) is None
    assert set(vc.noted_role("t4", late)) == {"role", "dial", "copy_of"}


def _mix(sources, **master):
    from synth2.project import MasterSettings

    return mixing.mix_sources(sources, SR, band_mode=False, settings=MasterSettings(**master), reverb_wet=1.0)


def test_the_mixer_gives_a_track_with_a_role_its_chain_and_leaves_the_others_alone():
    x, _ = voice()
    y, _ = voice(levels=(0.1, 0.2, 0.05, 0.25), seed=3)
    src = lambda key, audio, **kw: mixing.Source(key=key, name=key, kind="vocal", audio=np.asarray(audio, np.float64),
                                                 reverb_send=0.25, **kw)
    for key in ("a", "b"):
        vc.note_role(key, x, "")
    before = _mix([src("a", x), src("b", y)])
    assert "role" not in before.info["vocal_strips"]["a"]
    vc.note_role("a", x, "lead", "crisp")
    after = _mix([src("a", x), src("b", y)])
    strips = after.info["vocal_strips"]
    assert strips["a"]["role"] == "lead" and strips["a"]["dial"] == "crisp" and "rider" in strips["a"]["chain"]
    assert np.array_equal(after.stems["b"], before.stems["b"])  # a track without a role is mixed as before
    assert not np.array_equal(after.stems["a"], before.stems["a"])
    want, _ = vc.apply_role(x, SR, "lead", "crisp")
    assert np.allclose(after.stems["a"][:, 0], want * mixing.pan_gains(0.0)[0], atol=1e-6)
    # the lead's send follows the dial: twice the track's send at smooth, 0.6 of it at crisp
    solo = lambda dial: (vc.note_role("a", x, "lead", dial), _mix([src("a", x)]))[1]
    wet = lambda m: float(np.sqrt(np.mean(m.wet ** 2)))
    assert wet(solo("smooth")) / wet(solo("balanced")) == pytest.approx(2.0, rel=0.15)
    assert wet(solo("crisp")) / wet(solo("balanced")) == pytest.approx(0.6, rel=0.15)
    # switched off like the strip: per track, or for the whole mix
    vc.note_role("a", x, "lead", "crisp")
    off = _mix([src("a", x, overrides={"chain": False}), src("b", y)])
    assert "a" not in off.info.get("vocal_strips", {}) and np.allclose(off.stems["a"][:, 0], x * mixing.pan_gains(0.0)[0])
    assert "vocal_strips" not in _mix([src("a", x)], vocal_chain=False).info
    # other audio under the same track id (another engine sang it): the note is not used
    assert "role" not in _mix([src("a", y)]).info["vocal_strips"]["a"]
    # an ad-lib ducks under the lead of the same mix
    vc.note_role("a", x, "lead", "crisp")
    vc.note_role("b", y, "adlib")
    both = _mix([src("a", x), src("b", y)])
    assert both.info["vocal_strips"]["b"]["duck_db"] == 2.0
    vc.note_role("a", x, "")
    assert "duck_db" not in _mix([src("a", x), src("b", y)]).info["vocal_strips"]["b"]
    for key in ("a", "b"):
        vc.note_role(key, x, "")


def test_projects_and_scores_carry_the_role():
    text = ('tempo: 120\ntrack: Lead voice=nova role=lead\nC4 q la\n'
            'track: Double voice=nova pan=-0.3 role="double:Lead"\nC4 q la\ntrack: Old voice=aurora\nC4 q la\n')
    p = parse_score(text)
    assert [t.role for t in p.tracks] == ["lead", "double:Lead", ""]
    again = parse_score(project_to_score(p))
    assert [t.role for t in again.tracks] == ["lead", "double:Lead", ""]
    d = p.to_dict()
    assert d["tracks"][0]["role"] == "lead" and "role" not in d["tracks"][2]  # old projects keep their shape
    assert [t.role for t in Project.from_dict(d).tracks] == ["lead", "double:Lead", ""]
    assert Track().role == "" and Track.from_dict({"name": "x"}).role == ""
    assert "role=" not in project_to_score(parse_score("tempo: 120\ntrack: A\nC4 q la\n"))
    odd = parse_score("tempo: 120\ntrack: A role=juggler\nC4 q la\n")
    assert odd.tracks[0].role == "juggler"  # kept, with a warning; the engines ignore it
