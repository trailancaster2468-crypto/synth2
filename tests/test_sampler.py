"""Tests for the sample player (synth2.arranger.sampler) and its formats. Every fixture is
built here: tiny SoundFont 2 and SFZ files and WAV folders made of sines and noise."""
from __future__ import annotations

import struct

import numpy as np
import pytest

from synth2.arranger import sampler, samples, sf2
from synth2.arranger.sampler import Envelope, Region, SampleInstrument, render_note, render_region
from synth2.arranger.samples import SampleError

SR = 22050
PERIOD = 84  # samples per cycle of the fixture sines at their root key
F_ROOT = SR / PERIOD  # 262.5 Hz, the recorded pitch of key 60 in the fixtures


# ------------------------------------------------------------------------------------- helpers

def sine(n, period=PERIOD, amp=0.5, harmonics=()):
    t = np.arange(n) / period
    y = np.sin(2 * np.pi * t)
    for k, a in harmonics:
        y = y + a * np.sin(2 * np.pi * k * t)
    return (amp * y).astype(np.float32)


def freq_of(x, sr, lo=30.0, hi=8000.0):
    """Frequency of the strongest partial of a mono signal (parabolic peak of a Hann spectrum)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    spec = np.abs(np.fft.rfft(x * np.hanning(n), 8 * n))
    f = np.fft.rfftfreq(8 * n, 1 / sr)
    band = (f >= lo) & (f <= hi)
    i = int(np.argmax(np.where(band, spec, 0)))
    a, b, c = np.log(spec[i - 1] + 1e-12), np.log(spec[i] + 1e-12), np.log(spec[i + 1] + 1e-12)
    return f[i] + 0.5 * (a - c) / (a - 2 * b + c) * (f[1] - f[0])


def cents(f, ref):
    return 1200 * np.log2(f / ref)


def level_db(x, f, sr):
    """Level in dB of the partial at frequency f relative to full scale."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    t = np.arange(n) / sr
    w = np.hanning(n)
    return 20 * np.log10(abs(np.sum(x * w * np.exp(-2j * np.pi * f * t))) / (w.sum() / 2) + 1e-12)


def write_sf2(path, smpls, instruments, presets):
    """A minimal SoundFont 2 file.

    smpls: [{name, data (mono float), rate, root, loop (start, end)}]
    instruments: [{name, zones: [{gens: {oper: amount | (lo, hi)}, sample: i, mods: [...]}], global: {...}}]
    presets: [{name, bank, program, zones: [{gens: {...}, instrument: i}]}]
    """
    def pad20(s):
        return s.encode("latin-1")[:19].ljust(20, b"\0")

    def gen_records(gens, last):
        order = sorted(gens.items(), key=lambda kv: (kv[0] != 43, kv[0] != 44, kv[0]))
        out = b""
        for oper, amount in order + [last]:
            if isinstance(amount, tuple):
                out += struct.pack("<HBB", oper, amount[0], amount[1])
            else:
                out += struct.pack("<Hh", oper, amount)
        return out, len(order) + 1

    smpl, shdr, pos = b"", b"", 0
    for s in smpls:
        pcm = np.round(np.clip(np.asarray(s["data"], np.float64), -1, 1) * 32767).astype("<i2")
        smpl += pcm.tobytes() + b"\0" * 92
        loop = s.get("loop", (0, len(pcm)))
        shdr += pad20(s["name"]) + struct.pack("<IIIIIBbHH", pos, pos + len(pcm), pos + loop[0], pos + loop[1],
                                               s.get("rate", SR), s.get("root", 60), s.get("correction", 0), 0,
                                               s.get("type", 1))
        pos += len(pcm) + 46
    shdr += pad20("EOS") + b"\0" * 26

    def hydra(items, zone_key, terminal_oper, name_fmt):
        hdr, bag, gen, mod = b"", b"", b"", b""
        n_bag = n_gen = n_mod = 0
        for it in items:
            hdr += name_fmt(it, n_bag)
            zones = it["zones"]
            if it.get("global"):
                recs = b"".join(struct.pack("<Hh", o, a) if not isinstance(a, tuple) else struct.pack("<HBB", o, *a)
                                for o, a in it["global"].items())
                bag += struct.pack("<HH", n_gen, n_mod)
                gen += recs
                n_gen += len(it["global"])
                n_bag += 1
            for z in zones:
                bag += struct.pack("<HH", n_gen, n_mod)
                recs, k = gen_records(z.get("gens", {}), (terminal_oper, z[zone_key]))
                gen += recs
                n_gen += k
                for m in z.get("mods", []):
                    mod += struct.pack("<HHhHH", *m)
                    n_mod += 1
                n_bag += 1
        hdr += name_fmt(None, n_bag)
        bag += struct.pack("<HH", n_gen, n_mod)
        gen += struct.pack("<Hh", 0, 0)
        mod += b"\0" * 10
        return hdr, bag, mod, gen

    inst, ibag, imod, igen = hydra(
        instruments, "sample", 53, lambda it, b: pad20(it["name"] if it else "EOI") + struct.pack("<H", b))
    phdr, pbag, pmod, pgen = hydra(
        presets, "instrument", 41,
        lambda it, b: pad20(it["name"] if it else "EOP") + struct.pack("<HHHIII", it["program"] if it else 0,
                                                                       it["bank"] if it else 0, b, 0, 0, 0))

    def chunk(cid, body):
        return cid + struct.pack("<I", len(body)) + body + (b"\0" if len(body) & 1 else b"")

    info = b"INFO" + chunk(b"ifil", struct.pack("<HH", 2, 1)) + chunk(b"isng", b"EMU8000\0") + chunk(b"INAM", b"test\0\0")
    sdta = b"sdta" + chunk(b"smpl", smpl)
    pdta = b"pdta" + b"".join(chunk(c, b) for c, b in ((b"phdr", phdr), (b"pbag", pbag), (b"pmod", pmod),
                                                       (b"pgen", pgen), (b"inst", inst), (b"ibag", ibag),
                                                       (b"imod", imod), (b"igen", igen), (b"shdr", shdr)))
    body = b"sfbk" + chunk(b"LIST", info) + chunk(b"LIST", sdta) + chunk(b"LIST", pdta)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def timecents(seconds):
    return int(round(1200 * np.log2(seconds)))


@pytest.fixture(scope="module")
def font(tmp_path_factory):
    """A SoundFont with a looped sine multisample (two roots, two velocity layers) and a
    three-piece kit with a choked open hat."""
    rng = np.random.default_rng(3)
    n = PERIOD * 60
    soft = sine(n, amp=0.25)
    loud = sine(n, amp=0.5, harmonics=((3, 0.4),))
    high = sine(n, period=PERIOD // 2, amp=0.4)  # one octave up, root 72
    t = np.arange(SR // 2) / SR
    kick = (0.9 * np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.08)).astype(np.float32)
    snare = (0.6 * rng.standard_normal(len(t)) * np.exp(-t / 0.05)).astype(np.float32)
    hat = (0.4 * rng.standard_normal(SR // 10) * np.exp(-np.arange(SR // 10) / SR / 0.01)).astype(np.float32)
    ohat = (0.4 * rng.standard_normal(SR) * np.exp(-np.arange(SR) / SR / 0.4)).astype(np.float32)
    loop = (PERIOD * 20, PERIOD * 50)
    smpls = [{"name": "soft", "data": soft, "loop": loop, "root": 60},
             {"name": "loud", "data": loud, "loop": loop, "root": 60},
             {"name": "high", "data": high, "loop": loop, "root": 72},
             {"name": "kick", "data": kick}, {"name": "snare", "data": snare}, {"name": "hat", "data": hat},
             {"name": "ohat", "data": ohat}]
    env = {34: timecents(0.05), 38: timecents(0.5), 54: 1}
    instruments = [
        {"name": "sines", "global": env, "zones": [
            {"gens": {43: (0, 65), 44: (0, 79)}, "sample": 0},
            {"gens": {43: (0, 65), 44: (80, 127)}, "sample": 1},
            {"gens": {43: (66, 127)}, "sample": 2}]},
        {"name": "kit", "zones": [
            {"gens": {43: (36, 36), 58: 36}, "sample": 3},
            {"gens": {43: (38, 38), 58: 38, 17: 100}, "sample": 4},
            {"gens": {43: (42, 42), 58: 42, 57: 1}, "sample": 5},
            {"gens": {43: (46, 46), 58: 46, 57: 1}, "sample": 6}]},
    ]
    presets = [{"name": "Sine Piano", "bank": 0, "program": 0, "zones": [{"instrument": 0}]},
               {"name": "Sine Bass", "bank": 0, "program": 33, "zones": [{"gens": {51: -12}, "instrument": 0}]},
               {"name": "Test Kit", "bank": 128, "program": 0, "zones": [{"instrument": 1}]}]
    path = tmp_path_factory.mktemp("sf2") / "test.sf2"
    write_sf2(path, smpls, instruments, presets)
    return sf2.load_sf2(str(path))


# ---------------------------------------------------------------------------------- envelopes

def test_envelope_shapes_and_end():
    t = np.linspace(0, 2, 2001)
    e = Envelope(delay=0.1, attack=0.2, hold=0.1, decay=0.4, sustain=0.5, release=0.3, shape="exp")
    y = e.at(t, 1.0)
    assert y[50] == 0 and abs(y[200] - 0.5) < 1e-6 and abs(y[300] - 1) < 1e-9 and abs(y[400] - 1) < 1e-9
    assert abs(y[1000] - 0.5) < 0.01 and y[1299] < 0.5 * np.exp(-8.9) * 1.1
    assert np.all(np.diff(y[400:]) <= 1e-12)
    assert abs(e.end(1.0) - 1.3) < 1e-9 and e.end(None) is None
    d = Envelope(attack=0.0, decay=1.0, sustain=0.1, release=0.5, shape="db")
    y = d.at(t, 1.5)
    assert abs(20 * np.log10(y[100]) + 10) < 0.01  # 100 dB per decay time
    assert abs(y[1000] - 0.1) < 1e-9  # the sustain level holds
    assert abs(20 * np.log10(y[1600]) - (-20 - 20)) < 0.05  # release: 100 dB per release time
    assert abs(d.end(1.5) - (1.5 + 0.5 * 0.7)) < 1e-6
    pluck = Envelope(decay=2.0, sustain=0.0, shape="db")
    assert abs(pluck.end(None) - 1.8) < 1e-9 and pluck.end(0.5) < 0.6


# --------------------------------------------------------------------------------- the player

def loop_region(**kw):
    smp = samples.from_array(sine(PERIOD * 60), SR, name="sine")
    base = dict(sample=smp, root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20, loop_end=PERIOD * 50,
                ampeg=Envelope(attack=0.01, release=0.05))
    base.update(kw)
    return Region(**base)


@pytest.mark.parametrize("key", [36, 48, 59, 60, 61, 67, 72, 84, 96])
def test_pitch_within_a_cent(key):
    y = render_region(loop_region(), key, 100, 1.0, 44100)
    seg = y[4410:44100, 0]
    f = freq_of(seg, 44100)
    assert abs(cents(f, F_ROOT * 2 ** ((key - 60) / 12))) < 1.0


def test_tuning_keytrack_and_pitch_argument():
    f = freq_of(render_region(loop_region(tune=50.0), 60, 100, 1.0, SR)[2205:, 0], SR)
    assert abs(cents(f, F_ROOT) - 50) < 1.0
    f = freq_of(render_region(loop_region(keytrack=0.0), 72, 100, 1.0, SR)[2205:, 0], SR)
    assert abs(cents(f, F_ROOT)) < 1.0
    f = freq_of(render_region(loop_region(), 60, 100, 1.0, SR, pitch=60.25)[2205:, 0], SR)
    assert abs(cents(f, F_ROOT) - 25) < 1.0


def test_loop_is_continuous_and_clean():
    for key in (60, 53, 67, 72):
        y = render_region(loop_region(), key, 127, 3.0, SR)[:, 0]
        f0 = F_ROOT * 2 ** ((key - 60) / 12)
        body = y[SR // 2:3 * SR - 100].astype(np.float64)
        # no jump larger than the sine's own steepest step, and nothing but the sine in the spectrum
        assert np.abs(np.diff(body)).max() <= 1.03 * np.abs(body).max() * 2 * np.pi * f0 / SR
        seg = body[:2 * SR]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
        fr = np.fft.rfftfreq(len(seg), 1 / SR)
        rest = spec[np.abs(fr - f0) > 40].max()
        assert 20 * np.log10(rest / spec.max()) < -70, key


def test_loop_sustain_plays_the_tail_after_release():
    smp_data = np.concatenate([sine(PERIOD * 50), sine(PERIOD * 40, period=PERIOD // 2)])  # tail one octave up
    smp = samples.from_array(smp_data, SR)
    reg = Region(sample=smp, root=60, loop_mode="loop_sustain", loop_start=PERIOD * 20, loop_end=PERIOD * 50,
                 ampeg=Envelope(release=5.0, shape="db"))
    y = render_region(reg, 60, 100, 1.0, SR)[:, 0]
    assert abs(cents(freq_of(y[2000:20000], SR), F_ROOT)) < 1
    tail = y[SR + PERIOD * 31:SR + PERIOD * 60]
    assert abs(cents(freq_of(tail, SR), 2 * F_ROOT)) < 3
    assert len(y) <= SR + PERIOD * 70 + 8  # ends with the sample
    held = render_region(Region(sample=smp, root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20,
                                loop_end=PERIOD * 50, ampeg=Envelope(release=0.2)), 60, 100, 2.0, SR)
    assert abs(len(held) - int(2.2 * SR)) < 16


def test_no_aliasing_when_transposed_up():
    # a bright sample two octaves up: partials that would fold back are removed, the rest are exact
    harm = tuple((k, 1.0 / k) for k in range(2, 40))
    smp = samples.from_array(sine(PERIOD * 80, amp=0.2, harmonics=harm), SR)
    reg = Region(sample=smp, root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20, loop_end=PERIOD * 60)
    y = render_region(reg, 84, 127, 1.5, SR)[SR // 4:SR + SR // 4, 0].astype(np.float64)
    spec = np.abs(np.fft.rfft(y * np.hanning(len(y))))
    fr = np.fft.rfftfreq(len(y), 1 / SR)
    f0 = 4 * F_ROOT
    near = np.min(np.abs(fr[:, None] - f0 * np.arange(1, 12)[None, :]), axis=1) < 25
    assert 20 * np.log10(spec[~near].max() / spec.max()) < -60
    assert abs(cents(freq_of(y, SR), f0)) < 1


def test_interpolation_error_is_small_when_transposed_down():
    y = render_region(loop_region(), 55, 127, 1.0, 44100)[8820:, 0].astype(np.float64)
    f0 = F_ROOT * 2 ** (-5 / 12)
    seg = y[:22050]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1 / 44100)
    assert 20 * np.log10(spec[np.abs(fr - f0) > 40].max() / spec.max()) < -70


def test_sample_rate_conversion_keeps_pitch():
    smp = samples.from_array(sine(PERIOD * 60), 48000)
    reg = Region(sample=smp, root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20, loop_end=PERIOD * 50)
    f = freq_of(render_region(reg, 60, 100, 1.0, 44100)[4410:, 0], 44100)
    assert abs(cents(f, 48000 / PERIOD)) < 1


def test_unison_one_shot_is_a_copy():
    rng = np.random.default_rng(1)
    data = (0.3 * rng.standard_normal(5000)).astype(np.float32)
    reg = Region(sample=samples.from_array(data, SR), root=60, loop_mode="one_shot", amp_veltrack=0.0)
    y = render_region(reg, 60, 100, 0.01, SR)
    assert len(y) == 5000
    assert np.allclose(y[:-30, 0], data[:-30], atol=1e-6) and np.array_equal(y[:, 0], y[:, 1])


def test_envelope_times_in_audio():
    reg = loop_region(ampeg=Envelope(delay=0.1, attack=0.2, hold=0.0, decay=0.0, sustain=1.0, release=0.4, shape="db"))
    y = np.abs(render_region(reg, 60, 127, 1.0, SR)[:, 0])
    peak = lambda a, b: float(y[int(a * SR):int(b * SR)].max())  # noqa: E731
    assert peak(0.0, 0.095) < 1e-4
    assert abs(peak(0.19, 0.21) / peak(0.5, 0.9) - 0.5) < 0.05  # half way up the linear attack
    assert abs(peak(0.29, 0.32) / peak(0.5, 0.9) - 1.0) < 0.03
    assert abs(20 * np.log10(peak(1.19, 1.21) / peak(0.5, 0.9)) + 50) < 2.5  # 100 dB per 0.4 s
    assert len(y) < int((1.0 + 0.4 * 0.9 + 0.01) * SR)


def test_note_off_and_cut_do_not_click():
    reg = loop_region(ampeg=Envelope(attack=0.0, release=0.0))
    for kw in ({}, {"cut": 0.3}):
        y = render_region(reg, 60, 127, 0.5, SR, **kw)[:, 0].astype(np.float64)
        step = np.abs(np.diff(np.concatenate([y, [0.0]])))
        sine_step = 0.5 * 2 * np.pi * F_ROOT / SR
        assert step[100:].max() < 2.5 * sine_step
    assert len(render_region(reg, 60, 127, 0.5, SR, cut=0.3)) <= int((0.3 + sampler.OFF_TIME) * SR) + 2


def test_velocity_curve_and_tracking():
    reg = loop_region()
    loud = np.abs(render_region(reg, 60, 127, 0.5, SR)).max()
    half = np.abs(render_region(reg, 60, 63.5, 0.5, SR)).max()
    assert abs(half / loud - 0.25) < 0.01
    flat = loop_region(amp_veltrack=0.0)
    assert abs(np.abs(render_region(flat, 60, 20, 0.5, SR)).max() / loud - 1) < 0.01
    lin = loop_region(vel_curve=1.0)
    assert abs(np.abs(render_region(lin, 60, 63.5, 0.5, SR)).max() / loud - 0.5) < 0.01


def test_pan_and_stereo_samples():
    left = render_region(loop_region(pan=-1.0), 60, 127, 0.3, SR)
    assert np.abs(left[:, 1]).max() < 1e-6 < np.abs(left[:, 0]).max()
    centre = render_region(loop_region(), 60, 127, 0.3, SR)
    assert np.allclose(centre[:, 0], centre[:, 1])
    st = np.stack([sine(PERIOD * 60), 0.5 * sine(PERIOD * 60, period=PERIOD // 2)], axis=1)
    reg = Region(sample=samples.from_array(st, SR), root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20,
                 loop_end=PERIOD * 50)
    y = render_region(reg, 62, 127, 1.0, SR)
    assert abs(cents(freq_of(y[2205:, 0], SR), F_ROOT * 2 ** (2 / 12))) < 1
    assert abs(cents(freq_of(y[2205:, 1], SR), 2 * F_ROOT * 2 ** (2 / 12))) < 1


def test_filter_cutoff_resonance_key_and_velocity_tracking():
    harm = tuple((k, 1.0) for k in range(2, 20))
    smp = samples.from_array(sine(PERIOD * 80, amp=0.04, harmonics=harm), SR)
    kw = dict(sample=smp, root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20, loop_end=PERIOD * 60)
    f1, f8 = F_ROOT, 8 * F_ROOT

    def tilt(reg, key=60, vel=127):
        y = render_region(reg, key, vel, 1.0, SR)[SR // 4:SR, 0]
        k = 2 ** ((key - 60) / 12)
        return level_db(y, f8 * k, SR) - level_db(y, f1 * k, SR)

    flat = tilt(Region(**kw))
    assert abs(flat) < 0.5
    lp = tilt(Region(cutoff=4 * F_ROOT, **kw))
    assert abs(lp - flat + 11.4) < 1.0  # two-pole low-pass one octave above its cutoff
    assert tilt(Region(cutoff=4 * F_ROOT, fil_type="lpf_1p", **kw)) > lp + 4
    assert tilt(Region(cutoff=4 * F_ROOT, fil_type="lpf_4p", **kw)) < lp - 8
    assert tilt(Region(cutoff=4 * F_ROOT, fil_type="hpf_2p", **kw)) > 10
    y = render_region(Region(cutoff=4 * F_ROOT, resonance=12.0, **kw), 60, 127, 1.0, SR)[SR // 4:SR, 0]
    y0 = render_region(Region(**kw), 60, 127, 1.0, SR)[SR // 4:SR, 0]
    assert abs(level_db(y, 4 * F_ROOT, SR) - level_db(y0, 4 * F_ROOT, SR) - 12) < 1.0
    # full key tracking: the same tilt an octave up; velocity opens the filter
    tracked = Region(cutoff=4 * F_ROOT, fil_keytrack=100.0, **kw)
    assert abs(tilt(tracked, key=72) - lp) < 1.5
    vel = Region(cutoff=F_ROOT, fil_veltrack=2400.0, **kw)
    assert tilt(vel, vel=127) > tilt(vel, vel=20) + 15


def test_filter_envelope_opens_and_closes():
    harm = tuple((k, 1.0) for k in range(2, 20))
    smp = samples.from_array(sine(PERIOD * 80, amp=0.04, harmonics=harm), SR)
    reg = Region(sample=smp, root=60, loop_mode="loop_continuous", loop_start=PERIOD * 20, loop_end=PERIOD * 60,
                 cutoff=F_ROOT, fileg=Envelope(attack=0.0, decay=0.5, sustain=0.0, shape="lin"), fileg_depth=4800.0)
    y = render_region(reg, 60, 127, 1.5, SR)[:, 0]
    early, late = y[int(0.02 * SR):int(0.12 * SR)], y[int(1.0 * SR):int(1.4 * SR)]
    assert level_db(early, 8 * F_ROOT, SR) - level_db(early, F_ROOT, SR) > -6
    assert level_db(late, 8 * F_ROOT, SR) - level_db(late, F_ROOT, SR) < -30


def test_vibrato_and_bend():
    reg = loop_region(vib_depth=50.0, vib_freq=5.0)
    y = render_region(reg, 60, 127, 2.0, SR)[:, 0]
    fs = [freq_of(y[a:a + 1024], SR) for a in range(4410, 40000, 256)]
    dev = np.array([cents(f, F_ROOT) for f in fs])
    assert 35 < dev.max() < 60 and -60 < dev.min() < -35
    bend = np.linspace(0.0, 2.0, SR)  # two semitones up over a second, then held
    y = render_region(loop_region(), 60, 127, 2.0, SR, bend=bend)[:, 0]
    assert abs(cents(freq_of(y[int(1.2 * SR):int(1.9 * SR)], SR), F_ROOT) - 200) < 1.5
    assert abs(cents(freq_of(y[int(0.45 * SR):int(0.55 * SR)], SR), F_ROOT) - 100) < 12


def kit_instrument():
    rng = np.random.default_rng(5)
    shots = [samples.from_array(0.3 * rng.standard_normal(4000).astype(np.float32), SR, name=f"s{i}") for i in range(3)]
    regions = [Region(sample=s, lokey=38, hikey=38, root=38, loop_mode="one_shot", seq_length=3, seq_position=i + 1)
               for i, s in enumerate(shots)]
    regions += [Region(sample=shots[0], lokey=40, hikey=40, root=40, loop_mode="one_shot", lorand=0.0, hirand=0.5),
                Region(sample=shots[1], lokey=40, hikey=40, root=40, loop_mode="one_shot", lorand=0.5, hirand=1.0)]
    return SampleInstrument("kit", regions, drums=True), shots


def test_round_robin_and_random_layers_are_deterministic():
    inst, shots = kit_instrument()
    picks = [inst.select(38, 100, seq=i)[0].sample.name for i in range(7)]
    assert picks == ["s0", "s1", "s2", "s0", "s1", "s2", "s0"]
    a = [render_note(inst, 38, 100, 0.1, SR, seq=i) for i in range(3)]
    b = [render_note(inst, 38, 100, 0.1, SR, seq=i) for i in range(3)]
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], a[1]) and not np.array_equal(a[1], a[2])
    assert inst.select(40, 100, rand=0.2)[0].sample.name == "s0"
    assert inst.select(40, 100, rand=0.7)[0].sample.name == "s1"
    assert len(inst.select(40, 100, rand=0.999)) == 1


def test_release_sample_plays_at_note_off():
    rng = np.random.default_rng(2)
    click = samples.from_array(0.5 * rng.standard_normal(800).astype(np.float32), SR)
    inst = SampleInstrument("x", [loop_region(ampeg=Envelope(release=0.02)),
                                  Region(sample=click, root=60, trigger="release", amp_veltrack=0.0, rt_decay=6.0)])
    y = render_note(inst, 60, 127, 0.5, SR)[:, 0]
    assert np.abs(y[int(0.5 * SR) + 10:int(0.5 * SR) + 700]).max() > 0.3
    quiet = render_note(inst, 60, 127, 2.0, SR)[:, 0]
    a, b = y[int(0.5 * SR) + 600:int(0.5 * SR) + 790], quiet[int(2.0 * SR) + 600:int(2.0 * SR) + 790]
    assert abs(20 * np.log10(np.abs(b).max() / np.abs(a).max()) + 9) < 1.0  # 6 dB per second held


def test_prepared_audio_cache_is_bounded(monkeypatch):
    sampler.clear_cache()
    monkeypatch.setitem(sampler.QUALITY, "cache_mb", 1)
    for i in range(12):
        smp = samples.from_array(sine(60000), SR, name=f"big{i}")
        render_region(Region(sample=smp, root=60), 62, 100, 0.2, SR)
    assert sampler.cache_bytes() <= (1 << 20) + 60000 * 2 * 4 + 64
    sampler.clear_cache()
    assert sampler.cache_bytes() == 0


# ----------------------------------------------------------------------------------------- WAV

@pytest.mark.parametrize("bits,flt", [(8, False), (16, False), (24, False), (32, False), (32, True), (64, True)])
def test_wav_formats_round_trip(tmp_path, bits, flt):
    rng = np.random.default_rng(bits)
    data = np.clip(0.3 * rng.standard_normal((3000, 2)), -1, 1)
    path = tmp_path / "x.wav"
    samples.write_wav(str(path), data, 32000, bits=bits, float_format=flt, loop=(100, 2000), root=57)
    smp = samples.open_sample(str(path))
    assert (smp.rate, smp.frames, smp.channels, smp.loop, smp.root) == (32000, 3000, 2, (100, 2000), 57)
    tol = {8: 0.02, 16: 1e-4}.get(bits, 1e-6)
    assert np.abs(smp.all() - data).max() < tol
    assert np.abs(smp.read(1000, 1010) - data[1000:1010]).max() < tol
    assert smp.read(2990, 3010).shape == (20, 2) and not smp.read(2990, 3010)[10:].any()


def test_wav_errors_are_plain(tmp_path):
    bad = tmp_path / "bad.wav"
    bad.write_bytes(b"RIFF\x10\0\0\0WAVEjunk")
    with pytest.raises(SampleError):
        samples.open_sample(str(bad))
    with pytest.raises(SampleError, match="not found"):
        samples.open_sample(str(tmp_path / "missing.wav"))
    adpcm = tmp_path / "adpcm.wav"
    body = b"WAVE" + b"fmt " + struct.pack("<IHHIIHH", 16, 2, 1, 22050, 11025, 256, 4) + b"data" + struct.pack("<I", 4) + b"\0" * 4
    adpcm.write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)
    with pytest.raises(SampleError, match="compressed"):
        samples.open_sample(str(adpcm))
    with pytest.raises(SampleError, match="unsupported sample format"):
        (tmp_path / "x.mp3").write_bytes(b"0")
        samples.open_sample(str(tmp_path / "x.mp3"))


def test_names_in_file_names():
    assert samples.note_from_name("Piano_C#4.wav") == 61 and samples.note_from_name("bass-Eb1_v3.wav") == 27
    assert samples.note_from_name("A3.wav") == 57 and samples.note_from_name("epiano_060.wav") == 60
    assert samples.note_from_name("Gs2 loud.wav") == 44 and samples.note_from_name("kick.wav") is None
    assert samples.layer_from_name("snare_v3_rr2.wav") == (3, 2)
    assert samples.layer_from_name("C4_mf.wav") == (80, None) and samples.layer_from_name("kick-vel100.wav")[0] == 100


# ----------------------------------------------------------------------------------------- SF2

def test_sf2_presets_and_lookup(font):
    assert [(p.bank, p.program, p.name) for p in font.presets] == [
        (0, 0, "Sine Piano"), (0, 33, "Sine Bass"), (128, 0, "Test Kit")]
    assert font.find("0:33").name == "Sine Bass" and font.find((128, 0)).name == "Test Kit"
    assert font.find(33).name == "Sine Bass" and font.find("sine bass").name == "Sine Bass"
    assert font.find("bass").name == "Sine Bass" and font.find("nothing") is None
    with pytest.raises(SampleError):
        font.instrument("nothing")
    assert font.instrument("Test Kit").drums and not font.instrument(0).drums


def test_sf2_pitch_loop_and_preset_tuning(font):
    piano = font.instrument("Sine Piano")
    for key in (48, 60, 64, 72, 79):
        y = render_note(piano, key, 100, 1.5, 44100)
        assert abs(cents(freq_of(y[8820:44100, 0], 44100), F_ROOT * 2 ** ((key - 60) / 12))) < 1.5, key
    y = render_note(piano, 60, 60, 3.0, SR)[SR // 2:3 * SR - 100, 0].astype(np.float64)
    assert np.abs(np.diff(y)).max() <= 1.05 * np.abs(y).max() * 2 * np.pi * F_ROOT / SR
    bass = render_note(font.instrument("0:33"), 60, 100, 1.0, SR)  # coarse tune -12 at the preset level
    assert abs(cents(freq_of(bass[4410:, 0], SR), F_ROOT / 2)) < 1.5


def test_sf2_velocity_layers(font):
    piano = font.instrument(0)
    soft = render_note(piano, 60, 60, 1.0, SR)[SR // 4:SR, 0]
    loud = render_note(piano, 60, 110, 1.0, SR)[SR // 4:SR, 0]
    third = lambda y: level_db(y, 3 * F_ROOT, SR) - level_db(y, F_ROOT, SR)  # noqa: E731
    assert third(soft) < -50 and abs(third(loud) - 20 * np.log10(0.4)) < 1.0
    assert [r.sample.name for r in piano.select(60, 79)] == ["soft"]
    assert [r.sample.name for r in piano.select(60, 80)] == ["loud"]
    assert [r.sample.name for r in piano.select(70, 80)] == ["high"]


def test_sf2_envelope_from_generators(font):
    y = np.abs(render_note(font.instrument(0), 60, 127, 1.0, SR)[:, 0])
    peak = lambda a, b: float(y[int(a * SR):int(b * SR)].max())  # noqa: E731
    assert abs(peak(0.02, 0.03) / peak(0.3, 0.9) - 0.5) < 0.1  # attack 0.05 s
    assert abs(20 * np.log10(peak(1.095, 1.105) / peak(0.3, 0.9)) + 20) < 2.0  # release: 100 dB in 0.5 s


def test_sf2_kit_pan_and_exclusive_class(font):
    kit = font.instrument((128, 0))
    assert kit.keys() == [36, 38, 42, 46]
    snare = render_note(kit, 38, 100, 0.3, SR)
    assert np.abs(snare[:, 1]).max() > 1.2 * np.abs(snare[:, 0]).max()  # pan 100 of 500 to the right
    groups, off_by = sampler.note_groups(kit.select(46, 100))
    assert groups == {1} and off_by == {1} and sampler.note_groups(kit.select(36, 100)) == (set(), set())
    full = render_note(kit, 46, 100, 0.9, SR)
    choked = render_note(kit, 46, 100, 0.9, SR, cut=0.1)
    assert len(choked) <= int((0.1 + 0.01) * SR) + 2 < len(full)
    assert np.array_equal(choked[:int(0.1 * SR)], full[:int(0.1 * SR)])


def test_sf2_velocity_to_filter_modulator(tmp_path):
    harm = tuple((k, 1.0) for k in range(2, 20))
    data = sine(PERIOD * 80, amp=0.04, harmonics=harm)
    smpls = [{"name": "bright", "data": data, "loop": (PERIOD * 20, PERIOD * 60)}]
    zone = {"gens": {54: 1, 8: 9600}, "sample": 0}
    mod = (0x0102, 8, -3600, 0, 0)  # velocity (negative, linear) to filter cutoff
    write_sf2(tmp_path / "plain.sf2", smpls, [{"name": "i", "zones": [zone]}],
              [{"name": "p", "bank": 0, "program": 0, "zones": [{"instrument": 0}]}])
    write_sf2(tmp_path / "mod.sf2", smpls, [{"name": "i", "zones": [dict(zone, mods=[mod])]}],
              [{"name": "p", "bank": 0, "program": 0, "zones": [{"instrument": 0}]}])

    def tilt(path, vel):
        y = render_note(sf2.load_sf2(str(path)).instrument(0), 60, vel, 1.0, SR)[SR // 4:SR, 0]
        return level_db(y, 8 * F_ROOT, SR) - level_db(y, F_ROOT, SR)

    assert abs(tilt(tmp_path / "plain.sf2", 127) - tilt(tmp_path / "plain.sf2", 30)) < 0.5
    assert tilt(tmp_path / "mod.sf2", 127) > tilt(tmp_path / "mod.sf2", 30) + 12
    assert abs(tilt(tmp_path / "mod.sf2", 127) - tilt(tmp_path / "plain.sf2", 127)) < 0.5


def test_sf2_errors(tmp_path):
    junk = tmp_path / "junk.sf2"
    junk.write_bytes(b"RIFF\x04\0\0\0WAVE")
    with pytest.raises(SampleError, match="not a SoundFont"):
        sf2.load_sf2(str(junk))
    sf3 = tmp_path / "set.sf3"
    sf3.write_bytes(b"RIFF\x04\0\0\0sfbk")
    with pytest.raises(SampleError, match="sf3"):
        sf2.load_sf2(str(sf3))
    with pytest.raises(SampleError):
        sf2.load_sf2(str(tmp_path / "missing.sf2"))


# ------------------------------------------------------------------------- in an arrangement

import copy  # noqa: E402
import json  # noqa: E402

from synth2 import arranger  # noqa: E402
from synth2.arranger import render_backing, soundbank, validate_arrangement  # noqa: E402
from synth2.project import Project  # noqa: E402

BPM = 120.0
QUIET = {"timing_ms": 0, "velocity": 0}


def band(instruments, patterns, parts, bars=2, **top):
    for spec in instruments.values():
        spec.setdefault("humanize", dict(QUIET))
    arr = {"format": "synth2-arrangement", "version": 1, "key": "A minor", "energy_response": 0,
           "instruments": instruments, "patterns": patterns,
           "sections": [{"name": "A", "start_bar": 1, "bars": bars, "chords": ["Am"], "energy": 0.8, "parts": parts}]}
    arr.update(top)
    return arr


def project(arr):
    p = Project.new("Sampler test", bpm=BPM)
    p.key = "A minor"
    p.backing = copy.deepcopy(arr)
    return p


def stems_of(arr, sr=SR, t1=4.5):
    return {st.name: st for st in render_backing(project(arr), sr, 0.0, t1)}


LINE = {"line": {"kind": "notes", "score": "A3 h E4 h"}}
BEAT = {"beat": {"kind": "drums", "lanes": {"kick": "x.......x.......", "snare": "....x.......x...",
                                            "hat": "x.x.x.x.x.x.x.x.", "ohat": "..............x."}}}


def test_instrument_plays_a_soundfont_preset(font):
    arr = band({"keys": {"preset": "piano", "sound": {"path": font.path, "preset": "Sine Piano"}}}, LINE, {"keys": "line"})
    assert validate_arrangement(arr) == []
    st = stems_of(arr)["keys"]
    assert st.sound.endswith("0:0 Sine Piano") and st.warnings == () and st.role == "keys"
    a3 = st.audio[int(0.2 * SR):int(0.9 * SR), 0]
    e4 = st.audio[int(1.2 * SR):int(1.9 * SR), 0]
    assert abs(cents(freq_of(a3, SR), F_ROOT * 2 ** (-3 / 12))) < 2
    assert abs(cents(freq_of(e4, SR), F_ROOT * 2 ** (4 / 12))) < 2
    # level-matched to the synthesised piano it replaces
    synth = stems_of(band({"keys": {"preset": "piano"}}, LINE, {"keys": "line"}))["keys"]
    ratio = sampler.loudness(st.audio, SR) / sampler.loudness(synth.audio, SR)
    assert 0.5 < ratio < 2.0


def test_general_midi_table_picks_the_program_and_presets_by_name(font):
    # bass_pick asks for program 34, then 33: the font has 33
    res = soundbank.resolve({"path": font.path}, synth_preset="bass_pick", low=28)
    assert res.label.endswith("0:33 Sine Bass") and res.error is None
    assert soundbank.resolve({"path": font.path, "preset": "bass"}, synth_preset="piano").label.endswith("Sine Bass")
    assert soundbank.resolve({"path": font.path, "preset": "Acoustic Grand Piano"}).label.endswith("Sine Piano")
    assert soundbank.resolve({"path": font.path, "preset": "0:33"}).label.endswith("Sine Bass")
    assert soundbank.resolve({"path": font.path}, drums=True).label.endswith("128:0 Test Kit")
    missing = soundbank.resolve({"path": font.path, "preset": "Tuba"})
    assert missing.instrument is None and "no preset 'Tuba'" in missing.error
    # a sampled preset name instead of a synth preset: the synth preset of its family stands in
    arr = band({"b": {"sound": font.path, "preset": "slap bass"}, "s": {"sound": font.path, "preset": "Sine Piano",
                                                                        "fallback": "organ"}},
               LINE, {"b": "line", "s": "line"})
    comp = arranger.compile_arrangement(arr)
    assert comp.instruments["b"].preset == "bass_slap" and comp.instruments["b"].sound[0]["preset"] == "slap bass"
    assert comp.instruments["s"].preset == "organ"


@pytest.mark.parametrize("sound", ["no-such-set", "missing/folder", "junk.sf2", "set.sf3",
                                   {"set": "no-such-set", "preset": "0:33", "attack": 0.2}])
def test_missing_sound_falls_back_to_the_synth_bit_for_bit(tmp_path, monkeypatch, sound):
    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path / "sounds"))
    monkeypatch.chdir(tmp_path)
    (tmp_path / "junk.sf2").write_bytes(b"not a soundfont at all")
    (tmp_path / "set.sf3").write_bytes(b"RIFF\x04\0\0\0sfbk")
    insts = {"drums": {"type": "drums", "split": "groups"}, "bass": {"preset": "bass_pick"}}
    pats = dict(BEAT, **LINE)
    parts = {"drums": "beat", "bass": "line"}
    plain = stems_of(band(copy.deepcopy(insts), pats, parts))
    insts["drums"]["sound"] = sound
    insts["bass"]["sound"] = sound
    arr = band(insts, pats, parts)
    assert not [m for m in validate_arrangement(arr) if m.startswith("error")]
    got = stems_of(arr)
    assert set(got) == set(plain) == {"drums kick", "drums snare", "drums hats", "bass"}
    for name in plain:
        assert np.array_equal(got[name].audio, plain[name].audio), name
        assert got[name].sound == "" and len(got[name].warnings) == 1
        assert "plays instead" in got[name].warnings[0]
    rep = arranger.sound_report(project(arr))
    assert len(rep["warnings"]) == 2 and rep["instruments"]["bass"]["plays"] == "synth"
    summary = arranger.summarize(arr, project(arr))
    assert sum("plays instead" in m for m in summary["problems"]) == 2


def test_not_installed_set_is_a_validation_warning(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path))
    arr = band({"bass": {"preset": "bass_pick", "sound": "fancy-bass"}}, LINE, {"bass": "line"})
    msgs = validate_arrangement(arr)
    assert len(msgs) == 1 and msgs[0].startswith("warning") and "synth2 sounds install fancy-bass" in msgs[0]
    bad = band({"bass": {"preset": "bass_pick", "sound": {"preset": 3}}, "x": {"preset": "piano", "sound": 7},
                "y": {"preset": "piano", "sound": {"set": "a", "tune": "sharp", "key_range": ["C2"], "zap": 1}}},
               LINE, {"bass": "line", "x": "line", "y": "line"})
    msgs = validate_arrangement(bad)
    assert sum(m.startswith("error") for m in msgs) == 4 and any("unknown key 'zap'" in m for m in msgs)


def install_set(root, sid, kind, files, src=None, **extra):
    """An installed sound set in the layout of synth2.sounds."""
    folder = root / sid / "files"
    folder.mkdir(parents=True)
    if src is not None:
        (folder / files[0]).write_bytes(open(src, "rb").read())
    manifest = {"format": "synth2-sound", "version": 1, "id": sid, "kind": kind, "name": sid, "files": files}
    manifest.update(extra)
    (root / sid / "synth2-sound.json").write_text(json.dumps(manifest))
    return folder


def test_installed_set_by_id_with_credits(font, tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path))
    install_set(tmp_path, "tiny-gm", "soundfont", ["tiny.sf2"], src=font.path, credits=["Tiny GM by Nobody (CC0)"],
                tags=["gm"])
    arr = band({"drums": {"type": "drums", "sound": "tiny-gm"}, "bass": {"preset": "bass_pick", "sound": "tiny-gm"}},
               dict(BEAT, **LINE), {"drums": "beat", "bass": "line"})
    assert validate_arrangement(arr) == []
    got = stems_of(arr)
    assert got["bass"].sound == "tiny-gm: 0:33 Sine Bass" and got["bass"].credits == ("Tiny GM by Nobody (CC0)",)
    assert got["drums kick"].sound == "tiny-gm: 128:0 Test Kit"
    rep = arranger.sound_report(project(arr))
    assert rep["credits"] == ["Tiny GM by Nobody (CC0)"] and rep["warnings"] == []
    assert rep["instruments"]["bass"] == {"type": "synth", "sound": ["tiny-gm"], "plays": "sampled",
                                          "label": "tiny-gm: 0:33 Sine Bass"}
    listed = arranger.list_sounds(detail=True)
    assert [e["id"] for e in listed] == ["tiny-gm"]
    assert [p["preset"] for p in listed[0]["presets"]] == ["0:0", "0:33"] and listed[0]["kits"][0]["name"] == "Test Kit"


def test_sound_options_and_layers(font):
    def keys(sound, pats=LINE):
        return stems_of(band({"keys": {"preset": "piano", "sound": sound}}, pats, {"keys": "line"}))["keys"].audio

    base = {"path": font.path, "preset": "Sine Piano", "normalize": False}
    plain = keys(base)
    # tuning and transposition
    f = freq_of(keys(dict(base, tune=30))[int(0.2 * SR):int(0.9 * SR), 0], SR)
    assert abs(cents(f, F_ROOT * 2 ** (-3 / 12)) - 30) < 2
    f = freq_of(keys(dict(base, transpose=12))[int(0.2 * SR):int(0.9 * SR), 0], SR)
    assert abs(cents(f, 2 * F_ROOT * 2 ** (-3 / 12))) < 2
    # envelope overrides: a slow attack, a short release
    slow = keys(dict(base, attack=0.5))
    assert np.abs(slow[int(0.05 * SR):int(0.1 * SR)]).max() < 0.25 * np.abs(plain[int(0.05 * SR):int(0.1 * SR)]).max()
    short = keys(dict(base, release=0.02), {"line": {"kind": "notes", "score": "A3 q r q r h"}})
    long = keys(dict(base, release=1.0), {"line": {"kind": "notes", "score": "A3 q r q r h"}})
    after = slice(int(0.6 * SR), int(0.9 * SR))
    assert np.abs(short[after]).max() < 1e-3 < np.abs(long[after]).max()
    # level trim and velocity sensitivity
    assert abs(np.abs(keys(dict(base, gain_db=-6))).max() / np.abs(plain).max() - 0.501) < 0.01
    soft = {"line": {"kind": "notes", "notes": [[0, "A3", 2, 0.4]]}}
    assert np.abs(keys(dict(base, vel_sens=0.0), soft)).max() > 2.5 * np.abs(keys(base, soft)).max()
    # a filter closes the bright layer
    loud = {"line": {"kind": "notes", "notes": [[0, "C4", 2, 1.0]]}}
    seg = slice(int(0.3 * SR), int(0.9 * SR))
    bright, dark = keys(base, loud)[seg, 0], keys(dict(base, cutoff=300.0), loud)[seg, 0]
    third = lambda y: level_db(y, 3 * F_ROOT, SR) - level_db(y, F_ROOT, SR)  # noqa: E731
    assert third(dark) < third(bright) - 12
    # vibrato
    vib = keys(dict(base, vibrato=0.5, vibrato_rate=6.0), {"line": {"kind": "notes", "notes": [[0, "C4", 4, 0.8]]}})
    devs = [cents(freq_of(vib[a:a + 1024, 0], SR), F_ROOT) for a in range(8000, 30000, 256)]
    assert max(devs) > 30 and min(devs) < -30
    # two layers play together; a key range splits the keyboard
    both = keys([base, dict(base, transpose=12)])[int(0.2 * SR):int(0.9 * SR), 0]
    assert abs(level_db(both, F_ROOT * 2 ** (-3 / 12), SR) - level_db(both, 2 * F_ROOT * 2 ** (-3 / 12), SR)) < 6.5
    split = keys([dict(base, key_range=["C0", "C4"]), dict(base, key_range=["C#4", "C8"], transpose=12)])
    assert abs(cents(freq_of(split[int(0.2 * SR):int(0.9 * SR), 0], SR), F_ROOT * 2 ** (-3 / 12))) < 2
    assert abs(cents(freq_of(split[int(1.2 * SR):int(1.9 * SR), 0], SR), 2 * F_ROOT * 2 ** (4 / 12))) < 2


def test_mono_cut_and_glide(font):
    pats = {"line": {"kind": "notes", "notes": [[0, "A3", 3.9, 0.8], [2, "C4", 2, 0.8]]}}
    sound = {"path": font.path, "preset": "Sine Piano", "normalize": False, "release": 0.3}

    def keys(**kw):
        return stems_of(band({"keys": {"preset": "piano", "sound": dict(sound, **kw)}}, pats, {"keys": "line"}))["keys"].audio

    poly = keys()[int(1.3 * SR):int(1.9 * SR), 0]
    mono = keys(mono=True)[int(1.3 * SR):int(1.9 * SR), 0]
    fa, fc = F_ROOT * 2 ** (-3 / 12), F_ROOT
    assert level_db(poly, fa, SR) > level_db(poly, fc, SR) - 12  # both notes sound
    assert level_db(mono, fa, SR) < level_db(mono, fc, SR) - 40  # the first note was cut by the second
    glide = keys(mono=True, glide=0.4)
    assert -290 < cents(freq_of(glide[int(1.05 * SR):int(1.1 * SR), 0], SR), fc) < -150  # still on its way up
    assert abs(cents(freq_of(glide[int(1.5 * SR):int(1.9 * SR), 0], SR), fc)) < 2


# ---------------------------------------------------------------------------------------- kits

def test_sampled_kit_stems_choke_and_missing_drums(font):
    pats = {"beat": {"kind": "drums", "lanes": {"kick": "x.......x.......", "snare": "....x.......x...",
                                                "hat": "x.x.....x.x.x.x.", "ohat": "....x...........",
                                                "crash": "x...............", "cowbell": "..x............."}}}
    arr = band({"drums": {"type": "drums", "sound": {"path": font.path, "gate": 1.0}}}, pats, {"drums": "beat"})
    got = stems_of(arr)
    assert set(got) == {"drums kick", "drums snare", "drums hat", "drums ohat", "drums crash", "drums cowbell"}
    assert all(st.role == "drums" for st in got.values())
    assert got["drums kick"].sound.endswith("128:0 Test Kit")
    note = got["drums kick"].warnings
    assert len(note) == 1 and "has no cowbell, crash; synthesised" in note[0]
    # the open hat on beat 2 (0.5 s) rings for a second in the kit; the closed hat at beat 3 (1.0 s) chokes it
    oh = np.abs(got["drums ohat"].audio).max(axis=1)
    assert oh[int(0.6 * SR):int(0.95 * SR)].max() > 0.01
    assert oh[int(1.02 * SR):int(1.9 * SR)].max() < 1e-4
    unchoked = stems_of(band({"drums": {"type": "drums", "sound": {"path": font.path, "gate": 1.0}}},
                             {"beat": {"kind": "drums", "lanes": {"ohat": "....x..........."}}}, {"drums": "beat"}))
    assert np.abs(unchoked["drums ohat"].audio[int(1.02 * SR):int(1.4 * SR)]).max() > 0.005
    # the kick is the kit's sine at 60 Hz, not the synthesised kick
    kick = got["drums kick"].audio[int(0.01 * SR):int(0.25 * SR), 0]
    assert abs(freq_of(kick, SR, lo=20, hi=400) - 60) < 2


def test_per_drum_tuning_level_pan_and_own_sample(font, tmp_path):
    pats = {"beat": {"kind": "drums", "lanes": {"kick": "x...............", "snare": "........x......."}}}
    t = np.arange(SR // 2) / SR
    samples.write_wav(str(tmp_path / "boom.wav"), 0.8 * np.sin(2 * np.pi * 90 * t) * np.exp(-t / 0.1), SR)

    def kit(voices, **kw):
        arr = band({"drums": {"type": "drums", "sound": dict({"path": font.path, "normalize": False}, **kw),
                              "voices": voices}}, pats, {"drums": "beat"}, sample_dir=str(tmp_path))
        assert not [m for m in validate_arrangement(arr) if m.startswith("error")]
        return stems_of(arr)

    plain = kit({})
    seg = slice(int(0.01 * SR), int(0.25 * SR))
    tuned = kit({"kick": {"tune": 12, "gain_db": -6, "pan": -1.0}})
    assert abs(freq_of(tuned["drums kick"].audio[seg, 0], SR, lo=20, hi=400) - 120) < 3
    assert abs(np.abs(tuned["drums kick"].audio).max() / np.abs(plain["drums kick"].audio).max() - 0.5 * 2 ** 0.5) < 0.05
    assert np.abs(tuned["drums kick"].audio[:, 1]).max() < 1e-6
    assert np.array_equal(tuned["drums snare"].audio, plain["drums snare"].audio)
    own = kit({"kick": {"sample": "boom.wav"}})
    assert abs(freq_of(own["drums kick"].audio[seg, 0], SR, lo=20, hi=400) - 90) < 2
    other_key = kit({"kick": {"note": 38}})  # the kick lane plays the kit's snare key
    assert np.abs(other_key["drums kick"].audio).max() > 0.01
    assert freq_of(other_key["drums kick"].audio[seg, 0], SR, lo=20, hi=8000) > 300
    short = kit({"snare": {"length": 0.05}})
    after = slice(int(1.08 * SR), int(1.4 * SR))
    assert np.abs(short["drums snare"].audio[after]).max() < 1e-4 < np.abs(plain["drums snare"].audio[after]).max()


def test_synth_kit_split_per_drum_is_the_same_audio_on_more_stems():
    lanes = {"kick": "x.......x.......", "snare": "....x.......x...", "clap": "....x...........", "rim": "..x.............",
             "hat": "x.x.x.x.x.x.x.x.", "ohat": "..............x.", "tom1": "............x...", "tom3": ".............x..",
             "crash": "x...............", "ride": "....x...x...x...", "shaker": "xxxx............", "tamb": "......x.........",
             "cowbell": "..........x.....", "snap": "............x...", "conga_hi": "...x............", "n75": ".x.............."}
    pats = {"beat": {"kind": "drums", "lanes": lanes}}
    groups = stems_of(band({"drums": {"type": "drums"}}, pats, {"drums": "beat"}))
    each = stems_of(band({"drums": {"type": "drums", "split": "each"}}, pats, {"drums": "beat"}))
    one = stems_of(band({"drums": {"type": "drums", "split": False}}, pats, {"drums": "beat"}))
    assert set(groups) == {"drums kick", "drums snare", "drums hats", "drums cymbals"}
    assert set(each) == {f"drums {k}" for k in ("kick", "snare", "clap", "rim", "snap", "toms", "hat", "ohat", "shaker",
                                                "tamb", "cowbell", "crash", "ride", "congas", "claves")}
    assert set(one) == {"drums"}
    total = lambda d: sum(st.audio.astype(np.float64) for st in d.values())  # noqa: E731
    assert np.allclose(total(each), total(groups), atol=2e-6) and np.allclose(total(one), total(groups), atol=2e-6)
    assert np.array_equal(each["drums kick"].audio, groups["drums kick"].audio)
    ref = arranger.reference()
    assert ref["drum_stems_each"]["tom2"] == "toms" and ref["drum_stems_each"]["ridebell"] == "ride"
    assert "sound" in ref["sampled_sounds"]["how"] and ref["sampled_sounds"]["drum_keys"]["kick"] == [36, 35]


# ----------------------------------------------------------------------------------------- SFZ

from synth2.arranger import folder, sfz  # noqa: E402


@pytest.fixture(scope="module")
def sfz_dir(tmp_path_factory):
    """An SFZ multisample (looped sines, two velocity layers, a release click) and an SFZ kit
    (round-robin snare, choked open hat) over WAV files, with an include and a define."""
    root = tmp_path_factory.mktemp("sfz")
    smp = root / "Samples"
    smp.mkdir()
    rng = np.random.default_rng(11)
    n = PERIOD * 60
    samples.write_wav(str(smp / "soft C4.wav"), sine(n, amp=0.25), SR)
    samples.write_wav(str(smp / "loud_c4.wav"), sine(n, amp=0.5, harmonics=((3, 0.4),)), SR, bits=24)
    samples.write_wav(str(smp / "high_c5.wav"), sine(n, period=PERIOD // 2, amp=0.4), SR, float_format=True, bits=32,
                      loop=(PERIOD * 10, PERIOD * 40))
    samples.write_wav(str(smp / "click.wav"), 0.5 * rng.standard_normal(600), SR)
    t = np.arange(SR // 2) / SR
    samples.write_wav(str(smp / "kick.wav"), 0.9 * np.sin(2 * np.pi * 60 * t) * np.exp(-t / 0.08), SR)
    for i in range(3):
        samples.write_wav(str(smp / f"snare{i + 1}.wav"), 0.5 * rng.standard_normal(4000) * np.exp(-np.arange(4000) / 900), SR)
    samples.write_wav(str(smp / "hat.wav"), 0.3 * rng.standard_normal(1500) * np.exp(-np.arange(1500) / 200), SR)
    samples.write_wav(str(smp / "ohat.wav"), 0.3 * rng.standard_normal(SR) * np.exp(-np.arange(SR) / 9000), SR)
    (root / "env.inc").write_text("ampeg_attack=0.05 ampeg_release=$REL // shared envelope\n")
    (root / "sines.sfz").write_text(f"""// a test instrument
#define $REL 0.3
<control> default_path=Samples/ note_offset=0
<global> #include "env.inc"
 loop_mode=loop_continuous loop_start={PERIOD * 20} loop_end={PERIOD * 50 - 1} bogus_opcode=3 width_oncc21=50
<group> lokey=40 hikey=65 pitch_keycenter=c4
<region> sample=soft C4.wav hivel=79
<region> sample=LOUD_C4.wav lovel=80   // found without regard to case
<group> lokey=66
<region> sample=high_c5.wav pitch_keycenter=72 loop_start={PERIOD * 10} loop_end={PERIOD * 40 - 1}
<region> sample=not_there.wav key=20
<region> sample=click.wav trigger=release loop_mode=one_shot lokey=0 hikey=127 amp_veltrack=0
<region> sample=soft C4.wav key=30 locc64=64
<region> sample=soft C4.wav key=31 sw_last=25 sw_lokey=24 sw_hikey=25 sw_default=24
<region> sample=soft C4.wav key=32 sw_last=24
""")
    (root / "kit.sfz").write_text("""<control> default_path=Samples\\
<global> loop_mode=one_shot
<region> key=36 sample=kick.wav
<group> key=38 seq_length=3
<region> sample=snare1.wav seq_position=1
<region> sample=snare2.wav seq_position=2
<region> sample=snare3.wav seq_position=3
<group> group=1
<region> key=42 sample=hat.wav
<region> key=46 sample=ohat.wav off_by=1 off_mode=fast
""")
    return root


def test_sfz_parser_headers_values_with_spaces_and_inheritance():
    control, regions = sfz.parse("<control> default_path=My Samples/ <global> volume=-3 <group> lokey=c2 hikey=40\n"
                                 "<region> sample=A b c.wav tune=5 <region> sample=d.wav lokey=41\n"
                                 "<master> pan=10 <region> sample=e.wav")
    assert control == {"default_path": "My Samples/"}
    path = {"default_path": "My Samples/"}  # every region carries the <control> values in force where it stands
    assert regions[0] == dict(path, volume="-3", lokey="c2", hikey="40", sample="A b c.wav", tune="5")
    assert regions[1]["lokey"] == "41" and regions[1]["hikey"] == "40"
    assert regions[2] == dict(path, volume="-3", pan="10", sample="e.wav")  # a new <master> ends the group
    _c, later = sfz.parse("<control> default_path=a/ <region> sample=x.wav <control> default_path=b/ <region> sample=y.wav")
    assert [r["default_path"] for r in later] == ["a/", "b/"]


def test_sfz_instrument_pitch_loop_layers_and_envelope(sfz_dir):
    inst = sfz.load_sfz(str(sfz_dir / "sines.sfz"))
    for key in (55, 60, 65, 66, 72, 77):
        y = render_note(inst, key, 100, 1.5, 44100)
        assert abs(cents(freq_of(y[8820:44100, 0], 44100), F_ROOT * 2 ** ((key - 60) / 12))) < 1.5, key
    y = render_note(inst, 62, 60, 3.0, SR)[SR // 2:3 * SR - 100, 0].astype(np.float64)
    assert np.abs(np.diff(y)).max() <= 1.05 * np.abs(y).max() * 2 * np.pi * F_ROOT * 2 ** (2 / 12) / SR
    third = lambda y: level_db(y, 3 * F_ROOT, SR) - level_db(y, F_ROOT, SR)  # noqa: E731
    assert third(render_note(inst, 60, 60, 1.0, SR)[SR // 4:SR, 0]) < -50
    assert abs(third(render_note(inst, 60, 110, 1.0, SR)[SR // 4:SR, 0]) - 20 * np.log10(0.4)) < 1.0
    env = np.abs(render_note(inst, 60, 127, 1.0, SR)[:, 0])
    assert abs(env[int(0.02 * SR):int(0.03 * SR)].max() / env[int(0.3 * SR):int(0.9 * SR)].max() - 0.5) < 0.1
    assert len(env) <= int(1.3 * SR) + 600 + 8  # release 0.3 s from the include, then the release click
    assert np.abs(render_note(inst, 60, 127, 1.0, SR)[SR + 20:SR + 500]).max() > 0.2  # the release sample
    # regions behind a controller or another keyswitch are left out; the default keyswitch plays
    assert inst.select(30, 100) == [] and inst.select(31, 100) == [] and len(inst.select(32, 100)) == 1
    text = " | ".join(inst.warnings)
    assert "1 samples not found (first: not_there.wav)" in text and "bogus_opcode" in text
    assert "1 controller modulations ignored" in text and "controller away from its default" in text
    assert "other keyswitch articulations" in text


def test_sfz_kit_round_robin_is_the_same_in_any_window(sfz_dir):
    kit = sfz.load_sfz(str(sfz_dir / "kit.sfz"))
    assert kit.drums and kit.keys() == [36, 38, 42, 46]
    assert [kit.select(38, 100, seq=i)[0].sample.name for i in range(4)] == ["snare1", "snare2", "snare3", "snare1"]
    pats = {"beat": {"kind": "drums", "lanes": {"kick": "x.......x.......", "snare": "..x...x...x...x.",
                                                "hat": "x.x.x...x.x.x.x.", "ohat": "....x..........."}}}
    arr = band({"drums": {"type": "drums", "sound": str(sfz_dir / "kit.sfz")}}, pats, {"drums": "beat"}, bars=4)
    p = project(arr)
    full = {st.name: st.audio for st in render_backing(p, SR, 0.0, 8.0)}
    again = {st.name: st.audio for st in render_backing(p, SR, 0.0, 8.0)}
    part = {st.name: st.audio for st in render_backing(p, SR, 5.0, 7.0)}
    assert set(full) == {"drums kick", "drums snare", "drums hat", "drums ohat"}
    for name in full:
        assert np.array_equal(full[name], again[name])
        assert np.allclose(part[name], full[name][5 * SR:7 * SR], atol=1e-6), name
    # successive snare hits are different takes
    sn = full["drums snare"][:, 0]

    def hit_at(t):
        seg = sn[int(t * SR) - 4:int(t * SR) + 2100]
        return seg[np.flatnonzero(np.abs(seg) > 1e-9)[0]:][:1900]

    hits = [hit_at(t) for t in (0.25, 0.75, 1.25, 1.75)]
    assert not np.allclose(hits[0], hits[1], atol=1e-3) and not np.allclose(hits[1], hits[2], atol=1e-3)
    assert np.allclose(hits[0], hits[3], atol=1e-6)
    # the open hat (0.5 s) is choked by the closed hat at 1.0 s
    oh = np.abs(full["drums ohat"]).max(axis=1)
    assert oh[int(0.9 * SR):int(0.99 * SR)].max() > 1e-3 and oh[int(1.03 * SR):int(1.9 * SR)].max() < 1e-5


def test_sfz_in_an_arrangement_and_broken_files(sfz_dir, tmp_path):
    arr = band({"keys": {"preset": "piano", "sound": {"path": "sines.sfz", "normalize": False}}}, LINE, {"keys": "line"},
               sample_dir=str(sfz_dir))
    st = stems_of(arr)["keys"]
    assert st.sound == "sines.sfz" and any("not supported and ignored" in w for w in st.warnings)
    assert abs(cents(freq_of(st.audio[int(0.2 * SR):int(0.9 * SR), 0], SR), F_ROOT * 2 ** (-3 / 12))) < 2
    (tmp_path / "empty.sfz").write_text("<region> sample=nothing.wav\n<region> sample=*sine\n<effect> bus=main\n")
    (tmp_path / "binary.sfz").write_bytes(bytes(range(256)) * 20)
    for name in ("empty.sfz", "binary.sfz"):
        res = soundbank.resolve({"path": str(tmp_path / name)}, synth_preset="piano")
        assert res.instrument is None and "no playable samples" in res.error
    got = stems_of(band({"keys": {"preset": "piano", "sound": str(tmp_path / "empty.sfz")}}, LINE, {"keys": "line"}))
    assert got["keys"].sound == "" and "plays instead" in got["keys"].warnings[0]


def test_sfz_crossfades_filter_and_offsets(tmp_path):
    harm = tuple((k, 1.0) for k in range(2, 20))
    samples.write_wav(str(tmp_path / "a.wav"), sine(PERIOD * 80, amp=0.04, harmonics=harm), SR)
    (tmp_path / "x.sfz").write_text(f"""<global> loop_mode=loop_continuous loop_start={PERIOD * 20} loop_end={PERIOD * 60 - 1}
 pitch_keycenter=60 amp_veltrack=0
<region> sample=a.wav lokey=0 hikey=59 cutoff={4 * F_ROOT} fil_type=lpf_2p resonance=0 fil_keytrack=100 fil_keycenter=60
<region> sample=a.wav key=60 xfin_lovel=20 xfin_hivel=100 pitch_keycenter=60
<region> sample=a.wav lokey=61 hikey=72 transpose=12 tune=-100 volume=-6 pan=100
<region> sample=a.wav lokey=73 hikey=80 cutoff=500 fil_type=brf_2p
""")
    inst = sfz.load_sfz(str(tmp_path / "x.sfz"))
    y = render_note(inst, 48, 100, 1.0, SR)[SR // 4:SR, 0]
    assert abs(level_db(y, 8 * F_ROOT / 2, SR) - level_db(y, F_ROOT / 2, SR) + 11.4) < 1.5  # filter follows the key
    full = np.abs(render_note(inst, 60, 100, 0.5, SR)).max()
    assert abs(np.abs(render_note(inst, 60, 60, 0.5, SR)).max() / full - np.sqrt(0.5)) < 0.02
    assert np.abs(render_note(inst, 60, 20, 0.5, SR)).max() < 1e-6
    up = render_note(inst, 66, 100, 1.0, SR)
    f_up = F_ROOT * 2 ** ((66 - 60 + 11) / 12)  # transpose=12 and tune=-100
    assert abs(cents(freq_of(up[SR // 4:SR, 1], SR, hi=1.5 * f_up), f_up)) < 2
    assert np.abs(up[:, 0]).max() < 1e-6
    assert inst.select(75, 100)[0].cutoff is None and any("fil_type=brf_2p" in w for w in inst.warnings)


# --------------------------------------------------------------------------------- WAV folders

def test_drum_names():
    name = folder.drum_voice_from_name
    assert name("Kick 01.wav") == "kick" and name("BD_hard.wav") == "kick" and name("kit/Snare/hard.wav") == "snare"
    assert name("HiHat Open.wav") == "ohat" and name("hh_closed_2.wav") == "hat" and name("ClosedHat.wav") == "hat"
    assert name("OpenHat3.wav") == "ohat" and name("pedal hh.wav") == "phat" and name("Ride Bell.wav") == "ridebell"
    assert name("crash_a.wav") == "crash" and name("Tom Low.wav") == "tom3" and name("tom1.wav") == "tom1"
    assert name("FloorTom.wav") == "tom3" and name("Conga Low.wav") == "conga_lo" and name("cowbell.wav") == "cowbell"
    assert name("sidestick.wav") == "rim" and name("Clap.wav") == "clap" and name("808/tamb.wav") == "tamb"
    assert name("texture.wav") is None and name("pad_C3.wav") is None


def test_folder_kit_maps_drums_layers_and_round_robins(tmp_path):
    rng = np.random.default_rng(4)
    kit = tmp_path / "MyKit"
    (kit / "Snare").mkdir(parents=True)
    t = np.arange(SR // 4) / SR
    samples.write_wav(str(kit / "Kick.wav"), 0.9 * np.sin(2 * np.pi * 55 * t) * np.exp(-t / 0.08), SR)
    for layer, amp in (("v1", 0.1), ("v2", 0.6)):
        for rr in (1, 2):
            samples.write_wav(str(kit / "Snare" / f"hit_{layer}_rr{rr}.wav"),
                              amp * rng.standard_normal(3000) * np.exp(-np.arange(3000) / 700), SR)
    samples.write_wav(str(kit / "HiHat Closed.wav"), 0.3 * rng.standard_normal(1200) * np.exp(-np.arange(1200) / 150), SR)
    samples.write_wav(str(kit / "HiHat Open.wav"), 0.3 * rng.standard_normal(SR) * np.exp(-np.arange(SR) / 9000), SR)
    samples.write_wav(str(kit / "Crash.wav"), 0.3 * rng.standard_normal(SR) * np.exp(-np.arange(SR) / 9000), SR)
    samples.write_wav(str(kit / "zap.wav"), 0.3 * rng.standard_normal(900), SR)
    (kit / "notes.txt").write_text("not audio")
    inst = folder.folder_instrument(str(kit), True)
    assert inst.drums and inst.keys() == [36, 38, 42, 46, 49, 84]
    assert inst.info["voices"] == {"kick": 36, "snare": 38, "hat": 42, "ohat": 46, "crash": 49, "n84": "zap.wav"}
    assert [r.sample.name for r in inst.select(38, 30, seq=0)] == ["hit_v1_rr1"]
    assert [r.sample.name for r in inst.select(38, 30, seq=1)] == ["hit_v1_rr2"]
    assert [r.sample.name for r in inst.select(38, 120, seq=0)] == ["hit_v2_rr1"]
    assert sampler.note_groups(inst.select(46, 100)) == ({1}, {1}) and "name no drum" in inst.warnings[0]
    pats = {"beat": {"kind": "drums", "lanes": {"kick": "x.......", "snare": "....x...", "hat": "x.x.x.x.", "ohat": "......x.",
                                                "crash": "x...............", "n84": "...x............",
                                                "clap": "....x..........."}}}
    arr = band({"drums": {"type": "drums", "sound": "MyKit"}}, pats, {"drums": "beat"}, sample_dir=str(tmp_path))
    got = stems_of(arr)
    assert set(got) == {"drums kick", "drums snare", "drums hat", "drums ohat", "drums crash", "drums perc", "drums clap"}
    assert got["drums kick"].sound == "MyKit" and "has no clap; synthesised" in " ".join(got["drums kick"].warnings)
    assert abs(freq_of(got["drums kick"].audio[200:5000, 0], SR, lo=20, hi=400) - 55) < 3
    assert folder.folder_instrument(str(kit)).drums  # guessed from the names


def test_folder_multisample_from_note_names(tmp_path):
    inst_dir = tmp_path / "Tines"
    inst_dir.mkdir()
    for note, per in (("C3", PERIOD * 2), ("C4", PERIOD), ("C5", PERIOD // 2)):
        samples.write_wav(str(inst_dir / f"tine_{note}.wav"), sine(per * 120, period=per, amp=0.4) *
                          np.exp(-np.arange(per * 120) / (per * 40)), SR, loop=None)
    samples.write_wav(str(inst_dir / "noise.wav"), np.zeros(100), SR)
    inst = folder.folder_instrument(str(inst_dir), False)
    assert not inst.drums and inst.info["notes"] == [48, 60, 72]
    assert [(r.lokey, r.hikey, r.root) for r in inst.regions] == [(0, 54, 48), (55, 66, 60), (67, 127, 72)]
    for key in (45, 54, 55, 62, 70):
        y = render_note(inst, key, 100, 0.6, SR)[1000:9000, 0]
        assert abs(cents(freq_of(y, SR), F_ROOT * 2 ** ((key - 60) / 12))) < 2, key
    assert "name no note" in inst.warnings[0]
    got = stems_of(band({"keys": {"preset": "epiano", "sound": str(inst_dir)}}, LINE, {"keys": "line"}))
    assert got["keys"].sound == "Tines"
    one = folder.single_sample(str(inst_dir / "tine_C4.wav"), False)
    assert one.regions[0].root == 60 and one.keys() == list(range(128))
    with pytest.raises(SampleError, match="no sample files"):
        folder.folder_instrument(str(tmp_path / "Tines" / ".."), False) if False else folder.folder_instrument(str(tmp_path / "empty_dir_x"), False)


def test_flac_needs_soundfile_or_says_so(tmp_path, monkeypatch):
    data = sine(PERIOD * 40)
    path = tmp_path / "tone_C4.flac"
    if samples.have_soundfile():
        import soundfile as sf

        sf.write(str(path), data, SR, subtype="PCM_16")
        smp = samples.open_sample(str(path))
        assert (smp.rate, smp.frames, smp.channels) == (SR, len(data), 1)
        assert np.abs(smp.read(100, 600)[:, 0] - data[100:600]).max() < 1e-4
        inst = folder.single_sample(str(path), False)
        assert abs(cents(freq_of(render_note(inst, 64, 100, 0.1, SR)[:3000, 0], SR), F_ROOT * 2 ** (4 / 12))) < 3
    else:
        path.write_bytes(b"fLaC")
    import builtins

    real_import = builtins.__import__

    def no_soundfile(name, *a, **kw):
        if name == "soundfile":
            raise ImportError("no soundfile")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", no_soundfile)
    with pytest.raises(SampleError, match="needs the optional 'soundfile' package"):
        samples.open_sample(str(path))
    res = soundbank.resolve({"path": str(path)}, synth_preset="piano")
    assert res.instrument is None and "soundfile" in res.error


def test_sfz_controllers_at_their_defaults_velocity_curve_and_accent_layers(tmp_path):
    samples.write_wav(str(tmp_path / "a.wav"), sine(PERIOD * 80), SR)
    (tmp_path / "cc.sfz").write_text(f"""#define $REL 72
<control> set_cc$REL=64 set_hdcc30=0.5
<global> sample=a.wav loop_mode=loop_continuous loop_start={PERIOD * 20} loop_end={PERIOD * 60 - 1} pitch_keycenter=60
<region> key=60 ampeg_release_oncc$REL=2
<region> key=61 amp_velcurve_64=1 amp_veltrack=100
<region> key=62 xfin_locc30=0 xfin_hicc30=127 volume=0
<region> key=63 xfout_locc30=0 xfout_hicc30=60
<region> key=64 amplitude=50
""")
    inst = sfz.load_sfz(str(tmp_path / "cc.sfz"))
    assert abs(inst.select(60, 100)[0].ampeg.release - (0.001 + 2 * 64 / 127)) < 1e-6
    assert abs(len(render_note(inst, 60, 100, 0.2, SR)) / SR - (0.2 + 0.001 + 2 * 64 / 127)) < 0.01
    full = np.abs(render_note(inst, 61, 127, 0.3, SR)).max()
    assert abs(np.abs(render_note(inst, 61, 64, 0.3, SR)).max() / full - 1.0) < 0.01  # the curve is flat from 64 up
    assert abs(np.abs(render_note(inst, 61, 32, 0.3, SR)).max() / full - 0.5) < 0.01
    assert abs(np.abs(render_note(inst, 62, 127, 0.3, SR)).max() / full - np.sqrt(0.5)) < 0.01
    assert inst.select(63, 100) == []  # faded out completely at the controller's default
    assert abs(np.abs(render_note(inst, 64, 127, 0.3, SR)).max() / full - 0.5) < 0.01
    assert not any("ignored" in w for w in inst.warnings)
    # kick.wav next to kick-accent.wav: two velocity layers
    kit = tmp_path / "CR"
    kit.mkdir()
    rng = np.random.default_rng(8)
    for name in ("kick", "kick-accent", "conga-l", "conga-h", "tom-h", "tom-ll", "cowb", "hihat", "hihat-open"):
        samples.write_wav(str(kit / f"{name}.wav"), 0.3 * rng.standard_normal(800), SR)
    inst = folder.folder_instrument(str(kit), True)
    assert inst.info["voices"] == {"kick": 36, "conga_lo": 64, "conga_hi": 63, "tom1": 50, "tom3b": 41, "cowbell": 56,
                                   "hat": 42, "ohat": 46}
    assert [r.sample.name for r in inst.select(36, 90)] == ["kick"]
    assert [r.sample.name for r in inst.select(36, 115)] == ["kick-accent"]


def test_sfz_set_presets_with_defines_controllers_and_keyswitch(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path))
    root = install_set(tmp_path, "blend", "sfz", ["Programs/main.sfz"], presets=[
        {"name": "Dark", "file": "Programs/main.sfz", "defines": {"$dir": "../Samples"}, "cc": {"100": 0}},
        {"name": "Bright", "file": "Programs/main.sfz", "defines": {"$dir": "../Samples"}, "cc": {"100": 127}}])
    (root / "Programs").mkdir()
    (root / "Samples").mkdir()
    samples.write_wav(str(root / "Samples" / "dark.wav"), sine(PERIOD * 60), SR)
    samples.write_wav(str(root / "Samples" / "bright.wav"), sine(PERIOD * 60, period=PERIOD // 2), SR)
    (root / "Programs" / "main.sfz").write_text("""<control> default_path=$dir/ set_cc100=0
<global> pitch_keycenter=60 loop_mode=one_shot
<region> sample=dark.wav hicc100=63 sw_last=24 sw_lokey=24 sw_hikey=25 sw_default=24
<region> sample=bright.wav locc100=64 sw_last=24
<region> sample=bright.wav sw_last=25 transpose=12
""")
    dark = soundbank.resolve({"set": "blend", "preset": "Dark", "normalize": False})
    bright = soundbank.resolve({"set": "blend", "preset": "bright", "normalize": False})
    assert dark.label == "blend: Dark" and bright.label == "blend: Bright"
    f = lambda res: freq_of(render_note(res.instrument, 60, 100, 0.2, SR)[500:4000, 0], SR)  # noqa: E731
    assert abs(cents(f(dark), F_ROOT)) < 2 and abs(cents(f(bright), 2 * F_ROOT)) < 2
    # the layer's own controller value and keyswitch
    own = soundbank.resolve({"set": "blend", "preset": "Dark", "cc": {"100": 100}, "normalize": False})
    assert abs(cents(f(own), 2 * F_ROOT)) < 2
    ks = soundbank.resolve({"set": "blend", "keyswitch": 25, "normalize": False})
    assert abs(cents(f(ks), 4 * F_ROOT)) < 2
    listed = arranger.list_sounds(detail=True)[0]
    assert [p["name"] for p in listed["presets"]] == ["Dark", "Bright"]
    arr = band({"g": {"preset": "gtr_clean", "sound": {"set": "blend", "preset": "Bright", "cc": {"100": 127},
                                                        "keyswitch": "C1"}}}, LINE, {"g": "line"})
    assert validate_arrangement(arr) == [] and stems_of(arr)["g"].sound == "blend: Bright"


# --------------------------------------------------------------------------------------- tools

def test_list_sounds_tool(font, tmp_path, monkeypatch):
    from synth2.ai.tools import Session, ToolError, call_tool

    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path))
    s = Session(Project.new("t", bpm=BPM))
    empty = call_tool(s, "list_sounds", {})
    assert empty["sets"] == [] and "synth2 sounds install" in empty["hint"]
    install_set(tmp_path, "tiny-gm", "soundfont", ["tiny.sf2"], src=font.path, credits=["Tiny GM (CC0)"], tags=["gm"],
                license="CC0-1.0")
    sets = call_tool(s, "list_sounds", {})["sets"]
    assert sets == [{"id": "tiny-gm", "kind": "soundfont", "name": "tiny-gm", "tags": ["gm"], "license": "CC0-1.0",
                     "credits": ["Tiny GM (CC0)"], "presets": 2, "kits": ["Test Kit"]}]
    one = call_tool(s, "list_sounds", {"set": "tiny-gm", "search": "bass"})["set"]
    assert one["presets"] == [{"preset": "0:33", "name": "Sine Bass"}] and one["presets_matching"] == 1
    assert one["kits"] == []
    with pytest.raises(ToolError, match="no installed sound set"):
        call_tool(s, "list_sounds", {"set": "nope"})
    arr = band({"bass": {"preset": "bass_pick", "sound": "tiny-gm"}, "keys": {"preset": "piano", "sound": "absent"}},
               LINE, {"bass": "line", "keys": "line"})
    r = call_tool(s, "set_backing", {"arrangement": arr})
    assert r["ok"] and any("'absent' is not installed" in w for w in r["warnings"])
    assert r["summary"]["instruments"]["bass"]["sound"] == ["tiny-gm"]
    used = call_tool(s, "list_sounds", {})["backing"]
    assert used["instruments"]["bass"]["label"] == "tiny-gm: 0:33 Sine Bass" and used["credits"] == ["Tiny GM (CC0)"]
    assert len(used["warnings"]) == 1 and "keys: sound set 'absent' is not installed" in used["warnings"][0]
    summary = call_tool(s, "get_backing", {})["backing"]
    assert any("absent" in m and "plays instead" in m for m in summary["problems"])
    assert "sampled_sounds" in call_tool(s, "get_backing", {"detail": "reference"})


def test_any_window_of_a_sampled_part_is_the_same_audio(font):
    arr = band({"keys": {"preset": "piano", "sound": {"path": font.path, "preset": "Sine Piano", "release": 0.6},
                         "eq": {"bands": [{"hz": 800, "db": -4, "q": 1}], "low_cut": 80}, "humanize": {"timing_ms": 3}},
                "drums": {"type": "drums", "sound": {"path": font.path, "gate": 0.5}, "eq": {"high_cut": 6000}}},
               dict(LINE, **BEAT), {"keys": "line", "drums": "beat"}, bars=4)
    p = project(arr)
    full = {st.name: st.audio for st in render_backing(p, SR, 0.0, 8.0)}
    part = {st.name: st.audio for st in render_backing(p, SR, 2.7, 5.3)}
    a = int(round(2.7 * SR))
    assert set(part) == set(full) and "keys" in full
    n = len(part["keys"])
    assert np.allclose(part["keys"], full["keys"][a:a + n], atol=1e-6)  # the EQ of a pitched part is per note
    for name in full:  # a kit's EQ runs over the rendered buffer: the same but for its last milliseconds
        m = n - int(0.05 * SR)
        assert np.allclose(part[name][:m], full[name][a:a + m], atol=2e-5), name


def test_in_memory_sample_keys_follow_the_content_not_the_object():
    a = samples.from_array(sine(PERIOD * 40), SR)
    b = samples.from_array(sine(PERIOD * 40), SR)
    c = samples.from_array(0.5 * sine(PERIOD * 40), SR)
    assert a.key == b.key
    assert a.key != c.key
