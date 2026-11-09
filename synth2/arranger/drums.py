"""Synthesised drum kit: kick, snare, clap, rim, hats, cymbals, toms and small percussion.

Each kit voice is a one-shot rendered once per (parameters, round-robin variant,
velocity layer) and cached; the sequencer then just adds scaled copies at the hit
times, which makes thousands of hits cost almost nothing.  Every voice takes a
parameter dict (see :data:`DRUM_DEFAULTS`) so an arrangement can retune the kit,
e.g. ``{"kick": {"pitch": 50, "decay": 0.5, "click": 0.8}}``.
"""
from __future__ import annotations

import zlib
from functools import lru_cache

import numpy as np

from .dsp import bandpass_noise, fft_filter, noise, osc, saturate, sine

# kit voice -> default parameters (all optional in an arrangement)
DRUM_DEFAULTS: dict[str, dict] = {
    "kick": {"pitch": 52.0, "sweep": 3.4, "sweep_time": 0.032, "decay": 0.42, "click": 0.6, "drive": 0.35,
             "level": 0.321, "pan": 0.0},
    "snare": {"tone": 195.0, "tone_decay": 0.07, "snappy": 0.75, "noise_decay": 0.11, "crack": 0.6,
              "brightness": 0.6, "drive": 0.25, "level": 0.169, "pan": 0.0},
    "clap": {"tone": 1150.0, "decay": 0.16, "spread": 0.011, "bursts": 4, "level": 0.845, "pan": 0.0},
    "rim": {"tone": 1650.0, "decay": 0.05, "level": 0.924, "pan": -0.1},
    "hat": {"tone": 1.0, "decay": 0.045, "brightness": 0.6, "level": 0.604, "pan": 0.22},
    "ohat": {"tone": 1.0, "decay": 0.42, "brightness": 0.6, "level": 0.378, "pan": 0.22},
    "crash": {"tone": 1.0, "decay": 2.2, "brightness": 0.6, "level": 0.427, "pan": -0.3},
    "ride": {"tone": 1.0, "decay": 1.4, "bell": 0.4, "level": 0.162, "pan": 0.32},
    "tom1": {"pitch": 185.0, "decay": 0.32, "level": 0.208, "pan": 0.25},
    "tom2": {"pitch": 138.0, "decay": 0.38, "level": 0.197, "pan": 0.0},
    "tom3": {"pitch": 98.0, "decay": 0.45, "level": 0.186, "pan": -0.25},
    "shaker": {"decay": 0.05, "brightness": 0.6, "level": 0.168, "pan": 0.4},
    "tamb": {"decay": 0.16, "level": 0.147, "pan": -0.4},
    "cowbell": {"tone": 1.0, "decay": 0.09, "level": 0.183, "pan": -0.2},
    "snap": {"decay": 0.04, "level": 0.533, "pan": 0.1},
}
DRUM_VOICES = tuple(DRUM_DEFAULTS)
DRUM_ALIASES = {
    "bd": "kick", "bass_drum": "kick", "k": "kick",
    "sd": "snare", "s": "snare",
    "cp": "clap", "handclap": "clap",
    "rs": "rim", "rimshot": "rim", "stick": "rim",
    "hh": "hat", "ch": "hat", "chh": "hat", "closed_hat": "hat", "hihat": "hat",
    "oh": "ohat", "ohh": "ohat", "open_hat": "ohat", "openhat": "ohat",
    "cr": "crash", "cy": "crash", "cymbal": "crash",
    "rd": "ride",
    "ht": "tom1", "tom_hi": "tom1", "hi_tom": "tom1",
    "mt": "tom2", "tom_mid": "tom2", "mid_tom": "tom2", "tom": "tom2",
    "lt": "tom3", "tom_lo": "tom3", "low_tom": "tom3", "floor_tom": "tom3",
    "sh": "shaker", "tambourine": "tamb", "tb": "tamb", "cb": "cowbell", "fingersnap": "snap",
}
# stem grouping of kit voices
DRUM_GROUPS = {
    "kick": ("kick",),
    "snare": ("snare", "clap", "rim", "snap", "tom1", "tom2", "tom3"),
    "hats": ("hat", "ohat", "shaker", "tamb", "cowbell"),
    "cymbals": ("crash", "ride"),
}
GROUP_OF = {v: g for g, vs in DRUM_GROUPS.items() for v in vs}
N_VARIANTS = 4


def canonical_voice(name: str) -> str | None:
    n = str(name).strip().lower()
    if n in DRUM_DEFAULTS:
        return n
    return DRUM_ALIASES.get(n)


def _t(n, sr):
    return np.arange(n, dtype=np.float32) / sr


def _metal(n, sr, freqs, rng):
    """Six detuned band-limited square waves (the 808/909 'metal' source)."""
    out = np.zeros(n, np.float32)
    for f in freqs:
        out += osc("square", f * (1 + rng.uniform(-0.004, 0.004)), n, sr, rng.uniform())
    return out / len(freqs)


_HAT_FREQS = (205.3, 304.4, 369.6, 522.7, 540.0, 800.0)


def _kick(sr, p, rng, layer):
    dur = float(p["decay"])
    n = int(sr * (dur * 1.25 + 0.02))
    t = _t(n, sr)
    f_end = float(p["pitch"]) * (1 + rng.uniform(-0.004, 0.004))
    f_start = f_end * float(p["sweep"]) * (0.9 + 0.1 * layer)
    f = f_end + (f_start - f_end) * np.exp(-t / float(p["sweep_time"]))
    body = sine(f, n, sr, phase0=0.0)
    hold = 0.012
    env = np.where(t < hold, 1.0, np.exp(-(t - hold) / (dur * 0.42))).astype(np.float32)
    env *= np.minimum(1.0, t / 0.0006)
    body *= env
    # beater click: a short band of noise plus a tiny high blip
    nc = int(0.012 * sr)
    click = bandpass_noise(nc, sr, rng, 1500, 9000)[:nc] * np.exp(-_t(nc, sr) / 0.0016)
    click += 0.6 * sine(2900.0, nc, sr) * np.exp(-_t(nc, sr) / 0.002)
    out = saturate(body, float(p["drive"]))
    out[:nc] += click * float(p["click"]) * 0.5 * (0.6 + 0.4 * layer)  # after the drive, so it stays crisp
    return out


def _snare(sr, p, rng, layer):
    nd = float(p["noise_decay"])
    n = int(sr * (nd * 3.2 + 0.05))
    t = _t(n, sr)
    f0 = float(p["tone"]) * (1 + rng.uniform(-0.006, 0.006))
    f1 = f0 * (1 + 0.55 * np.exp(-t / 0.01))
    td = float(p["tone_decay"])
    tone = sine(f1, n, sr) * np.exp(-t / td)
    tone += 0.55 * sine(f1 * 1.69, n, sr) * np.exp(-t / (td * 0.6))
    tone += 0.25 * sine(f1 * 2.71, n, sr) * np.exp(-t / (td * 0.4))
    bright = float(p["brightness"]) * (0.7 + 0.3 * layer)
    nz = bandpass_noise(n, sr, rng, 1100 + 600 * bright, 7000 + 7000 * bright)
    nz += 0.35 * bandpass_noise(n, sr, rng, 250, 1100)
    att = np.minimum(1.0, t / 0.0008)
    nz *= att * np.exp(-t / nd)
    nc = int(0.01 * sr)
    crack = bandpass_noise(nc, sr, rng, 2500, 12000)[:nc] * np.exp(-_t(nc, sr) / 0.0025)
    out = tone * 0.9 + nz * float(p["snappy"]) * 1.4
    out[:nc] += crack * float(p["crack"]) * 1.2
    return saturate(out * 0.8, float(p["drive"]))


def _clap(sr, p, rng, layer, stereo_offset=0.0):
    decay = float(p["decay"])
    spread = float(p["spread"])
    bursts = int(p["bursts"])
    n = int(sr * (spread * bursts + decay * 3.5 + 0.02))
    t = _t(n, sr)
    env = np.zeros(n, np.float32)
    for i in range(bursts):
        t0 = i * spread * (1 + rng.uniform(-0.15, 0.15)) + stereo_offset * (i > 0)
        m = t >= t0
        env[m] += np.exp(-(t[m] - t0) / 0.0032) * (0.8 + 0.2 * rng.uniform())
    t_last = (bursts - 1) * spread
    m = t >= t_last
    env[m] += 0.55 * np.exp(-(t[m] - t_last) / decay)
    tone = float(p["tone"])
    nz = bandpass_noise(n, sr, rng, tone * 0.62, tone * 2.6, slope=1.5)
    nz += 0.25 * bandpass_noise(n, sr, rng, 3000, 10000)
    return nz * env * 1.1


def _rim(sr, p, rng, layer):
    n = int(sr * (float(p["decay"]) * 4 + 0.01))
    t = _t(n, sr)
    f = float(p["tone"])
    out = sine(f, n, sr) * np.exp(-t / 0.006) + 0.6 * sine(f * 0.49, n, sr) * np.exp(-t / 0.01)
    out += 0.5 * bandpass_noise(n, sr, rng, 1500, 6000) * np.exp(-t / 0.004)
    out *= np.exp(-t / float(p["decay"])) * 1.4
    return out


def _hat(sr, p, rng, layer, open_=False):
    decay = float(p["decay"])
    n = int(sr * (decay * (3.0 if open_ else 3.4) + 0.01))
    t = _t(n, sr)
    tone = float(p["tone"])
    metal = _metal(n, sr, [f * tone * 1.0 for f in _HAT_FREQS], rng)
    bright = float(p["brightness"]) * (0.75 + 0.25 * layer)
    src = metal * 1.2 + 0.5 * noise(n, rng)
    lo = 6500 + 2500 * bright
    hp = fft_filter(src, sr, lambda f: (1.0 / np.sqrt(1.0 + (lo / np.maximum(f, 1.0)) ** 6))
                    * (1.0 + 0.8 * np.exp(-0.5 * ((f - 10500) / 2500) ** 2)))
    att = np.minimum(1.0, t / 0.0005)
    if open_:
        env = 0.65 * np.exp(-t / (decay * 0.5)) + 0.35 * np.exp(-t / 0.02)
    else:
        env = np.exp(-t / (decay * 0.42))
    return hp * env * att * 1.5


def _crash(sr, p, rng, layer, ride=False):
    decay = float(p["decay"])
    n = int(sr * (decay * 1.2 + 0.05))
    t = _t(n, sr)
    tone = float(p["tone"])
    out = np.zeros((n, 2), np.float32)
    for ch in range(2):
        r = np.random.default_rng(rng.integers(1 << 30))
        freqs = [f * tone * (2.2 if not ride else 1.6) for f in _HAT_FREQS]
        metal = _metal(n, sr, freqs, r)
        if ride:
            src = metal * 1.4 + 0.25 * noise(n, r)
            band = fft_filter(src, sr, lambda f: (1.0 / np.sqrt(1.0 + (3200.0 / np.maximum(f, 1.0)) ** 4))
                              / np.sqrt(1.0 + (f / 13000.0) ** 2))
            env = 0.7 * np.exp(-t / (decay * 0.3)) + 0.3 * np.exp(-t / 0.03)
            bell = float(p.get("bell", 0.4))
            ping = sum(sine(bf * tone, n, sr, r.uniform(0, 6.28)) * g for bf, g in
                       ((2420.0, 0.5), (3510.0, 0.35), (5210.0, 0.25)))
            band = band + ping * bell * 0.25 * np.exp(-t / (decay * 0.35))
        else:
            src = metal * 0.7 + 0.8 * noise(n, r)
            bright = float(p["brightness"])
            band = fft_filter(src, sr, lambda f: (1.0 / np.sqrt(1.0 + ((3000.0 + 2000 * bright) /
                                                                            np.maximum(f, 1.0)) ** 4))
                              * (1.0 + 0.6 * np.exp(-0.5 * ((f - 7500) / 3000) ** 2)))
            env = 0.75 * np.exp(-t / (decay * 0.28)) + 0.25 * np.exp(-t / 0.05)
        att = np.minimum(1.0, t / 0.0015)
        out[:, ch] = band * env * att * 1.3
    return out


def _tom(sr, p, rng, layer):
    decay = float(p["decay"])
    n = int(sr * (decay * 1.4 + 0.02))
    t = _t(n, sr)
    f_end = float(p["pitch"]) * (1 + rng.uniform(-0.005, 0.005))
    f = f_end * (1 + 0.45 * np.exp(-t / 0.05))
    body = sine(f, n, sr) * np.exp(-t / (decay * 0.38))
    body += 0.3 * sine(f * 1.5, n, sr) * np.exp(-t / (decay * 0.18))
    nz = bandpass_noise(n, sr, rng, 300, 4000) * np.exp(-t / 0.012) * 0.5
    out = body + nz
    out *= np.minimum(1.0, t / 0.0008)
    return saturate(out, 0.2)


def _shaker(sr, p, rng, layer):
    decay = float(p["decay"])
    n = int(sr * (decay * 4 + 0.02))
    t = _t(n, sr)
    b = float(p["brightness"])
    nz = bandpass_noise(n, sr, rng, 3500 + 2500 * b, 14000)
    env = np.minimum(1.0, t / 0.012) * np.exp(-np.maximum(t - 0.012, 0) / decay)
    return nz * env * 1.2


def _tamb(sr, p, rng, layer):
    decay = float(p["decay"])
    n = int(sr * (decay * 3.5 + 0.02))
    t = _t(n, sr)
    jingles = sum(sine(f * (1 + rng.uniform(-0.01, 0.01)), n, sr, rng.uniform(0, 6.28))
                  for f in (5300.0, 6650.0, 7900.0, 9450.0))
    nz = bandpass_noise(n, sr, rng, 5000, 14000)
    env = np.minimum(1.0, t / 0.002) * (0.6 * np.exp(-t / decay) + 0.4 * np.exp(-t / (decay * 0.3)))
    return (0.35 * jingles / 4 + nz) * env * 1.1


def _cowbell(sr, p, rng, layer):
    decay = float(p["decay"])
    n = int(sr * (decay * 5 + 0.02))
    t = _t(n, sr)
    tone = float(p["tone"])
    src = osc("square", 540.0 * tone, n, sr) + osc("square", 800.0 * tone, n, sr)
    band = fft_filter(src, sr, lambda f: 1.0 / np.sqrt(1.0 + (2.2 * (f / 2600.0 - 2600.0 / np.maximum(f, 1.0))) ** 2))
    env = np.minimum(1.0, t / 0.001) * (0.7 * np.exp(-t / decay) + 0.3 * np.exp(-t / (decay * 3)))
    return band * env * 0.9


def _snap(sr, p, rng, layer):
    decay = float(p["decay"])
    n = int(sr * (decay * 4 + 0.01))
    t = _t(n, sr)
    nz = bandpass_noise(n, sr, rng, 1800, 7000, slope=1.5)
    env = np.minimum(1.0, t / 0.0006) * np.exp(-t / decay)
    return (nz + 0.4 * sine(2300.0, n, sr) * np.exp(-t / 0.004)) * env * 1.2


_SYNTH = {
    "kick": _kick, "snare": _snare, "rim": _rim, "tom1": _tom, "tom2": _tom, "tom3": _tom,
    "shaker": _shaker, "tamb": _tamb, "cowbell": _cowbell, "snap": _snap,
}


def _freeze(d: dict) -> tuple:
    return tuple(sorted((k, float(v) if isinstance(v, (int, float)) else str(v)) for k, v in d.items()))


@lru_cache(maxsize=1024)
def _one_shot_cached(voice: str, params: tuple, variant: int, layer: int, sr: int) -> np.ndarray:
    p = dict(params)
    seed = (zlib.crc32(voice.encode()) % 10007) * 97 + variant * 7919 + layer * 131 + 12345
    rng = np.random.default_rng(abs(seed))
    lay = (0.55, 0.8, 1.0)[layer]
    if voice == "clap":
        left = _clap(sr, p, np.random.default_rng(abs(seed)), lay)
        right = _clap(sr, p, np.random.default_rng(abs(seed) + 1), lay, stereo_offset=0.0012)
        n = max(len(left), len(right))
        out = np.zeros((n, 2), np.float32)
        out[:len(left), 0] = left
        out[:len(right), 1] = right
    elif voice in ("crash", "ride"):
        out = _crash(sr, p, rng, lay, ride=(voice == "ride"))
    elif voice in ("hat", "ohat"):
        mono = _hat(sr, p, rng, lay, open_=(voice == "ohat"))
        out = np.stack([mono, mono], axis=1)
    else:
        mono = _SYNTH[voice](sr, p, rng, lay)
        out = np.stack([mono, mono], axis=1)
    out = out.astype(np.float32)
    # normalise: every voice peaks at 1.0 on its hardest layer, softer layers a little lower
    out *= (0.82 + 0.18 * lay) / max(float(np.abs(out).max()), 1e-6)
    # click-free tail
    fo = min(len(out) // 2, int(0.006 * sr))
    if fo > 1:
        out[-fo:] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
    out.setflags(write=False)
    return out


def drum_params(voice: str, overrides: dict | None) -> dict:
    p = dict(DRUM_DEFAULTS[voice])
    if overrides:
        p.update({k: v for k, v in overrides.items() if k in p or k in ("bell",)})
    return p


def one_shot(voice: str, params: dict, velocity: float, variant: int, sr: int) -> np.ndarray:
    """Stereo one-shot for a hit (read-only cached array; scale a copy by the velocity)."""
    layer = 0 if velocity < 0.6 else 1 if velocity < 0.88 else 2
    synth_params = {k: v for k, v in params.items() if k not in ("level", "pan")}
    return _one_shot_cached(voice, _freeze(synth_params), variant % N_VARIANTS, layer, sr)


def velocity_gain(velocity: float) -> float:
    return float(max(0.0, velocity)) ** 1.35

