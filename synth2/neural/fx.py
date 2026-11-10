"""Character effects for the neural engine's voices: ring modulation, comb, bit crush, tremolo,
chorus, drive and a band-pass, and presets made of them (``alien``, ``robot``, ``radio``,
``crush``, ``double``). numpy only.

An effect chain is given as a preset name, ``"name:amount"`` (amount 0..1), a dict with a
``type``, or a list of these::

    y = apply_fx(x, 44100, "alien:0.6")
    y = apply_fx(x, 44100, [{"type": "band", "lo_hz": 300, "hi_hz": 3400}, "double:0.5"])

Every effect is a function of absolute time: ``x[0]`` is the sample at ``t0`` seconds,
oscillators and sample-and-hold grids are counted from time 0, and no effect looks ahead or
remembers more than :func:`fx_tail` seconds. A piece of a signal that starts ``fx_tail``
seconds early therefore gets exactly the samples a whole render gets::

    tail = fx_tail(spec)
    a = int(round((t - tail) * sr))
    piece = apply_fx(x[a:], sr, spec, t0=a / sr)      # piece[n - a] == apply_fx(x, sr, spec)[n] from t on

Levels: every preset keeps the level of a sung voice within about 1 dB. ``drive`` and
``crush`` depend on the level of their input; they are tuned for a voice that peaks between
0.3 and 1.0.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

# type -> parameter -> (default, lowest, highest); values outside the range are clipped
FX_TYPES: dict[str, dict[str, tuple[float, float, float]]] = {
    # x * ((1 - mix) + mix * gain * sin(2 pi freq t))
    "ring": {"freq": (60.0, 0.1, 20000.0), "mix": (0.5, 0.0, 1.0), "gain": (math.sqrt(2.0), 0.0, 4.0)},
    # x + mix * (x delayed by delay_ms, recirculated with feedback); parallel 1: the delayed copy is
    # taken from the input of the chain instead of from the effect before
    "comb": {"delay_ms": (5.0, 0.05, 100.0), "feedback": (0.0, -0.9, 0.9), "mix": (0.5, 0.0, 1.0),
             "parallel": (0.0, 0.0, 1.0)},
    # sample-and-hold at rate_hz and rounding to bits; mu > 0 rounds on a mu-law scale (255: the
    # telephone's), which keeps the grit in proportion to the voice at any level
    "crush": {"bits": (8.0, 2.0, 24.0), "rate_hz": (11025.0, 100.0, 384000.0), "mix": (1.0, 0.0, 1.0),
              "mu": (0.0, 0.0, 65535.0)},
    # a sine gain at rate Hz that dips by depth; the mean power is kept
    "tremolo": {"rate": (6.0, 0.01, 1000.0), "depth": (0.5, 0.0, 1.0)},
    # voices copies delayed by delay_ms +- depth_ms, each wandering at about rate Hz
    "chorus": {"voices": (2.0, 1.0, 8.0), "delay_ms": (20.0, 1.0, 100.0), "depth_ms": (2.0, 0.0, 20.0),
               "rate": (0.5, 0.01, 20.0), "mix": (0.5, 0.0, 1.0)},
    # tanh soft clipping after gain_db; quiet passages keep their level
    "drive": {"gain_db": (12.0, 0.0, 40.0), "mix": (1.0, 0.0, 1.0)},
    # 12 dB/octave high-pass at lo_hz (0: none) and 24 dB/octave low-pass at hi_hz
    "band": {"lo_hz": (300.0, 0.0, 20000.0), "hi_hz": (3400.0, 100.0, 200000.0), "mix": (1.0, 0.0, 1.0)},
    # a level change
    "gain": {"db": (0.0, -40.0, 20.0)},
}
# the parameter an amount scales
AMOUNT_PARAM = {"ring": "mix", "comb": "mix", "crush": "mix", "tremolo": "depth", "chorus": "mix", "drive": "mix",
                "band": "mix", "gain": "db"}

FX_PRESETS: dict[str, list[dict]] = {
    "none": [],
    # 62 Hz ring modulation and a 4.5 ms comb beside it
    "alien": [{"type": "ring", "freq": 62.0, "mix": 0.27, "gain": 1.6},
              {"type": "comb", "delay_ms": 4.5, "feedback": 0.0, "mix": 0.21, "parallel": 1.0},
              {"type": "gain", "db": 1.5}],
    # a resonating comb (a metal tube at 167 Hz) and a fast flutter
    "robot": [{"type": "comb", "delay_ms": 6.0, "feedback": 0.5, "mix": 0.35},
              {"type": "tremolo", "rate": 35.0, "depth": 0.25},
              {"type": "gain", "db": -1.4}],
    # a small loudspeaker's band and a little saturation
    "radio": [{"type": "band", "lo_hz": 300.0, "hi_hz": 4000.0},
              {"type": "drive", "gain_db": 9.0},
              {"type": "gain", "db": 1.0}],
    # an early sampler: 6 bits on a mu-law scale at 22 kHz
    "crush": [{"type": "crush", "bits": 6.0, "rate_hz": 22050.0, "mu": 255.0}],
    # two more singers a few milliseconds behind
    "double": [{"type": "chorus", "voices": 2.0, "delay_ms": 18.0, "depth_ms": 1.5, "rate": 0.6, "mix": 0.5}],
}

COMB_FLOOR = 1e-5   # a comb's recirculation is cut off below this gain
BAND_FLOOR = 1e-6   # the band-pass forgets what has decayed to this fraction (see _band_memory)
MIN_RATE = 8000     # lowest sample rate fx_tail allows for when it rounds delays up to samples


# ---- reading effect chains ---------------------------------------------------------------------------
def _amount(raw) -> float:
    try:
        a = float(raw)
    except (TypeError, ValueError):
        a = math.nan
    if not math.isfinite(a):
        raise ValueError(f"effect amount {raw!r} is not a number; use name:amount with an amount from 0 to 1")
    return min(1.0, max(0.0, a))


def _preset(name, amount: float) -> list[dict]:
    key = str(name).strip().lower()
    if key not in FX_PRESETS:
        raise ValueError(f"unknown effect {str(name).strip()!r}; valid effects: {', '.join(sorted(FX_PRESETS))}")
    return [e for item in FX_PRESETS[key] for e in _effect(item, amount)]


def _effect(item: dict, amount: float = 1.0) -> list[dict]:
    """One effect dict with every parameter filled in and its amount applied ([] for amount 0)."""
    if "preset" in item and "type" not in item:
        extra = sorted(str(k) for k in item if k not in ("preset", "amount"))
        if extra:
            raise ValueError(f"an effect given by 'preset' takes 'amount' only, not {', '.join(extra)}")
        return _preset(item["preset"], amount * _amount(item.get("amount", 1.0)))
    kind = str(item.get("type", "")).strip().lower()
    if kind not in FX_TYPES:
        raise ValueError(f"unknown effect type {item.get('type')!r}; valid types: {', '.join(sorted(FX_TYPES))}")
    params = FX_TYPES[kind]
    out: dict = {"type": kind}
    for key in item:
        if key not in params and key not in ("type", "amount"):
            raise ValueError(f"effect type {kind!r} has no parameter {key!r}; its parameters: "
                             f"{', '.join(sorted(params))}")
    for key, (default, lo, hi) in params.items():
        raw = item.get(key, default)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            v = math.nan
        if not math.isfinite(v) or isinstance(raw, bool):
            raise ValueError(f"parameter {key!r} of effect type {kind!r} must be a number, got {raw!r}")
        out[key] = min(hi, max(lo, v))
    if kind == "band" and out["lo_hz"] > 0:
        out["lo_hz"] = max(20.0, out["lo_hz"])
        if out["hi_hz"] <= out["lo_hz"]:
            raise ValueError(f"effect type 'band' needs hi_hz above lo_hz, got lo_hz={out['lo_hz']:g} and "
                             f"hi_hz={out['hi_hz']:g}")
    amount *= _amount(item.get("amount", 1.0))
    out[AMOUNT_PARAM[kind]] *= amount
    return [out] if amount > 0 else []


def parse_fx(spec) -> list[dict]:
    """The effect chain ``spec`` as a list of effect dicts with every parameter filled in.

    ``spec`` is None, ``""`` or ``"none"`` (no effect), a preset name, ``"name:amount"``, a dict
    ``{"type": ..., parameters..., "amount": ...}`` or ``{"preset": name, "amount": ...}``, or a
    list of these. An amount (0..1, clipped) scales the effect's wet mix or depth; amount 0
    removes it. Raises :class:`ValueError` naming the valid effects, types or parameters."""
    if spec is None:
        return []
    if isinstance(spec, str):
        name, sep, amount = spec.partition(":")
        if not name.strip() and not sep:
            return []
        return _preset(name, _amount(amount) if sep else 1.0)
    if isinstance(spec, dict):
        return _effect(spec)
    if isinstance(spec, (list, tuple)):
        return [e for item in spec for e in parse_fx(item)]
    raise ValueError("an effect is a preset name, 'name:amount', a dict with a 'type' or a list of these, "
                     f"got {spec!r}")


# ---- how much of the past each effect uses --------------------------------------------------------------
def _comb_terms(feedback: float) -> int:
    """Passes of the delay line that are kept: the first, and those recirculated above COMB_FLOOR."""
    fb = abs(feedback)
    return 1 if fb < COMB_FLOOR else 1 + int(math.ceil(math.log(COMB_FLOOR) / math.log(fb)))


def _band_memory(lo: float, hi: float) -> float:
    """Seconds after which the band-pass's response to an impulse is below BAND_FLOOR of its
    peak: 14.5 time constants of the high-pass's poles plus those of the two low-pass sections."""
    return (3.3 / lo if lo > 0 else 0.0) + 6.0 / hi


def _memory(e: dict) -> float:
    """Seconds of input before a sample that effect ``e`` uses, rounded up for any sample rate
    of at least MIN_RATE."""
    kind = e["type"]
    if kind == "comb":
        return _comb_terms(e["feedback"]) * (e["delay_ms"] * 1e-3 + 0.5 / MIN_RATE)
    if kind == "crush":
        return 1.0 / e["rate_hz"] + 2.0 / MIN_RATE
    if kind == "chorus":
        return (e["delay_ms"] + e["depth_ms"]) * 1e-3 + 3.0 / MIN_RATE
    if kind == "band":
        return _band_memory(e["lo_hz"], e["hi_hz"]) + 1.0 / MIN_RATE
    return 0.0


def fx_tail(spec) -> float:
    """Seconds of pre-roll the chain needs: a piece of a signal processed from this long before
    a point equals, from that point on, the same samples of the whole signal processed at once.
    0 for chains without memory."""
    return float(sum(_memory(e) for e in parse_fx(spec)))


# ---- the effects ------------------------------------------------------------------------------------
# Each takes the signal so far (float64), the input of the chain, the absolute index of the first
# sample, the sample rate, its parameters and the seed. Per sample they use only elementwise
# operations on absolute quantities, so a sample's value does not depend on where the buffer starts.
def _index(n0: int, n: int) -> np.ndarray:
    return n0 + np.arange(n, dtype=np.int64)


def _ring(y, src, n0, sr, p, seed):
    freq = min(p["freq"], 0.49 * sr)
    carrier = np.sin((2.0 * np.pi * freq / sr) * _index(n0, len(y)))
    return y * ((1.0 - p["mix"]) + (p["mix"] * p["gain"]) * carrier)


def _comb(y, src, n0, sr, p, seed):
    delay = max(1, int(round(p["delay_ms"] * 1e-3 * sr)))
    tap = src if p["parallel"] >= 0.5 else y
    wet = np.zeros(len(y))
    g = 1.0
    for k in range(1, _comb_terms(p["feedback"]) + 1):
        shift = k * delay
        if shift >= len(y):
            break
        wet[shift:] += g * tap[:len(y) - shift]
        g *= p["feedback"]
    return y + p["mix"] * wet


def _crush(y, src, n0, sr, p, seed):
    n = len(y)
    rate = min(p["rate_hz"], float(sr))
    cell = np.floor(_index(n0 - 1, n + 1) * (rate / sr))  # the hold cell of each sample and of the one before
    first = np.maximum.accumulate(np.where(cell[1:] != cell[:-1], np.arange(n), -1))  # where each cell began
    held = np.where(first >= 0, y[np.maximum(first, 0)], 0.0)  # a cell that began before the buffer holds 0
    steps = 2.0 ** (p["bits"] - 1.0)
    mu = p["mu"]
    if mu >= 1.0:
        scale = math.log1p(mu)
        code = np.round(np.sign(held) * np.log1p(mu * np.abs(held)) * (steps / scale)) / steps
        wet = np.sign(code) * np.expm1(np.abs(code) * scale) / mu
    else:
        wet = np.round(held * steps) / steps
    return (1.0 - p["mix"]) * y + p["mix"] * wet


def _tremolo(y, src, n0, sr, p, seed):
    d = p["depth"]
    lfo = np.sin((2.0 * np.pi * p["rate"] / sr) * _index(n0, len(y)))
    norm = math.sqrt((1.0 - d / 2.0) ** 2 + d * d / 8.0)  # root mean square of the gain
    return y * (((1.0 - d / 2.0) + (d / 2.0) * lfo) / norm)


def _chorus(y, src, n0, sr, p, seed):
    n = len(y)
    voices = int(round(p["voices"]))
    base = max(2.0, p["delay_ms"] * 1e-3 * sr)
    depth = min(p["depth_ms"] * 1e-3 * sr, base - 1.0)  # the delay stays at least one sample
    pad = int(math.ceil(base + depth)) + 3
    ypad = np.concatenate([np.zeros(pad), y])
    k = np.arange(n, dtype=np.int64) + pad
    idx = _index(n0, n)
    start = float(np.random.default_rng(abs(int(seed))).uniform())
    wet = np.zeros(n)
    for v in range(voices):
        rate = p["rate"] * (1.0 + 0.17 * v)
        delay = base + depth * np.sin((2.0 * np.pi * rate / sr) * idx + 2.0 * np.pi * (start + v / voices))
        whole = np.floor(delay)
        t = 1.0 - (delay - whole)  # the tap lies t past sample i
        i = k - whole.astype(np.int64) - 1
        p0, p1, p2, p3 = ypad[i - 1], ypad[i], ypad[i + 1], ypad[i + 2]
        # Catmull-Rom interpolation between p1 and p2
        wet += p1 + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3 + t * (3.0 * (p1 - p2) + p3 - p0)))
    mix = p["mix"]
    return (y + (mix / voices) * wet) / math.sqrt(1.0 + mix * mix / voices)


def _drive(y, src, n0, sr, p, seed):
    g = 10.0 ** (p["gain_db"] / 20.0)
    return (1.0 - p["mix"]) * y + (p["mix"] / g) * np.tanh(g * y)


def _biquad(kind: str, f: float) -> tuple[float, float, float, float, float]:
    """A second-order Butterworth section at ``f`` (cycles per sample): ``(b0, b1, b2, a1, a2)``."""
    w = 2.0 * math.pi * f
    c, alpha = math.cos(w), math.sin(w) / math.sqrt(2.0)
    b1 = (1.0 - c) if kind == "lp" else -(1.0 + c)
    a0 = 1.0 + alpha
    return abs(b1) / 2.0 / a0, b1 / a0, abs(b1) / 2.0 / a0, -2.0 * c / a0, (1.0 - alpha) / a0


@lru_cache(maxsize=16)
def _band_kernel(sr: int, lo: float, hi: float) -> np.ndarray:
    """The band-pass's response to an impulse, as long as ``_band_memory``."""
    sections = [_biquad("hp", lo / sr)] if lo > 0 else []
    if hi < 0.45 * sr:
        sections += [_biquad("lp", hi / sr)] * 2
    h = [1.0] + [0.0] * (max(1, int(math.ceil(_band_memory(lo, hi) * sr))) - 1)
    for b0, b1, b2, a1, a2 in sections:
        x1 = x2 = y1 = y2 = 0.0
        for i, x0 in enumerate(h):
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            x2, x1, y2, y1 = x1, x0, y1, y0
            h[i] = y0
    kernel = np.array(h if sections else [1.0])
    kernel.setflags(write=False)
    return kernel


def _band(y, src, n0, sr, p, seed):
    h = _band_kernel(int(sr), p["lo_hz"], p["hi_hz"])
    wet = np.convolve(np.concatenate([np.zeros(len(h) - 1), y]), h, "valid")
    return (1.0 - p["mix"]) * y + p["mix"] * wet


def _gain(y, src, n0, sr, p, seed):
    return y * 10.0 ** (p["db"] / 20.0)


_RUN = {"ring": _ring, "comb": _comb, "crush": _crush, "tremolo": _tremolo, "chorus": _chorus, "drive": _drive,
        "band": _band, "gain": _gain}


def apply_fx(x, sr: int, spec, *, t0: float = 0.0, seed: int = 0) -> np.ndarray:
    """``x`` (mono, ``x[0]`` at ``t0`` seconds) through the effect chain ``spec``. Returns float32
    of the same length. ``seed`` sets where the chorus voices start their wandering."""
    chain = parse_fx(spec)
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"apply_fx expects mono audio, got an array of shape {x.shape}")
    if not chain or not len(x):
        return x.astype(np.float32)
    n0 = int(round(float(t0) * sr))
    src = x.astype(np.float64)
    y = src
    for e in chain:
        y = _RUN[e["type"]](y, src, n0, sr, e, seed)
    return y.astype(np.float32)
