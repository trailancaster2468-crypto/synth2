"""Vectorised DSP building blocks for the arranger (pure numpy).

* band-limited wavetable oscillators (additive tables, one per harmonic budget, so
  nothing aliases): saw, square, pulse 25/12.5 %, triangle, sine, organ
* ADSR envelopes
* a time-varying resonant filter in the STFT domain (per-frame magnitude response:
  low-pass 12/24 dB, high-pass, band-pass) plus static FFT EQ
* saturation, stereo helpers, a synthetic-IR convolution reverb and a tempo-synced
  ping-pong delay

Everything works on float32 arrays; stereo audio is shape ``(n, 2)``.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

TABLE_SIZE = 2048
_HARM_STEPS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160,
               192, 224, 256, 320, 384, 448, 512, 640, 768, 896, 1023)
WAVES = ("saw", "square", "pulse25", "pulse12", "triangle", "sine", "organ", "soft_saw", "hollow")


# ---------------------------------------------------------------------------------------- oscillators

def _harmonics(wave: str, nh: int) -> np.ndarray:
    k = np.arange(1, nh + 1, dtype=np.float64)
    if wave == "saw":
        a = 2.0 / (math.pi * k)
    elif wave == "soft_saw":  # saw with a gentle 6 dB/oct extra roll-off above the 8th harmonic
        a = 2.0 / (math.pi * k) / np.sqrt(1.0 + (k / 8.0) ** 2)
    elif wave == "square":
        a = np.where(k % 2 == 1, 4.0 / (math.pi * k), 0.0)
    elif wave in ("pulse25", "pulse12"):
        w = 0.25 if wave == "pulse25" else 0.125
        a = 2.0 / (math.pi * k) * np.sin(math.pi * k * w) * 1.6
    elif wave == "triangle":
        a = np.where(k % 2 == 1, 8.0 / (math.pi ** 2 * k ** 2), 0.0)
    elif wave == "organ":  # drawbars 8' 4' 2 2/3' 2' 1 3/5' 1'
        a = np.zeros(nh)
        for h, lvl in ((1, 1.0), (2, 0.7), (3, 0.45), (4, 0.4), (5, 0.12), (6, 0.25), (8, 0.18)):
            if h <= nh:
                a[h - 1] = lvl * 0.55
    elif wave == "hollow":  # odd-heavy, clarinet-ish
        a = np.where(k % 2 == 1, 3.0 / (math.pi * k), 0.6 / (math.pi * k ** 2))
    else:  # sine
        a = np.zeros(nh)
        a[0] = 1.0
    return a


@lru_cache(maxsize=512)
def wavetable(wave: str, nh: int) -> tuple[np.ndarray, np.ndarray]:
    """One cycle of ``wave`` with ``nh`` harmonics (TABLE_SIZE samples) and its forward differences
    (for linear interpolation), both float32 and read-only."""
    nh = max(1, min(nh, TABLE_SIZE // 2 - 1))
    spec = np.zeros(TABLE_SIZE // 2 + 1, dtype=np.complex128)
    a = _harmonics(wave, nh)
    # sine phase for saw-like waves (ramp), cosine phase for pulses keeps the waveform peak low
    if wave in ("pulse25", "pulse12"):
        spec[1:nh + 1] = a * TABLE_SIZE / 2
    else:
        spec[1:nh + 1] = -1j * a * TABLE_SIZE / 2
    tab = np.fft.irfft(spec, TABLE_SIZE)
    tab -= tab.mean()
    tab = tab.astype(np.float32)
    diff = (np.roll(tab, -1) - tab).astype(np.float32)
    tab.setflags(write=False)
    diff.setflags(write=False)
    return tab, diff


def harmonic_budget(f_max: float, sr: int) -> int:
    """Largest tabulated harmonic count whose top harmonic stays below ~0.45 * sr."""
    nh = int(0.45 * sr / max(f_max, 1.0))
    best = 1
    for h in _HARM_STEPS:
        if h <= nh:
            best = h
        else:
            break
    return best


_TABLE_BITS = int(math.log2(TABLE_SIZE))
_FRAC_BITS = 32 - _TABLE_BITS
_FRAC_SCALE = np.float32(1.0 / (1 << _FRAC_BITS))
_FRAC_MASK = np.uint32((1 << _FRAC_BITS) - 1)
_ARANGE: list[np.ndarray] = [np.zeros(0, np.uint32)]


def _arange_u32(n: int) -> np.ndarray:
    if len(_ARANGE[0]) < n:
        _ARANGE[0] = np.arange(max(n, 1 << 20), dtype=np.uint32)
    return _ARANGE[0][:n]


def osc(wave: str, freq, n: int, sr: int, phase0: float = 0.0) -> np.ndarray:
    """Band-limited oscillator. ``freq`` is a scalar (Hz) or a per-sample array of length n.

    A 32-bit fixed-point phase accumulator (wrapping integer arithmetic) indexes a band-limited
    wavetable with linear interpolation, all in 32-bit arrays.
    """
    if n <= 0:
        return np.zeros(0, np.float32)
    p0 = np.uint32(int((phase0 % 1.0) * 4294967296.0) & 0xFFFFFFFF)
    if np.isscalar(freq):
        f_max = float(freq)
        inc = np.uint32(int(round(float(freq) / sr * 4294967296.0)) & 0xFFFFFFFF)
        ph = _arange_u32(n) * inc
        ph += p0
    else:
        freq = np.asarray(freq, dtype=np.float64)
        f_max = float(freq.max()) if len(freq) else 1.0
        inc = (np.clip(freq, 0.0, sr * 0.499) * (4294967296.0 / sr)).astype(np.uint32)
        ph = np.cumsum(inc, dtype=np.uint32)
        ph -= inc[0]
        ph += p0
    tab, diff = wavetable(wave, harmonic_budget(f_max, sr))
    i0 = ph >> np.uint32(_FRAC_BITS)
    ph &= _FRAC_MASK
    frac = ph.astype(np.float32)
    frac *= _FRAC_SCALE
    out = diff[i0]
    out *= frac
    out += tab[i0]
    return out


def sine(freq, n: int, sr: int, phase0: float = 0.0) -> np.ndarray:
    """Plain sine (scalar or per-sample frequency), float32."""
    if np.isscalar(freq):
        ph = phase0 + np.arange(n, dtype=np.float64) * (2 * math.pi * float(freq) / sr)
    else:
        ph = phase0 + np.cumsum(np.asarray(freq, dtype=np.float64)) * (2 * math.pi / sr)
    return np.sin(ph).astype(np.float32)


def noise(n: int, rng: np.random.Generator, channels: int = 1) -> np.ndarray:
    if channels == 1:
        return rng.standard_normal(n, dtype=np.float32)
    return rng.standard_normal((n, channels), dtype=np.float32)


# ------------------------------------------------------------------------------------------ envelopes

def adsr(n: int, sr: int, attack: float, decay: float, sustain: float, release: float, gate: float,
         attack_curve: float = 0.0) -> np.ndarray:
    """ADSR envelope of n samples. ``gate`` (seconds) is when the release starts.

    attack is linear (``attack_curve`` > 0 bends it exponential-ish), decay and release are
    exponential (``decay`` = time to fall ~95 % of the way to the sustain level, ``release`` =
    time to fall ~95 % to silence).
    """
    t = np.arange(n, dtype=np.float32) / sr
    a = max(attack, 0.0005)
    env = np.empty(n, np.float32)
    att = t < a
    x = t[att] / a
    env[att] = x if attack_curve <= 0 else (1 - np.exp(-x * (1 + 4 * attack_curve))) / (1 - math.exp(-(1 + 4 * attack_curve)))
    d = max(decay, 0.001)
    rest = ~att
    env[rest] = sustain + (1.0 - sustain) * np.exp(-(t[rest] - a) * (3.0 / d))
    if gate < n / sr:
        g = max(gate, 0.0)
        # level reached at the gate
        if g < a:
            lvl = g / a
        else:
            lvl = sustain + (1.0 - sustain) * math.exp(-(g - a) * (3.0 / d))
        r = max(release, 0.003)
        rel = t >= g
        env[rel] = lvl * np.exp(-(t[rel] - g) * (3.0 / r))
    return env


def fade_edges(x: np.ndarray, sr: int, fade_in: float = 0.0015, fade_out: float = 0.004) -> np.ndarray:
    """Short raised-cosine fades at both ends (in place) so note boundaries never click."""
    n = len(x)
    fi = min(n // 2, int(fade_in * sr))
    fo = min(n // 2, int(fade_out * sr))
    if fi > 1:
        w = (0.5 - 0.5 * np.cos(np.linspace(0, math.pi, fi, dtype=np.float32)))
        x[:fi] *= w if x.ndim == 1 else w[:, None]
    if fo > 1:
        w = (0.5 + 0.5 * np.cos(np.linspace(0, math.pi, fo, dtype=np.float32)))
        x[n - fo:] *= w if x.ndim == 1 else w[:, None]
    return x


# -------------------------------------------------------------------------------------------- filters

def stft_size(sr: int) -> int:
    return int(2 ** round(math.log2(sr * 0.0232)))


@lru_cache(maxsize=16)
def _stft_window(n_fft: int) -> np.ndarray:
    return (0.5 - 0.5 * np.cos(2 * math.pi * np.arange(n_fft) / n_fft)).astype(np.float32)


def filter_response(freqs: np.ndarray, cutoff: np.ndarray, resonance: float, ftype: str = "lp",
                    slope: int = 24) -> np.ndarray:
    """Magnitude response (frames x bins). cutoff: per-frame Hz (shape (F, 1)). resonance 0..1."""
    q = 0.707 + resonance ** 1.5 * 9.0
    r = freqs[None, :] / cutoff
    if ftype == "hp":
        r = 1.0 / np.maximum(r, 1e-6)
    if ftype == "bp":
        qq = 0.7 + resonance * 6.0
        rr = np.maximum(r, 1e-6)
        return 1.0 / np.sqrt(1.0 + (qq * (rr - 1.0 / rr)) ** 2)
    r2 = r * r
    h = 1.0 / np.sqrt((1.0 - r2) ** 2 + r2 / (q * q))
    if slope >= 24:
        h = h * (1.0 / np.sqrt((1.0 - r2) ** 2 + r2 / 0.36))
    return h


def eq_curve(freqs: np.ndarray, eq: dict | None) -> np.ndarray | None:
    """Static magnitude curve from an EQ dict: low_cut, high_cut (Hz, 12 dB/oct), and peaking
    ``bands``: [{hz, db, q}] plus ``low_shelf``/``high_shelf`` {hz, db}. None when flat."""
    if not eq:
        return None
    f = np.maximum(freqs, 1.0)
    h = np.ones_like(f)
    if eq.get("low_cut"):
        r = f / float(eq["low_cut"])
        h *= r * r / np.sqrt((1 - r * r) ** 2 + (r / 0.707) ** 2)
    if eq.get("high_cut"):
        r = f / float(eq["high_cut"])
        h *= 1.0 / np.sqrt((1 - r * r) ** 2 + (r / 0.707) ** 2)
    for b in eq.get("bands", []) or []:
        hz, db, q = float(b.get("hz", 1000)), float(b.get("db", 0)), float(b.get("q", 1.0))
        if db:
            lo = np.log2(f / hz)
            bw = 1.0 / max(q, 0.1)
            h *= 10 ** (db / 20 * np.exp(-0.5 * (lo / (bw * 0.6)) ** 2))
    for key, sign in (("low_shelf", -1), ("high_shelf", 1)):
        sh = eq.get(key)
        if sh and sh.get("db"):
            x = np.log2(f / float(sh.get("hz", 200 if sign < 0 else 5000)))
            s = 1.0 / (1.0 + np.exp(-sign * x * 3.0))
            h *= 10 ** (float(sh["db"]) / 20 * s)
    return h


def stft_filter(x: np.ndarray, sr: int, mask_fn, frame0: int = 0) -> np.ndarray:
    """Apply a time-varying magnitude mask: ``mask_fn(frame_times_s, freqs) -> (F, bins)``.

    x is mono (n,) or stereo (n, 2). Frames are centred on multiples of hop from the start of x
    (offset by ``frame0`` samples for the frame time stamps). Hann analysis+synthesis, 75 % overlap.
    """
    n = len(x)
    if n == 0:
        return x
    n_fft = stft_size(sr)
    hop = n_fft // 4
    win = _stft_window(n_fft)
    pad = n_fft
    n_frames = (n + 2 * pad - n_fft) // hop + 1
    total = (n_frames - 1) * hop + n_fft
    shape = (total,) + x.shape[1:]
    xp = np.zeros(shape, np.float32)
    xp[pad:pad + n] = x
    frames = np.lib.stride_tricks.sliding_window_view(xp, n_fft, axis=0)[::hop]  # (F, ch?, n_fft)
    spec = np.fft.rfft(frames * win, axis=-1)  # (F, [2,] bins)
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    times = (np.arange(n_frames) * hop + n_fft // 2 - pad + frame0) / sr
    mask = mask_fn(times, freqs).astype(np.float32)
    if x.ndim == 2:
        spec *= mask[:, None, :]
    else:
        spec *= mask
    y = np.fft.irfft(spec, n_fft, axis=-1).astype(np.float32) * win  # (F, [2,] n_fft)
    out = np.zeros(shape, np.float32)
    # overlap-add: frames k, k+4, k+8... do not overlap each other
    for k in range(4):
        sub = y[k::4]
        m = len(sub)
        if m == 0:
            continue
        start = k * hop
        if x.ndim == 2:
            block = np.transpose(sub, (0, 2, 1)).reshape(m * n_fft, 2)
        else:
            block = sub.reshape(m * n_fft)
        end = min(start + m * n_fft, total)
        out[start:end] += block[:end - start]
    out *= 1.0 / 1.5  # sum of hann^2 at 75 % overlap
    return out[pad:pad + n]


def fft_filter(x: np.ndarray, sr: int, curve_fn) -> np.ndarray:
    """Static zero-phase filter over the whole signal: ``curve_fn(freqs) -> gains``."""
    n = len(x)
    if n == 0:
        return x
    pad = min(int(0.25 * sr), max(1024, n))
    m = 1 << int(math.ceil(math.log2(n + pad)))
    spec = np.fft.rfft(x, m, axis=0)
    g = curve_fn(np.fft.rfftfreq(m, 1.0 / sr))
    spec *= g[:, None] if x.ndim == 2 else g
    return np.fft.irfft(spec, m, axis=0)[:n].astype(np.float32)


def bandpass_noise(n: int, sr: int, rng: np.random.Generator, lo: float, hi: float, channels: int = 1,
                   slope: float = 2.0) -> np.ndarray:
    """Noise shaped to a band (smooth edges); zero-phase FFT filter."""
    x = noise(n, rng, channels)

    def curve(f):
        f = np.maximum(f, 1.0)
        g = 1.0 / np.sqrt(1.0 + (lo / f) ** (2 * slope)) if lo > 0 else np.ones_like(f)
        if hi:
            g = g / np.sqrt(1.0 + (f / hi) ** (2 * slope))
        return g

    return fft_filter(x, sr, curve)


# ---------------------------------------------------------------------------------- shaping / stereo

def saturate(x: np.ndarray, drive: float) -> np.ndarray:
    """tanh saturation, level-compensated. drive 0..1 (0 = clean)."""
    if drive <= 0:
        return x
    g = 1.0 + drive * 14.0
    return (np.tanh(x * g) / math.tanh(g) * (0.35 + 0.65 / (1 + drive * 1.5))).astype(np.float32)


def pan_gains(pan: float) -> tuple[float, float]:
    """Constant-power pan gains for pan -1..1 (centre = 0.707 each, so normalised to 1 at centre)."""
    a = (max(-1.0, min(1.0, pan)) + 1) * math.pi / 4
    return math.cos(a) * math.sqrt(2), math.sin(a) * math.sqrt(2)


def to_stereo(x: np.ndarray, pan: float = 0.0) -> np.ndarray:
    gl, gr = pan_gains(pan)
    return np.stack([x * gl, x * gr], axis=1).astype(np.float32)


def set_width(x: np.ndarray, width: float) -> np.ndarray:
    if x.ndim != 2 or abs(width - 1.0) < 1e-6:
        return x
    mid = 0.5 * (x[:, 0] + x[:, 1])
    side = 0.5 * (x[:, 0] - x[:, 1]) * width
    return np.stack([mid + side, mid - side], axis=1).astype(np.float32)


def add_at(buf: np.ndarray, sig: np.ndarray, start: int):
    """buf[start:start+len(sig)] += sig, clipped to the buffer (start may be negative)."""
    n = len(sig)
    a, b = start, start + n
    if b <= 0 or a >= len(buf):
        return
    s0 = max(0, -a)
    s1 = n - max(0, b - len(buf))
    buf[max(a, 0):max(a, 0) + (s1 - s0)] += sig[s0:s1]


# ------------------------------------------------------------------------------------- time effects

@lru_cache(maxsize=32)
def reverb_ir(sr: int, decay: float, size: float = 0.5, damp: float = 0.5, seed: int = 7,
              low_cut: float = 0.0) -> np.ndarray:
    """Synthetic stereo room/plate impulse response (decorrelated noise, frequency-dependent decay)."""
    rng = np.random.default_rng(seed)
    n = int(sr * min(max(decay, 0.1), 6.0) * 1.1)
    t = np.arange(n, dtype=np.float32) / sr
    pre = int(sr * (0.004 + 0.02 * size))
    ir = np.zeros((n, 2), np.float32)
    # three bands with their own decay: highs die first (air absorption, damping)
    lo = bandpass_noise(n, sr, rng, 0, 700, 2)
    mid = bandpass_noise(n, sr, rng, 700, 4000, 2)
    hi = bandpass_noise(n, sr, rng, 4000, 0, 2)
    k = 6.9 / max(decay, 0.05)  # -60 dB at `decay` seconds
    for band, mult in ((lo, 0.9), (mid, 1.0), (hi, 1.0 + 2.5 * damp)):
        ir += band * np.exp(-t * k * mult)[:, None]
    # soft onset (diffusion build-up) and pre-delay
    build = 1 - np.exp(-t / (0.004 + 0.02 * size))
    ir *= build[:, None]
    ir = np.concatenate([np.zeros((pre, 2), np.float32), ir])[:n]
    # a few early reflections
    for i, (d, g) in enumerate(((0.011, 0.5), (0.017, 0.4), (0.023, 0.33), (0.031, 0.25))):
        j = pre + int(d * sr * (0.5 + size))
        if j < n:
            ir[j, i % 2] += g * 3.0
    if low_cut:  # keep the low end out of the tail (folded into the IR: no extra filtering per signal)
        ir = fft_filter(ir, sr, lambda f: 1.0 / np.sqrt(1.0 + (low_cut / np.maximum(f, 1.0)) ** 4))
    ir /= np.sqrt((ir ** 2).sum(axis=0, keepdims=True) + 1e-9)
    ir.setflags(write=False)
    return ir


def convolve_stereo(x: np.ndarray, ir: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """Overlap-add FFT convolution of stereo x (n, 2) with stereo ir (m, 2), output length n."""
    n = len(x)
    m = len(ir)
    if n == 0:
        return x.copy()
    nfft = 1 << int(math.ceil(math.log2(block + m - 1)))
    H = np.fft.rfft(ir, nfft, axis=0)
    out = np.zeros((n + m, 2), np.float32)
    for s in range(0, n, block):
        seg = x[s:s + block]
        if not np.any(seg):
            continue
        y = np.fft.irfft(np.fft.rfft(seg, nfft, axis=0) * H, nfft, axis=0)
        L = len(seg) + m - 1
        out[s:s + L] += y[:L].astype(np.float32)
    return out[:n]


def apply_reverb(x: np.ndarray, sr: int, amount: float, decay: float = 1.2, size: float = 0.5,
                 damp: float = 0.5, low_cut: float = 250.0) -> np.ndarray:
    """Mix a synthetic reverb into stereo x. amount 0..1 is the wet level (0.2 = subtle)."""
    if amount <= 0 or not np.any(x):
        return x
    ir = reverb_ir(sr, round(decay, 3), round(size, 3), round(damp, 3), 7, float(low_cut or 0.0))
    wet = convolve_stereo(x, ir)
    return (x + wet * amount * 0.6).astype(np.float32)


def apply_delay(x: np.ndarray, sr: int, delay_s: float, feedback: float = 0.35, mix: float = 0.25,
                pingpong: bool = True, tone: float = 3500.0) -> np.ndarray:
    """Tempo-synced echo built from shifted copies (feedback^k), each a little darker, ping-ponged."""
    if mix <= 0 or delay_s <= 0 or not np.any(x):
        return x
    d = int(round(delay_s * sr))
    if d <= 0:
        return x
    mono = fft_filter(x.mean(axis=1), sr, lambda f: 1.0 / np.sqrt(1.0 + (f / tone) ** 2)
                      / np.sqrt(1.0 + (180.0 / np.maximum(f, 1.0)) ** 2))
    out = x.copy()
    g = 1.0
    k = 1
    while g > 0.02 and k < 24:
        g = feedback ** (k - 1)
        if g <= 0.02:
            break
        tap = mono * (g * mix)
        if pingpong:
            ch = (k - 1) % 2
            add_at(out[:, ch], tap, d * k)
            add_at(out[:, 1 - ch], tap * 0.25, d * k)
        else:
            add_at(out[:, 0], tap, d * k)
            add_at(out[:, 1], tap, d * k)
        k += 1
    return out


def db(x: float) -> float:
    return 20 * math.log10(max(x, 1e-12))


def undb(x: float) -> float:
    return 10 ** (x / 20)
