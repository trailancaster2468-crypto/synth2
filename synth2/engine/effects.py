"""Mixing and effects: pan, algorithmic convolution reverb, EQ helpers, limiter."""
from __future__ import annotations

import numpy as np


def pan_stereo(mono: np.ndarray, pan: float, gain_db: float = 0.0) -> np.ndarray:
    """Constant-power pan; ``pan`` in [-1, 1]."""
    g = 10 ** (gain_db / 20.0)
    theta = (np.clip(pan, -1, 1) + 1) * np.pi / 4
    # unity gain in the centre, +3 dB on the hard side
    return np.stack([mono * np.cos(theta), mono * np.sin(theta)], axis=1) * (g * np.sqrt(2))


def reverb_ir(sr: int, room: float, seed: int = 1) -> np.ndarray:
    """Synthetic stereo hall impulse response (n, 2) with frequency-dependent decay."""
    room = float(np.clip(room, 0.0, 1.0))
    rt60 = 0.35 + 2.4 * room
    length = int(sr * min(rt60 * 1.2 + 0.1, 4.0))
    rng = np.random.default_rng(seed)
    t = np.arange(length) / sr
    out = np.zeros((length, 2))
    nfft = 1 << int(np.ceil(np.log2(length)))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    # bands with their own decay (highs decay faster)
    bands = [(0, 400, 1.15), (400, 2000, 1.0), (2000, 6000, 0.7), (6000, sr / 2 + 1, 0.45)]
    for ch in range(2):
        acc = np.zeros(length)
        for lo, hi, k in bands:
            noise = rng.standard_normal(nfft)
            spec = np.fft.rfft(noise)
            spec[(freqs < lo) | (freqs >= hi)] = 0
            band = np.fft.irfft(spec, n=nfft)[:length]
            acc += band * np.exp(-6.91 * t / (rt60 * k))
        # soft onset (diffusion build-up) and pre-delay
        pre = int(0.018 * sr)
        acc *= 1 - np.exp(-t / 0.012)
        acc = np.concatenate([np.zeros(pre), acc])[:length]
        # early reflections
        for d, g in [(0.011, 0.5), (0.019, 0.35), (0.027, 0.3), (0.037, 0.22), (0.051, 0.16)]:
            dd = int((d + 0.003 * ch) * sr)
            if dd < length:
                acc[dd] += g * (1 if (ch + int(d * 1000)) % 2 else -1) * 3.0
        out[:, ch] = acc
    out /= np.sqrt(np.sum(out ** 2) / 2.0) + 1e-12
    return out


def fft_convolve(x: np.ndarray, ir: np.ndarray, block: int = 1 << 16) -> np.ndarray:
    """Overlap-add FFT convolution of a mono signal with a mono IR."""
    m = len(ir)
    nfft = 1 << int(np.ceil(np.log2(block + m - 1)))
    H = np.fft.rfft(ir, nfft)
    y = np.zeros(len(x) + m - 1)
    for s in range(0, len(x), block):
        seg = x[s:s + block]
        yb = np.fft.irfft(np.fft.rfft(seg, nfft) * H, nfft)[: len(seg) + m - 1]
        y[s:s + len(yb)] += yb
    return y


def apply_reverb(bus: np.ndarray, sr: int, room: float) -> np.ndarray:
    """Return the wet signal (n + tail, 2) for a stereo send bus."""
    ir = reverb_ir(sr, room)
    mono_in = bus.mean(axis=1)
    wet = np.stack([fft_convolve(mono_in, ir[:, 0]), fft_convolve(mono_in, ir[:, 1])], axis=1)
    return wet * 0.6


def high_pass(x: np.ndarray, sr: int, fc: float = 70.0) -> np.ndarray:
    """Zero-phase spectral high-pass (removes rumble / DC)."""
    n = x.shape[0]
    nfft = 1 << int(np.ceil(np.log2(max(n, 2))))
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    resp = 1.0 / np.sqrt(1.0 + (fc / np.maximum(freqs, 1e-3)) ** 4)
    X = np.fft.rfft(x, nfft, axis=0)
    if x.ndim == 2:
        X *= resp[:, None]
    else:
        X *= resp
    return np.fft.irfft(X, nfft, axis=0)[:n]


def limiter(x: np.ndarray, sr: int, threshold_db: float = -1.0, release: float = 0.08) -> np.ndarray:
    """Peak limiter with smoothed gain reduction plus a soft clipper for safety."""
    thr = 10 ** (threshold_db / 20.0)
    peak = np.max(np.abs(x), axis=1) if x.ndim == 2 else np.abs(x)
    if peak.max() <= thr:
        return x
    hop = max(1, int(sr * 0.002))
    nb = int(np.ceil(len(peak) / hop))
    padded = np.pad(peak, (0, nb * hop - len(peak)))
    blocks = padded.reshape(nb, hop).max(axis=1)
    # look-ahead of one block
    blocks = np.maximum(blocks, np.concatenate([blocks[1:], blocks[-1:]]))
    g = np.minimum(1.0, thr / np.maximum(blocks, 1e-12))
    coef = np.exp(-hop / (release * sr))
    sm = np.empty_like(g)
    cur = 1.0
    for i, gi in enumerate(g):
        cur = gi if gi < cur else gi + (cur - gi) * coef
        sm[i] = cur
    gs = np.interp(np.arange(len(peak)), np.arange(nb) * hop + hop / 2, sm)
    y = x * (gs[:, None] if x.ndim == 2 else gs)
    knee = 0.9 * thr
    a = np.abs(y)
    over = a > knee
    if over.any():
        y = y.copy()
        y[over] = np.sign(y[over]) * (knee + (thr - knee) * np.tanh((a[over] - knee) / (thr - knee)))
    return y
