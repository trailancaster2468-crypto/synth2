"""Sample-rate conversion for the neural engine: a Kaiser windowed-sinc polyphase resampler.

The models sing at their own rate (44.1 kHz); this converts to the project rate. It is
numpy only and works on absolute sample positions, so a long signal can be converted in
pieces that join without a seam::

    y = resample(x, 44100, 48000)                    # the whole signal
    o0, n = out_span(in_start, len(piece), 44100, 48000)
    y_piece = resample(piece, 44100, 48000, in_start=in_start)   # == y[o0:o0 + n] ...

... as long as ``piece`` carries :func:`margin` extra input samples on each side of the
part that is used (see :func:`in_span`). Output samples closer than the margin to the edge
of the data see zeros beyond it, exactly as the edges of the whole signal do.

Filter: cutoff in the middle of the band from 0.85 to 1.0 of the lower Nyquist frequency,
Kaiser window designed for 90 dB. Measured (``tests/test_neural_util.py``): passband
ripple under 0.01 dB up to 0.85 of the lower Nyquist, images and aliases more than 80 dB
down above it.
"""
from __future__ import annotations

from functools import lru_cache
from math import gcd

import numpy as np

PASSBAND = 0.85       # flat up to this fraction of the lower Nyquist frequency
ATTEN_DB = 90.0       # design stop-band attenuation
BLOCK = 1 << 16       # output samples computed per block
_TABLE_LIMIT = 1 << 22  # largest tap table (phases x taps) kept in memory


def _ratio(sr_in: int, sr_out: int) -> tuple[int, int]:
    """``(up, down)``: the rate pair reduced by its gcd."""
    try:
        a, b = int(sr_in), int(sr_out)
        exact = a == sr_in and b == sr_out
    except (TypeError, ValueError):
        exact = False
    if not exact or a <= 0 or b <= 0:
        raise ValueError(f"sample rates must be positive integers, got {sr_in!r} and {sr_out!r}")
    g = gcd(a, b)
    return b // g, a // g


def _design(up: int, down: int) -> tuple[int, float, float]:
    """``(half, cutoff, beta)``: filter half-width in input samples, cutoff as a fraction of
    the input Nyquist frequency and the Kaiser beta."""
    scale = min(1.0, up / down)                    # lower Nyquist over input Nyquist
    width = np.pi * (1.0 - PASSBAND) * scale       # transition band, radians per input sample
    half = int(np.ceil((ATTEN_DB - 7.95) / (2.285 * width) / 2.0)) + 1
    return half, 0.5 * (1.0 + PASSBAND) * scale, 0.1102 * (ATTEN_DB - 8.7)


def margin(sr_in: int, sr_out: int) -> int:
    """Input samples on each side of an output sample's position that take part in it.

    A piece of a longer signal converts to exactly the same output as the whole signal for
    every output sample that has this many real input samples on both sides."""
    up, down = _ratio(sr_in, sr_out)
    return 0 if up == down else _design(up, down)[0]


def out_span(in_start: int, n_in: int, sr_in: int, sr_out: int) -> tuple[int, int]:
    """``(out_start, n_out)``: the output samples whose times fall inside the input's span
    ``[in_start, in_start + n_in) / sr_in``. Spans of adjacent inputs are adjacent."""
    up, down = _ratio(sr_in, sr_out)
    in_start, n_in = int(in_start), max(0, int(n_in))
    lo = -((-in_start * up) // down)
    hi = -((-(in_start + n_in) * up) // down)
    return lo, hi - lo


def in_span(out_start: int, n_out: int, sr_in: int, sr_out: int) -> tuple[int, int]:
    """``(in_start, n_in)``: the input samples that output samples ``[out_start, out_start +
    n_out)`` depend on, margin included."""
    up, down = _ratio(sr_in, sr_out)
    out_start, n_out = int(out_start), max(0, int(n_out))
    if n_out == 0:
        return (out_start * down) // up, 0
    m = margin(sr_in, sr_out)
    lo = (out_start * down) // up - m
    hi = ((out_start + n_out - 1) * down) // up + m
    return lo, hi - lo + 1


def _taps(frac: np.ndarray, half: int, cutoff: float, beta: float) -> np.ndarray:
    """Filter taps for output positions ``frac`` (0..1) past an input sample: row ``i`` holds
    the weights of the ``2 * half`` input samples from ``half - 1`` before to ``half`` after."""
    t = frac[:, None] + (half - 1 - np.arange(2 * half))[None, :]
    u = np.clip(1.0 - (t / half) ** 2, 0.0, None)
    h = cutoff * np.sinc(cutoff * t) * np.i0(beta * np.sqrt(u)) / np.i0(beta)
    return h / h.sum(axis=1, keepdims=True)


@lru_cache(maxsize=16)
def _table(up: int, down: int) -> np.ndarray:
    half, cutoff, beta = _design(up, down)
    tab = np.ascontiguousarray(_taps(np.arange(up) / up, half, cutoff, beta))
    tab.setflags(write=False)
    return tab


def resample(x, sr_in: int, sr_out: int, *, in_start: int = 0,
             out_start: int | None = None, n_out: int | None = None) -> np.ndarray:
    """Convert mono ``x`` from ``sr_in`` to ``sr_out``. Returns float32.

    ``x[k]`` is the sample at time ``(in_start + k) / sr_in`` and the signal is zero
    elsewhere; output sample ``j`` is at time ``(out_start + j) / sr_out``. ``out_start`` and
    ``n_out`` default to :func:`out_span`, the output samples inside the input's span.
    Equal rates copy (or shift) the samples exactly."""
    up, down = _ratio(sr_in, sr_out)
    x = np.asarray(x)
    if x.ndim != 1:
        raise ValueError(f"resample expects mono audio, got an array of shape {x.shape}")
    in_start = int(in_start)
    span_start, span_n = out_span(in_start, len(x), sr_in, sr_out)
    if out_start is None:
        out_start = span_start
    out_start = int(out_start)
    if n_out is None:
        n_out = span_start + span_n - out_start
    n_out = max(0, int(n_out))
    y = np.zeros(n_out, np.float32)
    if n_out == 0 or len(x) == 0:
        return y
    if up == down:
        a = max(out_start, in_start)
        b = min(out_start + n_out, in_start + len(x))
        if b > a:
            y[a - out_start:b - out_start] = x[a - in_start:b - in_start]
        return y
    half, cutoff, beta = _design(up, down)
    table = _table(up, down) if up * 2 * half <= _TABLE_LIMIT else None
    in_end = in_start + len(x)
    # output samples outside [lo, hi) are further than the filter reach from any input
    lo = max(out_start, -((-(in_start - half) * up) // down))
    hi = min(out_start + n_out, ((in_end + half) * up) // down + 1)
    block = BLOCK if table is not None else max(256, (1 << 21) // (2 * half))
    inv = pow(down, -1, up) if up > 1 else 0
    width = 2 * half
    for n0 in range(lo, hi, block):
        n1 = min(hi, n0 + block)
        q_lo = (n0 * down) // up
        q_hi = ((n1 - 1) * down) // up
        base = q_lo - half + 1                     # absolute index of seg[0]
        seg = np.zeros(q_hi - q_lo + width)
        a = max(base, in_start)
        b = min(base + len(seg), in_end)
        if b <= a:
            continue
        seg[a - base:b - base] = x[a - in_start:b - in_start]
        out = y[n0 - out_start:n1 - out_start]
        if table is None:
            n = np.arange(n0, n1, dtype=np.int64)
            pos = n * down
            taps = _taps((pos % up) / up, half, cutoff, beta)
            idx = (pos // up - q_lo)[:, None] + np.arange(width)[None, :]
            out[:] = np.einsum("ij,ij->i", seg[idx], taps)
            continue
        step = seg.strides[0]
        for r in range(up):
            first = n0 + ((r * inv - n0) % up)     # first output in the block with phase r
            if first >= n1:
                continue
            cnt = (n1 - 1 - first) // up + 1
            s = (first * down) // up - q_lo
            win = np.lib.stride_tricks.as_strided(seg[s:], (cnt, width), (down * step, step),
                                                  writeable=False)
            out[first - n0::up] = win @ table[r]
    return y
