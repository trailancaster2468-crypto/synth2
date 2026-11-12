"""Mixing and mastering DSP for full songs (numpy only).

Everything here is vectorised, works on a coarse envelope grid or runs FFTs in float32 on a
small thread pool (numpy's FFTs and convolutions release the GIL), so a 3-minute stereo song at
44.1 kHz with ~15 stems mixes and masters in about 10 s on 4 cores.

Building blocks
---------------
* **EQ** (:class:`EQBand`, :func:`eq`): RBJ-cookbook biquads (high/low-pass, shelves, peaking
  bells, band-pass, notch). Their exact causal IIR responses are multiplied, the product is
  turned into a truncated FIR (tail below -90 dB) and applied by overlap-save FFT convolution
  (:func:`fir_mimo`, which skips silent stretches). :func:`filter_mimo` applies a matrix of such
  responses plus side outputs (for dynamic EQ) in one FFT pass.
* **Dynamics** on a 1 ms envelope grid (:data:`GRID_S`): :func:`compressor_db` (RMS or peak
  detector, soft knee, attack / release, makeup), :func:`dynamic_band_db` (de-esser and de-harsh
  bands), :func:`kick_duck_db` (sidechain ducking from kick times), :func:`activity` (vocal
  activity for ducking and carving the band). Attack / release smoothing is vectorised
  (:func:`ar_smooth`).
* **Stereo / time**: :func:`width_matrix` (M/S width with mono bass), :func:`pan_gains`,
  :func:`pingpong_delay` (tempo-synced, darkening repeats), :func:`convolve_reverb`.
* **Metering**: ITU-R BS.1770-4 / EBU R128 loudness (:func:`loudness_stats`: integrated with the
  absolute and relative gates, momentary and short-term max, loudness range) and a 4x-oversampled
  true-peak meter (:func:`true_peak`, :func:`true_peak_envelope`).
* **Mastering**: :func:`limiter_gain` / :func:`limit` (true-peak look-ahead limiter, provably below
  its ceiling on the sample grid, inter-sample peaks checked) and :func:`master_to_loudness`
  (gain + limiter iterated onto an integrated-LUFS target).

The song mix (:func:`mix_sources` and :func:`master_bus`) wires these into a vocal bus (channel
strip :func:`vocal_strip_preset`, reverb and delay sends), an instrument bus (role presets
:data:`ROLE_PRESETS` / :data:`DRUM_GROUP_PRESETS`, transient limiting, kick sidechain, vocal-keyed
ducking and carving, a bounded low-end balance, bus compression and an automatic vocal-to-band
balance) and a master chain (glue compressor, loudness target, true-peak limiter).
"""
from __future__ import annotations

import math
import os
from collections import deque
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

GRID_S = 0.001  # envelope grid (s): dynamics, ducking and automation live on this grid
SQRT2 = math.sqrt(2.0)


def grid_hop(sr: int) -> int:
    """Samples per envelope-grid block (about 1 ms)."""
    return max(1, int(round(sr * GRID_S)))


def db_to_gain(db):
    return np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)


def gain_to_db(g, floor: float = 1e-12):
    return 20.0 * np.log10(np.maximum(np.asarray(g, dtype=np.float64), floor))


def _r(x, nd=2):
    """Round for JSON reports; None for non-finite values."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


def mix_threads() -> int:
    """Worker threads for per-source processing (numpy's FFTs release the GIL). Override with the
    ``SYNTH2_MIX_THREADS`` environment variable (1 = serial)."""
    env = os.environ.get("SYNTH2_MIX_THREADS")
    if env and env.strip().isdigit():
        return max(1, int(env))
    return max(1, min(4, os.cpu_count() or 1))


def _run_jobs(fn, items: list, threads: int) -> Iterator:
    """``fn`` over ``items``, results in order; threaded when worthwhile. At most ``threads``
    jobs are running or waiting to be read at a time, so results do not pile up behind a slow
    job."""
    if threads <= 1 or len(items) <= 1:
        for it in items:
            yield fn(it)
        return
    workers = min(threads, len(items))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        todo = iter(items)
        futs: deque = deque()
        for it in todo:
            futs.append(ex.submit(fn, it))
            if len(futs) >= workers:
                break
        while futs:
            res = futs.popleft().result()
            for it in todo:
                futs.append(ex.submit(fn, it))
                break
            yield res
            del res


# =============================================================================== EQ / filters
@dataclass(frozen=True)
class EQBand:
    """One EQ band. ``kind`` is one of ``hp`` / ``lp`` (``order`` 1, 2 or 4, Butterworth for
    2 and 4), ``lowshelf`` / ``highshelf`` (``q`` is the shelf Q), ``peak`` (bell),
    ``bandpass`` (0 dB peak gain) or ``notch``. ``freq`` in Hz, ``gain_db`` for shelves / bells."""

    kind: str
    freq: float
    gain_db: float = 0.0
    q: float = 0.7071
    order: int = 2


_KINDS = ("hp", "lp", "lowshelf", "highshelf", "peak", "bandpass", "notch")
_ALIASES = {"highpass": "hp", "hpf": "hp", "lowpass": "lp", "lpf": "lp", "bell": "peak", "peaking": "peak",
            "ls": "lowshelf", "hs": "highshelf", "low_shelf": "lowshelf", "high_shelf": "highshelf", "bp": "bandpass"}


def as_band(spec) -> EQBand:
    """An :class:`EQBand` from a band, a dict ``{kind, freq, gain_db, q, order}`` or a sequence
    ``(kind, freq, gain_db=0, q=0.7071, order=2)``."""
    if isinstance(spec, EQBand):
        b = spec
    elif isinstance(spec, dict):
        b = EQBand(str(spec["kind"]), float(spec["freq"]), float(spec.get("gain_db", spec.get("gain", 0.0))),
                   float(spec.get("q", 0.7071)), int(spec.get("order", 2)))
    else:
        s = list(spec)
        b = EQBand(str(s[0]), float(s[1]), float(s[2]) if len(s) > 2 else 0.0,
                   float(s[3]) if len(s) > 3 else 0.7071, int(s[4]) if len(s) > 4 else 2)
    kind = _ALIASES.get(b.kind.lower(), b.kind.lower())
    if kind not in _KINDS:
        raise ValueError(f"unknown EQ band kind {b.kind!r}; use one of {', '.join(_KINDS)}")
    if not (b.freq > 0 and b.q > 0):
        raise ValueError(f"EQ band needs freq > 0 and q > 0, got {b}")
    return EQBand(kind, float(b.freq), float(b.gain_db), float(b.q), int(b.order))


def biquads(band: EQBand, sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """The band as a list of ``(b, a)`` coefficient triples (RBJ audio-EQ cookbook)."""
    band = as_band(band)
    kind, f0, q = band.kind, band.freq, band.q
    if kind in ("lp", "peak", "lowshelf", "highshelf", "bandpass", "notch") and f0 >= 0.45 * sr:
        if kind in ("lp", "peak", "bandpass", "notch"):
            return []  # nothing to do this close to (or above) Nyquist; a band-pass up there is silence
        f0 = 0.45 * sr
    f0 = min(f0, 0.49 * sr)
    w0 = 2 * math.pi * f0 / sr
    cw, sw = math.cos(w0), math.sin(w0)
    if kind in ("hp", "lp"):
        order = band.order
        if order == 1:
            k = math.tan(w0 / 2)
            if kind == "lp":
                b, a = [k / (1 + k), k / (1 + k), 0.0], [1.0, (k - 1) / (k + 1), 0.0]
            else:
                b, a = [1 / (1 + k), -1 / (1 + k), 0.0], [1.0, (k - 1) / (k + 1), 0.0]
            return [(np.array(b), np.array(a))]
        qs = [q] if order == 2 else ([0.5412, 1.3066] if order == 4 else [q])
        out = []
        for qq in qs:
            al = sw / (2 * qq)
            if kind == "lp":
                b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
            else:
                b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
            out.append((np.array(b), np.array([1 + al, -2 * cw, 1 - al])))
        return out
    A = 10 ** (band.gain_db / 40.0)
    al = sw / (2 * q)
    if kind == "peak":
        if band.gain_db == 0:
            return []
        b = [1 + al * A, -2 * cw, 1 - al * A]
        a = [1 + al / A, -2 * cw, 1 - al / A]
    elif kind == "bandpass":
        b, a = [al, 0.0, -al], [1 + al, -2 * cw, 1 - al]
    elif kind == "notch":
        b, a = [1.0, -2 * cw, 1.0], [1 + al, -2 * cw, 1 - al]
    else:
        if band.gain_db == 0:
            return []
        sa = 2 * math.sqrt(A) * al
        if kind == "lowshelf":
            b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
            a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
        else:
            b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
            a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return [(np.array(b, dtype=np.float64), np.array(a, dtype=np.float64))]


def biquad_response(coeffs: Iterable[tuple[np.ndarray, np.ndarray]], w: np.ndarray) -> np.ndarray:
    """Complex response of a cascade of biquads at angular frequencies ``w`` (rad / sample)."""
    z1 = np.exp(-1j * w)
    z2 = z1 * z1
    h = np.ones(len(w), dtype=np.complex128)
    for b, a in coeffs:
        h *= (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
    return h


def eq_response(bands: Iterable, sr: int, freqs_hz: np.ndarray) -> np.ndarray:
    """Complex response of an EQ curve (a list of band specs) at ``freqs_hz``."""
    w = 2 * np.pi * np.asarray(freqs_hz, dtype=np.float64) / sr
    coeffs = [c for b in bands for c in biquads(as_band(b), sr)]
    return biquad_response(coeffs, w)


_GRID_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _dense_z(sr: int) -> tuple[np.ndarray, np.ndarray]:
    """``(z^-1, z^-2)`` on the dense design grid (cached per sample rate)."""
    if sr not in _GRID_CACHE:
        k = dense_size(sr)
        z1 = np.exp(-2j * np.pi * np.arange(k // 2 + 1) / k)
        _GRID_CACHE[sr] = (z1, z1 * z1)
        while len(_GRID_CACHE) > 4:
            _GRID_CACHE.pop(next(iter(_GRID_CACHE)))
    return _GRID_CACHE[sr]


def dense_size(sr: int) -> int:
    """FFT size of the dense design grid (about 1.5 s): long enough for every IR used here."""
    return 1 << int(math.ceil(math.log2(max(1024, 1.5 * sr))))


def dense_freqs(sr: int) -> np.ndarray:
    k = dense_size(sr)
    return np.fft.rfftfreq(k, 1.0 / sr)


def eq_dense(bands: Iterable, sr: int) -> np.ndarray:
    """The EQ response on the dense design grid (see :func:`filter_mimo`)."""
    z1, z2 = _dense_z(sr)
    h = np.ones(len(z1), dtype=np.complex128)
    for spec in bands:
        for b, a in biquads(as_band(spec), sr):
            h *= (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
    return h


def _fir_len(h: np.ndarray, tol: float = 1e-9) -> int:
    """Length that keeps all but ``tol`` of the energy of each IR in ``h`` (..., K)."""
    e = h.reshape(-1, h.shape[-1]) ** 2
    tot = e.sum(axis=1)
    lens = []
    for row, t in zip(e, tot):
        if t <= 0:
            continue
        tail = np.cumsum(row[::-1])[::-1]  # energy from i to the end
        idx = np.nonzero(tail <= tol * t)[0]
        lens.append(int(idx[0]) if len(idx) else len(row))
    return max([64] + lens)


SILENCE = 1e-7  # input frames whose peak stays below this (-140 dBFS) are not filtered (output 0)


def filter_mimo(x: np.ndarray, H: np.ndarray, sr: int, max_len: int | None = None, threads: int = 1) -> np.ndarray:
    """Apply a matrix of filters by overlap-save FFT convolution: ``y[o] = sum_c h[o, c] * x[c]``.

    ``x`` is ``(C, n)``; ``H`` is ``(O, C, F)``, complex responses on the dense grid of
    :func:`dense_size` (``F = K // 2 + 1``), zeros for no connection. The responses are made causal
    FIRs (they are the exact responses of causal IIR filters, so this matches running the IIR up
    to the -90 dB truncation of its tail). Returns ``(O, n)`` float32 (see :func:`fir_mimo`).
    """
    K = dense_size(sr)
    h = np.fft.irfft(H, K, axis=-1)
    active = np.array([[np.any(H[o, c] != 0) for c in range(H.shape[1])] for o in range(H.shape[0])])
    L = _fir_len(h[active]) if active.any() else 64
    L = min(L, max_len or K // 2, K // 2)
    h = h[..., :L].copy()
    fade = min(64, L // 8)
    if fade > 1:
        h[..., L - fade:] *= np.hanning(2 * fade)[fade:]
    h[~active] = 0.0
    return fir_mimo(x, h, threads)


def fir_mimo(x: np.ndarray, h: np.ndarray, threads: int = 1) -> np.ndarray:
    """Overlap-save FFT convolution with a matrix of FIRs: ``y[o] = sum_c h[o, c] * x[c]``,
    ``x`` (C, n), ``h`` (O, C, L) (all-zero rows are skipped). The FFTs run in float32 (about
    -140 dB of rounding) and stretches where the input is silent (below :data:`SILENCE`) are
    skipped. Returns ``(O, n)`` float32, the first ``n`` samples of the convolution."""
    x = np.asarray(x)
    C, n = x.shape
    O, _, L = h.shape
    active = np.any(h != 0, axis=-1)
    N = min(1 << 18, max(1 << 15, 1 << int(math.ceil(math.log2(8 * L)))))
    N = max(N, 1 << int(math.ceil(math.log2(2 * L))))
    hop = N - L + 1
    Hn = np.fft.rfft(h, N, axis=-1).astype(np.complex64)
    nb = max(1, -(-n // hop))
    xp = np.zeros((C, (nb + 1) * hop + N), dtype=np.float32)
    xp[:, L - 1:L - 1 + n] = x
    # a frame (N <= 2 hop samples) is silent when the two hop-blocks it spans are
    bpk = np.abs(xp[:, :(nb + 2) * hop].reshape(C, nb + 2, hop)).max(axis=(0, 2))
    live = np.nonzero((bpk[:nb] > SILENCE) | (bpk[1:nb + 1] > SILENCE))[0]
    frames = sliding_window_view(xp, N, axis=-1)[:, ::hop][:, :nb]  # (C, nb, N) view
    out = np.zeros((O, nb * hop), dtype=np.float32)
    per = max(1, (1 << 22) // (N * C))

    def run(j0):  # chunks write disjoint parts of `out`, so they can run on threads
        js = live[j0:j0 + per]
        X = np.fft.rfft(frames[:, js], axis=-1)  # (C, k, F) complex64
        y = np.zeros((O, len(js), hop), dtype=np.float32)
        for o in range(O):
            Y = None
            for c in range(C):
                if active[o, c]:
                    Y = Hn[o, c] * X[c] if Y is None else Y + Hn[o, c] * X[c]
            if Y is not None:
                y[o] = np.fft.irfft(Y, N, axis=-1)[..., L - 1:]
        if js[-1] - js[0] == len(js) - 1:  # contiguous run of frames
            out[:, js[0] * hop:(js[-1] + 1) * hop] = y.reshape(O, -1)
        else:
            for i, j in enumerate(js):
                out[:, j * hop:(j + 1) * hop] = y[:, i]

    for _ in _run_jobs(run, list(range(0, len(live), per)), threads):
        pass
    return out[:, :n]


def convolve_reverb(send: np.ndarray, ir: np.ndarray, threads: int = 1) -> np.ndarray:
    """Mono send (n,) convolved with a stereo IR (m, 2): the wet signal (n + m - 1, 2) float32."""
    send = np.asarray(send)
    m = len(ir)
    x = np.concatenate([send.astype(np.float32), np.zeros(m - 1, np.float32)])
    h = np.asarray(ir, dtype=np.float64).T[:, None, :]  # (2, 1, m)
    return fir_mimo(x[None, :], h, threads).T


def eq(x: np.ndarray, bands: Iterable, sr: int, threads: int = 1) -> np.ndarray:
    """Apply an EQ curve (list of band specs) to ``x`` (``(n,)`` or ``(n, C)``, each channel the
    same). Returns float32."""
    bands = list(bands)
    x = np.asarray(x)
    if not bands or x.size == 0:
        return x.astype(np.float32)
    Hd = eq_dense(bands, sr)
    if x.ndim == 1:
        return filter_mimo(x[None, :], Hd[None, None, :], sr, threads=threads)[0]
    C = x.shape[1]
    H = np.zeros((C, C, len(Hd)), dtype=np.complex128)
    for c in range(C):
        H[c, c] = Hd
    return filter_mimo(x.T, H, sr, threads=threads).T


# =============================================================================== envelopes
def block_power(x: np.ndarray, hop: int) -> np.ndarray:
    """Mean square per ``hop``-sample block (channels averaged); the last block may be partial."""
    x = np.asarray(x)
    n = x.shape[0]
    C = 1 if x.ndim == 1 else x.shape[1]
    full = n // hop
    nb = max(1, -(-n // hop))
    out = np.zeros(nb)
    if full:
        r = x[:full * hop].reshape(full, hop * C)
        out[:full] = np.einsum("ij,ij->i", r, r, dtype=np.float64) / (hop * C)
    if nb > full and n > full * hop:
        t = np.asarray(x[full * hop:], dtype=np.float64)
        out[full] = float(np.sum(t * t)) / (t.size)
    return out


def block_peak(x: np.ndarray, hop: int) -> np.ndarray:
    a = np.abs(np.asarray(x))
    if a.ndim == 2:
        a = a.max(axis=1)
    nb = max(1, -(-len(a) // hop))
    if nb * hop != len(a):
        a = np.concatenate([a, np.zeros(nb * hop - len(a), a.dtype)])
    return a.reshape(nb, hop).max(axis=1).astype(np.float64)


def grid_to_samples(env: np.ndarray, n: int, hop: int, dtype=np.float64) -> np.ndarray:
    """Per-sample curve from a grid envelope: block ``i`` ramps linearly from ``env[i]`` to
    ``env[i + 1]`` (grid values sit at the block starts; detectors measure the block that follows)."""
    env = np.asarray(env, dtype=np.float64)
    nb = -(-n // hop)
    if len(env) < nb:
        env = np.concatenate([env, np.full(nb - len(env), env[-1] if len(env) else 1.0)])
    env = env[:nb]
    d = np.empty(nb)
    d[:-1] = env[1:] - env[:-1]
    d[-1] = 0.0
    r = (np.arange(hop) / hop).astype(dtype)
    out = env.astype(dtype)[:, None] + d.astype(dtype)[:, None] * r[None, :]
    return out.reshape(-1)[:n]


def _moving_mean(x: np.ndarray, w: int, centred: bool = True) -> np.ndarray:
    """Mean over a window of ``w`` samples, centred or trailing (``x[n-w+1..n]``); the windows
    are shortened (and renormalised) at the edges."""
    w = max(1, int(w))
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if w == 1 or n == 0:
        return x.copy()
    a = w // 2 if centred else w - 1
    cp = np.zeros(n + w + 1)
    np.cumsum(x, out=cp[a + 1:a + 1 + n])
    cp[a + 1 + n:] = cp[a + n]
    out = cp[w:w + n] - cp[:n]
    out /= w
    # edge windows hold fewer samples
    k = min(n, w)
    idx = np.unique(np.concatenate([np.arange(k), np.arange(n - k, n)]))
    cnt = np.minimum(idx - a + w, n) - np.maximum(idx - a, 0)
    out[idx] *= w / cnt
    return out


def _sliding_max(x: np.ndarray, w: int, forward: bool = False) -> np.ndarray:
    """Max over a window of ``w`` samples: ``x[n-w+1..n]`` (or ``x[n..n+w-1]`` when ``forward``)."""
    return -_sliding_min(-x, w, forward)


def _sliding_min(x: np.ndarray, w: int, forward: bool = True) -> np.ndarray:
    """Min over ``x[n..n+w-1]`` (forward) or ``x[n-w+1..n]``, by doubling: O(n log w), vectorised."""
    w = max(1, int(w))
    x = np.asarray(x, dtype=np.float64)
    if w == 1:
        return x.copy()
    if not forward:
        return _sliding_min(x[::-1], w, True)[::-1].copy()
    cur = x.copy()  # cur[i] = min over x[i..i+span-1] (windows clipped at the end)
    span = 1
    while span * 2 <= w:
        np.minimum(cur[:-span], cur[span:], out=cur[:-span])
        span *= 2
    if span < w:
        sh = w - span
        np.minimum(cur[:-sh], cur[sh:], out=cur[:-sh])
    return cur


def _one_pole(x: np.ndarray, a: float, y0: float = 0.0) -> np.ndarray:
    """``y[n] = a y[n-1] + (1 - a) x[n]`` (vectorised: closed form within chunks, carried across)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if a <= 0 or n == 0:
        return x.copy()
    la = -math.log(a)
    chunk = max(1, min(n, int(250.0 / la)))
    pad = (-n) % chunk
    xr = np.concatenate([x, np.zeros(pad)]).reshape(-1, chunk)
    k = np.arange(chunk, dtype=np.float64)
    w = np.exp(-la * k)  # a^k
    S = np.cumsum(xr * np.exp(la * k), axis=1) * w * (1 - a)
    ak1 = w * a  # a^(k+1)
    prev = y0
    for c in range(S.shape[0]):
        S[c] += ak1 * prev
        prev = S[c, -1]
    return S.reshape(-1)[:n]


def _peak_release(x: np.ndarray, c: float, y0: float = 0.0) -> np.ndarray:
    """``y[n] = max(x[n], c y[n-1])`` for ``x >= 0`` (instant attack, exponential release), via a
    running max in the log domain."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if c <= 0:
        return x.copy()
    logc = math.log(c)
    idx = np.arange(1, n + 1, dtype=np.float64)
    with np.errstate(divide="ignore"):
        u = np.log(np.maximum(x, 0.0)) - idx * logc
        u0 = math.log(y0) if y0 > 0 else -np.inf
    v = np.maximum.accumulate(np.maximum(u, u0))
    with np.errstate(invalid="ignore", over="ignore"):
        y = np.exp(v + idx * logc)
    return np.nan_to_num(y, nan=0.0, posinf=0.0)


def ar_smooth(target: np.ndarray, attack_blocks: float, release_blocks: float, start: float | None = None) -> np.ndarray:
    """Attack / release smoothing of a grid signal (time constants in blocks): a peak follower
    with exponential release (towards the signal's minimum), then a one-pole attack filter.
    Rising values follow with the attack time, falling ones with the release."""
    x = np.asarray(target, dtype=np.float64)
    if len(x) == 0:
        return np.zeros(0)
    lo = float(x.min()) if start is None else min(float(x.min()), float(start))
    s0 = (x[0] if start is None else float(start)) - lo
    cr = math.exp(-1.0 / release_blocks) if release_blocks > 0 else 0.0
    ca = math.exp(-1.0 / attack_blocks) if attack_blocks > 0 else 0.0
    y = _peak_release(x - lo, cr, s0)
    return _one_pole(y, ca, s0) + lo


# =============================================================================== dynamics
def static_gr_db(level_db: np.ndarray, threshold_db: float, ratio: float, knee_db: float = 6.0) -> np.ndarray:
    """Gain reduction (dB, >= 0) of a soft-knee compressor curve."""
    level_db = np.asarray(level_db, dtype=np.float64)
    slope = 1.0 - 1.0 / max(ratio, 1.0)
    over = level_db - threshold_db
    k = max(knee_db, 0.0)
    if k <= 0:
        return np.maximum(over, 0.0) * slope
    gr = np.where(over > k / 2, over * slope, 0.0)
    m = np.abs(over) <= k / 2
    gr[m] = slope * (over[m] + k / 2) ** 2 / (2 * k)
    return gr


def detector_db(x: np.ndarray, sr: int, detector: str = "rms", window: float = 0.01) -> np.ndarray:
    """Level (dB) on the envelope grid: RMS over ``window`` seconds (centred), or block peak."""
    hop = grid_hop(sr)
    if detector == "peak":
        return gain_to_db(block_peak(x, hop), 1e-10)
    p = block_power(x, hop)
    w = max(1, int(round(window / GRID_S)))
    return 10 * np.log10(np.maximum(_moving_mean(p, w), 1e-20))


def compressor_db(level_db: np.ndarray, threshold_db: float, ratio: float, attack: float = 0.01,
                  release: float = 0.12, knee_db: float = 6.0, makeup_db: float = 0.0,
                  max_gr_db: float | None = None) -> np.ndarray:
    """Gain curve (dB, on the grid) of a feed-forward compressor driven by ``level_db``.

    ``attack`` / ``release`` are time constants in seconds; ``max_gr_db`` caps the reduction."""
    gr = static_gr_db(level_db, threshold_db, ratio, knee_db)
    if max_gr_db is not None:
        gr = np.minimum(gr, max_gr_db)
    gr = ar_smooth(gr, attack / GRID_S, release / GRID_S, start=0.0)
    return makeup_db - gr


def compress(x: np.ndarray, sr: int, threshold_db: float = -18.0, ratio: float = 3.0, attack: float = 0.01,
             release: float = 0.12, knee_db: float = 6.0, makeup_db: float = 0.0, detector: str = "rms",
             window: float = 0.01, key: np.ndarray | None = None) -> np.ndarray:
    """Compress ``x`` (mono or stereo-linked); ``key`` is an optional external sidechain signal."""
    x = np.asarray(x, dtype=np.float64)
    lvl = detector_db(x if key is None else key, sr, detector, window)
    g = grid_to_samples(db_to_gain(compressor_db(lvl, threshold_db, ratio, attack, release, knee_db, makeup_db)),
                        len(x), grid_hop(sr))
    return x * (g if x.ndim == 1 else g[:, None])


def relative_threshold(level_db: np.ndarray, offset_db: float, percentile: float = 95.0,
                       floor_db: float = -60.0) -> float:
    """``percentile`` of the active part (above ``floor_db``) of a level curve, plus ``offset_db``:
    a gain-staged threshold, so a compressor does the same job at any input level."""
    act = level_db[level_db > floor_db]
    if len(act) < 10:
        return 0.0
    return float(np.percentile(act, percentile)) + offset_db


def dynamic_band_db(level_full_db: np.ndarray, level_band_db: np.ndarray, rel_threshold_db: float,
                    ratio: float = 3.0, max_cut_db: float = 6.0, attack: float = 0.002, release: float = 0.05,
                    floor_db: float = -60.0) -> np.ndarray:
    """Gain (dB, <= 0) for a dynamic EQ band: cuts the band when it rises more than
    ``rel_threshold_db`` above (or less below) the full-band level (a de-esser / de-harsher)."""
    over = (level_band_db - level_full_db) - rel_threshold_db
    gr = np.clip(over * (1 - 1 / ratio), 0.0, max_cut_db)
    gr[level_band_db < floor_db] = 0.0
    return -ar_smooth(gr, attack / GRID_S, release / GRID_S, start=0.0)


def kick_duck_db(kick_times: Sequence[float], n_blocks: int, sr: int, depth_db: float = 6.0,
                 attack: float = 0.004, hold: float = 0.012, release: float = 0.16) -> np.ndarray:
    """Sidechain ducking curve (dB, <= 0, on the grid) from kick times in seconds (relative to the
    buffer start). The gain falls along a raised cosine over ``attack`` (ending on the kick, since
    the kicks are known in advance), holds ``depth_db`` for ``hold`` and recovers over ``release``
    (raised cosine in dB). Overlapping ducks take the deeper value."""
    hop = grid_hop(sr)
    env = np.zeros(n_blocks)
    if depth_db <= 0 or n_blocks == 0:
        return env
    blk = hop / sr
    t_grid = (np.arange(n_blocks) + 0.5) * blk
    span = attack + hold + release
    for tk in np.asarray(list(kick_times), dtype=np.float64):
        if not np.isfinite(tk) or tk + hold + release < 0 or tk - attack > t_grid[-1] + blk:
            continue
        i0 = max(0, int(math.floor((tk - attack) / blk)))
        i1 = min(n_blocks, int(math.ceil((tk - attack + span) / blk)) + 1)
        t = t_grid[i0:i1] - tk
        shape = np.zeros(len(t))
        a = (t >= -attack) & (t < 0)
        if attack > 0:
            shape[a] = 0.5 * (1 - np.cos(np.pi * (t[a] + attack) / attack))
        shape[(t >= 0) & (t < hold)] = 1.0
        r = (t >= hold) & (t < hold + release)
        shape[r] = 0.5 * (1 + np.cos(np.pi * (t[r] - hold) / release))
        env[i0:i1] = np.minimum(env[i0:i1], -depth_db * shape)
    return env


def activity(x: np.ndarray, sr: int, threshold_db: float = -45.0, range_db: float = 8.0, attack: float = 0.015,
             hold: float = 0.12, release: float = 0.25) -> np.ndarray:
    """Activity (0..1 on the grid) of a signal, e.g. "the vocals are singing": the 10 ms RMS level
    mapped from ``threshold_db`` (0) to ``threshold_db + range_db`` (1), held for ``hold`` seconds
    and smoothed with ``attack`` / ``release``."""
    lvl = detector_db(x, sr, "rms", 0.01)
    a = np.clip((lvl - threshold_db) / max(range_db, 1e-6), 0.0, 1.0)
    a = _sliding_max(a, max(1, int(round(hold / GRID_S))), forward=False)
    return np.clip(ar_smooth(a, attack / GRID_S, release / GRID_S, start=0.0), 0.0, 1.0)


# =============================================================================== stereo / time
def pan_gains(pan: float) -> tuple[float, float]:
    """Channel gains of the studio's constant-power pan law (unity in the centre, +3 dB hard side)."""
    theta = (float(np.clip(pan, -1, 1)) + 1) * math.pi / 4
    return math.cos(theta) * SQRT2, math.sin(theta) * SQRT2


def width_matrix(width: float, sr: int, mono_below_hz: float | None = None) -> np.ndarray:
    """2x2 dense responses (2, 2, F) of a mid/side width stage: side x ``width``, and the side
    high-passed at ``mono_below_hz`` (mono bass). Identity for width 1 and no mono bass."""
    F = dense_size(sr) // 2 + 1
    s = np.full(F, float(width), dtype=np.complex128)
    if mono_below_hz:
        s = s * eq_dense([("hp", mono_below_hz, 0, 0.7071, 2)], sr)
    M = np.empty((2, 2, F), dtype=np.complex128)
    M[0, 0] = M[1, 1] = 0.5 + 0.5 * s
    M[0, 1] = M[1, 0] = 0.5 - 0.5 * s
    return M


def stereo_width(x: np.ndarray, sr: int, width: float = 1.0, mono_below_hz: float | None = None) -> np.ndarray:
    """Stereo width on an (n, 2) signal (see :func:`width_matrix`)."""
    if width == 1.0 and not mono_below_hz:
        return np.asarray(x, dtype=np.float64).copy()
    return filter_mimo(np.asarray(x, dtype=np.float64).T, width_matrix(width, sr, mono_below_hz), sr).T


def pingpong_delay(x: np.ndarray, sr: int, delay_s: float, feedback: float = 0.35, hp_hz: float = 300.0,
                   lp_hz: float = 6000.0, damp_hz: float = 4500.0, max_repeats: int = 16) -> np.ndarray:
    """Ping-pong delay of a mono send: repeat k (k = 1, 2, ...) comes after ``k * delay_s``
    with gain ``feedback ** (k - 1)``, alternating left / right (first repeat left). The send is
    band-limited (``hp_hz`` / ``lp_hz``) and every repeat gets darker (``damp_hz`` low-pass per
    repeat). Returns the wet signal ``(len(x) + tail, 2)``; it is exact (frequency domain)."""
    x = np.asarray(x)
    if x.ndim == 2:
        x = x.mean(axis=1)
    fb = float(np.clip(feedback, 0.0, 0.95))
    reps = 1 if fb <= 0 else min(max_repeats, max(1, int(math.ceil(math.log(1e-3) / math.log(fb))) + 1))
    d = max(1.0, delay_s * sr)
    tail = int(math.ceil(reps * d + 0.25 * sr))

    def network(z_step, tone, damp):
        HL = np.zeros(len(tone), dtype=np.complex128)
        HR = np.zeros(len(tone), dtype=np.complex128)
        term = tone * z_step  # repeat k: fb^(k-1) * damp^(k-1) * step^k
        step = z_step * damp * fb
        for k in range(1, reps + 1):
            if k % 2:
                HL += term
            else:
                HR += term
            if k < reps:
                term = term * step
        return HL, HR

    tone_bands = [("hp", hp_hz, 0, 0.7071, 2), ("lp", lp_hz, 0, 0.7071, 2)]
    damp_bands = [("lp", damp_hz, 0, 0.7071, 1)]
    # design the network's impulse responses on a grid long enough for all the repeats, then run
    # them as FIRs (overlap-save, float32, silence skipped)
    K2 = 1 << int(math.ceil(math.log2(2 * (reps * d + 0.2 * sr))))
    f = np.fft.rfftfreq(K2, 1.0 / sr)
    HL, HR = network(np.exp(-1j * (2 * np.pi * f / sr) * d), eq_response(tone_bands, sr, f),
                     eq_response(damp_bands, sr, f))
    h = np.fft.irfft(np.stack([HL, HR]), K2, axis=-1)
    Lh = min(K2 // 2, _fir_len(h))
    h = h[:, :Lh]
    xx = np.concatenate([x.astype(np.float32), np.zeros(tail, np.float32)])
    return fir_mimo(xx[None, :], h[:, None, :], mix_threads()).T


# =============================================================================== loudness / peaks
def k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """BS.1770 K-weighting (pre-filter shelf + RLB high-pass) for any sample rate.

    The analog prototypes (f0, gain, Q) are those of libebur128, which reproduce the 48 kHz
    coefficients printed in ITU-R BS.1770-4 to about 1e-8."""
    f0, G, Q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    K = math.tan(math.pi * f0 / sr)
    Vh = 10 ** (G / 20.0)
    Vb = Vh ** 0.4996667741545416
    a0 = 1.0 + K / Q + K * K
    b1 = np.array([(Vh + Vb * K / Q + K * K) / a0, 2.0 * (K * K - Vh) / a0, (Vh - Vb * K / Q + K * K) / a0])
    a1 = np.array([1.0, 2.0 * (K * K - 1.0) / a0, (1.0 - K / Q + K * K) / a0])
    f0, Q = 38.13547087602444, 0.5003270373238773
    K = math.tan(math.pi * f0 / sr)
    d = 1.0 + K / Q + K * K
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (K * K - 1.0) / d, (1.0 - K / Q + K * K) / d])
    return [(b1, a1), (b2, a2)]


def _as_channels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x[None, :] if x.ndim == 1 else x.T


def k_filter(x: np.ndarray, sr: int, threads: int | None = None) -> np.ndarray:
    """K-weighted channels ``(C, n)``."""
    ch = _as_channels(x)
    z1, z2 = _dense_z(sr)
    Hd = np.ones(len(z1), dtype=np.complex128)
    for b, a in k_weighting(sr):
        Hd *= (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)
    C = ch.shape[0]
    H = np.zeros((C, C, len(Hd)), dtype=np.complex128)
    for c in range(C):
        H[c, c] = Hd
    return filter_mimo(ch, H, sr, threads=mix_threads() if threads is None else threads)


def _step_power(zz: np.ndarray, sr: int) -> np.ndarray:
    """Mean of ``zz`` (summed K-weighted channel power per sample) per 100 ms step (complete steps)."""
    step = int(round(0.1 * sr))
    ns = len(zz) // step
    if ns == 0:
        return np.zeros(0)
    return zz[:ns * step].reshape(ns, step).mean(axis=1)


def _lufs(p):
    return -0.691 + 10 * np.log10(np.maximum(p, 1e-30))


def _gated_integrated(steps: np.ndarray) -> float | None:
    """Integrated loudness from 100 ms step powers (400 ms blocks, 75 % overlap, both gates)."""
    if len(steps) < 4:
        return None
    blocks = _moving_mean(steps, 4, centred=False)[3:]
    lk = _lufs(blocks)
    g1 = blocks[lk > -70.0]
    if len(g1) == 0:
        return None
    rel = _lufs(g1.mean()) - 10.0
    g2 = blocks[(lk > -70.0) & (lk > rel)]
    return float(_lufs(g2.mean())) if len(g2) else None


def loudness_stats(x: np.ndarray, sr: int) -> dict:
    """BS.1770-4 / EBU R128 loudness of ``x`` ((n,) mono or (n, C), all channel weights 1):
    ``integrated`` (LUFS, gated), ``momentary_max`` (400 ms), ``short_term_max`` (3 s) and
    ``range`` (LRA, EBU Tech 3342). Values are None for silence or audio shorter than 400 ms."""
    z = k_filter(x, sr)
    zz = np.sum(z * z, axis=0)
    return _stats_from_steps(_step_power(zz, sr))


def _stats_from_steps(steps: np.ndarray) -> dict:
    out = {"integrated": _gated_integrated(steps), "momentary_max": None, "short_term_max": None, "range": None}
    if len(steps) >= 4:
        mom = _lufs(_moving_mean(steps, 4, centred=False)[3:])
        out["momentary_max"] = float(mom.max()) if mom.max() > -70 else None
    if len(steps) >= 30:
        st = _lufs(_moving_mean(steps, 30, centred=False)[29:])
        out["short_term_max"] = float(st.max()) if st.max() > -70 else None
        g = st[st > -70]
        if len(g):
            pw = 10 ** ((g + 0.691) / 10)
            rel = _lufs(pw.mean()) - 20
            g = g[g > rel]
            if len(g) > 1:
                out["range"] = float(np.percentile(g, 95) - np.percentile(g, 10))
    return out


def integrated_loudness(x: np.ndarray, sr: int) -> float | None:
    return loudness_stats(x, sr)["integrated"]


_TP_TAPS = 16  # taps per interpolation phase


def _tp_filters(oversample: int = 4) -> list[np.ndarray]:
    half = _TP_TAPS // 2
    j = np.arange(-(half - 1), half + 1, dtype=np.float64)  # tap offsets
    beta = 7.0
    out = []
    for k in range(1, oversample):
        t = j - k / oversample
        win = np.i0(beta * np.sqrt(np.clip(1 - (t / (half + 0.5)) ** 2, 0, 1))) / np.i0(beta)
        c = np.sinc(t) * win
        out.append(c / c.sum())
    return out


def _tp_interp_max(ch: np.ndarray, oversample: int = 4) -> np.ndarray:
    """max |interpolated value| over channels and phases between every sample m and m+1 (threaded)."""
    half = _TP_TAPS // 2
    jobs = [(c, f) for c in ch for f in _tp_filters(oversample)]

    def one(job):
        c, f = job
        return np.abs(np.convolve(c, f[::-1])[half:half + len(c)])  # full[i] = sum x[i - T + 1 + k] f[k]

    best = None
    for a in _run_jobs(one, jobs, mix_threads()):
        best = a if best is None else np.maximum(best, a, out=best)
    return best


_TP_L1 = float(max(np.abs(c).sum() for c in _tp_filters(4)))  # bound: |interpolated| <= L1 * local max


def _interp_abs_max(ch: np.ndarray, idx: np.ndarray, oversample: int = 4) -> np.ndarray:
    """max |value| over all channels of the interpolated points between sample m and m+1, for m in idx."""
    half = _TP_TAPS // 2
    F = np.stack(_tp_filters(oversample), axis=1)  # (T, phases)
    out = np.zeros(len(idx))
    off = np.arange(-(half - 1), half + 1)
    for c in ch:
        xp = np.concatenate([np.zeros(half), c, np.zeros(half + 1)])
        for s in range(0, len(idx), 1 << 18):
            j = idx[s:s + (1 << 18)]
            win = xp[(j + half)[:, None] + off[None, :]]
            np.maximum(out[s:s + len(j)], np.abs(win @ F).max(axis=1), out=out[s:s + len(j)])
    return out


def _tp_candidates(a: np.ndarray, floor: float) -> np.ndarray:
    """Sample indices m whose interpolated points (between m and m+1) can exceed ``floor``."""
    half = _TP_TAPS // 2
    loc = _sliding_max(a, _TP_TAPS, forward=True)  # max over a[i..i+T-1]
    loc = np.concatenate([np.full(half - 1, loc[0]), loc])[:len(a)]  # -> max over a[m-half+1..m+half]
    return np.nonzero(loc * _TP_L1 > floor)[0]


def true_peak_envelope(x: np.ndarray, floor: float = 0.0, oversample: int = 4) -> np.ndarray:
    """Per-sample true-peak envelope: max |value| over all channels at the sample and at the
    ``oversample``-x interpolated points on both sides of it. Interpolated points that cannot
    reach ``floor`` are skipped (the envelope is then the sample value, below ``floor``)."""
    ch = _as_channels(x)
    env = np.abs(ch).max(axis=0)
    n = len(env)
    if n == 0:
        return env
    idx = _tp_candidates(env, floor) if floor > 0 else np.arange(n)
    if len(idx) == 0:
        return env
    if len(idx) > n // 16:
        best = _tp_interp_max(ch, oversample)
        np.maximum(env, best, out=env)
        np.maximum(env[1:], best[:-1], out=env[1:])
        return env
    vals = _interp_abs_max(ch, idx, oversample)  # idx is sorted and unique
    env[idx] = np.maximum(env[idx], vals)
    nxt = idx + 1 < n
    j = idx[nxt] + 1
    env[j] = np.maximum(env[j], vals[nxt])
    return env


def true_peak(x: np.ndarray, oversample: int = 4) -> float:
    """True peak (linear, 4x oversampled as in BS.1770-4 Annex 2). Exact for the interpolator
    used, but only the points that can beat the sample peak are computed."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return 0.0
    ch = _as_channels(x)
    a = np.abs(ch).max(axis=0)
    m = float(a.max())
    if m <= 0:
        return 0.0
    idx = _tp_candidates(a, m)
    if len(idx) > len(a) // 16:  # dense: one convolution per phase is cheaper than gathering
        m = max(m, float(_tp_interp_max(ch, oversample).max()))
    elif len(idx):
        m = max(m, float(_interp_abs_max(ch, idx, oversample).max()))
    return m


def true_peak_db(x: np.ndarray) -> float:
    return float(gain_to_db(true_peak(x)))


# =============================================================================== limiter / mastering
LIMITER_BLOCK = 32  # samples per block of the limiter's release stage


def _release_blocks(r1: np.ndarray, sr: int, release: float, release_slow: float, slow_window: float) -> np.ndarray:
    """Released reduction (dB) per limiter block: ``R[b] = max(r1[b], c R[b-1])`` with time constant
    ``release`` (vectorised in the log domain); reduction sustained over ``slow_window`` releases
    with ``release_slow`` (program-dependent release)."""
    B = LIMITER_BLOCK
    nb = len(r1)
    idx = np.arange(nb, dtype=np.float64)

    def _rel(r, tau):
        logc = -B / max(tau * sr, 1.0)
        with np.errstate(divide="ignore"):
            u = np.log(r) - idx * logc
        v = np.maximum.accumulate(u)
        with np.errstate(invalid="ignore", over="ignore"):
            out = np.exp(v + idx * logc)
        return np.nan_to_num(out, nan=0.0, posinf=0.0)

    r = _rel(r1, release)
    if release_slow and release_slow > release:
        sw = max(1, int(round(slow_window * sr / B)))
        r = np.maximum(r, np.minimum(_rel(_moving_mean(r1, sw, centred=True) * 0.85, release_slow), r1.max()))
    return r


def _block_reduce(x: np.ndarray, B: int, fn, fill: float) -> np.ndarray:
    nb = -(-len(x) // B)
    if nb * B != len(x):
        x = np.concatenate([x, np.full(nb * B - len(x), fill)])
    return fn(x.reshape(nb, B), axis=1)


def limiter_gain(env: np.ndarray, sr: int, ceiling: float, lookahead: float = 0.0015, release: float = 0.05,
                 release_slow: float = 0.35, slow_window: float = 0.05) -> np.ndarray:
    """Per-sample gain of a look-ahead peak limiter for a peak envelope ``env``.

    The required gain ``ceiling / env`` is min-filtered over the look-ahead window (so the gain is
    down before a peak arrives), its reduction (dB) is released exponentially on 32-sample blocks
    (time constant ``release`` for short peaks; reduction sustained over ``slow_window`` releases
    with ``release_slow``), and the result is smoothed by a trailing moving average as long as the
    look-ahead. By construction ``gain[n] * env[n] <= ceiling`` at every sample.
    """
    env = np.asarray(env, dtype=np.float64)
    n = len(env)
    if n == 0:
        return np.ones(0)
    greq = np.minimum(1.0, ceiling / np.maximum(env, 1e-12))
    if greq.min() >= 1.0:
        return np.ones(n)
    L = max(1, int(round(lookahead * sr)))
    g1 = _sliding_min(greq, L, forward=True)
    return _limiter_finish(g1, greq, sr, L, _release_blocks(-20 * np.log10(_block_reduce(g1, LIMITER_BLOCK, np.min, 1.0)),
                                                         sr, release, release_slow, slow_window))


def _limiter_finish(g1: np.ndarray, greq: np.ndarray, sr: int, L: int, r_blk: np.ndarray) -> np.ndarray:
    """Samples from the released block reduction: the released curve starts after each block (so
    it never undercuts ``g1``), then the look-ahead-long moving average and the safety clamp."""
    n = len(g1)
    B = LIMITER_BLOCK
    g_rel = np.interp(np.arange(n, dtype=np.float64), (np.arange(len(r_blk)) + 1.0) * B - 1.0, 10 ** (-r_blk / 20.0))
    g2 = np.minimum(g1, g_rel)
    g3 = _moving_mean(g2, L, centred=False)
    return np.minimum(g3, greq)


def limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0, lookahead: float = 0.0015, release: float = 0.05,
          true_peak_mode: bool = True) -> np.ndarray:
    """True-peak limiter (stereo-linked). The output's 4x-oversampled peak is <= ``ceiling_db``."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    ceiling = 10 ** (ceiling_db / 20.0) * 10 ** (-0.02 / 20)  # a hair under the ceiling
    if true_peak_mode:
        if float(np.max(np.abs(x))) * _TP_L1 <= ceiling:
            return x.copy()  # not even an inter-sample peak can reach the ceiling
        env = true_peak_envelope(x, ceiling)
    else:
        env = np.abs(_as_channels(x)).max(axis=0)
    if float(env.max()) <= ceiling:
        return x.copy()
    g = limiter_gain(env, sr, ceiling, lookahead, release)
    y = x * (g if x.ndim == 1 else g[:, None])
    if true_peak_mode:
        tp = true_peak(y)
        lim = 10 ** (ceiling_db / 20.0)
        if tp > lim:  # inter-sample residue of a fast gain change: trim the whole thing (tiny)
            y *= lim / tp
    return y


def master_to_loudness(x: np.ndarray, sr: int, target_lufs: float | None, ceiling_db: float = -1.0,
                       lookahead: float = 0.0015, release: float = 0.05, limiter: bool = True,
                       max_gain_db: float = 30.0, max_limiting_db: float = 9.0) -> tuple[np.ndarray, dict]:
    """Gain + true-peak limiter iterated so the output's integrated loudness hits ``target_lufs``.

    Without a target the input is only limited. The loudness of each candidate is predicted from
    the K-weighted input times the gain curve (cheap), and the final result is measured exactly.
    If the target would need more than ``max_limiting_db`` of peak gain reduction the gain stops
    there (the report says so). Without the limiter the gain stops at the true-peak ceiling.
    Returns ``(y, info)``; ``info`` has ``gain_db`` (static gain), ``limiter_max_db`` and
    ``limiter_mean_db`` (gain reduction where it acts)."""
    x = np.asarray(x, dtype=np.float64)
    info: dict[str, Any] = {"gain_db": 0.0, "gain": 1.0, "limiter_max_db": 0.0, "limiter_mean_db": 0.0,
                            "target_reached": True, "true_peak": 0.0}
    if x.size == 0:
        return x.copy(), info
    ceiling_lin = 10 ** (ceiling_db / 20.0)
    pk = float(np.max(np.abs(x)))
    if pk <= 0:
        return x.copy(), info
    G_db = 0.0
    L0 = None
    if target_lufs is not None:
        z = k_filter(x, sr)
        zz = np.sum(z * z, axis=0)
        del z
        L0 = _gated_integrated(_step_power(zz, sr))
    # the envelope only needs true-peak detail where the gains tried below can push it over the ceiling
    g_hi = 0.0 if L0 is None else min(max_gain_db, target_lufs - L0 + max_limiting_db + 6.0)
    env = true_peak_envelope(x, ceiling_lin * 10 ** (-g_hi / 20.0))
    pk = float(env.max())
    lim = ceiling_lin * 10 ** (-0.02 / 20)
    L = max(1, int(round(lookahead * sr)))
    if target_lufs is not None and L0 is not None:
        G_db = float(np.clip(target_lufs - L0, -60.0, max_gain_db))
        if not limiter:
            G_db = min(G_db, ceiling_db - float(gain_to_db(pk)))
        else:
            # iterate on 32-sample blocks: the look-ahead minimum of min(1, lim / (G env)) is
            # min(1, lim / (G max(env over the window))), so the envelope maxima are computed once
            B = LIMITER_BLOCK
            env_blk = _block_reduce(_sliding_max(env, L, forward=True), B, np.max, 0.0)
            zz_blk = _block_reduce(zz, B, np.sum, 0.0)
            step = int(round(0.1 * sr))
            ns = len(zz) // step
            sid = np.minimum((np.arange(len(zz_blk)) * B + B // 2) // step, ns)  # block -> 100 ms step

            def predict(gdb):
                G = 10 ** (gdb / 20.0)
                r1 = np.maximum(0.0, 20 * np.log10(np.maximum(G * env_blk, 1e-12) / lim))
                r = _release_blocks(r1, sr, release, 0.35, 0.05)
                p = np.bincount(sid, weights=zz_blk * 10 ** (-r / 10.0), minlength=ns + 1)[:ns] / step
                return _gated_integrated(p * G * G), float(r.max())

            for _ in range(10):
                Lp, gr_max = predict(G_db)
                if Lp is None:
                    break
                err = target_lufs - Lp
                if gr_max > max_limiting_db and err > 0:
                    # back off to the gain that limits by max_limiting_db at most
                    G_db -= gr_max - max_limiting_db
                    info["target_reached"] = False
                    break
                if abs(err) < 0.02:
                    break
                G_db = float(np.clip(G_db + err * (1.15 if err > 0 else 1.0), -60.0, max_gain_db))
    G = 10 ** (G_db / 20.0)
    if limiter:
        g = limiter_gain(env * G, sr, lim, lookahead, release)
        if target_lufs is not None and L0 is not None and info["target_reached"]:
            # the sample-level gain smooths the attacks a little more than the block prediction:
            # one exact correction step
            Lx = _gated_integrated(_step_power(zz * (G * g) ** 2, sr))
            if Lx is not None and abs(target_lufs - Lx) > 0.03:
                G_db = float(np.clip(G_db + (target_lufs - Lx), -60.0, max_gain_db))
                G = 10 ** (G_db / 20.0)
                g = limiter_gain(env * G, sr, lim, lookahead, release)
        gG = g * G
        y = x * (gG[:, None] if x.ndim == 2 else gG)
        # inter-sample check of the output where it can come near the ceiling (the gain may vary
        # within the interpolation kernel, so the local gain maximum is used)
        est = env * _sliding_max(gG, _TP_TAPS, forward=True)
        tp = _true_peak_near(y, est, ceiling_lin * 10 ** (-0.5 / 20))
        if tp > ceiling_lin:
            y *= ceiling_lin / tp
            tp = ceiling_lin
        red = -gain_to_db(g)
        info["limiter_max_db"] = _r(red.max())
        act = red[red > 0.05]
        info["limiter_mean_db"] = _r(act.mean()) if len(act) else 0.0
    else:
        y = x * G
        tp = float(env.max()) * G
    info["gain_db"] = _r(G_db)
    info["gain"] = G
    info["true_peak"] = tp
    return y, info


def _true_peak_near(y: np.ndarray, est: np.ndarray, thr: float) -> float:
    """True peak of ``y`` computing the interpolated points only where ``est`` (an estimate of the
    per-sample true-peak envelope) exceeds ``thr``; everywhere else the sample peak counts."""
    ch = _as_channels(y)
    m = float(np.max(np.abs(ch))) if ch.size else 0.0
    idx = np.nonzero(est > thr)[0]
    if len(idx) > len(est) // 16:
        return true_peak(y)
    if len(idx):
        lo = np.maximum(idx - 1, 0)  # the points before the sample too
        m = max(m, float(_interp_abs_max(ch, np.unique(np.concatenate([lo, idx]))).max()))
    return m


# =============================================================================== mix presets
# role -> default channel processing for backing stems. eq: band specs; comp: compressor with a
# gain-staged threshold (offset from the stem's 95th-percentile RMS level) or None; sc: kick
# sidechain depth factor (x MasterSettings.sidechain_db); duck / carve: factors for the
# vocal-keyed band ducking and 1.5-4 kHz carving; reverb: send to the song reverb; width:
# extra stereo width (the arranger already sets each instrument's width, so 1.0); mono_below: Hz
# below which the stem is made mono (keeps the low end centred); crest_db: transient limiter that
# keeps peaks within this many dB of the stem's loud (95th-percentile 10 ms RMS) level, so clicky
# drum hits do not make the master limiter duck the whole mix.
ROLE_PRESETS: dict[str, dict[str, Any]] = {
    "drums": {"eq": [("hp", 30, 0, 0.7071, 4), ("peak", 420, -1.5, 1.0)],
              "comp": {"ratio": 3.0, "offset_db": -5.0, "attack": 0.015, "release": 0.12, "detector": "rms"},
              "sc": 0.5, "duck": 0.3, "carve": 0.35, "reverb": 0.04, "width": 1.0, "mono_below": 120.0,
              "crest_db": 14.0},
    "bass": {"eq": [("hp", 38, 0, 0.7071, 4), ("peak", 260, -1.5, 1.0), ("peak", 2800, -2.0, 0.9)],
             "comp": {"ratio": 3.0, "offset_db": -5.0, "attack": 0.01, "release": 0.10, "detector": "rms"},
             "sc": 1.0, "duck": 0.3, "carve": 0.5, "reverb": 0.0, "width": 1.0, "mono_below": 160.0},
    "keys": {"eq": [("hp", 90, 0, 0.7071), ("peak", 320, -1.5, 1.0)],
             "comp": None, "sc": 0.8, "duck": 1.0, "carve": 1.0, "reverb": 0.10, "width": 1.0, "mono_below": 150.0},
    "lead": {"eq": [("hp", 140, 0, 0.7071), ("peak", 350, -1.5, 1.0)],
             "comp": {"ratio": 2.0, "offset_db": -4.0, "attack": 0.008, "release": 0.08, "detector": "rms"},
             "sc": 0.6, "duck": 1.3, "carve": 1.2, "reverb": 0.12, "width": 1.0, "mono_below": 150.0},
    "pad": {"eq": [("hp", 170, 0, 0.7071), ("peak", 400, -2.0, 0.8)],
            "comp": None, "sc": 1.25, "duck": 1.2, "carve": 1.2, "reverb": 0.08, "width": 1.0, "mono_below": 200.0},
    "perc": {"eq": [("hp", 160, 0, 0.7071)], "crest_db": 12.0,
             "comp": None, "sc": 0.5, "duck": 0.5, "carve": 0.5, "reverb": 0.08, "width": 1.0, "mono_below": 150.0},
    "fx": {"eq": [("hp", 120, 0, 0.7071)], "crest_db": 16.0,
           "comp": None, "sc": 1.0, "duck": 0.8, "carve": 0.8, "reverb": 0.15, "width": 1.0, "mono_below": 150.0},
}

# drum sub-stems (the arranger can split a kit into "<name> kick", "... snare", "... hats", "... cymbals")
DRUM_GROUP_PRESETS: dict[str, dict[str, Any]] = {
    "kick": {"eq": [("hp", 32, 0, 0.7071, 4), ("peak", 380, -2.0, 1.2),
                    ("peak", 3500, 1.0, 1.0)],
             "comp": {"ratio": 3.0, "offset_db": -4.0, "attack": 0.012, "release": 0.08, "detector": "rms"},
             "duck": 0.0, "carve": 0.0, "reverb": 0.0, "width": 1.0, "mono_below": 200.0, "crest_db": None},
    "snare": {"eq": [("hp", 90, 0, 0.7071), ("peak", 450, -1.5, 1.0)],
              "comp": {"ratio": 3.0, "offset_db": -5.0, "attack": 0.01, "release": 0.1, "detector": "rms"},
              "duck": 0.2, "carve": 0.4, "reverb": 0.05, "width": 1.0, "mono_below": 150.0, "crest_db": 12.0},
    "hats": {"eq": [("hp", 300, 0, 0.7071)], "comp": None,
             "duck": 0.3, "carve": 0.5, "reverb": 0.02, "width": 1.0, "mono_below": None, "crest_db": 10.0},
    "cymbals": {"eq": [("hp", 250, 0, 0.7071)], "comp": None,
                "duck": 0.5, "carve": 0.6, "reverb": 0.03, "width": 1.0, "mono_below": None, "crest_db": 10.0},
}
DRUM_GROUPS = tuple(DRUM_GROUP_PRESETS)  # the four group stems; the entries added below are single drums


def _drum(base: str, hp: float, **over) -> dict[str, Any]:
    """The preset of group ``base`` with the high-pass at ``hp`` Hz (the bands after it kept) and ``over``."""
    pre = dict(DRUM_GROUP_PRESETS[base])
    pre["eq"] = [("hp", hp, 0, 0.7071), *pre["eq"][1:]]
    pre.update(over)
    return pre


# Stems of a kit split per kind of drum ("<name> clap", "... toms", "... congas", ...). Each keeps the
# treatment of the group it used to be mixed in (snare: clap, rim, snap, toms; hats: hat, ohat, shaker, tamb,
# cowbell; cymbals: crash, ride) with a high-pass under its own lowest partial; a click gets the transient
# limiter instead of the compressor. Hand drums and the small percussion start from one preset ("perc", the
# perc role's own values); what sounds in the vocal's band (cuica, whistle) is carved and ducked more.
_HAND = {"eq": [("hp", 160, 0, 0.7071)], "comp": None, "duck": 0.5, "carve": 0.5, "reverb": 0.08, "width": 1.0,
         "mono_below": 150.0, "crest_db": 12.0}
DRUM_GROUP_PRESETS.update({
    "clap": _drum("snare", 160),
    "rim": _drum("snare", 200, comp=None),
    "snap": _drum("snare", 300, comp=None),
    "toms": {"eq": [("hp", 55, 0, 0.7071), ("peak", 420, -2.0, 1.0)],
             "comp": {"ratio": 3.0, "offset_db": -5.0, "attack": 0.015, "release": 0.12, "detector": "rms"},
             "duck": 0.2, "carve": 0.3, "reverb": 0.05, "width": 1.0, "mono_below": 120.0, "crest_db": 13.0},
    "hat": _drum("hats", 300),
    "ohat": _drum("hats", 300),
    "shaker": _drum("hats", 400),
    "tamb": _drum("hats", 350),
    "cowbell": _drum("hats", 250, crest_db=12.0),
    "crash": _drum("cymbals", 250),
    "ride": _drum("cymbals", 250, duck=0.4),
    "perc": dict(_HAND),
    "bongos": dict(_HAND, eq=[("hp", 140, 0, 0.7071)]),
    "congas": dict(_HAND, eq=[("hp", 100, 0, 0.7071)], mono_below=120.0),
    "timbales": dict(_HAND, eq=[("hp", 120, 0, 0.7071)]),
    "agogo": dict(_HAND, eq=[("hp", 300, 0, 0.7071)], mono_below=None),
    "claves": dict(_HAND, eq=[("hp", 500, 0, 0.7071)], mono_below=None),
    "woodblock": dict(_HAND, eq=[("hp", 300, 0, 0.7071)], mono_below=None),
    "triangle": dict(_HAND, eq=[("hp", 800, 0, 0.7071)], duck=0.3, carve=0.3, reverb=0.04, mono_below=None),
    "guiro": dict(_HAND, eq=[("hp", 300, 0, 0.7071)], mono_below=None),
    "cuica": dict(_HAND, eq=[("hp", 180, 0, 0.7071)], carve=0.8),
    "whistle": dict(_HAND, eq=[("hp", 700, 0, 0.7071)], duck=0.6, carve=1.0, mono_below=None),
    "vibraslap": dict(_HAND, eq=[("hp", 300, 0, 0.7071)], mono_below=None),
})


def role_preset(role: str, name: str = "") -> dict[str, Any]:
    """The default processing of a backing stem: its role's preset, refined for drum sub-stems
    (by the last word of the stem name). A group stem (kick, snare, hats, cymbals) is known by
    that word alone; a single drum's stem by the word after the kit's name ("kit clap"), so an
    instrument that is simply called "shaker" keeps its role's preset. Unknown roles are
    treated as keys."""
    pre = dict(ROLE_PRESETS.get(role, ROLE_PRESETS["keys"]))
    if role in ("drums", "perc"):
        words = str(name).strip().lower().split()
        last = words[-1] if words else ""
        if last in DRUM_GROUPS or (len(words) > 1 and last in DRUM_GROUP_PRESETS):
            pre.update(DRUM_GROUP_PRESETS[last])
    return pre


CARVE_HZ, CARVE_Q = 2600.0, 0.8  # vocal pocket carved out of the band while the vocals sing
VBR_BAND = (1000.0, 5000.0)  # vocal-to-band ratio is measured in this band
MASTER_EQ = [("hp", 25, 0, 0.7071, 2), ("highshelf", 12000, 1.0, 0.7)]  # folded into every source
RETURN_EQ = [("hp", 220, 0, 0.7071, 2), ("highshelf", 7000, -3.0, 0.7)]  # reverb / delay returns (band mode)


def vocal_strip_preset(band: bool, lowest_hz: float | None = None) -> dict[str, Any]:
    """The vocal channel strip: EQ, de-harsh and de-ess bands, compressor.

    With a band (``band=True``) the vocal gets more presence, air and compression so it stays on
    top; vocal-only renders get a gentler version. The high-pass follows the lowest sung note
    (0.7 x its f0, clamped to 40..90 Hz) so bass voices keep their fundamental."""
    hp = 90.0 if not lowest_hz else float(np.clip(0.7 * lowest_hz, 40.0, 90.0))
    if band:
        return {"eq": [("hp", hp, 0, 0.7071, 2), ("peak", 300, -2.0, 1.0), ("peak", 3200, 3.0, 0.9),
                       ("highshelf", 10000, 2.5, 0.7)],
                "deharsh": {"freq": 3200.0, "q": 1.4, "rel_db": -4.0, "max_cut_db": 3.0},
                "deess": {"freq": 6800.0, "q": 1.0, "rel_db": -6.0, "max_cut_db": 6.0},
                "comp": {"threshold_db": -26.0, "ratio": 3.5, "attack": 0.005, "release": 0.09, "knee_db": 6.0,
                         "makeup_db": 3.0}}
    return {"eq": [("hp", hp, 0, 0.7071, 2), ("peak", 300, -1.0, 1.0), ("peak", 3200, 1.5, 0.9),
                   ("highshelf", 10000, 1.5, 0.7)],
            "deharsh": None,
            "deess": {"freq": 6800.0, "q": 1.0, "rel_db": -6.0, "max_cut_db": 5.0},
            "comp": {"threshold_db": -23.0, "ratio": 2.5, "attack": 0.006, "release": 0.10, "knee_db": 6.0,
                     "makeup_db": 1.5}}


# =============================================================================== the song mix
@dataclass
class Source:
    """One input of the mix.

    ``kind`` is ``"vocal"`` (``audio`` mono (n,), dry from the voice engine) or ``"backing"``
    (``audio`` (n, 2) from the arranger, with ``role``). Gains / pans are the track or stem
    settings; ``overrides`` are the per-track / per-stem entries of ``MasterSettings.mix_overrides``.
    """

    key: str
    name: str
    kind: str
    audio: "np.ndarray | Sparse"
    role: str = "vocal"
    gain_db: float = 0.0
    pan: float = 0.0
    reverb_send: float = 0.0
    delay_send: float = 0.0
    sidechain: bool = False
    lowest_hz: float | None = None
    overrides: dict = field(default_factory=dict)
    # vocal sources: the engine that sang the track, the track's role in the mix ("" for none; see
    # vocal_chain.ROLES), the dial setting it was sung at and, for a double, the role of the track its
    # audio was copied from. A role's chain is applied to what an engine of vocal_chain.ROLE_ENGINES sang.
    engine: str = ""
    vocal_role: str = ""
    dial: str = ""
    copy_of: str | None = None


SPARSE_BLOCK = 1024  # samples per block when audio is cut into its sounding segments


class Sparse:
    """Audio that is silent most of the time, held as its sounding segments.

    ``segments`` is a list of ``(start, data)``: float32 arrays ``(m,)`` or ``(m, channels)``
    that start at sample ``start``, in order and without overlap. The rest of the ``length``
    samples is silence. ``channels`` is 0 for mono ``(n,)`` audio."""

    __slots__ = ("length", "channels", "segments")

    def __init__(self, length: int, channels: int = 0, segments: list | None = None):
        self.length = int(length)
        self.channels = int(channels)
        self.segments = list(segments or [])

    @classmethod
    def from_dense(cls, x: np.ndarray, floor: float = 0.0, length: int | None = None) -> "Sparse":
        """The segments of ``x`` ((n,) or (n, C)): runs of :data:`SPARSE_BLOCK`-sample blocks whose
        peak is above ``floor`` (0: only blocks of exact zeros are left out, so :meth:`dense`
        gives ``x`` back). ``length`` is the length it stands for (default ``len(x)``)."""
        x = np.asarray(x)
        n = len(x)
        ch = 0 if x.ndim == 1 else x.shape[1]
        out = cls(n if length is None else length, ch)
        if n == 0:
            return out
        B = SPARSE_BLOCK
        full = n // B
        peak = np.zeros(-(-n // B))
        if full:
            r = x[:full * B].reshape(full, -1)
            peak[:full] = np.maximum(r.max(axis=1), -r.min(axis=1))
        if full * B < n:
            peak[full] = float(np.max(np.abs(x[full * B:])))
        edges = np.diff(np.concatenate([[0], (peak > floor).astype(np.int8), [0]]))
        for a, b in zip(np.nonzero(edges == 1)[0], np.nonzero(edges == -1)[0]):
            out.segments.append((int(a) * B, np.array(x[a * B:min(b * B, n)], dtype=np.float32)))
        return out

    def __len__(self) -> int:
        return self.length

    @property
    def nbytes(self) -> int:
        return sum(seg.nbytes for _, seg in self.segments)

    def dense(self, dtype=np.float32, length: int | None = None) -> np.ndarray:
        """The full-length array."""
        n = self.length if length is None else int(length)
        out = np.zeros((n, self.channels) if self.channels else n, dtype=dtype)
        for a, seg in self.segments:
            m = min(n, a + len(seg)) - a
            if m > 0:
                out[a:a + m] = seg[:m]
        return out

    def add_to(self, out: np.ndarray, gain: float | None = None) -> None:
        """``out += self`` (times ``gain``, a float32 factor), segment by segment."""
        n = len(out)
        for a, seg in self.segments:
            m = min(n, a + len(seg)) - a
            if m > 0:
                out[a:a + m] += seg[:m] if gain is None else seg[:m] * np.float32(gain)

    def scale(self, g) -> None:
        """Multiply in place by a factor or by a per-sample curve ``(n,)``."""
        if np.ndim(g) == 0:
            for _, seg in self.segments:
                seg *= np.float32(g)
            return
        for a, seg in self.segments:
            cur = np.asarray(g[a:a + len(seg)], dtype=np.float32)
            seg[:len(cur)] *= cur[:, None] if self.channels else cur

    def crop(self, a: int, b: int) -> None:
        """Keep samples ``[a, b)``; sample ``a`` becomes sample 0."""
        a = max(0, int(a))
        b = min(self.length, int(b))
        segs = []
        for s0, seg in self.segments:
            lo, hi = max(a, s0), min(b, s0 + len(seg))
            if hi > lo:
                segs.append((lo - a, seg[lo - s0:hi - s0]))
        self.segments = segs
        self.length = max(0, b - a)

    def peak(self) -> float:
        return max((float(max(seg.max(), -seg.min())) for _, seg in self.segments if seg.size), default=0.0)

    def sum_squares(self) -> float:
        tot = 0.0
        for _, seg in self.segments:
            f = seg.reshape(-1)
            tot += float(np.einsum("i,i->", f, f, dtype=np.float64))
        return tot


def _dense(audio) -> np.ndarray:
    return audio.dense() if isinstance(audio, Sparse) else np.asarray(audio)


class StemStore(Mapping):
    """The stems of a mix: a read-only mapping ``key -> (n, 2) float32``.

    A song has many stems and most of them are silent most of the time, so the store keeps each
    stem as its sounding segments (:class:`Sparse`; blocks under :data:`SILENCE` count as
    silence) and builds the full-length array when the stem is read. Every read (``store[key]``,
    ``.values()``, ``.items()``) returns a new array."""

    def __init__(self, length: int = 0):
        self.length = int(length)
        self._stems: dict[str, Sparse] = {}

    def __getitem__(self, key) -> np.ndarray:
        return self._stems[key].dense(length=self.length)

    def __iter__(self):
        return iter(self._stems)

    def __len__(self) -> int:
        return len(self._stems)

    def __contains__(self, key) -> bool:
        return key in self._stems

    def put(self, key: str, stem: "Sparse | np.ndarray") -> None:
        self._stems[key] = stem if isinstance(stem, Sparse) else Sparse.from_dense(stem, SILENCE)

    def sparse(self, key: str) -> Sparse:
        """The stem as it is held."""
        return self._stems[key]

    def scale(self, g, keys: Iterable[str] | None = None) -> None:
        """Multiply the stems (all, or ``keys``) in place by a factor or a per-sample curve ``(n,)``."""
        for k in (self._stems if keys is None else keys):
            self._stems[k].scale(g)

    def crop(self, a: int, b: int | None = None) -> None:
        """Keep samples ``[a, b)`` of every stem."""
        b = self.length if b is None else min(self.length, int(b))
        for st in self._stems.values():
            st.length = self.length
            st.crop(a, b)
        self.length = max(0, b - max(0, int(a)))

    def peak(self, key: str | None = None) -> float:
        if key is not None:
            return self._stems[key].peak()
        return max((st.peak() for st in self._stems.values()), default=0.0)

    def rms(self, key: str) -> float:
        size = self.length * 2
        return math.sqrt(self._stems[key].sum_squares() / size) if size else 0.0

    @property
    def nbytes(self) -> int:
        return sum(st.nbytes for st in self._stems.values())


@dataclass
class MixResult:
    stems: StemStore  # key -> (n, 2) float32, dry and post-bus (they sum to `dry`)
    groups: dict[str, str]  # key -> "vocal" | "backing"
    dry: np.ndarray  # (n, 2) float32 sum of the stems
    wet: np.ndarray  # (n, 2) float64 reverb + delay returns (not in any stem)
    vocal_activity: np.ndarray | None  # grid activity of the vocal bus
    info: dict


def _merge(preset: dict, over: dict) -> dict:
    out = dict(preset)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def _processed_vocal(src: Source, sr: int, band: bool, chain: bool, extra_eq: list,
                     x: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
    """Vocal channel strip on the mono source (``x``: its audio as a full-length array, when the
    caller has it): EQ (+ de-harsh / de-ess dynamic bands) -> compressor.
    Returns the processed mono signal (float32) and strip info."""
    x = _dense(src.audio) if x is None else x
    info: dict[str, Any] = {}
    if not chain or not np.any(x):
        if extra_eq:
            return eq(x, extra_eq, sr).astype(np.float32), info
        return x.astype(np.float32), info
    pre = _merge(vocal_strip_preset(band, src.lowest_hz), src.overrides)
    bands = [*pre.get("eq", []), *(src.overrides.get("eq_add") or []), *extra_eq]
    Hd = eq_dense(bands, sr)
    dyn = [(k, pre[k]) for k in ("deharsh", "deess") if pre.get(k)]
    H = np.zeros((1 + len(dyn), 1, len(Hd)), dtype=np.complex128)
    H[0, 0] = Hd
    for i, (_, d) in enumerate(dyn):
        H[1 + i, 0] = Hd * eq_dense([("bandpass", d["freq"], 0, d["q"])], sr)
    outs = filter_mimo(x[None, :], H, sr)
    y = outs[0]
    hop = grid_hop(sr)
    n = len(y)
    if dyn:
        lvl_full = detector_db(y, sr, "rms", 0.005)
        for i, (name, d) in enumerate(dyn):
            bp = outs[1 + i]
            gdb = dynamic_band_db(lvl_full, detector_db(bp, sr, "rms", 0.005), d["rel_db"],
                                  ratio=d.get("ratio", 3.0), max_cut_db=d["max_cut_db"])
            info[f"{name}_max_db"] = _r(-gdb.min())
            info[f"{name}_active_pct"] = _r(100.0 * np.mean(gdb < -1.0), 1)
            y = y + (grid_to_samples(db_to_gain(gdb), n, hop, np.float32) - np.float32(1.0)) * bp
    comp = pre.get("comp")
    if comp:
        lvl = detector_db(y, sr, "rms", 0.01)
        gdb = compressor_db(lvl, comp["threshold_db"], comp["ratio"], comp.get("attack", 0.005),
                            comp.get("release", 0.1), comp.get("knee_db", 6.0), comp.get("makeup_db", 0.0))
        info["comp_max_gr_db"] = _r(comp.get("makeup_db", 0.0) - gdb.min())
        y = y * grid_to_samples(db_to_gain(gdb), n, hop, np.float32)
    if pre.get("crest_db") is not None:  # opt-in transient limiter (``mix_overrides`` ``crest_db``) for peaky
        # voices (buzzy low pulse trains, shouts): peaks stay within crest_db of the voice's loud
        # (95th-percentile 10 ms RMS) level, as for drums, so they do not make the master limiter duck the mix
        lvl = 10 * np.log10(np.maximum(_moving_mean(block_power(y, hop), 10), 1e-20))
        ceil_db = relative_threshold(lvl, float(pre["crest_db"]))
        g = limiter_gain(np.abs(y), sr, 10 ** (ceil_db / 20.0), lookahead=0.001, release=0.03, release_slow=0.0)
        if g.min() < 1.0:
            y = y * g.astype(np.float32)
            info["transient_max_db"] = _r(-gain_to_db(g.min()))
    return y.astype(np.float32, copy=False), info


def _bp_power(x_mono: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Grid block power of ``x_mono`` band-limited to ``lo..hi`` Hz (4th-order slopes)."""
    y = eq(x_mono, [("hp", lo, 0, 0.7071, 4), ("lp", hi, 0, 0.7071, 4)], sr, threads=mix_threads())
    return block_power(y, grid_hop(sr))


def _vbr(pv: np.ndarray, pb: np.ndarray, act: np.ndarray | None) -> float | None:
    """Vocal-to-band ratio (dB) from grid band powers, over the blocks where the vocals sing."""
    nb = min(len(pv), len(pb))
    pv, pb = pv[:nb], pb[:nb]
    m = np.ones(nb, dtype=bool) if act is None else (np.asarray(act)[:nb] > 0.5)
    if len(m) < nb:
        m = np.pad(m, (0, nb - len(m)))
    m &= pb > 1e-12
    if m.sum() < 50:
        return None
    return float(10 * np.log10(max(pv[m].sum(), 1e-30) / pb[m].sum()))


def vocal_to_band_ratio(vocal: np.ndarray, band: np.ndarray, sr: int, act: np.ndarray | None = None,
                        lo: float = VBR_BAND[0], hi: float = VBR_BAND[1]) -> float | None:
    """Vocal-to-band power ratio (dB) in ``lo..hi`` Hz while the vocals sing (grid activity > 0.5;
    everywhere when ``act`` is None). None when they never overlap."""
    mono = lambda a: np.asarray(a).mean(axis=1) if np.ndim(a) == 2 else np.asarray(a)  # noqa: E731
    return _vbr(_bp_power(mono(vocal), sr, lo, hi), _bp_power(mono(band), sr, lo, hi), act)


def _vocal_job(s: Source, sr: int, n: int, S: "_Settings", band_mode: bool, master_eq: list,
               role: dict | None = None, lead_env: np.ndarray | None = None) -> dict:
    """One vocal source through its channel: the role's chain (``role``, from
    :func:`.vocal_chain.source_role`) or the channel strip, the in-channel effects of
    ``mix_overrides`` ``fx``, fader and pan. Returns the stem and the mono channel (for the
    sends) as :class:`Sparse`."""
    from . import vocal_chain

    scale = 1.0
    x = _dense(s.audio)
    fx = s.overrides.get("fx") or None
    if role is not None:  # a track with a role in the mix (vocal_chain.py) gets that role's chain, not the strip
        y, si = vocal_chain.apply_role(x, sr, role["role"], role.get("dial"), lead_env, role.get("copy_of"),
                                       float(s.overrides.get("pan", s.pan)), fx)
        scale = float(si.get("reverb_scale", 1.0))
        if master_eq:
            y = eq(y, master_eq, sr).astype(np.float32)
    else:
        y, si = _processed_vocal(s, sr, band_mode, S.vocal_chain and s.overrides.get("chain", True), master_eq, x)
        if fx:
            y, fi = vocal_chain.apply_fx(y, sr, fx, lead_env)
            si = {**si, **fi}
    del x
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    mono = Sparse.from_dense(y[:n], SILENCE)
    del y
    gl, gr = pan_gains(s.overrides.get("pan", s.pan))
    g = 10 ** ((s.gain_db + float(s.overrides.get("gain_db", 0.0))) / 20.0)
    stem = Sparse(n, 2)
    for a, seg in mono.segments:
        st = np.empty((len(seg), 2), dtype=np.float32)
        np.multiply(seg, np.float32(g * gl), out=st[:, 0])
        np.multiply(seg, np.float32(g * gr), out=st[:, 1])
        stem.segments.append((a, st))
    return {"stem": stem, "mono": mono, "g": g, "info": si, "reverb_scale": scale}


def _low_end_trim(sources: list[Source], sr: int, master_eq: list, S: "_Settings") -> tuple[float, float | None]:
    """Predicted sub balance of the band and the low-shelf trim (dB, <= 0) that brings its 30-60 Hz
    octave to at most ``S.sub_max_db`` above its 120-240 Hz octave (at most 4 dB of cut). The band
    spectrum is estimated from frames sampled across each raw stem, weighted by the stem's role EQ
    and gain (cheap; no filtering pass)."""
    if S.sub_max_db is None or not sources:
        return 0.0, None
    nfft = 1 << int(round(math.log2(max(256, 0.19 * sr))))
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    win = np.hanning(nfft)
    total = np.zeros(len(f))
    for s in sources:
        x = _dense(s.audio)
        n = len(x)
        if n < nfft:
            continue
        k = min(256, max(1, n // nfft))
        starts = np.linspace(0, n - nfft, k).astype(np.int64)
        frames = x[starts[:, None] + np.arange(nfft)[None, :]]
        del x
        if frames.ndim == 3:
            frames = frames.mean(axis=2)
        frames = frames.astype(np.float64) * win
        P = np.mean(np.abs(np.fft.rfft(frames, axis=1)) ** 2, axis=0)
        pre = _merge(role_preset(s.role, s.name), s.overrides)
        bands = [*pre.get("eq", []), *(s.overrides.get("eq_add") or []), *master_eq]
        g = 10 ** ((s.gain_db + float(s.overrides.get("gain_db", 0.0))) / 10.0)
        total += P * np.abs(eq_response(bands, sr, f)) ** 2 * g
    sub = total[(f >= 30) & (f < 60)].sum()
    lowmid = total[(f >= 120) & (f < 240)].sum()
    if sub <= 0 or lowmid <= 0:
        return 0.0, None
    ratio = float(10 * np.log10(sub / lowmid))
    return -float(np.clip(ratio - S.sub_max_db, 0.0, 4.0)), ratio


def _band_job(s: Source, sr: int, n: int, S: "_Settings", band_mode: bool, master_eq: list, act: np.ndarray,
              have_vocals: bool, kicks: Sequence[float], bus_eq: list = ()) -> dict | None:
    hop = grid_hop(sr)
    nb = max(1, -(-n // hop))
    F = dense_size(sr) // 2 + 1
    pre = _merge(role_preset(s.role, s.name), s.overrides)
    if isinstance(s.audio, Sparse) and not s.audio.segments:
        return None
    x = _dense(s.audio)
    if x.ndim == 1:
        x = np.stack([x, x], axis=1)
    if len(x) != n:
        x = np.pad(x[:n], ((0, max(0, n - len(x))), (0, 0)))
    if not np.any(x):
        return None
    gl, gr = pan_gains(s.overrides.get("pan", s.pan))
    L = x[:, 0].astype(np.float32) * np.float32(gl)
    R = x[:, 1 if x.shape[1] > 1 else 0].astype(np.float32) * np.float32(gr)
    mid = (L + R) * np.float32(0.5)
    side = (L - R) * np.float32(0.5)
    del L, R, x
    bands = [*pre.get("eq", []), *(s.overrides.get("eq_add") or []), *bus_eq, *master_eq]
    Hd = eq_dense(bands, sr) if bands else np.ones(F, dtype=np.complex128)
    Hs = Hd * float(pre.get("width", 1.0)) * float(S.band_width)
    if pre.get("mono_below"):
        Hs = Hs * eq_dense([("hp", pre["mono_below"], 0, 0.7071, 2)], sr)
    carve_amt = float(pre.get("carve", 0.0)) * S.vocal_carve_db
    use_carve = band_mode and have_vocals and carve_amt > 0
    H = np.zeros((3 if use_carve else 2, 2, F), dtype=np.complex128)
    H[0, 0], H[1, 1] = Hd, Hs
    if use_carve:
        H[2, 0] = Hd * eq_dense([("bandpass", CARVE_HZ, 0, CARVE_Q)], sr)
    outs = filter_mimo(np.stack([mid, side]), H, sr)
    del mid, side
    m_out, s_out = outs[0], outs[1]
    gdb = np.zeros(nb)
    comp = pre.get("comp")
    sinfo: dict[str, Any] = {"name": s.name, "role": s.role}
    if pre.get("crest_db") is not None:  # transient limiter (sample accurate, 1 ms look-ahead)
        p = block_power(m_out, hop) + block_power(s_out, hop)
        lvl = 10 * np.log10(np.maximum(_moving_mean(p, 10), 1e-20))
        ceil_db = relative_threshold(lvl, float(pre["crest_db"]))
        env = np.abs(m_out) + np.abs(s_out)  # >= max(|L|, |R|)
        g = limiter_gain(env, sr, 10 ** (ceil_db / 20.0), lookahead=0.001, release=0.03, release_slow=0.0)
        if g.min() < 1.0:
            g = g.astype(np.float32)
            m_out *= g
            s_out *= g
            if len(outs) > 2:
                outs[2] *= g
            sinfo["transient_max_db"] = _r(-gain_to_db(g.min()))
        del env, g
    if comp:
        if comp.get("detector", "rms") == "peak":
            lvl = gain_to_db(np.maximum(block_peak(m_out, hop), block_peak(s_out, hop)) * 2.0, 1e-10)
        else:
            p = block_power(m_out, hop) + block_power(s_out, hop)  # = mean power of L and R
            lvl = 10 * np.log10(np.maximum(_moving_mean(p, max(1, int(round(0.01 / GRID_S)))), 1e-20))
        thr = relative_threshold(lvl, comp.get("offset_db", -5.0))
        c_db = compressor_db(lvl, thr, comp["ratio"], comp.get("attack", 0.01), comp.get("release", 0.1),
                             comp.get("knee_db", 6.0))
        gdb += c_db[:nb]
        sinfo["comp_max_gr_db"] = _r(-c_db.min())
    if s.sidechain and len(kicks):
        depth = float(s.overrides.get("sidechain_db", S.sidechain_db * float(pre.get("sc", 1.0))))
        gdb += kick_duck_db(kicks, nb, sr, depth, release=S.sidechain_release)
        sinfo["sidechain_db"] = _r(depth)
    duck = float(pre.get("duck", 0.0)) * S.band_duck_db
    if have_vocals and duck > 0:
        gdb += -duck * act
        sinfo["duck_db"] = _r(duck)
    g_static = 10 ** ((s.gain_db + float(s.overrides.get("gain_db", 0.0))) / 20.0)
    gs = grid_to_samples(db_to_gain(gdb) * g_static, n, hop, np.float32)
    m_out *= gs
    s_out *= gs
    if use_carve:
        cg = grid_to_samples(db_to_gain(-carve_amt * act) - 1.0, n, hop, np.float32)
        cg *= gs
        m_out += outs[2] * cg
        sinfo["carve_db"] = _r(carve_amt)
        del cg
    del gs
    st = np.empty((n, 2), dtype=np.float32)
    np.add(m_out, s_out, out=st[:, 0])
    np.subtract(m_out, s_out, out=st[:, 1])
    rs = float(s.overrides.get("reverb", float(pre.get("reverb", 0.0)) * S.band_reverb))
    send = Sparse.from_dense(m_out * np.float32(rs), SILENCE) if rs > 0 else None
    return {"stem": Sparse.from_dense(st, SILENCE), "send": send, "info": sinfo}


def mix_sources(sources: list[Source], sr: int, *, band_mode: bool, settings: Any, kicks: Sequence[float] = (),
                bpm: float = 120.0, reverb_room: float = 0.55, reverb_wet: float = 1.0,
                progress=None, threads: int | None = None, fixed: dict | None = None,
                consume: bool = False) -> MixResult:
    """Mix vocal and backing sources into stems, a dry sum and the effect returns.

    Vocal path: channel strip (``settings.vocal_chain``; for a source with a role,
    ``Source.vocal_role``, that an engine of ``vocal_chain.ROLE_ENGINES`` sang, the role's chain
    and effects instead) -> the in-channel effects of ``mix_overrides`` ``fx`` (``eq``,
    ``duck_db`` under the tracks with the role ``lead``, ``slap``, ``room``; for a source with a
    role they replace the role's own, entry by entry) -> fader / pan -> stem; sends to the
    reverb (track ``reverb_send``) and the ping-pong delay (``settings.vocal_delay``).
    Backing path (``band_mode``): balance -> mid/side -> role EQ (mid) and EQ x width with mono
    bass (side), one FFT pass -> role compressor -> kick sidechain duck (stems flagged
    ``sidechain``) -> vocal-keyed ducking and a 1.5-4 kHz carve of the mid -> gain -> instrument
    bus (auto vocal balance, bus compressor). The master EQ is folded into every source.
    ``settings`` is a ``MasterSettings`` (missing fields fall back to defaults). Sources are
    processed on :func:`mix_threads` threads and summed in order (the result does not depend on
    the thread count). ``fixed`` may hold ``band_gain_db``, ``bus_threshold_db`` and
    ``low_end_trim_db`` from an earlier (full) mix to use instead of measuring them (section
    previews then match the full song); ``info`` reports the values used.

    A source's ``audio`` may be a :class:`Sparse`; the stems come back in a :class:`StemStore`.
    Only the sources being processed (one per thread) are held at full length. With ``consume``
    the mixer lets go of each source's audio once it is mixed (``Source.audio`` is then None).
    """
    fixed = fixed or {}
    S = _Settings(settings)
    threads = mix_threads() if threads is None else max(1, int(threads))
    n = max((len(s.audio) for s in sources), default=0)
    hop = grid_hop(sr)
    nb = max(1, -(-n // hop))
    info: dict[str, Any] = {}
    master_eq = list(S.master_eq) if band_mode else []
    stems = StemStore(n)
    groups: dict[str, str] = {}
    rev_send = np.zeros(n, dtype=np.float32)  # vocal reverb send (mono)
    rev_band = np.zeros(n, dtype=np.float32)  # band reverb send, scaled by the band gain later
    dly_send = np.zeros(n, dtype=np.float32)
    vocal_bus = np.zeros((n, 2), dtype=np.float32)
    vinfo = {}
    vocals = [s for s in sources if s.kind == "vocal"]
    backing = [s for s in sources if s.kind != "vocal"]
    done = 0
    roles: dict = {}
    env = None
    if vocals:
        from . import vocal_chain

        if S.vocal_chain:
            roles = {s.key: r for s in vocals
                     if s.overrides.get("chain", True) and (r := vocal_chain.source_role(s)) is not None}
        ducked = any(vocal_chain.ROLE_FX.get(r["role"], {}).get("duck_db") for r in roles.values()) \
            or any((s.overrides.get("fx") or {}).get("duck_db") for s in vocals)
        if ducked:  # where the lead sings: what the ad-libs, the crowd and tracks with fx.duck_db duck under
            env = vocal_chain.leads_env(((_dense(s.audio), s.gain_db) for s in vocals
                                         if vocal_chain.parse_role(s.vocal_role)[0] == "lead"), sr, n)
    for s, r in zip(vocals, _run_jobs(lambda s: _vocal_job(s, sr, n, S, band_mode, master_eq, roles.get(s.key), env),
                                      vocals, threads)):
        done += 1
        if progress:
            progress(done / max(1, len(sources)), f"mixed {s.name}")
        st, y, g = r["stem"], r["mono"], r["g"]
        st.add_to(vocal_bus)
        rs = float(s.overrides.get("reverb", s.reverb_send * r.get("reverb_scale", 1.0)))
        if rs > 0:
            y.add_to(rev_send, g * rs)
        ds = float(s.overrides.get("delay", s.delay_send))
        if ds > 0 and S.vocal_delay > 0:
            y.add_to(dly_send, g * ds)
        stems.put(s.key, st)
        groups[s.key] = "vocal"
        if r["info"]:
            vinfo[s.name] = r["info"]
        del r, st, y
        if consume:
            s.audio = None
    if vinfo:
        info["vocal_strips"] = vinfo
    have_vocals = bool(vocals) and bool(np.any(vocal_bus))
    act = activity(vocal_bus, sr, threshold_db=S.activity_threshold_db) if have_vocals else np.zeros(nb)
    act = act[:nb] if len(act) >= nb else np.pad(act, (0, nb - len(act)))

    # ---------------------------------------------------------------- backing stems
    band_bus = np.zeros((n, 2), dtype=np.float32)
    band_keys = []
    binfo = []
    bus_eq: list = []
    if band_mode and backing:
        if fixed.get("low_end_trim_db") is not None:
            trim = float(fixed["low_end_trim_db"])
        else:
            trim, sub_ratio = _low_end_trim(backing, sr, master_eq, S)
            if sub_ratio is not None:
                info["band_sub_vs_lowmid_db"] = _r(sub_ratio)
        info["low_end_trim_db"] = trim
        if trim < -0.1:  # a bounded low shelf on the whole band (folded into every stem's EQ)
            bus_eq = [("lowshelf", 70.0, trim, 0.7071)]
    job = lambda s: _band_job(s, sr, n, S, band_mode, master_eq, act, have_vocals, kicks, bus_eq)  # noqa: E731
    for s, r in zip(backing, _run_jobs(job, backing, threads)):
        done += 1
        if progress:
            progress(done / max(1, len(sources)), f"mixed {s.name}")
        if consume:
            s.audio = None
        if r is None:
            continue
        r["stem"].add_to(band_bus)
        if r["send"] is not None:
            r["send"].add_to(rev_band)
        stems.put(s.key, r["stem"])
        groups[s.key] = "backing"
        band_keys.append(s.key)
        binfo.append(r["info"])
        del r
    if binfo:
        info["backing"] = binfo

    # ---------------------------------------------------------------- instrument bus
    # bus compressor first (its threshold is gain-staged, so it does not depend on the band gain),
    # then the band gain that puts the vocals `vocal_balance_db` above the compressed band
    band_gain_db = float(S.backing_gain_db)
    vbr_before = None
    have_band = bool(band_keys) and bool(np.any(band_bus))
    bus_env = np.zeros(nb)
    if have_band and S.band_glue > 0:
        lvl = detector_db(band_bus, sr, "rms", 0.02)
        thr = fixed.get("bus_threshold_db")
        if thr is None:
            thr = relative_threshold(lvl, -2.0 - 3.0 * S.band_glue)
        bus_env = compressor_db(lvl, thr, 1.0 + 1.5 * S.band_glue, 0.02, 0.18, 6.0)[:nb]
        info["band_bus_gr_max_db"] = _r(-bus_env.min())
        info["bus_threshold_db"] = float(thr)
    if have_band and have_vocals:
        pv = _bp_power(vocal_bus.mean(axis=1), sr, *VBR_BAND)
        pb = _bp_power(band_bus.mean(axis=1), sr, *VBR_BAND)
        pb = pb * db_to_gain(2 * bus_env[:len(pb)])
        vbr_before = _vbr(pv, pb, act)
        if fixed.get("band_gain_db") is not None:
            band_gain_db = float(fixed["band_gain_db"])
        elif vbr_before is not None and S.vocal_balance_db is not None:
            band_gain_db += float(np.clip(vbr_before - S.vocal_balance_db, -30.0, 12.0))
    elif have_band and fixed.get("band_gain_db") is not None:
        band_gain_db = float(fixed["band_gain_db"])
    if have_band:
        gb = grid_to_samples(db_to_gain(bus_env + band_gain_db), n, hop, np.float32)
        band_bus *= gb[:, None]
        stems.scale(gb, band_keys)
        del gb
        rev_send += rev_band * np.float32(10 ** (band_gain_db / 20.0))
        if have_vocals and vbr_before is not None:
            info["vocal_to_band_db"] = _r(_vbr(pv, pb * db_to_gain(2 * band_gain_db), act))
            info["vocal_to_band_db_before_balance"] = _r(vbr_before)
    del rev_band
    info["band_gain_db"] = band_gain_db if have_band else None

    # ---------------------------------------------------------------- effect returns
    from . import effects

    wet = np.zeros((n, 2))
    if reverb_wet > 0 and np.any(rev_send):
        rv = convolve_reverb(rev_send, effects.reverb_ir(sr, reverb_room), threads)[:n]
        rv *= np.float32(0.6 * reverb_wet)  # effects.apply_reverb's return level
        if band_mode:
            rv = eq(rv, [*RETURN_EQ, *master_eq], sr, threads)
        wet[:len(rv)] += rv
        del rv
    if S.vocal_delay > 0 and np.any(dly_send):
        d = pingpong_delay(dly_send, sr, S.delay_beats * 60.0 / max(bpm, 1e-3), S.delay_feedback)[:n]
        if band_mode and master_eq:
            d = eq(d, master_eq, sr, threads)
        if have_vocals and S.delay_duck_db > 0:  # the repeats step back while the vocals sing
            d *= grid_to_samples(db_to_gain(-S.delay_duck_db * act), len(d), hop, np.float32)[:, None]
        wet[:len(d)] += d * S.vocal_delay
        info["delay_s"] = _r(S.delay_beats * 60.0 / max(bpm, 1e-3), 4)
    dry = vocal_bus
    dry += band_bus
    return MixResult(stems, groups, dry, wet, act if have_vocals else None, info)


def glue_env_db(x: np.ndarray, sr: int, amount: float, threshold_db: float | None = None) -> np.ndarray:
    """Master glue compressor curve (dB, grid): gentle ratio (1 + 2 x amount), threshold
    ``3 + 3 x amount`` dB under the loud (95th percentile) 50 ms RMS level, 30 ms attack, 250 ms
    release, 8 dB knee: about 1-3 dB on the loudest parts at amount 0.5."""
    if amount <= 0:
        return np.zeros(max(1, -(-len(x) // grid_hop(sr))))
    lvl = detector_db(x, sr, "rms", 0.05)
    thr = relative_threshold(lvl, -(3.0 + 3.0 * amount)) if threshold_db is None else threshold_db
    return compressor_db(lvl, thr, 1.0 + 2.0 * amount, 0.03, 0.25, 8.0, max_gr_db=6.0)


def glue_threshold_db(x: np.ndarray, sr: int, amount: float) -> float:
    """The gain-staged threshold :func:`glue_env_db` uses for ``x``."""
    return relative_threshold(detector_db(x, sr, "rms", 0.05), -(3.0 + 3.0 * amount))


class _Settings:
    """Mixer settings with defaults, read from a MasterSettings (or anything with attributes)."""

    DEFAULTS = {
        "vocal_chain": True, "vocal_delay": 0.0, "delay_beats": 0.5, "delay_feedback": 0.35, "delay_duck_db": 6.0,
        "band_duck_db": 1.5, "vocal_carve_db": 3.0, "sidechain_db": 6.0, "sidechain_release": 0.16,
        "band_width": 1.0, "band_glue": 0.5, "band_reverb": 0.5, "vocal_balance_db": 4.0, "backing_gain_db": 0.0,
        "master_eq": None, "activity_threshold_db": -45.0, "sub_max_db": 4.0,
    }

    def __init__(self, s: Any):
        for k, v in self.DEFAULTS.items():
            val = getattr(s, k, v) if s is not None else v
            setattr(self, k, v if val is None and k not in ("vocal_balance_db", "sub_max_db") else val)
        if self.master_eq is None:
            self.master_eq = list(MASTER_EQ)


def master_bus(dry: np.ndarray, wet: np.ndarray, sr: int, *, glue: float = 0.5, target_lufs: float | None = -11.0,
               ceiling_db: float = -1.0, limiter: bool = True, gain_db: float = 0.0,
               fixed: dict | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """The master chain of a song: glue compressor -> loudness gain -> true-peak limiter.

    ``dry`` (the sum of the stems) and ``wet`` (effect returns) are (n, 2). The master EQ is
    already folded into every source by :func:`mix_sources`. ``gain_db`` trims after the loudness
    target (so the target becomes ``target_lufs + gain_db``). ``fixed`` may hold
    ``glue_threshold_db`` and ``master_gain_db`` (the static gain before the limiter, trim
    included) from an earlier full mix, used instead of measuring this audio. Returns
    ``(master, stem_gain, info)`` where ``stem_gain`` (n,) is the gain the stems need to stay
    summing to the dry part of the master (glue and loudness gain; the limiter is not applied to
    stems)."""
    fixed = fixed or {}
    x = dry + wet
    n = len(x)
    hop = grid_hop(sr)
    info: dict[str, Any] = {}
    if glue > 0:
        thr = fixed.get("glue_threshold_db")
        thr = glue_threshold_db(x, sr, glue) if thr is None else float(thr)
        genv = glue_env_db(x, sr, glue, thr)
        info["glue_gr_max_db"] = _r(-genv.min()) if len(genv) else 0.0
        info["glue_threshold_db"] = thr
        x = x * grid_to_samples(db_to_gain(genv), n, hop)[:, None]
        gg = grid_to_samples(db_to_gain(genv), n, hop)
    else:
        info["glue_gr_max_db"] = 0.0
        gg = np.ones(n)
    target = None
    if fixed.get("master_gain_db") is not None:
        G = 10 ** (float(fixed["master_gain_db"]) / 20.0)
        y, li = master_to_loudness(x * G, sr, None, ceiling_db, limiter=limiter)
        li["gain"] = G
        li["gain_db"] = float(fixed["master_gain_db"])
    elif target_lufs is None:
        G = 10 ** (gain_db / 20.0)
        y, li = master_to_loudness(x * G, sr, None, ceiling_db, limiter=limiter)
        li["gain"] = G
        li["gain_db"] = float(gain_db)
    else:
        target = target_lufs + gain_db
        y, li = master_to_loudness(x, sr, target, ceiling_db, limiter=limiter)
        G = li["gain"]
    info.update({"master_gain_db": float(20 * np.log10(G)), "limiter_max_db": li["limiter_max_db"],
                 "limiter_mean_db": li["limiter_mean_db"], "target_lufs": _r(target),
                 "target_reached": li["target_reached"], "true_peak": li["true_peak"]})
    return y, gg * G, info
