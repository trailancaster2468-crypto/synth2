"""Audio analysis used to verify renders: f0 tracking, levels, pitch accuracy.

These functions let an AI (or a test) "listen" to a render objectively:
did every note land on pitch, is anything clipping, how much is voiced, etc.
"""
from __future__ import annotations

import numpy as np

from .timing import midi_to_note


def _frames(x: np.ndarray, size: int, hop: int) -> np.ndarray:
    n = 1 + max(0, (len(x) - size) // hop)
    idx = np.arange(size)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


def estimate_f0(mono: np.ndarray, sr: int, hop_s: float = 0.005, fmin: float = 60.0, fmax: float = 1200.0,
                threshold: float = 0.15) -> tuple[np.ndarray, np.ndarray]:
    """YIN pitch tracker. Returns (frame centre times in s, f0 in Hz with 0 for unvoiced)."""
    x = np.asarray(mono, dtype=np.float64)
    if x.ndim == 2:
        x = x.mean(axis=1)
    tau_max = int(sr / fmin) + 2
    tau_min = max(2, int(sr / fmax))
    half = max(tau_max, int(0.025 * sr))
    W = half + tau_max
    hop = max(1, int(hop_s * sr))
    if len(x) < W:
        x = np.pad(x, (0, W - len(x)))
    fr = _frames(x, W, hop)
    nfft = 1 << int(np.ceil(np.log2(2 * W)))
    A = np.fft.rfft(fr[:, :half], nfft)
    Bf = np.fft.rfft(fr, nfft)
    cross = np.fft.irfft(np.conj(A) * Bf, nfft)[:, :tau_max]
    sq = fr ** 2
    cs = np.concatenate([np.zeros((len(fr), 1)), np.cumsum(sq, axis=1)], axis=1)
    e0 = cs[:, half][:, None]
    taus = np.arange(tau_max)
    et = cs[:, taus + half] - cs[:, taus]
    d = e0 + et - 2 * cross
    d[:, 0] = 0
    cum = np.cumsum(d[:, 1:], axis=1) / np.arange(1, tau_max)[None, :]
    cmnd = np.ones_like(d)
    cmnd[:, 1:] = d[:, 1:] / np.maximum(cum, 1e-12)
    f0 = np.zeros(len(fr))
    energy = e0[:, 0] / half
    silent = energy < 1e-7
    for i in range(len(fr)):
        if silent[i]:
            continue
        c = cmnd[i]
        below = np.nonzero(c[tau_min:] < threshold)[0]
        if len(below) == 0:
            # YIN fallback: accept the global minimum when it is still clearly periodic
            # (breathy voices rarely dip below the strict threshold)
            cmin = float(np.min(c[tau_min:]))
            if cmin > 0.35:
                continue
            # first dip that comes close to the global minimum (avoids period multiples)
            t = int(np.nonzero(c[tau_min:] <= cmin + 0.06)[0][0]) + tau_min
        else:
            t = below[0] + tau_min
        while t + 1 < tau_max and c[t + 1] < c[t]:
            t += 1
        if 1 <= t < tau_max - 1:
            a, b, cc = c[t - 1], c[t], c[t + 1]
            den = a - 2 * b + cc
            shift = 0.5 * (a - cc) / den if abs(den) > 1e-12 else 0.0
        else:
            shift = 0.0
        f0[i] = sr / (t + shift)
    times = (np.arange(len(fr)) * hop + half / 2) / sr
    return times, f0


def analyze_audio(audio: np.ndarray, sr: int) -> dict:
    a = np.asarray(audio, dtype=np.float64)
    mono = a.mean(axis=1) if a.ndim == 2 else a
    peak = float(np.max(np.abs(a))) if a.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if a.size else 0.0
    db = lambda v: round(20 * np.log10(max(v, 1e-9)), 2)  # noqa: E731
    hop = int(0.02 * sr)
    if len(mono) > hop:
        blocks = mono[: len(mono) // hop * hop].reshape(-1, hop)
        blk_db = 10 * np.log10(np.mean(blocks ** 2, axis=1) + 1e-12)
        silence_ratio = float(np.mean(blk_db < -60))
        loud = blk_db[blk_db > -60]
        lufs_like = float(np.mean(loud)) if loud.size else -120.0
    else:
        silence_ratio, lufs_like = 1.0, -120.0
    times, f0 = estimate_f0(mono, sr)
    voiced = f0 > 0
    spec = np.abs(np.fft.rfft(mono[: min(len(mono), sr * 30)]))
    freqs = np.fft.rfftfreq(min(len(mono), sr * 30), 1 / sr)
    centroid = float(np.sum(spec * freqs) / max(np.sum(spec), 1e-12)) if spec.size else 0.0
    return {
        "duration_s": round(len(mono) / sr, 3),
        "peak_db": db(peak),
        "rms_db": db(rms),
        "active_level_db": round(lufs_like, 2),
        "crest_factor_db": round(db(peak) - db(rms), 2),
        "clipped_samples": int(np.sum(np.abs(a) >= 0.999)),
        "silence_ratio": round(silence_ratio, 3),
        "voiced_ratio": round(float(np.mean(voiced)) if len(voiced) else 0.0, 3),
        "median_f0_hz": round(float(np.median(f0[voiced])), 2) if voiced.any() else None,
        "f0_range": [midi_to_note(69 + 12 * np.log2(np.percentile(f0[voiced], 5) / 440)),
                     midi_to_note(69 + 12 * np.log2(np.percentile(f0[voiced], 95) / 440))] if voiced.sum() > 10 else None,
        "spectral_centroid_hz": round(centroid, 1),
    }


def pitch_accuracy(project, result, max_list: int = 10) -> dict:
    """Compare the sung f0 of each rendered stem with the notes it should sing.

    For every note longer than 0.12 s the median f0 over the central part of the
    note is compared to the note's pitch (+detune).  Vibrato averages out.
    """
    sr = result.sample_rate
    out = {"tracks": []}
    b2s = project.tempo.beat_to_sec
    for tr in project.tracks:
        stem = result.stems.get(tr.id)
        if stem is None:
            continue
        times, f0 = estimate_f0(stem.mean(axis=1), sr)
        times = times + result.start_seconds
        errors = []
        missing = []
        for n in tr.notes:
            if (n.lyric or "").strip().lower() in ("br", "breath", "", "r"):
                continue
            a, b = b2s(n.onset), b2s(n.end)
            if b - a < 0.12:
                continue
            # skip the transition into the note; keep the rest (short notes end in consonants)
            lo, hi = a + min(0.25 * (b - a), 0.12), b - 0.1 * (b - a)
            m = (times >= lo) & (times <= hi)
            if not m.any():
                continue
            seg = f0[m]
            v = seg[seg > 0]
            if len(v) < max(3, 0.3 * m.sum()):
                missing.append({"id": n.id, "beat": n.onset, "note": midi_to_note(n.pitch), "lyric": n.lyric})
                continue
            target = n.pitch + float(n.attrs.get("detune", 0) or 0) / 100.0
            sung = 69 + 12 * np.log2(np.median(v) / 440.0)
            cents = float((sung - target) * 100)
            # octave errors from the tracker are not pitch errors of the singer
            if abs(abs(cents) - 1200) < 60:
                cents = cents - np.sign(cents) * 1200
            errors.append({"id": n.id, "beat": n.onset, "note": midi_to_note(n.pitch), "lyric": n.lyric,
                           "cents": round(cents, 1)})
        abs_err = np.array([abs(e["cents"]) for e in errors]) if errors else np.zeros(0)
        out["tracks"].append({
            "id": tr.id, "name": tr.name,
            "notes_checked": len(errors),
            "mean_abs_cents": round(float(abs_err.mean()), 1) if len(abs_err) else None,
            "max_abs_cents": round(float(abs_err.max()), 1) if len(abs_err) else None,
            "within_25_cents": round(float(np.mean(abs_err <= 25)), 3) if len(abs_err) else None,
            "worst": sorted(errors, key=lambda e: -abs(e["cents"]))[:max_list],
            "unvoiced_notes": missing[:max_list],
        })
    return out
