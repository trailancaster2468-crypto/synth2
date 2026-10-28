"""Segments + control curves -> frame-level synthesis parameters.

Targets are written per segment and then smoothed, which yields natural
coarticulation: formants glide between phonemes over ~30 ms while consonant
amplitudes stay crisp.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..phonemes.inventory import Phoneme
from .planner import Plan, Segment

NEUTRAL = (520.0, 1450.0, 2500.0)


@dataclass
class FrameParams:
    t0: float
    frame_period: float
    f0: np.ndarray  # Hz
    formants: np.ndarray  # (n, 5) Hz
    bandwidths: np.ndarray  # (n, 5) Hz
    voiced: np.ndarray  # linear amplitude of the voiced (periodic) source
    asp: np.ndarray  # linear amplitude of unvoiced aspiration (h, stop release, breath)
    breath: np.ndarray  # breathiness amount (-1..1) -> aspiration riding on voicing
    fric: np.ndarray  # linear amplitude of frication noise
    fric_fc: np.ndarray
    fric_bw: np.ndarray
    nasal: np.ndarray  # 0..1 nasal coupling
    nasal_zero: np.ndarray  # Hz
    tension: np.ndarray  # -1..1
    loudness_db: np.ndarray
    voicing: np.ndarray  # 0..1

    @property
    def n(self) -> int:
        return len(self.f0)

    def times(self) -> np.ndarray:
        return self.t0 + np.arange(self.n) * self.frame_period


def _hann_smooth(x: np.ndarray, width_frames: float) -> np.ndarray:
    w = int(round(width_frames))
    if w < 2:
        return x
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    pad = w
    xp = np.pad(x, (pad, pad), mode="edge")
    return np.convolve(xp, k, mode="same")[pad:-pad]


def _fill_nan(x: np.ndarray, default: float) -> np.ndarray:
    m = np.isnan(x)
    if m.all():
        return np.full_like(x, default)
    if m.any():
        idx = np.arange(len(x))
        x = x.copy()
        x[m] = np.interp(idx[m], idx[~m], x[~m])
    return x


def _segment_formants(s: Segment) -> tuple[float, float, float] | None:
    p = s.ph
    if p.kind == "silence":
        return None
    neighbour = s.next_vowel or s.prev_vowel
    nf = neighbour.formants if neighbour is not None and neighbour.kind == "vowel" else NEUTRAL
    if p.coarticulate:
        return nf
    if p.kind in ("stop", "fricative", "affricate"):
        w = 0.55 if p.kind == "stop" else 0.6
        return tuple(w * a + (1 - w) * b for a, b in zip(p.formants, nf))  # type: ignore[return-value]
    if p.kind == "flap":
        return tuple(0.6 * a + 0.4 * b for a, b in zip(p.formants, nf))  # type: ignore[return-value]
    return p.formants


def _levels(s: Segment, voice) -> tuple[float, float, float]:
    """(voiced, aspiration, frication) linear amplitudes for a segment, relative to a vowel = 1."""
    p: Phoneme = s.ph
    db = lambda x: 10.0 ** (x / 20.0)  # noqa: E731
    k = p.kind
    if k == "silence":
        return 0.0, 0.0, 0.0
    if k == "breath":
        return 0.0, db(-22) * s.level, 0.0
    if k == "aspirate":
        return 0.0, db(-15), 0.0
    if k in ("vowel", "diphthong", "nasal", "liquid", "glide", "flap"):
        return db(p.amp_db), 0.0, 0.0
    if k == "fricative":
        fric = db(p.fric[2] - FRIC_OFFSET_DB) if p.fric else 0.0
        v = db(p.amp_db) if p.voiced else 0.0
        return v, 0.0, fric
    if k == "stop":
        g = p.fric[2] if p.fric else -12.0
        if s.part == "closure":
            return (db(-28) if p.voiced else 0.0), 0.0, 0.0
        if s.part == "burst":
            return (db(-18) if p.voiced else 0.0), db(-26), db(g - 6)
        if s.part == "asp":
            return 0.0, db(-17), db(g - 22)
    return 0.0, 0.0, 0.0


FRIC_OFFSET_DB = 11.0


def build_params(plan: Plan, t: np.ndarray, voice, pitch_semitones: np.ndarray, loud_db: np.ndarray,
                 tension: np.ndarray, breath: np.ndarray, gender: np.ndarray, tone_shift: np.ndarray,
                 voicing: np.ndarray) -> FrameParams:
    n = len(t)
    fp = (t[1] - t[0]) if n > 1 else 0.003
    t0 = t[0]
    F = np.full((n, 3), np.nan)
    voiced = np.zeros(n)
    asp = np.zeros(n)
    fric = np.zeros(n)
    fric_fc = np.full(n, np.nan)
    fric_bw = np.full(n, np.nan)
    nasal = np.zeros(n)
    nzero = np.full(n, np.nan)
    sharp = np.zeros(n, dtype=bool)  # frames where amplitude changes must stay sharp (bursts)

    def idx(a, b):
        i0 = int(np.clip(np.floor((a - t0) / fp + 0.5), 0, n))
        i1 = int(np.clip(np.floor((b - t0) / fp + 0.5), 0, n))
        if i1 <= i0 and i0 < n and b > a:
            i1 = i0 + 1
        return i0, i1

    for s in plan.segments:
        i0, i1 = idx(s.start, s.end)
        if i1 <= i0:
            continue
        f = _segment_formants(s)
        if f is not None:
            F[i0:i1] = f
        v, a, fr = _levels(s, voice)
        voiced[i0:i1] = np.maximum(voiced[i0:i1], v)
        asp[i0:i1] = np.maximum(asp[i0:i1], a)
        if fr > 0:
            fric[i0:i1] = np.maximum(fric[i0:i1], fr)
            fc, bw, _ = s.ph.fric
            fric_fc[i0:i1] = fc
            fric_bw[i0:i1] = bw
        if s.part == "burst":
            sharp[max(i0 - 1, 0):i1 + 1] = True
        if s.ph.kind == "nasal":
            nasal[i0:i1] = 1.0
            nzero[i0:i1] = s.ph.nasal_zero or 1500.0

    # natural attack at vowel-initial phrase starts, release at phrase ends
    env = np.ones(n)
    for u in plan.units:
        if u.breath:
            continue
        if u.phrase_start and u.lead <= 0.0:
            i0, i1 = idx(u.start - 0.01, u.start + 0.06)
            if i1 > i0:
                env[i0:i1] *= np.linspace(0.0, 1.0, i1 - i0) ** 1.5
        elif u.phrase_start:
            a = u.start - u.lead
            i0, i1 = idx(a, a + min(0.03, 0.5 * u.lead))
            if i1 > i0:
                env[i0:i1] *= np.linspace(0.0, 1.0, i1 - i0)
        if u.phrase_end:
            i0, i1 = idx(u.end - 0.07, u.end + 0.005)
            if i1 > i0:
                env[i0:i1] *= np.linspace(1.0, 0.0, i1 - i0) ** 0.8

    # smoothing (in frames)
    ms = lambda x: x / 1000.0 / fp  # noqa: E731
    F = np.stack([_hann_smooth(np.log(_fill_nan(F[:, k], NEUTRAL[k])), ms(36)) for k in range(3)], axis=1)
    F = np.exp(F)
    voiced = _hann_smooth(voiced, ms(14)) * env
    asp_s = _hann_smooth(asp, ms(12))
    fric_s = _hann_smooth(fric, ms(8))
    fric_s[sharp] = np.maximum(fric_s[sharp], fric[sharp] * 0.85)
    asp = asp_s
    fric = fric_s
    fric_fc = _hann_smooth(_fill_nan(fric_fc, 4000.0), ms(10))
    fric_bw = _fill_nan(fric_bw, 3000.0)
    nasal = _hann_smooth(nasal, ms(24))
    nzero = _fill_nan(nzero, 1500.0)

    # voice & gender scaling of the vocal tract
    f0 = 440.0 * 2.0 ** ((pitch_semitones - 69.0) / 12.0)
    scale = voice.formant_scale * 2.0 ** (-0.18 * gender) * 2.0 ** (tone_shift / 1200.0)
    Fs = F * scale[:, None]
    # vowel modification at high pitch: keep F1 above f0 (formant tuning)
    Fs[:, 0] = np.maximum(Fs[:, 0], 1.05 * f0 * (voiced > 0.05) + 0.0)
    Fs[:, 1] = np.maximum(Fs[:, 1], Fs[:, 0] + 250.0)
    Fs[:, 2] = np.maximum(Fs[:, 2], Fs[:, 1] + 300.0)
    f45 = np.stack([voice.f4 * scale / voice.formant_scale, voice.f5 * scale / voice.formant_scale], axis=1)
    f45[:, 0] = np.maximum(f45[:, 0], Fs[:, 2] + 400)
    f45[:, 1] = np.maximum(f45[:, 1], f45[:, 0] + 500)
    formants = np.concatenate([Fs, f45], axis=1)

    bscale = voice.bandwidth_scale * (1.0 + 0.35 * np.clip(breath, 0, 1)) * (1.0 - 0.12 * np.clip(tension, -1, 1))
    B = np.stack([
        55.0 + 0.045 * formants[:, 0] + 45.0 * nasal,
        70.0 + 0.035 * formants[:, 1],
        110.0 + 0.025 * formants[:, 2],
        np.full(n, 200.0),
        np.full(n, 270.0),
    ], axis=1) * bscale[:, None]

    return FrameParams(
        t0=t0, frame_period=fp, f0=f0, formants=formants, bandwidths=B, voiced=voiced, asp=asp,
        breath=breath, fric=fric, fric_fc=fric_fc * np.sqrt(voice.formant_scale), fric_bw=fric_bw,
        nasal=nasal, nasal_zero=nzero * scale, tension=tension, loudness_db=loud_db, voicing=voicing,
    )
