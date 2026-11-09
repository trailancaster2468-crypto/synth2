"""Pulse-synchronous harmonic + noise vocoder.

Voiced sound is synthesised WORLD-style: one impulse response per glottal
pulse, overlap-added at the pulse position with sub-sample accuracy.  Each
response is the product of

* a **glottal flow-derivative pulse** (KLGLOTT88 shape, closed-form spectrum)
  whose open quotient and return-phase tilt follow Fant's ``Rd`` voice-quality
  parameter, driven by tension and breathiness.  The open phase is
  *anticausal* (it precedes the glottal closure instant), so the excitation is
  mixed-phase like a real voice instead of a buzzy minimum-phase spike, and it
  has no DC.  Lax voices get a strong fundamental (H1-H2 ~ +10 dB) and a steep
  spectral tilt, pressed voices a weak H1 and a bright top end;
* a minimum-phase **vocal tract** built from the real cepstrum: formant
  cascade, higher poles up to Nyquist (a real tract keeps resonating every
  ~1 kHz), a nasal pole-zero pair, and voice colour (singer's formant,
  brightness).

Each pulse spectrum is loudness-normalised (see :func:`_voiced_spectra`), so
vowels sung at the same level sound equally loud whatever their formants.
Spectral envelopes are evaluated on a coarse frame grid (every ``GRID``
frames, ~9 ms) and interpolated per pulse: formant tracks are smoothed over
~36 ms, so this is inaudible (median log-spectral difference 0.05 dB) and
makes synthesis several times faster than evaluating them per pulse.

Noise (aspiration, frication, breath) is synthesised with a short-time inverse
FFT using shaped random spectra; breath noise riding on voicing reuses the
pulse spectra minus a frequency-dependent harmonics-to-noise ratio and is
modulated by the glottal cycle.

Everything is vectorised with numpy (per-pulse FFTs in single precision);
rendering runs ~25x faster than real time at 44.1 kHz.
"""
from __future__ import annotations

import numpy as np

from .params import FrameParams

LN10_20 = np.log(10.0) / 20.0
OUTPUT_GAIN = 0.1  # global calibration (see tests/test_engine.py::test_loudness_calibration)
VOICED_NORM = 0.85  # fraction of the envelope loudness that is normalised away for voiced pulses
ALIGN_NORM = 0.7  # fraction of the harmonic/formant alignment gain that is compensated
# the harmonics are taken to carry at least this fraction of the envelope power, which caps the alignment
# gain at +14 dB: a lone harmonic far down the envelope (f0 near Nyquist) must not be boosted by 200 dB
ALIGN_FLOOR = 0.01
DENSE_HARM = 0.5  # harmonic/envelope power ratio when the harmonics sample the envelope densely (low f0)
F0_MIN = 30.0  # Hz: f0 floor for the glottal open phase and pulse level (a fry pulse keeps a short open phase)
F0_MAX = 0.45  # x sr: the voiced source fades out from 0.40 sr up to here, where no harmonic fits below Nyquist
FORMANT_MAX = 0.47  # x sr: resonators fade out from 0.42 sr up to here instead of folding back to sr - F
GRID = 3  # spectral envelopes are evaluated every GRID frames (~8.7 ms) and interpolated per pulse
FRAC_STEPS = 64  # sub-sample pulse placement resolution (1/64 sample)
SAMPLE_CHUNK = 1 << 16  # per-sample bookkeeping (glottal phase, breath gate) is done in blocks this long
PULSE_POST_S = 0.0232  # seconds of vocal-tract ring-down kept after each glottal closure
RD_MODAL = 0.9  # Fant's Rd for a neutral (tension 0) voice with source_tilt == 11
FA_REF_HZ = 300.0  # f0 at which the return-phase corner equals f0 / (2 pi Ra)
# KLGLOTT88 pulses have unit closure strength; this restores the absolute voiced level
# (test_loudness_calibration: a default /a/ at E4 sits near -20 dBFS RMS)
SOURCE_GAIN_DB = 10.9


def fft_size(sr: int) -> int:
    """Pulse FFT length: ~46 ms of signal (2048 at 44.1/48 kHz, 1024 at 22.05 kHz)."""
    return 1 << int(round(np.log2(sr * 0.0464)))


def _nyquist_fade(F: np.ndarray, sr: int) -> np.ndarray:
    """Weight (0..1) of a resonator at F Hz: 1 up to 0.42 sr, smoothly 0 from FORMANT_MAX * sr.

    A digital resonator above Nyquist folds back to sr - F, a spurious resonance that a
    higher sample rate would not have, so resonators near Nyquist fade out instead.
    """
    x = np.clip((FORMANT_MAX * sr - F) / (0.05 * sr), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _resonator_db(freqs: np.ndarray, F: np.ndarray, B: np.ndarray, sr: int,
                  weight: np.ndarray | None = None) -> np.ndarray:
    """Magnitude (dB) of DC-normalised digital 2-pole resonators.

    freqs: (nb,) Hz; F, B: (k, m) -> returns (k, nb) summed over the m resonators, each
    scaled in dB by ``weight`` (k, m) when given.
    Uses the closed form |1 - c z^-1 + r^2 z^-2|^2 = 1 + c^2 + r^4 - 2c(1 + r^2)cos w + 2r^2 cos 2w.
    """
    w = 2 * np.pi * freqs / sr
    cw = np.cos(w)[None, :]
    c2w = np.cos(2 * w)[None, :]
    prod = np.ones((F.shape[0], len(freqs)))
    r_all = np.exp(-np.pi * B / sr)
    c_all = 2 * r_all * np.cos(2 * np.pi * F / sr)
    for j in range(F.shape[1]):
        r = r_all[:, j:j + 1]
        r2 = r * r
        c = c_all[:, j:j + 1]
        num = 1 - c + r2
        den2 = (1 + c * c + r2 * r2) - (2 * c * (1 + r2)) * cw + (2 * r2) * c2w
        if weight is not None and np.any(weight[:, j] < 1.0):
            prod *= ((num * num) / np.maximum(den2, 1e-30)) ** weight[:, j:j + 1]
        else:
            prod *= (num * num) / np.maximum(den2, 1e-30)
    return 10 * np.log10(np.maximum(prod, 1e-300))


def _antiresonator_db(freqs, Fz, Bz, sr):
    return -_resonator_db(freqs, Fz[:, None], Bz[:, None], sr)


def _higher_poles_db(freqs, f5, spacing, sr):
    """Higher-pole correction (dB, (k, nb)): the tract resonances above F5 up to Nyquist.

    A real vocal tract keeps resonating every ~1 kHz; a DC-normalised cascade
    that stops at F5 (plus a few poles) falls off at ~ -12 dB/oct per pole above
    the last one, a brick wall that removed everything above ~7 kHz.  The poles
    only depend on (F5, spacing), which hardly ever change within a track, so
    each distinct pair (quantised to 1/256 octave) is evaluated once.
    """
    q = np.round(np.log2(np.stack([f5, spacing], axis=1)) * 256.0)
    keys, inv = np.unique(q, axis=0, return_inverse=True)
    rows = np.empty((len(keys), len(freqs)))
    for i, (k5, ks) in enumerate(keys):
        top, sp = 2.0 ** (k5 / 256.0), 2.0 ** (ks / 256.0)
        fk = top + sp * np.arange(1, 64)
        fk = fk[fk < 0.47 * sr]
        # radiation and wall losses widen the high resonances
        bk = np.minimum(450.0 + 200.0 * np.arange(len(fk)), 2500.0)
        rows[i] = _resonator_db(freqs, fk[None, :], bk[None, :], sr)[0] if len(fk) else 0.0
    return rows[inv.ravel()]


def vocal_tract_db(freqs, formants, bandwidths, nasal, nasal_zero, sr):
    """Vocal-tract magnitude (dB, (k, nb)): F1-F5 cascade, higher poles and nasal coupling."""
    f5 = formants[:, 4]
    # a tube of length L resonates every c/2L ~ F5/4.5
    spacing = np.maximum(f5 - formants[:, 3], 0.22 * f5)
    # formants pushed towards Nyquist (gender / toneShift at a low sample rate) fade out like the higher poles
    fade = _nyquist_fade(formants, sr) if np.any(formants > 0.42 * sr) else None
    db = _resonator_db(freqs, formants, bandwidths, sr, fade) + _higher_poles_db(freqs, f5, spacing, sr)
    if np.any(nasal > 0.01):
        # nasal coupling adds a pole-zero pair (Klatt's FNP/FNZ): the oral side-branch zero
        # plus a nasal pole just below it.  A lone DC-normalised zero would rise +12 dB/oct
        # forever (nasals brighter than vowels at 8-16 kHz); the pair settles at
        # (Fp/Fz)^2 = -4 dB above the zero, so the murmur stays duller than a vowel.
        fz = nasal_zero
        z = _antiresonator_db(freqs, fz, np.full_like(fz, 120.0), sr)
        z += _resonator_db(freqs, (0.8 * fz)[:, None], (80.0 + 0.05 * fz)[:, None], sr)
        db += (nasal * _nyquist_fade(fz, sr))[:, None] * z
    return db


# ------------------------------------------------------------------ glottal source
def rd_glottal(f0, tension, breath, voice):
    """Glottal pulse shape from Fant's Rd parameter (LF-model regression, Fant 1995).

    Rd runs from ~0.3 (pressed) through ~1 (modal) to ~2.7 (breathy).  Tension
    lowers it, breathiness raises it, and a voice's ``source_tilt`` sets its
    neutral point (11 dB/oct = modal).  Returns ``(open_quotient, Fa)`` where
    ``Fa = 1 / (2 pi Ta)`` is the corner of the return-phase spectral tilt.
    """
    t = np.clip(tension, -1, 1)
    b = np.clip(breath, -1, 1)
    rd = (RD_MODAL * 2.0 ** ((voice.source_tilt - 11.0) / 6.0)
          * 2.0 ** (-1.2 * t + 0.6 * np.clip(b, 0, 1) + 0.3 * np.clip(b, -1, 0)))
    rd = np.clip(rd, 0.3, 2.7)
    ra = (-1.0 + 4.8 * rd) / 100.0
    rk = (22.4 + 11.8 * rd) / 100.0
    rg = rk / (4.0 * (0.11 * rd / (0.5 + 1.2 * rk) - ra))
    oq = np.clip((1.0 + rk) / (2.0 * rg), 0.3, 0.95)
    # Ta does not grow in proportion to the period at low pitch (male voices keep a fairly
    # abrupt closure and more high-frequency energy than female ones, Hanson & Chuang 1999),
    # so Fa follows f0 only weakly
    fa = FA_REF_HZ * (np.maximum(f0, 30.0) / FA_REF_HZ) ** 0.25 / (2 * np.pi * ra)
    return oq, fa


def glottal_spectrum(freqs: np.ndarray, f0: np.ndarray, oq: np.ndarray) -> np.ndarray:
    """Closed-form spectrum of a KLGLOTT88 flow-derivative pulse, (k, nb) complex.

    E(t) = a (2t - 3t^2/Te) on the open phase [0, Te], Te = Oq / f0, with
    ``a = 1/Te`` so the closure step is 1.  Time is referenced to the glottal
    closure instant (the open phase is anticausal), so the waveform is
    band-limited by construction (no aliasing) and E has zero mean (no DC).
    """
    te = (oq / np.maximum(f0, 1.0))[:, None]
    x = 2 * np.pi * np.maximum(freqs, 1e-6)[None, :] * te
    # G / Te = e^{jx} (-2/x^2 - 6j/x^3) - 4/x^2 + j (6/x^3 - 1/x), evaluated in real arithmetic
    ix = 1.0 / x
    ix2 = ix * ix
    a = -2.0 * ix2
    b = -6.0 * ix2 * ix
    nb = len(freqs)
    if nb > 2 and freqs[0] == 0 and np.allclose(np.diff(freqs), freqs[1]):
        # uniform bins: e^{jx} is a geometric series along frequency (2x faster than sin + cos)
        z = np.empty((len(te), nb), dtype=complex)
        z[:, 0] = 1.0
        z[:, 1:] = np.exp(2j * np.pi * freqs[1] * te)
        z = np.cumprod(z, axis=1)
        c, s = z.real, z.imag
    else:
        c, s = np.cos(x), np.sin(x)
    g = np.empty(x.shape, dtype=complex)
    g.real = (a * c - b * s - 4.0 * ix2) * te
    g.imag = (a * s + b * c - b - ix) * te
    g[:, freqs <= 0] = 0.0
    return g


def colour_db(freqs, tension, voice):
    """Voice colour (dB, (k, nb)): singer's formant cluster, brightness shelf, HF smoothing."""
    sfc = 2900.0 * np.sqrt(voice.formant_scale)
    sf_gain = voice.singer_formant_db * (1.0 + 0.6 * np.clip(tension, -1, 1))
    octs = np.log2(np.maximum(freqs, 1.0) / sfc)
    sf = sf_gain[:, None] * np.exp(-0.5 * (octs[None, :] / 0.3) ** 2)
    bright = voice.brightness_db * (1.0 / (1.0 + (5200.0 / np.maximum(freqs, 1.0)) ** 4))
    # gentle high-frequency rolloff to keep the top end smooth
    hf = -10 * np.log10(1 + (freqs / 12000.0) ** 6)
    return sf + (bright + hf)[None, :]


def source_db(freqs, f0, tension, breath, voice):
    """Glottal source (flow derivative, i.e. including lip radiation) + voice colour, dB (k, nb)."""
    oq, fa = rd_glottal(f0, tension, breath, voice)
    g = np.abs(glottal_spectrum(freqs, f0, oq))
    tilt = -10 * np.log10(1 + (freqs[None, :] / fa[:, None]) ** 2)
    return 20 * np.log10(np.maximum(g, 1e-12)) + tilt + colour_db(freqs, tension, voice)


def loudness_weight(f: np.ndarray) -> np.ndarray:
    """Power weighting approximating the ISO 226 80-phon equal-loudness contour (1 at 1 kHz).

    Half of IEC 61672 A-weighting in dB (A-weighting itself models ~40 phon and
    would over-penalise the low harmonics of a voice heard at singing levels):
    -9.6 dB at 100 Hz, -4.3 dB at 250 Hz, +0.6 dB at 2 kHz.
    """
    f2 = np.maximum(f, 1.0) ** 2
    ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2))
                                  * (f2 + 12194.0 ** 2))
    return ra / 0.7943


def _voiced_spectra(fp: FrameParams, idx: np.ndarray, voice, sr: int, n_fft: int):
    """Normalised voiced pulse spectra (complex, (k, n_fft//2+1)) at frames ``idx``.

    Returns ``(H, oq, harm)``: ``harm`` (k,) is the amplitude of the harmonics relative
    to the smooth envelope, ``sqrt(min(p_harm / (DENSE_HARM p_env), 1))``, which scales
    the breath noise so it follows the periodic power (see :func:`_noise`).

    The minimum-phase part (vocal tract, return-phase tilt, colour) is built
    from the real cepstrum; the glottal open phase is multiplied in with its own
    (anticausal) phase.  Each spectrum is then loudness-normalised so phoneme
    loudness comes from the amplitude tracks, not from the incidental energy of
    the formant pattern:

    * VOICED_NORM of the loudness-weighted envelope power is removed (closed vowels
      would otherwise be ~10 dB quiet), and
    * ALIGN_NORM of the harmonic/formant alignment gain: the power the
      harmonics actually carry relative to the smooth envelope.  Without it a
      formant sitting on H1 (e.g. /iy/ near E4) or between two harmonics makes
      one vowel up to ~10 dB louder than the next at the same pitch; keeping
      30 % of it preserves the natural partial modulation of vibrato.  The
      correction is capped (``ALIGN_FLOOR``) for a lone harmonic far down the
      envelope, e.g. f0 near Nyquist.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    f0 = fp.f0[idx]
    ten = fp.tension[idx]
    br = fp.breath[idx]
    oq, fa = rd_glottal(f0, ten, br, voice)
    db = vocal_tract_db(freqs, fp.formants[idx], fp.bandwidths[idx], fp.nasal[idx], fp.nasal_zero[idx], sr)
    db += colour_db(freqs, ten, voice)
    db -= 10 * np.log10(1 + (freqs[None, :] / fa[:, None]) ** 2)
    g = glottal_spectrum(freqs, np.maximum(f0, F0_MIN), oq)
    fold = np.zeros(n_fft)
    fold[0] = 1
    fold[1:n_fft // 2] = 2
    fold[n_fft // 2] = 1
    cep = np.fft.irfft(db * LN10_20, n=n_fft, axis=1) * fold[None, :]
    H = np.exp(np.fft.rfft(cep, axis=1)) * g
    P = (H.real ** 2 + H.imag ** 2) * loudness_weight(freqs)[None, :]
    p_env = np.maximum(np.mean(P, axis=1) * 2.0, 1e-30)  # loudness-weighted power of the smooth envelope
    # weighted power at the harmonics: (2 f0 / sr) * sum_k W |H(k f0)|^2 (linear interpolation in bins)
    fn = np.maximum(fp.f0[idx], 30.0)
    kmax = int(np.ceil(0.5 * sr / fn.min()))
    pos = fn[:, None] * np.arange(1, kmax + 1)[None, :] * (n_fft / sr)
    valid = pos < n_fft // 2
    pos = np.where(valid, pos, 0.0)
    i0 = np.floor(pos).astype(int)
    w = pos - i0
    ph = np.take_along_axis(P, i0, axis=1) * (1 - w) + np.take_along_axis(P, i0 + 1, axis=1) * w
    p_harm = np.maximum((2 * fn / sr) * np.sum(ph * valid, axis=1), 1e-30)
    gain = p_env ** (-0.5 * VOICED_NORM) * np.maximum(p_harm / p_env, ALIGN_FLOOR) ** (-0.5 * ALIGN_NORM)
    H *= (gain * 10 ** (SOURCE_GAIN_DB / 20.0))[:, None]
    harm = np.sqrt(np.minimum(p_harm / (DENSE_HARM * p_env), 1.0))
    return H, oq, harm


def _ola(out: np.ndarray, starts: np.ndarray, blocks: np.ndarray):
    """Overlap-add ``blocks`` (k, L) into ``out`` at integer ``starts`` (vectorised via bincount)."""
    keep = (starts >= 0) & (starts + blocks.shape[1] <= len(out))
    if not keep.all():
        starts, blocks = starts[keep], blocks[keep]
    if len(starts) == 0:
        return
    lo = int(starts.min())
    L = blocks.shape[1]
    idx = (starts - lo)[:, None] + np.arange(L)[None, :]
    span = int(idx.max()) + 1
    out[lo:lo + span] += np.bincount(idx.ravel(), weights=blocks.ravel(), minlength=span)


def _interp_frames(arr: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation of per-frame values at fractional frame positions."""
    n = arr.shape[0]
    p = np.clip(pos, 0, n - 1)
    i0 = np.floor(p).astype(int)
    i1 = np.minimum(i0 + 1, n - 1)
    w = p - i0
    if arr.ndim == 1:
        return arr[i0] * (1 - w) + arr[i1] * w
    return arr[i0] * (1 - w)[:, None] + arr[i1] * w[:, None]


def _pulse_positions(fp: FrameParams, voice, sr: int, hop: float, n_samples: int, rng: np.random.Generator):
    """Glottal closure instants (fractional samples) and the glottal phase per sample.

    f0 gets a slow "flutter" (band-limited random FM) and every pulse a small
    independent timing displacement (cycle-to-cycle jitter).  Displacing pulses
    instead of integrating random periods keeps the long-term pitch exact.

    The per-sample phase is integrated in blocks of ``SAMPLE_CHUNK`` samples (the
    running sum carries over, so the result equals one cumsum over the render) and
    only its fractional part is kept, in single precision: ~4 bytes per output
    sample instead of several full-length float64 temporaries.
    """
    flt = rng.standard_normal(fp.n // 4 + 2) if voice.jitter > 0 else None
    flt_pos = np.linspace(0, fp.n, len(flt)) if flt is not None else None
    frames = np.arange(fp.n)
    f0 = np.minimum(fp.f0, F0_MAX * sr)  # the source is silent above (see synthesize): bound the pulse rate
    gph = np.empty(n_samples, dtype=np.float32)
    pos_l, f0_l = [], []
    carry = 0.0
    for s0 in range(0, n_samples, SAMPLE_CHUNK):
        s1 = min(s0 + SAMPLE_CHUNK, n_samples)
        frame_pos = np.arange(s0, s1) / hop
        f0_s = np.interp(frame_pos, frames, f0)
        if flt is not None:
            f0_s *= 1.0 + 0.7 * voice.jitter * np.interp(frame_pos, flt_pos, flt)
        phase = np.cumsum(np.concatenate(([carry], f0_s / sr)))  # phase[j + 1] belongs to sample s0 + j
        carry = float(phase[-1])
        cyc = np.floor(phase)
        j = np.nonzero(np.diff(cyc))[0]
        frac = (phase[j + 1] - cyc[j + 1]) / np.maximum(f0_s[j] / sr, 1e-9)
        pos_l.append(s0 + j - frac)
        f0_l.append(f0_s[j])
        gph[s0:s1] = (phase - cyc)[1:]
    pos = np.concatenate(pos_l) if pos_l else np.zeros(0)
    f0_at = np.concatenate(f0_l) if f0_l else np.zeros(0)
    if voice.jitter > 0 and len(pos):
        # ~0.5 % local jitter (Praat definition) for the default voice.jitter of 0.004
        pos = pos + (0.65 * voice.jitter * sr / f0_at) * np.clip(rng.standard_normal(len(pos)), -3, 3)
        pos = np.maximum.accumulate(pos)
    return pos, gph


def synthesize(fp: FrameParams, voice, sr: int, rng: np.random.Generator) -> np.ndarray:
    hop = fp.frame_period * sr
    n_fft = fft_size(sr)
    n_noise = max(64, n_fft // 4)  # noise frames of ~11.6 ms, 4x overlap at the frame hop
    post = int(round(PULSE_POST_S * sr))
    n_samples = int(np.ceil(fp.n * hop)) + post

    gain = 10.0 ** (fp.loudness_db / 20.0) * OUTPUT_GAIN * 10 ** (voice.loudness_db / 20.0)
    v_amp = fp.voiced * np.clip(fp.voicing, 0, 1) * (1.0 - 0.25 * np.clip(fp.breath, 0, 1)) * gain
    # no harmonic of an f0 near or above Nyquist fits in the band: the voiced source (and the breath
    # noise riding on it) fades out between 0.40 sr and F0_MAX * sr instead of aliasing
    v_amp = v_amp * np.clip((F0_MAX * sr - fp.f0) / (0.05 * sr), 0.0, 1.0)

    # ---------------------------------------------------------------- pulses
    pos, phase = _pulse_positions(fp, voice, sr, hop, n_samples, rng)
    fpos = pos / hop
    va = _interp_frames(v_amp, fpos)
    keep = va > 1e-5
    pos, fpos, va = pos[keep], fpos[keep], va[keep]

    # pulse envelopes are interpolated between grid frames GRID * g0 and GRID * (g0 + 1)
    n_grid = (fp.n - 1) // GRID + 1
    gpos = np.clip(fpos, 0, fp.n - 1) / GRID
    g0 = np.floor(gpos).astype(int)
    gw = (gpos - g0).astype(np.float32)
    g1 = np.minimum(g0 + 1, n_grid - 1)

    # the longest open phase among the grid spectra the pulses use decides how many samples precede
    # each closure; the pulse FFT (~46 ms) doubles until it holds that open phase plus the ring-down
    # (low bass notes), so the open phase is never cut into a per-period step
    if len(pos):
        fr = np.minimum(np.unique(np.concatenate([g0, g1])) * GRID, fp.n - 1)
        oq_r, _ = rd_glottal(fp.f0[fr], fp.tension[fr], fp.breath[fr], voice)
        pre = int(np.ceil(np.max(oq_r * sr / np.maximum(fp.f0[fr], F0_MIN)))) + 8
    else:
        pre = 8
    while pre > 0.4 * n_fft:
        n_fft *= 2
    lead = pre + 8  # samples of headroom before the first closure in the output buffer
    out = np.zeros(n_samples + post + lead, dtype=np.float32)  # single precision: 4 bytes per sample

    nb = n_fft // 2 + 1
    kbin = np.arange(nb)
    # sub-sample delays as a table of linear-phase ramps (no transcendental maths per pulse)
    ramps = np.exp(-2j * np.pi * kbin[None, :] * (np.arange(FRAC_STEPS + 1) / FRAC_STEPS)[:, None]
                   / n_fft).astype(np.complex64)
    shimmer = voice.shimmer_db * LN10_20

    # magnitude of the voiced spectrum at the noise resolution, reused to shape breath noise
    ratio = n_fft // n_noise
    vmag = np.zeros((n_grid, n_noise // 2 + 1), dtype=np.float32)
    have = np.zeros(n_grid, dtype=bool)

    f0p = _interp_frames(fp.f0, fpos)
    amp = va * np.sqrt(sr / np.maximum(f0p, F0_MIN))
    if shimmer > 0:
        amp = amp * np.exp(shimmer * rng.standard_normal(len(amp)))
    amp = amp.astype(np.float32)
    ip = np.floor(pos).astype(int)
    q = np.rint((pos - ip) * FRAC_STEPS).astype(int)

    taper = np.ones(pre + post, dtype=np.float32)
    tl = post // 4
    taper[-tl:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, tl))
    taper[:4] = np.linspace(0.0, 1.0, 6)[1:5]

    CH = 256
    for c0 in range(0, len(pos), CH):
        sl = slice(c0, c0 + CH)
        a, b = g0[sl], g1[sl]
        need, inv = np.unique(np.concatenate([a, b]), return_inverse=True)
        Hg, _, hfac = _voiced_spectra(fp, np.minimum(need * GRID, fp.n - 1), voice, sr, n_fft)
        vmag[need] = np.abs(Hg[:, ::ratio]) * hfac[:, None]
        have[need] = True
        Hg = Hg.astype(np.complex64)
        k = len(a)
        w = gw[sl][:, None]
        H = Hg[inv[:k]] * (1 - w) + Hg[inv[k:]] * w
        H *= ramps[q[sl]]
        H *= amp[sl][:, None]
        y = np.fft.irfft(H, n=n_fft, axis=1)  # single precision
        # closure at index 0; the anticausal open phase wrapped around to the end of the buffer
        blocks = np.concatenate([y[:, n_fft - pre:], y[:, :post]], axis=1) * taper[None, :]
        _ola(out, ip[sl] - pre + lead, blocks)

    # breath noise reads vmag on every grid row next to a voiced frame: fill the rows no pulse
    # needed (pulses more than two rows apart below ~57 Hz, voicing edges) so it does not drop out
    fv = np.nonzero(v_amp > 1e-6)[0] // GRID
    rows = np.unique(np.minimum(np.concatenate([fv, fv + 1]), n_grid - 1))
    rows = rows[~have[rows]]
    for c0 in range(0, len(rows), CH):
        r = rows[c0:c0 + CH]
        Hg, _, hfac = _voiced_spectra(fp, np.minimum(r * GRID, fp.n - 1), voice, sr, n_fft)
        vmag[r] = np.abs(Hg[:, ::ratio]) * hfac[:, None]

    voiced = out[lead:lead + n_samples]
    # ---------------------------------------------------------------- noise
    voiced += _noise(fp, voice, sr, rng, gain, v_amp, phase, hop, n_samples, vmag, n_noise)
    return voiced


def _noise(fp: FrameParams, voice, sr, rng, gain, v_amp, phase, hop, n_samples, vmag, N) -> np.ndarray:
    """Breath + aspiration + frication noise via short-time inverse FFT overlap-add.

    * air: breath noise riding on the voice.  Its spectrum is the voiced spectrum
      (``vmag``: pulse spectra on the envelope grid, at this resolution, scaled by
      the harmonic/envelope power ratio so it follows the power the harmonics
      really carry, also at high f0 where a few harmonics sample a formant-rich
      envelope) minus a harmonics-to-noise ratio (HNR) that falls with frequency,
      so a breathy voice is airy at the top while the low harmonics stay clean.
      It is pulsed by the glottal cycle (``phase``: glottal phase per sample, in
      cycles), peaking around maximum glottal flow.
    * aspiration: h / stop release / breath intake / whisper, shaped by the vocal tract.
    * frication: resonance-shaped noise for s, sh, f, bursts...

    Aspiration and frication share one random spectrum (independent Gaussian
    noises add in power), and each component is only synthesised on the frames
    where it is active.
    """
    H = int(round(hop))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    n = fp.n
    breath = np.clip(fp.breath, -1, 1)
    whisper = fp.voiced * (1.0 - np.clip(fp.voicing, 0, 1)) * 0.35
    asp = (fp.asp + whisper) * gain
    air = v_amp
    fric = fp.fric * gain
    # single precision accumulators (the output is float32 anyway): ~8 bytes per output sample
    noise = np.zeros(max(n * H, n_samples) + 3 * N, dtype=np.float32)
    air_out = np.zeros_like(noise)
    win = np.hanning(N + 1)[:-1].astype(np.float32)
    scale = np.sqrt(8.0 * H / 3.0)  # Hann^2 overlap-add at hop H: E[noise^2] = mean(|mag|^2)
    off = N - N // 2 + H // 2  # frame i is centred on sample i * H + H/2 (offset by N to stay positive)

    def _synth(target, fi, mag):
        X = rng.standard_normal((len(fi), len(freqs), 2), dtype=np.float32).view(np.complex64)[..., 0]
        X *= (mag * np.sqrt(0.5)).astype(np.float32)
        seg = np.fft.irfft(X, n=N, axis=1) * (win * scale)[None, :]
        _ola(target, fi * H + off, seg)

    CH = 512
    # -- aspiration + frication (consonant frames)
    cons = np.nonzero((asp > 1e-6) | (fric > 1e-6))[0]
    if len(cons):
        hp_asp = -10 * np.log10(1 + (400.0 / np.maximum(freqs, 1.0)) ** 4)
        # radiation (+6 dB/oct) on a glottal turbulence source that itself falls above ~2.5 kHz:
        # aspiration energy sits in the F2-F4 region of the vowel being shaped
        rad = 10 * np.log10(1 + (freqs / 350.0) ** 2) - 10 * np.log10(1 + (freqs / 2500.0) ** 2) - 6.0
        for c0 in range(0, len(cons), CH):
            fi = cons[c0:c0 + CH]
            vt = vocal_tract_db(freqs, fp.formants[fi], fp.bandwidths[fi], fp.nasal[fi], fp.nasal_zero[fi], sr)
            # aspiration: vocal-tract shaped, power-normalised
            asp_db = vt + (rad + hp_asp)[None, :]
            asp_p = 10 ** (asp_db / 10.0)
            asp_p *= (asp[fi] ** 2 / np.mean(asp_p, axis=1))[:, None]
            # frication: resonance around fc with a place-dependent high-pass
            fc = fp.fric_fc[fi][:, None]
            bw = fp.fric_bw[fi][:, None]
            x = (freqs[None, :] - fc) / (0.5 * bw)
            res = 1.0 / np.sqrt(1.0 + x ** 4)
            hp = 1.0 / np.sqrt(1.0 + (np.minimum(0.45 * fc, 2600.0) / np.maximum(freqs[None, :], 1.0)) ** 6)
            fric_p = ((res + 0.06) * hp) ** 2
            fric_p *= (fric[fi] ** 2 / np.mean(fric_p, axis=1))[:, None]
            _synth(noise, fi, np.sqrt(asp_p + fric_p))
    # -- breath noise riding on the voice
    voiced = np.nonzero(air > 1e-6)[0]
    if len(voiced):
        octs = np.log2(np.maximum(freqs, 250.0) / 250.0)
        # HNR at 250 Hz: ~36 dB for a modal voice, lower when breathy / for airy voices.  Glottal
        # turbulence noise is mostly high-frequency (it replaces the harmonics around F3 in a
        # breathy voice while H1 stays strong), so the HNR falls steeply with frequency:
        # modal ~14 dB at 4 kHz, breathy ~0 dB.
        hnr0 = 36.0 - 13.0 * np.clip(breath, 0, 1) - 4.0 * np.clip(breath, -1, 0) \
            + 4.0 * np.clip(fp.tension, -1, 1) - (voice.aspiration_db + 30.0)
        hnr_slope = 5.5  # dB per octave
        hnr_lin = 10 ** (-hnr0 / 20.0)
        slope_lin = (10 ** (hnr_slope * octs / 20.0)).astype(np.float32)
        gp = voiced / GRID
        g0 = np.minimum(np.floor(gp).astype(int), len(vmag) - 1)
        g1 = np.minimum(g0 + 1, len(vmag) - 1)
        gw = (gp - g0).astype(np.float32)[:, None]
        for c0 in range(0, len(voiced), CH):
            sl = slice(c0, c0 + CH)
            fi = voiced[sl]
            mag = vmag[g0[sl]] * (1 - gw[sl]) + vmag[g1[sl]] * gw[sl]
            mag *= slope_lin[None, :]
            mag *= (hnr_lin[fi] * air[fi]).astype(np.float32)[:, None]
            _synth(air_out, fi, mag)
    noise = noise[N:N + n_samples]
    air_out = air_out[N:N + n_samples]
    # breath noise is pulsed by the glottal cycle: strongest near maximum flow, late in the
    # open phase (the closure instant is at integer phase); gated in blocks to bound memory
    norm = 1.0 / np.sqrt(0.3 ** 2 + 2 * 0.3 * 1.4 * 0.375 + 1.4 ** 2 * 35.0 / 128.0)  # unit RMS over a cycle
    for s0 in range(0, n_samples, SAMPLE_CHUNK):
        s1 = min(s0 + SAMPLE_CHUNK, n_samples)
        ph = phase[s0:s1] - np.floor(phase[s0:s1])
        gate = 0.3 + 1.4 * (0.5 - 0.5 * np.cos(2 * np.pi * (ph - 0.3))) ** 2
        noise[s0:s1] += air_out[s0:s1] * (gate * norm)
    return noise
