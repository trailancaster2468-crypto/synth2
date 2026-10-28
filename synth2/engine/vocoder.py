"""Pulse-synchronous harmonic + noise vocoder.

Voiced sound is synthesised WORLD-style: one minimum-phase impulse response
per glottal pulse, computed from a spectral envelope (glottal source x vocal
tract formants x lip radiation x singer's formant), overlap-added at the pulse
position with sub-sample accuracy.  Noise (aspiration, frication, breath) is
synthesised with a short-time inverse FFT using shaped random spectra, and
breath noise riding on voicing is modulated by the glottal cycle.

Everything is vectorised with numpy; rendering runs many times faster than
real time.
"""
from __future__ import annotations

import numpy as np

from .params import FrameParams

N_FFT = 2048
PULSE_LEN = 1024
NOISE_FFT = 512
LN10_20 = np.log(10.0) / 20.0
OUTPUT_GAIN = 0.1  # global calibration (see tests/test_engine.py::test_loudness_calibration)
VOICED_NORM = 0.85  # fraction of envelope-energy normalisation for voiced pulses


def _resonator_db(freqs: np.ndarray, F: np.ndarray, B: np.ndarray, sr: int) -> np.ndarray:
    """Magnitude (dB) of DC-normalised digital 2-pole resonators.

    freqs: (nb,) Hz; F, B: (k, m) -> returns (k, nb) summed over the m resonators.
    Uses the closed form |1 - c z^-1 + r^2 z^-2|^2 = 1 + c^2 + r^4 - 2c(1 + r^2)cos w + 2r^2 cos 2w.
    """
    w = 2 * np.pi * freqs / sr
    cw = np.cos(w)[None, :]
    c2w = np.cos(2 * w)[None, :]
    prod = np.ones((F.shape[0], len(freqs)))
    for j in range(F.shape[1]):
        r = np.exp(-np.pi * B[:, j] / sr)[:, None]
        r2 = r * r
        c = 2 * r * np.cos(2 * np.pi * F[:, j] / sr)[:, None]
        den2 = (1 + c * c + r2 * r2) - (2 * c * (1 + r2)) * cw + (2 * r2) * c2w
        num = 1 - c + r2
        prod *= (num * num) / np.maximum(den2, 1e-30)
    return 10 * np.log10(np.maximum(prod, 1e-300))


def _antiresonator_db(freqs, Fz, Bz, sr):
    return -_resonator_db(freqs, Fz[:, None], Bz[:, None], sr)


def vocal_tract_db(freqs, formants, bandwidths, nasal, nasal_zero, sr):
    # higher-pole correction: a real tract keeps resonating above F5 (~1 formant/kHz),
    # without these the DC-normalised cascade would kill everything above ~6 kHz
    f5 = formants[:, 4:5]
    spacing = np.maximum(f5 - formants[:, 3:4], 700.0)
    extra_f = np.concatenate([f5 + spacing, f5 + 2 * spacing, f5 + 3 * spacing], axis=1)
    extra_f = np.minimum(extra_f, 0.45 * sr)
    extra_b = np.tile(np.array([[450.0, 650.0, 900.0]]), (len(formants), 1))
    db = _resonator_db(freqs, np.concatenate([formants, extra_f], axis=1),
                       np.concatenate([bandwidths, extra_b], axis=1), sr)
    if np.any(nasal > 0.01):
        z = _antiresonator_db(freqs, nasal_zero, np.full_like(nasal_zero, 120.0), sr)
        db += nasal[:, None] * z
    return db


def source_db(freqs, f0, tension, breath, voice):
    """Glottal source + radiation + voice colour (dB), shape (k, nb)."""
    tilt = voice.source_tilt - 3.5 * tension + 2.5 * np.clip(breath, 0, None)  # (k,)
    fc = voice.source_corner * (1.0 + 0.35 * tension)
    fc = np.maximum(fc, 0.8 * f0)
    ratio = (freqs[None, :] / fc[:, None]) ** 2
    src = -(tilt[:, None] / 6.0) * 10 * np.log10(1 + ratio)
    rad = 10 * np.log10(1 + (freqs[None, :] / 350.0) ** 2) - 10 * np.log10(1 + (60.0 / np.maximum(freqs[None, :], 1)) ** 2)
    sfc = 2900.0 * np.sqrt(voice.formant_scale)
    sf_gain = voice.singer_formant_db * (1.0 + 0.6 * tension)
    octs = np.log2(np.maximum(freqs, 1.0) / sfc)
    sf = sf_gain[:, None] * np.exp(-0.5 * (octs[None, :] / 0.3) ** 2)
    bright = voice.brightness_db * (1.0 / (1.0 + (5200.0 / np.maximum(freqs, 1.0)) ** 4))
    # gentle high-frequency rolloff to keep the top end smooth
    hf = -10 * np.log10(1 + (freqs / 12000.0) ** 6)
    return src + rad + sf + bright[None, :] + hf[None, :]


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


def synthesize(fp: FrameParams, voice, sr: int, rng: np.random.Generator) -> np.ndarray:
    hop = fp.frame_period * sr
    n_samples = int(np.ceil(fp.n * hop)) + PULSE_LEN
    out = np.zeros(n_samples + PULSE_LEN)
    frame_pos_of_sample = np.arange(n_samples) / hop

    gain = 10.0 ** (fp.loudness_db / 20.0) * OUTPUT_GAIN * 10 ** (voice.loudness_db / 20.0)
    v_amp = fp.voiced * np.clip(fp.voicing, 0, 1) * (1.0 - 0.25 * np.clip(fp.breath, 0, 1)) * gain

    # ---------------------------------------------------------------- pulses
    f0_s = np.interp(frame_pos_of_sample, np.arange(fp.n), fp.f0)
    if voice.jitter > 0:
        jit = rng.standard_normal(fp.n // 4 + 2)
        jit = np.interp(frame_pos_of_sample, np.linspace(0, fp.n, len(jit)), jit)
        f0_s = f0_s * (1.0 + voice.jitter * jit)
    phase = np.cumsum(f0_s / sr)
    cyc = np.floor(phase)
    idx = np.nonzero(np.diff(cyc))[0] + 1
    frac = (phase[idx] - cyc[idx]) / np.maximum(f0_s[idx] / sr, 1e-9)
    pos = idx - frac  # sub-sample pulse positions
    fpos = pos / hop
    va = _interp_frames(v_amp, fpos)
    keep = va > 1e-5
    pos, fpos, va = pos[keep], fpos[keep], va[keep]

    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    taper = np.ones(PULSE_LEN)
    tl = PULSE_LEN // 4
    taper[-tl:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, tl))
    kbin = np.arange(len(freqs))
    fold = np.zeros(N_FFT)
    fold[0] = 1
    fold[1:N_FFT // 2] = 2
    fold[N_FFT // 2] = 1
    shimmer = voice.shimmer_db * LN10_20

    CH = 192
    for c0 in range(0, len(pos), CH):
        p = pos[c0:c0 + CH]
        fpp = fpos[c0:c0 + CH]
        k = len(p)
        f0p = _interp_frames(fp.f0, fpp)
        F = _interp_frames(fp.formants, fpp)
        B = _interp_frames(fp.bandwidths, fpp)
        nas = _interp_frames(fp.nasal, fpp)
        nz = _interp_frames(fp.nasal_zero, fpp)
        ten = _interp_frames(fp.tension, fpp)
        br = _interp_frames(fp.breath, fpp)
        db = vocal_tract_db(freqs, F, B, nas, nz, sr) + source_db(freqs, f0p, ten, br, voice)
        # level normalisation: phoneme loudness comes from the amplitude tracks, not from
        # the incidental energy of the formant pattern (closed vowels would be ~10 dB quiet)
        pdb = 10 * np.log10(np.mean(10 ** (db / 10.0), axis=1))
        db -= VOICED_NORM * pdb[:, None]
        ln = db * LN10_20
        cep = np.fft.irfft(ln, n=N_FFT, axis=1) * fold[None, :]
        H = np.exp(np.fft.rfft(cep, axis=1))
        ip = np.floor(p).astype(int)
        fr = p - ip
        H *= np.exp(-2j * np.pi * kbin[None, :] * fr[:, None] / N_FFT)
        amp = va[c0:c0 + CH] * np.sqrt(sr / np.maximum(f0p, 30.0))
        if shimmer > 0:
            amp = amp * np.exp(shimmer * rng.standard_normal(k))
        pulses = np.fft.irfft(H, n=N_FFT, axis=1)[:, :PULSE_LEN] * taper[None, :] * amp[:, None]
        _ola(out, ip, pulses)

    # ---------------------------------------------------------------- noise
    out[:n_samples] += _noise(fp, voice, sr, rng, gain, v_amp, phase, hop, n_samples)
    return out[:n_samples]


def _noise(fp: FrameParams, voice, sr, rng, gain, v_amp, phase, hop, n_samples) -> np.ndarray:
    """Breath + aspiration + frication noise via short-time inverse FFT overlap-add.

    * air: breath noise riding on the voice.  Its spectrum is the voiced spectrum
      minus a harmonics-to-noise ratio (HNR) that falls with frequency, so a
      breathy voice is airy at the top while the low harmonics stay clean.
    * aspiration: h / stop release / breath intake / whisper, shaped by the vocal tract.
    * frication: resonance-shaped noise for s, sh, f, bursts...
    """
    N = NOISE_FFT
    H = int(round(hop))
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    n = fp.n
    breath = np.clip(fp.breath, -1, 1)
    whisper = fp.voiced * (1.0 - np.clip(fp.voicing, 0, 1)) * 0.35
    asp = (fp.asp + whisper) * gain
    air = v_amp
    fric = fp.fric * gain
    active = (asp > 1e-6) | (air > 1e-6) | (fric > 1e-6)
    noise = np.zeros(max(n * H, n_samples) + 3 * N)
    air_out = np.zeros_like(noise)
    if not active.any():
        return noise[:n_samples]
    win = np.hanning(N + 1)[:-1]
    scale = np.sqrt(N / 1.5)
    hp_asp = -10 * np.log10(1 + (400.0 / np.maximum(freqs, 1.0)) ** 4)
    rad = 10 * np.log10(1 + (freqs / 350.0) ** 2) - 6.0
    octs = np.log2(np.maximum(freqs, 250.0) / 250.0)
    # HNR at 250 Hz: ~27 dB for a modal voice, lower when breathy / for airy voices
    hnr0 = 30.0 - 13.0 * np.clip(breath, 0, 1) - 4.0 * np.clip(breath, -1, 0) \
        + 4.0 * np.clip(fp.tension, -1, 1) - (voice.aspiration_db + 30.0)
    hnr_slope = 4.2  # dB per octave
    frames = np.nonzero(active)[0]
    CH = 512
    for c0 in range(0, len(frames), CH):
        fi = frames[c0:c0 + CH]
        k = len(fi)
        vt = vocal_tract_db(freqs, fp.formants[fi], fp.bandwidths[fi], fp.nasal[fi], fp.nasal_zero[fi], sr)
        # aspiration: vocal-tract shaped, power-normalised
        asp_db = vt + rad[None, :] + hp_asp[None, :]
        asp_db -= 10 * np.log10(np.mean(10 ** (asp_db / 10.0), axis=1))[:, None]
        mag = 10 ** (asp_db / 20.0) * asp[fi][:, None]
        # air: voiced spectrum (normalised exactly like the pulses) minus the HNR curve
        vdb = vt + source_db(freqs, fp.f0[fi], fp.tension[fi], breath[fi], voice)
        vdb -= VOICED_NORM * 10 * np.log10(np.mean(10 ** (vdb / 10.0), axis=1))[:, None]
        air_db = vdb - (hnr0[fi][:, None] - hnr_slope * octs[None, :])
        air_mag = 10 ** (air_db / 20.0) * air[fi][:, None]
        # frication: resonance around fc with a place-dependent high-pass
        fc = fp.fric_fc[fi][:, None]
        bw = fp.fric_bw[fi][:, None]
        x = (freqs[None, :] - fc) / (0.5 * bw)
        res = 1.0 / np.sqrt(1.0 + x ** 4)
        hp = 1.0 / np.sqrt(1.0 + (np.minimum(0.45 * fc, 2600.0) / np.maximum(freqs[None, :], 1.0)) ** 6)
        fric_shape = (res + 0.06) * hp
        fric_shape /= np.sqrt(np.mean(fric_shape ** 2, axis=1))[:, None]
        fric_mag = fric_shape * fric[fi][:, None]
        shape = (k, len(freqs))
        X = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        Y = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        Z = (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)
        starts = fi * H - N // 2 + H // 2 + N  # offset by N so indices stay positive
        seg = np.fft.irfft(X * mag + Y * fric_mag, n=N, axis=1) * win[None, :] * scale
        _ola(noise, starts, seg)
        seg = np.fft.irfft(Z * air_mag, n=N, axis=1) * win[None, :] * scale
        _ola(air_out, starts, seg)
    noise = noise[N:N + n_samples]  # power-normalised: E[noise^2] = amplitude^2
    air_out = air_out[N:N + n_samples]
    # breath noise is pulsed by the glottal cycle (open phase)
    ph = phase[:n_samples] - np.floor(phase[:n_samples])
    gate = 0.3 + 1.4 * (0.5 - 0.5 * np.cos(2 * np.pi * ph)) ** 2
    gate /= np.sqrt(np.mean(gate ** 2))
    return noise + air_out * gate
