"""Voice-quality numbers for a sung clip: how breathy, how noisy, how steady.

usage: python3 -I voiceq.py FILE_OR_DIR [...] [--json OUT]

Per file (all over "active" frames = within 35 dB of the loudest frame):
  cpp      median cepstral peak prominence, dB. THE breathiness measure: higher = clearer
           phonation, lower = breathy / hoarse / whispery.
  cpp_v    the same over voiced frames only.
  hnr      median harmonics-to-noise ratio of voiced frames, dB (autocorrelation method).
  voiced   share of active frames that are voiced (0..1). Low = a lot of noise-only time.
  hf       energy above 5 kHz relative to 100 Hz..5 kHz on voiced frames, dB. High = airy.
  lvl_sd   standard deviation of the voiced-frame level, dB. High = pumping / fading notes.
  dip      10th percentile of voiced-frame level minus the median, dB.
  run      median length of an unbroken voiced run, ms. Short = choppy phonation.

Import: from voiceq import measure; measure(x, sr) -> dict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SR = 16000
N = 1024  # 64 ms
HOP = 160  # 10 ms
NFFT = 2048


def _resample(x, sr, to):
    if sr == to:
        return x
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(int(sr), int(to))
    return resample_poly(x, to // g, sr // g).astype(np.float32)


def _frames(x, n, hop):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    k = 1 + (len(x) - n) // hop
    idx = np.arange(n)[None, :] + hop * np.arange(k)[:, None]
    return x[idx]


def measure(x, sr):
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    full = x
    x = _resample(x, sr, SR)
    fr = _frames(x, N, HOP)
    win = np.hanning(N).astype(np.float32)
    rms = 20 * np.log10(np.sqrt((fr**2).mean(axis=1)) + 1e-9)
    active = rms > rms.max() - 35.0
    fw = fr * win

    # cepstral peak prominence
    spec = 20 * np.log10(np.abs(np.fft.rfft(fw, NFFT, axis=1)) + 1e-7)
    cep = 20 * np.log10(np.abs(np.fft.irfft(spec, axis=1))[:, : NFFT // 2] + 1e-7)
    q = np.arange(NFFT // 2) / SR
    lo, hi = int(SR / 600), int(SR / 60)
    reg = slice(int(0.001 * SR), NFFT // 2)
    A = np.stack([q[reg], np.ones(reg.stop - reg.start)], axis=1)
    coef, *_ = np.linalg.lstsq(A, cep[:, reg].T, rcond=None)
    pk = lo + np.argmax(cep[:, lo:hi], axis=1)
    cpp = cep[np.arange(len(cep)), pk] - (coef[0] * q[pk] + coef[1])

    # harmonics-to-noise ratio (Boersma: autocorrelation of the windowed frame over the window's)
    ac = np.fft.irfft(np.abs(np.fft.rfft(fw, 2 * N, axis=1)) ** 2, axis=1)[:, :N]
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, 2 * N)) ** 2)[:N]
    r = (ac / (ac[:, :1] + 1e-12)) / (wac / wac[0])[None, :]
    rmax = np.clip(r[:, lo:hi].max(axis=1), 1e-3, 0.999)
    hnr = 10 * np.log10(rmax / (1 - rmax))
    voiced = active & (rmax > 0.45)

    # airiness: high band against the voice band, at the original rate
    n2 = int(round(N * sr / SR))
    h2 = int(round(HOP * sr / SR))
    f2 = _frames(full, n2, h2)[: len(fr)] * np.hanning(n2)
    p2 = np.abs(np.fft.rfft(f2, axis=1)) ** 2
    f = np.fft.rfftfreq(n2, 1 / sr)
    top = min(16000.0, sr / 2 - 1)
    hfr = 10 * np.log10((p2[:, (f >= 5000) & (f <= top)].sum(axis=1) + 1e-12)
                        / (p2[:, (f >= 100) & (f < 5000)].sum(axis=1) + 1e-12))
    v2 = voiced[: len(hfr)]

    runs, n = [], 0
    for v in voiced:
        if v:
            n += 1
        elif n:
            runs.append(n)
            n = 0
    if n:
        runs.append(n)

    def med(a):
        return float(np.median(a)) if len(a) else float("nan")

    lv = rms[voiced]
    return {
        "cpp": round(med(cpp[active]), 2),
        "cpp_v": round(med(cpp[voiced]), 2),
        "hnr": round(med(hnr[voiced]), 2),
        "voiced": round(float(voiced.sum() / max(1, active.sum())), 3),
        "hf": round(med(hfr[v2]), 2),
        "lvl_sd": round(float(lv.std()) if len(lv) else float("nan"), 2),
        "dip": round(float(np.percentile(lv, 10) - np.median(lv)) if len(lv) else float("nan"), 2),
        "run": round(med(np.array(runs)) * 1000 * HOP / SR, 0),
        "seconds": round(len(full) / sr, 2),
    }


def main(argv):
    import soundfile as sf

    out = None
    if "--json" in argv:
        i = argv.index("--json")
        out = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    files = []
    for a in argv:
        p = Path(a)
        files += sorted(list(p.glob("*.wav")) + list(p.glob("*.mp3")) + list(p.glob("*.flac"))) if p.is_dir() else [p]
    res = {}
    keys = ["cpp", "cpp_v", "hnr", "voiced", "hf", "lvl_sd", "dip", "run", "seconds"]
    print(f"{'file':38s}" + "".join(f"{k:>8s}" for k in keys))
    for p in files:
        x, sr = sf.read(str(p), dtype="float32", always_2d=False)
        m = measure(x, sr)
        res[str(p)] = m
        print(f"{p.name[:38]:38s}" + "".join(f"{m[k]:8.2f}" for k in keys))
    if out:
        Path(out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
