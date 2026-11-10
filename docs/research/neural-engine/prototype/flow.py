"""How continuous a sung clip is: holes, short voiced runs, noise puffs.

usage: python3 -I flow.py FILE [...]
Per file, inside phrases (stretches of sound with no silence longer than 0.3 s):
  holes/s   dips of 20-250 ms where the level falls more than 12 dB under the phrase's median level
  hole%     share of in-phrase time spent in such dips
  voiced%   share of in-phrase frames that are voiced
  run       median unbroken voiced run, ms
  breaks/s  ends of voiced runs per second (how often the voice stops)
  puffs/s   noise bursts: frames that are unvoiced, within 15 dB of the phrase level, 30-200 ms long
  puff_dB   level of those bursts against the phrase's median voiced level (0 = as loud as the vowels)
  lvl_sd    standard deviation of the voiced-frame level, dB
  phr       median phrase length, s
"""
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
sys.path.insert(0, str(Path(__file__).parent))
from voiceq import _resample, _frames, SR, N, HOP

def analyse(x, sr):
    if x.ndim > 1: x = x.mean(axis=1)
    x = _resample(x.astype(np.float32), sr, SR)
    fr = _frames(x, N, HOP)
    # short level (20 ms) for holes, long frames for voicing
    n2 = 320
    f2 = _frames(np.pad(x, (n2 // 2, n2 // 2)), n2, HOP)[: len(fr)]
    lev = 20 * np.log10(np.sqrt((f2 ** 2).mean(axis=1)) + 1e-9)
    win = np.hanning(N).astype(np.float32)
    fw = fr * win
    ac = np.fft.irfft(np.abs(np.fft.rfft(fw, 2 * N, axis=1)) ** 2, axis=1)[:, :N]
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, 2 * N)) ** 2)[:N]
    r = (ac / (ac[:, :1] + 1e-12)) / (wac / wac[0])[None, :]
    lo, hi = int(SR / 600), int(SR / 60)
    rmax = r[:, lo:hi].max(axis=1)
    k = min(len(lev), len(rmax)); lev = lev[:k]; rmax = rmax[:k]
    top = np.percentile(lev, 95)
    sound = lev > top - 40
    # phrases: bridge silences shorter than 0.3 s
    idx = np.flatnonzero(sound)
    if not len(idx): return None
    phrases, a, p = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - p > 30: phrases.append((a, p + 1)); a = i
        p = i
    phrases.append((a, p + 1))
    phrases = [(a, b) for a, b in phrases if b - a >= 30]
    T = holes = holet = brk = puffs = pufft = 0; runs = []; plv = []; vl = []; vshare = 0
    for a, b in phrases:
        L = lev[a:b]; V = (rmax[a:b] > 0.45) & (L > top - 35)
        if V.sum() < 5: continue
        med = np.median(L[V]); T += b - a; vshare += V.sum(); vl += list(L[V])
        low = L < med - 12
        def spans(m):
            out, s = [], None
            for i, v in enumerate(m):
                if v and s is None: s = i
                if not v and s is not None: out.append((s, i)); s = None
            if s is not None: out.append((s, len(m)))
            return out
        for s, e in spans(low):
            if 2 <= e - s <= 25 and s > 0 and e < len(L): holes += 1; holet += e - s
        vr = spans(V); runs += [e - s for s, e in vr]; brk += max(0, len(vr) - 1)
        noise = (~V) & (L > med - 15)
        for s, e in spans(noise):
            if 3 <= e - s <= 20: puffs += 1; pufft += e - s; plv.append(L[s:e].max() - med)
    sec = T * HOP / SR
    return dict(holes=holes / sec, holep=100 * holet / T, voiced=100 * vshare / T, run=float(np.median(runs)) * 10 if runs else 0,
                breaks=brk / sec, puffs=puffs / sec, puffdb=float(np.mean(plv)) if plv else float("nan"),
                lvl_sd=float(np.std(vl)), phr=float(np.median([(b - a) for a, b in phrases])) * HOP / SR, sec=sec)

if __name__ == "__main__":
    print(f"{'file':44s} holes/s hole% voiced%  run  breaks/s puffs/s puff_dB lvl_sd  phr")
    for a in sys.argv[1:]:
        x, sr = sf.read(a, dtype="float32", always_2d=False)
        m = analyse(x, sr)
        if m: print(f"{Path(a).name[:44]:44s} {m['holes']:6.2f} {m['holep']:5.1f} {m['voiced']:6.1f} {m['run']:5.0f} {m['breaks']:7.2f} {m['puffs']:7.2f} {m['puffdb']:7.1f} {m['lvl_sd']:6.2f} {m['phr']:5.2f}")
