"""Consonant measures of a sung clip against its phoneme times, for the reference songs and for our takes.

usage: cd /home/user/synth2 && python3 -I $S/clar/cm.py ref [NAME ...]            (the bank's sample songs, q7)
       cd /home/user/synth2 && python3 -I $S/clar/cm.py ours WAV DUMP.json [...]  (--offset S: the wav starts S s
                                                                                   after the dump's time 0)
       import cm; row = cm.measure(x, sr, phones)       # phones = [(t0, t1, phoneme)]

Levels are RMS in 10 ms windows every 5 ms, dB re the clip's median vowel level (median over the vowels of each
vowel's median level, full band). Per consonant class:
  n, len    number, median length (ms)
  med       median over the consonants of their median level (how loud the consonant is as a whole)
  peak      median of their loudest 10 ms, full band
  hf        median of their loudest 10 ms above 4 kHz (noise: bursts, sibilance, aspiration)
  dip       median of their quietest 10 ms (the closure of a stop: how far the level falls)
For the stops and affricates that a vowel follows (the released ones; a stop before another consonant or a rest
has no release to measure) also `burst`: the loudest 10 ms above 4 kHz between the stop's start and 15 ms after
its end, and `rise`: the largest rise of the level above 2 kHz over 10 ms inside that span (how sudden the release
is); `nrel` = how many of the class are released.
Classes (EN | ZH): stop = p t k (aspirated in ZH) | aff = ch (c ch q) | ustop = ZH only: b d g z zh j, voiceless
and unaspirated | vstop = b d g jh dx (EN) | sib = s sh (s sh x) | fric = f th hh (f h) | vfric = z zh v dh (r) |
nasal | glide.
Modulation: the level envelope inside phrases (linear RMS, 5 ms hop, divided by its mean), Welch spectrum over 2 s;
`mod` = rms modulation depth in 2-4, 4-8, 8-16 Hz and 2-20 Hz, in dB (20 log10). The 4-16 Hz band is the syllable and
consonant rate; a levelled, reverberant voice loses depth there.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
FRAME = 512 / 44100
EN = {"stop": "p t k", "aff": "ch", "vstop": "b d g jh dx", "sib": "s sh", "fric": "f th hh", "vfric": "z zh v dh",
      "nasal": "m n ng", "glide": "l r w y _r"}
ZH = {"stop": "p t k", "aff": "c ch q", "ustop": "b d g z zh j", "sib": "s sh x", "fric": "f h", "vfric": "r",
      "nasal": "m n", "glide": "l y w y0 w0"}
ORDER = ("vowel", "nasal", "glide", "vfric", "vstop", "ustop", "fric", "sib", "aff", "stop")
HOP_S = 0.005


def klass(p):
    if p in ("SP", "AP"):
        return p
    lang, _, b = p.rpartition("/")
    for k, v in (ZH if lang == "zh" else EN).items():
        if b in v.split():
            return k
    return "vowel"


def band(x, sr, lo=None, hi=None):
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / sr)
    if lo:
        X[f < lo] = 0
    if hi:
        X[f > hi] = 0
    return np.fft.irfft(X, len(x)).astype(np.float32)


def levels(x, sr, win=0.010, hop=HOP_S):
    """Level (dB) of `win` s windows every `hop` s; entry i covers [i hop, i hop + win)."""
    h, n = int(round(hop * sr)), int(round(win * sr))
    p = np.concatenate([[0.0], np.cumsum(x.astype(np.float64) ** 2)])
    k = max(0, (len(x) - n) // h + 1)
    i = np.arange(k) * h
    return 10 * np.log10((p[i + n] - p[i]) / n + 1e-12)


def modulation(x, sr):
    """rms modulation depth (dB) of the level envelope inside phrases, per band (Hz)."""
    h = int(round(HOP_S * sr))
    k = len(x) // h
    e = np.sqrt((x[: k * h].reshape(k, h).astype(np.float64) ** 2).mean(axis=1))
    lv = 20 * np.log10(e + 1e-9)
    act = lv > np.percentile(lv, 95) - 40
    idx = np.flatnonzero(act)
    if len(idx) < 400:
        return {}
    # phrases: silences shorter than 0.3 s are part of the phrase
    cuts = np.flatnonzero(np.diff(idx) > int(0.3 / HOP_S))
    a = np.concatenate([[idx[0]], idx[cuts + 1]])
    b = np.concatenate([idx[cuts], [idx[-1]]]) + 1
    fs = 1.0 / HOP_S
    n = int(2.0 * fs)
    win = np.hanning(n)
    acc, cnt = np.zeros(n // 2 + 1), 0
    for s0, s1 in zip(a, b):
        seg = e[s0:s1]
        if len(seg) < n:
            continue
        for j in range(0, len(seg) - n + 1, n // 2):
            w = seg[j:j + n]
            w = w / (w.mean() + 1e-12) - 1.0
            acc += np.abs(np.fft.rfft(w * win)) ** 2 / (win ** 2).sum() / n * 2  # one-sided power per bin
            cnt += 1
    if not cnt:
        return {}
    psd = acc / cnt
    f = np.fft.rfftfreq(n, 1 / fs)
    out = {}
    for name, (lo, hi) in {"2-4": (2, 4), "4-8": (4, 8), "8-16": (8, 16), "2-20": (2, 20)}.items():
        m = (f >= lo) & (f < hi)
        out[name] = round(float(10 * np.log10(psd[m].sum() + 1e-12)), 2)
    return out


def measure(x, sr, phones, detail=False):
    """Dict: {"vowel_db", "classes": {class: {n, len, med, peak, hf, dip[, burst, rise]}}, "mod": {...},
    "cons_share": share of the sung time taken by consonants, "syll": vowels per second of sung time}."""
    x = np.asarray(x, np.float32)
    if x.ndim > 1:
        x = x.mean(axis=1)
    lv = levels(x, sr)
    lh = levels(band(x, sr, lo=4000.0), sr)
    l2 = levels(band(x, sr, lo=2000.0), sr)
    rows = []
    for n_, (a, b, p) in enumerate(phones):
        k = klass(p)
        if k in ("SP",):
            continue
        i0 = int(round(a / HOP_S))
        i1 = max(i0 + 1, int(round((b - 0.010) / HOP_S)) + 1)
        if i1 > len(lv) or i0 < 0:
            continue
        r = {"k": k, "p": p, "t": a, "len": b - a, "med": float(np.median(lv[i0:i1])), "peak": float(lv[i0:i1].max()),
             "hf": float(lh[i0:i1].max()), "dip": float(lv[i0:i1].min())}
        if k in ("stop", "aff", "ustop", "vstop") and n_ + 1 < len(phones) and klass(phones[n_ + 1][2]) == "vowel":
            j1 = min(len(lv), i1 + 3)
            r["burst"] = float(lh[i0:j1].max())
            seg = l2[i0:j1 + 1]
            r["rise"] = float((seg[2:] - seg[:-2]).max()) if len(seg) > 2 else 0.0
        rows.append(r)
    vow = [r["med"] for r in rows if r["k"] == "vowel"]
    if len(vow) < 3:
        return {}
    ref = float(np.median(vow))
    by = defaultdict(list)
    for r in rows:
        by[r["k"]].append(r)
    out = {"vowel_db": round(ref, 2), "classes": {}}
    for k, rs in by.items():
        c = {"n": len(rs), "len": round(1000 * float(np.median([r["len"] for r in rs])), 1)}
        for q in ("med", "peak", "hf", "dip"):
            c[q] = round(float(np.median([r[q] for r in rs])) - ref, 1)
        rel = [r for r in rs if "burst" in r]
        if rel:
            c["nrel"] = len(rel)
            c["burst"] = round(float(np.median([r["burst"] for r in rel])) - ref, 1)
            c["rise"] = round(float(np.median([r["rise"] for r in rel])), 1)
        out["classes"][k] = c
    sung = [r for r in rows if r["k"] != "AP"]
    tot = sum(r["len"] for r in sung)
    out["cons_share"] = round(sum(r["len"] for r in sung if r["k"] != "vowel") / max(tot, 1e-9), 3)
    out["syll"] = round(len(vow) / max(tot, 1e-9), 2)
    out["mod"] = modulation(x, sr)
    if detail:
        out["rows"] = rows
    return out


def ref_phones(name):
    m = json.load(open(S / f"q7/out/meta/00_ceiling_{name}_human.json"))
    out = []
    for s in m["segs"]:
        e = np.concatenate([[0], np.cumsum(s["durs"])]) * FRAME + s["offset"]
        out += [(float(e[i]), float(e[i + 1]), p) for i, p in enumerate(s["phones"])]
    return out


def dump_phones(dumpf, offset=0.0):
    out = []
    for ph in (json.load(open(dumpf)) if not isinstance(dumpf, list) else dumpf):
        e = (ph["f_start"] + np.concatenate([[0], np.cumsum(ph["durs"])])) * FRAME - offset
        out += [(float(e[i]), float(e[i + 1]), p) for i, p in enumerate(ph["seq"])]
    return out


COLS = ("len", "med", "peak", "hf", "dip", "burst", "rise")


def fmt(name, r, classes=ORDER):
    s = [f"{name}: vowel {r['vowel_db']:.1f} dB, consonants {100 * r['cons_share']:.0f} % of sung time, {r['syll']:.1f} vowels/s, "
         f"mod dB " + " ".join(f"{k} {v:.1f}" for k, v in r.get("mod", {}).items())]
    s.append(f"  {'class':6s} {'n':>4s} " + " ".join(f"{c:>6s}" for c in COLS))
    for k in classes:
        c = r["classes"].get(k)
        if c:
            s.append(f"  {k:6s} {c['n']:4d} " + " ".join(f"{c[q]:6.1f}" if q in c else "      " for q in COLS))
    return "\n".join(s)


def pool(rows):
    """Median of each class measure over several clips' rows (classes weighted equally per clip)."""
    out = {"classes": {}}
    for k in ORDER:
        cs = [r["classes"][k] for r in rows if k in r.get("classes", {})]
        if cs:
            out["classes"][k] = {q: round(float(np.median([c[q] for c in cs])), 1) for q in cs[0] if q != "n"}
            out["classes"][k]["n"] = int(sum(c["n"] for c in cs))
    out["vowel_db"] = float(np.median([r["vowel_db"] for r in rows]))
    out["cons_share"] = float(np.mean([r["cons_share"] for r in rows]))
    out["syll"] = float(np.mean([r["syll"] for r in rows]))
    ks = rows[0].get("mod", {}).keys()
    out["mod"] = {k: round(float(np.mean([r["mod"][k] for r in rows if r.get("mod")])), 2) for k in ks}
    return out


REF_NAMES = ("lianrenxin", "sajiao", "zaijian")


def reference(names=REF_NAMES, cache=True):
    """Measures of the reference songs (cached in clar/ref.json)."""
    f = S / "clar" / "ref.json"
    if cache and f.exists():
        return json.loads(f.read_text())
    import soundfile as sf
    out = {}
    for n in names:
        x, sr = sf.read(S / f"q7/out/00_ceiling_{n}_human.wav", dtype="float32")
        out[n] = measure(x, sr, ref_phones(n))
    out["pooled"] = pool([out[n] for n in names])
    f.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    import soundfile as sf
    a = [v for v in sys.argv[1:] if not v.startswith("--")]
    off = float(sys.argv[sys.argv.index("--offset") + 1]) if "--offset" in sys.argv else 0.0
    if "--offset" in sys.argv:
        a.remove(sys.argv[sys.argv.index("--offset") + 1])
    if a[0] == "ref":
        r = reference(cache=False)
        for n in list(a[1:] or REF_NAMES) + ["pooled"]:
            print(fmt("ref " + n, r[n]))
    else:
        for w, d in zip(a[1::2], a[2::2]):
            x, sr = sf.read(w, dtype="float32")
            print(fmt(Path(w).parent.name + "/" + Path(w).stem, measure(x, sr, dump_phones(d, off))))
