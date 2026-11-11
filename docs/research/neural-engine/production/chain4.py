"""A vocal chain for the neural lead, built from Synth2's mixing primitives (synth2/engine/mixing.py; numpy only).

    y, info = chain4.process(x, sr, chain4.LEAD)          # mono in, mono out, same length
    st = chain4.adt(x, sr, ...)                           # a double made from the take itself
    wet = chain4.room(x, sr, ...)                         # short mono room, to add to the channel

Stages of `process` in the order of chain4.LEAD (a stage is switched off by setting its entry to None):
  eq        static tone: low cut, a small presence bell, no air
  deess     dynamic cut of the 5-9 kHz band when it rises over the full-band level
  rider     slow gain ride of the sung level toward the take's median, bounded, frozen in silence
  comp1     the leveller: a low-threshold 2:1 compressor that follows the syllables
  comp2     a compressor on the loud syllables. Both read the sung level only: noise frames (consonants, breaths,
            gaps) get the gain of the sung frames next to them, so compression does not lift them
  cons      consonant leveller: noise frames in the consonants' level range move toward one level under the sung
            level (quiet fricatives up, loud bursts down); the breaths under that range are left alone
  expand, bursts   (not in LEAD) downward expander far under the sung level; a cap on loud noise frames
  breaths   quiet noise in the gaps (breaths) gets no more gain than the median sung frame plus `max_up_db`
  crest     transient limiter: peaks stay within `crest` dB of the loud level
A config is a dict {stage name: settings}; the stages run in the dict's order and a name may carry a digit
("comp1", "comp2").
The output has the input's active level (one static gain), so the channel's fader keeps its meaning.
`info` reports every stage's gain curve (max, mean, share of time active) and `gr_speed`: the fastest change of the
summed dynamic gain over sung frames, dB per 100 ms (pumping shows up as a high value together with a deep range).
"""
from __future__ import annotations

import numpy as np

from synth2.engine import mixing as mx

# the lead: Qixuan five semitones under the written key (fundamentals E4-C5, 330-520 Hz)
LEAD = {
    # low cut, a small presence bell, no air (the bank's top above 9 kHz is aspiration noise). Each dB of the bell
    # costs about 0.3 dB of the hnr measure; a cut of the fundamental's range (480 Hz) cost 0.45 dB and is left out
    "eq": [["hp", 110, 0, 0.7071, 4], ["peak", 3200, 1.5, 0.8], ["highshelf", 10000, -2.0, 0.7]],
    "deess": {"freq": 6800.0, "q": 1.0, "rel_db": -7.0, "max_cut_db": 6.0, "ratio": 3.0},
    "rider": {"window": 0.12, "range_db": 8.0, "smooth": 0.04, "strength": 1.0},
    # the leveller: 2:1 from 26 dB under the loud level, fast enough to follow a syllable
    "comp1": {"offset_db": -26.0, "ratio": 2.0, "attack": 0.004, "release": 0.02, "knee_db": 10.0},
    # the loud syllables
    "comp2": {"offset_db": -6.0, "ratio": 3.0, "attack": 0.002, "release": 0.08, "knee_db": 6.0},
    "cons": {"target_db": -8.0, "ratio": 2.0, "max_up_db": 4.0, "max_down_db": 9.0, "floor_db": -24.0, "width_db": 6.0},
    "breaths": {"under_db": -18.0, "width_db": 4.0, "max_up_db": 3.0},
    "crest": 11.0,
}
# the harmony and the double: the same chain with the consonants pushed back and a darker top
HARMONY = dict(LEAD, eq=[["hp", 110, 0, 0.7071, 4], ["highshelf", 5000, -4.0, 0.7]],
               deess={"freq": 6800.0, "q": 1.0, "rel_db": -10.0, "max_cut_db": 10.0, "ratio": 4.0},
               cons={"target_db": -18.0, "ratio": 4.0, "max_up_db": 0.0, "max_down_db": 15.0, "floor_db": -30.0, "width_db": 6.0})


def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-20))


def level_db(x, sr, window):
    """RMS level (dB) on the 1 ms grid over a centred window (s)."""
    p = mx.block_power(x, mx.grid_hop(sr))
    return _db(mx._moving_mean(p, max(1, int(round(window / mx.GRID_S)))))


def periodicity(x, sr):
    """0..1 on the 1 ms grid: the highest normalised autocorrelation of the signal under 1.5 kHz at a lag of a sung
    pitch (70-600 Hz), in 48 ms frames every 4 ms. Vowels and voiced consonants are near 1; fricatives, stop bursts
    and breaths (also the low, formant-shaped noise of an inhale) are low."""
    hop = mx.grid_hop(sr)
    nb = max(1, -(-len(x) // hop))
    lo = mx.eq(x, [("lp", 1500, 0, 0.7071, 4)], sr)
    d = max(1, sr // 5500)
    z = lo[::d]
    r = sr / d
    n, h = int(0.048 * r), max(1, int(round(0.004 * r)))
    if len(z) < n + h:
        return np.zeros(nb)
    k = 1 + (len(z) - n) // h
    idx = np.arange(n)[None, :] + h * np.arange(k)[:, None]
    win = np.hanning(n)
    fr = z[idx] * win
    ac = np.fft.irfft(np.abs(np.fft.rfft(fr, 2 * n, axis=1)) ** 2, axis=1)[:, :n]
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, 2 * n)) ** 2)[:n]
    e0 = ac[:, 0]
    rr = ac / (e0[:, None] + 1e-20) / (wac / wac[0])[None, :]
    a, b = max(2, int(r / 600)), min(n - 1, int(r / 70))
    rmax = np.clip(rr[:, a:b].max(axis=1), 0.0, 1.0)
    rmax[e0 < e0.max() * 1e-7] = 0.0  # silence
    t_fr = (np.arange(k) * h + n / 2) * d / sr
    return np.interp((np.arange(nb) + 0.5) * hop / sr, t_fr, rmax)


def noise_share(x, sr):
    """0..1 on the grid: 1 where the frame is noise (consonant noise, a breath, a gap), 0 where it is sung:
    the periodicity mapped from 0.6 (0) down to 0.4 (1). On a relaxed-verse take 95 % of the breath frames are under
    0.47 and 94 % of the other active frames over it."""
    return np.clip((0.6 - periodicity(x, sr)) / 0.2, 0.0, 1.0)


def sung_level(lvl, noise, floor_db=30.0):
    """(median level of the sung frames, mask of sung frames): frames within `floor_db` of the loud level that are
    not noise."""
    act = lvl > np.percentile(lvl, 95) - floor_db
    sung = act & (noise < 0.5)
    if sung.sum() < 20:
        sung = act
    return float(np.median(lvl[sung])) if sung.any() else -60.0, sung


def active_rms(x, lvl, hop):
    """RMS of the blocks within 40 dB of the loud (95th percentile) level."""
    k = min(len(lvl), len(x) // hop)
    act = lvl[:k] > np.percentile(lvl, 95) - 40
    return float(np.sqrt(np.mean(x[: k * hop].reshape(k, hop)[act] ** 2))) if act.any() else 0.0


def _fill(x, mask):
    """x with every entry where mask is False replaced by the last entry before it where mask is True (-inf dB
    before the first)."""
    idx = np.where(mask, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], -200.0)


def sung_env(lvl, sung, window):
    """Level (dB) of the sung frames around every block: their power mean over a centred `window` (s); where less
    than half of the window is sung, the higher of the nearest such values before and after."""
    w = max(1, int(round(window / mx.GRID_S)))
    num = mx._moving_mean(np.where(sung, 10 ** (lvl / 10), 0.0), w)
    den = mx._moving_mean(sung.astype(np.float64), w)
    have = den > 0.5  # the window's centre lies inside a sung stretch, not on the tail of one
    env = _db(num / np.maximum(den, 1e-9))
    return np.where(have, env, np.maximum(_fill(env, have), _fill(env[::-1], have[::-1])[::-1]))


def _stat(info, name, g_db):
    info[name] = {"max_db": round(float(-g_db.min()), 2), "mean_db": round(float(-g_db.mean()), 2),
                  "active_pct": round(float(100 * np.mean(g_db < -1.0)), 1)}


def process(x, sr, cfg=None, return_gain=False):
    cfg = LEAD if cfg is None else cfg
    x = np.asarray(x, np.float64)
    n, hop = len(x), mx.grid_hop(sr)
    info: dict = {}
    if not np.any(x):
        return x.astype(np.float32), info
    lvl0 = level_db(x, sr, 0.02)
    rms_in = active_rms(x, lvl0, hop)
    y = x
    total = np.zeros(len(lvl0))  # summed dynamic gain (dB) on the grid
    quiet0 = sung0 = None

    def gain(g_db):
        nonlocal y, total
        total[: len(g_db)] += g_db[: len(total)]
        y = y * mx.grid_to_samples(mx.db_to_gain(g_db), n, hop)

    for name, c in cfg.items():  # stages run in the order of the dict; the kind is the key without its digits
        if c is None:
            continue
        kind = name.rstrip("0123456789")
        if kind == "eq":
            y = mx.eq(y, c, sr)
        elif kind == "deess":
            bp = mx.eq(y, [("bandpass", c["freq"], 0, c["q"])], sr)
            g = mx.dynamic_band_db(mx.detector_db(y, sr, "rms", 0.005), mx.detector_db(bp, sr, "rms", 0.005), c["rel_db"],
                                   ratio=c.get("ratio", 3.0), max_cut_db=c["max_cut_db"])
            _stat(info, name, g)
            y = y + (mx.grid_to_samples(mx.db_to_gain(g), n, hop) - 1.0) * bp
        elif kind == "bursts":
            noise = noise_share(y, sr)
            lvl = level_db(y, sr, 0.01)
            ref, _ = sung_level(level_db(y, sr, 0.02), noise)
            over = np.maximum(lvl - (ref + c["cap_db"]), 0.0) * (1 - 1 / c["ratio"]) * noise
            g = -mx.ar_smooth(np.minimum(over, c["max_cut_db"]), c["attack"] / mx.GRID_S, c["release"] / mx.GRID_S, start=0.0)
            _stat(info, name, g)
            gain(g)
        elif kind == "rider":
            noise = noise_share(y, sr)
            lvl = level_db(y, sr, 0.02)
            ref, sung = sung_level(lvl, noise)
            # the sung level around each block (power mean over the window, sung frames only; in a gap the higher
            # of the levels before and after it, so a breath is not lifted like the tail of the note before it)
            want = c.get("strength", 1.0) * (ref - sung_env(lvl, sung, c["window"]))
            g = mx._moving_mean(np.clip(want, -c["range_db"], c["range_db"]), max(1, int(round(c["smooth"] / mx.GRID_S))))
            info[name] = {"max_up_db": round(float(g.max()), 2), "max_down_db": round(float(-g.min()), 2),
                          "sd_db": round(float(g[sung].std()), 2)}
            gain(g)
        elif kind == "comp":
            lvl = mx.detector_db(y, sr, "rms", c.get("window", 0.01))
            thr = mx.relative_threshold(lvl, c["offset_db"])
            det = lvl
            if c.get("follow_sung", True):
                # noise frames (consonants, breaths, the gaps) are compressed like the sung frames around them, so
                # the compressor does not lift them against the vowels: there the detector reads the sung level
                # of the surrounding 0.3 s (the nearest sung stretch on either side when there is none that close)
                _, sung = sung_level(level_db(y, sr, 0.02), noise_share(y, sr))
                det = np.where(sung, lvl, sung_env(lvl, sung, c.get("follow_window", 0.3)))
            g = mx.compressor_db(det, thr, c["ratio"], c["attack"], c["release"], c.get("knee_db", 6.0),
                                 max_gr_db=c.get("max_gr_db"))
            _stat(info, name, g)
            gain(g)
        elif kind == "cons":
            # consonant leveller: noise frames in the consonants' level range are moved toward `target_db` under
            # the sung level (quiet fricatives up, loud bursts down); the breaths, under `floor_db`, are left alone
            noise = noise_share(y, sr)
            lvl = level_db(y, sr, 0.015)
            ref, _ = sung_level(level_db(y, sr, 0.02), noise)
            rel = lvl - ref
            g = np.clip((c["target_db"] - rel) * (1 - 1 / c.get("ratio", 2.0)), -c.get("max_down_db", 9.0), c.get("max_up_db", 4.0))
            g = g * noise * np.clip((rel - c["floor_db"]) / c.get("width_db", 6.0), 0.0, 1.0)
            g = mx._moving_mean(g, 5)
            info[name] = {"max_up_db": round(float(g.max()), 2), "max_down_db": round(float(-g.min()), 2),
                          "up_pct": round(float(100 * np.mean(g > 1.0)), 1), "down_pct": round(float(100 * np.mean(g < -1.0)), 1)}
            gain(g)
        elif kind == "breaths":
            # breaths and other quiet noise in the gaps: the stages before give them the gain of the sung frames
            # around them, and next to a soft line that is a lift of 10 dB and more. Their summed gain is held at
            # `max_up_db` over the median gain of the sung frames. "Quiet noise" = noise frames whose level in the
            # input is more than `under_db` under the input's sung level (the consonants are louder than that).
            if quiet0 is None:
                noise0 = noise_share(x, sr)
                ref0, sung0 = sung_level(lvl0, noise0)
                quiet0 = noise0 * np.clip(((ref0 + c["under_db"]) - level_db(x, sr, 0.03)) / c.get("width_db", 4.0), 0.0, 1.0)
            med = float(np.median(total[sung0])) if sung0.any() else 0.0
            g = -np.maximum(total - (med + c["max_up_db"]), 0.0) * quiet0
            g = mx._moving_mean(g, 9)
            info[name] = {"max_db": round(float(-g.min()), 2), "active_pct": round(float(100 * np.mean(g < -1.0)), 1)}
            gain(g)
        elif kind == "expand":
            # downward expander under the sung level: what the compressors lifted out of the gaps between the
            # lines (the engine's low noise, tails) goes back down
            noise = noise_share(y, sr)
            lvl = level_db(y, sr, 0.03)
            ref, _ = sung_level(level_db(y, sr, 0.02), noise)
            under = np.maximum((ref + c["offset_db"]) - lvl, 0.0) * (c["ratio"] - 1.0)
            g = -mx.ar_smooth(np.minimum(under, c["max_db"]), c.get("attack", 0.03) / mx.GRID_S,
                              c.get("release", 0.004) / mx.GRID_S, start=0.0)
            info[name] = {"max_db": round(float(-g.min()), 2), "active_pct": round(float(100 * np.mean(g < -1.0)), 1)}
            gain(g)
        elif kind == "crest":
            lvl = _db(mx._moving_mean(mx.block_power(y, hop), 10))
            ceil_db = mx.relative_threshold(lvl, float(c))
            gl = mx.limiter_gain(np.abs(y), sr, 10 ** (ceil_db / 20.0), lookahead=0.001, release=0.03, release_slow=0.0)
            info[name + "_max_db"] = round(float(-mx.gain_to_db(gl.min())), 2)
            y = y * gl
        else:
            raise KeyError(f"unknown stage {name!r}")
    # static make-up: the input's active level
    lvl1 = level_db(y, sr, 0.02)
    rms_out = active_rms(y, lvl1, hop)
    mk = rms_in / rms_out if rms_in > 0 and rms_out > 0 else 1.0
    y = y * mk
    info["makeup_db"] = round(float(20 * np.log10(mk)), 2)
    # how fast the summed gain moves while the voice sings (dB per 100 ms), and its range there
    noise = noise_share(y, sr)
    _, sung = sung_level(lvl1, noise)
    dg = np.abs(total[100:] - total[:-100])[sung[100:] & sung[:-100]]
    info["gr_speed"] = {"p50": round(float(np.percentile(dg, 50)), 2), "p95": round(float(np.percentile(dg, 95)), 2),
                        "max": round(float(dg.max()), 2)} if len(dg) else {}
    info["gain_range_db"] = [round(float(np.percentile(total[sung], q)), 2) for q in (1, 50, 99)] if sung.any() else []
    if return_gain:
        return y.astype(np.float32), info, total
    return y.astype(np.float32), info


def frac_delay(x, delay_samples):
    """x delayed by a time-varying number of samples (array, >= 0), linear interpolation."""
    pos = np.arange(len(x)) - delay_samples
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    a = np.where((i >= 0) & (i < len(x)), x[np.clip(i, 0, len(x) - 1)], 0.0)
    b = np.where((i + 1 >= 0) & (i + 1 < len(x)), x[np.clip(i + 1, 0, len(x) - 1)], 0.0)
    return a * (1 - f) + b * f


def adt(x, sr, delay=0.018, depth=0.004, rate=0.31, detune_cents=6.0, dark=None, phase=0.0):
    """A double made from the take itself: delayed by `delay` +- `depth` s (a slow sine at `rate` Hz, which bends the
    pitch by up to 2 pi rate depth), detuned by a constant `detune_cents` (the delay drifts by that much and snaps
    back in silence, see below), with the EQ `dark`."""
    x = np.asarray(x, np.float64)
    t = np.arange(len(x)) / sr
    d = delay + depth * np.sin(2 * np.pi * rate * t + phase)
    if detune_cents:
        # constant detune = a delay that grows (or shrinks) at a constant rate; it is reset at every silence longer
        # than 80 ms so that it never leaves [delay - 6 ms, delay + 6 ms] for long
        rate_s = 2 ** (detune_cents / 1200.0) - 1.0  # s of delay per s; positive detune = delay shrinking
        lvl = level_db(x, sr, 0.02)
        quiet = lvl < np.percentile(lvl, 95) - 45
        hop = mx.grid_hop(sr)
        drift = np.zeros(len(lvl))
        acc, run = 0.0, 0
        for i in range(len(lvl)):
            if quiet[i]:
                run += 1
                if run > 80:
                    acc = 0.0
            else:
                run = 0
                acc -= rate_s * mx.GRID_S
                if abs(acc) > 0.006:
                    acc = float(np.sign(acc)) * 0.006
            drift[i] = acc
        d = d + mx.grid_to_samples(drift, len(x), hop)
    y = frac_delay(x, np.maximum(d, 0.0) * sr)
    if dark:
        y = mx.eq(y, dark, sr)
    return y.astype(np.float32)


def room_ir(sr, decay=0.18, predelay=0.012, seed=7, lo=300.0, hi=6000.0):
    """Mono impulse response of a small room: band-limited noise with an exponential decay (`decay` = RT60, s) after
    `predelay`, unit energy."""
    n = int(sr * (predelay + decay * 1.3))
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr - predelay
    h = rng.standard_normal(n) * np.where(t >= 0, np.exp(-6.91 * np.maximum(t, 0) / decay), 0.0)
    h = mx.eq(h, [("hp", lo, 0, 0.7071, 2), ("lp", hi, 0, 0.7071, 2)], sr)
    return h / (np.sqrt(np.sum(h ** 2)) + 1e-12)


def room(x, sr, decay=0.18, predelay=0.012, level_db_=-12.0, seed=7):
    """The wet signal of the small room for the mono channel `x` (same length), `level_db_` under the dry level."""
    from synth2.engine import effects

    h = room_ir(sr, decay, predelay, seed)
    return (effects.fft_convolve(np.asarray(x, np.float64), h)[: len(x)] * 10 ** (level_db_ / 20)).astype(np.float32)
