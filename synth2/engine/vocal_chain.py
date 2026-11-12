"""The vocal chain of the neural voice, and what a vocal track gets for its role in the mix.

A neural voice needs other processing than the formant engine's channel strip: its level
moves more from syllable to syllable, its consonants are noise the bank made (not shaped by
rules), and a compressor that reads the whole signal lifts breaths and consonants against the
vowels. The chain here reads the sung level only. Built from :mod:`.mixing` (numpy only).

    y, info = process(x, sr, lead_chain("crisp"))     # mono in, mono out, same length
    y, info = apply_role(x, sr, "harmony")            # the chain and effects of a role
    d = adt(x, sr, **DOUBLE)                          # a double made from the take itself

Stages of :func:`process` (a config is ``{stage name: settings}``; the stages run in the
dict's order, a name may carry a digit as in ``comp1`` / ``comp2``, None switches one off):

=========  ==============================================================================
eq         static tone: low cut, a small presence bell, no air
deess      dynamic cut of the 5-9 kHz band when it rises over the full-band level
rider      slow gain ride of the sung level toward the take's median, bounded, frozen in
           silence
comp       a compressor that reads the sung level only: noise frames (consonants, breaths,
           gaps) get the gain of the sung frames next to them, so it does not lift them
cons       consonant leveller: noise frames in the consonants' level range move toward one
           level under the sung level; with ``max_up_db: 0`` it is a limiter for loud ones
bursts     a cap on loud noise frames
expand     downward expander far under the sung level
lift       gain on sibilants, optionally only above a frequency, up to a ceiling
trans      transient emphasis above a frequency: releases become more sudden
breaths    quiet noise in the gaps gets no more gain than the median sung frame plus
           ``max_up_db``
crest      transient limiter: peaks stay within this many dB of the loud level
=========  ==============================================================================

``cons``, ``bursts`` and ``lift`` find their frames with a noise detector: ``"tilt"`` (the
level above 4 kHz against the level under 1.5 kHz, held for ``sustain`` s: sibilants and
fricatives wherever they stand) or ``"periodicity"`` (no pitch under 1.5 kHz in 48 ms frames:
long or isolated noise; it reads a sibilant between two vowels as sung). ``lift`` uses the
tilt detector unless told otherwise, the other two the periodicity detector.

The output has the input's active level (one static gain), so a fader keeps its meaning.

Roles (:data:`ROLES`): ``lead`` goes through the chain of the track's dial setting
(:func:`lead_chain`: the smoother the dial, the more levelling and the more room); ``answer``
(the lead's short lines elsewhere) through the same chain plus a little room; ``harmony``
through a chain that pushes the consonants back and darkens the top; ``double`` through that
chain and :func:`adt` (the track's audio is then a copy of the lead's take, see
``synth2/neural/renderer.py``); ``adlib`` is band-limited, ducked under the lead and gets one
repeat; ``crowd`` is darkened, ducked and put in a room.

The mixer applies a role to the vocal sources that carry one and were sung by an engine of
:data:`ROLE_ENGINES` (:func:`source_role`); every other source keeps the channel strip.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from . import mixing as mx

ROLES = ("lead", "double", "harmony", "answer", "adlib", "crowd")
DIALS = ("smooth", "balanced", "crisp")
DEFAULT_DIAL = "crisp"

# The base chain (the smooth setting): a lead around E4-C5.
LEAD = {
    # low cut, a small presence bell, no air (the top above 9 kHz of a neural voice is aspiration noise)
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
_LIMIT = {"replace": True, "target_db": -10.0, "ratio": 2.0, "max_up_db": 0.0, "max_down_db": 6.0, "floor_db": -24.0,
          "width_db": 6.0}
_LIFT = {"hf": 2500.0, "floor_db": -24.0, "width_db": 6.0, "detector": "tilt", "sustain": 0.03, "ceiling_db": -5.0}
# What each dial setting changes in the lead's chain (merged over LEAD by `dial_chain`): the
# crisper the setting, the less the level is flattened and the more the consonants stand out.
DIAL_CHAINS: dict[str, dict] = {
    "smooth": {},
    "balanced": {"rider": {"range_db": 5.0, "strength": 0.6}, "comp1": {"ratio": 1.25}, "cons": dict(_LIMIT),
                 "lift": dict(_LIFT, db=2.0)},
    "crisp": {"rider": {"range_db": 4.0, "strength": 0.5}, "comp1": None, "comp2": {"ratio": 2.0}, "cons": None,
              "trans": {"lo": 2000.0, "amount": 0.4, "max_db": 5.0}, "lift": dict(_LIFT, db=2.5),
              "cons2": dict(_LIMIT)},
}
# Factor on the lead's reverb send per dial setting: a room fills the stop closures (with the
# default send of 0.25 this gives 0.50, 0.25 and 0.15).
REVERB_SCALE = {"smooth": 2.0, "balanced": 1.0, "crisp": 0.6}
# the harmony and the double: the base chain with the consonants pushed back and a darker top
HARMONY = dict(LEAD, eq=[["hp", 110, 0, 0.7071, 4], ["highshelf", 5000, -4.0, 0.7]],
               deess={"freq": 6800.0, "q": 1.0, "rel_db": -10.0, "max_cut_db": 10.0, "ratio": 4.0},
               cons={"target_db": -18.0, "ratio": 4.0, "max_up_db": 0.0, "max_down_db": 15.0, "floor_db": -30.0,
                     "width_db": 6.0})
# the copies made from a take (`adt`): the double left and right of the lead, the harmony's other side
DOUBLE = {"delay": 0.018, "depth": 0.003, "rate": 0.31, "detune_cents": 6.0,
          "dark": [["highshelf", 4500, -5.0, 0.7], ["peak", 3000, -2.0, 1.0]]}
DOUBLE_R = dict(DOUBLE, delay=0.027, rate=0.23, detune_cents=-5.0, phase=1.6)
HARMONY_COPY = {"delay": 0.022, "depth": 0.003, "rate": 0.27, "detune_cents": -7.0, "phase": 0.8, "dark": None}
_BAND = [["hp", 330, 0, 0.7071, 2], ["lp", 5200, 0, 0.7071, 2]]
# What a role gets after its chain, in this order: eq (tone), duck_db (gain reduction while the
# lead sings), slap (one repeat), room (a small mono room, its wet level under the dry).
ROLE_FX: dict[str, dict] = {
    "lead": {},
    "double": {},
    "harmony": {"eq": [["peak", 2800, -3.0, 0.9]], "room": {"decay": 0.28, "predelay": 0.030, "level_db": -13.0}},
    "answer": {"eq": [["highshelf", 6000, -1.5, 0.7]], "room": {"decay": 0.22, "predelay": 0.020, "level_db": -16.0}},
    "adlib": {"eq": _BAND, "duck_db": 2.0, "slap": {"time": 0.095, "level_db": -9.0, "lp": 3200.0}},
    "crowd": {"eq": [["hp", 170, 0, 0.7071, 2], ["highshelf", 3800, -4.0, 0.7]], "duck_db": 3.0,
              "room": {"decay": 0.38, "predelay": 0.038, "level_db": -9.0}},
}
PERIODICITY_BLOCK = 4096  # frames the periodicity detector transforms at a time


def parse_role(role) -> tuple[str, str | None]:
    """``(role, source)`` of a track's role text: ``"lead"``, ``"ad-lib"`` (= ``adlib``),
    ``"double:Harmony"`` (a double of the track named Harmony). ``("", None)`` for no role or
    one that is not in :data:`ROLES`."""
    text = str(role or "").strip()
    name, _, source = text.partition(":")
    name = name.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
    if name not in ROLES:
        return "", None
    return name, (source.strip() or None)


def dial_chain(over, base=None) -> dict:
    """``base`` (default :data:`LEAD`) with the stages of ``over``: an existing stage is
    replaced in place (None = off; a dict is merged into the stage's settings unless it
    carries ``"replace": True``), new stages go in before ``breaths``."""
    base = dict(LEAD if base is None else base)
    over = dict(over or {})
    new = {k: v for k, v in over.items() if k not in base}
    out: dict = {}
    for k, v in base.items():
        if k == "breaths":
            out.update(new)
        if k in over:
            o = over[k]
            if isinstance(o, dict) and isinstance(v, dict) and not o.get("replace"):
                o = dict(v, **o)
            elif isinstance(o, dict):
                o = {a: b for a, b in o.items() if a != "replace"}
            out[k] = o
        else:
            out[k] = v
    if "breaths" not in base:
        out.update(new)
    return {k: ({a: b for a, b in v.items() if a != "replace"} if isinstance(v, dict) else v) for k, v in out.items()}


def lead_chain(dial: str = DEFAULT_DIAL) -> dict:
    """The lead's chain at a dial setting."""
    return dial_chain(DIAL_CHAINS.get(dial, DIAL_CHAINS[DEFAULT_DIAL]))


def role_preset(role: str, dial: str = DEFAULT_DIAL, copy_of: str | None = None, pan: float = 0.0) -> dict:
    """What a role gets: ``chain`` (a config of :func:`process` or None), ``adt`` (settings of
    :func:`adt` or None), the entries of :data:`ROLE_FX` and ``reverb_scale`` (a factor on the
    track's reverb send). ``copy_of`` is the role of the track a double's audio was copied from
    (None: the double sang its own take and gets no ``adt``); a double on the right
    (``pan > 0``) gets the second set of delay and detune values."""
    role = parse_role(role)[0]
    out = {"role": role, "chain": None, "adt": None, "reverb_scale": 1.0, **ROLE_FX.get(role, {})}
    if role in ("lead", "answer"):
        out["chain"] = lead_chain(dial)
        if role == "lead":
            out["reverb_scale"] = REVERB_SCALE.get(dial, 1.0)
    elif role in ("harmony", "double"):
        out["chain"] = dict(HARMONY)
        if role == "double" and copy_of:
            out["adt"] = dict(HARMONY_COPY if copy_of == "harmony" else (DOUBLE_R if pan > 0 else DOUBLE))
    elif role == "adlib":
        out["chain"] = {"eq": LEAD["eq"]}  # static: a gain that moves would make stamped copies differ
    return out


# ---- measures on the 1 ms grid --------------------------------------------------------------------
def _db(p):
    return 10 * np.log10(np.maximum(p, 1e-20))


def level_db(x, sr, window):
    """RMS level (dB) on the 1 ms grid over a centred window (s)."""
    p = mx.block_power(x, mx.grid_hop(sr))
    return _db(mx._moving_mean(p, max(1, int(round(window / mx.GRID_S)))))


def periodicity(x, sr):
    """0..1 on the 1 ms grid: the highest normalised autocorrelation of the signal under
    1.5 kHz at a lag of a sung pitch (70-600 Hz), in 48 ms frames every 4 ms. Vowels and voiced
    consonants are near 1; fricatives, stop bursts and breaths are low."""
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
    win = np.hanning(n)
    wac = np.fft.irfft(np.abs(np.fft.rfft(win, 2 * n)) ** 2)[:n]
    a, b = max(2, int(r / 600)), min(n - 1, int(r / 70))
    e0, rmax = np.empty(k), np.empty(k)
    for j in range(0, k, PERIODICITY_BLOCK):  # frame blocks: the same numbers with bounded memory
        idx = np.arange(n)[None, :] + h * np.arange(j, min(k, j + PERIODICITY_BLOCK))[:, None]
        ac = np.fft.irfft(np.abs(np.fft.rfft(z[idx] * win, 2 * n, axis=1)) ** 2, axis=1)[:, :n]
        e0[j:j + len(ac)] = ac[:, 0]
        rr = ac / (ac[:, :1] + 1e-20) / (wac / wac[0])[None, :]
        rmax[j:j + len(ac)] = np.clip(rr[:, a:b].max(axis=1), 0.0, 1.0)
    rmax[e0 < e0.max() * 1e-7] = 0.0  # silence
    t_fr = (np.arange(k) * h + n / 2) * d / sr
    return np.interp((np.arange(nb) + 0.5) * hop / sr, t_fr, rmax)


def noise_share(x, sr):
    """0..1 on the grid: 1 where the frame is noise (consonant noise, a breath, a gap), 0 where
    it is sung: the periodicity mapped from 0.6 (0) down to 0.4 (1)."""
    return np.clip((0.6 - periodicity(x, sr)) / 0.2, 0.0, 1.0)


def tilt_share(x, sr, sustain: float = 0.03):
    """0..1 on the grid: 1 where the level above 4 kHz is 3 dB over the level under 1.5 kHz, 0
    where it is 3 dB under it, held for ``sustain`` s (a fricative; a stop's burst is shorter)."""
    tilt = (level_db(mx.eq(x, [("hp", 4000.0, 0, 0.7071, 2)], sr), sr, 0.01)
            - level_db(mx.eq(x, [("lp", 1500.0, 0, 0.7071, 2)], sr), sr, 0.01))
    noise = np.clip(tilt / 6.0 + 0.5, 0.0, 1.0)
    w = max(1, int(round(sustain / mx.GRID_S)))
    pad = np.pad(noise, (w // 2, w - w // 2 - 1), mode="edge")
    return sliding_window_view(pad, w).min(axis=1)


def _noise(x, sr, c: dict, default: str):
    """A stage's noise frames by its ``detector`` setting."""
    if c.get("detector", default) == "tilt":
        return tilt_share(x, sr, c.get("sustain", 0.03))
    return noise_share(x, sr)


def sung_level(lvl, noise, floor_db=30.0):
    """``(median level of the sung frames, mask of sung frames)``: frames within ``floor_db``
    of the loud level that are not noise."""
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
    """``x`` with every entry where mask is False replaced by the last entry before it where
    mask is True (-200 dB before the first)."""
    idx = np.where(mask, np.arange(len(x)), -1)
    np.maximum.accumulate(idx, out=idx)
    return np.where(idx >= 0, x[np.maximum(idx, 0)], -200.0)


def sung_env(lvl, sung, window):
    """Level (dB) of the sung frames around every block: their power mean over a centred
    ``window`` (s); where less than half of the window is sung, the higher of the nearest such
    values before and after."""
    w = max(1, int(round(window / mx.GRID_S)))
    num = mx._moving_mean(np.where(sung, 10 ** (lvl / 10), 0.0), w)
    den = mx._moving_mean(sung.astype(np.float64), w)
    have = den > 0.5  # the window's centre lies inside a sung stretch, not on the tail of one
    env = _db(num / np.maximum(den, 1e-9))
    return np.where(have, env, np.maximum(_fill(env, have), _fill(env[::-1], have[::-1])[::-1]))


def _stat(info, name, g_db):
    info[name] = {"max_db": round(float(-g_db.min()), 2), "mean_db": round(float(-g_db.mean()), 2),
                  "active_pct": round(float(100 * np.mean(g_db < -1.0)), 1)}


# ---- the chain ------------------------------------------------------------------------------------
def process(x, sr, cfg=None, return_gain=False):
    """``(y, info)``: mono ``x`` through the stages of ``cfg`` (default :func:`lead_chain`),
    float32, the same length and active level. ``info`` reports every stage's gain curve and
    ``gr_speed``: the fastest change of the summed dynamic gain over sung frames, dB per
    100 ms. With ``return_gain`` the summed dynamic gain (dB on the 1 ms grid) is returned too."""
    cfg = lead_chain() if cfg is None else cfg
    x = np.asarray(x, np.float64)
    n, hop = len(x), mx.grid_hop(sr)
    info: dict = {}
    if not np.any(x):
        return (x.astype(np.float32), info, np.zeros(0)) if return_gain else (x.astype(np.float32), info)
    lvl0 = level_db(x, sr, 0.02)
    rms_in = active_rms(x, lvl0, hop)
    y = x
    total = np.zeros(len(lvl0))  # summed dynamic gain (dB) on the grid
    quiet0 = sung0 = None

    def gain(g_db):
        nonlocal y, total
        total[: len(g_db)] += g_db[: len(total)]
        y = y * mx.grid_to_samples(mx.db_to_gain(g_db), n, hop)

    for name, c in cfg.items():  # the kind of a stage is its name without the digits
        if c is None:
            continue
        kind = name.rstrip("0123456789")
        if kind == "eq":
            y = mx.eq(y, c, sr)
        elif kind == "deess":
            bp = mx.eq(y, [("bandpass", c["freq"], 0, c["q"])], sr)
            g = mx.dynamic_band_db(mx.detector_db(y, sr, "rms", 0.005), mx.detector_db(bp, sr, "rms", 0.005),
                                   c["rel_db"], ratio=c.get("ratio", 3.0), max_cut_db=c["max_cut_db"])
            _stat(info, name, g)
            y = y + (mx.grid_to_samples(mx.db_to_gain(g), n, hop) - 1.0) * bp
        elif kind == "bursts":
            noise = _noise(y, sr, c, "periodicity")
            lvl = level_db(y, sr, 0.01)
            ref, _ = sung_level(level_db(y, sr, 0.02), noise_share(y, sr) if "detector" in c else noise)
            over = np.maximum(lvl - (ref + c["cap_db"]), 0.0) * (1 - 1 / c["ratio"]) * noise
            g = -mx.ar_smooth(np.minimum(over, c["max_cut_db"]), c["attack"] / mx.GRID_S, c["release"] / mx.GRID_S,
                              start=0.0)
            _stat(info, name, g)
            gain(g)
        elif kind == "rider":
            noise = noise_share(y, sr)
            lvl = level_db(y, sr, 0.02)
            ref, sung = sung_level(lvl, noise)
            # the sung level around each block (power mean over the window, sung frames only; in a gap the
            # higher of the levels before and after it, so a breath is not lifted like the tail of a note)
            want = c.get("strength", 1.0) * (ref - sung_env(lvl, sung, c["window"]))
            g = mx._moving_mean(np.clip(want, -c["range_db"], c["range_db"]),
                                max(1, int(round(c["smooth"] / mx.GRID_S))))
            info[name] = {"max_up_db": round(float(g.max()), 2), "max_down_db": round(float(-g.min()), 2),
                          "sd_db": round(float(g[sung].std()), 2)}
            gain(g)
        elif kind == "comp":
            lvl = mx.detector_db(y, sr, "rms", c.get("window", 0.01))
            thr = mx.relative_threshold(lvl, c["offset_db"])
            det = lvl
            if c.get("follow_sung", True):
                # on noise frames the detector reads the sung level of the surrounding 0.3 s (the nearest sung
                # stretch on either side when there is none that close)
                _, sung = sung_level(level_db(y, sr, 0.02), noise_share(y, sr))
                det = np.where(sung, lvl, sung_env(lvl, sung, c.get("follow_window", 0.3)))
            g = mx.compressor_db(det, thr, c["ratio"], c["attack"], c["release"], c.get("knee_db", 6.0),
                                 max_gr_db=c.get("max_gr_db"))
            _stat(info, name, g)
            gain(g)
        elif kind == "cons":
            # noise frames in the consonants' level range move toward `target_db` under the sung level; the
            # breaths, under `floor_db`, are left alone
            sung_noise = noise_share(y, sr)
            noise = _noise(y, sr, c, "periodicity") if "detector" in c else sung_noise
            lvl = level_db(y, sr, 0.015)
            ref, _ = sung_level(level_db(y, sr, 0.02), sung_noise)
            rel = lvl - ref
            g = np.clip((c["target_db"] - rel) * (1 - 1 / c.get("ratio", 2.0)), -c.get("max_down_db", 9.0),
                        c.get("max_up_db", 4.0))
            g = g * noise * np.clip((rel - c["floor_db"]) / c.get("width_db", 6.0), 0.0, 1.0)
            g = mx._moving_mean(g, 5)
            info[name] = {"max_up_db": round(float(g.max()), 2), "max_down_db": round(float(-g.min()), 2),
                          "up_pct": round(float(100 * np.mean(g > 1.0)), 1),
                          "down_pct": round(float(100 * np.mean(g < -1.0)), 1)}
            gain(g)
        elif kind == "breaths":
            # the stages before give quiet noise in the gaps the gain of the sung frames around it; its summed
            # gain is held at `max_up_db` over the median gain of the sung frames. "Quiet noise" = noise frames
            # whose level in the input is more than `under_db` under the input's sung level
            if quiet0 is None:
                noise0 = noise_share(x, sr)
                ref0, sung0 = sung_level(lvl0, noise0)
                quiet0 = noise0 * np.clip(((ref0 + c["under_db"]) - level_db(x, sr, 0.03)) / c.get("width_db", 4.0),
                                          0.0, 1.0)
            med = float(np.median(total[sung0])) if sung0.any() else 0.0
            g = -np.maximum(total - (med + c["max_up_db"]), 0.0) * quiet0
            g = mx._moving_mean(g, 9)
            info[name] = {"max_db": round(float(-g.min()), 2), "active_pct": round(float(100 * np.mean(g < -1.0)), 1)}
            gain(g)
        elif kind == "expand":
            noise = noise_share(y, sr)
            lvl = level_db(y, sr, 0.03)
            ref, _ = sung_level(level_db(y, sr, 0.02), noise)
            under = np.maximum((ref + c["offset_db"]) - lvl, 0.0) * (c["ratio"] - 1.0)
            g = -mx.ar_smooth(np.minimum(under, c["max_db"]), c.get("attack", 0.03) / mx.GRID_S,
                              c.get("release", 0.004) / mx.GRID_S, start=0.0)
            info[name] = {"max_db": round(float(-g.min()), 2), "active_pct": round(float(100 * np.mean(g < -1.0)), 1)}
            gain(g)
        elif kind == "lift":
            # noise frames whose level is over `floor_db` re the sung level (so not the breaths) get up to `db`
            # of gain, no further than `ceiling_db` re the sung level; with `hf` only the band above it
            lvl = level_db(y, sr, 0.015)
            ref, _ = sung_level(level_db(y, sr, 0.02), noise_share(y, sr))
            noise = _noise(y, sr, c, "tilt")
            rel = lvl - ref
            up = np.full(len(rel), float(c["db"]))
            if c.get("ceiling_db") is not None:
                up = np.minimum(up, np.maximum(c["ceiling_db"] - rel, 0.0))
            g = up * noise * np.clip((rel - c.get("floor_db", -30.0)) / c.get("width_db", 6.0), 0.0, 1.0)
            g = mx._moving_mean(g, 5)
            info[name] = {"max_db": round(float(g.max()), 2), "active_pct": round(float(100 * np.mean(g > 1.0)), 1)}
            if c.get("hf"):
                part = mx.eq(y, [("hp", float(c["hf"]), 0, 0.7071, 2)], sr)
                y = y + (mx.grid_to_samples(mx.db_to_gain(g), n, hop) - 1.0) * part
            else:
                gain(g)
        elif kind == "trans":
            # where the level of the band above `lo` Hz rises faster than its slow envelope (a stop's release,
            # the start of a sibilant), that band gets `amount` dB per dB of the difference, up to `max_db`
            part = mx.eq(y, [("hp", float(c.get("lo", 2000.0)), 0, 0.7071, 2)], sr)
            fast = _db(mx._moving_mean(mx.block_power(part, hop), max(1, int(round(c.get("fast", 0.005) / mx.GRID_S)))))
            w = max(1, int(round(c.get("slow", 0.05) / mx.GRID_S)))
            slow = _db(mx._moving_mean(mx.block_power(part, hop), w))
            slow = np.concatenate([np.full(w // 2, slow[0]), slow])[: len(slow)]  # the slow envelope looks back
            lvl = level_db(y, sr, 0.02)
            act = np.clip((fast - (np.percentile(lvl, 95) - c.get("gate_db", 45.0))) / 6.0, 0.0, 1.0)
            g = np.clip((fast - slow) * c.get("amount", 0.5), 0.0, c.get("max_db", 6.0)) * act
            g = mx.ar_smooth(g, 0.001 / mx.GRID_S, c.get("release", 0.02) / mx.GRID_S, start=0.0)
            info[name] = {"max_db": round(float(g.max()), 2), "active_pct": round(float(100 * np.mean(g > 1.0)), 1)}
            y = y + (mx.grid_to_samples(mx.db_to_gain(g), n, hop) - 1.0) * part
        elif kind == "crest":
            lvl = _db(mx._moving_mean(mx.block_power(y, hop), 10))
            ceil_db = mx.relative_threshold(lvl, float(c))
            gl = mx.limiter_gain(np.abs(y), sr, 10 ** (ceil_db / 20.0), lookahead=0.001, release=0.03,
                                 release_slow=0.0)
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
    _, sung = sung_level(lvl1, noise_share(y, sr))
    dg = np.abs(total[100:] - total[:-100])[sung[100:] & sung[:-100]]
    info["gr_speed"] = {"p50": round(float(np.percentile(dg, 50)), 2), "p95": round(float(np.percentile(dg, 95)), 2),
                        "max": round(float(dg.max()), 2)} if len(dg) else {}
    info["gain_range_db"] = [round(float(np.percentile(total[sung], q)), 2) for q in (1, 50, 99)] if sung.any() else []
    if return_gain:
        return y.astype(np.float32), info, total
    return y.astype(np.float32), info


# ---- copies, repeats and rooms --------------------------------------------------------------------
def frac_delay(x, delay_samples):
    """``x`` delayed by a time-varying number of samples (array, >= 0), linear interpolation."""
    pos = np.arange(len(x)) - delay_samples
    i = np.floor(pos).astype(np.int64)
    f = pos - i
    a = np.where((i >= 0) & (i < len(x)), x[np.clip(i, 0, len(x) - 1)], 0.0)
    b = np.where((i + 1 >= 0) & (i + 1 < len(x)), x[np.clip(i + 1, 0, len(x) - 1)], 0.0)
    return a * (1 - f) + b * f


def adt(x, sr, delay=0.018, depth=0.004, rate=0.31, detune_cents=6.0, dark=None, phase=0.0):
    """A double made from the take itself: delayed by ``delay`` +- ``depth`` s (a slow sine at
    ``rate`` Hz), detuned by a constant ``detune_cents`` (a delay that drifts at a constant
    rate and snaps back in every silence longer than 80 ms, so that it stays within 6 ms of
    ``delay``), with the EQ ``dark``. Two voices from one take: no second take's timing to
    blur the consonants."""
    x = np.asarray(x, np.float64)
    t = np.arange(len(x)) / sr
    d = delay + depth * np.sin(2 * np.pi * rate * t + phase)
    if detune_cents and np.any(x):
        rate_s = 2 ** (detune_cents / 1200.0) - 1.0  # s of delay per s; positive detune = delay shrinking
        lvl = level_db(x, sr, 0.02)
        quiet = lvl < np.percentile(lvl, 95) - 45
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
        d = d + mx.grid_to_samples(drift, len(x), mx.grid_hop(sr))
    y = frac_delay(x, np.maximum(d, 0.0) * sr)
    if dark:
        y = mx.eq(y, dark, sr)
    return y.astype(np.float32)


def slap(x, sr, time=0.1, level_db=-10.0, lp=3000.0):
    """``x`` plus one band-limited repeat ``time`` s later, ``level_db`` under it."""
    k = int(round(time * sr))
    y = np.zeros(len(x), np.float64)
    y[k:] = x[: len(x) - k]
    y = mx.eq(y, [["hp", 300, 0, 0.7071, 2], ["lp", lp, 0, 0.7071, 2]], sr)
    return (x + y * 10 ** (level_db / 20)).astype(np.float32)


def room_ir(sr, decay=0.18, predelay=0.012, seed=7, lo=300.0, hi=6000.0):
    """Mono impulse response of a small room: band-limited noise with an exponential decay
    (``decay`` = RT60, s) after ``predelay``, unit energy."""
    n = int(sr * (predelay + decay * 1.3))
    rng = np.random.default_rng(seed)
    t = np.arange(n) / sr - predelay
    h = rng.standard_normal(n) * np.where(t >= 0, np.exp(-6.91 * np.maximum(t, 0) / decay), 0.0)
    h = mx.eq(h, [("hp", lo, 0, 0.7071, 2), ("lp", hi, 0, 0.7071, 2)], sr)
    return h / (np.sqrt(np.sum(h ** 2)) + 1e-12)


def room(x, sr, decay=0.18, predelay=0.012, level_db=-12.0, seed=7):
    """The wet signal of the small room for the mono channel ``x`` (same length),
    ``level_db`` under the dry level."""
    from .effects import fft_convolve

    h = room_ir(sr, decay, predelay, seed)
    return (fft_convolve(np.asarray(x, np.float64), h)[: len(x)] * 10 ** (level_db / 20)).astype(np.float32)


def sing_env(x, sr):
    """0..1 per sample: where a take sings (within 25 dB of its loud level; 60 ms hold, 30 ms
    attack, 180 ms release). What the other roles duck under."""
    n = len(x)
    hop = max(1, int(round(0.01 * sr)))
    nb = max(1, n // hop)
    if n < hop or not np.any(x):
        return np.zeros(n, np.float32)
    e = np.sqrt(np.mean(np.asarray(x[: nb * hop], np.float64).reshape(nb, hop) ** 2, axis=1) + 1e-12)
    lv = 20 * np.log10(e)
    on = np.zeros(nb)
    if np.any(lv > -80):
        on = (lv > np.percentile(lv[lv > -80], 90) - 25.0).astype(np.float64)
    hold = 6
    on = np.max(np.pad(on, (hold, 0))[np.arange(hold + 1)[None, :] + np.arange(nb)[:, None]], axis=1)
    g, v = np.zeros(nb), 0.0
    for i in range(nb):
        v += (on[i] - v) * (1 - np.exp(-0.01 / (0.03 if on[i] > v else 0.18)))
        g[i] = v
    return np.interp(np.arange(n), np.arange(nb) * hop + hop / 2, g).astype(np.float32)


def apply_fx(y, sr, fx, lead_env=None) -> tuple[np.ndarray, dict]:
    """``(y, info)``: mono ``y`` with the in-channel effects ``fx``, in this order: ``eq`` (tone),
    ``duck_db`` (gain reduction where ``lead_env`` says the lead sings; no duck without it),
    ``slap`` (one repeat, settings of :func:`slap`), ``room`` (a small mono room, settings of
    :func:`room`). An entry that is missing or None is skipped."""
    info: dict = {}
    y = np.asarray(y, np.float32)
    if not fx or not np.any(y):
        return y, info
    if fx.get("eq"):
        y = mx.eq(y.astype(np.float64), fx["eq"], sr).astype(np.float32)
    if fx.get("duck_db") and lead_env is not None and len(lead_env):
        env = np.zeros(len(y), np.float32)
        m = min(len(y), len(lead_env))
        env[:m] = lead_env[:m]
        y = y * (10 ** (-fx["duck_db"] * env / 20)).astype(np.float32)
        info["duck_db"] = fx["duck_db"]
    if fx.get("slap"):
        y = slap(y, sr, **fx["slap"])
    if fx.get("room"):
        y = y + room(y, sr, **fx["room"])
    return np.asarray(y, np.float32), info


def apply_role(x, sr, role, dial: str = DEFAULT_DIAL, lead_env=None, copy_of: str | None = None,
               pan: float = 0.0, fx: dict | None = None) -> tuple[np.ndarray, dict]:
    """``(y, info)``: mono ``x`` with what its role gets (:func:`role_preset`): the chain, the
    double's delay and detune, then the in-channel effects (:func:`apply_fx`: tone, the duck
    under the lead, a repeat, a room). ``fx`` replaces single entries of the role's effects for
    this track (``{"slap": None}``: no repeat). ``lead_env`` is :func:`sing_env` of the lead's
    takes. ``info["reverb_scale"]`` is the factor the role puts on the track's reverb send."""
    pre = role_preset(role, dial, copy_of, pan)
    y = np.asarray(x, np.float32)
    info: dict = {"role": pre["role"], "reverb_scale": pre["reverb_scale"]}
    if not pre["role"] or not np.any(y):
        return y, info
    if pre["role"] in ("lead", "answer"):
        info["dial"] = dial if dial in DIAL_CHAINS else DEFAULT_DIAL
    if pre["chain"]:
        y, info["chain"] = process(y, sr, pre["chain"])
    if pre["adt"]:
        y = adt(y, sr, **pre["adt"])
        info["adt"] = {k: v for k, v in pre["adt"].items() if k != "dark"}
    y, fx_info = apply_fx(y, sr, {**pre, **(fx or {})}, lead_env)
    info.update(fx_info)
    return y, info


# ---- which sources of a mix have a role -----------------------------------------------------------
ROLE_ENGINES = ("neural",)  # engines whose takes get a role's chain; any other engine keeps the channel strip


def source_role(source) -> dict | None:
    """``{"role", "dial", "copy_of"}`` of a vocal source of the mix that gets its role's chain:
    one with a role (``vocal_role``) that an engine of :data:`ROLE_ENGINES` sang. None for any
    other source."""
    role = parse_role(getattr(source, "vocal_role", ""))[0]
    if not role or getattr(source, "engine", "") not in ROLE_ENGINES:
        return None
    dial = getattr(source, "dial", "")
    return {"role": role, "dial": dial if dial in DIAL_CHAINS else DEFAULT_DIAL,
            "copy_of": getattr(source, "copy_of", None) or None}


def leads_env(leads, sr: int, n: int) -> np.ndarray | None:
    """Where the lead sings over ``n`` samples: :func:`sing_env` of the sum of ``leads``, an
    iterable of ``(mono audio, gain_db)``. None without a lead. What the roles and tracks with
    a ``duck_db`` duck under."""
    total = None
    for audio, gain_db in leads:
        if total is None:
            total = np.zeros(n)
        audio = np.asarray(audio)
        m = min(n, len(audio))
        total[:m] += audio[:m].astype(np.float64) * 10 ** (gain_db / 20.0)
    return None if total is None else sing_env(total, sr)
