"""From the planner's timed units to what the models are fed: phrases, word groups, frames.

Pure functions (numpy only, no model access): the duration model's prediction is passed in.
The alignment follows OpenUtau's DiffSinger phonemizer: every word group starts at a vowel
(or a rest) on its note onset, the consonants before a vowel trail the previous group and
share that group's time in the proportions the duration model predicts.

A phrase's frames are laid out as ``[head SP][phonemes ...][tail SP]`` on one global frame
grid (frame ``k`` is at ``k * frame`` seconds), so phrases can be rendered independently and
placed by their first frame.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

HEAD = 8  # frames of SP before a phrase
TAIL = 10  # and after it
PHRASE_GAP = 0.30  # s of silence that splits phrases
SP_GAP = 0.045  # s: shorter gaps inside a phrase are sung through (legato)
PAD = 0.5  # s given to the duration model before the first note
MAX_PHRASE = 20.0  # s: a longer phrase is split at its longest internal gap
MIN_PHONE = 0.5  # frames: the least a predicted phoneme length counts for when a group is shared out

SLOW_NOTE = 0.25  # s: notes shorter than this are sung in slow motion
SLOW_MAX = 2.0  # largest slow-motion factor
SLOW_MIN = 1.02  # a phrase whose largest factor is below this is rendered at its real speed
VOICING_FLOOR = 6.0  # dB under the running median a vowel's voicing may fall
VOICING_WINDOW = 1.5  # s of vowel time the floor's running median covers
MIN_VOWEL_SHARE = 0.5  # of its note, the most ``min_vowel`` gives a vowel

JA_VOWELS = {"a", "i", "u", "e", "o", "N"}
REST_SYMBOLS = ("SP", "AP")


def build_phrases(plan) -> list[list]:
    """Split ``plan.units`` into phrases (lists of units) at silences of ``PHRASE_GAP`` or more."""
    phrases, cur, prev_end = [], None, None
    for u in plan.units:
        st = u.start - (u.lead or 0.0)
        if cur is None or st - prev_end >= PHRASE_GAP:
            cur = []
            phrases.append(cur)
        cur.append(u)
        prev_end = u.end
    return phrases


def split_long(phrases: list[list], max_len: float = MAX_PHRASE) -> tuple[list[list], int]:
    """Phrases longer than ``max_len`` seconds cut at their longest internal gap (the one
    nearest the middle among equals), repeatedly. Returns ``(phrases, number of cuts)``."""
    out, cuts = [], 0
    todo = list(reversed(phrases))
    while todo:
        ph = todo.pop()
        if len(ph) < 2 or ph[-1].end - ph[0].start <= max_len:
            out.append(ph)
            continue
        mid = 0.5 * (ph[0].start + ph[-1].end)
        k = max(range(1, len(ph)), key=lambda i: (round(ph[i].start - (ph[i].lead or 0.0) - ph[i - 1].end, 4),
                                                  -abs(ph[i].start - mid)))
        cuts += 1
        todo.append(ph[k:])
        todo.append(ph[:k])
    return out, cuts


def phrase_groups(units, plan, convert) -> tuple[list[dict], float]:
    """OpenUtau-style word groups for one phrase.

    Each group starts at a vowel, ``SP`` or ``AP``; consonants trail the previous starter.
    ``convert(phoneme, ja, next_phoneme)`` gives the bank's symbol for a Synth2 phoneme.
    Returns ``(groups, end)``: a group is ``dict(pos, phones, midi)`` with ``vowel=True`` on
    the groups a nucleus starts; the first group is the leading ``SP``.
    """
    by_index = {tm.index: tm.pitch for tm in plan.timings}

    def midi(u):
        # u.notes holds track note indices; plan.timings skips breaths and closures, so the note
        # is looked up by index (a breath takes the pitch of the nearest pitched note)
        if not u.notes or not by_index:
            return 60
        k = min(by_index, key=lambda i: (min(abs(i - j) for j in u.notes), i))
        return int(round(by_index[k]))

    first = units[0]
    groups = [dict(pos=None, phones=["SP"], midi=midi(first))]
    for k, u in enumerate(units):
        ja = any(n[-1] in JA_VOWELS for n in u.nuclei) if u.nuclei else False
        if k > 0 and u.start - units[k - 1].end >= SP_GAP:
            groups.append(dict(pos=units[k - 1].end, phones=["SP"], midi=midi(u)))
        if u.breath:
            groups.append(dict(pos=u.start, phones=["AP"], midi=midi(u)))
            continue
        if not u.nuclei:
            if u.closure:
                groups.append(dict(pos=u.start, phones=["SP"], midi=midi(u)))
            else:
                ph = list(u.phones)
                groups[-1]["phones"] += [convert(p, ja, ph[i + 1] if i + 1 < len(ph) else None)
                                         for i, p in enumerate(ph)]
            continue
        flat = list(u.onset) + [p for nuc in u.nuclei for p in nuc] + list(u.coda)
        sym = [convert(p, ja, flat[i + 1] if i + 1 < len(flat) else None) for i, p in enumerate(flat)]
        at = len(u.onset)
        groups[-1]["phones"] += sym[:at]
        span = u.end - u.start
        for j, nuc in enumerate(u.nuclei):
            groups[-1]["phones"] += sym[at:at + len(nuc) - 1]
            at += len(nuc)
            groups.append(dict(pos=u.start + span * j / len(u.nuclei), phones=[sym[at - 1]], midi=midi(u),
                               vowel=True))
        groups[-1]["phones"] += sym[at:]
    if len(groups) < 2:
        # nothing but consonants: sing them from the unit's start
        groups.append(dict(pos=first.start, phones=["SP"], midi=midi(first)))
    groups[0]["pos"] = groups[1]["pos"] - PAD
    return groups, units[-1].end


def dur_inputs(groups, end: float, frame: float) -> tuple[list[str], list[int], list[int], list[int]]:
    """``(phones, word_div, word_dur, ph_midi)`` for the duration model."""
    phones = [p for g in groups for p in g["phones"]]
    word_div = [len(g["phones"]) for g in groups]
    pos = [g["pos"] for g in groups] + [end]
    fr = [int(round(p / frame)) for p in pos]
    word_dur = [max(1, b - a) for a, b in zip(fr, fr[1:])]
    ph_midi = [g["midi"] for g in groups for _ in g["phones"]]
    return phones, word_div, word_dur, ph_midi


def align(groups, end: float, pred, frame: float, min_vowel: float | None = None) -> list[float]:
    """OpenUtau alignment: the start time (s) of every phoneme after the leading ``SP``.

    ``pred`` is the duration model's length (frames) of every phoneme. ``min_vowel`` (s): a
    vowel the predicted lengths leave shorter than this takes the difference from the
    consonants that share its note, up to ``MIN_VOWEL_SHARE`` of the note.
    """
    pred = np.asarray(pred, np.float64)
    counts = [len(g["phones"]) for g in groups]
    idx = np.concatenate([[0], np.cumsum(counts)])
    pos = [g["pos"] for g in groups] + [end]
    starts = []
    # first group: SP + onset consonants keep their predicted length, ending at the first vowel
    cons = pred[1:idx[1]] * frame
    lim = max(0.0, pos[1] - pos[0] - 2 * frame)
    if cons.sum() > lim and cons.sum() > 0:
        cons = cons * lim / cons.sum()
    c = pos[1] - cons.sum()
    for d in cons:
        starts.append(c)
        c += d
    for g in range(1, len(groups)):
        seg = np.maximum(pred[idx[g]:idx[g + 1]], MIN_PHONE)
        span = pos[g + 1] - pos[g]
        seg = seg * span / seg.sum()
        if min_vowel and groups[g].get("vowel") and len(seg) > 1:
            want = min(float(min_vowel), MIN_VOWEL_SHARE * span)
            if seg[0] < want:
                seg[1:] *= (span - want) / seg[1:].sum()
                seg[0] = want
        c = pos[g]
        for d in seg:
            starts.append(c)
            c += d
    return starts


def phrase_frames(groups, end: float, pred, frame: float, min_vowel: float | None = None
                  ) -> tuple[list[str], list[int], int, list[bool]]:
    """The phrase on the global frame grid: ``(seq, durs, f_start, is_vowel)``.

    ``seq`` is ``SP``, the phonemes, ``SP``; ``durs`` their lengths in frames (every phoneme
    at least one); ``f_start`` the global index of the first frame; ``is_vowel`` flags the
    phonemes that start a note.
    """
    phones = [p for g in groups for p in g["phones"]]
    starts = align(groups, end, pred, frame, min_vowel)
    b = [int(round(s / frame)) for s in starts] + [int(round(end / frame))]
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1] + 1)
    durs = [HEAD] + [y - x for x, y in zip(b, b[1:])] + [TAIL]
    seq = ["SP"] + phones[1:] + ["SP"]
    is_vowel = [False] + [j == 0 and bool(g.get("vowel", False)) and p not in REST_SYMBOLS
                          for g in groups for j, p in enumerate(g["phones"])][1:] + [False]
    return seq, durs, b[0] - HEAD, is_vowel


def phrase_bounds(units, frame: float) -> tuple[float, float]:
    """Seconds outside which a phrase has no frames, whatever the duration model predicts.

    The head starts at most ``PAD`` and ``HEAD`` frames before the first unit; the end moves
    past the last unit by at most one frame per phoneme (each phoneme gets at least a frame).
    """
    phones = sum(len(u.phones) + 2 for u in units)
    return units[0].start - PAD - (HEAD + 1) * frame, units[-1].end + (TAIL + 1 + phones) * frame


# ---- slow-motion rendering ------------------------------------------------------------------------
def slow_factors(groups, end: float, slow_note: float = SLOW_NOTE, slow_max: float = SLOW_MAX) -> np.ndarray:
    """Per phoneme (leading ``SP`` included): how many times slower it is sung than written.

    A note shorter than ``slow_note`` is stretched to ``slow_note`` (at most ``slow_max``
    times); silence is never stretched and the consonants before a note follow that note.
    """
    pos = [g["pos"] for g in groups] + [end]
    kg = [float(np.clip(slow_note / max(b - a, 1e-3), 1.0, slow_max)) for a, b in zip(pos, pos[1:])]
    k = []
    for g, grp in enumerate(groups):
        rest = grp["phones"][0] in REST_SYMBOLS
        own = kg[min(g + 1, len(groups) - 1)] if rest else kg[g]
        k += [1.0 if p in REST_SYMBOLS else own for p in grp["phones"]]
    return np.array(k)


def slow_grid(durs, k) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Phoneme lengths (frames) stretched by ``k``, and the frame positions that map each grid
    onto the other: ``(slow durs, to_true, to_slow)``.

    Phoneme edges of the two grids coincide; inside a phoneme the map is linear.
    """
    edges = np.concatenate([[0.0], np.cumsum(durs)])
    slow = np.maximum(np.diff(np.round(np.concatenate([[0.0], np.cumsum(np.asarray(durs) * k)]))), 1).astype(int)
    edges2 = np.concatenate([[0.0], np.cumsum(slow)])
    to_true = np.interp(np.arange(int(edges2[-1])) + 0.5, edges2, edges) - 0.5
    to_slow = np.interp(np.arange(int(edges[-1])) + 0.5, edges, edges2) - 0.5
    return [int(d) for d in slow], to_true, to_slow


def warp_rows(x, at) -> np.ndarray:
    """Rows of ``x`` (frames, bins) at fractional frame positions, linearly interpolated."""
    x = np.asarray(x)
    if len(x) < 2:
        return np.repeat(x[:1], len(at), axis=0).astype(np.float32)
    at = np.clip(at, 0, len(x) - 1)
    i = np.minimum(at.astype(int), len(x) - 2)
    f = (at - i)[:, None]
    return (x[i] * (1 - f) + x[i + 1] * f).astype(np.float32)


# ---- vowel voicing floor --------------------------------------------------------------------------
def _running_median(x, win: int) -> np.ndarray:
    """Median of the ``win`` samples centred on each sample (reflected at the ends)."""
    n = len(x)
    if n <= win:
        return np.full(n, np.median(x))
    h = win // 2
    xp = np.concatenate([x[h:0:-1], x, x[-2:-h - 2:-1]])
    return np.median(xp[np.arange(win)[None, :] + np.arange(n)[:, None]], axis=1)


def floor_voicing(voicing, seq, durs, is_vowel, frame: float, floor_db: float | None = VOICING_FLOOR,
                  window: float = VOICING_WINDOW) -> np.ndarray:
    """Lift the vowel frames whose predicted voicing collapsed.

    In fast lines a variance model predicts some short vowels 10 to 60 dB under the vowels
    around them and the acoustic model then whispers the syllable. Frames of vowels
    (``is_vowel``, one flag per phoneme of ``seq``) are held at or above the running median of
    the phrase's vowel voicing minus ``floor_db``. A vowel that dies away into a rest keeps
    its predicted decay (its trailing low frames are left alone, unless the whole vowel is
    low). Consonants and rests keep the prediction. ``floor_db=None`` returns it unchanged.
    """
    if floor_db is None:
        return voicing
    idx = np.concatenate([[0], np.cumsum(durs)])
    vmask = np.repeat(np.asarray(is_vowel, bool), durs)
    if vmask.sum() < 4:
        return voicing
    out = voicing.copy()
    floor = np.full(len(out), -np.inf)
    floor[vmask] = _running_median(out[vmask], int(round(window / frame))) - floor_db
    for i in np.flatnonzero(is_vowel):
        a, b = idx[i], idx[i + 1]
        low = out[a:b] < floor[a:b]
        if not low.any():
            continue
        if i + 1 < len(seq) and seq[i + 1] in REST_SYMBOLS and not low.all():
            k = b - a
            while low[k - 1]:
                k -= 1
            low[k:] = False
        out[a:b] = np.where(low, floor[a:b], out[a:b])
    return out


# ---- noise ----------------------------------------------------------------------------------------
def phrase_seed(seed: int, f_start: int, n: int) -> int:
    """31-bit seed from the track seed and the phrase's first frame and length, not from its
    notes or lyrics: with the noise unchanged, editing a note leaves the rest of the phrase's
    take in place."""
    key = [int(seed), int(f_start), int(n)]
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big") & 0x7FFFFFFF
