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

# Consonant classes by phoneme name without the language prefix (English and Japanese sets).
CONS_CLASSES = ("stop", "vstop", "aff", "fric", "vfric", "nasal", "glide")
CONS_CLASS: dict[str, str] = {}
for _k, _v in {"stop": "p t k ky kw py ty cl", "vstop": "b d g dx jh by dy gy gw", "aff": "ch ts",
               "fric": "s sh f th hh h hy", "vfric": "z zh v dh j", "nasal": "m n ng my ny",
               "glide": "l r w y _r ry"}.items():
    for _p in _v.split():
        CONS_CLASS[_p] = _k
VOICED_CONS = ("nasal", "glide", "vstop", "vfric")  # the classes a voicing floor applies to
BURST_RISE = 0.6  # of a stop's rise in level: where its release is taken to start
SLOW_CONS_MAX = 4.0  # largest slow-motion factor a consonant gets from a minimum sung length

BREATH_FALL = 0.09  # s over which a breath's noise dies away before the consonant after it
BREATH_RELEASE = 0.1  # s left to the phrase before, ahead of a breath at the start of a phrase


def cons_class(p: str) -> str | None:
    """Class of a bank phoneme (``"en/s"`` -> ``"fric"``); None for vowels, ``SP`` and ``AP``."""
    if p in REST_SYMBOLS:
        return None
    return CONS_CLASS.get(p.split("/")[-1])


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


def phrase_groups(units, plan, convert, frame: float | None = None, breath: dict | None = None,
                  prev_end: float | None = None) -> tuple[list[dict], float]:
    """OpenUtau-style word groups for one phrase.

    Each group starts at a vowel, ``SP`` or ``AP``; consonants trail the previous starter.
    ``convert(phoneme, ja, next_phoneme)`` gives the bank's symbol for a Synth2 phoneme.
    Returns ``(groups, end)``: a group is ``dict(pos, phones, midi)`` with ``vowel=True`` on
    the groups a nucleus starts; the first group is the leading ``SP``. With ``frame`` (s) a
    group never starts less than a frame after the group before it.

    ``breath``: None, or ``dict(min_gap, fill, length, first, min_len, breathiness)``. A gap of
    at least ``min_gap`` s between two notes gets a breath (``AP``) that ends where the next
    note's consonants start: the whole gap up to ``fill`` s, the last ``length`` s of a longer
    one. The phrase itself starts with a breath of ``length`` s when the silence before it
    (``prev_end``: the end of the phrase before; None: the start of the track, breathed when
    ``first``) has room for it. ``breath["avoid"]`` lists ``(start, end)`` spans in seconds
    where no breath is put: a breath that would overlap one of them stays a rest.
    """
    by_index = {tm.index: tm.pitch for tm in plan.timings}

    def midi(u):
        # u.notes holds track note indices; plan.timings skips breaths and closures, so the note
        # is looked up by index (a breath takes the pitch of the nearest pitched note)
        if not u.notes or not by_index:
            return 60
        k = min(by_index, key=lambda i: (min(abs(i - j) for j in u.notes), i))
        return int(round(by_index[k]))

    def clear(a, b):
        return not any(a < y and x < b for x, y in (breath or {}).get("avoid", ()))

    first = units[0]
    groups = [dict(pos=None, phones=["SP"], midi=midi(first))]
    if breath and not first.breath:
        on = first.start - (first.lead or 0.0)
        room = 1e9 if prev_end is None else on - prev_end - breath.get("release", BREATH_RELEASE)
        if (prev_end is not None or breath.get("first", True)) and room >= breath["min_gap"] \
                and clear(on - min(breath["length"], room), on):
            groups.append(dict(pos=on - min(breath["length"], room), phones=["AP"], midi=midi(first)))
    for k, u in enumerate(units):
        ja = any(n[-1] in JA_VOWELS for n in u.nuclei) if u.nuclei else False
        gap = u.start - units[k - 1].end if k > 0 else 0.0
        if k > 0 and gap >= SP_GAP:
            on = u.start - (u.lead or 0.0)
            if breath and not u.breath and not units[k - 1].breath and gap >= breath["min_gap"] \
                    and clear(max(units[k - 1].end, on - breath["length"]), on):
                if gap > breath["fill"]:
                    groups.append(dict(pos=units[k - 1].end, phones=["SP"], midi=midi(u)))
                    groups.append(dict(pos=on - breath["length"], phones=["AP"], midi=midi(u)))
                else:
                    groups.append(dict(pos=units[k - 1].end, phones=["AP"], midi=midi(u)))
            else:
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
    if frame:
        for a, b in zip(groups, groups[1:]):  # a breath placed from the end never starts before the group before it
            b["pos"] = max(b["pos"], a["pos"] + frame)
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


def align(groups, end: float, pred, frame: float, min_vowel: float | None = None, cons_cap: dict | None = None,
          cons_cap_keep: float = 0.0, cons_cap_share: float | None = None, onset_keep: bool = False) -> list[float]:
    """OpenUtau alignment: the start time (s) of every phoneme after the leading ``SP``.

    ``pred`` is the duration model's length (frames) of every phoneme. ``min_vowel`` (s): a
    vowel the predicted lengths leave shorter than this takes the difference from the
    consonants that share its note, up to ``MIN_VOWEL_SHARE`` of the note.

    ``cons_cap`` (``{class: s}``): no consonant that shares a note with a vowel is longer than
    its class's cap; the vowel takes the rest. ``cons_cap_keep``: no cap takes a consonant
    under this share of the length the duration model gave it. ``cons_cap_share``: the caps
    bind only where the consonants of a note take more than this share of the note, and then
    only as far as it takes to bring them down to that share. ``onset_keep``: the consonants
    after a rest or a breath keep their predicted length (as those at the start of the phrase
    do) instead of sharing the rest's time in proportion.
    """
    pred = np.asarray(pred, np.float64)
    classes = [cons_class(p) for g in groups for p in g["phones"]]
    cap = np.array([(cons_cap or {}).get(c, np.inf) if c is not None else np.inf for c in classes])
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
        if onset_keep and len(seg) > 1 and groups[g]["phones"][0] in REST_SYMBOLS:
            c = seg[1:] * frame
            room = max(span - 2 * frame, 0.5 * span)
            if c.sum() > room:
                c *= room / c.sum()
            seg = np.concatenate([[span - c.sum()], c])
        else:
            seg = seg * span / seg.sum()
        if groups[g].get("vowel") and len(seg) > 1:
            over = np.maximum(seg[1:] - np.maximum(cap[idx[g] + 1:idx[g + 1]], cons_cap_keep * seg[1:]), 0.0)
            if cons_cap_share is not None and over.any():
                # s the consonants are over their share of the note; the caps take no more than that
                over *= min(1.0, max(0.0, seg[1:].sum() - cons_cap_share * span) / over.sum())
            if over.any():
                seg[1:] -= over
                seg[0] += over.sum()
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


def phrase_frames(groups, end: float, pred, frame: float, min_vowel: float | None = None, min_breath: float = 0.0,
                  **caps) -> tuple[list[str], list[int], int, list[bool]]:
    """The phrase on the global frame grid: ``(seq, durs, f_start, is_vowel)``.

    ``seq`` is ``SP``, the phonemes, ``SP``; ``durs`` their lengths in frames (every phoneme
    at least one); ``f_start`` the global index of the first frame; ``is_vowel`` flags the
    phonemes that start a note. A breath left shorter than ``min_breath`` s is a rest.
    ``caps`` are the consonant options of :func:`align`.
    """
    phones = [p for g in groups for p in g["phones"]]
    starts = align(groups, end, pred, frame, min_vowel, **caps)
    b = [int(round(s / frame)) for s in starts] + [int(round(end / frame))]
    for i in range(1, len(b)):
        b[i] = max(b[i], b[i - 1] + 1)
    durs = [HEAD] + [y - x for x, y in zip(b, b[1:])] + [TAIL]
    seq = ["SP"] + phones[1:] + ["SP"]
    if min_breath:
        seq = ["SP" if p == "AP" and d * frame < min_breath else p for p, d in zip(seq, durs)]
    is_vowel = [False] + [j == 0 and bool(g.get("vowel", False)) and p not in REST_SYMBOLS
                          for g in groups for j, p in enumerate(g["phones"])][1:] + [False]
    return seq, durs, b[0] - HEAD, is_vowel


def phrase_bounds(units, frame: float, before: float = 0.0) -> tuple[float, float]:
    """Seconds outside which a phrase has no frames, whatever the duration model predicts.

    The head starts at most ``PAD`` and ``HEAD`` frames before the first unit's consonants
    (and ``before`` s more: the breath a phrase may start with); the end moves past the last
    unit by at most one frame per phoneme (each phoneme gets at least a frame).
    """
    phones = sum(len(u.phones) + 2 for u in units)
    start = units[0].start - (units[0].lead or 0.0) - before
    return start - PAD - (HEAD + 1) * frame, units[-1].end + (TAIL + 1 + phones) * frame


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


def slow_consonants(k, seq, durs, frame: float, least: dict, most: float = SLOW_CONS_MAX) -> np.ndarray:
    """Slow-motion factors ``k`` (one per phoneme of ``seq``) raised so that a consonant of a
    class in ``least`` (``{class: s}``) is sung at least that long, at most ``most`` times its
    real length: the acoustic model gets the time a stop needs for a closure and a release."""
    k = np.array(k, np.float64)
    for i, p in enumerate(seq):
        m = least.get(cons_class(p))
        if m:
            k[i] = float(np.clip(m / (durs[i] * frame), k[i], max(k[i], most)))
    return k


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


def squeeze_bursts(mel, mel_slow, seq, durs, durs_slow, classes) -> np.ndarray:
    """Replace the linearly squeezed rows of the consonants of ``classes`` (stops, affricates)
    by rows cut out of the slow-motion mel, frame for frame.

    The frames kept are the slow-motion frames around the release, found as the first frame
    whose level in the upper half of the mel bins has risen ``BURST_RISE`` of the way from the
    closure's lowest level to the consonant's highest; one frame of closure before it for
    every two frames of the consonant. The rest of the closure and the end of the aspiration
    are dropped. Linear interpolation, which the other phonemes keep, averages each frame with
    its neighbours: a burst one frame long is averaged with the closure before it.
    """
    et = np.concatenate([[0], np.cumsum(durs)]).astype(int)
    es = np.concatenate([[0], np.cumsum(durs_slow)]).astype(int)
    mel = mel.copy()
    half = mel_slow.shape[1] // 2
    for i, p in enumerate(seq):
        nt, ns = int(durs[i]), int(durs_slow[i])
        if cons_class(p) not in classes or ns <= nt:
            continue
        e = mel_slow[es[i]:es[i + 1], half:].mean(axis=1)
        rise = e - np.minimum.accumulate(e)
        b = int(np.argmax(rise >= BURST_RISE * rise.max())) if rise.max() > 0 else ns - 1
        a = int(np.clip(b - max(1, nt // 2), 0, ns - nt))
        mel[et[i]:et[i + 1]] = mel_slow[es[i] + a:es[i] + a + nt]
    return mel


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


# ---- consonants and breaths in the variance curves -------------------------------------------------
def breath_shape(n: int, frame: float) -> np.ndarray:
    """dB under its peak for the ``n`` frames of a breath: in from -6, loudest after a third, a
    slow fall to -4 and out to -22 over the last ``BREATH_FALL`` s."""
    t = (np.arange(n) + 0.5) * frame
    end = n * frame
    fall = max(end - BREATH_FALL, 0.5 * end)
    return np.interp(t, [0.0, 0.35 * fall, fall, end], [-6.0, 0.0, -4.0, -22.0])


def shape_consonants(var: dict, seq, durs, is_vowel, frame: float, voicing_floor: dict | None = None,
                     breath_db: float | None = None, window: float = VOICING_WINDOW) -> dict:
    """The variance curves on consonant and breath frames (``var`` is changed in place).

    ``voicing_floor`` (``{class: dB}``): frames of voiced consonants (nasals, glides, voiced
    stops and fricatives) are held at or above the vowel voicing around them (running median
    over the phrase's vowels, interpolated across the consonants) minus their class's value,
    so the voice does not stop on them. ``breath_db``: the breathiness of a breath (``AP``)
    peaks at this (:func:`breath_shape`) and its voicing is -96. The voicing and breathiness
    curves are clipped to -96..0 first.
    """
    idx = np.concatenate([[0], np.cumsum(durs)]).astype(int)
    cls = [cons_class(p) for p in seq]
    for k in ("voicing", "breathiness"):
        if k in var:
            var[k] = np.clip(var[k], -96, 0)
    if voicing_floor is not None and "voicing" in var:
        v = var["voicing"]
        vmask = np.repeat(np.asarray(is_vowel, bool), durs)
        if vmask.sum() >= 4:
            at = np.flatnonzero(vmask)
            ref = np.interp(np.arange(len(v)), at, _running_median(v[vmask], int(round(window / frame))))
            for i, c in enumerate(cls):
                fl = voicing_floor.get(c) if c in VOICED_CONS else None
                if fl is not None:
                    a, b = idx[i], idx[i + 1]
                    v[a:b] = np.maximum(v[a:b], ref[a:b] - fl)
        var["voicing"] = v
    if breath_db is not None and "breathiness" in var:
        br = var["breathiness"]
        for i, p in enumerate(seq):
            if p == "AP":
                a, b = idx[i], idx[i + 1]
                br[a:b] = breath_db + breath_shape(b - a, frame)
                if "voicing" in var:
                    var["voicing"][a:b] = -96.0
        var["breathiness"] = br
    return var


def vowel_weight(durs, is_vowel) -> np.ndarray:
    """One weight per frame: 1 inside vowels, 0.5 on a vowel's first and last frame, 0 elsewhere."""
    w = np.zeros(int(sum(durs)), np.float32)
    idx = np.concatenate([[0], np.cumsum(durs)]).astype(int)
    for i in np.flatnonzero(is_vowel):
        a, b = idx[i], idx[i + 1]
        w[a:b] = 1.0
        w[a] = w[b - 1] = 0.5
    return w


def consonant_gain(wav, seq, durs, gain: dict, hop: int, sr: int, hf: float | None = None,
                   tail: float = 0.010) -> np.ndarray:
    """A fixed gain (dB) per consonant class on a phrase's waveform: ``gain = {class: dB}``.

    With ``hf`` (Hz) only the band above it is changed (the noise of bursts, aspiration and
    sibilance; the voiced part under it stays). Unvoiced stops and affricates take the gain up
    to ``tail`` s into the next phoneme (the release runs into the vowel). 4 ms ramps.
    """
    idx = (np.concatenate([[0], np.cumsum(durs)]) * hop).astype(int)
    g = np.zeros(len(wav))
    for i, p in enumerate(seq):
        c = cons_class(p)
        d = gain.get(c) if c is not None else None
        if not d:
            continue
        b = idx[i + 1] + (int(tail * sr) if c in ("stop", "aff") else 0)
        g[idx[i]:min(b, len(wav))] = d
    if not g.any():
        return wav
    r = int(0.004 * sr)
    win = np.hanning(2 * r + 1)
    g = np.convolve(np.pad(g, r, mode="edge"), win / win.sum(), mode="valid")[: len(wav)]
    lin = 10 ** (g / 20) - 1.0
    if hf:
        from ..engine import mixing as mx

        part = mx.eq(wav.astype(np.float64), [("hp", float(hf), 0, 0.7071, 2)], sr)
    else:
        part = wav.astype(np.float64)
    return (wav + lin * part).astype(np.float32)


# ---- noise ----------------------------------------------------------------------------------------
def phrase_seed(seed: int, f_start: int, n: int) -> int:
    """31-bit seed from the track seed and the phrase's first frame and length, not from its
    notes or lyrics: with the noise unchanged, editing a note leaves the rest of the phrase's
    take in place."""
    key = [int(seed), int(f_start), int(n)]
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big") & 0x7FFFFFFF
