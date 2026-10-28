"""Segment planner: notes + lyrics -> timed phoneme segments.

Follows vocal-synth timing conventions: the vowel (syllable nucleus) lands
exactly on the note onset, onset consonants are placed *before* the note
(borrowing time from the previous note or rest), coda consonants close the
syllable just before the next onset, melisma notes extend the vowel, and
diphthongs glide to their second target at the end of the sung span.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import phonemes as ph
from ..phonemes.inventory import Phoneme, get
from .expression import NoteTiming, Style

PHRASE_GAP = 0.2  # seconds of silence that separate phrases


@dataclass
class Segment:
    ph: Phoneme
    start: float
    end: float
    part: str = "main"  # main | closure | burst | asp | fric | glide
    note: int = -1
    unit: int = -1
    level: float = 1.0  # extra amplitude scaling
    prev_vowel: Phoneme | None = None
    next_vowel: Phoneme | None = None

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class Unit:
    """A sung syllable spanning one note plus any melisma continuation notes."""

    notes: list[int]
    phones: list[str]
    start: float
    end: float
    breath: bool = False
    lead: float = 0.0
    phrase_start: bool = False
    phrase_end: bool = False
    onset: list[str] = field(default_factory=list)
    nuclei: list[list[str]] = field(default_factory=list)  # [ [consonants-before], vowel ]...
    coda: list[str] = field(default_factory=list)


@dataclass
class Plan:
    segments: list[Segment]
    timings: list[NoteTiming]
    units: list[Unit]
    note_phonemes: dict[str, str]
    warnings: list[str]
    start: float
    end: float


def _cons_dur(p: Phoneme, before_vowel: bool) -> float:
    if p.kind == "stop":
        return p.dur + (p.aspiration if before_vowel else 0.0)
    return p.dur


def _split_syllable(phones: list[str]) -> tuple[list[str], list[tuple[list[str], str]], list[str]]:
    """Split into onset consonants, [(inter-consonants, nucleus)], coda consonants."""
    nuc_idx = [i for i, p in enumerate(phones) if get(p).is_vowel or p == "N"]
    if not nuc_idx:
        # no vowel: sustain the most sonorous consonant, or add a schwa
        son = [i for i, p in enumerate(phones) if get(p).is_sonorant]
        if son:
            nuc_idx = [son[-1]]
        else:
            phones = phones + ["ax"]
            nuc_idx = [len(phones) - 1]
    onset = phones[: nuc_idx[0]]
    nuclei = [([], phones[nuc_idx[0]])]
    for a, b in zip(nuc_idx, nuc_idx[1:]):
        nuclei.append((phones[a + 1:b], phones[b]))
    coda = phones[nuc_idx[-1] + 1:]
    return onset, nuclei, coda


def plan_track(notes, tempo, language: str, style: Style, rng: np.random.Generator) -> Plan:
    warnings: list[str] = []
    notes = sorted(notes, key=lambda n: (n.onset, -n.pitch))
    # trim overlaps (a single voice is monophonic)
    clean = []
    for n in notes:
        if clean and n.onset < clean[-1].end - 1e-6:
            prev = clean[-1]
            if n.onset - prev.onset <= 1e-6:
                warnings.append(f"note {n.id} at beat {n.onset:g} starts with another note; dropped")
                continue
            warnings.append(f"notes {prev.id} and {n.id} overlap; first note shortened")
            prev = _shorten(prev, n.onset - prev.onset)
            clean[-1] = prev
        if n.duration > 1e-6:
            clean.append(n)
    notes = clean

    try:
        sylls = ph.resolve_notes(notes, language)
    except Exception as exc:  # bad explicit phonemes: resolve one by one
        warnings.append(f"lyric resolution: {exc}")
        sylls = []
        for n in notes:
            try:
                sylls += ph.resolve_notes([n], language)
            except Exception:
                warnings.append(f"note {n.id} lyric {n.lyric!r} invalid; sung as 'la'")
                sylls.append(ph.Syllable(["l", "aa"]))

    on = np.array([tempo.beat_to_sec(n.onset) for n in notes])
    off = np.array([tempo.beat_to_sec(n.end) for n in notes])
    if style.timing_humanize > 0 and len(notes):
        jit = rng.normal(0, style.timing_humanize, len(notes))
        on = on + jit
        off = np.maximum(off + np.roll(jit, -1) * 0.0, on + 0.03)

    # ---- build units
    units: list[Unit] = []
    note_unit = [-1] * len(notes)
    last_vowel = "aa"
    for i, (n, s) in enumerate(zip(notes, sylls)):
        if s.rest:
            continue
        if s.melisma:
            if units and units[-1].notes[-1] == i - 1 and on[i] - off[i - 1] < 0.08:
                units[-1].notes.append(i)
                units[-1].end = off[i]
                note_unit[i] = len(units) - 1
                continue
            phones = [last_vowel]
        else:
            phones = list(s.phones)
        if not phones:
            continue
        u = Unit([i], phones, on[i], off[i], breath=s.breath)
        units.append(u)
        note_unit[i] = len(units) - 1
        vs = [p for p in phones if get(p).is_vowel]
        if vs:
            last_vowel = get(vs[-1]).parts[-1] if get(vs[-1]).parts else vs[-1]

    # ---- phrase structure and consonant timing
    for k, u in enumerate(units):
        prev = units[k - 1] if k > 0 else None
        nxt = units[k + 1] if k + 1 < len(units) else None
        u.phrase_start = prev is None or u.start - prev.end >= PHRASE_GAP
        u.phrase_end = nxt is None or nxt.start - u.end >= PHRASE_GAP
        if u.breath:
            continue
        u.onset, nuclei, u.coda = _split_syllable(u.phones)
        u.nuclei = [list(c) + [v] for c, v in nuclei]
        cl = float(notes[u.notes[0]].attr("consonant_length", 1.0) or 1.0)
        d_on = sum(_cons_dur(get(p), True) for p in u.onset) * cl
        if d_on <= 0:
            u.lead = 0.0
            continue
        if prev is None or u.start - prev.end > 0.6:
            room = 0.45
        else:
            gap = max(0.0, u.start - prev.end)
            last_note_dur = off[prev.notes[-1]] - on[prev.notes[-1]]
            room = gap + 0.55 * last_note_dur
        scale = min(1.0, room / d_on)
        u.lead = d_on * max(scale, 0.3)

    segments: list[Segment] = []
    for k, u in enumerate(units):
        nxt = units[k + 1] if k + 1 < len(units) else None
        if u.breath:
            segments.append(Segment(get("br"), u.start, u.end, note=u.notes[0], unit=k, level=0.8))
            continue
        # onset consonants (placed before the vowel)
        t = u.start - u.lead
        d_nom = [(_cons_dur(get(p), True)) for p in u.onset]
        tot = sum(d_nom) or 1.0
        for p, d in zip(u.onset, d_nom):
            dd = u.lead * d / tot
            segments += _expand_consonant(get(p), t, t + dd, True, u.notes[0], k)
            t += dd
        # end of the sung span
        span_end = u.end
        if nxt is not None and nxt.start - nxt.lead < span_end + 1e-6:
            span_end = nxt.start - nxt.lead
        span_end = max(span_end, u.start + 0.03)
        # coda consonants
        d_coda = [_cons_dur(get(p), False) for p in u.coda]
        coda_total = sum(d_coda)
        if u.phrase_end and coda_total:
            coda_total_allowed = min(coda_total, 0.45 * (span_end - u.start))
        else:
            coda_total_allowed = min(coda_total, 0.4 * (span_end - u.start))
        coda_start = span_end - coda_total_allowed
        # nuclei
        vstart, vend = u.start, coda_start
        n_nuc = len(u.nuclei)
        slot = (vend - vstart) / n_nuc
        for j, group in enumerate(u.nuclei):
            cons, vowel = group[:-1], group[-1]
            a = vstart + j * slot
            b = a + slot
            if cons:
                cd = [_cons_dur(get(p), True) for p in cons]
                ctot = min(sum(cd), 0.45 * slot)
                cscale = ctot / (sum(cd) or 1)
                # inter-syllable consonants steal from the previous nucleus
                c_t = a - ctot if j > 0 else a
                if j > 0:
                    # shorten the previous nucleus to make room
                    for s in segments:
                        if s.unit == k and (s.ph.is_vowel or s.ph.name == "N") and s.end > c_t:
                            s.end = max(s.start + 0.005, c_t)
                for p, d in zip(cons, cd):
                    segments += _expand_consonant(get(p), c_t, c_t + d * cscale, True, u.notes[0], k)
                    c_t += d * cscale
                if j == 0:
                    a = c_t
            segments += _vowel_segments(get(vowel), a, b, u, k, last=(j == n_nuc - 1))
        t = coda_start
        for p, d in zip(u.coda, d_coda):
            dd = coda_total_allowed * d / (coda_total or 1)
            segments += _expand_consonant(get(p), t, t + dd, False, u.notes[-1], k, final=u.phrase_end)
            t += dd

    # automatic breaths before phrases
    if style.auto_breath:
        for k, u in enumerate(units):
            if not u.phrase_start or u.breath:
                continue
            prev_end = units[k - 1].end if k > 0 else -1.0
            if k > 0 and units[k - 1].breath:
                continue
            gap = (u.start - u.lead) - max(prev_end, 0.0) if k > 0 else u.start - u.lead
            if gap < 0.45:
                continue
            dur = min(0.32, 0.5 * gap)
            end = u.start - u.lead - 0.05
            segments.append(Segment(get("br"), end - dur, end, note=u.notes[0], unit=k, level=0.55))

    segments.sort(key=lambda s: s.start)
    # attach neighbouring vowels for coarticulation
    vowel_idx = [i for i, s in enumerate(segments) if s.ph.is_vowel]
    vi = 0
    prev_v = None
    for i, s in enumerate(segments):
        while vi < len(vowel_idx) and vowel_idx[vi] < i:
            prev_v = segments[vowel_idx[vi]].ph
            vi += 1
        nxt_v = segments[vowel_idx[vi]].ph if vi < len(vowel_idx) else None
        if vi < len(vowel_idx) and vowel_idx[vi] == i:
            nxt_v = s.ph
        s.prev_vowel = prev_v
        s.next_vowel = nxt_v

    # ---- note timings for the expression engine
    timings: list[NoteTiming] = []
    note_phonemes: dict[str, str] = {}
    for i, n in enumerate(notes):
        ui = note_unit[i]
        if ui < 0:
            continue
        u = units[ui]
        first = u.notes[0] == i
        timings.append(NoteTiming(
            index=i, start=on[i], end=off[i],
            pitch=n.pitch + float(n.attr("detune", 0.0) or 0.0) / 100.0,
            attrs=n.attrs, phrase_start=u.phrase_start and first,
            phrase_end=u.phrase_end and u.notes[-1] == i,
            lead=u.lead if first else 0.0, melisma=not first,
        ))
        note_phonemes[n.id] = " ".join(u.phones) if first else "-"
    start = min((s.start for s in segments), default=0.0)
    end = max((s.end for s in segments), default=0.0)
    return Plan(segments, timings, units, note_phonemes, warnings, start, end)


def _shorten(note, new_dur):
    from dataclasses import replace

    return replace(note, duration=new_dur)


def _vowel_segments(p: Phoneme, a: float, b: float, u: Unit, k: int, last: bool) -> list[Segment]:
    if b <= a:
        b = a + 0.01
    if p.kind == "diphthong":
        p1, p2 = get(p.parts[0]), get(p.parts[1])
        span = b - a
        glide = min(0.2, 0.32 * span) if last else 0.45 * span
        return [
            Segment(p1, a, b - glide, note=u.notes[0], unit=k),
            Segment(p2, b - glide, b, part="glide", note=u.notes[-1], unit=k),
        ]
    return [Segment(p, a, b, note=u.notes[0], unit=k)]


def _expand_consonant(p: Phoneme, a: float, b: float, before_vowel: bool, note: int, unit: int,
                      final: bool = False) -> list[Segment]:
    d = max(b - a, 0.005)
    b = a + d
    if p.kind == "affricate":
        stop, fric = get(p.parts[0]), get(p.parts[1])
        c_end = a + 0.42 * d
        return [
            Segment(stop, a, c_end - 0.006, part="closure", note=note, unit=unit),
            Segment(stop, c_end - 0.006, c_end, part="burst", note=note, unit=unit),
            Segment(fric, c_end, b, part="fric", note=note, unit=unit),
        ]
    if p.kind == "stop":
        if p.name == "q":
            return [Segment(p, a, b, part="closure", note=note, unit=unit)]
        burst = 0.010 if not p.voiced else 0.007
        asp = p.aspiration if (before_vowel and not p.voiced) else (0.025 if final and not p.voiced else 0.0)
        nominal = p.dur + asp
        scale = d / nominal if nominal > 0 else 1.0
        asp *= min(scale, 1.0)
        burst = min(burst, 0.3 * d)
        closure = max(d - burst - asp, 0.002)
        segs = [Segment(p, a, a + closure, part="closure", note=note, unit=unit),
                Segment(p, a + closure, a + closure + burst, part="burst", note=note, unit=unit)]
        if asp > 0.001:
            segs.append(Segment(p, a + closure + burst, b, part="asp", note=note, unit=unit))
        return segs
    return [Segment(p, a, b, note=note, unit=unit)]
