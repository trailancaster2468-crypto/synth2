"""MusicXML import and export (``.musicxml`` / ``.xml`` and compressed ``.mxl``).

Reading (:func:`read_musicxml`, :func:`loads_musicxml`) builds one Synth2
track per ``<part>`` (``score-partwise``; ``score-timewise`` is converted):

* positions follow ``<divisions>``, ``<backup>`` and ``<forward>``; only the
  first staff and its first voice (voice ``1`` when present) form the
  monophonic line; chords keep their top note; grace and cue notes are skipped;
* tied notes are merged (a tied continuation without a lyric extends the
  note it is tied to);
* lyrics (verse 1): ``syllabic`` ``single`` -> the word, ``begin`` -> the full
  word joined from its syllables, ``middle`` / ``end`` -> ``"+"``; notes without
  a lyric inside a slur, under an ``<extend/>`` or inside an unfinished word
  become ``"-"`` melismas; other lyric-less notes sing ``"la"``;
* tempo from ``<sound tempo>`` (preferred) or ``<metronome>``; the first time
  signature and key (``fifths`` + ``mode`` -> ``"D major"``) set the project's;
* microtonal ``<alter>`` values become the ``detune`` attribute (cents).

Writing (:func:`write_musicxml`, :func:`dumps_musicxml`) produces MusicXML 4.0
partwise with ``divisions`` 480: one part per track, proper measures from the
project time signature, notes split at bar lines (and into notatable values)
with ties, rests in the gaps, word syllables as ``begin`` / ``middle`` / ``end``,
melismas as a slur plus ``<extend/>``, and tempo changes as
``<direction>``/``<sound tempo>`` in the first part.  A path ending in ``.mxl``
is written as a compressed MusicXML archive.
"""
from __future__ import annotations

import io
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from ..project import Note, Project, Track
from ..timing import TempoMap, TempoPoint
from .midi import fifths_to_key, key_to_fifths

DIVISIONS = 480
MELISMA_LYRICS = {"-", "~", "ー"}
_STEP = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_UNIT_BEATS = {
    "maxima": Fraction(32), "long": Fraction(16), "breve": Fraction(8), "whole": Fraction(4), "half": Fraction(2),
    "quarter": Fraction(1), "eighth": Fraction(1, 2), "16th": Fraction(1, 4), "32nd": Fraction(1, 8),
    "64th": Fraction(1, 16), "128th": Fraction(1, 32), "256th": Fraction(1, 64),
}
_SHARP_NAMES = [("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0), ("G", 1), ("A", 0), ("A", 1), ("B", 0)]
_FLAT_NAMES = [("C", 0), ("D", -1), ("D", 0), ("E", -1), ("E", 0), ("F", 0), ("G", -1), ("G", 0), ("A", -1), ("A", 0), ("B", -1), ("B", 0)]


def _has_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def _strip_ns(root: ET.Element) -> ET.Element:
    for el in root.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return root


def _frac(text: str | None) -> Fraction:
    try:
        return Fraction(str(text).strip()) if text not in (None, "") else Fraction(0)
    except (ValueError, ZeroDivisionError):
        return Fraction(0)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _xml_from_mxl(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = z.namelist()
        target = None
        if "META-INF/container.xml" in names:
            container = _strip_ns(ET.fromstring(z.read("META-INF/container.xml")))
            for rf in container.iter("rootfile"):
                fp = rf.get("full-path")
                media = rf.get("media-type") or ""
                if fp and fp in names and ("musicxml" in media or not media or fp.lower().endswith((".xml", ".musicxml"))):
                    target = fp
                    break
        if target is None:
            cands = [n for n in names if n.lower().endswith((".musicxml", ".xml")) and not n.startswith("META-INF/")]
            if not cands:
                raise ValueError("no MusicXML score found in .mxl archive")
            target = cands[0]
        return z.read(target)


@dataclass
class _XNote:
    onset: Fraction
    duration: Fraction
    pitch: int
    lyric: str | None = None
    detune: float = 0.0

    @property
    def end(self) -> Fraction:
        return self.onset + self.duration


def _staff(el: ET.Element) -> int:
    s = (el.findtext("staff") or "1").strip()
    return int(s) if s.isdigit() else 1


def _pitch_of(el: ET.Element) -> tuple[int, float] | None:
    p = el.find("pitch")
    if p is None:
        return None
    step = (p.findtext("step") or "C").strip().upper()
    try:
        alter = float(p.findtext("alter") or 0)
        octave = int(p.findtext("octave") or 4)
    except ValueError:
        return None
    exact = (octave + 1) * 12 + _STEP.get(step, 0) + alter
    midi = math.floor(exact + 0.5)  # quarter tones round up consistently
    return midi, round((exact - midi) * 100.0, 3)


def _clean_syllable(text: str) -> str:
    s = text.strip()
    if s.strip("-"):
        s = s.strip("-").strip()
    return s


def _lyric_of(els: list[ET.Element]) -> ET.Element | None:
    for e in els:
        lyrics = e.findall("lyric")
        if not lyrics:
            continue
        for ly in lyrics:
            if (ly.get("number") or "1") == "1":
                return ly
        return lyrics[0]
    return None


def _direction_tempo(el: ET.Element) -> float | None:
    snd = el.find("sound") if el.tag != "sound" else el
    if snd is not None and snd.get("tempo"):
        try:
            bpm = float(snd.get("tempo"))
            if bpm > 0:
                return bpm
        except ValueError:
            pass
    for met in el.iter("metronome"):
        unit = (met.findtext("beat-unit") or "").strip()
        pm = met.findtext("per-minute") or ""
        m = re.search(r"\d+(?:\.\d+)?", pm)
        if unit in _UNIT_BEATS and m:
            dots = len(met.findall("beat-unit-dot"))
            return float(m.group(0)) * float(_UNIT_BEATS[unit]) * (2.0 - 0.5 ** dots)
    return None


class _PartReader:
    """Walks one part's measures and builds its monophonic note line."""

    def __init__(self, measures: list[ET.Element], tempos: dict[Fraction, float], info: dict):
        self.measures = measures
        self.tempos = tempos
        self.info = info
        self.out: list[_XNote] = []
        self.word_idx: int | None = None
        self.word_parts: list[str] = []
        self.slurs: set[str] = set()
        self.extend = False
        self.phrase = False  # a note has been sung since the last rest
        notes = [n for m in measures for n in m.findall("note")]
        self.main_staff = min((_staff(n) for n in notes), default=1)
        voices = [(n.findtext("voice") or "").strip() for n in notes
                  if _staff(n) == self.main_staff and n.find("pitch") is not None]
        voices = [v for v in voices if v]
        self.main_voice = "1" if "1" in voices else (voices[0] if voices else None)
        self.has_lyrics = any(n.find("lyric/text") is not None for n in notes)

    # -- word handling ----------------------------------------------------
    def _finish_word(self):
        if self.word_idx is not None and self.word_parts:
            self.out[self.word_idx].lyric = "".join(self.word_parts)
        self.word_idx, self.word_parts = None, []

    def _in_line(self, el: ET.Element) -> bool:
        if _staff(el) != self.main_staff:
            return False
        voice = (el.findtext("voice") or "").strip()
        return not (voice and self.main_voice and voice != self.main_voice)

    def _flush(self, group: list[ET.Element], start: Fraction, duration: Fraction):
        if not group or not self._in_line(group[0]):
            return
        group = [e for e in group if e.find("cue") is None]
        if not group:
            return
        pitched = [(e, *pp) for e in group if (pp := _pitch_of(e)) is not None]
        if not pitched:  # rest (or unpitched percussion)
            self.extend = False
            self.phrase = False
            return
        el, pitch, detune = max(pitched, key=lambda x: x[1] + x[2] / 100.0)
        ties = {t.get("type") for t in el.findall("tie")} | {t.get("type") for t in el.findall("notations/tied")}
        tie_stop = bool(ties & {"stop", "continue"})
        ly = _lyric_of([el] + [e for e in group if e is not el])
        text = None
        syllabic = "single"
        has_extend = False
        extend_stop = False
        if ly is not None:
            texts = [t.text or "" for t in ly.findall("text")]
            joined = " ".join(_clean_syllable(t) for t in texts if t.strip())
            text = joined or None
            syllabic = (ly.findtext("syllabic") or "single").strip()
            ext = ly.find("extend")
            has_extend = ext is not None and ext.get("type") != "stop"
            extend_stop = ext is not None and ext.get("type") == "stop"
        in_slur = bool(self.slurs)
        prev = self.out[-1] if self.out else None
        contiguous = prev is not None and prev.end == start
        if tie_stop and text is None and prev is not None and prev.pitch == pitch and contiguous:
            prev.duration += duration
        else:
            note = _XNote(start, duration, pitch, detune=detune)
            if text is not None:
                if syllabic == "begin":
                    self._finish_word()
                    self.word_idx, self.word_parts = len(self.out), [text]
                    note.lyric = text
                elif syllabic in ("middle", "end") and self.word_idx is not None:
                    self.word_parts.append(text)
                    note.lyric = "+"
                else:
                    self._finish_word()
                    note.lyric = text
                self.extend = has_extend
            elif ly is not None and (has_extend or extend_stop):
                note.lyric = "-"
            elif self.has_lyrics and self.out and (in_slur or (self.phrase and (self.extend or self.word_idx is not None or tie_stop))):
                note.lyric = "-"  # a slur still marks a melisma when it spans a rest
            self.out.append(note)
            if text is not None and syllabic == "end" and self.word_idx is not None:
                self._finish_word()
        if extend_stop:
            self.extend = False
        self.phrase = True
        slur_events = [s for e in group for s in e.findall("notations/slur")]
        for s in slur_events:
            if s.get("type") == "stop":
                self.slurs.discard(s.get("number") or "1")
        for s in slur_events:
            if s.get("type") == "start":
                self.slurs.add(s.get("number") or "1")

    def _attributes(self, el: ET.Element):
        div = el.findtext("divisions")
        if div:
            d = _frac(div)
            if d > 0:
                self.divisions = d
        t = el.find("time")
        if t is not None and t.findtext("beats"):
            try:
                beats = sum(int(x) for x in re.split(r"\+", t.findtext("beats").strip()))
                beat_type = int((t.findtext("beat-type") or "4").strip())
                self.bar = Fraction(beats * 4, beat_type)
                self.info.setdefault("time_signature", (beats, beat_type))
            except ValueError:
                pass
        k = el.find("key")
        if k is not None and k.findtext("fifths") is not None and "key" not in self.info:
            try:
                mode = (k.findtext("mode") or "major").strip().lower()
                self.info["key"] = fifths_to_key(int(k.findtext("fifths").strip()), mode == "minor")
            except ValueError:
                pass

    def read(self) -> list[_XNote]:
        self.divisions = Fraction(1)
        self.bar = Fraction(4)
        m_start = Fraction(0)
        for m in self.measures:
            pos = Fraction(0)
            mlen = Fraction(0)
            group: list[ET.Element] = []
            g_start = g_dur = Fraction(0)
            content = False
            for el in m:
                tag = el.tag
                if tag == "note":
                    if el.find("chord") is not None and group:
                        group.append(el)
                        continue
                    self._flush(group, m_start + g_start, g_dur)
                    group = []
                    if el.find("grace") is not None:
                        continue
                    dur = _frac(el.findtext("duration")) / self.divisions
                    group, g_start, g_dur = [el], pos, dur
                    pos += dur
                    mlen = max(mlen, pos)
                    content = True
                    continue
                self._flush(group, m_start + g_start, g_dur)
                group = []
                if tag == "attributes":
                    self._attributes(el)
                elif tag == "backup":
                    pos = max(Fraction(0), pos - _frac(el.findtext("duration")) / self.divisions)
                elif tag == "forward":
                    pos += _frac(el.findtext("duration")) / self.divisions
                    mlen = max(mlen, pos)
                    content = True
                elif tag in ("direction", "sound"):
                    bpm = _direction_tempo(el)
                    if bpm:
                        at = m_start + pos + _frac(el.findtext("offset")) / self.divisions
                        self.tempos.setdefault(max(Fraction(0), at), bpm)
            self._flush(group, m_start + g_start, g_dur)
            m_start += mlen if content or mlen else self.bar
        self._finish_word()
        return self.out


def _part_measures(root: ET.Element) -> list[tuple[str, list[ET.Element]]]:
    if root.tag == "score-timewise":
        parts: dict[str, list[ET.Element]] = {}
        for m in root.findall("measure"):
            for p in m.findall("part"):
                parts.setdefault(p.get("id") or "", []).append(p)
        return list(parts.items())
    return [(p.get("id") or "", p.findall("measure")) for p in root.findall("part")]


def loads_musicxml(data: bytes | str, title: str | None = None) -> Project:
    """Parse MusicXML text/bytes (or ``.mxl`` zip bytes) into a :class:`Project`."""
    if isinstance(data, bytes) and data[:2] == b"PK":
        data = _xml_from_mxl(data)
    root = _strip_ns(ET.fromstring(data))
    if root.tag not in ("score-partwise", "score-timewise"):
        raise ValueError(f"not a MusicXML score (root element <{root.tag}>)")
    names = {}
    for sp in root.iter("score-part"):
        nm = (sp.findtext("part-name") or "").strip() or (sp.findtext("part-abbreviation") or "").strip()
        names[sp.get("id") or ""] = nm
    work_title = (root.findtext("work/work-title") or "").strip() or (root.findtext("movement-title") or "").strip()
    project = Project(title=work_title or title or "Untitled")
    composer = root.find("identification/creator[@type='composer']")
    if composer is not None and (composer.text or "").strip():
        project.metadata["composer"] = composer.text.strip()

    tempos: dict[Fraction, float] = {}
    info: dict = {}
    for i, (pid, measures) in enumerate(_part_measures(root)):
        notes = _PartReader(measures, tempos, info).read()
        if not notes:
            continue
        track = project.add_track(names.get(pid) or f"Part {i + 1}")
        for xn in notes:
            attrs = {"detune": xn.detune} if xn.detune else {}
            track.notes.append(Note(float(xn.onset), float(xn.duration), xn.pitch, xn.lyric or "la", attrs=attrs))
        if any(_has_kana(n.lyric) for n in track.notes):
            track.language = "ja"
        track.sort()
    if tempos:
        pts = sorted(tempos.items())
        if pts[0][0] > 0:
            pts.insert(0, (Fraction(0), pts[0][1]))
        project.tempo = TempoMap([TempoPoint(float(b), float(v)) for b, v in pts])
    if "time_signature" in info:
        project.time_signature = info["time_signature"]
    if "key" in info:
        project.key = info["key"]
    return project


def read_musicxml(path: str | Path) -> Project:
    """Read a ``.musicxml`` / ``.xml`` / ``.mxl`` file."""
    path = Path(path)
    return loads_musicxml(path.read_bytes(), title=path.stem)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
_BASE_TYPES = [("breve", 3840), ("whole", 1920), ("half", 960), ("quarter", 480), ("eighth", 240),
               ("16th", 120), ("32nd", 60), ("64th", 30), ("128th", 15)]


def _note_values() -> list[tuple[int, str, int, bool, float]]:
    """(ticks, type, dots, triplet, cost) for every notatable value at 480 divisions."""
    vals: dict[int, tuple[int, str, int, bool, float]] = {}
    for name, t in _BASE_TYPES:
        for dots, mul in ((0, Fraction(1)), (1, Fraction(3, 2)), (2, Fraction(7, 4))):
            v = t * mul
            if v.denominator == 1 and int(v) not in vals:
                vals[int(v)] = (int(v), name, dots, False, 1.0 + 0.1 * dots)
    for name, t in _BASE_TYPES:
        v = Fraction(2 * t, 3)
        if v.denominator == 1 and int(v) not in vals:
            vals[int(v)] = (int(v), name, 0, True, 1.5)
    return sorted(vals.values(), reverse=True)


_VALUES = _note_values()
_VALUE_INFO = {v[0]: v for v in _VALUES}
_DP: list[tuple[float, int]] = [(0.0, 0)]  # best (cost, first value) for each tick count


def _decompose(ticks: int, max_pieces: int = 6) -> list[int] | None:
    """Split ``ticks`` into the fewest notatable values (``None`` if impossible)."""
    while len(_DP) <= ticks:
        r = len(_DP)
        best = (math.inf, 0)
        for v, _, _, _, cost in _VALUES:
            if v <= r and _DP[r - v][0] + cost < best[0]:
                best = (_DP[r - v][0] + cost, v)
        _DP.append(best)
    if ticks <= 0 or _DP[ticks][0] == math.inf:
        return None
    out = []
    r = ticks
    while r > 0:
        v = _DP[r][1]
        out.append(v)
        r -= v
    if len(out) > max_pieces:
        return None
    return sorted(out, reverse=True)


def _mono_ticks(track: Track) -> list[tuple[int, int, Note]]:
    out: list[list] = []
    for n in sorted(track.notes, key=lambda n: (n.onset, -n.pitch)):
        on = max(0, int(round(n.onset * DIVISIONS)))
        off = int(round(n.end * DIVISIONS))
        if out and out[-1][0] == on:
            continue
        if out and out[-1][1] > on:
            out[-1][1] = on
        out.append([on, off, n])
    return [(a, b, n) for a, b, n in out if b > a]


def _lyric_plan(notes: list[Note]):
    """Per note: (syllabic, text) or None, plus slur start/stop and extend sets."""
    from ..phonemes.english import split_word_text

    n = len(notes)
    plan: list[tuple[str, str] | None] = [None] * n
    slur_start, slur_stop, extend = set(), set(), set()
    i = 0
    while i < n:
        lyr = (notes[i].lyric or "la").strip() or "la"
        if lyr in MELISMA_LYRICS and i > 0:
            j = i
            while j + 1 < n and (notes[j + 1].lyric or "").strip() in MELISMA_LYRICS:
                j += 1
            slur_start.add(i - 1)
            slur_stop.add(j)
            if plan[i - 1] is not None:
                extend.add(i - 1)
            i = j + 1
            continue
        j = i + 1
        while j < n and (notes[j].lyric or "").strip() == "+":
            j += 1
        k = j - i
        parts = split_word_text(lyr, k) if k > 1 and re.fullmatch(r"[^\s\-\[\].]+", lyr) else []
        if k > 1 and len(parts) == k and all(p.strip() for p in parts):
            plan[i] = ("begin", parts[0])
            for m in range(1, k):
                plan[i + m] = ("end" if m == k - 1 else "middle", parts[m])
        else:
            plan[i] = ("single", lyr)
            for m in range(1, k):
                plan[i + m] = ("single", "+")
        i = j
    return plan, slur_start, slur_stop, extend


def _fmt_num(x: float) -> str:
    x = float(x)
    return str(int(x)) if x.is_integer() else f"{x:.6g}"


def _direction(parent: ET.Element, bpm: float, offset: int):
    d = ET.SubElement(parent, "direction", placement="above")
    dt = ET.SubElement(d, "direction-type")
    met = ET.SubElement(dt, "metronome")
    ET.SubElement(met, "beat-unit").text = "quarter"
    ET.SubElement(met, "per-minute").text = _fmt_num(round(bpm, 2))
    if offset:
        ET.SubElement(d, "offset", sound="yes").text = str(offset)
    ET.SubElement(d, "sound", tempo=_fmt_num(round(bpm, 6)))


def _part_xml(part: ET.Element, track: Track, project: Project, n_measures: int, bar: int, first: bool,
              fifths: int, minor: bool):
    mono = _mono_ticks(track)
    notes = [n for _, _, n in mono]
    plan, slur_start, slur_stop, extend = _lyric_plan(notes)
    total = n_measures * bar

    # segments covering the whole part: (start, end, note index or None)
    segs: list[tuple[int, int, int | None]] = []
    cursor = 0
    for i, (on, off, _) in enumerate(mono):
        if on > cursor:
            segs.append((cursor, on, None))
        segs.append((on, off, i))
        cursor = off
    if cursor < total:
        segs.append((cursor, total, None))

    # split at bar lines and into notatable values -> pieces per measure
    pieces: list[list[tuple[int, int, int | None, bool]]] = [[] for _ in range(n_measures)]
    for s, e, idx in segs:
        t = s
        while t < e:
            mi = t // bar
            if mi >= n_measures:
                break
            chunk_end = min(e, (mi + 1) * bar)
            length = chunk_end - t
            if idx is None and length == bar:
                pieces[mi].append((t, length, None, True))  # whole-measure rest
            else:
                vals = _decompose(length) or [length]
                p = t
                for v in vals:
                    pieces[mi].append((p, v, idx, False))
                    p += v
            t = chunk_end
    first_piece: dict[int, tuple[int, int]] = {}
    last_piece: dict[int, tuple[int, int]] = {}
    for mi, plist in enumerate(pieces):
        for k, (_, _, idx, _) in enumerate(plist):
            if idx is not None:
                first_piece.setdefault(idx, (mi, k))
                last_piece[idx] = (mi, k)

    tempo_ticks = [(int(round(p.beat * DIVISIONS)), p.bpm) for p in project.tempo.points] if first else []
    names = _SHARP_NAMES if fifths >= 0 else _FLAT_NAMES
    pitches = [n.pitch for n in notes]
    low_clef = bool(pitches) and sorted(pitches)[len(pitches) // 2] < 55

    for mi in range(n_measures):
        m = ET.SubElement(part, "measure", number=str(mi + 1))
        if mi == 0:
            attrs = ET.SubElement(m, "attributes")
            ET.SubElement(attrs, "divisions").text = str(DIVISIONS)
            key = ET.SubElement(attrs, "key")
            ET.SubElement(key, "fifths").text = str(fifths)
            ET.SubElement(key, "mode").text = "minor" if minor else "major"
            time = ET.SubElement(attrs, "time")
            ET.SubElement(time, "beats").text = str(project.time_signature[0])
            ET.SubElement(time, "beat-type").text = str(project.time_signature[1])
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "F" if low_clef else "G"
            ET.SubElement(clef, "line").text = "4" if low_clef else "2"
        m_start, m_end = mi * bar, (mi + 1) * bar
        plist = pieces[mi]
        for k, (start, length, idx, whole_rest) in enumerate(plist):
            nxt = plist[k + 1][0] if k + 1 < len(plist) else m_end
            for tt, bpm in tempo_ticks:
                if start <= tt < nxt or (k == 0 and m_start <= tt < start):
                    _direction(m, bpm, max(0, tt - start))
            el = ET.SubElement(m, "note")
            if idx is None:
                if whole_rest:
                    ET.SubElement(el, "rest", measure="yes")
                else:
                    ET.SubElement(el, "rest")
                ET.SubElement(el, "duration").text = str(length)
                ET.SubElement(el, "voice").text = "1"
                info = None if whole_rest else _VALUE_INFO.get(length)
                if info:
                    _type_elements(el, info)
                continue
            n = notes[idx]
            is_first = first_piece[idx] == (mi, k)
            is_last = last_piece[idx] == (mi, k)
            pitch = max(0, min(127, int(n.pitch)))
            step, alter = names[pitch % 12]
            p = ET.SubElement(el, "pitch")
            ET.SubElement(p, "step").text = step
            if alter:
                ET.SubElement(p, "alter").text = str(alter)
            ET.SubElement(p, "octave").text = str(pitch // 12 - 1)
            ET.SubElement(el, "duration").text = str(length)
            if not is_first:
                ET.SubElement(el, "tie", type="stop")
            if not is_last:
                ET.SubElement(el, "tie", type="start")
            ET.SubElement(el, "voice").text = "1"
            info = _VALUE_INFO.get(length)
            if info:
                _type_elements(el, info)
            notations = []
            if not is_first:
                notations.append(("tied", "stop"))
            if not is_last:
                notations.append(("tied", "start"))
            if is_first and idx in slur_start:
                notations.append(("slur", "start"))
            if is_last and idx in slur_stop:
                notations.append(("slur", "stop"))
            if notations:
                ne = ET.SubElement(el, "notations")
                for tag, typ in notations:
                    attrs = {"type": typ, "number": "1"} if tag == "slur" else {"type": typ}
                    ET.SubElement(ne, tag, **attrs)
            if is_first and plan[idx] is not None:
                syl, text = plan[idx]
                ly = ET.SubElement(el, "lyric", number="1")
                ET.SubElement(ly, "syllabic").text = syl
                ET.SubElement(ly, "text").text = text
                if idx in extend:
                    ET.SubElement(ly, "extend")


def _type_elements(el: ET.Element, info: tuple[int, str, int, bool, float]):
    _, name, dots, triplet, _ = info
    ET.SubElement(el, "type").text = name
    for _ in range(dots):
        ET.SubElement(el, "dot")
    if triplet:
        tm = ET.SubElement(el, "time-modification")
        ET.SubElement(tm, "actual-notes").text = "3"
        ET.SubElement(tm, "normal-notes").text = "2"


def dumps_musicxml(project: Project) -> str:
    """Serialise ``project`` as MusicXML 4.0 (partwise) text."""
    root = ET.Element("score-partwise", version="4.0")
    work = ET.SubElement(root, "work")
    ET.SubElement(work, "work-title").text = project.title
    ident = ET.SubElement(root, "identification")
    if project.metadata.get("composer"):
        ET.SubElement(ident, "creator", type="composer").text = str(project.metadata["composer"])
    enc = ET.SubElement(ident, "encoding")
    ET.SubElement(enc, "software").text = "Synth2 Vocal Studio"
    plist = ET.SubElement(root, "part-list")
    for i, t in enumerate(project.tracks):
        sp = ET.SubElement(plist, "score-part", id=f"P{i + 1}")
        ET.SubElement(sp, "part-name").text = t.name
    bar = max(1, int(round(project.beats_per_bar() * DIVISIONS)))
    end = max([int(round(n.end * DIVISIONS)) for t in project.tracks for n in t.notes] + [1])
    n_measures = max(1, -(-end // bar))
    k = key_to_fifths(project.key) or (0, False)
    for i, t in enumerate(project.tracks):
        part = ET.SubElement(root, "part", id=f"P{i + 1}")
        _part_xml(part, t, project, n_measures, bar, i == 0, k[0], k[1])
    if hasattr(ET, "indent"):
        ET.indent(root, space="  ")
    header = ('<?xml version="1.0" encoding="UTF-8" standalone="no"?>\n'
              '<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
              '"http://www.musicxml.org/dtds/partwise.dtd">\n')
    return header + ET.tostring(root, encoding="unicode") + "\n"


_CONTAINER = ('<?xml version="1.0" encoding="UTF-8"?>\n'
              '<container><rootfiles><rootfile full-path="score.musicxml" '
              'media-type="application/vnd.recordare.musicxml+xml"/></rootfiles></container>\n')


def write_musicxml(project: Project, path: str | Path) -> Path:
    """Write ``project`` as MusicXML; ``.mxl`` paths get a compressed archive."""
    path = Path(path)
    text = dumps_musicxml(project)
    if path.suffix.lower() == ".mxl":
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(zipfile.ZipInfo("mimetype"), "application/vnd.recordare.musicxml")
            z.writestr("META-INF/container.xml", _CONTAINER, compress_type=zipfile.ZIP_DEFLATED)
            z.writestr("score.musicxml", text, compress_type=zipfile.ZIP_DEFLATED)
    else:
        path.write_text(text, encoding="utf-8")
    return path
