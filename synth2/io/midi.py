"""Standard MIDI File (SMF) import and export, written from scratch (no ``mido``).

Reading (:func:`read_midi`, :func:`loads_midi`):

* SMF format 0, 1 (and 2, treated like 1), running status, sysex skipping,
  RIFF ``RMID`` wrappers and SMPTE time division (approximated).
* Tempo map (``FF 51``), time signature (``FF 58``), key signature (``FF 59``),
  track names (``FF 03``, falling back to the instrument name ``FF 04``).
* Lyrics: ``FF 05`` events are attached to the note that starts at the same
  tick (with a small tolerance).  When the file contains no ``FF 05`` at all,
  ``FF 01`` text events are used instead (karaoke ``/`` and ``\\`` line markers
  are stripped and ``@`` header lines ignored).  Hyphenated syllables
  (``"twin-" "kle"``) become ``"twinkle" "+"``.  Notes without a lyric sing
  ``"la"``.
* One project :class:`~synth2.project.Track` per MIDI track that contains
  notes (per channel for format 0).  Drum-channel (10) notes are ignored when
  other notes exist.  Overlapping notes are trimmed into a monophonic line:
  for notes starting together the highest wins, otherwise the most recent
  note wins and the previous one is truncated.
* Note velocities become the ``velocity`` attribute, normalised to the
  track's median velocity (so a file played at a constant velocity imports
  with neutral dynamics).

Writing (:func:`write_midi`, :func:`dumps_midi`): format 1 at 480 PPQ with a
conductor track (title, time signature, key signature, tempo map) followed by
one track per project track, with the ``FF 05`` lyric event placed right
before each note-on.
"""
from __future__ import annotations

import bisect
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

from ..project import Note, Project, Track
from ..timing import TempoMap, TempoPoint

PPQ = 480
DRUM_CHANNEL = 9

_MAJOR_KEYS = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]
_MINOR_KEYS = ["Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", "G#", "D#", "A#"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


class MidiError(ValueError):
    """Raised for data that is not a readable Standard MIDI File."""


# ---------------------------------------------------------------------------
# key signature helpers (shared with the MusicXML module)
# ---------------------------------------------------------------------------
def fifths_to_key(fifths: int, minor: bool = False) -> str:
    """``(2, False)`` -> ``"D major"``; ``(-3, True)`` -> ``"C minor"``."""
    f = max(-7, min(7, int(fifths)))
    return f"{(_MINOR_KEYS if minor else _MAJOR_KEYS)[f + 7]} {'minor' if minor else 'major'}"


def key_to_fifths(key: str) -> tuple[int, bool] | None:
    """Parse ``"Bb major"``, ``"F# minor"``, ``"Am"`` ... into ``(fifths, minor)``."""
    m = re.match(r"^\s*([A-Ga-g])\s*([#b♯♭]?)\s*(major|minor|maj|min|m|)\s*$", str(key or ""), re.I)
    if not m:
        return None
    letter, acc, mode = m.group(1).upper(), m.group(2), m.group(3).lower()
    minor = mode in ("minor", "min", "m")
    acc = {"♯": "#", "♭": "b"}.get(acc, acc)
    table = _MINOR_KEYS if minor else _MAJOR_KEYS
    name = letter + acc
    if name in table:
        return table.index(name) - 7, minor
    pc = (_PC[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12
    for f in sorted(range(-7, 8), key=abs):
        n = table[f + 7]
        npc = (_PC[n[0]] + (1 if n[1:] == "#" else -1 if n[1:] == "b" else 0)) % 12
        if npc == pc:
            return f, minor
    return None


# ---------------------------------------------------------------------------
# low-level parsing
# ---------------------------------------------------------------------------
@dataclass
class RawNote:
    start: int  # ticks
    end: int
    pitch: int
    velocity: int
    channel: int


@dataclass
class RawTrack:
    """Everything of interest found in one ``MTrk`` chunk."""

    name: str = ""
    instrument: str = ""
    notes: list[RawNote] = field(default_factory=list)
    lyrics: list[tuple[int, str]] = field(default_factory=list)
    texts: list[tuple[int, str]] = field(default_factory=list)
    tempos: list[tuple[int, int]] = field(default_factory=list)  # (tick, microseconds per quarter)
    time_signatures: list[tuple[int, int, int]] = field(default_factory=list)  # (tick, num, den)
    key_signatures: list[tuple[int, int, bool]] = field(default_factory=list)  # (tick, fifths, minor)
    markers: list[tuple[int, str]] = field(default_factory=list)
    end_tick: int = 0


def _decode_text(b: bytes) -> str:
    for enc in ("utf-8", "cp932"):
        try:
            return b.decode(enc)
        except UnicodeDecodeError:
            pass
    return b.decode("latin-1")


def _read_vlq(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    for _ in range(4):
        b = data[pos]
        pos += 1
        value = (value << 7) | (b & 0x7F)
        if not b & 0x80:
            return value, pos
    return value, pos


def _parse_track(data: bytes) -> RawTrack:
    tr = RawTrack()
    pos = 0
    tick = 0
    status: int | None = None
    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    n = len(data)
    try:
        while pos < n:
            delta, pos = _read_vlq(data, pos)
            tick += delta
            if pos >= n:
                break
            b = data[pos]
            if b == 0xFF:  # meta event
                mtype = data[pos + 1]
                length, pos = _read_vlq(data, pos + 2)
                payload = data[pos:pos + length]
                pos += length
                if mtype == 0x2F:
                    break
                if mtype == 0x03 and not tr.name:
                    tr.name = _decode_text(payload).strip()
                elif mtype == 0x04 and not tr.instrument:
                    tr.instrument = _decode_text(payload).strip()
                elif mtype == 0x05:
                    tr.lyrics.append((tick, _decode_text(payload)))
                elif mtype == 0x01:
                    tr.texts.append((tick, _decode_text(payload)))
                elif mtype == 0x06:
                    tr.markers.append((tick, _decode_text(payload)))
                elif mtype == 0x51 and len(payload) >= 3:
                    us = (payload[0] << 16) | (payload[1] << 8) | payload[2]
                    if us > 0:
                        tr.tempos.append((tick, us))
                elif mtype == 0x58 and len(payload) >= 2:
                    tr.time_signatures.append((tick, payload[0], 2 ** payload[1]))
                elif mtype == 0x59 and len(payload) >= 2:
                    sf = payload[0] - 256 if payload[0] > 127 else payload[0]
                    tr.key_signatures.append((tick, sf, bool(payload[1])))
                continue
            if b in (0xF0, 0xF7):  # sysex / escape
                length, pos = _read_vlq(data, pos + 1)
                pos += length
                continue
            if b & 0x80:
                status = b
                pos += 1
            elif status is None:  # stray data byte without running status
                pos += 1
                continue
            if status >= 0xF0:  # system common/real-time bytes should not appear in files
                pos += {0xF1: 1, 0xF2: 2, 0xF3: 1}.get(status, 0)
                status = None
                continue
            kind, ch = status & 0xF0, status & 0x0F
            nbytes = 1 if kind in (0xC0, 0xD0) else 2
            args = data[pos:pos + nbytes]
            pos += nbytes
            if len(args) < nbytes:
                break
            if kind == 0x90 and args[1] > 0:
                open_notes.setdefault((ch, args[0]), []).append((tick, args[1]))
            elif kind == 0x80 or kind == 0x90:
                stack = open_notes.get((ch, args[0]))
                if stack:
                    start, vel = stack.pop(0)
                    tr.notes.append(RawNote(start, tick, args[0], vel, ch))
    except IndexError:  # truncated track: keep what we have
        pass
    tr.end_tick = tick
    for (ch, pitch), stack in open_notes.items():
        for start, vel in stack:
            tr.notes.append(RawNote(start, max(tick, start), pitch, vel, ch))
    tr.notes = [x for x in tr.notes if x.end > x.start]
    tr.notes.sort(key=lambda x: (x.start, x.pitch))
    return tr


def parse_smf(data: bytes) -> tuple[int, int, list[RawTrack]]:
    """Parse SMF bytes into ``(format, ticks_per_quarter, tracks)``.

    SMPTE time division is converted to an equivalent PPQ at 120 bpm.
    """
    start = data.find(b"MThd")
    if start < 0:
        raise MidiError("not a Standard MIDI File (missing MThd header)")
    hlen = struct.unpack(">I", data[start + 4:start + 8])[0]
    if hlen < 6:
        raise MidiError("invalid MThd header")
    fmt, ntrks, division = struct.unpack(">HHH", data[start + 8:start + 14])
    if division & 0x8000:
        fps = 256 - (division >> 8)
        tpf = division & 0xFF
        ppq = max(1, fps * tpf // 2)
    else:
        ppq = division or PPQ
    pos = start + 8 + hlen
    tracks: list[RawTrack] = []
    while pos + 8 <= len(data) and len(tracks) < max(ntrks, 1) + 64:
        cid = data[pos:pos + 4]
        size = struct.unpack(">I", data[pos + 4:pos + 8])[0]
        body = data[pos + 8:pos + 8 + size]
        pos += 8 + size
        if cid == b"MTrk":
            tracks.append(_parse_track(body))
    return fmt, ppq, tracks


# ---------------------------------------------------------------------------
# raw data -> Project
# ---------------------------------------------------------------------------
def monophonic(notes: list[RawNote]) -> list[RawNote]:
    """Trim overlapping notes into a single line.

    Notes starting at the same tick keep only the highest; otherwise a note
    that starts while the previous one still sounds truncates it.
    """
    out: list[RawNote] = []
    for x in sorted(notes, key=lambda n: (n.start, -n.pitch)):
        if out and out[-1].start == x.start:
            continue
        if out and out[-1].end > x.start:
            out[-1].end = x.start
        out.append(RawNote(x.start, x.end, x.pitch, x.velocity, x.channel))
    return [x for x in out if x.end > x.start]


def _clean_lyric(text: str) -> str:
    s = text.replace("\x00", "").strip()
    while s[:1] in ("/", "\\"):
        s = s[1:].lstrip()
    return s


def _assign_lyrics(notes: list[RawNote], events: list[tuple[int, str]], tol: int) -> list[str | None]:
    """Attach lyric events to notes: exact tick matches first, then nearest within ``tol``."""
    out: list[str | None] = [None] * len(notes)
    events = sorted(events, key=lambda e: e[0])
    ticks = [e[0] for e in events]
    used = [False] * len(events)
    for i, x in enumerate(notes):
        lo = bisect.bisect_left(ticks, x.start)
        hi = bisect.bisect_right(ticks, x.start)
        texts = [events[k][1] for k in range(lo, hi) if not used[k]]
        if texts:
            for k in range(lo, hi):
                used[k] = True
            out[i] = "".join(texts)
    for i, x in enumerate(notes):
        if out[i] is not None:
            continue
        lo = bisect.bisect_left(ticks, x.start - tol)
        hi = bisect.bisect_right(ticks, x.start + tol)
        best = None
        for k in range(lo, hi):
            if not used[k] and (best is None or abs(ticks[k] - x.start) < abs(ticks[best] - x.start)):
                best = k
        if best is not None:
            used[best] = True
            out[i] = events[best][1]
    return out


def _join_hyphenated(lyrics: list[str]) -> list[str]:
    """``["twin-", "kle", "lit-", "tle"]`` -> ``["twinkle", "+", "little", "+"]``."""
    out = list(lyrics)
    i = 0
    while i < len(out):
        s = out[i]
        if len(s) > 1 and s.endswith("-") and s.strip("-"):
            parts = [s.rstrip("-")]
            j = i + 1
            while j < len(out):
                t = out[j]
                if t in ("-", "+"):  # melisma / explicit continuation inside the word
                    j += 1
                    continue
                parts.append(t.strip("-") if t.strip("-") else t)
                out[j] = "+"
                j += 1
                if not t.endswith("-"):
                    break
            out[i] = "".join(parts)
            i = j
        else:
            i += 1
    return out


def _has_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def loads_midi(data: bytes, title: str | None = None) -> Project:
    """Build a :class:`Project` from Standard MIDI File bytes."""
    fmt, ppq, tracks = parse_smf(data)

    # --- global maps -----------------------------------------------------
    tempo_events = sorted((e for t in tracks for e in t.tempos), key=lambda e: e[0])
    points: dict[int, float] = {}
    for tick, us in tempo_events:
        points[tick] = 60_000_000.0 / us
    if not points or min(points) > 0:
        points.setdefault(0, 120.0)
    # microsecond resolution makes e.g. 140 bpm read back as 140.00014: keep 3 decimals
    tempo = TempoMap([TempoPoint(tick / ppq, round(bpm, 3)) for tick, bpm in sorted(points.items())])

    sigs = sorted((e for t in tracks for e in t.time_signatures), key=lambda e: e[0])
    keys = sorted((e for t in tracks for e in t.key_signatures), key=lambda e: e[0])

    project = Project(title=title or "Untitled", tempo=tempo)
    if sigs:
        project.time_signature = (int(sigs[0][1]), int(sigs[0][2]))
    if keys:
        project.key = fifths_to_key(keys[0][1], keys[0][2])
    meta: dict = {"format": fmt, "ppq": ppq}
    if len(sigs) > 1:
        meta["time_signatures"] = [{"beat": round(t / ppq, 6), "numerator": a, "denominator": b} for t, a, b in sigs]
    markers = [(round(t / ppq, 6), s) for tr in tracks for t, s in tr.markers if s.strip()]
    if markers:
        meta["markers"] = [{"beat": b, "text": s} for b, s in sorted(markers)]

    if tracks and not tracks[0].notes and tracks[0].name and fmt != 0:
        project.title = tracks[0].name
    elif fmt == 0 and tracks and tracks[0].name:
        project.title = tracks[0].name

    # --- lyric sources ---------------------------------------------------
    use_lyrics = any(t.lyrics for t in tracks)

    def lyric_events(t: RawTrack) -> list[tuple[int, str]]:
        if use_lyrics:
            return t.lyrics
        return [(tick, s) for tick, s in t.texts if not s.lstrip().startswith("@")]

    orphan_lyrics = [e for t in tracks if not t.notes for e in lyric_events(t)]

    # --- note units: one per track (format 1/2) or per channel (format 0) ---
    units: list[tuple[str, list[RawNote], list[tuple[int, str]]]] = []
    for t in tracks:
        if not t.notes:
            continue
        events = lyric_events(t) or orphan_lyrics
        base = t.name or t.instrument or f"Track {len(units) + 1}"
        if fmt == 0:
            channels = sorted({x.channel for x in t.notes})
            for ch in channels:
                name = base if len(channels) == 1 else f"{base} ch{ch + 1}"
                units.append((name, [x for x in t.notes if x.channel == ch], events))
        else:
            units.append((base, list(t.notes), events))

    def is_drum(notes: list[RawNote]) -> bool:
        return all(x.channel == DRUM_CHANNEL for x in notes)

    if any(not is_drum(u[1]) for u in units):
        units = [(nm, [x for x in ns if x.channel != DRUM_CHANNEL], ev) for nm, ns, ev in units if not is_drum(ns)]

    tol = max(1, ppq // 16)
    for name, raw_notes, events in units:
        line = monophonic(raw_notes)
        if not line:
            continue
        lyrics = [_clean_lyric(s) if s is not None else "" for s in _assign_lyrics(line, events, tol)]
        lyrics = _join_hyphenated(lyrics)
        med = median(x.velocity for x in line) or 100
        track = project.add_track(name)
        for x, lyric in zip(line, lyrics):
            attrs = {}
            ratio = x.velocity / med
            if abs(ratio - 1.0) > 0.05:
                attrs["velocity"] = round(min(2.0, ratio), 3)
            track.notes.append(Note(x.start / ppq, (x.end - x.start) / ppq, x.pitch, lyric or "la", attrs=attrs))
        if any(_has_kana(n.lyric) for n in track.notes):
            track.language = "ja"
        track.sort()
    project.metadata["midi"] = meta
    return project


def read_midi(path: str | Path) -> Project:
    """Read a ``.mid`` / ``.midi`` file into a :class:`Project`."""
    path = Path(path)
    return loads_midi(path.read_bytes(), title=path.stem)


# ---------------------------------------------------------------------------
# Project -> SMF bytes
# ---------------------------------------------------------------------------
def _vlq(value: int) -> bytes:
    value = max(0, int(value))
    out = [value & 0x7F]
    value >>= 7
    while value:
        out.append(0x80 | (value & 0x7F))
        value >>= 7
    return bytes(reversed(out))


def _meta(mtype: int, payload: bytes) -> bytes:
    return bytes([0xFF, mtype]) + _vlq(len(payload)) + payload


def _chunk(events: list[tuple[int, int, int, bytes]]) -> bytes:
    """Encode ``(tick, order, seq, event bytes)`` into an ``MTrk`` chunk."""
    events = sorted(events, key=lambda e: (e[0], e[1], e[2]))
    body = bytearray()
    last = 0
    for tick, _, _, ev in events:
        body += _vlq(tick - last) + ev
        last = tick
    body += _vlq(0) + _meta(0x2F, b"")
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def dumps_midi(project: Project) -> bytes:
    """Serialise ``project`` as an SMF format-1 file (bytes) at 480 PPQ."""
    def tk(beat: float) -> int:
        return max(0, int(round(beat * PPQ)))

    conductor: list[tuple[int, int, int, bytes]] = [(0, 0, 0, _meta(0x03, project.title.encode("utf-8")))]
    num, den = project.time_signature
    dd = max(1, int(den)).bit_length() - 1  # denominator as a power of two
    conductor.append((0, 0, 1, _meta(0x58, bytes([int(num) & 0xFF, dd, 24, 8]))))
    k = key_to_fifths(project.key)
    if k is not None:
        conductor.append((0, 0, 2, _meta(0x59, bytes([k[0] & 0xFF, 1 if k[1] else 0]))))
    for i, p in enumerate(project.tempo.points):
        us = max(1, min(0xFFFFFF, int(round(60_000_000.0 / p.bpm))))
        conductor.append((tk(p.beat), 1, i, _meta(0x51, us.to_bytes(3, "big"))))
    chunks = [_chunk(conductor)]

    channels = [c for c in range(16) if c != DRUM_CHANNEL]
    for ti, track in enumerate(project.tracks):
        ch = channels[ti % len(channels)]
        events: list[tuple[int, int, int, bytes]] = [(0, 0, 0, _meta(0x03, track.name.encode("utf-8")))]
        for i, n in enumerate(sorted(track.notes, key=lambda n: (n.onset, n.pitch))):
            on = tk(n.onset)
            off = max(on + 1, tk(n.end))
            pitch = max(0, min(127, int(n.pitch)))
            vel = max(1, min(127, int(round(float(n.attrs.get("velocity") or 1.0) * 100))))
            events.append((on, 2, 2 * i, _meta(0x05, (n.lyric or "la").encode("utf-8"))))
            events.append((on, 2, 2 * i + 1, bytes([0x90 | ch, pitch, vel])))
            events.append((off, 1, i, bytes([0x80 | ch, pitch, 0])))
        chunks.append(_chunk(events))
    header = b"MThd" + struct.pack(">IHHH", 6, 1, len(chunks), PPQ)
    return header + b"".join(chunks)


def write_midi(project: Project, path: str | Path) -> Path:
    """Write ``project`` to ``path`` as a Standard MIDI File."""
    path = Path(path)
    path.write_bytes(dumps_midi(project))
    return path
