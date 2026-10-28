"""Musical time handling: tempo maps, beat <-> seconds conversion, note names.

All positions inside a Synth2 project are expressed in *beats* (quarter notes)
as floats.  This keeps projects easy for both humans and AI models to write:
``onset=4.0, duration=0.5`` is "an eighth note on the downbeat of bar 2" in 4/4.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_NOTE_RE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]{0,2})(-?\d+)\s*$")
_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_midi(name: str | int | float) -> int:
    """Convert a note name such as ``"C4"``, ``"F#3"`` or ``"Bb5"`` to a MIDI number.

    Integers (or numeric strings) are passed through. C4 = 60 (middle C).
    """
    if isinstance(name, (int, float)):
        return int(round(name))
    s = str(name).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    m = _NOTE_RE.match(s)
    if not m:
        raise ValueError(f"invalid note name: {name!r}")
    letter, acc, octave = m.groups()
    value = _SEMITONE[letter.upper()]
    for ch in acc:
        value += 1 if ch in "#♯" else -1
    return value + (int(octave) + 1) * 12


def midi_to_note(midi: int) -> str:
    midi = int(round(midi))
    return f"{NOTE_NAMES[midi % 12]}{midi // 12 - 1}"


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def hz_to_midi(hz):
    import numpy as np

    hz = np.asarray(hz, dtype=float)
    with np.errstate(divide="ignore"):
        return 69.0 + 12.0 * np.log2(np.maximum(hz, 1e-9) / 440.0)


@dataclass
class TempoPoint:
    beat: float
    bpm: float


@dataclass
class TempoMap:
    """Piecewise-constant tempo map (tempo changes take effect at their beat)."""

    points: list[TempoPoint] = field(default_factory=lambda: [TempoPoint(0.0, 120.0)])

    def __post_init__(self):
        self._rebuild()

    def _rebuild(self):
        pts = sorted(self.points, key=lambda p: p.beat)
        if not pts or pts[0].beat > 0:
            pts.insert(0, TempoPoint(0.0, pts[0].bpm if pts else 120.0))
        self.points = pts
        self._beats = [p.beat for p in pts]
        self._secs = [0.0]
        for a, b in zip(pts, pts[1:]):
            self._secs.append(self._secs[-1] + (b.beat - a.beat) * 60.0 / a.bpm)

    def set(self, beat: float, bpm: float):
        if bpm <= 0:
            raise ValueError("bpm must be positive")
        self.points = [p for p in self.points if abs(p.beat - beat) > 1e-9]
        self.points.append(TempoPoint(float(beat), float(bpm)))
        self._rebuild()

    def bpm_at(self, beat: float) -> float:
        i = max(0, bisect.bisect_right(self._beats, beat) - 1)
        return self.points[i].bpm

    def beat_to_sec(self, beat: float) -> float:
        i = max(0, bisect.bisect_right(self._beats, beat) - 1)
        p = self.points[i]
        return self._secs[i] + (beat - p.beat) * 60.0 / p.bpm

    def sec_to_beat(self, sec: float) -> float:
        i = max(0, bisect.bisect_right(self._secs, sec) - 1)
        p = self.points[i]
        return p.beat + (sec - self._secs[i]) * p.bpm / 60.0

    def to_list(self):
        return [{"beat": p.beat, "bpm": p.bpm} for p in self.points]

    @classmethod
    def from_list(cls, data):
        if not data:
            return cls()
        return cls([TempoPoint(float(d["beat"]), float(d["bpm"])) for d in data])


DURATION_NAMES = {
    "w": 4.0, "whole": 4.0,
    "h": 2.0, "half": 2.0,
    "q": 1.0, "quarter": 1.0,
    "e": 0.5, "eighth": 0.5, "8": 0.5,
    "s": 0.25, "sixteenth": 0.25, "16": 0.25,
    "t": 0.125, "32": 0.125,
}


def parse_duration(token: str | float | int) -> float:
    """Parse a duration in beats.

    Accepts numbers (``1.5``), fractions (``3/2``) and note values: ``w h q e s t``
    with optional dots (``q.`` = 1.5, ``q..`` = 1.75), triplet suffix ``3``
    (``e3`` = 1/3 beat) and ties with ``+`` (``h+e``).
    """
    if isinstance(token, (int, float)):
        return float(token)
    tok = str(token).strip().lower()
    if "+" in tok:
        return sum(parse_duration(t) for t in tok.split("+"))
    try:
        if "/" in tok:
            a, b = tok.split("/", 1)
            return float(a) / float(b)
        return float(tok)
    except ValueError:
        pass
    dots = len(tok) - len(tok.rstrip("."))
    tok = tok.rstrip(".")
    triplet = tok.endswith("3") and tok[:-1] in DURATION_NAMES
    if triplet:
        tok = tok[:-1]
    if tok not in DURATION_NAMES:
        raise ValueError(f"invalid duration: {token!r}")
    base = DURATION_NAMES[tok]
    value = base * (2.0 - 0.5 ** dots)
    if triplet:
        value *= 2.0 / 3.0
    return value
