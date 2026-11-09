"""Music theory for the arranger: chord symbols, chord-relative degree tokens and voicings.

Chord symbols (``"Am"``, ``"Am/G"``, ``"F#dim"``, ``"E7"``, ``"Cmaj7"``, ``"Dsus4"``,
``"G5"``, ``"Bb"``) are parsed into a :class:`Chord` with a root pitch class, the
chord's intervals and an optional slash bass.  Patterns address pitches relative to
the chord that is playing with *degree tokens* (``R``, ``1``, ``3``, ``5``, ``7``,
``8``, ``b7``, ``5'``, ``R,`` ...), so one bass or arp pattern follows any chord
progression.  Chord tones come from the chord itself; other degrees (2, 4, 6 and a
7th the chord does not contain) come from the current key's scale, so a ``7`` over
``F`` in A minor is E, over ``G`` it is F and over ``E`` it is D.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..timing import note_to_midi

_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# quality suffix -> intervals (semitones above the root). Longest suffixes are matched first.
QUALITIES: dict[str, tuple[int, ...]] = {
    "": (0, 4, 7), "maj": (0, 4, 7), "M": (0, 4, 7),
    "m": (0, 3, 7), "min": (0, 3, 7), "-": (0, 3, 7),
    "dim": (0, 3, 6), "o": (0, 3, 6), "°": (0, 3, 6),
    "aug": (0, 4, 8), "+": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "sus": (0, 5, 7),
    "5": (0, 7),
    "6": (0, 4, 7, 9), "m6": (0, 3, 7, 9),
    "7": (0, 4, 7, 10), "dom7": (0, 4, 7, 10),
    "maj7": (0, 4, 7, 11), "M7": (0, 4, 7, 11), "Δ7": (0, 4, 7, 11), "Δ": (0, 4, 7, 11),
    "m7": (0, 3, 7, 10), "min7": (0, 3, 7, 10), "-7": (0, 3, 7, 10),
    "mmaj7": (0, 3, 7, 11), "mM7": (0, 3, 7, 11),
    "m7b5": (0, 3, 6, 10), "ø": (0, 3, 6, 10), "ø7": (0, 3, 6, 10),
    "dim7": (0, 3, 6, 9), "o7": (0, 3, 6, 9), "°7": (0, 3, 6, 9),
    "7sus4": (0, 5, 7, 10), "7sus": (0, 5, 7, 10),
    "add9": (0, 4, 7, 14), "madd9": (0, 3, 7, 14), "add2": (0, 2, 4, 7),
    "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14), "m9": (0, 3, 7, 10, 14),
    "11": (0, 4, 7, 10, 14, 17), "m11": (0, 3, 7, 10, 14, 17),
    "13": (0, 4, 7, 10, 14, 21),
    "7b9": (0, 4, 7, 10, 13), "7#9": (0, 4, 7, 10, 15),
    "aug7": (0, 4, 8, 10), "+7": (0, 4, 8, 10),
}
_QUAL_ORDER = sorted(QUALITIES, key=len, reverse=True)
_CHORD_RE = re.compile(r"^([A-Ga-g])([#b♯♭]?)(.*?)(?:/([A-Ga-g])([#b♯♭]?))?$")
NO_CHORD = ("N.C.", "NC", "N.C", "n.c.", "nc", "x", "X")

MAJOR = (0, 2, 4, 5, 7, 9, 11)
MINOR = (0, 2, 3, 5, 7, 8, 10)


def _pc(letter: str, acc: str) -> int:
    return (_LETTER_PC[letter.upper()] + (1 if acc in ("#", "♯") else -1 if acc in ("b", "♭") else 0)) % 12


@dataclass(frozen=True)
class Chord:
    """A parsed chord symbol. ``root`` and ``bass`` are pitch classes 0..11."""

    name: str
    root: int
    intervals: tuple[int, ...]
    bass: int

    @property
    def is_minor(self) -> bool:
        return 3 in self.intervals and 4 not in self.intervals

    @property
    def pcs(self) -> list[int]:
        return [(self.root + i) % 12 for i in self.intervals]

    def tone(self, which: int) -> int | None:
        """Interval of chord tone 3, 5 or 7 when the chord has one."""
        iv = set(self.intervals)
        if which == 3:
            for c in (4, 3, 5, 2):  # sus chords: the sus tone stands in for the third
                if c in iv:
                    return c
        elif which == 5:
            for c in (7, 6, 8):
                if c in iv:
                    return c
        elif which == 7:
            for c in (10, 11, 9):
                if c in iv:
                    return c
        return None


def parse_chord(sym: str) -> Chord | None:
    """Parse a chord symbol. Returns ``None`` for "no chord" (``N.C.``). Raises ValueError when invalid."""
    s = str(sym).strip()
    if s in NO_CHORD:
        return None
    m = _CHORD_RE.match(s)
    if not m:
        raise ValueError(f"invalid chord {sym!r}: expected e.g. 'Am', 'F#dim', 'E7', 'Cmaj7', 'Am/G', 'G5'")
    letter, acc, qual, bl, bacc = m.groups()
    q = None
    for cand in _QUAL_ORDER:
        if qual == cand:
            q = cand
            break
    if q is None:
        raise ValueError(f"unknown chord quality {qual!r} in {sym!r}; known: "
                         f"{', '.join(repr(k) for k in sorted(set(QUALITIES), key=len) if k)}")
    root = _pc(letter, acc)
    bass = _pc(bl, bacc or "") if bl else root
    return Chord(s, root, QUALITIES[q], bass)


def scale_for(key: str | None) -> tuple[int, tuple[int, ...]]:
    """(tonic pc, scale intervals) of a key like 'A minor'; defaults to C major when unknown."""
    try:
        from ..ai.composer import MODES, parse_key

        k = parse_key(key or "C major")
        scale = MODES[k.mode]
        if len(scale) != 7:  # pentatonic/blues keys: fall back to the parent mode
            scale = MINOR if 3 in scale else MAJOR
        return k.tonic, tuple(scale)
    except (ValueError, ImportError):
        return 0, MAJOR


# ------------------------------------------------------------------------------- degree tokens

# token -> (degree number, accidental semitones, octave shift)
_DEG_RE = re.compile(r"^([b#]?)(R|r|1|2|3|4|5|6|7|8|9|10|11|12|13)([',]*)$")
# single-character step aliases used in compact step strings
COMPACT_ALIASES = {"L": "R,", "H": "5'", "F": "5,", "T": "3'", "O": "8'"}


def degree_semitones(token: str, chord: Chord, key: tuple[int, tuple[int, ...]]) -> int:
    """Semitones from the chord's *bass/root reference* for a degree token.

    ``R`` is the chord's bass note (the slash bass for ``Am/G``), ``1`` the chord root.
    Every other degree is measured from the chord root: 3/5/7 use the chord's own
    third/fifth/seventh when it has one, otherwise (and for 2, 4, 6, 9, 11, 13) the
    diatonic note of the key that many scale steps above the root.  ``8`` is the root an
    octave up, ``9`` = 2 + octave, ``10`` = 3 + octave, ``12`` = 5 + octave.  ``b``/``#``
    alter by a semitone; each ``'`` raises and each ``,`` lowers an octave.

    Returns the pitch offset relative to the chord root placed at 0 (``R`` may be negative,
    the slash bass below the root).
    """
    t = COMPACT_ALIASES.get(token, token)
    m = _DEG_RE.match(t)
    if not m:
        raise ValueError(f"invalid degree token {token!r}: use R 1 2 3 4 5 6 7 8 9 10 11 12 13 with optional "
                         f"b/# prefix and ' (octave up) or , (octave down) suffixes, e.g. 'b7', \"5'\", 'R,'")
    acc, deg, octs = m.groups()
    shift = 12 * (octs.count("'") - octs.count(","))
    alter = 1 if acc == "#" else -1 if acc == "b" else 0
    if deg in ("R", "r"):
        down = (chord.root - chord.bass) % 12
        return -down + shift + alter
    n = int(deg)
    octave = 0
    while n > 7:
        n -= 7
        octave += 12
    if n == 1:
        base = 0
    elif n in (3, 5, 7) and chord.tone(n) is not None:
        base = chord.tone(n) % 12
    else:
        base = _diatonic(chord.root, n, key, chord)
    return base + octave + shift + alter


def _diatonic(root: int, n: int, key: tuple[int, tuple[int, ...]], chord: Chord) -> int:
    """Interval above ``root`` of the scale note ``n-1`` steps up (n = 2..7) in the key."""
    tonic, scale = key
    pcs = [(tonic + i) % 12 for i in scale]
    if root in pcs:
        idx = pcs.index(root)
        target = pcs[(idx + n - 1) % 7]
        return (target - root) % 12
    # chord root outside the key (e.g. Bb in A minor): use the chord's own quality
    return (MINOR if chord.is_minor else MAJOR)[n - 1]


def place_in_register(pc: int, low: int) -> int:
    """Lowest MIDI pitch >= ``low`` with pitch class ``pc``."""
    return low + ((pc - low) % 12)


def parse_pitch(p) -> int:
    """MIDI number from a note name ('A4'), a MIDI int or a numeric string."""
    return note_to_midi(p)


# ------------------------------------------------------------------------------------ voicings

VOICINGS = ("triad", "close", "power", "power8", "open", "full", "shell", "seventh", "octave", "root", "cluster", "spread")


def voice_chord(chord: Chord, voicing: str, center: int, low: int, prev: list[int] | None = None,
                key: tuple[int, tuple[int, ...]] | None = None) -> list[int]:
    """MIDI pitches for ``chord``.

    voicing: ``triad``/``close`` (the chord as written, at most 4 notes, in close position; the
    inversion nearest ``center`` and the previous chord, for smooth voice leading),
    ``seventh`` (close position with the 7th from key/chord), ``power`` (root + fifth, root
    placed at or above ``low``), ``power8`` (root, fifth, octave), ``open`` (root low, then
    fifth + third an octave up: a wide pop/rock spread), ``full`` (guitar-like six-string
    spread: root, 5, 8, 10, 12, 15), ``shell`` (root + 3 + 7), ``octave`` (root + octave),
    ``root`` (root only), ``cluster`` (root, 2, 3, 5 close), ``spread`` (root low, triad
    close position near ``center`` on top).
    """
    root_low = place_in_register(chord.root, low)
    iv3 = chord.tone(3)
    iv5 = chord.tone(5) if chord.tone(5) is not None else 7
    k = key or (0, MAJOR)
    if voicing == "power":
        return [root_low, root_low + iv5]
    if voicing == "power8":
        return [root_low, root_low + iv5, root_low + 12]
    if voicing == "octave":
        return [root_low, root_low + 12]
    if voicing == "root":
        return [root_low]
    if voicing == "open":
        out = [root_low, root_low + iv5 + 12]
        if iv3 is not None:
            out.append(root_low + iv3 + 12)
        return sorted(out)
    if voicing == "full":
        out = [root_low, root_low + iv5, root_low + 12]
        if iv3 is not None:
            out.append(root_low + 12 + iv3)
        out += [root_low + 12 + iv5, root_low + 24]
        return sorted(out)
    if voicing == "shell":
        iv7 = degree_semitones("7", chord, k) % 12
        return [root_low] + ([root_low + iv3] if iv3 is not None else []) + [root_low + iv7]
    if voicing == "cluster":
        iv2 = degree_semitones("2", chord, k) % 12
        return sorted({root_low, root_low + iv2, root_low + (iv3 if iv3 is not None else 5), root_low + iv5})
    # close-position voicings: choose the inversion nearest the centre / previous chord
    # the chord as written (7ths and extensions included, at most 4 notes), close position
    pcs = [0] + ([iv3] if iv3 is not None else []) + [iv5]
    pcs += [i for i in chord.intervals if i % 12 not in (0, iv3, iv5)][: 4 - len(pcs)]
    if voicing == "seventh" and chord.tone(7) is None:
        pcs.append(degree_semitones("7", chord, k) % 12)
    pcs = sorted({p % 12 for p in pcs})
    cands = []
    base_pcs = [(chord.root + p) % 12 for p in pcs]
    for inv in range(len(base_pcs)):
        order = base_pcs[inv:] + base_pcs[:inv]
        for start_oct in range(center - 18, center + 7):
            if start_oct % 12 != order[0]:
                continue
            notes = [start_oct]
            for pc in order[1:]:
                nxt = notes[-1] + ((pc - notes[-1]) % 12 or 12)
                notes.append(nxt)
            cands.append(notes)
    if not cands:
        return [root_low]

    def cost(ns):
        mean = sum(ns) / len(ns)
        c = abs(mean - center)
        if prev:
            pm = sum(prev) / len(prev)
            c = 0.5 * c + abs(mean - pm) + 0.15 * sum(min(abs(n - p) for p in prev) for n in ns)
        if min(ns) < low:
            c += 24
        return c

    best = min(cands, key=cost)
    if voicing == "spread":
        return [root_low] + [p for p in best if p > root_low]
    return best
