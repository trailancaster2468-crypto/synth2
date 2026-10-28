"""Pure music-theory and editing helpers behind the AI tools.

Everything here works on :class:`~synth2.project.Note` lists and returns *new*
note objects (the inputs are never mutated), so the tool layer can apply the
results atomically and keep undo snapshots simple.

* keys and scales: :func:`parse_key`, :func:`scale_degrees`, :func:`scale_degree`,
  :func:`in_scale`, :func:`snap_to_scale`, :func:`step_index`, :func:`step_to_pitch`,
  :func:`diatonic_shift`
* note transforms: :func:`transpose`, :func:`harmonize`, :func:`quantize`,
  :func:`resolve_overlaps`, :func:`double`
* lyrics: :func:`lyric_tokens`, :func:`fit_lyrics`

Pitches are MIDI note numbers, times are beats.
"""
from __future__ import annotations

import copy
import re
from typing import Iterable

from ..project import Note, _new_id
from ..timing import NOTE_NAMES, midi_to_note, note_to_midi, parse_duration

# --------------------------------------------------------------------------- keys

MODES: dict[str, tuple[int, ...]] = {
    "major": (0, 2, 4, 5, 7, 9, 11),
    "minor": (0, 2, 3, 5, 7, 8, 10),
    "harmonic minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic minor": (0, 2, 3, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "major pentatonic": (0, 2, 4, 7, 9),
    "minor pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
}
MODE_ALIASES = {
    "": "major", "maj": "major", "ionian": "major", "major scale": "major",
    "m": "minor", "min": "minor", "aeolian": "minor", "natural minor": "minor", "minor scale": "minor",
    "harmonic": "harmonic minor", "harm minor": "harmonic minor", "harmonic_minor": "harmonic minor",
    "melodic": "melodic minor", "mel minor": "melodic minor", "melodic_minor": "melodic minor",
    "pentatonic": "major pentatonic", "major_pentatonic": "major pentatonic",
    "minor_pentatonic": "minor pentatonic", "maj pentatonic": "major pentatonic",
    "min pentatonic": "minor pentatonic", "dor": "dorian", "phr": "phrygian", "lyd": "lydian",
    "mix": "mixolydian", "loc": "locrian",
}
_KEY_RE = re.compile(r"^\s*([A-Ga-g])([#b♯♭]?)\s*(.*?)\s*$")
_LETTER_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_LETTERS = "CDEFGAB"
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


class KeyInfo(tuple):
    """``(tonic_pc, mode, scale_pcs)`` - a plain 3-tuple with named accessors and the spelled tonic."""

    def __new__(cls, tonic: int, mode: str, scale: tuple[int, ...], tonic_name: str = "C"):
        obj = super().__new__(cls, (tonic, mode, tuple(scale)))
        obj.tonic_name = tonic_name
        return obj

    def __getnewargs__(self):
        return (self[0], self[1], self[2], self.tonic_name)

    @property
    def tonic(self) -> int:  # pitch class 0..11 (C = 0)
        return self[0]

    @property
    def mode(self) -> str:  # normalised mode name, e.g. "major", "minor", "dorian"
        return self[1]

    @property
    def scale(self) -> tuple[int, ...]:  # absolute pitch classes of the scale, starting at the tonic
        return self[2]

    @property
    def intervals(self) -> tuple[int, ...]:
        return MODES[self.mode]

    @property
    def name(self) -> str:
        return f"{self.tonic_name} {self.mode}"


def parse_key(key: str | KeyInfo | None) -> KeyInfo:
    """Parse ``"A minor"``, ``"F# major"``, ``"Bbm"``, ``"D dorian"``, ``"C"``.

    Returns ``(tonic_pc, mode, scale_pcs)`` as a :class:`KeyInfo` named tuple
    (``scale_pcs`` are absolute pitch classes, e.g. A minor -> (9, 11, 0, 2, 4, 5, 7)).
    """
    if isinstance(key, KeyInfo):
        return key
    if key is None or not str(key).strip():
        key = "C major"
    m = _KEY_RE.match(str(key))
    if not m:
        raise ValueError(f"invalid key {key!r}: expected a tonic and mode like 'C major', 'A minor', "
                         f"'F# minor', 'Bb major' or 'D dorian'")
    letter, acc, mode_txt = m.groups()
    mode_txt = re.sub(r"[\s_\-]+", " ", mode_txt.strip())
    # "Am" style: lower-case "m" means minor; "M" means major
    if mode_txt == "M":
        mode_txt = "major"
    mode_txt = mode_txt.lower()
    mode = MODE_ALIASES.get(mode_txt, mode_txt)
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode_txt!r} in key {key!r}; valid modes: {', '.join(MODES)}")
    tonic = (_LETTER_PC[letter.upper()] + (1 if acc in "#♯" and acc else -1 if acc else 0)) % 12
    acc_txt = {"♯": "#", "♭": "b"}.get(acc, acc)
    tonic_name = letter.upper() + acc_txt
    scale = tuple((tonic + iv) % 12 for iv in MODES[mode])
    return KeyInfo(tonic, mode, scale, tonic_name)


def scale_degrees(key: str | KeyInfo) -> list[str]:
    """Spelled note names of the scale, e.g. ``"F major"`` -> ``['F', 'G', 'A', 'Bb', 'C', 'D', 'E']``."""
    k = parse_key(key)
    if len(k.scale) == 7:
        start = _LETTERS.index(k.tonic_name[0])
        out = []
        for i, pc in enumerate(k.scale):
            letter = _LETTERS[(start + i) % 7]
            diff = (pc - _LETTER_PC[letter] + 6) % 12 - 6
            if abs(diff) <= 2:
                out.append(letter + ("#" * diff if diff > 0 else "b" * -diff))
            else:  # pragma: no cover - cannot happen for 7-note modes
                out.append(NOTE_NAMES[pc])
        return out
    flats = "b" in k.tonic_name or k.tonic_name == "F" or (
        k.mode in ("minor pentatonic", "blues") and k.tonic_name in ("C", "G", "D"))
    names = _FLAT_NAMES if flats else NOTE_NAMES
    return [names[pc] for pc in k.scale]


def _rel(pitch: int, k: KeyInfo) -> tuple[int, int]:
    """(octave, pitch class relative to tonic)."""
    rel = int(pitch) - k.tonic
    return rel // 12, rel % 12


def step_index(pitch: int, key: str | KeyInfo) -> tuple[int, int]:
    """Absolute diatonic step of the scale tone at or below ``pitch``, plus the chromatic remainder.

    ``step = octave * len(scale) + degree_index``; ``step_to_pitch(step) + remainder == pitch``.
    """
    k = parse_key(key)
    ivs = k.intervals
    octave, pc = _rel(pitch, k)
    degree = max(i for i, iv in enumerate(ivs) if iv <= pc)
    return octave * len(ivs) + degree, pc - ivs[degree]


def step_to_pitch(step: int, key: str | KeyInfo) -> int:
    """Inverse of :func:`step_index` for scale tones."""
    k = parse_key(key)
    ivs = k.intervals
    octave, degree = divmod(int(step), len(ivs))
    return k.tonic + 12 * octave + ivs[degree]


def scale_degree(pitch: int | str, key: str | KeyInfo) -> int | None:
    """1-based scale degree of ``pitch`` in ``key``, or ``None`` when it is chromatic."""
    k = parse_key(key)
    pc = note_to_midi(pitch) % 12
    return k.scale.index(pc) + 1 if pc in k.scale else None


def in_scale(pitch: int | str, key: str | KeyInfo) -> bool:
    return scale_degree(pitch, key) is not None


def snap_to_scale(pitch: int | str, key: str | KeyInfo, direction: str = "nearest") -> int:
    """Move a chromatic pitch to a scale tone (``nearest`` ties go down, or ``up`` / ``down``)."""
    k = parse_key(key)
    p = note_to_midi(pitch)
    if p % 12 in k.scale:
        return p
    if direction not in ("nearest", "up", "down"):
        raise ValueError(f"direction must be 'nearest', 'up' or 'down', got {direction!r}")
    for d in range(1, 12):
        if direction in ("nearest", "down") and (p - d) % 12 in k.scale:
            return p - d
        if direction in ("nearest", "up") and (p + d) % 12 in k.scale:
            return p + d
    return p  # pragma: no cover


_MAJOR_FLATS = {3, 8, 10}
_MINOR_FLATS = {1}
_FLAT_SIDE = {"major": _MAJOR_FLATS, "lydian": _MAJOR_FLATS, "mixolydian": _MAJOR_FLATS,
              "major pentatonic": _MAJOR_FLATS, "minor": _MINOR_FLATS, "dorian": _MINOR_FLATS,
              "harmonic minor": _MINOR_FLATS, "melodic minor": _MINOR_FLATS}


def _diatonic_intervals(k: KeyInfo, steps: int) -> set[int]:
    """All semitone sizes that a diatonic interval of ``steps`` scale steps takes in this scale."""
    n = len(k.intervals)
    out = set()
    for d in range(n):
        a = step_to_pitch(d, k)
        out.add(step_to_pitch(d + steps, k) - a)
    return out


def diatonic_shift(pitch: int, steps: int, key: str | KeyInfo, chromatic: str = "smart") -> tuple[int, bool]:
    """Move ``pitch`` by ``steps`` scale steps (``+2`` = a third above in a 7-note scale).

    Returns ``(new_pitch, was_chromatic)``.  Scale tones move strictly diatonically.
    Chromatic source notes (not in the key) are handled by ``chromatic``:

    * ``"smart"`` (default): try the diatonic harmony of the scale tone below and
      above the source and keep the one whose interval to the source is a size
      that interval normally has in this key (e.g. a 3 or 4 semitone third),
      avoiding tritones; if neither fits, move in parallel with the lower neighbour.
    * ``"parallel"``: keep the exact chromatic offset (shift the lower neighbour
      diatonically, then re-apply the alteration).
    * ``"snap"``: snap the result into the key (diatonic harmony of the nearest scale tone).
    """
    k = parse_key(key)
    pitch = int(pitch)
    step, alt = step_index(pitch, k)
    if alt == 0 or steps == 0:
        return step_to_pitch(step + steps, k) + (alt if steps == 0 else 0), alt != 0
    lower = step_to_pitch(step + steps, k)
    if chromatic == "parallel":
        return lower + alt, True
    upper = step_to_pitch(step + 1 + steps, k)
    if chromatic == "snap":
        src_snap = snap_to_scale(pitch, k)
        s2, _ = step_index(src_snap, k)
        return step_to_pitch(s2 + steps, k), True
    good = {abs(iv) for iv in _diatonic_intervals(k, steps)}
    no_tritone = {iv for iv in good if iv % 12 != 6} or good
    # read the chromatic note as a lowered upper neighbour (b3, b6, b7 in major; b2 in minor)
    # or as a raised lower neighbour (#1, #4, #5, leading tones); that decides ties
    flat_like = (pitch - k.tonic) % 12 in _FLAT_SIDE.get(k.mode, set())
    for cand in ((upper, lower) if flat_like else (lower, upper)):
        if abs(cand - pitch) in no_tritone:
            return cand, True
    return lower + alt, True


# --------------------------------------------------------------------- note transforms

def _copy_note(n: Note, keep_id: bool = True, **changes) -> Note:
    c = Note(n.onset, n.duration, n.pitch, n.lyric, n.phonemes, copy.deepcopy(n.attrs), n.id if keep_id else _new_id())
    for k, v in changes.items():
        setattr(c, k, v)
    return c


def _check_pitch(p: int, what: str = "pitch") -> int:
    if not 0 <= p <= 127:
        raise ValueError(f"{what} {p} ({midi_to_note(p) if -12 < p < 140 else p}) is outside MIDI range 0-127")
    return p


def transpose(notes: Iterable[Note], semitones: int = 0, *, steps: int = 0,
              key: str | KeyInfo | None = None, chromatic: str = "smart") -> list[Note]:
    """Transpose notes chromatically by ``semitones`` and/or diatonically by ``steps`` in ``key``.

    Note ids are kept (this is an edit, not a copy).
    """
    out = []
    k = parse_key(key) if steps else None
    for n in notes:
        p = n.pitch
        if steps:
            p, _ = diatonic_shift(p, steps, k, chromatic)
        p += int(semitones)
        out.append(_copy_note(n, pitch=_check_pitch(p, f"transposed note at beat {n.onset:g}")))
    return out


def _phrases(notes: list[Note], gap: float = 0.5) -> list[list[Note]]:
    phrases: list[list[Note]] = []
    prev = None
    for n in sorted(notes, key=lambda x: x.onset):
        if prev is None or n.onset - prev.end >= gap - 1e-9 or (prev.lyric or "").lower() in ("br", "breath"):
            phrases.append([])
        phrases[-1].append(n)
        prev = n
    return phrases


def fit_range(notes: list[Note], low: int, high: int, gap: float = 0.5) -> tuple[list[Note], int]:
    """Octave-shift notes into ``[low, high]``.

    Whole phrases (separated by rests of at least ``gap`` beats) are shifted
    together when that brings them into range, to avoid octave leaps inside a
    phrase; any remaining stragglers are shifted individually.  Returns
    ``(notes, number_of_notes_moved)``.  Notes are modified in place.
    """
    moved = 0
    if high - low < 11:
        high = low + 11  # an octave is needed for octave shifting to always succeed
    for ph in _phrases(notes, gap):
        best_k, best_bad = 0, None
        for k in range(-4, 5):
            bad = sum(1 for n in ph if not low <= n.pitch + 12 * k <= high)
            score = (bad, abs(k))
            if best_bad is None or score < best_bad:
                best_k, best_bad = k, score
        for n in ph:
            p = n.pitch + 12 * best_k
            while p < low:
                p += 12
            while p > high:
                p -= 12
            if p != n.pitch:
                moved += 1
                n.pitch = p
    return notes, moved


def harmonize(notes: Iterable[Note], steps: int, key: str | KeyInfo, *, voice_range: tuple[int, int] | None = None,
              chromatic: str = "smart", keep_lyrics: bool = True, octave: int = 0,
              stats: dict | None = None) -> list[Note]:
    """Diatonic harmony line for ``notes``.

    ``steps`` is the interval in scale steps: ``+2`` = a third above, ``-2`` = a
    third below, ``+4`` = a fifth above, ``-3`` = a fourth below, ``+5`` = a sixth
    above, ``7`` = an octave (7-note scales).  ``octave`` adds whole octaves.
    Chromatic source notes are handled as described in :func:`diatonic_shift`.
    ``voice_range=(low, high)`` octave-shifts phrases into a singable range.
    Returns new notes with new ids; lyrics, phoneme overrides and attributes are
    copied when ``keep_lyrics`` (so the harmony sings the same words).
    If ``stats`` is given it is filled with ``chromatic_notes`` and ``range_shifted``.
    """
    k = parse_key(key)
    out = []
    n_chrom = 0
    for n in sorted(notes, key=lambda x: x.onset):
        p, chrom = diatonic_shift(n.pitch, int(steps), k, chromatic)
        n_chrom += chrom
        c = _copy_note(n, keep_id=False, pitch=p + 12 * int(octave))
        if not keep_lyrics:
            c.lyric, c.phonemes = "la", None
        out.append(c)
    moved = 0
    if voice_range is not None:
        out, moved = fit_range(out, int(voice_range[0]), int(voice_range[1]))
    for c in out:
        c.pitch = max(0, min(127, c.pitch))
    if stats is not None:
        stats.update({"chromatic_notes": n_chrom, "range_shifted": moved})
    return out


def _grid_value(grid: float | str) -> float:
    g = parse_duration(grid)
    if not g > 0:
        raise ValueError(f"grid must be positive, got {grid!r}")
    return g


def resolve_overlaps(notes: list[Note], min_length: float = 0.0625) -> list[Note]:
    """Make a note list monophonic: sort, push colliding onsets forward, trim overlapping ends.

    Notes are modified in place and returned sorted.  No note ever ends up with a
    length below ``min_length`` (beats).
    """
    notes.sort(key=lambda n: (n.onset, n.pitch))
    for i in range(1, len(notes)):
        prev, cur = notes[i - 1], notes[i]
        if cur.onset < prev.onset + min_length - 1e-9:
            end = cur.end
            cur.onset = prev.onset + min_length
            cur.duration = max(end - cur.onset, min_length)
        if prev.end > cur.onset + 1e-9:
            prev.duration = max(cur.onset - prev.onset, min_length)
    return notes


def quantize(notes: Iterable[Note], grid: float | str, strength: float = 1.0, *, ends: bool = True,
             offset: float = 0.0, min_length: float | None = None) -> list[Note]:
    """Snap note onsets (and ends, when ``ends``) towards a grid.

    ``grid`` is in beats (``0.25`` = sixteenth) or a note value (``"s"``, ``"e"``,
    ``"e3"`` for eighth triplets).  ``strength`` 0..1 moves part of the way.
    Notes never become zero-length (minimum one grid step at full strength) and
    overlaps created by snapping are resolved.  Ids are kept.
    """
    g = _grid_value(grid)
    strength = float(min(max(strength, 0.0), 1.0))
    if min_length is None:
        min_length = g if strength >= 1.0 else max(g * strength, 1e-3)

    def snap(x: float) -> float:
        target = round((x - offset) / g) * g + offset
        return x + (target - x) * strength

    out = []
    for n in notes:
        on = snap(n.onset)
        end = snap(n.end) if ends else on + n.duration
        if end - on < min_length - 1e-9:
            end = on + min_length
        out.append(_copy_note(n, onset=round(on, 9), duration=round(end - on, 9)))
    return resolve_overlaps(out, min_length=min(min_length, g))


def double(notes: Iterable[Note], detune_cents: float = 7.0, delay_beats: float = 0.0,
           octave: int = 0) -> list[Note]:
    """Copy notes for a doubled / unison track with a small detune and optional delay (new ids)."""
    out = []
    for n in notes:
        c = _copy_note(n, keep_id=False, onset=n.onset + delay_beats, pitch=_check_pitch(n.pitch + 12 * octave))
        c.attrs["detune"] = float(c.attrs.get("detune") or 0.0) + float(detune_cents)
        out.append(c)
    return out


# ------------------------------------------------------------------------------ lyrics

_BRACKET_RE = re.compile(r"\[[^\]]*\]|\S+")


def lyric_tokens(text: str, language: str = "en") -> list[str]:
    """Split free text into per-note lyric tokens (via :func:`synth2.phonemes.split_lyrics`).

    ``[...]`` explicit-phoneme groups, ``br`` breaths and ``-`` / ``+`` tokens are kept intact.
    """
    from ..phonemes import split_lyrics

    out: list[str] = []
    chunk: list[str] = []

    def flush():
        if chunk:
            out.extend(split_lyrics(" ".join(chunk), language))
            chunk.clear()

    for m in _BRACKET_RE.finditer(text.replace("\n", " ")):
        tok = m.group(0)
        if tok.startswith("[") and tok.endswith("]"):
            flush()
            out.append("[" + " ".join(tok[1:-1].split()) + "]")
        elif tok.lower() in ("br", "breath"):
            flush()
            out.append("br")
        else:
            chunk.append(tok)
    flush()
    return out


def fit_lyrics(notes: Iterable[Note], text: str, language: str = "en", *, keep_breaths: bool = True) -> list[Note]:
    """Distribute free text over notes, one syllable per note.

    Multi-syllable English words become ``word`` + ``+`` continuation notes.
    More syllables than notes: the remaining words are joined onto the last
    note's lyric (sung together on it).  Fewer: remaining notes get ``-``
    (melisma, holding the previous vowel).  Notes whose lyric is ``br`` are
    skipped when ``keep_breaths``.  Phoneme overrides of re-lyriced notes are
    cleared.  Returns copies (same ids) in onset order.
    """
    out = [_copy_note(n) for n in sorted(notes, key=lambda x: x.onset)]
    targets = [n for n in out if not (keep_breaths and (n.lyric or "").strip().lower() in ("br", "breath"))]
    tokens = lyric_tokens(text, language)
    if not targets:
        return out
    for i, n in enumerate(targets):
        n.phonemes = None
        if i < len(targets) - 1 or len(tokens) <= len(targets):
            n.lyric = tokens[i] if i < len(tokens) else "-"
        else:
            rest = tokens[i:]
            words = [t for t in rest if t not in ("+", "-", "~", "+~", "ー")]
            n.lyric = " ".join(words) if words else rest[0]
    return out
