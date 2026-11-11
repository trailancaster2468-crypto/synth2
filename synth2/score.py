"""Compact text score notation (``.s2s``): the language AI models write.

Example::

    # comments start with '#' (at line start or after whitespace; 'F#4' is a pitch)
    title: Twinkle
    tempo: 112            # bpm; "tempo: 90 @ 32" = tempo change at beat 32
    time: 4/4
    key: C major

    track: Lead voice=aurora style=pop lang=en gain=0 pan=0
    C4 q twinkle  C4 q +  G4 q twinkle  G4 q +  |  A4 q little  A4 q +  G4 h star
    r q                                 # rest
    E4 e. la  F4 s -                    # dotted eighth, then a melisma sixteenth
    G4 h+q oh{vibrato_depth=0.5,tension=0.3}   # tied duration, per-note attrs
    A4 q [s t aa r]                     # explicit phonemes as the lyric
    B4 q star[s t aa r]                 # lyric "star" with a phoneme override
    @ 32                                # jump to beat 32 (absolute)
    curve: loudness 0:0 4:-3 8:0 (linear)

Grammar summary
---------------
* Header lines ``title:``, ``tempo:``/``bpm:``, ``time:``, ``key:``,
  ``sample_rate:``, ``engine:`` (the singing engine of the whole project:
  ``formant``, ``neural`` or ``neural:<voicebank id>``; ``auto`` or no line =
  the best installed engine) and
  ``master: gain=0 room=0.55 wet=1 normalize=true limiter=true``.
* ``track: NAME key=value ...`` starts a track.  Options: ``voice``, ``style``
  (defaults to the voice's own style), ``lang``, ``gain`` (dB), ``pan`` (-1..1),
  ``reverb`` (send 0..1), ``mute``, ``solo``, ``seed``, ``engine`` (this track's
  engine, same values as the header), ``role`` (the track's role in the mix for
  the neural engine: ``lead``, ``double``, ``harmony``, ``answer``, ``adlib``,
  ``crowd``; ``"double:NAME"`` doubles track NAME), ``voice.<field>=v`` (voice tweaks such as
  ``voice.breathiness=0.3``; ``voice.neural.<name>=v`` tunes the neural engine,
  e.g. ``voice.neural.gender=0.2`` or ``voice.neural.fx.alien=0.6``) and
  ``style.<field>=v`` (expression-style overrides such as
  ``style.vibrato_depth=1.3``).
* A note is ``PITCH DURATION LYRIC``.  ``PITCH``: note name (``C4``, ``F#3``,
  ``Bb5``) or MIDI number; ``r`` is a rest (``r DURATION``).  ``DURATION``: any
  :func:`synth2.timing.parse_duration` token (``w h q e s t``, dots ``q.``,
  triplets ``e3``, ties ``h+q``, numbers in beats ``1.5``, fractions ``3/2``).
  ``LYRIC``: one token: a word/syllable, ``-`` (melisma), ``+`` (next syllable
  of the previous word), ``br`` (breath), ``[ph ph]`` explicit phonemes, or a
  ``"quoted string"``.  ``word[ph ph]`` sets a phoneme override for the word and
  ``{k=v,...}`` after the lyric sets note attributes.  The lyric may be
  omitted (it defaults to ``la``) when the next token starts a new note.
* ``|`` bar lines are ignored for timing (a warning is produced when a bar line
  is not on a bar boundary).  ``@ BEAT`` (or ``@ bar N``) moves the position.
  Notes are sequential from the track's current position.
* Inside a track, ``tempo: BPM`` adds a tempo change at the current position and
  ``curve: NAME beat:value ... (linear|cubic|step)`` adds automation points
  (absolute beats).

Public API: :func:`parse_score`, :func:`parse_score_ex`, :func:`project_to_score`,
:func:`parse_notes`, :func:`parse_snippet`, :func:`notes_to_score`, :class:`ScoreError`.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

from .project import CURVE_SPECS, NOTE_ATTRIBUTES, Curve, MasterSettings, Note, Project, Track
from .timing import midi_to_note, note_to_midi, parse_duration

__all__ = [
    "ScoreError", "Snippet", "parse_score", "parse_score_ex", "project_to_score", "parse_notes",
    "parse_snippet", "notes_to_score", "duration_token",
]

HEADER_KEYS = ("title", "tempo", "bpm", "time", "meter", "key", "sample_rate", "engine", "master")
TRACK_OPTIONS = ("voice", "style", "lang", "language", "gain", "gain_db", "pan", "reverb", "reverb_send",
                 "mute", "solo", "seed", "engine", "role", "voice.<field>", "voice.neural.<name>", "style.<field>")
INTERPOLATIONS = ("linear", "cubic", "step")
DEFAULT_LYRIC = "la"

_DIRECTIVE_RE = re.compile(r"^([A-Za-z_]+)\s*:(.*)$")
_PITCH_NAME_RE = re.compile(r"^[A-Ga-g][#b♯♭]{0,2}-?\d+$")
_REST_TOKENS = {"r", "R", "rest"}
_BAR_RE = re.compile(r"^[|:]*\|[|:]*$")
_SAFE_WORD_RE = re.compile(r"^[^\s\"\[\]{}#@|]+$")
_SAFE_NAME_RE = re.compile(r"^[\w.\-'()!?,&/+]+(?: [\w.\-'()!?,&/+]+)*$")
_SAFE_VALUE_RE = re.compile(r"^[A-Za-z_][\w.\-]*$")
# an engine spec that needs no quotes: formant, neural, neural:<voicebank id>
_SAFE_ENGINE_RE = re.compile(r"^[A-Za-z]+(?::[\w.\-]+)?$")
# voice parameters of the neural engine: neural.gender, neural.fx.alien, ...
_NEURAL_PARAM_RE = re.compile(r"^neural(?:\.[A-Za-z_][\w\-]*)+$")
_EPS = 1e-7


class ScoreError(ValueError):
    """A score syntax/semantic error, with the 1-based line number and offending token."""

    def __init__(self, message: str, line: int | None = None, token: str | None = None, source: str | None = None):
        self.message = message
        self.line = line
        self.token = token
        self.source = source
        loc = []
        if line is not None:
            loc.append(f"line {line}")
        if token is not None:
            loc.append(f"token {token!r}")
        text = (", ".join(loc) + ": " if loc else "") + message
        if source is not None and source.strip():
            text += f"\n    > {source.strip()}"
        super().__init__(text)


@dataclass
class Snippet:
    """Result of :func:`parse_snippet`: notes plus anything else the snippet contained."""

    notes: list[Note] = field(default_factory=list)
    curves: dict[str, Curve] = field(default_factory=dict)
    tempos: list[tuple[float, float]] = field(default_factory=list)  # (beat, bpm)
    start_beat: float = 0.0
    end_beat: float = 0.0  # position after the last note/rest (where the next note would go)
    warnings: list[str] = field(default_factory=list)


# ------------------------------------------------------------------------- lexing

def _scan_string(s: str, i: int) -> int:
    """Index just past the JSON-style string starting at ``s[i] == '"'``, or -1."""
    j = i + 1
    while j < len(s):
        c = s[j]
        if c == "\\":
            j += 2
            continue
        if c == '"':
            return j + 1
        j += 1
    return -1


def _scan_group(s: str, i: int) -> int:
    """Index just past the ``[...]`` / ``{...}`` group starting at ``s[i]``, or -1."""
    close = "]" if s[i] == "[" else "}"
    j = i + 1
    while j < len(s):
        c = s[j]
        if c == '"':
            j = _scan_string(s, j)
            if j < 0:
                return -1
            continue
        if c == close:
            return j + 1
        j += 1
    return -1


def _strip_comment(line: str) -> str:
    i = 0
    n = len(line)
    while i < n:
        c = line[i]
        if c == '"':
            j = _scan_string(line, i)
            if j < 0:
                return line
            i = j
            continue
        if c in "[{":
            j = _scan_group(line, i)
            if j < 0:
                return line
            i = j
            continue
        if c == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i]
        i += 1
    return line


def _tokenize(s: str, lineno: int | None, raw: str) -> list[str]:
    toks: list[str] = []
    i, n = 0, len(s)
    while i < n:
        if s[i].isspace():
            i += 1
            continue
        start = i
        while i < n and not s[i].isspace():
            c = s[i]
            if c == '"':
                j = _scan_string(s, i)
                if j < 0:
                    raise ScoreError("unterminated quoted string (missing closing '\"')", lineno, s[i:], raw)
                i = j
            elif c in "[{":
                j = _scan_group(s, i)
                if j < 0:
                    close = "]" if c == "[" else "}"
                    raise ScoreError(f"missing closing '{close}'", lineno, s[i:], raw)
                i = j
            else:
                i += 1
        toks.append(s[start:i])
    return toks


def _unquote(s: str, lineno=None, raw=None) -> str:
    s = s.strip()
    if s.startswith('"'):
        try:
            v = json.loads(s)
        except ValueError:
            raise ScoreError("invalid quoted string", lineno, s, raw) from None
        if not isinstance(v, str):
            raise ScoreError("expected a quoted string", lineno, s, raw)
        return v
    return s


# ---------------------------------------------------------------- token classifiers

def _parse_pitch(tok: str) -> int | None:
    """MIDI number for a pitch token, ``None`` if it is not a pitch."""
    if re.fullmatch(r"-?\d+", tok):
        return int(tok)
    if _PITCH_NAME_RE.match(tok):
        try:
            return note_to_midi(tok)
        except ValueError:
            return None
    return None


def _is_pitch_like(tok: str) -> bool:
    return tok in _REST_TOKENS or _parse_pitch(tok) is not None


def _parse_dur(tok: str) -> float | None:
    if not tok or tok[0] in '"[{@|' or tok in _REST_TOKENS:
        return None
    try:
        v = parse_duration(tok)
    except (ValueError, ZeroDivisionError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _parse_number(tok: str, what: str, lineno, raw) -> float:
    try:
        v = float(tok)
    except (TypeError, ValueError):
        raise ScoreError(f"expected a number for {what}", lineno, tok, raw) from None
    if not math.isfinite(v):
        raise ScoreError(f"{what} must be finite", lineno, tok, raw)
    return v


def _parse_engine(v: str, lineno, tok, raw) -> str:
    """The canonical engine spec for a header or track value (``""`` for auto)."""
    from .engine.backend import normalize_engine

    spec = _unquote(v, lineno, raw)
    try:
        return normalize_engine(spec)
    except ValueError as exc:
        raise ScoreError(str(exc), lineno, tok, raw) from None


def _parse_bool(v: str, what: str, lineno, raw) -> bool:
    low = v.lower()
    if low in ("1", "true", "yes", "on"):
        return True
    if low in ("0", "false", "no", "off"):
        return False
    raise ScoreError(f"{what} must be true or false", lineno, v, raw)


def _parse_value(v: str):
    v = v.strip()
    if v.startswith('"'):
        return json.loads(v)
    low = v.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if low in ("null", "none"):
        return None
    try:
        f = float(v)
        if math.isfinite(f):
            return f
    except ValueError:
        pass
    return v


def _attr_problem(k: str, v) -> str | None:
    """Why ``v`` is not a valid value for the known note attribute ``k`` (``None`` if it is fine).

    Unknown attributes are not checked (the engine ignores them); ``null`` is always allowed.
    """
    if k not in NOTE_ATTRIBUTES or v is None:
        return None
    if k == "language":
        return None if isinstance(v, str) else f"attribute {k!r} must be a language code such as en or ja"
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
        return f"attribute {k!r} must be a finite number, got {v!r} - {NOTE_ATTRIBUTES[k][1]}"
    return None


def _split_commas(s: str) -> list[str]:
    parts, cur, i = [], [], 0
    while i < len(s):
        c = s[i]
        if c == '"':
            j = _scan_string(s, i)
            j = len(s) if j < 0 else j
            cur.append(s[i:j])
            i = j
            continue
        if c == ",":
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(c)
        i += 1
    parts.append("".join(cur))
    return [p.strip() for p in parts if p.strip()]


# ------------------------------------------------------------------------ the parser

class _Parser:
    def __init__(self, mode: str, start_beat: float = 0.0, language: str = "en", beats_per_bar: float = 4.0):
        self.mode = mode  # "score" | "snippet"
        self.warnings: list[str] = []
        self.project = Project(title="Untitled") if mode == "score" else None
        self.language = language
        self.track: Track | None = None
        self.pos = float(start_beat)
        self.start_beat = float(start_beat)
        self.last_note: Note | None = None
        self.bpb = float(beats_per_bar)
        # snippet results
        self.notes: list[Note] = []
        self.curves: dict[str, Curve] = {}
        self.tempos: list[tuple[float, float]] = []
        self._header_tempo_set = False
        # curve points collected while parsing, keyed by id(curve): (curve, {round(beat, 9): (beat, value)});
        # written back sorted once at the end of parse() (per-point list rebuilds were quadratic)
        self._curve_points: dict[int, tuple[Curve, dict]] = {}

    # -- helpers ---------------------------------------------------------------
    def warn(self, lineno, msg):
        self.warnings.append(f"line {lineno}: {msg}")

    def _notes_target(self) -> list[Note]:
        if self.mode == "snippet":
            return self.notes
        if self.track is None:
            self._new_track("Lead", {}, None, None)
        return self.track.notes

    def _language(self) -> str:
        if self.mode == "score" and self.track is not None:
            return self.track.language
        return self.language

    # -- top level --------------------------------------------------------------
    def parse(self, text: str):
        if text.startswith("﻿"):
            text = text[1:]
        # split on real newlines only: str.splitlines() also breaks on \f, \v, \x1c-\x1e, \x85, U+2028/9,
        # which would turn comment text into notes and shift line numbers
        for lineno, raw in enumerate(text.replace("\r\n", "\n").replace("\r", "\n").split("\n"), 1):
            line = _strip_comment(raw).strip()
            if not line:
                continue
            m = _DIRECTIVE_RE.match(line)
            if m:
                self.directive(m.group(1).lower(), m.group(2).strip(), lineno, raw)
            else:
                self.note_line(line, lineno, raw)
        for curve, merged in self._curve_points.values():
            curve.points = sorted(merged.values(), key=lambda p: p[0])
        if self.mode == "score":
            for t in self.project.tracks:
                t.sort()
                for name in list(t.curves):
                    t.curves[name].sort()
        return self

    # -- directives ---------------------------------------------------------------
    def directive(self, key: str, value: str, lineno: int, raw: str):
        if key == "track":
            if self.mode == "snippet":
                raise ScoreError("'track:' lines are not allowed in a note snippet; pass only note lines such as "
                                 "'C4 q la  D4 q la', rests 'r q', '@ BEAT', 'tempo:' and 'curve:' lines "
                                 "(load a full score with parse_score / new_project(score=...))", lineno, "track:", raw)
            self.track_line(value, lineno, raw)
            return
        if key == "curve":
            self.curve_line(value, lineno, raw)
            return
        if key in ("tempo", "bpm"):
            self.tempo_line(value, lineno, raw)
            return
        if key in ("title", "time", "meter", "key", "sample_rate", "engine", "master"):
            if self.mode == "snippet":
                raise ScoreError(f"'{key}:' is a score header and is not allowed in a note snippet; set it with "
                                 f"update_project instead", lineno, f"{key}:", raw)
            self.header(key, value, lineno, raw)
            return
        valid = ", ".join(f"'{k}:'" for k in HEADER_KEYS + ("track", "curve"))
        raise ScoreError(f"unknown directive '{key}:'; valid directives are {valid}", lineno, f"{key}:", raw)

    def header(self, key, value, lineno, raw):
        p = self.project
        if key == "title":
            p.title = _unquote(value, lineno, raw) or "Untitled"
        elif key in ("time", "meter"):
            m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", value)
            if not m:
                raise ScoreError("time signature must look like 4/4, 3/4 or 6/8", lineno, value, raw)
            num, den = int(m.group(1)), int(m.group(2))
            if not 1 <= num <= 64 or den not in (1, 2, 4, 8, 16, 32):
                raise ScoreError("time signature numerator must be 1-64 and denominator a power of two (1-32)",
                                 lineno, value, raw)
            p.time_signature = (num, den)
            self.bpb = p.beats_per_bar()
        elif key == "key":
            k = _unquote(value, lineno, raw)
            try:
                from .ai.composer import parse_key

                parse_key(k)
            except ValueError as exc:
                self.warn(lineno, f"key {k!r} not understood ({exc}); harmony tools will assume C major")
            p.key = k
        elif key == "sample_rate":
            sr = int(_parse_number(value, "sample_rate", lineno, raw))
            if not 8000 <= sr <= 192000:
                raise ScoreError("sample_rate must be between 8000 and 192000", lineno, value, raw)
            p.sample_rate = sr
        elif key == "engine":
            p.engine = _parse_engine(value, lineno, value, raw)
        elif key == "master":
            self.master_line(value, lineno, raw)

    def master_line(self, value, lineno, raw):
        m = self.project.master
        names = {"gain": "gain_db", "gain_db": "gain_db", "room": "reverb_room", "reverb_room": "reverb_room",
                 "wet": "reverb_wet", "reverb_wet": "reverb_wet", "normalize": "normalize", "limiter": "limiter"}
        for tok in _tokenize(value, lineno, raw):
            if "=" not in tok:
                raise ScoreError("master settings are key=value pairs (gain, room, wet, normalize, limiter)",
                                 lineno, tok, raw)
            k, v = tok.split("=", 1)
            attr = names.get(k.lower())
            if attr is None:
                raise ScoreError(f"unknown master setting {k!r}; valid: gain, room, wet, normalize, limiter",
                                 lineno, tok, raw)
            if attr in ("normalize", "limiter"):
                setattr(m, attr, _parse_bool(v, k, lineno, raw))
            else:
                setattr(m, attr, _parse_number(v, k, lineno, raw))

    def tempo_line(self, value, lineno, raw):
        m = re.fullmatch(r"([^\s@]+)\s*(?:@\s*([^\s]+))?", value)
        if not m:
            raise ScoreError("tempo must look like 'tempo: 120' or 'tempo: 90 @ 32'", lineno, value, raw)
        bpm = _parse_number(m.group(1), "tempo (bpm)", lineno, raw)
        if not 1 <= bpm <= 2000:
            raise ScoreError("tempo must be between 1 and 2000 bpm", lineno, m.group(1), raw)
        if m.group(2) is not None:
            beat = _parse_number(m.group(2), "tempo change beat", lineno, raw)
        else:
            beat = self.pos if (self.track is not None or self.mode == "snippet") else 0.0
        if beat < 0:
            raise ScoreError("tempo change beat must be >= 0", lineno, m.group(2), raw)
        if self.mode == "snippet":
            self.tempos.append((beat, bpm))
        else:
            self.project.tempo.set(beat, bpm)

    def track_line(self, value, lineno, raw):
        toks = _tokenize(value, lineno, raw)
        name_parts, opts = [], {}
        for tok in toks:
            if "=" in tok and not tok.startswith('"'):
                k, v = tok.split("=", 1)
                opts[k.strip().lower() if not k.startswith(("voice.", "style.")) else k.strip()] = (v, tok)
            else:
                if opts:
                    raise ScoreError("the track name must come before the key=value options "
                                     "(quote names that contain spaces or '=')", lineno, tok, raw)
                name_parts.append(_unquote(tok, lineno, raw))
        name = " ".join(name_parts) or f"Track {len(self.project.tracks) + 1}"
        self._new_track(name, opts, lineno, raw)

    def _new_track(self, name, opts, lineno, raw):
        from .engine.expression import STYLES, Style
        from .voices import VOICES, Voice

        p = self.project
        t = p.add_track(name)
        self.track = t
        self.pos = 0.0
        self.last_note = None
        style_set = False
        voice_fields = {f for f in Voice.__dataclass_fields__} - {"id", "name", "description", "tags", "gender",
                                                                     "default_style"}
        style_fields = {f for f in Style.__dataclass_fields__} - {"name", "description"}
        for k, (v, tok) in opts.items():
            if k.startswith("voice."):
                f = k[6:]
                if f not in voice_fields and not _NEURAL_PARAM_RE.match(f):
                    raise ScoreError(f"unknown voice parameter {f!r}; valid: {', '.join(sorted(voice_fields))}, "
                                     f"and neural.<name> for the neural engine (e.g. neural.gender, "
                                     f"neural.fx.alien)", lineno, tok, raw)
                t.voice_params[f] = _parse_number(v, k, lineno, raw)
            elif k.startswith("style."):
                f = k[6:]
                if f not in style_fields:
                    raise ScoreError(f"unknown style parameter {f!r}; valid: {', '.join(sorted(style_fields))}",
                                     lineno, tok, raw)
                try:
                    val = _parse_value(v)
                except ValueError:  # bad JSON string, e.g. "\x"
                    raise ScoreError(f"{k} must be a number", lineno, tok, raw) from None
                if isinstance(val, bool):
                    val = float(val)
                if not isinstance(val, float):
                    raise ScoreError(f"{k} must be a number", lineno, tok, raw)
                t.voice_params[k] = val
            elif k == "voice":
                t.voice = _unquote(v, lineno, raw)
                if t.voice not in VOICES:
                    self.warn(lineno, f"unknown voice {t.voice!r}; built-in voices: {', '.join(VOICES)}")
            elif k == "style":
                t.style = _unquote(v, lineno, raw)
                style_set = True
                if t.style not in STYLES:
                    self.warn(lineno, f"unknown style {t.style!r}; valid styles: {', '.join(STYLES)}")
            elif k in ("lang", "language"):
                t.language = _unquote(v, lineno, raw)
                if t.language not in ("en", "ja"):
                    self.warn(lineno, f"language {t.language!r} is not supported; use 'en' or 'ja'")
            elif k in ("gain", "gain_db"):
                t.gain_db = _parse_number(v, k, lineno, raw)
            elif k == "pan":
                t.pan = _parse_number(v, k, lineno, raw)
                if not -1 <= t.pan <= 1:
                    raise ScoreError("pan must be between -1 (left) and 1 (right)", lineno, tok, raw)
            elif k in ("reverb", "reverb_send"):
                t.reverb_send = _parse_number(v, k, lineno, raw)
            elif k == "mute":
                t.mute = _parse_bool(v, k, lineno, raw)
            elif k == "solo":
                t.solo = _parse_bool(v, k, lineno, raw)
            elif k == "seed":
                t.seed = int(_parse_number(v, k, lineno, raw))
            elif k == "engine":
                t.engine = _parse_engine(v, lineno, tok, raw)
            elif k == "role":
                from .engine.vocal_chain import ROLES, parse_role

                t.role = _unquote(v, lineno, raw).strip()
                if t.role and not parse_role(t.role)[0]:
                    self.warn(lineno, f"unknown role {t.role!r}; roles: {', '.join(ROLES)} "
                                      "(a double may name its track: \"double:Lead\")")
            else:
                raise ScoreError(f"unknown track option {k!r}; valid options: {', '.join(TRACK_OPTIONS)}",
                                 lineno, tok, raw)
        if not style_set and t.voice in VOICES:
            t.style = VOICES[t.voice].default_style

    def curve_line(self, value, lineno, raw):
        if self.mode == "score" and self.track is None:
            raise ScoreError("'curve:' lines must come after a 'track:' line", lineno, "curve:", raw)
        toks = _tokenize(value, lineno, raw)
        if not toks:
            raise ScoreError("curve needs a name and points, e.g. 'curve: loudness 0:0 4:-3 8:0 (linear)'",
                             lineno, "curve:", raw)
        name = toks[0]
        if name not in CURVE_SPECS:
            lower = {k.lower(): k for k in CURVE_SPECS}
            if name.lower() in lower:
                name = lower[name.lower()]
            else:
                raise ScoreError(f"unknown curve {name!r}; valid curves: {', '.join(CURVE_SPECS)}", lineno, name, raw)
        spec = CURVE_SPECS[name]
        target = self.curves if self.mode == "snippet" else self.track.curves
        curve = target.get(name)
        if curve is None:
            curve = target[name] = Curve()
        entry = self._curve_points.get(id(curve))
        if entry is None:
            entry = self._curve_points[id(curve)] = (curve, {round(b, 9): (b, v) for b, v in curve.points})
        merged = entry[1]
        for tok in toks[1:]:
            low = tok.strip("()").lower()
            if tok.startswith("(") or low in INTERPOLATIONS or low.startswith("interp"):
                low = low.split("=", 1)[-1]
                if low not in INTERPOLATIONS:
                    raise ScoreError(f"unknown interpolation {low!r}; valid: {', '.join(INTERPOLATIONS)}",
                                     lineno, tok, raw)
                curve.interpolation = low
                continue
            if ":" not in tok:
                raise ScoreError("curve points are beat:value pairs like 4:-3", lineno, tok, raw)
            b, v = tok.split(":", 1)
            beat = _parse_number(b, "curve point beat", lineno, raw)
            val = _parse_number(v, "curve point value", lineno, raw)
            if not spec[1] <= val <= spec[2]:
                self.warn(lineno, f"{name} value {val:g} is outside {spec[1]:g}..{spec[2]:g} {spec[3]} "
                                  f"and will be clamped when rendering")
            merged[round(beat, 9)] = (beat, val)  # a later point at the same beat replaces the earlier one

    # -- notes -------------------------------------------------------------------------
    def note_line(self, line, lineno, raw):
        toks = _tokenize(line, lineno, raw)
        target = self._notes_target()
        i = 0
        n = len(toks)
        while i < n:
            tok = toks[i]
            if _BAR_RE.match(tok):
                rel = (self.pos) / self.bpb if self.bpb > 0 else 0.0
                if abs(rel - round(rel)) > 1e-6:
                    self.warn(lineno, f"bar line at beat {self.pos:g} is not on a bar boundary "
                                      f"({self.bpb:g} beats per bar); check the durations before it")
                i += 1
                continue
            if tok.startswith("@"):
                i = self._at(toks, i, lineno, raw)
                continue
            if tok.startswith("{"):
                if self.last_note is None:
                    raise ScoreError("note attributes {...} must follow a note", lineno, tok, raw)
                end = _scan_group(tok, 0)
                if end != len(tok):
                    raise ScoreError("unexpected text after {attributes}; put all attributes in one {...} group "
                                     "and separate notes with spaces", lineno, tok, raw)
                self.last_note.attrs.update(self._attrs(tok, lineno, raw))
                i += 1
                continue
            if tok in _REST_TOKENS:
                if i + 1 >= n:
                    raise ScoreError("rest is missing its duration (e.g. 'r q' or 'r 2')", lineno, tok, raw)
                d = _parse_dur(toks[i + 1])
                if d is None or d <= 0:
                    raise ScoreError(_dur_error(toks[i + 1]), lineno, toks[i + 1], raw)
                self.pos += d
                i += 2
                if i < n and not self._starts_item(toks, i):
                    raise ScoreError("rests take no lyric; expected a new note after 'r DURATION' (for an audible "
                                     "breath write a short note with lyric br, e.g. 'C4 e br')", lineno, toks[i], raw)
                continue
            pitch = _parse_pitch(tok)
            if pitch is None:
                raise ScoreError("invalid pitch: expected a note name like C4, F#3, Bb5, a MIDI number 0-127, "
                                 "or 'r' for a rest (notes are 'PITCH DURATION LYRIC')", lineno, tok, raw)
            if not 0 <= pitch <= 127:
                raise ScoreError(f"pitch {pitch} is outside the MIDI range 0-127", lineno, tok, raw)
            if i + 1 >= n:
                raise ScoreError(f"note {tok} is missing its duration (e.g. '{tok} q la')", lineno, tok, raw)
            d = _parse_dur(toks[i + 1])
            if d is None or d <= 0:
                raise ScoreError(_dur_error(toks[i + 1]), lineno, toks[i + 1], raw)
            lyric, phonemes, attrs = DEFAULT_LYRIC, None, {}
            j = i + 2
            if j < n and not self._starts_item(toks, j):
                lyric, phonemes, attrs = self._lyric(toks[j], lineno, raw)
                j += 1
            note = Note(self.pos, d, pitch, lyric, phonemes, attrs)
            self._check_lyric(note, lineno)
            target.append(note)
            self.last_note = note
            self.pos += d
            i = j

    def _starts_item(self, toks, j) -> bool:
        """Whether ``toks[j]`` starts something other than a lyric (bar, @, attrs, new note)."""
        tok = toks[j]
        if _BAR_RE.match(tok) or tok.startswith(("@", "{")):
            return True
        if _is_pitch_like(tok):
            return j + 1 >= len(toks) or _parse_dur(toks[j + 1]) is not None
        return False

    def _at(self, toks, i, lineno, raw) -> int:
        tok = toks[i]
        rest = tok[1:]
        j = i + 1
        if not rest:
            if j >= len(toks):
                raise ScoreError("'@' must be followed by a beat number, e.g. '@ 16' or '@ bar 5'", lineno, tok, raw)
            rest = toks[j]
            j += 1
        if rest.lower() == "bar":
            if j >= len(toks):
                raise ScoreError("'@ bar' must be followed by a bar number (1 = first bar)", lineno, tok, raw)
            bar = _parse_number(toks[j], "bar number", lineno, raw)
            if bar < 1:
                raise ScoreError("bar numbers start at 1", lineno, toks[j], raw)
            self.pos = (bar - 1) * self.bpb
            return j + 1
        beat = _parse_number(rest, "'@' position (beats)", lineno, raw)
        if beat < 0:
            raise ScoreError("'@' position must be >= 0", lineno, rest, raw)
        self.pos = beat
        return j

    def _lyric(self, tok, lineno, raw):
        phonemes = None
        attrs: dict = {}
        if tok.startswith('"'):
            end = _scan_string(tok, 0)
            lyric = _unquote(tok[:end], lineno, raw)
            rest = tok[end:]
        elif tok.startswith("["):
            end = _scan_group(tok, 0)
            inner = tok[1:end - 1]
            phones = self._phonemes(inner, lineno, tok, raw)
            lyric = "[" + " ".join(phones) + "]"
            rest = tok[end:]
        else:
            m = re.match(r"[^\[{]*", tok)
            lyric = m.group(0)
            rest = tok[m.end():]
        if rest.startswith("["):
            end = _scan_group(rest, 0)
            phonemes = " ".join(self._phonemes(rest[1:end - 1], lineno, tok, raw)) or None
            rest = rest[end:]
        if rest.startswith("{"):
            end = _scan_group(rest, 0)
            attrs = self._attrs(rest[:end], lineno, raw)
            rest = rest[end:]
        if rest:
            raise ScoreError("unexpected text after the lyric; a lyric is one token, optionally followed by "
                             "[phonemes] and {attributes} (quote lyrics containing spaces)", lineno, tok, raw)
        return lyric, phonemes, attrs

    def _phonemes(self, inner, lineno, tok, raw) -> list[str]:
        from .phonemes import parse_phoneme_string

        try:
            parse_phoneme_string(inner)
        except ValueError as exc:
            raise ScoreError(f"{exc}; see list_phonemes for the phoneme inventory", lineno, tok, raw) from None
        return [p for p in re.split(r"[\s,]+", inner.strip()) if p]

    def _attrs(self, tok, lineno, raw) -> dict:
        inner = tok[1:-1]
        out = {}
        for part in _split_commas(inner):
            if "=" not in part:
                raise ScoreError("note attributes are key=value pairs, e.g. {vibrato_depth=0.5,tension=0.3}",
                                 lineno, tok, raw)
            k, v = part.split("=", 1)
            k = k.strip()
            if k not in NOTE_ATTRIBUTES:
                self.warn(lineno, f"unknown note attribute {k!r} (kept, but the engine ignores it); "
                                  f"valid: {', '.join(NOTE_ATTRIBUTES)}")
            try:
                val = _parse_value(v)
            except ValueError:
                raise ScoreError(f"invalid value for attribute {k!r}", lineno, tok, raw) from None
            problem = _attr_problem(k, val)
            if problem:
                raise ScoreError(problem, lineno, tok, raw)
            if k == "language" and val is not None and val not in ("en", "ja"):
                self.warn(lineno, f"language {val!r} is not supported; use 'en' or 'ja'")
            out[k] = val
        return out

    def _check_lyric(self, note: Note, lineno):
        lang = note.attrs.get("language") or self._language()
        if lang == "ja" and not note.phonemes and not note.lyric.startswith("["):
            from .phonemes import japanese

            if japanese.is_kana(note.lyric):
                morae = [m for m in japanese.kana_to_romaji(note.lyric) if m != "-"]
                if len(morae) > 1:
                    self.warn(lineno, f"lyric {note.lyric!r} has {len(morae)} morae on one note; Japanese is "
                                      f"normally sung one mora per note")


def _dur_error(tok: str) -> str:
    return (f"invalid duration {tok!r}: use w h q e s t (whole..32nd), dots 'q.', triplets 'e3', ties 'h+q', "
            f"or a positive number of beats like 1.5")


# -------------------------------------------------------------------------- public API

def parse_score_ex(text: str) -> tuple[Project, list[str]]:
    """Parse a full score; returns ``(project, warnings)``. Raises :class:`ScoreError`."""
    p = _Parser("score").parse(text)
    return p.project, p.warnings


def parse_score(text: str) -> Project:
    """Parse ``.s2s`` score text into a :class:`~synth2.project.Project`. Raises :class:`ScoreError`."""
    return parse_score_ex(text)[0]


def parse_snippet(text: str, start_beat: float = 0.0, language: str = "en", beats_per_bar: float = 4.0) -> Snippet:
    """Parse note lines (plus ``@``, ``|``, ``tempo:`` and ``curve:`` lines) starting at ``start_beat``.

    ``@ BEAT`` positions and curve point beats are absolute project beats.
    """
    p = _Parser("snippet", start_beat, language, beats_per_bar).parse(text)
    return Snippet(p.notes, p.curves, p.tempos, float(start_beat), p.pos, p.warnings)


def parse_notes(text: str, start_beat: float = 0.0, language: str = "en") -> list[Note]:
    """Parse note lines such as ``"C4 q twinkle C4 q + G4 h star"`` into notes starting at ``start_beat``."""
    return parse_snippet(text, start_beat, language).notes


# ----------------------------------------------------------------------- serialisation

def _num(x: float) -> str:
    x = float(x)
    if x == 0:
        return "0"
    r = round(x, 6)
    if abs(x - r) < 1e-10:
        s = f"{r:.6f}".rstrip("0").rstrip(".")
        return s if s not in ("-0", "") else "0"
    return repr(x)


_BASE = {"w": 4.0, "h": 2.0, "q": 1.0, "e": 0.5, "s": 0.25, "t": 0.125}


def _token_table() -> list[tuple[str, float]]:
    out = []
    for name, v in _BASE.items():
        out.append((name, v))
    for name, v in _BASE.items():
        out.append((name + ".", v * 1.5))
    for name, v in _BASE.items():
        out.append((name + "3", v * 2 / 3))
    for name, v in _BASE.items():
        out.append((name + "..", v * 1.75))
    return out


_SINGLE = _token_table()
_PLAIN = [(n, v) for n, v in _SINGLE if not n.endswith("3")]


def duration_token(d: float, allow_number: bool = True) -> str | None:
    """Spell a duration (beats) as note-value tokens when exact (``q``, ``e.``, ``h+q``), else a number."""
    d = float(d)
    for name, v in _SINGLE:
        if abs(v - d) < 1e-9:
            return name
    best = None
    # tied pairs: prefer plain values (no triplets), larger value first
    for pool in (_PLAIN, _SINGLE):
        for a, va in pool:
            for b, vb in pool:
                if va >= vb and abs(va + vb - d) < 1e-9:
                    cand = f"{a}+{b}"
                    if best is None or (cand.count(".") + cand.count("3"), len(cand)) < (
                            best.count(".") + best.count("3"), len(best)):
                        best = cand
        if best:
            return best
    return _num(d) if allow_number else None


def _lyric_token(n: Note) -> str:
    lyric = n.lyric if n.lyric is not None else ""
    if re.fullmatch(r"\[[^\[\]{}\"]*\]", lyric) and lyric == "[" + " ".join(lyric[1:-1].split()) + "]":
        try:
            from .phonemes import parse_phoneme_string

            parse_phoneme_string(lyric[1:-1])
            tok = lyric
        except ValueError:
            tok = _quote(lyric)
    elif lyric and _SAFE_WORD_RE.match(lyric) and not _is_pitch_like(lyric):
        tok = lyric
    else:
        tok = _quote(lyric)
    if n.phonemes and _override_problem(n.phonemes) is None:
        tok += "[" + " ".join(re.split(r"[\s,]+", n.phonemes.strip())) + "]"
    attrs = [(k, v) for k, v in n.attrs.items() if _attr_problem(k, v) is None]
    if attrs:
        tok += "{" + ",".join(f"{k}={_attr_value(v)}" for k, v in attrs) + "}"
    return tok


def _override_problem(phonemes: str) -> str | None:
    """Why the phoneme override would not parse back from ``word[...]`` (``None`` if it would)."""
    if re.search(r"[\[\]{}\"]", phonemes):
        return "it contains brackets or quotes"
    from .phonemes import parse_phoneme_string

    try:
        parse_phoneme_string(phonemes)
    except ValueError as exc:
        return str(exc)
    return None


def _dropped_comment(n: Note) -> str | None:
    """A ``#`` comment naming the note fields :func:`_lyric_token` left out because they would not parse back."""
    dropped = []
    if n.phonemes and _override_problem(n.phonemes) is not None:
        dropped.append(f"phoneme override {json.dumps(n.phonemes)}")
    for k, v in n.attrs.items():
        if _attr_problem(k, v) is not None:
            dropped.append(f"attribute {k}={json.dumps(v, default=str)}")
    if not dropped:
        return None
    # json.dumps (ensure_ascii) keeps user text on this one line
    return f"# dropped invalid {', '.join(dropped)} of {midi_to_note(n.pitch)} at beat {_num(n.onset)}"


_LINE_SEPARATORS = {"\x85": "\\u0085", "\u2028": "\\u2028", "\u2029": "\\u2029"}


def _quote(s: str) -> str:
    """JSON-quote ``s``; also escape the line separators ``json.dumps(ensure_ascii=False)`` leaves raw."""
    q = json.dumps(s, ensure_ascii=False)
    for ch, esc in _LINE_SEPARATORS.items():
        q = q.replace(ch, esc)
    return q


def _attr_value(v) -> str:
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return _num(v)
    s = str(v)
    if _SAFE_VALUE_RE.match(s) and s.lower() not in ("true", "false", "null", "none", "inf", "nan", "infinity"):
        return s
    return _quote(s)


def _name_token(s: str) -> str:
    if s and _SAFE_NAME_RE.match(s) and "=" not in s:
        return s
    return _quote(s)


def _value_token(s: str) -> str:
    return s if s and _SAFE_VALUE_RE.match(s) else _quote(s)


def _engine_token(spec: str) -> str:
    return spec if _SAFE_ENGINE_RE.match(spec) else _quote(spec)


def notes_to_score(notes: list[Note], start_beat: float = 0.0, beats_per_bar: float = 4.0) -> list[str]:
    """Serialise notes as score lines (one line per bar), starting from position ``start_beat``."""
    lines: list[str] = []
    cur: list[str] = []
    comments: list[str] = []  # notes about the current line, written after it
    cur_bar = None
    pos = float(start_beat)
    bpb = beats_per_bar if beats_per_bar > 0 else 4.0

    def bar_of(x: float) -> int:
        return int(math.floor(x / bpb + 1e-9))

    def flush():
        if cur:
            lines.append("  ".join(cur))
            cur.clear()
        lines.extend(comments)
        comments.clear()

    def emit(token: str, at: float):
        nonlocal cur_bar
        b = bar_of(at)
        if cur and b != cur_bar:
            flush()
        if not cur:
            cur_bar = b
        cur.append(token)

    for n in sorted(notes, key=lambda x: x.onset):
        gap = n.onset - pos
        if abs(gap) > _EPS:
            tok = duration_token(gap, allow_number=False) if 0 < gap <= 2 * bpb else None
            if tok is not None and (pos > start_beat + _EPS or gap < bpb):
                emit(f"r {tok}", pos)
                pos += parse_duration(tok)
            else:
                flush()
                lines.append(f"@ {_num(n.onset)}")
                pos = n.onset
        dtok = duration_token(n.duration)
        emit(f"{midi_to_note(n.pitch)} {dtok} {_lyric_token(n)}", n.onset)
        dropped = _dropped_comment(n)
        if dropped:
            comments.append(dropped)
        pos += parse_duration(dtok)  # track the position exactly the way the parser will
    flush()
    return lines


def _track_line(t: Track, index: int) -> str:
    parts = ["track:", _name_token(t.name), f"voice={_value_token(t.voice)}", f"style={_value_token(t.style)}",
             f"lang={_value_token(t.language)}", f"gain={_num(t.gain_db)}", f"pan={_num(t.pan)}"]
    if abs(t.reverb_send - 0.25) > 1e-12:
        parts.append(f"reverb={_num(t.reverb_send)}")
    if t.mute:
        parts.append("mute=true")
    if t.solo:
        parts.append("solo=true")
    if t.seed != index * 7919 + 17:
        parts.append(f"seed={t.seed}")
    if t.engine:
        parts.append(f"engine={_engine_token(t.engine)}")
    if t.role:
        parts.append(f"role={_value_token(t.role)}")
    for k, v in t.voice_params.items():
        parts.append(f"{k}={_num(v)}" if k.startswith("style.") else f"voice.{k}={_num(v)}")
    return " ".join(parts)


def project_to_score(project: Project, tracks: list[Track] | None = None) -> str:
    """Serialise a project (or a subset of its tracks) to score text. Round-trips with :func:`parse_score`."""
    p = project
    out = ["# Synth2 score (.s2s) - notes are PITCH DURATION LYRIC; durations w h q e s t, '.', '3', '+'",
           f"title: {_name_token(p.title)}"]
    pts = p.tempo.points
    out.append(f"tempo: {_num(pts[0].bpm)}")
    for tp in pts[1:]:
        out.append(f"tempo: {_num(tp.bpm)} @ {_num(tp.beat)}")
    out.append(f"time: {p.time_signature[0]}/{p.time_signature[1]}")
    key_safe = re.fullmatch(r"[A-Ga-g][#b♯♭]?(?: ?[A-Za-z]+)*", p.key or "")
    out.append(f"key: {p.key if key_safe else _name_token(p.key)}")
    if p.sample_rate != 44100:
        out.append(f"sample_rate: {p.sample_rate}")
    if p.engine:
        out.append(f"engine: {_engine_token(p.engine)}")
    m, d = p.master, MasterSettings()
    if m != d:
        out.append(f"master: gain={_num(m.gain_db)} room={_num(m.reverb_room)} wet={_num(m.reverb_wet)} "
                   f"normalize={'true' if m.normalize else 'false'} limiter={'true' if m.limiter else 'false'}")
    bpb = p.beats_per_bar()
    emitted = 0  # the parser derives default seeds from the track's position in *this* text
    for t in p.tracks:
        if tracks is not None and t not in tracks:
            continue
        out.append("")
        out.append(_track_line(t, emitted))
        emitted += 1
        out.extend(notes_to_score(t.notes, 0.0, bpb))
        for name, c in t.curves.items():
            if not c.points:
                continue
            pts_txt = [f"{_num(b)}:{_num(v)}" for b, v in c.points]
            for k in range(0, len(pts_txt), 16):
                out.append(f"curve: {name} " + " ".join(pts_txt[k:k + 16]) + f" ({c.interpolation})")
    return "\n".join(out) + "\n"
