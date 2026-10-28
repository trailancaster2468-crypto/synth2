"""AI tool layer: a :class:`Session` plus a JSON tool registry.

Every studio capability is exposed as a tool with a JSON Schema, so AI models
(MCP clients, the Claude-powered director, the REST API) can compose, edit,
render and check vocals at the same level a human uses the GUI::

    from synth2.ai.tools import Session, TOOLS, call_tool, ToolError

    s = Session()
    call_tool(s, "write_melody", {"track": 0, "score": "@ 4 C4 q twinkle C4 q + G4 q twinkle G4 q +"})
    call_tool(s, "render", {})

Conventions: time is in **beats** (quarter notes, beat 0 = song start), pitch is
a MIDI number or a note name ("C4" = 60), curves follow
:data:`synth2.project.CURVE_SPECS`.  Every mutating tool pushes an undo
snapshot (only when the project actually changed) and is atomic: if it fails
the project is left untouched.  Results are JSON-serialisable dicts; bad input
raises :class:`ToolError` with a message written for the model to fix its call.
"""
from __future__ import annotations

import copy
import difflib
import json
import math
import re
import threading
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Callable

from .. import score as score_mod
from ..engine.expression import STYLES, Style, get_style
from ..phonemes import english, japanese, lyric_phonemes, parse_phoneme_string, split_lyrics
from ..phonemes.inventory import PHONEMES
from ..project import CURVE_SPECS, NOTE_ATTRIBUTES, Note, Project, Track, _new_id
from ..timing import TempoPoint, midi_to_note, note_to_midi, parse_duration
from ..voices import VOICES, Voice
from ..voices import list_voices as _list_voices
from . import composer

__all__ = ["Session", "TOOLS", "call_tool", "ToolError", "tool_names", "MUTATING_TOOLS"]

MAX_HISTORY = 100


class ToolError(Exception):
    """Invalid tool call. The message explains what was wrong and how to fix it."""


# ============================================================================ session

class Session:
    """Editing session: the current project, its file path, the last render and undo/redo history."""

    def __init__(self, project: Project | None = None, path: str | Path | None = None):
        self.project: Project = project if project is not None else _default_project()
        self.path: str | None = str(path) if path else None
        self.last_render = None  # synth2.engine.RenderResult | None
        self.last_render_path: str | None = None
        self.last_render_state: str | None = None  # fingerprint of the project that was rendered
        self.last_render_range: tuple[float | None, float | None] = (None, None)  # beats
        self.undo_stack: list[tuple[str, dict]] = []
        self.redo_stack: list[tuple[str, dict]] = []
        self.max_history = MAX_HISTORY
        self.lock = threading.RLock()

    # -- construction -------------------------------------------------------------
    @classmethod
    def open(cls, path: str | Path) -> "Session":
        """Load a project file (any supported format) into a new session."""
        p = Path(path)
        project = _load_any(p)
        return cls(project, p if _is_native(p) else p.with_name(_stem(p) + ".s2p.json"))

    # -- history --------------------------------------------------------------------
    def snapshot(self) -> dict:
        return copy.deepcopy(self.project.to_dict())

    def restore(self, snap: dict):
        self.project = Project.from_dict(copy.deepcopy(snap))

    def push_undo(self, label: str, snap: dict | None = None):
        """Record ``snap`` (default: the current state) as the state to return to on undo."""
        self.undo_stack.append((label, snap if snap is not None else self.snapshot()))
        if len(self.undo_stack) > self.max_history:
            del self.undo_stack[: len(self.undo_stack) - self.max_history]
        self.redo_stack.clear()

    def replace_project(self, project: Project, label: str = "replace project"):
        """Swap in a whole new project (undoable)."""
        with self.lock:
            self.push_undo(label)
            self.project = project

    def undo(self) -> str | None:
        with self.lock:
            if not self.undo_stack:
                return None
            label, snap = self.undo_stack.pop()
            self.redo_stack.append((label, self.snapshot()))
            self.restore(snap)
            return label

    def redo(self) -> str | None:
        with self.lock:
            if not self.redo_stack:
                return None
            label, snap = self.redo_stack.pop()
            self.undo_stack.append((label, self.snapshot()))
            self.restore(snap)
            return label

    def call(self, name: str, args: dict | None = None) -> dict:
        return call_tool(self, name, args)


def _default_project() -> Project:
    p = Project.new("Untitled", bpm=120.0)
    p.tracks[0].style = VOICES[p.tracks[0].voice].default_style
    return p


# ============================================================================ registry

@dataclass
class _Tool:
    name: str
    description: str
    schema: dict
    handler: Callable[[Session, dict], dict]
    mutating: bool


_REGISTRY: dict[str, _Tool] = {}
TOOLS: list[dict] = []
MUTATING_TOOLS: set[str] = set()


def _tool(name: str, description: str, properties: dict | None = None, required: tuple = (),
          mutating: bool = False):
    def deco(fn):
        schema: dict = {"type": "object", "properties": properties or {}, "additionalProperties": False}
        if required:
            schema["required"] = list(required)
        _REGISTRY[name] = _Tool(name, description.strip(), schema, fn, mutating)
        TOOLS.append({"name": name, "description": description.strip(), "input_schema": schema})
        if mutating:
            MUTATING_TOOLS.add(name)
        return fn
    return deco


def tool_names() -> list[str]:
    return [t["name"] for t in TOOLS]


def call_tool(session: Session, name: str, args: dict | None = None) -> dict:
    """Run tool ``name`` with ``args`` on ``session``; returns a JSON-serialisable dict.

    Raises :class:`ToolError` for unknown tools and invalid arguments.  Mutating
    tools are atomic (the project is restored if they fail) and push one undo step.
    """
    tool = _REGISTRY.get(name)
    if tool is None:
        close = difflib.get_close_matches(str(name), list(_REGISTRY), n=3)
        hint = f" Did you mean {', '.join(close)}?" if close else ""
        raise ToolError(f"unknown tool {name!r}.{hint} Available tools: {', '.join(_REGISTRY)}")
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError(f"arguments for {name} must be a JSON object, got {type(args).__name__}")
    args = _coerce(dict(args), tool.schema, name, top=True)
    with session.lock:
        if not tool.mutating:
            try:
                return _jsonable(tool.handler(session, args))
            except ToolError:
                raise
            except (ValueError, KeyError, LookupError) as exc:
                raise ToolError(f"{name}: {_exc_text(exc)}") from exc
        before = session.snapshot()
        try:
            result = tool.handler(session, args)
        except ToolError:
            session.restore(before)
            raise
        except (ValueError, KeyError, LookupError) as exc:
            session.restore(before)
            raise ToolError(f"{name}: {_exc_text(exc)}") from exc
        except Exception:
            session.restore(before)
            raise
        if session.snapshot() != before:
            session.push_undo(_label(name, args), before)
        return _jsonable(result)


def _exc_text(exc: BaseException) -> str:
    if isinstance(exc, KeyError) and exc.args:
        return str(exc.args[0])
    return str(exc)


def _label(name: str, args: dict) -> str:
    bits = []
    for k, v in args.items():
        if isinstance(v, (str, int, float, bool)):
            s = str(v).replace("\n", " ")
            bits.append(f"{k}={s[:24] + '...' if len(s) > 24 else s}")
        if len(bits) >= 3:
            break
    return name + (f"({', '.join(bits)})" if bits else "")


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, Path):
        return str(x)
    try:  # numpy scalars / arrays
        import numpy as np

        if isinstance(x, np.ndarray):
            return _jsonable(x.tolist())
        if isinstance(x, np.generic):
            return _jsonable(x.item())
    except ImportError:  # pragma: no cover
        pass
    return str(x)


# --------------------------------------------------------------------- arg validation

def _type_name(v) -> str:
    if v is None:
        return "null"
    return {bool: "boolean", int: "integer", float: "number", str: "string", list: "array",
            dict: "object"}.get(type(v), type(v).__name__)


def _try_type(v, t):
    if t == "null":
        return v is None, v
    if t == "boolean":
        if isinstance(v, bool):
            return True, v
        if isinstance(v, str) and v.lower() in ("true", "false"):
            return True, v.lower() == "true"
        return False, v
    if t == "integer":
        if isinstance(v, bool):
            return False, v
        if isinstance(v, int):
            return True, v
        if isinstance(v, float) and v.is_integer():
            return True, int(v)
        if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
            return True, int(v)
        return False, v
    if t == "number":
        if isinstance(v, bool):
            return False, v
        if isinstance(v, (int, float)):
            return math.isfinite(v), v
        if isinstance(v, str):
            try:
                f = float(v)
                return math.isfinite(f), f
            except ValueError:
                return False, v
        return False, v
    if t == "string":
        return isinstance(v, str), v
    if t == "array":
        if isinstance(v, (list, tuple)):
            return True, list(v)
        return False, v
    if t == "object":
        return isinstance(v, dict), v
    return True, v


def _coerce(value, schema: dict, path: str, top: bool = False):
    types = schema.get("type")
    if types is not None:
        tl = [types] if isinstance(types, str) else list(types)
        matched = None
        for t in tl:
            ok, v = _try_type(value, t)
            if ok:
                value, matched = v, t
                break
        if matched is None:
            shown = json.dumps(value, default=str)
            shown = shown if len(shown) <= 80 else shown[:77] + "..."
            raise ToolError(f"argument '{path}' must be {' or '.join(tl)}, got {_type_name(value)} {shown}")
    else:
        matched = None
    if "enum" in schema and value not in schema["enum"]:
        raise ToolError(f"argument '{path}' must be one of {', '.join(map(str, schema['enum']))}; got {value!r}")
    if matched in ("integer", "number"):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolError(f"argument '{path}' must be >= {schema['minimum']}, got {value}")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolError(f"argument '{path}' must be <= {schema['maximum']}, got {value}")
    if matched == "array":
        if "minItems" in schema and len(value) < schema["minItems"]:
            raise ToolError(f"argument '{path}' needs at least {schema['minItems']} item(s)")
        if "items" in schema:
            value = [_coerce(x, schema["items"], f"{path}[{i}]") for i, x in enumerate(value)]
    if matched == "object" and "properties" in schema:
        props = schema["properties"]
        if schema.get("additionalProperties") is False:
            unknown = [k for k in value if k not in props]
            if unknown:
                sugg = []
                for k in unknown:
                    close = difflib.get_close_matches(k, list(props), n=1)
                    sugg.append(f"'{k}'" + (f" (did you mean '{close[0]}'?)" if close else ""))
                raise ToolError(f"unknown argument {', '.join(sugg)} for {path}; valid arguments: "
                                f"{', '.join(props) or '(none)'}")
        for k in schema.get("required", []):
            if k not in value or value[k] is None:
                desc = props.get(k, {}).get("description", "")
                where = f"'{k}' for {path}" if top else f"'{path}.{k}'"
                raise ToolError(f"missing required argument {where}" + (f": {desc}" if desc else ""))
        for k, sub in props.items():
            if k in value and value[k] is not None:
                value[k] = _coerce(value[k], sub, k if top else f"{path}.{k}")
            elif k in value and value[k] is None and "null" not in _as_list(sub.get("type")):
                del value[k]  # treat explicit null as "not given"
    return value


def _as_list(t) -> list:
    if t is None:
        return []
    return [t] if isinstance(t, str) else list(t)


# ============================================================================ helpers

_TRACK_REF = {"type": ["integer", "string"],
              "description": "Track index (0 = first), track id, or track name (case-insensitive)."}
_PITCH = {"type": ["integer", "string"], "description": "MIDI number (60 = C4 = middle C) or note name like 'C4', 'F#3', 'Bb5'."}
_BEAT = {"type": "number", "description": "Position in beats (quarter notes; beat 0 = song start)."}
_IDS = {"type": "array", "items": {"type": "string"}, "description": "Note ids to select (from get_notes / get_project)."}
_START = {"type": "number", "description": "Select notes whose onset is >= start_beat (beats)."}
_END = {"type": "number", "description": "Select notes whose onset is < end_beat (beats)."}

LANGUAGES = ("en", "ja")
ATTR_RANGES: dict[str, tuple[float, float]] = {
    "vibrato_depth": (0.0, 3.0), "vibrato_rate": (0.0, 12.0), "vibrato_start": (0.0, 10.0),
    "transition": (0.0, 1.0), "overshoot": (0.0, 1.0), "consonant_length": (0.1, 4.0),
    "tension": (-1.0, 1.0), "breathiness": (-1.0, 1.0), "detune": (-1200.0, 1200.0), "velocity": (0.0, 2.0),
}

STYLE_FIELD_DOCS: dict[str, str] = {
    "transition": "base portamento (glide) time between legato notes, seconds",
    "transition_offset": "glide centre relative to the onset, in units of the glide time (negative = glide starts early)",
    "overshoot": "pitch overshoot after arriving at a new note, fraction of the interval",
    "preparation": "small dip away from the target just before a glide, fraction of the interval",
    "scoop": "semitones below the target that phrase-initial notes scoop up from",
    "scoop_time": "scoop duration, seconds",
    "fall": "semitones the pitch falls at phrase ends",
    "fall_time": "fall duration, seconds",
    "vibrato_depth": "multiplier on the voice's vibrato depth (0 = no automatic vibrato)",
    "vibrato_rate": "multiplier on the voice's vibrato rate",
    "vibrato_start": "seconds after onset before vibrato starts",
    "vibrato_fade": "vibrato fade-in time, seconds",
    "vibrato_min_note": "notes shorter than this many seconds get no vibrato",
    "drift_cents": "slow random pitch drift, cents",
    "phrase_arc_db": "crescendo into the middle of each phrase and back, dB",
    "note_swell_db": "loudness swell on long notes, dB",
    "end_taper_db": "loudness taper on phrase-final notes, dB",
    "pitch_loudness": "dB louder per semitone above the voice's middle range",
    "pitch_tension": "extra tension per semitone above the middle range",
    "tension": "base tension offset -1..1 (soft .. pressed)",
    "breathiness": "base breathiness offset -1..1",
    "end_breathiness": "extra breathiness on phrase endings",
    "auto_breath": "1/true = insert automatic breaths between phrases, 0/false = none",
    "timing_humanize": "random timing offset (standard deviation), seconds",
}
STYLE_FIELDS = [f.name for f in fields(Style) if f.name not in ("name", "description")]

VOICE_PARAM_DOCS: dict[str, str] = {
    "tension": "global tension -1..1 (soft/lax .. pressed/bright)",
    "breathiness": "global breath noise -1..1",
    "formant_scale": "vocal-tract size factor: higher = smaller, younger, brighter timbre (about 0.85-1.35)",
    "brightness_db": "high shelf above ~5 kHz, dB",
    "singer_formant_db": "ring / presence boost around 3 kHz, dB",
    "vibrato_rate": "natural vibrato rate, Hz",
    "vibrato_depth": "natural vibrato depth, semitones",
    "robotic": "0 = natural .. 1 = machine-like vocaloid character",
    "aspiration_db": "breath-noise level relative to the voiced sound, dB",
    "jitter": "cycle-to-cycle pitch roughness (fraction, about 0.001-0.01)",
    "shimmer_db": "cycle-to-cycle amplitude roughness, dB",
    "source_tilt": "glottal source roll-off, dB/octave (lower = brighter, buzzier)",
    "source_corner": "glottal source roll-off corner frequency, Hz",
    "loudness_db": "voice level offset, dB",
    "bandwidth_scale": "formant bandwidth multiplier (higher = duller, less resonant)",
    "f4": "4th formant frequency, Hz", "f5": "5th formant frequency, Hz",
    "range_low": "lowest comfortable MIDI note", "range_high": "highest comfortable MIDI note",
}
VOICE_PARAM_FIELDS = [f.name for f in fields(Voice)
                      if f.name not in ("id", "name", "description", "tags", "gender", "default_style")]

LYRIC_CONVENTIONS = (
    "Lyric conventions (one token per note): a word ('love', 'star'); '+' = next syllable of the previous word "
    "(the WHOLE word goes on the first note: 'hello' then '+' sings hel-lo; 'hel' then '+' would not); '-' = "
    "melisma, keep singing the previous vowel on a new pitch; 'br' = breath intake; '[s t aa r]' = explicit "
    "phonemes; Japanese kana or romaji, one mora per note ('さ', 'ku').")


def _track(session: Session, ref, required: bool = True) -> Track:
    p = session.project
    if ref is None:
        if not p.tracks:
            raise ToolError("the project has no tracks; create one with add_track")
        return p.tracks[0]
    if isinstance(ref, int) or (isinstance(ref, str) and re.fullmatch(r"\s*-?\d+\s*", ref)):
        i = int(ref)
        if not 0 <= i < len(p.tracks):
            raise ToolError(f"track index {i} is out of range; {_tracks_text(p)}")
        return p.tracks[i]
    ref_s = str(ref).strip()
    for t in p.tracks:
        if t.id == ref_s or t.name.lower() == ref_s.lower():
            return t
    raise ToolError(f"no track {ref!r}; {_tracks_text(p)}")


def _tracks_text(p: Project) -> str:
    if not p.tracks:
        return "the project has no tracks (use add_track)"
    return "tracks are: " + ", ".join(f"{i}={t.name!r} (id {t.id})" for i, t in enumerate(p.tracks))


def _pitch(v, what: str = "pitch") -> int:
    try:
        p = note_to_midi(v)
    except (ValueError, TypeError):
        raise ToolError(f"invalid {what} {v!r}: use a MIDI number (60 = C4) or a note name like 'C4', 'F#3', "
                        f"'Bb5' (note letter, optional # or b, octave number)") from None
    if not 0 <= p <= 127:
        raise ToolError(f"{what} {v!r} is outside the MIDI range 0-127")
    return p


def _duration(v, what: str = "duration") -> float:
    try:
        d = parse_duration(v)
    except (ValueError, ZeroDivisionError, TypeError):
        raise ToolError(f"invalid {what} {v!r}: give beats as a number (1 = quarter note, 0.5 = eighth) or a note "
                        f"value like 'q', 'e.', 'h+q', 'e3'") from None
    if not (math.isfinite(d) and d > 0):
        raise ToolError(f"{what} must be a positive number of beats, got {v!r}")
    return float(d)


def _note_view(n: Note) -> dict:
    d = {"id": n.id, "onset": round(n.onset, 6), "duration": round(n.duration, 6), "pitch": n.pitch,
         "note": midi_to_note(n.pitch), "lyric": n.lyric}
    if n.phonemes:
        d["phonemes"] = n.phonemes
    if n.attrs:
        d["attrs"] = dict(n.attrs)
    return d


def _range_text(notes) -> str | None:
    ps = [n.pitch for n in notes]
    return f"{midi_to_note(min(ps))}-{midi_to_note(max(ps))}" if ps else None


def _style_overrides(t: Track) -> dict:
    return {k[6:]: v for k, v in t.voice_params.items() if k.startswith("style.")}


def _track_info(p: Project, t: Track) -> dict:
    return {
        "index": p.tracks.index(t), "id": t.id, "name": t.name, "voice": t.voice, "style": t.style,
        "language": t.language, "gain_db": t.gain_db, "pan": t.pan, "mute": t.mute, "solo": t.solo,
        "reverb_send": t.reverb_send, "seed": t.seed,
        "voice_params": {k: v for k, v in t.voice_params.items() if not k.startswith("style.")},
        "style_overrides": _style_overrides(t), "notes": len(t.notes), "range": _range_text(t.notes),
        "start_beat": round(min((n.onset for n in t.notes), default=0.0), 6), "end_beat": round(t.end, 6),
        "curves": sorted(k for k, c in t.curves.items() if c.points),
    }


def _select(track: Track, args: dict, default_all: bool = True) -> list[Note]:
    ids = args.get("ids")
    start, end = args.get("start_beat"), args.get("end_beat")
    if ids:
        by_id = {n.id: n for n in track.notes}
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ToolError(f"note id(s) not found in track {track.name!r}: {', '.join(missing[:10])}; "
                            f"call get_notes to see the current ids")
        sel = [by_id[i] for i in dict.fromkeys(ids)]
    elif start is not None or end is not None:
        lo = -math.inf if start is None else float(start)
        hi = math.inf if end is None else float(end)
        if hi <= lo:
            raise ToolError(f"end_beat ({end}) must be greater than start_beat ({start})")
        sel = [n for n in track.notes if n.onset >= lo - 1e-9 and n.onset < hi - 1e-9]
    elif default_all:
        sel = list(track.notes)
    else:
        raise ToolError("select notes with 'ids', or 'start_beat'/'end_beat', or pass all=true")
    return sorted(sel, key=lambda n: n.onset)


def _validate_attrs(attrs: dict, allow_null: bool = True) -> dict:
    out = {}
    for k, v in attrs.items():
        if k not in NOTE_ATTRIBUTES:
            close = difflib.get_close_matches(k, list(NOTE_ATTRIBUTES), n=1)
            raise ToolError(f"unknown note attribute {k!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
                            + "; valid: " + ", ".join(f"{a} ({d[1]})" for a, d in NOTE_ATTRIBUTES.items()))
        if v is None:
            if not allow_null:
                raise ToolError(f"attribute {k!r} needs a value")
            out[k] = None
            continue
        if k == "language":
            if v not in LANGUAGES:
                raise ToolError(f"attribute language must be one of {', '.join(LANGUAGES)}, got {v!r}")
            out[k] = v
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            try:
                v = float(v)
            except (TypeError, ValueError):
                raise ToolError(f"attribute {k!r} must be a number, got {v!r}") from None
        lo, hi = ATTR_RANGES.get(k, (-math.inf, math.inf))
        if not (math.isfinite(v) and lo <= v <= hi):
            raise ToolError(f"attribute {k!r}={v} is out of range {lo:g}..{hi:g} ({NOTE_ATTRIBUTES[k][1]})")
        out[k] = float(v)
    return out


def _apply_attrs(n: Note, attrs: dict):
    for k, v in attrs.items():
        if v is None:
            n.attrs.pop(k, None)
        else:
            n.attrs[k] = v


def _check_lyric(lyric: str) -> str:
    lyric = str(lyric)
    exp = lyric.strip()
    if exp.startswith("[") and exp.endswith("]"):
        try:
            parse_phoneme_string(exp[1:-1])
        except ValueError as exc:
            raise ToolError(f"{exc}; call list_phonemes for valid phoneme names") from None
    return lyric


def _check_phonemes(ph) -> str | None:
    if ph is None or str(ph).strip() == "":
        return None
    s = str(ph).strip().strip("[]")
    try:
        parse_phoneme_string(s)
    except ValueError as exc:
        raise ToolError(f"{exc}; call list_phonemes for valid phoneme names") from None
    return " ".join(re.split(r"[\s,]+", s))


def _check_voice(v: str) -> str:
    if v in VOICES:
        return v
    low = str(v).lower()
    for vid, voice in VOICES.items():
        if low in (vid, voice.name.lower()):
            return vid
    raise ToolError(f"unknown voice {v!r}; valid voices: "
                    + ", ".join(f"{vid} ({x.description})" for vid, x in VOICES.items()))


def _check_style(s: str) -> str:
    if s in STYLES:
        return s
    if str(s).lower() in STYLES:
        return str(s).lower()
    raise ToolError(f"unknown style {s!r}; valid styles: " + ", ".join(STYLES))


def _check_language(lang: str) -> str:
    if lang not in LANGUAGES:
        raise ToolError(f"unknown language {lang!r}; valid: en (English), ja (Japanese)")
    return lang


def _check_key(key: str) -> str:
    try:
        return composer.parse_key(key).name
    except ValueError as exc:
        raise ToolError(str(exc)) from None


def _key_of(session: Session, key: str | None):
    try:
        return composer.parse_key(key or session.project.key)
    except ValueError as exc:
        raise ToolError(f"{exc} (set a valid key with update_project or pass key=...)") from None


def _overlap_warnings(track: Track, new_ids: set[str] | None = None) -> list[str]:
    out = []
    for a, b in track.overlaps():
        if new_ids is None or a.id in new_ids or b.id in new_ids:
            out.append(f"notes {a.id} ({a.note_name} @ {a.onset:g}) and {b.id} ({b.note_name} @ {b.onset:g}) "
                       f"overlap; a voice sings one note at a time, so the first will be cut short")
    if len(out) > 10:
        out = out[:10] + [f"... and {len(out) - 10} more overlaps"]
    return out


def _voice_range_warnings(track: Track, notes: list[Note]) -> list[str]:
    v = VOICES.get(track.voice)
    if v is None:
        return []
    out = [n for n in notes if n.pitch < v.range_low - 2 or n.pitch > v.range_high + 2]
    if not out:
        return []
    return [f"{len(out)} note(s) are outside {v.name}'s comfortable range {midi_to_note(v.range_low)}-"
            f"{midi_to_note(v.range_high)} (e.g. {out[0].note_name} at beat {out[0].onset:g}); transpose them "
            f"or pick another voice"]


def _parse_ts(v) -> tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        num, den = v
    else:
        m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", str(v))
        if not m:
            raise ToolError(f"time_signature must look like '4/4', '3/4' or '6/8', got {v!r}")
        num, den = m.groups()
    num, den = int(num), int(den)
    if not 1 <= num <= 64 or den not in (1, 2, 4, 8, 16, 32):
        raise ToolError("time signature numerator must be 1-64 and the denominator a power of two (1-32)")
    return num, den


def _stem(p: Path) -> str:
    name = p.name
    for suf in (".s2p.json", ".json", ".s2p", ".s2s", ".svp", ".mid", ".midi", ".musicxml", ".xml", ".mxl", ".ust",
                ".txt", ".wav"):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return p.stem


def _is_native(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith((".json", ".s2p", ".s2s"))


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s.strip()).strip("._") or "untitled"


def _load_any(path: Path) -> Project:
    if not path.exists():
        raise ToolError(f"file not found: {path}")
    name = path.name.lower()
    if name.endswith((".s2s", ".txt")):
        try:
            return score_mod.parse_score(path.read_text(encoding="utf-8"))
        except score_mod.ScoreError as exc:
            raise ToolError(f"{path.name}: {exc}") from None
    if name.endswith((".json", ".s2p")):
        try:
            return Project.from_json(path.read_text(encoding="utf-8"))
        except (ValueError, KeyError, TypeError) as exc:
            raise ToolError(f"{path.name} is not a valid Synth2 project: {exc}") from None
    try:
        from ..io import load_any
    except ImportError:
        load_any = None
    if load_any is not None:
        return load_any(path)
    return Project.load(path)


_FORMAT_EXT = {"json": ".s2p.json", "svp": ".svp", "midi": ".mid", "musicxml": ".musicxml", "ust": ".ust",
               "score": ".s2s"}
_EXT_FORMAT = {".s2p.json": "json", ".json": "json", ".s2p": "json", ".svp": "svp", ".mid": "midi", ".midi": "midi",
               ".musicxml": "musicxml", ".xml": "musicxml", ".mxl": "musicxml", ".ust": "ust", ".s2s": "score",
               ".txt": "score"}


def _format_of(path: Path) -> str | None:
    n = path.name.lower()
    for ext in sorted(_EXT_FORMAT, key=len, reverse=True):
        if n.endswith(ext):
            return _EXT_FORMAT[ext]
    return None


def _write_any(project: Project, path: Path, fmt: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        project.save(path)
        return
    if fmt == "score":
        path.write_text(score_mod.project_to_score(project), encoding="utf-8")
        return
    try:
        from ..io import export_any
    except ImportError:
        raise ToolError(f"exporting {fmt} is not available yet (synth2.io.export_any is missing)") from None
    export_any(project, path)


def _fingerprint(project: Project) -> str:
    return json.dumps(project.to_dict(), sort_keys=True, default=str)


# ============================================================================ tools: read

@_tool("get_project", """
Read the current project. Times are in beats (quarter notes, beat 0 = song start).
detail='summary' (default): title, tempo map, key, time signature, length in beats and seconds, and one entry per
track (index, id, name, voice, style, note count, pitch range, curves, lyrics preview).
detail='full': everything, including every note (id, onset, duration, pitch + note name, lyric, phonemes, attrs) and
curve points; optionally only one `track` and only notes with onset in [start_beat, end_beat).
detail='score': the project (or one `track`) as score-notation text - the compact language write_melody accepts and
the easiest way to review a melody. With start_beat/end_beat it returns just those notes as a snippet.
""", {
    "detail": {"type": "string", "enum": ["summary", "full", "score"], "description": "How much to return."},
    "track": _TRACK_REF, "start_beat": _START, "end_beat": _END,
})
def _get_project(s: Session, a: dict) -> dict:
    p = s.project
    detail = a.get("detail", "summary")
    if detail == "score":
        if a.get("start_beat") is not None or a.get("end_beat") is not None:
            t = _track(s, a.get("track"))
            notes = _select(t, a)
            start = notes[0].onset if notes else float(a.get("start_beat") or 0.0)
            lines = [f"@ {score_mod._num(start)}"] + score_mod.notes_to_score(notes, start, p.beats_per_bar())
            return {"track": t.name, "score": "\n".join(lines), "notes": len(notes)}
        tracks = [_track(s, a["track"])] if a.get("track") is not None else None
        return {"score": score_mod.project_to_score(p, tracks)}
    out = p.summary()
    out.update({
        "tempo": p.tempo.to_list(), "beats_per_bar": p.beats_per_bar(), "master": p.master.to_dict(),
        "sample_rate": p.sample_rate, "path": s.path, "last_render": s.last_render_path,
        "history": {"undo": len(s.undo_stack), "redo": len(s.redo_stack)},
    })
    try:
        out["scale"] = composer.scale_degrees(p.key)
    except ValueError:
        pass
    if detail == "summary":
        if a.get("track") is not None:
            out["track"] = _track_info(p, _track(s, a["track"]))
        return out
    tracks = [_track(s, a["track"])] if a.get("track") is not None else p.tracks
    full = []
    for t in tracks:
        info = _track_info(p, t)
        info["notes"] = [_note_view(n) for n in _select(t, a)]
        info["curves"] = {k: c.to_dict() for k, c in t.curves.items() if c.points}
        full.append(info)
    out["tracks"] = full
    return out


@_tool("get_notes", """
List the notes of one track (default: the first) as compact objects {id, onset, duration, pitch, note, lyric,
phonemes?, attrs?}; onset/duration in beats, pitch as MIDI number plus note name. Use the ids with edit_notes,
delete_notes, set_lyrics(ids=...), apply_expression(ids=...). Optional start_beat/end_beat select notes by onset.
format='score' returns the same notes as score-notation text instead.
""", {"track": _TRACK_REF, "start_beat": _START, "end_beat": _END,
      "format": {"type": "string", "enum": ["list", "score"], "description": "'list' (default) or 'score'."}})
def _get_notes(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    notes = _select(t, a)
    out = {"track": t.name, "track_id": t.id, "count": len(notes), "range": _range_text(notes)}
    if a.get("format") == "score":
        start = notes[0].onset if notes else 0.0
        out["score"] = "\n".join([f"@ {score_mod._num(start)}"]
                                 + score_mod.notes_to_score(notes, start, s.project.beats_per_bar()))
    else:
        out["notes"] = [_note_view(n) for n in notes]
    return out


@_tool("list_voices", """
List the available singer voices: id, name, description, gender, comfortable range (note names) and the expression
style each voice defaults to. Keep melodies inside a voice's range (a few semitones beyond is possible but strained).
""")
def _list_voices_tool(s: Session, a: dict) -> dict:
    return {"voices": _list_voices()}


@_tool("list_styles", """
List the automatic-expression styles (pop, ballad, rock, jazz, rnb, robotic, soft, classical, flat). A style
generates portamento, scoops, falls, vibrato, drift, phrase dynamics and breaths from the notes. Also returns the
style parameter names you can override per track with apply_expression(style_overrides=...). Set
include_parameters=true to get every style's full parameter values.
""", {"include_parameters": {"type": "boolean", "description": "Include each style's parameter values."}})
def _list_styles_tool(s: Session, a: dict) -> dict:
    styles = []
    for st in STYLES.values():
        d = {"name": st.name, "description": st.description}
        if a.get("include_parameters"):
            d["parameters"] = {k: getattr(st, k) for k in STYLE_FIELDS}
        styles.append(d)
    return {"styles": styles, "override_parameters": STYLE_FIELD_DOCS}


_PHONEME_EXAMPLES = {
    "m": "me", "n": "no", "ng": "sing", "l": "low", "r": "red", "dx": "butter (flap) / JA ra", "w": "we", "y": "yes",
    "s": "see", "z": "zoo", "sh": "she", "zh": "measure", "f": "fee", "v": "vee", "th": "thin", "dh": "this",
    "hh": "he", "ph": "JA fu", "c": "JA hi", "p": "pea", "b": "be", "t": "tea", "d": "do", "k": "key", "g": "go",
    "ch": "chew", "jh": "joy", "ts": "JA tsu", "dz": "JA zu", "q": "glottal stop / JA small tsu", "sil": "silence",
    "br": "breath intake",
}
_EN_PHONES = set(english.ARPA_MAP.values()) | {"ax", "dx", "q", "sil", "br"}
_JA_PHONES = {"a", "i", "u", "e", "o", "N", "k", "g", "s", "z", "t", "d", "n", "hh", "b", "p", "m", "y", "dx", "w",
              "sh", "ch", "ts", "dz", "jh", "ph", "v", "c", "q", "sil", "br"}


@_tool("list_phonemes", """
List the phoneme inventory used for explicit pronunciations: lyric '[s t aa r]' or a note's phonemes override.
English uses lower-case ARPABET (vowels aa ae ah ao aw ax ay eh er ey ih iy ow oy uh uw; consonants like s t k
sh th dh ng ...); Japanese uses a i u e o N plus consonants. language='en' | 'ja' | 'all' (default 'all').
""", {"language": {"type": "string", "enum": ["en", "ja", "all"], "description": "Filter by language."}})
def _list_phonemes_tool(s: Session, a: dict) -> dict:
    lang = a.get("language", "all")
    out = []
    for name, ph in PHONEMES.items():
        langs = [x for x, pool in (("en", _EN_PHONES), ("ja", _JA_PHONES)) if name in pool]
        if lang != "all" and lang not in langs:
            continue
        out.append({"name": name, "kind": ph.kind, "voiced": ph.voiced, "languages": langs,
                    "example": ph.description or _PHONEME_EXAMPLES.get(name, ""),
                    **({"parts": list(ph.parts)} if ph.parts else {})})
    return {"phonemes": out, "usage": "Lyric '[hh eh l ow]' sings those phonemes; or keep the lyric and set the "
                                      "note's phonemes field (edit_notes) to override pronunciation."}


@_tool("list_parameters", """
The studio's parameter reference: automation curves (name, default, min, max, unit, meaning) for set_curve; note
attributes for add_notes/edit_notes/apply_expression; voice parameters for update_track/apply_expression
voice_params; style override parameters; lyric conventions and duration tokens. Read this once before detailed
expression work.
""")
def _list_parameters_tool(s: Session, a: dict) -> dict:
    curves = {k: {"default": v[0], "min": v[1], "max": v[2], "unit": v[3], "description": v[4]}
              for k, v in CURVE_SPECS.items()}
    attrs = {k: {"default": v[0], "description": v[1],
                 **({"min": ATTR_RANGES[k][0], "max": ATTR_RANGES[k][1]} if k in ATTR_RANGES else {})}
             for k, v in NOTE_ATTRIBUTES.items()}
    base = VOICES["aurora"]
    voice_params = {k: {"description": VOICE_PARAM_DOCS.get(k, ""), "aurora_value": getattr(base, k)}
                    for k in VOICE_PARAM_FIELDS}
    return {
        "curves": curves, "note_attributes": attrs, "voice_params": voice_params,
        "style_overrides": STYLE_FIELD_DOCS, "lyrics": LYRIC_CONVENTIONS,
        "durations": "beats as numbers (1 = quarter) or tokens: w=4 h=2 q=1 e=0.5 s=0.25 t=0.125; dot 'q.'=1.5; "
                     "triplet 'e3'=1/3; tie 'h+q'=3",
        "interpolations": list(score_mod.INTERPOLATIONS),
    }


@_tool("phonemize", """
Show how text will be pronounced and split into notes. For each word: phonemes, syllables (one per note),
whether it is in the pronunciation dictionary (words that are not are guessed from spelling - consider explicit
phonemes), plus `lyric_tokens`: exactly the per-note tokens set_lyrics would assign ('+' marks continuation
syllables of a multi-syllable word). language 'en' (default) or 'ja'.
""", {"text": {"type": "string", "description": "Lyrics text, e.g. 'twinkle twinkle little star'."},
      "language": {"type": "string", "enum": ["en", "ja"], "description": "Language (default en)."}},
    required=("text",))
def _phonemize_tool(s: Session, a: dict) -> dict:
    lang = a.get("language", "en")
    text = a["text"]
    words = []
    for raw in re.findall(r"\[[^\]]*\]|\S+", text):
        if raw.startswith("["):
            words.append({"word": raw, "phonemes": " ".join(lyric_phonemes(raw, lang)), "explicit": True})
            continue
        if lang == "ja" or japanese.is_kana(raw):
            morae = japanese.morae(raw)
            words.append({"word": raw, "phonemes": " ".join(japanese.g2p(raw)),
                          "syllables": [" ".join(japanese.romaji_to_phonemes(m)) for m in morae if m != "-"],
                          "morae": morae})
            continue
        for w in re.findall(r"[A-Za-z']+", raw):
            phones = english.g2p_word(w)
            sylls = english.syllabify(phones)
            words.append({"word": w, "phonemes": " ".join(phones), "syllables": [" ".join(x) for x in sylls],
                          "in_dictionary": english.lookup(w) is not None})
    try:
        tokens = composer.lyric_tokens(text, lang)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    return {"words": words, "lyric_tokens": tokens, "notes_needed": len(tokens)}


# ============================================================================ tools: project

@_tool("new_project", """
Start a new project (undoable), replacing the current one. Either give `score` (full score-notation text with
title/tempo/key/track lines - see write_melody for the note syntax) or the basic settings: title, bpm, key
('C major', 'A minor', 'F# dorian'...), time_signature ('4/4'), voice and language for the first track 'Lead'.
""", {
    "title": {"type": "string"}, "bpm": {"type": "number", "minimum": 20, "maximum": 400},
    "key": {"type": "string", "description": "e.g. 'C major', 'A minor', 'Bb major', 'D dorian'."},
    "time_signature": {"type": "string", "description": "e.g. '4/4', '3/4', '6/8'."},
    "voice": {"type": "string", "description": "Voice id for the first track (see list_voices)."},
    "language": {"type": "string", "enum": list(LANGUAGES)},
    "score": {"type": "string", "description": "Complete score-notation text; overrides the other settings."},
}, mutating=True)
def _new_project(s: Session, a: dict) -> dict:
    warnings: list[str] = []
    if a.get("score"):
        try:
            p, warnings = score_mod.parse_score_ex(a["score"])
        except score_mod.ScoreError as exc:
            raise ToolError(f"score error: {exc}") from None
    else:
        voice = _check_voice(a.get("voice", "aurora"))
        lang = _check_language(a.get("language", "en"))
        p = Project.new(a.get("title", "Untitled"), bpm=float(a.get("bpm", 120.0)), voice=voice, language=lang)
        p.tracks[0].style = VOICES[voice].default_style
        if a.get("key"):
            p.key = _check_key(a["key"])
        if a.get("time_signature"):
            p.time_signature = _parse_ts(a["time_signature"])
    s.project = p
    s.path = None
    s.last_render = None
    s.last_render_path = None
    out = p.summary()
    if warnings:
        out["warnings"] = warnings
    return out


@_tool("update_project", """
Change project settings: title, key (used by add_harmony and diatonic transpose; e.g. 'A minor', 'Eb major'),
time_signature ('3/4'), sample_rate (Hz) and master settings {gain_db, reverb_room 0..1 (room size), reverb_wet
0..2 (scales all reverb sends), normalize (peak-normalise to -1 dBFS), limiter}. Tempo is set with set_tempo.
""", {
    "title": {"type": "string"},
    "key": {"type": "string"},
    "time_signature": {"type": "string"},
    "sample_rate": {"type": "integer", "minimum": 8000, "maximum": 192000},
    "master": {"type": "object", "additionalProperties": False, "properties": {
        "gain_db": {"type": "number", "minimum": -48, "maximum": 24},
        "reverb_room": {"type": "number", "minimum": 0, "maximum": 1},
        "reverb_wet": {"type": "number", "minimum": 0, "maximum": 2},
        "normalize": {"type": "boolean"}, "limiter": {"type": "boolean"}}},
}, mutating=True)
def _update_project(s: Session, a: dict) -> dict:
    p = s.project
    if "title" in a:
        p.title = a["title"]
    if "key" in a:
        p.key = _check_key(a["key"])
    if "time_signature" in a:
        p.time_signature = _parse_ts(a["time_signature"])
    if "sample_rate" in a:
        p.sample_rate = int(a["sample_rate"])
    for k, v in (a.get("master") or {}).items():
        setattr(p.master, k, v)
    try:
        scale = composer.scale_degrees(p.key)
    except ValueError:
        scale = None
    return {"title": p.title, "key": p.key, "scale": scale,
            "time_signature": f"{p.time_signature[0]}/{p.time_signature[1]}", "sample_rate": p.sample_rate,
            "master": p.master.to_dict()}


@_tool("set_tempo", """
Set the tempo in BPM (quarter notes per minute). beat=0 (default) sets the song tempo; another beat adds a tempo
change taking effect at that beat (note positions stay in beats, so their timing in seconds changes).
remove=true deletes the tempo change at `beat`; replace_all=true clears all changes first.
""", {
    "bpm": {"type": "number", "minimum": 20, "maximum": 400, "description": "Beats per minute."},
    "beat": {"type": "number", "minimum": 0, "description": "Where the tempo takes effect (default 0)."},
    "remove": {"type": "boolean", "description": "Remove the tempo change at `beat` instead."},
    "replace_all": {"type": "boolean", "description": "Clear every tempo change, then set bpm at beat 0."},
}, mutating=True)
def _set_tempo(s: Session, a: dict) -> dict:
    tm = s.project.tempo
    beat = float(a.get("beat", 0.0))
    if a.get("remove"):
        if beat <= 0:
            raise ToolError("cannot remove the initial tempo at beat 0; set a new bpm instead")
        pts = [p for p in tm.points if abs(p.beat - beat) > 1e-9]
        if len(pts) == len(tm.points):
            raise ToolError(f"no tempo change at beat {beat:g}; tempo map: {tm.to_list()}")
        tm.points = pts
        tm._rebuild()
    else:
        if "bpm" not in a:
            raise ToolError("missing required argument 'bpm' (beats per minute, e.g. 96)")
        if a.get("replace_all"):
            tm.points = [TempoPoint(0.0, float(a["bpm"]))]
            tm._rebuild()
        tm.set(beat if not a.get("replace_all") else 0.0, float(a["bpm"]))
    return {"tempo": tm.to_list(), "length_seconds": round(s.project.duration_seconds(), 2)}


# ============================================================================ tools: tracks

_TRACK_SETTINGS = {
    "name": {"type": "string"},
    "voice": {"type": "string", "description": "Voice id (see list_voices), e.g. aurora, seraph, atlas, ember, nova, pixel, basso."},
    "style": {"type": "string", "description": "Expression style (see list_styles). Default: the voice's own style."},
    "language": {"type": "string", "enum": list(LANGUAGES), "description": "Lyrics language: en or ja."},
    "gain_db": {"type": "number", "minimum": -60, "maximum": 24, "description": "Track volume in dB (0 = unity)."},
    "pan": {"type": "number", "minimum": -1, "maximum": 1, "description": "-1 = left, 0 = centre, 1 = right."},
    "reverb_send": {"type": "number", "minimum": 0, "maximum": 1, "description": "Reverb send 0..1 (default 0.25)."},
    "mute": {"type": "boolean"}, "solo": {"type": "boolean"},
    "voice_params": {"type": "object", "description": "Voice tweaks merged into the track, e.g. {\"breathiness\": 0.3, "
                     "\"tension\": -0.2}; null removes a key. See list_parameters for names."},
}


def _apply_voice_params(t: Track, vp: dict):
    for k, v in vp.items():
        if k not in VOICE_PARAM_FIELDS:
            close = difflib.get_close_matches(k, VOICE_PARAM_FIELDS, n=1)
            raise ToolError(f"unknown voice parameter {k!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
                            + f"; valid: {', '.join(VOICE_PARAM_FIELDS)}")
        if v is None:
            t.voice_params.pop(k, None)
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ToolError(f"voice parameter {k!r} must be a number, got {v!r}")
        if k in ("tension", "breathiness", "robotic") and not -1 <= v <= 1:
            raise ToolError(f"voice parameter {k!r} must be between -1 and 1")
        t.voice_params[k] = float(v)


def _apply_track_settings(t: Track, a: dict):
    if "name" in a:
        if not str(a["name"]).strip():
            raise ToolError("track name cannot be empty")
        t.name = str(a["name"]).strip()
    if "voice" in a:
        t.voice = _check_voice(a["voice"])
    if "style" in a:
        t.style = _check_style(a["style"])
    if "language" in a:
        t.language = _check_language(a["language"])
    for k in ("gain_db", "pan", "reverb_send"):
        if k in a:
            setattr(t, k, float(a[k]))
    for k in ("mute", "solo"):
        if k in a:
            setattr(t, k, bool(a[k]))
    if "seed" in a:
        t.seed = int(a["seed"])
    if a.get("voice_params"):
        _apply_voice_params(t, a["voice_params"])


@_tool("add_track", """
Add a new empty vocal track (one monophonic singer). Defaults: voice 'aurora', the voice's own style, language
'en', gain 0 dB, centre pan, reverb send 0.25. Returns the track's index and id; then fill it with write_melody or
add_notes. For a harmony part derived from an existing melody use add_harmony instead.
""", dict(_TRACK_SETTINGS), mutating=True)
def _add_track(s: Session, a: dict) -> dict:
    p = s.project
    voice = _check_voice(a.get("voice", "aurora"))
    name = str(a.get("name") or f"Track {len(p.tracks) + 1}").strip()
    t = p.add_track(name, voice=voice, language=_check_language(a.get("language", "en")))
    t.style = VOICES[voice].default_style
    _apply_track_settings(t, {k: v for k, v in a.items() if k not in ("voice", "language", "name")})
    return {"track": _track_info(p, t)}


@_tool("update_track", """
Change a track's settings (only the given fields): name, voice, style, language, gain_db, pan, reverb_send, mute,
solo, seed (random seed for the humanising variations), voice_params (merged; null removes a key).
""", {"track": _TRACK_REF, **_TRACK_SETTINGS, "seed": {"type": "integer"}}, required=("track",), mutating=True)
def _update_track(s: Session, a: dict) -> dict:
    t = _track(s, a["track"])
    _apply_track_settings(t, a)
    return {"track": _track_info(s.project, t)}


@_tool("remove_track", "Delete a track and all its notes and curves (undoable).",
       {"track": _TRACK_REF}, required=("track",), mutating=True)
def _remove_track(s: Session, a: dict) -> dict:
    t = _track(s, a["track"])
    s.project.tracks.remove(t)
    return {"removed": t.name, "tracks": [{"index": i, "id": x.id, "name": x.name}
                                          for i, x in enumerate(s.project.tracks)]}


@_tool("duplicate_track", """
Copy a track (notes, lyrics, curves, settings) into a new track - e.g. for a double-tracked lead or a unison/octave
part. Options: name, voice, transpose (semitones, e.g. -12 for an octave below), detune_cents (adds a small static
detune to every note for a natural doubling, e.g. 6), delay_beats (tiny timing offset, e.g. 0.02), gain_db, pan.
""", {"track": _TRACK_REF, "name": {"type": "string"}, "voice": {"type": "string"},
      "transpose": {"type": "integer", "minimum": -48, "maximum": 48},
      "detune_cents": {"type": "number", "minimum": -100, "maximum": 100},
      "delay_beats": {"type": "number", "minimum": -1, "maximum": 1},
      "gain_db": {"type": "number", "minimum": -60, "maximum": 24},
      "pan": {"type": "number", "minimum": -1, "maximum": 1}},
    required=("track",), mutating=True)
def _duplicate_track(s: Session, a: dict) -> dict:
    p = s.project
    src = _track(s, a["track"])
    t = Track.from_dict(copy.deepcopy(src.to_dict()))
    t.id = _new_id()
    t.name = a.get("name") or f"{src.name} (copy)"
    t.seed = len(p.tracks) * 7919 + 17
    t.solo = False
    if a.get("voice"):
        t.voice = _check_voice(a["voice"])
    try:
        t.notes = composer.double(src.notes, float(a.get("detune_cents", 0.0)), float(a.get("delay_beats", 0.0)))
        if a.get("transpose"):
            t.notes = composer.transpose(t.notes, int(a["transpose"]))
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    for n in t.notes:
        if not n.attrs.get("detune"):
            n.attrs.pop("detune", None)
    for k in ("gain_db", "pan"):
        if k in a:
            setattr(t, k, float(a[k]))
    p.tracks.append(t)
    return {"track": _track_info(p, t), "warnings": _voice_range_warnings(t, t.notes)}


# ============================================================================ tools: notes

_NOTE_SPEC = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "pitch": _PITCH,
        "onset": {"type": "number", "description": "Start in beats."},
        "duration": {"type": ["number", "string"], "description": "Length in beats (1 = quarter) or a note value like 'q', 'e.', 'h+q'."},
        "lyric": {"type": "string", "description": "One syllable/word, '+', '-', 'br' or '[phonemes]'. Default 'la'."},
        "phonemes": {"type": "string", "description": "Optional explicit pronunciation override, e.g. 's t aa r'."},
        "attrs": {"type": "object", "description": "Optional note attributes, e.g. {\"vibrato_depth\": 0.5, \"velocity\": 1.2}."},
    },
    "required": ["pitch", "onset", "duration"],
}


@_tool("add_notes", f"""
Add notes to a track. Each note: pitch (MIDI number or note name 'C4'), onset (beats from song start), duration
(beats, or a note value 'q', 'e.', 'h+q'), lyric (default 'la'), optional phonemes override and attrs. A track is
one singer, so notes should not overlap: overwrite=true first deletes existing notes that overlap the new ones.
{LYRIC_CONVENTIONS}
Example: notes=[{{"pitch":"E4","onset":4,"duration":1,"lyric":"hello"}},{{"pitch":"G4","onset":5,"duration":2,"lyric":"+"}}].
For whole melodies write_melody (score notation) is usually faster.
""", {"track": _TRACK_REF, "notes": {"type": "array", "items": _NOTE_SPEC, "minItems": 1},
      "overwrite": {"type": "boolean", "description": "Delete existing notes that overlap the new notes."}},
    required=("notes",), mutating=True)
def _add_notes(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    new = []
    for i, spec in enumerate(a["notes"]):
        onset = float(spec["onset"])
        if onset < 0:
            raise ToolError(f"notes[{i}].onset must be >= 0 (beats), got {onset}")
        n = Note(onset, _duration(spec["duration"], f"notes[{i}].duration"), _pitch(spec["pitch"], f"notes[{i}].pitch"),
                 _check_lyric(spec.get("lyric", "la")), _check_phonemes(spec.get("phonemes")),
                 {k: v for k, v in _validate_attrs(spec.get("attrs") or {}).items() if v is not None})
        new.append(n)
    removed = 0
    if a.get("overwrite"):
        keep = []
        for old in t.notes:
            if any(old.onset < n.end - 1e-9 and old.end > n.onset + 1e-9 for n in new):
                removed += 1
            else:
                keep.append(old)
        t.notes = keep
    t.notes.extend(new)
    t.sort()
    return {"track": t.name, "added": [_note_view(n) for n in new], "removed": removed,
            "warnings": _overlap_warnings(t, {n.id for n in new}) + _voice_range_warnings(t, new)}


@_tool("write_melody", """
Write a melody into a track using score notation - the fastest way to enter music. Notes are
'PITCH DURATION LYRIC' separated by spaces, placed one after another from start_beat (default: the end of the
track's last note, i.e. append; beat 0 for an empty track).
- PITCH: note name C4 (middle C), F#3, Bb5 or MIDI number; 'r DURATION' is a rest.
- DURATION: w=4 beats, h=2, q=1, e=1/2, s=1/4, t=1/8; dotted 'q.'=1.5; triplet 'e3'=1/3; tie 'h+q'=3; or a
  number of beats like 1.5.
- LYRIC: one token: syllable/word, '+' (next syllable of the previous word), '-' (melisma: hold the previous
  vowel), 'br' (breath), '[s t aa r]' explicit phonemes, "quoted text"; may be omitted (defaults to 'la', set real
  words later with set_lyrics). 'word[ph ph]' adds a phoneme override; '{velocity=1.2,vibrato_depth=0.6}' right
  after the lyric sets note attributes.
- '|' bar lines are optional (warned if not on a bar boundary - a cheap rhythm check); '@ 16' jumps to absolute
  beat 16 ('@ bar 5' to bar 5); new lines are just for readability; '#' starts a comment.
- 'curve: loudness 0:0 4:-3 (linear)' lines add automation (absolute beats); 'tempo: 90' adds a tempo change here.
Example: "@ bar 2  E4 q twinkle  E4 q +  B4 q twinkle  B4 q +  | C#5 q little  C#5 q +  B4 h star | r q ..."
('twinkle' then '+' spreads the word over two notes - always write the whole word on its first note).
replace=true first deletes existing notes overlapping the written span. Returns the new note ids, the span and
end_beat (where the next phrase would start), plus warnings to check.
""", {"track": _TRACK_REF,
      "score": {"type": "string", "description": "Score-notation note lines."},
      "start_beat": {"type": "number", "minimum": 0, "description": "Beat where the first note starts (default: append)."},
      "replace": {"type": "boolean", "description": "Delete existing notes overlapping the written span first."},
      "language": {"type": "string", "enum": list(LANGUAGES), "description": "Lyric language (default: the track's)."}},
    required=("score",), mutating=True)
def _write_melody(s: Session, a: dict) -> dict:
    p = s.project
    t = _track(s, a.get("track"))
    start = float(a["start_beat"]) if a.get("start_beat") is not None else t.end
    lang = a.get("language") or t.language
    try:
        snip = score_mod.parse_snippet(a["score"], start, lang, p.beats_per_bar())
    except score_mod.ScoreError as exc:
        raise ToolError(f"score error: {exc}") from None
    if not snip.notes and not snip.curves and not snip.tempos:
        raise ToolError("the score contained no notes; write notes like 'C4 q la  D4 q la  E4 h la'")
    removed = 0
    if snip.notes:
        lo = min(n.onset for n in snip.notes)
        hi = max(n.end for n in snip.notes)
        if a.get("replace"):
            before = len(t.notes)
            t.notes = [n for n in t.notes if not (n.onset < hi - 1e-9 and n.end > lo + 1e-9)]
            removed = before - len(t.notes)
    t.notes.extend(snip.notes)
    t.sort()
    for name, c in snip.curves.items():
        cur = t.curve(name)
        for b, v in c.points:
            cur.add(b, v)
        if c.interpolation != "cubic":
            cur.interpolation = c.interpolation
    for beat, bpm in snip.tempos:
        p.tempo.set(beat, bpm)
    ids = {n.id for n in snip.notes}
    warnings = list(snip.warnings) + _overlap_warnings(t, ids) + _voice_range_warnings(t, snip.notes)
    out = {
        "track": t.name, "added": len(snip.notes), "removed": removed,
        "note_ids": [n.id for n in snip.notes],
        "start_beat": round(min((n.onset for n in snip.notes), default=start), 6),
        "end_beat": round(snip.end_beat, 6), "range": _range_text(snip.notes),
        "lyrics": " ".join(n.lyric for n in snip.notes)[:300],
        "warnings": warnings,
    }
    if snip.notes and all(n.lyric == score_mod.DEFAULT_LYRIC for n in snip.notes):
        out["hint"] = "all lyrics are the default 'la'; call set_lyrics with the words for this span"
    if snip.curves:
        out["curves"] = sorted(snip.curves)
    if snip.tempos:
        out["tempo"] = p.tempo.to_list()
    return out


_CHANGE_PROPS = {
    "pitch": _PITCH,
    "onset": {"type": "number", "description": "New start in beats."},
    "duration": {"type": ["number", "string"], "description": "New length in beats or a note value."},
    "lyric": {"type": "string"},
    "phonemes": {"type": ["string", "null"], "description": "Pronunciation override; null or '' clears it."},
    "attrs": {"type": "object", "description": "Attributes to merge; a null value deletes that attribute."},
    "move_by": {"type": "number", "description": "Shift the onset by this many beats (negative = earlier)."},
    "transpose": {"type": "integer", "description": "Shift pitch by semitones."},
}


def _apply_changes(n: Note, ch: dict, where: str):
    if "pitch" in ch:
        n.pitch = _pitch(ch["pitch"], f"{where}.pitch")
    if "transpose" in ch:
        n.pitch = _pitch(n.pitch + int(ch["transpose"]), f"{where} transposed pitch")
    if "onset" in ch:
        n.onset = float(ch["onset"])
    if "move_by" in ch:
        n.onset += float(ch["move_by"])
    if n.onset < 0:
        raise ToolError(f"{where}: onset would become negative ({n.onset:g}); beats start at 0")
    if "duration" in ch:
        n.duration = _duration(ch["duration"], f"{where}.duration")
    if "lyric" in ch:
        n.lyric = _check_lyric(ch["lyric"])
    if "phonemes" in ch:
        n.phonemes = _check_phonemes(ch["phonemes"])
    if ch.get("attrs"):
        _apply_attrs(n, _validate_attrs(ch["attrs"]))


@_tool("edit_notes", """
Edit existing notes. Two ways (combinable):
1) edits=[{id, pitch?, onset?, duration?, lyric?, phonemes?, attrs?, move_by?, transpose?}, ...] changes individual
   notes by id (ids come from get_notes).
2) changes={...same fields except onset...} applied to every note selected by ids or start_beat/end_beat (onset
   in [start_beat, end_beat)), e.g. changes={"attrs":{"velocity":1.3}} for a louder passage or {"move_by":-0.5}.
Units: beats for onset/duration/move_by, semitones for transpose. attrs merge (null deletes a key);
phonemes null/'' clears an override.
""", {"track": _TRACK_REF,
      "edits": {"type": "array", "items": {"type": "object", "additionalProperties": False,
                                           "properties": {"id": {"type": "string"}, **_CHANGE_PROPS},
                                           "required": ["id"]}},
      "ids": _IDS, "start_beat": _START, "end_beat": _END,
      "changes": {"type": "object", "additionalProperties": False,
                  "properties": {k: v for k, v in _CHANGE_PROPS.items() if k != "onset"}}},
    mutating=True)
def _edit_notes(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    if not a.get("edits") and not a.get("changes"):
        raise ToolError("nothing to do: pass edits=[{id, ...}] and/or changes={...} with ids or a beat range")
    touched: dict[str, Note] = {}
    by_id = {n.id: n for n in t.notes}
    for i, e in enumerate(a.get("edits") or []):
        n = by_id.get(e["id"])
        if n is None:
            raise ToolError(f"edits[{i}]: note id {e['id']!r} not found in track {t.name!r}; call get_notes for ids")
        _apply_changes(n, e, f"edits[{i}]")
        touched[n.id] = n
    if a.get("changes"):
        if not (a.get("ids") or a.get("start_beat") is not None or a.get("end_beat") is not None):
            raise ToolError("changes={...} needs a selection: ids=[...] or start_beat/end_beat")
        for n in _select(t, a, default_all=False):
            _apply_changes(n, a["changes"], f"note {n.id}")
            touched[n.id] = n
    t.sort()
    return {"track": t.name, "edited": [_note_view(n) for n in touched.values()],
            "warnings": _overlap_warnings(t, set(touched)) + _voice_range_warnings(t, list(touched.values()))}


@_tool("delete_notes", """
Delete notes from a track by ids, by beat range (notes whose onset is in [start_beat, end_beat)), or all=true.
""", {"track": _TRACK_REF, "ids": _IDS, "start_beat": _START, "end_beat": _END,
      "all": {"type": "boolean", "description": "Delete every note in the track."}}, mutating=True)
def _delete_notes(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    sel = list(t.notes) if a.get("all") else _select(t, a, default_all=False)
    ids = {n.id for n in sel}
    t.notes = [n for n in t.notes if n.id not in ids]
    return {"track": t.name, "deleted": len(ids), "ids": sorted(ids), "remaining": len(t.notes)}


@_tool("set_lyrics", f"""
Set lyrics from free text, one syllable per note, over the notes selected by ids or start_beat/end_beat (default:
the whole track), in time order. Multi-syllable words are split automatically ('twinkle' -> 'twinkle' '+'), so
'twinkle twinkle little star' needs 7 notes; call phonemize first to see the exact token count. If the text has
more syllables than notes, the extra words are sung together on the last note; if fewer, the remaining notes get
'-' (melisma). Notes whose lyric is 'br' are skipped (kept as breaths) unless keep_breaths=false.
You can embed tokens directly: '-' (melisma), '+' (continue the word), 'br', '[phonemes]', and hyphenate words
yourself ('beau-ti-ful') to control the split. Japanese: kana or romaji, one mora per note.
""", {"track": _TRACK_REF, "text": {"type": "string", "description": "The words to sing."},
      "ids": _IDS, "start_beat": _START, "end_beat": _END,
      "language": {"type": "string", "enum": list(LANGUAGES), "description": "Default: the track's language."},
      "keep_breaths": {"type": "boolean", "description": "Skip notes whose lyric is 'br' (default true)."}},
    required=("text",), mutating=True)
def _set_lyrics(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    sel = _select(t, a)
    if not sel:
        raise ToolError(f"no notes selected in track {t.name!r}; write notes first or check start_beat/end_beat")
    lang = a.get("language") or t.language
    keep_breaths = a.get("keep_breaths", True)
    try:
        tokens = composer.lyric_tokens(a["text"], lang)
        fitted = composer.fit_lyrics(sel, a["text"], lang, keep_breaths=keep_breaths)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    by_id = {n.id: n for n in fitted}
    for n in t.notes:
        if n.id in by_id:
            n.lyric, n.phonemes = by_id[n.id].lyric, by_id[n.id].phonemes
    targets = [n for n in fitted if not (keep_breaths and n.lyric.strip().lower() in ("br", "breath"))]
    out = {"track": t.name, "notes": len(targets), "syllables": len(tokens),
           "assigned": [{"id": n.id, "onset": round(n.onset, 6), "note": n.note_name, "lyric": n.lyric} for n in fitted]}
    if len(tokens) > len(targets):
        out["warning"] = (f"{len(tokens)} syllables for {len(targets)} notes: the last note sings "
                          f"{targets[-1].lyric!r}; add notes or shorten the text")
    elif len(tokens) < len(targets):
        out["info"] = f"{len(targets) - len(tokens)} note(s) at the end got '-' (melisma) because the text was shorter"
    return out


@_tool("transpose", """
Transpose notes of a track (selected by ids or start_beat/end_beat; default all). semitones shifts chromatically
(12 = one octave up). steps shifts diatonically within the key (project key unless `key` is given): steps=1 moves
each note to the next scale tone, keeping the melody in key.
""", {"track": _TRACK_REF, "semitones": {"type": "integer", "minimum": -60, "maximum": 60},
      "steps": {"type": "integer", "minimum": -21, "maximum": 21}, "key": {"type": "string"},
      "ids": _IDS, "start_beat": _START, "end_beat": _END}, mutating=True)
def _transpose(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    if not a.get("semitones") and not a.get("steps"):
        raise ToolError("give semitones (chromatic, 12 = octave) and/or steps (diatonic scale steps)")
    sel = _select(t, a)
    key = _key_of(s, a.get("key")) if a.get("steps") else None
    try:
        moved = composer.transpose(sel, int(a.get("semitones", 0)), steps=int(a.get("steps", 0)), key=key)
    except ValueError as exc:
        raise ToolError(str(exc)) from None
    by_id = {n.id: n for n in moved}
    for n in t.notes:
        if n.id in by_id:
            n.pitch = by_id[n.id].pitch
    return {"track": t.name, "transposed": len(moved), "range": _range_text(moved),
            "warnings": _voice_range_warnings(t, moved)}


@_tool("quantize", """
Snap note timing to a grid. grid in beats (0.25 = sixteenth, 0.5 = eighth, 1/3 = eighth triplet) or a note value
('s', 'e', 'e3'). strength 0..1 (1 = fully on the grid). ends=true (default) also snaps note ends (lengths).
Notes never become zero-length and overlaps created by snapping are resolved (the earlier note is shortened).
Selection by ids or start_beat/end_beat; default the whole track.
""", {"track": _TRACK_REF, "grid": {"type": ["number", "string"], "description": "Grid size in beats or note value."},
      "strength": {"type": "number", "minimum": 0, "maximum": 1}, "ends": {"type": "boolean"},
      "ids": _IDS, "start_beat": _START, "end_beat": _END}, required=("grid",), mutating=True)
def _quantize(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    grid = _duration(a["grid"], "grid")
    sel = _select(t, a)
    q = composer.quantize(sel, grid, float(a.get("strength", 1.0)), ends=a.get("ends", True))
    by_id = {n.id: n for n in q}
    changed = 0
    for n in t.notes:
        if n.id in by_id:
            m = by_id[n.id]
            if abs(m.onset - n.onset) > 1e-9 or abs(m.duration - n.duration) > 1e-9:
                changed += 1
            n.onset, n.duration = m.onset, m.duration
    before = {n.id: (n.onset, n.duration) for n in t.notes}
    composer.resolve_overlaps(t.notes, min_length=min(grid, 0.0625))
    fixed = sum(1 for n in t.notes
                if abs(before[n.id][0] - n.onset) > 1e-9 or abs(before[n.id][1] - n.duration) > 1e-9)
    return {"track": t.name, "quantized": len(sel), "changed": changed, "overlaps_resolved": fixed, "grid": grid}


# ============================================================================ tools: curves & expression

def _points(raw: list, name: str) -> list[tuple[float, float]]:
    spec = CURVE_SPECS[name]
    out = []
    for i, pt in enumerate(raw):
        if isinstance(pt, dict):
            if "beat" not in pt or "value" not in pt:
                raise ToolError(f"points[{i}] must be [beat, value] or {{\"beat\": b, \"value\": v}}")
            b, v = pt["beat"], pt["value"]
        elif isinstance(pt, (list, tuple)) and len(pt) == 2:
            b, v = pt
        else:
            raise ToolError(f"points[{i}] must be [beat, value], got {pt!r}")
        try:
            b, v = float(b), float(v)
        except (TypeError, ValueError):
            raise ToolError(f"points[{i}] must contain numbers, got {pt!r}") from None
        if not (math.isfinite(b) and math.isfinite(v)):
            raise ToolError(f"points[{i}] must be finite numbers")
        if not spec[1] <= v <= spec[2]:
            raise ToolError(f"{name} value {v:g} at points[{i}] is outside {spec[1]:g}..{spec[2]:g} {spec[3]} "
                            f"({spec[4]})")
        out.append((b, v))
    return out


def _curve_name(name: str) -> str:
    if name in CURVE_SPECS:
        return name
    low = {k.lower(): k for k in CURVE_SPECS}
    if name.lower() in low:
        return low[name.lower()]
    raise ToolError(f"unknown curve {name!r}; valid curves: "
                    + ", ".join(f"{k} ({v[1]:g}..{v[2]:g} {v[3]}: {v[4]})" for k, v in CURVE_SPECS.items()))


@_tool("set_curve", """
Write automation points on a track's parameter curve. points=[[beat, value], ...] (beats from song start). Curves
(value range, default): loudness (-48..12 dB, 0), pitchDelta (-1200..1200 cents added to the generated pitch, 0),
tension (-1..1, 0), breathiness (-1..1, 0), gender (-1 feminine .. 1 masculine formants, 0), voicing (0 whisper ..
1 voiced, 1), toneShift (-1200..1200 cents formant shift, 0), vibratoEnv (0..2 multiplier on vibrato, 1).
mode='merge' (default) adds/replaces points at those beats; 'replace' clears the whole curve first; 'replace_range'
clears existing points between the first and last new beat. interpolation: 'cubic' (smooth, default), 'linear',
'step'. Example crescendo: curve='loudness', points=[[16,-6],[24,2]], interpolation='linear'. Before the first and
after the last point the curve holds the end values.
""", {"track": _TRACK_REF, "curve": {"type": "string", "description": "Curve name (see above)."},
      "points": {"type": "array", "items": {"type": ["array", "object"]}, "minItems": 1},
      "mode": {"type": "string", "enum": ["merge", "replace", "replace_range"]},
      "interpolation": {"type": "string", "enum": list(score_mod.INTERPOLATIONS)}},
    required=("curve", "points"), mutating=True)
def _set_curve(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    name = _curve_name(a["curve"])
    pts = _points(a["points"], name)
    c = t.curve(name)
    mode = a.get("mode", "merge")
    if mode == "replace":
        c.points = []
    elif mode == "replace_range":
        c.clear_range(min(b for b, _ in pts) - 1e-9, max(b for b, _ in pts) + 1e-9)
    for b, v in pts:
        c.add(b, v)
    if a.get("interpolation"):
        c.interpolation = a["interpolation"]
    return {"track": t.name, "curve": name, "interpolation": c.interpolation, "points": len(c.points),
            "span": [c.points[0][0], c.points[-1][0]] if c.points else None}


@_tool("clear_curve", """
Remove automation: one curve (or every curve when `curve` is omitted) of a track, optionally only the points between
start_beat and end_beat (inclusive). The parameter then falls back to its default.
""", {"track": _TRACK_REF, "curve": {"type": "string"},
      "start_beat": {"type": "number"}, "end_beat": {"type": "number"}}, mutating=True)
def _clear_curve(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    names = [_curve_name(a["curve"])] if a.get("curve") else list(t.curves)
    removed = 0
    lo = a.get("start_beat", -math.inf)
    hi = a.get("end_beat", math.inf)
    for name in names:
        c = t.curves.get(name)
        if c is None:
            continue
        before = len(c.points)
        c.clear_range(lo, hi)
        removed += before - len(c.points)
        if not c.points:
            del t.curves[name]
    return {"track": t.name, "removed_points": removed, "curves": sorted(k for k, c in t.curves.items() if c.points)}


@_tool("apply_expression", """
Shape the performance of a track:
- style: the automatic-expression preset (pop, ballad, rock, jazz, rnb, robotic, soft, classical, flat; see
  list_styles). It generates glides, scoops, falls, vibrato, drift, phrase dynamics and breaths.
- style_overrides: tweak style parameters for this track, e.g. {"vibrato_depth": 1.4, "scoop": 0.2, "fall": 0,
  "transition": 0.06} (units in list_styles; merged with previous overrides; null removes one;
  reset_overrides=true clears all first).
- voice_params: timbre tweaks, e.g. {"breathiness": 0.3, "tension": -0.2, "brightness_db": 2}.
- note_attrs: per-note attributes applied to notes selected by ids or start_beat/end_beat (default: all notes),
  e.g. {"vibrato_depth": 0.6} (semitones), {"velocity": 1.3} (0..2 dynamics), {"transition": 0.15} (glide
  seconds), {"tension": 0.4}, {"breathiness": 0.3}, {"detune": -10} (cents), {"consonant_length": 1.5}.
  null removes an attribute.
Returns the effective style parameters.
""", {"track": _TRACK_REF, "style": {"type": "string"},
      "style_overrides": {"type": "object"}, "reset_overrides": {"type": "boolean"},
      "voice_params": {"type": "object"}, "note_attrs": {"type": "object"},
      "ids": _IDS, "start_beat": _START, "end_beat": _END}, mutating=True)
def _apply_expression(s: Session, a: dict) -> dict:
    t = _track(s, a.get("track"))
    if not any(k in a for k in ("style", "style_overrides", "reset_overrides", "voice_params", "note_attrs")):
        raise ToolError("nothing to do: pass style, style_overrides, voice_params and/or note_attrs")
    if "style" in a:
        t.style = _check_style(a["style"])
    if a.get("reset_overrides"):
        for k in [k for k in t.voice_params if k.startswith("style.")]:
            del t.voice_params[k]
    for k, v in (a.get("style_overrides") or {}).items():
        if k not in STYLE_FIELDS:
            close = difflib.get_close_matches(k, STYLE_FIELDS, n=1)
            raise ToolError(f"unknown style parameter {k!r}" + (f" (did you mean {close[0]!r}?)" if close else "")
                            + f"; valid: {', '.join(STYLE_FIELDS)}")
        if v is None:
            t.voice_params.pop(f"style.{k}", None)
            continue
        if isinstance(v, bool):
            v = float(v)
        if not isinstance(v, (int, float)) or not math.isfinite(v):
            raise ToolError(f"style parameter {k!r} must be a number, got {v!r}")
        t.voice_params[f"style.{k}"] = float(v)
    if a.get("voice_params"):
        _apply_voice_params(t, a["voice_params"])
    n_attr = 0
    if a.get("note_attrs"):
        attrs = _validate_attrs(a["note_attrs"])
        for n in _select(t, a):
            _apply_attrs(n, attrs)
            n_attr += 1
    st = get_style(t.style, _style_overrides(t))
    return {"track": t.name, "style": t.style, "style_overrides": _style_overrides(t),
            "effective_style": {k: getattr(st, k) for k in STYLE_FIELDS},
            "voice_params": {k: v for k, v in t.voice_params.items() if not k.startswith("style.")},
            "notes_updated": n_attr}


@_tool("add_harmony", """
Create a harmony track from a source track's melody, diatonically in a key (project key by default). steps is the
interval in scale steps: 2 = a third above (the classic harmony), -2 = a third below, 4 = a fifth above, -3 = a
fourth below, 5 = a sixth above, 7/-7 = octave. Notes outside the key are handled sensibly ('smart'; or 'parallel'
keeps the exact chromatic interval, 'snap' forces scale tones). Lyrics are copied so both parts sing the same
words. By default the new track uses the source's voice and style (choose `voice` for contrast, e.g. atlas or
ember below a female lead), sits 4 dB lower and slightly panned, and phrases are octave-shifted into the voice's
range (fit_range=false to disable, or give range_low/range_high). start_beat/end_beat harmonize only part of the
song. into_track writes into an existing track instead (its notes in that span are replaced).
""", {"source_track": _TRACK_REF, "steps": {"type": "integer", "minimum": -14, "maximum": 14},
      "key": {"type": "string"}, "name": {"type": "string"}, "voice": {"type": "string"}, "style": {"type": "string"},
      "chromatic": {"type": "string", "enum": ["smart", "parallel", "snap"]},
      "fit_range": {"type": "boolean"}, "range_low": _PITCH, "range_high": _PITCH,
      "octave": {"type": "integer", "minimum": -3, "maximum": 3, "description": "Extra octaves to add."},
      "gain_db": {"type": "number", "minimum": -60, "maximum": 24},
      "pan": {"type": "number", "minimum": -1, "maximum": 1},
      "keep_lyrics": {"type": "boolean"}, "into_track": _TRACK_REF,
      "start_beat": _START, "end_beat": _END}, required=("steps",), mutating=True)
def _add_harmony(s: Session, a: dict) -> dict:
    p = s.project
    src = _track(s, a.get("source_track"))
    steps = int(a["steps"])
    if steps == 0 and not a.get("octave"):
        raise ToolError("steps=0 would only double the melody; use duplicate_track for unison, or e.g. steps=2")
    key = _key_of(s, a.get("key"))
    sel = _select(src, a)
    if not sel:
        raise ToolError(f"source track {src.name!r} has no notes in the selection")
    voice_id = _check_voice(a["voice"]) if a.get("voice") else src.voice
    voice = VOICES.get(voice_id)
    vr = None
    if a.get("range_low") is not None or a.get("range_high") is not None:
        lo = _pitch(a.get("range_low", voice.range_low if voice else 48), "range_low")
        hi = _pitch(a.get("range_high", voice.range_high if voice else 84), "range_high")
        if hi <= lo:
            raise ToolError("range_high must be above range_low")
        vr = (lo, hi)
    elif a.get("fit_range", True) and voice is not None:
        vr = (voice.range_low, voice.range_high)
    stats: dict = {}
    notes = composer.harmonize(sel, steps, key, voice_range=vr, chromatic=a.get("chromatic", "smart"),
                               keep_lyrics=a.get("keep_lyrics", True), octave=int(a.get("octave", 0)), stats=stats)
    if a.get("into_track") is not None:
        t = _track(s, a["into_track"])
        if t is src:
            raise ToolError("into_track must be a different track than source_track")
        lo_b, hi_b = min(n.onset for n in notes), max(n.end for n in notes)
        t.notes = [n for n in t.notes if not (n.onset < hi_b - 1e-9 and n.end > lo_b + 1e-9)]
        if a.get("voice"):
            t.voice = voice_id
    else:
        default_name = {2: "Harmony 3rd above", -2: "Harmony 3rd below", 4: "Harmony 5th above",
                        -3: "Harmony 4th below", 5: "Harmony 6th above", -5: "Harmony 6th below"}
        t = p.add_track(a.get("name") or default_name.get(steps, f"Harmony {steps:+d} steps"), voice=voice_id,
                        language=src.language)
        t.style = src.style
        t.voice_params = dict(src.voice_params)
        t.gain_db = src.gain_db - 4.0
        t.pan = max(-1.0, min(1.0, src.pan + (0.25 if steps > 0 else -0.25)))
        t.reverb_send = src.reverb_send
    if a.get("style"):
        t.style = _check_style(a["style"])
    for k in ("gain_db", "pan"):
        if k in a:
            setattr(t, k, float(a[k]))
    t.notes.extend(notes)
    t.sort()
    return {"track": _track_info(p, t), "notes": len(notes), "key": key.name, "steps": steps,
            "chromatic_source_notes": stats.get("chromatic_notes", 0),
            "octave_shifted_notes": stats.get("range_shifted", 0),
            "warnings": _overlap_warnings(t, {n.id for n in notes}) + _voice_range_warnings(t, notes)}


# ============================================================================ tools: checks

@_tool("check_project", """
Lint the project for problems a singer/producer would catch before rendering: overlapping notes, notes outside the
voice's range, notes too short to sing, the first note too close to beat 0 (its consonants get cut), multi-syllable
words squeezed onto one note, orphan '+'/'-' lyrics, long phrases with no rest to breathe, words missing from the
pronunciation dictionary, out-of-key notes, unknown voices/styles. Returns issues with severity
(error/warning/info) and note ids. Run it after writing and before render.
""", {"track": _TRACK_REF})
def _check_project(s: Session, a: dict) -> dict:
    p = s.project
    issues: list[dict] = []

    def add(sev, msg, track=None, ids=None):
        d = {"severity": sev, "message": msg}
        if track is not None:
            d["track"] = track.name
        if ids:
            d["note_ids"] = list(ids)[:12]
        issues.append(d)

    if not p.tracks:
        add("error", "the project has no tracks")
    tracks = [_track(s, a["track"])] if a.get("track") is not None else p.tracks
    try:
        key = composer.parse_key(p.key)
    except ValueError:
        key = None
        add("warning", f"project key {p.key!r} is not understood; set it with update_project (e.g. 'A minor')")
    b2s = p.tempo.beat_to_sec
    if p.tracks and all(t.mute for t in p.tracks):
        add("warning", "every track is muted; nothing will be heard")
    for t in tracks:
        notes = sorted(t.notes, key=lambda n: n.onset)
        if t.voice not in VOICES:
            add("error", f"unknown voice {t.voice!r}; valid: {', '.join(VOICES)}", t)
        if t.style not in STYLES:
            add("warning", f"unknown style {t.style!r}; valid: {', '.join(STYLES)}", t)
        if not notes:
            add("info", "track has no notes", t)
            continue
        for x, y in t.overlaps():
            add("error", f"notes overlap at beat {y.onset:g} ({x.note_name} '{x.lyric}' and {y.note_name} "
                         f"'{y.lyric}'); shorten or move one", t, [x.id, y.id])
        v = VOICES.get(t.voice)
        if v is not None:
            low = [n for n in notes if n.pitch < v.range_low - 2]
            high = [n for n in notes if n.pitch > v.range_high + 2]
            for group, word in ((low, "below"), (high, "above")):
                if group:
                    add("warning", f"{len(group)} note(s) {word} {v.name}'s range {midi_to_note(v.range_low)}-"
                                   f"{midi_to_note(v.range_high)}; transpose by an octave or change voice",
                        t, [n.id for n in group])
        if b2s(notes[0].onset) < 0.12:
            add("warning", f"first note starts at beat {notes[0].onset:g}; leading consonants and the breath need "
                           f"room - start the first phrase at beat 1 or later (a one-bar lead-in, e.g. beat 4, "
                           f"is typical)", t, [notes[0].id])
        short = [n for n in notes if b2s(n.end) - b2s(n.onset) < 0.07]
        if short:
            add("warning", f"{len(short)} note(s) shorter than 70 ms are hard to articulate", t, [n.id for n in short])
        prev = None
        for i, n in enumerate(notes):
            lyr = (n.lyric or "").strip()
            gap = n.onset - prev.end if prev is not None else math.inf
            plyr = (prev.lyric or "").strip() if prev is not None else ""
            if lyr == "+" and (prev is None or plyr in ("-", "br", "breath", "") or plyr.startswith("[")):
                add("warning", f"'+' at beat {n.onset:g} does not continue a word (previous lyric {plyr!r}); "
                               f"it will be sung as a melisma", t, [n.id])
            if lyr == "+" and t.language == "en" and not n.attrs.get("language"):
                j = i - 1
                while j >= 0 and (notes[j].lyric or "").strip() == "+":
                    j -= 1
                head = (notes[j].lyric or "").strip() if j >= 0 else ""
                if j >= 0 and re.fullmatch(r"[A-Za-z']+", head) and not notes[j].phonemes:
                    group = i - j + 1
                    k = english.count_syllables(head)
                    if k < group and (i + 1 >= len(notes) or (notes[i + 1].lyric or "").strip() != "+"):
                        add("warning", f"'{head}' has {k} syllable(s) but is spread over {group} notes with '+' "
                                       f"(beat {notes[j].onset:g}); the extra notes become melismas. Write the whole "
                                       f"word on the first note (e.g. 'twinkle' '+', not 'twin' '+')", t,
                            [notes[j].id, n.id])
            if lyr in ("-", "~") and gap >= 0.5:
                add("warning", f"melisma '-' at beat {n.onset:g} follows a rest; it has no vowel to hold - give it a "
                               f"syllable", t, [n.id])
            if lyr == "":
                add("warning", f"empty lyric at beat {n.onset:g} (it will be silent)", t, [n.id])
            if (t.language == "en" and not n.phonemes and re.fullmatch(r"[A-Za-z']+", lyr)
                    and not (n.attrs.get("language") == "ja")):
                nxt = notes[i + 1].lyric.strip() if i + 1 < len(notes) else ""
                k = english.count_syllables(lyr)
                if k > 1 and nxt != "+":
                    add("info", f"'{lyr}' ({k} syllables) is sung on a single note at beat {n.onset:g}; use "
                                f"set_lyrics on more notes or follow it with '+' notes", t, [n.id])
            prev = n
        # long phrases without breathing room
        run_start = notes[0]
        for x, y in zip(notes, notes[1:] + [None]):
            if y is None or y.onset - x.end >= 0.25 or (y.lyric or "").lower() in ("br", "breath"):
                if b2s(x.end) - b2s(run_start.onset) > 10.0:
                    add("info", f"phrase from beat {run_start.onset:g} to {x.end:g} lasts "
                                f"{b2s(x.end) - b2s(run_start.onset):.1f} s with no rest; add a short rest or a 'br' "
                                f"note so the singer can breathe", t, [run_start.id])
                if y is not None:
                    run_start = y
        if t.language == "en":
            unknown = sorted({w for n in notes if not n.phonemes and not n.lyric.startswith(("[", "."))
                              for w in re.findall(r"[A-Za-z']+", n.lyric)
                              if len(w) > 1 and w.lower() not in ("br",) and english.lookup(w) is None})
            if unknown:
                add("info", "pronunciation guessed for words not in the dictionary: " + ", ".join(unknown[:20])
                    + " - check with phonemize; split words across notes with '+' (not fragments), or set phonemes",
                    t)
        if key is not None:
            out_key = [n for n in notes if n.pitch % 12 not in key.scale]
            if out_key and len(out_key) > 0.25 * len(notes):
                add("info", f"{len(out_key)} of {len(notes)} notes are outside {key.name}; is the project key "
                            f"right? (add_harmony uses it)", t, [n.id for n in out_key])
    n_err = sum(i["severity"] == "error" for i in issues)
    return {"ok": n_err == 0, "errors": n_err, "warnings": sum(i["severity"] == "warning" for i in issues),
            "issues": issues[:100]}


# ============================================================================ tools: render

def _default_render_path(s: Session) -> Path:
    if s.path:
        p = Path(s.path)
        return p.with_name(_stem(p) + ".wav")
    return Path("out") / f"{_safe_name(s.project.title)}.wav"


@_tool("render", """
Synthesize the vocals to a WAV file and return the render report (duration, peak/RMS dB, render speed, per-track
warnings such as range problems or guessed pronunciations). Optional: path (default: next to the project file, or
./out/<title>.wav), start_beat/end_beat to render only a section (fast previews), tracks (list of track
indexes/ids/names; default all unmuted tracks, respecting solo), stems=true to also write one WAV per track,
bits 16/24/32, sample_rate. Afterwards call analyze_render to check pitch accuracy and levels.
""", {"path": {"type": "string"}, "start_beat": {"type": "number", "minimum": 0}, "end_beat": {"type": "number"},
      "tracks": {"type": "array", "items": {"type": ["integer", "string"]}},
      "stems": {"type": "boolean"}, "bits": {"type": "integer", "enum": [16, 24, 32]},
      "sample_rate": {"type": "integer", "enum": [16000, 22050, 24000, 32000, 44100, 48000, 96000]},
      "include_phonemes": {"type": "boolean", "description": "Include the per-note phonemes the engine sang."}})
def _render(s: Session, a: dict) -> dict:
    from ..engine import render_project

    p = s.project
    if not any(t.notes for t in p.tracks):
        raise ToolError("nothing to render: the project has no notes (use write_melody or add_notes)")
    track_ids = [_track(s, r).id for r in a["tracks"]] if a.get("tracks") else None
    sb, eb = a.get("start_beat"), a.get("end_beat")
    if sb is not None and eb is not None and eb <= sb:
        raise ToolError("end_beat must be greater than start_beat")
    res = render_project(p, sample_rate=a.get("sample_rate"), track_ids=track_ids, start_beat=sb, end_beat=eb)
    path = Path(a["path"]) if a.get("path") else _default_render_path(s)
    if path.suffix.lower() != ".wav":
        path = path.with_name(path.name + ".wav")
    bits = int(a.get("bits", 16))
    saved = res.save(path, bits=bits)
    s.last_render = res
    s.last_render_path = str(saved)
    s.last_render_state = _fingerprint(p)
    s.last_render_range = (sb, eb)
    report = copy.deepcopy(res.report)
    for tr in report.get("tracks", []):
        ph = tr.pop("phonemes", None)
        if a.get("include_phonemes") and ph is not None:
            tr["phonemes"] = ph
    out = {"path": str(saved), "report": report}
    if a.get("stems"):
        out["stems"] = [str(x) for x in res.save_stems(path.with_name(path.stem + "_stems"), bits=bits)]
    return out


@_tool("analyze_render", """
Analyse the last render: levels (peak/RMS/active level dB, clipped samples, silence and voiced ratios, median f0,
sung range, spectral centroid) and pitch accuracy per track (mean/max absolute error in cents vs the written notes,
the worst notes, notes that came out unvoiced), plus a list of `issues` with suggested fixes. Call render first.
Errors under ~20 cents are inaudible; >50 cents on a held note is worth fixing (check transitions/vibrato or use a
'pitchDelta' curve).
""", {"include_pitch": {"type": "boolean", "description": "Run the pitch-accuracy check (default true)."}})
def _analyze_render(s: Session, a: dict) -> dict:
    if s.last_render is None:
        raise ToolError("nothing has been rendered in this session yet; call render first")
    from ..analysis import analyze_audio, pitch_accuracy

    res = s.last_render
    audio = analyze_audio(res.audio, res.sample_rate)
    out = {"path": s.last_render_path, "audio": audio}
    issues = []
    if (audio.get("clipped_samples") or 0) > 0:
        issues.append("clipping: lower the master gain_db or track gains (or keep the limiter on)")
    if audio.get("peak_db") is not None and audio["peak_db"] < -18:
        issues.append(f"very quiet master (peak {audio['peak_db']} dB): enable master normalize or raise gain")
    if audio.get("voiced_ratio") is not None and audio["voiced_ratio"] < 0.15 and audio.get("duration_s", 0) > 0.5:
        issues.append("very little voiced sound: check lyrics/phonemes (consonant-only lyrics?) and voicing curves")
    if a.get("include_pitch", True):
        pa = pitch_accuracy(s.project, res)
        lo, hi = s.last_render_range
        if lo is not None or hi is not None:  # only judge notes inside the rendered section
            lo = -math.inf if lo is None else lo
            hi = math.inf if hi is None else hi
            for tr in pa.get("tracks", []):
                for key in ("worst", "unvoiced_notes"):
                    if isinstance(tr.get(key), list):
                        tr[key] = [w for w in tr[key] if lo - 1e-9 <= float(w.get("beat", lo)) < hi]
        out["pitch"] = pa
        for tr in pa.get("tracks", []):
            if tr.get("mean_abs_cents") is not None and tr["mean_abs_cents"] > 35:
                issues.append(f"track {tr['name']!r} is on average {tr['mean_abs_cents']} cents off: try a "
                              f"steadier style (e.g. less scoop/fall/vibrato via style_overrides) or check detune")
            for w in tr.get("worst", [])[:5]:
                if abs(w.get("cents", 0)) > 50:
                    issues.append(f"track {tr['name']!r}: note {w['id']} ({w['note']} '{w['lyric']}' at beat "
                                  f"{w['beat']:g}) is sung {w['cents']:+.0f} cents off")
            if tr.get("unvoiced_notes"):
                issues.append(f"track {tr['name']!r}: {len(tr['unvoiced_notes'])} note(s) produced no clear pitch "
                              f"(e.g. {tr['unvoiced_notes'][0]['note']} '{tr['unvoiced_notes'][0]['lyric']}'); "
                              f"check the lyric and note length")
    for w in (res.report or {}).get("warnings", [])[:10]:
        issues.append(f"render warning: {w}")
    if s.last_render_state is not None and s.last_render_state != _fingerprint(s.project):
        issues.append("the project changed since this render; render again to analyse the current version")
    out["issues"] = issues
    out["ok"] = not issues
    return out


# ============================================================================ tools: files

@_tool("save_project", """
Save the project. Default path: the current project path, else ./<title>.s2p.json. The format follows the
extension: .s2p.json/.json (native, lossless), .s2s (score text), or .svp/.mid/.musicxml/.ust (exports). Saving
to a native format makes it the session's project path.
""", {"path": {"type": "string"}})
def _save_project(s: Session, a: dict) -> dict:
    path = Path(a["path"]) if a.get("path") else (Path(s.path) if s.path else
                                                    Path(f"{_safe_name(s.project.title)}.s2p.json"))
    fmt = _format_of(path)
    if fmt is None:
        path = path.with_name(path.name + ".s2p.json")
        fmt = "json"
    _write_any(s.project, path, fmt)
    if fmt in ("json", "score"):
        s.path = str(path)
    return {"path": str(path), "format": fmt}


@_tool("load_project", """
Load a project file, replacing the current project (undoable). Supported: .s2p.json/.json (native), .s2s (score
text), .svp (Synthesizer V), .mid/.midi, .musicxml/.xml/.mxl, .ust (UTAU). Returns the project summary.
""", {"path": {"type": "string"}}, required=("path",), mutating=True)
def _load_project(s: Session, a: dict) -> dict:
    path = Path(a["path"])
    try:
        project = _load_any(path)
    except (OSError, ValueError, KeyError) as exc:
        raise ToolError(f"could not load {path}: {_exc_text(exc)}") from None
    s.project = project
    s.path = str(path) if _is_native(path) else str(path.with_name(_stem(path) + ".s2p.json"))
    s.last_render = None
    s.last_render_path = None
    return {"path": str(path), "project_path": s.path, "summary": project.summary()}


@_tool("export", """
Export the project to another format: format 'svp' (Synthesizer V Studio), 'midi', 'musicxml', 'ust' (UTAU, one
track), 'score' (.s2s text) or 'json'. The extension is added when missing; the format is inferred from the path
when omitted. track limits the export to one track. For audio use render.
""", {"path": {"type": "string"}, "format": {"type": "string", "enum": list(_FORMAT_EXT)}, "track": _TRACK_REF},
    required=("path",))
def _export(s: Session, a: dict) -> dict:
    path = Path(a["path"])
    fmt = a.get("format") or _format_of(path)
    if fmt is None:
        raise ToolError(f"cannot infer the export format from {path.name!r}; pass format "
                        f"({', '.join(_FORMAT_EXT)})")
    if _format_of(path) != fmt:
        path = path.with_name(path.name + _FORMAT_EXT[fmt])
    project = s.project
    if a.get("track") is not None:
        t = _track(s, a["track"])
        project = s.project.copy()
        project.tracks = [x for x in project.tracks if x.id == t.id]
    elif fmt == "ust" and len(project.tracks) > 1:
        project = s.project.copy()
        project.tracks = project.tracks[:1]
    _write_any(project, path, fmt)
    return {"path": str(path), "format": fmt, "tracks": len(project.tracks)}


# ============================================================================ tools: history

@_tool("undo", "Undo the last change(s) to the project. steps = how many (default 1).",
       {"steps": {"type": "integer", "minimum": 1, "maximum": 100}})
def _undo(s: Session, a: dict) -> dict:
    done = []
    for _ in range(int(a.get("steps", 1))):
        label = s.undo()
        if label is None:
            break
        done.append(label)
    if not done:
        raise ToolError("nothing to undo")
    return {"undone": done, "undo_left": len(s.undo_stack), "redo_available": len(s.redo_stack)}


@_tool("redo", "Redo change(s) previously undone. steps = how many (default 1).",
       {"steps": {"type": "integer", "minimum": 1, "maximum": 100}})
def _redo(s: Session, a: dict) -> dict:
    done = []
    for _ in range(int(a.get("steps", 1))):
        label = s.redo()
        if label is None:
            break
        done.append(label)
    if not done:
        raise ToolError("nothing to redo")
    return {"redone": done, "undo_left": len(s.undo_stack), "redo_available": len(s.redo_stack)}
