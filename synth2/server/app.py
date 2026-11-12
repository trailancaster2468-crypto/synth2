"""Synth2 Vocal Studio web server.

A dependency-free (stdlib ``http.server``) application that serves the single
page studio in ``static/`` and a small JSON REST API on top of the shared tool
layer (:mod:`synth2.ai.tools`).  Everything the browser does goes through the
same :class:`~synth2.ai.tools.Session` that the AI tools use, so GUI edits, AI
edits and undo/redo share one history.

Endpoints
---------
``GET  /``                       the studio (``static/index.html``) and its assets
``GET  /api/project``            project dict
``PUT  /api/project``            replace the project (validated, pushes an undo snapshot)
``GET  /api/tools``              tool registry (JSON schema)
``POST /api/tool``               ``{name, args}`` -> tool result, or ``{error}`` (400)
``GET  /api/engines``            singing engines, what can run here, voicebanks, credits, roles
``GET  /api/sounds``             installed sound sets (``?set=<id>&search=`` lists one set's presets)
``GET  /api/render/status``      progress of the render that is running, if any
``GET  /api/render/report``      the full report of the last render as JSON
``POST /api/render``             ``{start_beat?, end_beat?, track_ids?, engine?, quality?}`` -> ``audio/wav``,
                                 report JSON in the ``X-Render-Report`` header
``GET  /api/render/last``        the last rendered WAV
``GET  /api/meta``               voices, styles, curve specs, note attributes, phonemes, ...
``POST /api/ai``                 ``{message}`` -> ``Director.chat`` (501 without an API key)
``POST /api/import?filename=``   raw file body -> project (any supported format)
``GET  /api/export?format=``     svp | mid | musicxml | ust | s2s | json file download
``POST /api/save``               ``{path?}`` save to disk (defaults to the session path)
``POST /api/new``                ``{title?, bpm?}`` start a fresh project
``POST /api/undo`` / ``/api/redo``
``GET  /api/history``            ``{undo, redo}`` depth of the history stacks
``GET  /api/phonemes``           automatic phonemes per note, for display in the piano roll
``POST /api/split_lyrics``       ``{text, language?}`` -> per-note lyric tokens

Security
--------
The API reads and writes files on the server machine, so it only answers the
studio page itself:

* the ``Host`` header must name this machine (``localhost``, an IP address, the
  ``--host`` the server is bound to, or an ``--allow-host`` name), which defeats
  DNS rebinding;
* a state-changing request (anything but GET/HEAD) is refused when its
  ``Origin`` is not the server's own origin or ``Sec-Fetch-Site`` says it came
  from another site (CSRF);
* request bodies must use a Content-Type a cross-site page cannot send without a
  CORS preflight (which this server never approves): ``application/json`` for
  the JSON endpoints, e.g. ``application/octet-stream`` for ``/api/import``.

Run it with ``python -m synth2.server.app --port 8765 --project song.s2p.json``.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import ipaddress
import json
import math
import os
import re
import socket
import sys
import tempfile
import threading
import time
import traceback
import urllib.parse
import webbrowser
from dataclasses import fields as dc_fields
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from ..project import CURVE_SPECS, NOTE_ATTRIBUTES, Project

STATIC_DIR = Path(__file__).resolve().with_name("static")
MAX_BODY = 64 * 1024 * 1024  # imports can be large MusicXML/MIDI files
HISTORY_LIMIT = 200
RENDER_MARGIN_BEATS = 32.0  # render ranges are clamped to the song length plus this much
UNLOCKED_TOOLS = ("render",)  # tools that run on a snapshot of the project and change nothing in it
MIN_BPM, MAX_BPM = 20.0, 400.0  # same range as the new_project tool and the studio's tempo field

SAFE_METHODS = ("GET", "HEAD")
WILDCARD_HOSTS = ("", "0.0.0.0", "::")
# Content types a web page may send cross-site without a CORS preflight ("" = no Content-Type at all).
SIMPLE_CONTENT_TYPES = {"", "text/plain", "application/x-www-form-urlencoded", "multipart/form-data"}
_HOST_RE = re.compile(r"(?:\[(?P<v6>[0-9A-Fa-f:.]+)\]|(?P<name>[A-Za-z0-9._-]+))(?::\d{1,5})?")
_CHUNK_SIZE_RE = re.compile(rb"[0-9A-Fa-f]{1,16}")

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".ico": "image/x-icon",
    ".wav": "audio/wav",
    ".txt": "text/plain; charset=utf-8",
    ".woff2": "font/woff2",
}

# format name -> (file extension, MIME type)
EXPORT_FORMATS = {
    "svp": (".svp", "application/json"),
    "mid": (".mid", "audio/midi"),
    "midi": (".mid", "audio/midi"),
    "musicxml": (".musicxml", "application/vnd.recordare.musicxml+xml"),
    "xml": (".musicxml", "application/vnd.recordare.musicxml+xml"),
    "ust": (".ust", "text/plain; charset=utf-8"),
    "s2s": (".s2s", "text/plain; charset=utf-8"),
    "score": (".s2s", "text/plain; charset=utf-8"),
    "json": (".s2p.json", "application/json"),
}

IMPORT_EXTENSIONS = {".json", ".s2p", ".svp", ".mid", ".midi", ".musicxml", ".xml", ".mxl", ".ust", ".s2s", ".txt"}

# Editable per-track voice parameters (``Track.voice_params`` keys = Voice fields) for the inspector.
VOICE_PARAM_SPECS: list[tuple[str, float, float, float, str]] = [
    ("tension", -1.0, 1.0, 0.05, "Global vocal tension"),
    ("breathiness", -1.0, 1.0, 0.05, "Global breathiness"),
    ("formant_scale", 0.8, 1.4, 0.01, "Vocal-tract size (higher = smaller/brighter)"),
    ("vibrato_depth", 0.0, 1.5, 0.01, "Default vibrato depth (semitones)"),
    ("vibrato_rate", 3.0, 8.0, 0.1, "Default vibrato rate (Hz)"),
    ("brightness_db", -12.0, 12.0, 0.5, "High-shelf brightness (dB)"),
    ("singer_formant_db", 0.0, 12.0, 0.5, "Singer's formant boost around 3 kHz (dB)"),
    ("aspiration_db", -60.0, -10.0, 1.0, "Breath-noise level (dB)"),
    ("source_tilt", 6.0, 16.0, 0.5, "Glottal source roll-off (dB/oct)"),
    ("jitter", 0.0, 0.02, 0.0005, "Cycle-to-cycle pitch roughness"),
    ("shimmer_db", 0.0, 2.0, 0.05, "Cycle-to-cycle amplitude roughness (dB)"),
    ("robotic", 0.0, 1.0, 0.05, "Machine-tuned vocaloid character"),
    ("loudness_db", -12.0, 12.0, 0.5, "Voice loudness offset (dB)"),
]


def _log(msg: str) -> None:
    sys.stderr.write(f"[studio] {msg}\n")
    sys.stderr.flush()


# --------------------------------------------------------------------------
# Session compatibility layer
# --------------------------------------------------------------------------
class ToolUnavailable(Exception):
    """Raised when the tool layer (``synth2.ai.tools``) cannot be imported."""


def _tool_error_types() -> tuple[type[BaseException], ...]:
    try:
        from ..ai.tools import ToolError

        return (ToolError,)
    except Exception:  # pragma: no cover - tool layer missing
        return ()


class FallbackSession:
    """Minimal stand-in for :class:`synth2.ai.tools.Session` when the tool layer is missing.

    It keeps a project and an undo/redo history of project snapshots so the
    studio still works (editing, undo, rendering) without the AI layer.
    """

    def __init__(self, project: Project | None = None, path: str | None = None):
        self.project = project or Project.new()
        self.path = path
        self.last_render = None
        self.undo_stack: list[dict] = []
        self.redo_stack: list[dict] = []

    def push_undo(self, label: str = "") -> None:
        self.undo_stack.append(self.project.to_dict())
        del self.undo_stack[:-HISTORY_LIMIT]
        self.redo_stack.clear()

    def undo(self) -> bool:
        if not self.undo_stack:
            return False
        self.redo_stack.append(self.project.to_dict())
        self.project = Project.from_dict(self.undo_stack.pop())
        return True

    def redo(self) -> bool:
        if not self.redo_stack:
            return False
        self.undo_stack.append(self.project.to_dict())
        self.project = Project.from_dict(self.redo_stack.pop())
        return True


def _load_project_file(path: str | Path) -> Project:
    try:
        from ..io import load_any  # type: ignore[attr-defined]
    except ImportError:
        load_any = None
    if load_any is not None:
        return load_any(str(path))
    return Project.load(path)


def _make_session(project: Project | None, path: str | None):
    """Create a tool-layer Session (or the fallback) holding ``project``."""
    try:
        from ..ai.tools import Session  # type: ignore[attr-defined]
    except Exception as exc:  # tool layer missing or broken
        _log(f"tool layer unavailable ({exc.__class__.__name__}: {exc}); using built-in session")
        return FallbackSession(project, path)
    session = None
    for attempt in (
        lambda: Session(project=project, path=path),
        lambda: Session(project),
        lambda: Session(),
    ):
        try:
            session = attempt()
            break
        except TypeError:
            continue
    if session is None:  # pragma: no cover - very unusual constructor
        return FallbackSession(project, path)
    if project is not None:
        session.project = project
    if path is not None and hasattr(session, "path"):
        session.path = path
    if getattr(session, "project", None) is None:
        session.project = Project.new()
    return session


def _history_lists(session) -> tuple[list | None, list | None]:
    """Find the undo/redo stacks on a Session, whatever they are called."""
    undo = redo = None
    for name in ("undo_stack", "_undo_stack", "_undo", "undo_history", "history"):
        v = getattr(session, name, None)
        if isinstance(v, list):
            undo = v
            break
    for name in ("redo_stack", "_redo_stack", "_redo", "redo_history", "future"):
        v = getattr(session, name, None)
        if isinstance(v, list):
            redo = v
            break
    return undo, redo


def push_undo(session, label: str = "edit", new_path=False) -> None:
    """Push an undo snapshot of the session's current project (before a mutation).

    Pass ``new_path`` (``None`` or a path) when the step also switches the project file, so
    undo/redo restore the matching path and a later Save never writes into the wrong file.
    """
    # synth2.ai.tools.Session.push_undo(label) is the contract; the other names are defensive fallbacks.
    if new_path is not False and hasattr(session, "path"):
        try:
            session.push_undo(label, None, (session.path, new_path))
            return
        except (AttributeError, TypeError):
            pass
    for name in ("push_undo", "push_snapshot", "save_undo", "record_undo", "_push_undo"):
        fn = getattr(session, name, None)
        if callable(fn):
            try:
                fn(label)
            except TypeError:
                fn()
            return
    undo, redo = _history_lists(session)
    if undo is None:
        raise RuntimeError("Session exposes no undo mechanism")
    undo.append(session.project.copy())
    del undo[:-HISTORY_LIMIT]
    if redo is not None:
        redo.clear()


def history_state(session) -> dict:
    undo, redo = _history_lists(session)
    return {"undo": len(undo) if undo is not None else None, "redo": len(redo) if redo is not None else None}


def call_session_tool(session, name: str, args: dict | None = None) -> Any:
    """Call a tool on the session; falls back to built-in undo/redo/get_project."""
    args = args or {}
    if isinstance(session, FallbackSession):
        if name == "get_project":
            return {"project": session.project.to_dict()}
        if name in ("undo", "redo"):
            ok = getattr(session, name)()
            if not ok:
                raise ValueError(f"nothing to {name}")
            return {"ok": True, **history_state(session)}
        raise ToolUnavailable(f"tool layer (synth2.ai.tools) is not available; cannot call {name!r}")
    from ..ai.tools import call_tool

    return call_tool(session, name, args)


# --------------------------------------------------------------------------
# Studio state shared by all request handlers
# --------------------------------------------------------------------------
class StudioState:
    def __init__(self, project_path: str | None = None):
        self._lock = threading.RLock()
        self.ai_lock = threading.Lock()  # one Director.chat at a time (the director keeps one conversation)
        project = None
        path = None
        self.session = None
        if project_path:
            p = Path(project_path)
            if p.exists():
                try:
                    from ..ai.tools import Session

                    self.session = Session.open(p)  # non-native formats save to <stem>.s2p.json
                except ImportError:
                    project = _load_project_file(p)
                _log(f"loaded {p}")
            else:
                _log(f"{p} does not exist yet; it will be created on save")
            path = str(p)
        if self.session is None:
            self.session = _make_session(project, path)
        self.director = None
        self.last_wav: bytes | None = None
        self.last_report: dict | None = None
        self.revision = 0
        # the render that is running (``/api/render/status``): renders run one at a time, outside the
        # project lock, so every other request is answered while one is under way
        self.render_status: dict = {"active": False, "waiting": 0}
        self._status_lock = threading.Lock()

    def set_render_status(self, **changes) -> None:
        with self._status_lock:
            self.render_status = {**self.render_status, **changes}

    def render_waiting(self, delta: int) -> None:
        with self._status_lock:
            self.render_status = {**self.render_status,
                                  "waiting": max(0, int(self.render_status.get("waiting", 0)) + delta)}

    def get_render_status(self) -> dict:
        with self._status_lock:
            out = dict(self.render_status)
        if out.get("active") and out.get("started"):
            out["elapsed_s"] = round(time.time() - out["started"], 1)
        return out

    @property
    def lock(self):
        """The session's own lock when it has one (``synth2.ai.tools.Session.lock``).

        ``call_tool`` takes that lock for each tool call, so studio requests and the
        tool calls of a running AI Director exclude each other one call at a time,
        without the director holding a lock across its network round trips.
        """
        lock = getattr(self.session, "lock", None)
        return lock if lock is not None else self._lock

    @property
    def project(self) -> Project:
        return self.session.project

    def touch(self):
        self.revision += 1

    def history(self) -> dict:
        h = history_state(self.session)
        h["revision"] = self.revision
        h["path"] = getattr(self.session, "path", None)
        return h


# --------------------------------------------------------------------------
# Helpers used by handlers
# --------------------------------------------------------------------------
def director_status() -> dict:
    """Whether the AI Director can run (``anthropic`` installed and API credentials present)."""
    try:
        from ..ai.director import director_available  # type: ignore[attr-defined]
    except ImportError:
        director_available = None
    except Exception as exc:  # pragma: no cover - broken module
        return {"available": False, "reason": f"AI Director module unavailable: {exc}"}
    if director_available is not None:
        ok, reason = director_available()
        if not ok and "ANTHROPIC_API_KEY" not in reason:
            reason = f"{reason}. Set ANTHROPIC_API_KEY to enable the AI Director."
        return {"available": bool(ok), "reason": reason or ""}
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return {"available": False, "reason": "Set ANTHROPIC_API_KEY to enable the AI Director."}
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return {"available": False, "reason": "Install the 'anthropic' package (pip install anthropic) to enable the AI Director."}
    return {"available": False, "reason": "AI Director module (synth2.ai.director) is not available."}


def build_meta() -> dict:
    from ..voices import VOICES, list_voices
    from ..engine.expression import STYLES, list_styles

    try:
        from ..phonemes.inventory import inventory_table

        phonemes = inventory_table()
    except Exception:  # pragma: no cover
        phonemes = []
    voice_defaults = {}
    for vid, v in VOICES.items():
        d = v.to_dict()
        voice_defaults[vid] = {k: d[k] for k, *_ in VOICE_PARAM_SPECS if k in d}
    curves = {
        name: {"default": spec[0], "min": spec[1], "max": spec[2], "unit": spec[3], "description": spec[4]}
        for name, spec in CURVE_SPECS.items()
    }
    attrs = {}
    for name, (default, desc) in NOTE_ATTRIBUTES.items():
        attrs[name] = {"default": default, "description": desc, "type": "string" if name == "language" else "number"}
    try:
        from ..engine.expression import Style

        style_fields = [f.name for f in dc_fields(Style) if f.name not in ("name", "description")]
        style_values = {k: s.to_dict() for k, s in STYLES.items()}
    except Exception:  # pragma: no cover
        style_fields, style_values = [], {}
    tools_available = True
    try:
        from ..ai import tools as _tools  # noqa: F401
    except Exception:
        tools_available = False
    return {
        "app": "Synth2 Vocal Studio",
        "voices": list_voices(),
        "styles": list_styles(),
        "style_params": style_fields,
        "style_values": style_values,
        "curves": curves,
        "note_attributes": attrs,
        "voice_params": [
            {"name": n, "min": lo, "max": hi, "step": st, "description": d} for n, lo, hi, st, d in VOICE_PARAM_SPECS
        ],
        "voice_defaults": voice_defaults,
        "phonemes": phonemes,
        "languages": [{"id": "en", "name": "English"}, {"id": "ja", "name": "Japanese"}],
        "export_formats": ["svp", "mid", "musicxml", "ust", "s2s", "json"],
        "import_extensions": sorted(IMPORT_EXTENSIONS),
        "director": director_status(),
        "tools_available": tools_available,
    }


def compute_phonemes(project: Project) -> dict:
    """Automatic phonemes of every note, ``{"tracks": {tid: {nid: str}}, "errors": {tid: {nid: msg}}}``.

    A melisma note maps to ``"-"`` and a rest to ``""``.  Notes with invalid explicit
    phonemes are reported in ``errors`` and treated as automatic, so one bad note
    never hides the phonemes of the rest of the track.
    """
    from ..phonemes import explicit_phones, lyric_phonemes, parse_phoneme_string, resolve_notes

    out: dict[str, dict[str, str]] = {}
    errors: dict[str, dict[str, str]] = {}
    for t in project.tracks:
        notes = sorted(t.notes, key=lambda n: (n.onset, n.pitch))
        per: dict[str, str] = {}
        errs: dict[str, str] = {}
        clean = []
        for n in notes:
            m = n
            if n.phonemes:
                try:
                    parse_phoneme_string(n.phonemes)
                except Exception as exc:
                    errs[n.id] = str(exc)
                    m = copy.copy(n)
                    m.phonemes = None
            try:
                explicit_phones(m.lyric or "")
            except Exception as exc:
                errs[n.id] = str(exc)
                m = copy.copy(m)
                m.lyric = "la"
            clean.append(m)
        try:
            for n, syl in zip(notes, resolve_notes(clean, t.language)):
                per[n.id] = "-" if syl.melisma else ("" if syl.rest else " ".join(syl.phones))
        except Exception:
            for n, m in zip(notes, clean):
                try:
                    per[n.id] = " ".join(lyric_phonemes(m.lyric, m.attrs.get("language") or t.language))
                except Exception as exc:
                    per[n.id] = ""
                    errs.setdefault(n.id, str(exc))
        out[t.id] = per
        if errs:
            errors[t.id] = errs
    return {"tracks": out, "errors": errors}


def _render_lock():
    """The process-wide render lock of the tool layer (renders queue on it), or a private one without it."""
    try:
        from ..ai.tools import RENDER_LOCK

        return RENDER_LOCK
    except Exception:  # pragma: no cover - tool layer missing
        return _FALLBACK_RENDER_LOCK


_FALLBACK_RENDER_LOCK = threading.Lock()


def engine_args(engine=None, quality=None) -> tuple[str | None, str | None]:
    """Validate the ``engine`` and ``quality`` of a render request: ``(engine spec or None, quality or
    None)``. Raises ``ValueError`` with the reason for anything that is not a known engine spec (auto,
    formant, neural, neural:<voicebank id>) or quality (draft, normal, high)."""
    from ..engine.backend import normalize_engine, parse_quality

    for name, value in (("engine", engine), ("quality", quality)):
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{name} must be a string, got {type(value).__name__}")
    return normalize_engine(engine) or None, parse_quality(quality)


def render_wav(project: Project, start_beat=None, end_beat=None, track_ids=None, engine=None, quality=None,
               progress=None, started=None) -> tuple[bytes, dict, Any]:
    """Render ``project`` and return ``(wav bytes, report, RenderResult)``.

    ``engine`` and ``quality`` are those of :func:`synth2.engine.render_project`; ``progress`` is called
    with ``(fraction, message)`` while the render runs. One render runs at a time in the process:
    ``started`` is called when this one's turn has come."""
    from ..engine import render_project

    kwargs = {}
    if start_beat is not None:
        kwargs["start_beat"] = float(start_beat)
    if end_beat is not None:
        kwargs["end_beat"] = float(end_beat)
    if isinstance(track_ids, str):  # one id, not an iterable of one-character ids
        track_ids = [track_ids]
    if track_ids:
        kwargs["track_ids"] = list(track_ids)
    if engine:
        kwargs["engine"] = engine
    if quality:
        kwargs["quality"] = quality
    if progress is not None:
        kwargs["progress"] = progress
    with _render_lock():
        if started is not None:
            started()
        res = render_project(project, **kwargs)
    fd, tmp = tempfile.mkstemp(suffix=".wav", prefix="synth2-render-")
    os.close(fd)
    try:
        if hasattr(res, "save"):
            res.save(tmp, bits=16)
        else:  # pragma: no cover - contract says RenderResult.save exists
            from ..io.wav import write_wav

            write_wav(tmp, res.audio, res.sample_rate, bits=16)
        data = Path(tmp).read_bytes()
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    report = copy.deepcopy(dict(getattr(res, "report", None) or {}))
    report.setdefault("sample_rate", int(getattr(res, "sample_rate", 0) or 0))
    if hasattr(res, "start_seconds"):
        report["start_seconds"] = float(res.start_seconds)
    return data, report, res


def add_render_notes(project: Project, report: dict, quality: str | None = None) -> dict:
    """Put what the user has to see next to a render into ``report``: the ``engines`` that sang, the
    ``quality``, the ``credits`` to show when publishing (voicebank, vocoder, sound sets), ``licence_notes``
    and the ``warnings`` (with sampled sounds that fell back to the synth)."""
    try:
        from ..ai.tools import render_notes
        from ..engine.backend import current_quality

        notes = render_notes(project, report)
        report.update(engines=notes["engines"], voicebanks=notes["voicebanks"], credits=notes["credits"],
                      licence_notes=notes["licence_notes"], warnings=notes["warnings"],
                      quality=quality or current_quality())
    except Exception:  # pragma: no cover - tool layer missing: the report keeps what the engine gave
        report.setdefault("credits", [])
    return report


def engines_payload() -> dict:
    """``GET /api/engines``."""
    from ..ai.tools import ROLE_DOCS, engines_info

    out = engines_info()
    out["roles"] = [{"id": k, "description": v} for k, v in ROLE_DOCS.items()]
    return out


def _beat_value(value, name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number of beats")
    try:
        beat = float(value)
    except (ValueError, OverflowError):
        raise ValueError(f"{name} must be a number of beats, got {value!r}") from None
    if not math.isfinite(beat):
        raise ValueError(f"{name} must be a finite number of beats")
    return beat


def render_args(project: Project, start_beat=None, end_beat=None, track_ids=None) -> tuple[float | None, float | None, list[str] | None]:
    """Validate the ``/api/render`` arguments; returns ``(start_beat, end_beat, track_ids)`` or raises ``ValueError``.

    Beats must be finite, ``start_beat >= 0`` and ``end_beat > start_beat``.  Both are
    clamped to the song length plus :data:`RENDER_MARGIN_BEATS`, so no request can make
    the engine allocate audio far beyond the song (a range wholly past the end renders
    the usual short silence).  ``track_ids`` is a list of track ids (a single id string
    is accepted); unknown ids are an error, an empty list means every track.
    """
    start = _beat_value(start_beat, "start_beat")
    end = _beat_value(end_beat, "end_beat")
    if start is not None and start < 0:
        raise ValueError(f"start_beat must be >= 0, got {start:g}")
    if end is not None and end <= (start or 0.0):
        raise ValueError("end_beat must be greater than start_beat")
    limit = project.duration_beats() + RENDER_MARGIN_BEATS
    if start is not None:
        start = min(start, limit)
    if end is not None:
        end = min(end, limit)
        if start is not None and end <= start:
            end = None  # the whole range lies past the song: render from start_beat to the end
    if track_ids is not None:
        if isinstance(track_ids, str):
            track_ids = [track_ids]
        if not isinstance(track_ids, list) or not all(isinstance(t, str) for t in track_ids):
            raise ValueError("track_ids must be a list of track id strings")
        known = {t.id for t in project.tracks}
        unknown = [t for t in track_ids if t not in known]
        if unknown:
            raise ValueError(f"unknown track id(s): {', '.join(map(repr, unknown))}")
        track_ids = track_ids or None
    return start, end, track_ids


def _record_render(session, project: Project, res, start_beat, end_beat, engine=None, quality=None) -> None:
    """Let the tool layer know about a studio render (so ``analyze_render`` works on it)."""
    if not hasattr(session, "last_render"):
        return
    try:
        session.last_render = res
        if hasattr(session, "last_render_range"):
            session.last_render_range = (start_beat, end_beat)
        if hasattr(session, "last_render_state"):
            from ..ai.tools import _fingerprint  # type: ignore[attr-defined]

            session.last_render_state = _fingerprint(project, engine, quality)
            session.last_render_options = (engine, quality)
    except Exception:  # pragma: no cover - bookkeeping only
        pass


def report_header(report: dict, limit: int = 7000) -> str:
    """Encode a render report as ASCII-safe JSON for an HTTP header, trimming it if too large."""
    def enc(obj):
        return dumps_json(obj, ensure_ascii=True, separators=(",", ":"))

    s = enc(report)
    if len(s) <= limit:
        return s
    slim = {k: v for k, v in report.items() if k != "tracks"}
    tracks = []
    for t in report.get("tracks", []) or []:
        if isinstance(t, dict):
            tt = {k: v for k, v in t.items() if not isinstance(v, (list, dict)) or k == "warnings"}
            if isinstance(tt.get("warnings"), list):
                tt["warnings"] = [str(w)[:200] for w in tt["warnings"][:5]]
            tracks.append(tt)
    slim["tracks"] = tracks
    slim["truncated"] = True
    s = enc(slim)
    if len(s) <= limit:
        return s
    # the last resort keeps what must be shown with the audio; the full report is at /api/render/report
    keep = {k: slim[k] for k in ("engines", "voicebanks", "credits", "licence_notes") if k in slim}
    if isinstance(slim.get("warnings"), list):
        keep["warnings"] = [str(w)[:200] for w in slim["warnings"][:8]]
    s = enc({k: v for k, v in slim.items() if not isinstance(v, (list, dict))} | keep | {"truncated": True})
    if len(s) <= limit:
        return s
    return enc({k: v for k, v in slim.items() if not isinstance(v, (list, dict))} | {"truncated": True})


def export_project(project: Project, fmt: str) -> tuple[bytes, str, str]:
    """Export ``project`` to ``fmt``; returns ``(data, filename, mime)``."""
    fmt = (fmt or "json").lower().lstrip(".")
    if fmt not in EXPORT_FORMATS:
        raise ValueError(f"unknown export format {fmt!r}; use one of: svp, mid, musicxml, ust, s2s, json, wav")
    ext, mime = EXPORT_FORMATS[fmt]
    base = re.sub(r"[^A-Za-z0-9 _.-]+", "", project.title or "").strip() or "untitled"
    filename = base.replace(" ", "_") + ext
    if fmt == "json":
        return project.to_json().encode("utf-8"), filename, mime
    with tempfile.TemporaryDirectory(prefix="synth2-export-") as tmp:
        path = Path(tmp) / filename
        exported = False
        try:
            from ..io import export_any  # type: ignore[attr-defined]
        except ImportError:
            export_any = None
        if export_any is not None:
            written = export_any(project, str(path))
            if written:
                path = Path(written)
            exported = True
        elif fmt in ("s2s", "score"):
            from ..score import project_to_score  # type: ignore[import-not-found]

            path.write_text(project_to_score(project), encoding="utf-8")
            exported = True
        if not exported:
            raise RuntimeError("file export (synth2.io.export_any) is not available yet")
        if not path.exists():
            # some writers may append their own extension; pick whatever was written
            files = list(Path(tmp).iterdir())
            if not files:
                raise RuntimeError("exporter produced no file")
            path = files[0]
        return path.read_bytes(), filename, mime


# file name ending -> save format (longest ending wins), as in the tool layer's save_project
SAVE_FORMATS = {
    ".s2p.json": "json", ".json": "json", ".s2p": "json", ".s2s": "score", ".txt": "score", ".svp": "svp",
    ".mid": "midi", ".midi": "midi", ".musicxml": "musicxml", ".xml": "musicxml", ".mxl": "musicxml", ".ust": "ust",
}
NATIVE_SUFFIXES = (".json", ".s2p", ".s2s")  # lossless project files (``synth2.ai.tools._is_native``)


def save_format(path: str | Path) -> str | None:
    """The format :func:`save_project` writes for ``path`` (``None`` = not a known file type)."""
    name = Path(path).name.lower()
    for ext in sorted(SAVE_FORMATS, key=len, reverse=True):
        if name.endswith(ext):
            return SAVE_FORMATS[ext]
    return None


def is_project_file(path: str | Path) -> bool:
    """Whether ``path`` is a native (lossless) project file rather than an export such as ``.mid``."""
    return Path(path).name.lower().endswith(NATIVE_SUFFIXES)


def save_project(project: Project, path: str) -> str:
    """Write ``project`` to ``path``; the format follows the extension.

    A name without a known extension (``My Song v1.2``, ``take.final``) gets ``.s2p.json``
    appended, like the tool layer's ``save_project``.  Returns the path written.
    """
    p = Path(path).expanduser()
    fmt = save_format(p)
    if fmt is None:
        p = p.with_name(p.name + ".s2p.json")
        fmt = "json"
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "json":
        project.save(p)
    else:
        from ..io import export_any  # type: ignore[attr-defined]

        written = export_any(project, str(p))
        if written:
            p = Path(written)
    return str(p)


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class HttpError(Exception):
    def __init__(self, status: int, message: str, **extra):
        self.extra = extra
        super().__init__(message)
        self.status = status
        self.message = message


class StudioHandler(BaseHTTPRequestHandler):
    server_version = "Synth2Studio/1.0"
    protocol_version = "HTTP/1.1"

    # ---- plumbing ------------------------------------------------------
    @property
    def state(self) -> StudioState:
        return self.server.state  # type: ignore[attr-defined]

    def log_message(self, fmt, *args):  # silence default per-request noise
        pass

    def log_error(self, fmt, *args):
        if not getattr(self.server, "quiet", False):
            _log("error: " + (fmt % args))

    def _log_request(self, status: int):
        if getattr(self.server, "quiet", False):
            return
        ms = (time.perf_counter() - getattr(self, "_t0", time.perf_counter())) * 1000
        path = self.path if len(self.path) < 120 else self.path[:117] + "..."
        if path.startswith("/api/") or status >= 400:
            _log(f"{self.command} {path} {status} {ms:.0f}ms")

    def _send(self, status: int, body: bytes, ctype: str, headers: dict | None = None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")  # no embedding by other sites
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        self._log_request(status)

    def send_json(self, obj: Any, status: int = 200, headers: dict | None = None):
        body = dumps_json(obj, ensure_ascii=False).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8", headers)

    def send_error_json(self, status: int, message: str, **extra):
        self.send_json({"error": message, **extra}, status)

    def _consume_body(self) -> None:
        """Read the request body up front so keep-alive connections never see stale bytes.

        Handles ``Content-Length`` and ``Transfer-Encoding: chunked`` bodies.  When a body
        cannot be read completely (too large, malformed, unsupported encoding) the request
        fails and the connection is closed, so leftover bytes are never parsed as a request.
        """
        self._body = b""
        self._body_error = None
        try:
            te = self.headers.get_all("Transfer-Encoding")
            if te:
                codings = [c.strip().lower() for v in te for c in v.split(",") if c.strip()]
                if codings != ["chunked"]:
                    raise HttpError(501, f"unsupported Transfer-Encoding {', '.join(te)!r}")
                if self.headers.get("Content-Length") is not None:
                    self.close_connection = True  # ambiguous framing (RFC 9112 6.1): never reuse the connection
                self._body = self._read_chunked()
                return
            lengths = {v.strip() for v in self.headers.get_all("Content-Length") or []}
            if len(lengths) > 1:
                raise HttpError(400, "conflicting Content-Length headers")
            raw = lengths.pop() if lengths else "0"
            if not (raw.isascii() and raw.isdigit()):
                raise HttpError(400, f"invalid Content-Length {raw!r}")
            length = int(raw)
            if length > MAX_BODY:
                raise HttpError(413, "request body too large")
            if length > 0:
                self._body = self.rfile.read(length)
                if len(self._body) < length:
                    raise HttpError(400, "incomplete request body")
        except HttpError as exc:
            self._body = b""
            self._body_error = exc
            self.close_connection = True

    def _read_chunked(self) -> bytes:
        body = bytearray()
        while True:
            line = self.rfile.readline(1026)
            size_text = line.split(b";", 1)[0].strip()
            if not line.endswith(b"\n") or not _CHUNK_SIZE_RE.fullmatch(size_text):
                raise HttpError(400, "malformed chunked request body")
            size = int(size_text, 16)
            if size == 0:
                for _ in range(64):  # optional trailer fields, up to the blank line
                    trailer = self.rfile.readline(8194)
                    if trailer in (b"\r\n", b"\n"):
                        return bytes(body)
                    if not trailer.endswith(b"\n"):
                        break
                raise HttpError(400, "malformed chunked request body")
            if len(body) + size > MAX_BODY:
                raise HttpError(413, "request body too large")
            chunk = self.rfile.read(size)
            if len(chunk) < size or self.rfile.readline(3) not in (b"\r\n", b"\n"):
                raise HttpError(400, "malformed chunked request body")
            body += chunk

    def read_body(self) -> bytes:
        if getattr(self, "_body_error", None):
            raise self._body_error
        return getattr(self, "_body", b"")

    def read_json(self) -> dict:
        raw = self.read_body()
        if not raw.strip():
            return {}
        ctype = _media_type(self.headers.get("Content-Type"))
        if ctype != "application/json" and not ctype.endswith("+json"):
            raise HttpError(415, "the request body must be JSON sent with Content-Type: application/json")
        try:
            data = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant, parse_float=_finite_float)
        except (ValueError, RecursionError) as exc:  # includes JSONDecodeError, UnicodeDecodeError, NaN/Infinity
            raise HttpError(400, f"invalid JSON body: {exc}")
        if not isinstance(data, dict):
            raise HttpError(400, "JSON body must be an object")
        return data

    def _check_request(self, method: str) -> None:
        """Refuse requests that another web site's page could have made (CSRF) or read (DNS rebinding)."""
        hosts = self.headers.get_all("Host") or []
        if len(hosts) > 1:
            raise HttpError(400, "multiple Host headers")
        host = hosts[0].strip() if hosts else None
        if host is not None and not self.server.host_allowed(host):  # type: ignore[attr-defined]
            raise HttpError(403, f"forbidden: unexpected Host {host!r} (open the studio at "
                                 f"{self.server.url}, or start it with --allow-host)")  # type: ignore[attr-defined]
        if method in SAFE_METHODS:
            return
        origin = self.headers.get("Origin")
        if origin is not None and not _same_origin(origin, host):
            raise HttpError(403, "forbidden: cross-origin request")
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site not in ("", "same-origin", "none"):
            raise HttpError(403, "forbidden: cross-site request")
        if self._body and _media_type(self.headers.get("Content-Type")) in SIMPLE_CONTENT_TYPES:
            # a page on another site can send these without a CORS preflight
            raise HttpError(415, "send the request body with Content-Type: application/json "
                                 "(application/octet-stream for /api/import)")

    def _dispatch(self, method: str):
        self._t0 = time.perf_counter()
        self._consume_body()
        parsed = urllib.parse.urlsplit(self.path)
        self.query = {k: v[-1] for k, v in urllib.parse.parse_qs(parsed.query).items()}
        route = parsed.path.rstrip("/") or "/"
        handler = ROUTES.get((method, route))
        try:
            self._check_request(method)
            if self._body_error is not None:
                raise self._body_error
            if handler is not None:
                handler(self)
            elif method in ("GET", "HEAD") and not route.startswith("/api"):
                self.serve_static(parsed.path)
            elif any(r == route for (_m, r) in ROUTES):
                raise HttpError(405, f"method {method} not allowed on {route}")
            else:
                raise HttpError(404, f"not found: {route}")
        except HttpError as exc:
            self.send_error_json(exc.status, exc.message, **getattr(exc, "extra", {}))
        except BrokenPipeError:  # client went away
            pass
        except Exception as exc:  # pragma: no cover - unexpected server error
            traceback.print_exc()
            try:
                self.send_error_json(500, f"{exc.__class__.__name__}: {exc}")
            except Exception:
                pass

    def do_GET(self):
        self._dispatch("GET")

    def do_HEAD(self):
        self._dispatch("HEAD")

    def do_POST(self):
        self._dispatch("POST")

    def do_PUT(self):
        self._dispatch("PUT")

    def do_DELETE(self):
        self._dispatch("DELETE")

    # ---- static files --------------------------------------------------
    def serve_static(self, path: str):
        rel = urllib.parse.unquote(path).lstrip("/")
        if rel.startswith("static/"):
            rel = rel[len("static/"):]
        if rel in ("", "index.html"):
            rel = "index.html"
        target = (STATIC_DIR / rel).resolve()
        if STATIC_DIR not in target.parents and target != STATIC_DIR:
            raise HttpError(403, "forbidden")
        if not target.is_file():
            raise HttpError(404, f"not found: /{rel}")
        ctype = CONTENT_TYPES.get(target.suffix.lower(), "application/octet-stream")
        self._send(200, target.read_bytes(), ctype)

    # ---- API: project --------------------------------------------------
    def api_get_project(self):
        with self.state.lock:
            data = self.state.project.to_dict()
            rev = self.state.revision
            path = getattr(self.state.session, "path", None)
        self.send_json(data, headers={"X-Project-Revision": str(rev), "X-Project-Path": _ascii(str(path or ""))})

    def api_put_project(self):
        body = self.read_json()
        data = body.get("project") if isinstance(body.get("project"), dict) else body
        if not data:
            raise HttpError(400, "missing project: send the project JSON as the request body")
        try:
            project = Project.from_dict(copy.deepcopy(data))
            _check_finite(project)  # make sure it serialises as strict JSON
        except Exception as exc:
            raise HttpError(400, f"invalid project: {exc.__class__.__name__}: {exc}")
        st = self.state
        with st.lock:
            push_undo(st.session, "studio edit")
            st.session.project = project
            st.touch()
            hist = st.history()
        self.send_json({"ok": True, "history": hist, "revision": hist["revision"]})

    def api_history(self):
        with self.state.lock:
            hist = self.state.history()
        self.send_json(hist)

    def api_new(self):
        body = self.read_json()
        title = str(body.get("title") or "Untitled")
        bpm = body.get("bpm")
        if bpm is None or bpm == "":
            bpm = 120.0
        try:
            if isinstance(bpm, bool):
                raise TypeError
            bpm = float(bpm)
        except (TypeError, ValueError, OverflowError):
            raise HttpError(400, f"bpm must be a number, got {bpm!r}") from None
        if not (math.isfinite(bpm) and MIN_BPM <= bpm <= MAX_BPM):
            raise HttpError(400, f"bpm must be between {MIN_BPM:g} and {MAX_BPM:g}, got {bpm:g}")
        voice = str(body.get("voice") or "aurora")
        try:
            project = Project.new(title=title, bpm=bpm, voice=voice)
            _check_finite(project)
        except (TypeError, ValueError) as exc:
            raise HttpError(400, f"invalid project: {exc}") from None
        st = self.state
        with st.lock:
            push_undo(st.session, "new project", new_path=None)
            st.session.project = project
            if hasattr(st.session, "path"):
                st.session.path = None  # a new project has no file yet: Save asks for one
            st.touch()
            out = {"ok": True, "project": project.to_dict(), "history": st.history()}
        self.send_json(out)

    def _undo_redo(self, name: str):
        st = self.state
        with st.lock:
            try:
                result = call_session_tool(st.session, name, {})
            except ToolUnavailable as exc:
                raise HttpError(501, str(exc))
            except Exception as exc:
                raise HttpError(400, str(exc))
            st.touch()
            out = {"ok": True, "result": result, "project": st.project.to_dict(), "history": st.history()}
        self.send_json(out)

    def api_undo(self):
        self._undo_redo("undo")

    def api_redo(self):
        self._undo_redo("redo")

    # ---- API: tools ----------------------------------------------------
    def api_tools(self):
        try:
            from ..ai.tools import TOOLS
        except Exception as exc:
            raise HttpError(501, f"tool layer unavailable: {exc}")
        self.send_json(TOOLS)

    def api_tool(self):
        body = self.read_json()
        name = body.get("name")
        args = body.get("args") or {}
        if not isinstance(name, str) or not name:
            raise HttpError(400, "missing tool 'name'")
        if not isinstance(args, dict):
            raise HttpError(400, "'args' must be an object")
        st = self.state
        errors = _tool_error_types()
        # a render works on a snapshot (the tool layer takes the session lock only to copy the project), so a
        # long one does not hold up the other requests; every other tool runs under the project lock
        lock = contextlib.nullcontext() if name in UNLOCKED_TOOLS else st.lock
        with lock:
            try:
                result = call_session_tool(st.session, name, args)
            except ToolUnavailable as exc:
                raise HttpError(501, str(exc))
            except errors as exc:  # type: ignore[misc]
                self.send_error_json(400, str(exc), tool=name)
                return
            except (ValueError, KeyError, LookupError, TypeError, OSError) as exc:  # OSError: save/export paths
                self.send_error_json(400, f"{exc.__class__.__name__}: {exc}", tool=name)
                return
            if name not in UNLOCKED_TOOLS:
                st.touch()
        self.send_json(result)

    # ---- API: meta / phonemes / lyrics --------------------------------
    def api_meta(self):
        self.send_json(build_meta())

    def api_phonemes(self):
        with self.state.lock:
            project = self.state.project.copy()
        self.send_json(compute_phonemes(project))

    def api_split_lyrics(self):
        body = self.read_json()
        text = str(body.get("text") or "")
        lang = str(body.get("language") or "en")
        from ..phonemes import split_lyrics

        try:
            tokens = split_lyrics(text, lang)
        except Exception as exc:
            raise HttpError(400, str(exc))
        self.send_json({"tokens": tokens})

    # ---- API: render ---------------------------------------------------
    def _render(self, project: Project, start, end, track_ids, engine, quality) -> tuple[bytes, dict]:
        """Render ``project`` for a request: ``(wav bytes, report)``, recorded as the last render.

        Runs outside the project lock, so the studio keeps answering while a long (neural) render is under
        way; renders themselves run one at a time and ``/api/render/status`` shows the one that runs. An
        engine that cannot run here is a 409 whose body carries the ``install_hint``."""
        st = self.state
        try:
            import importlib

            importlib.import_module("synth2.engine.renderer")
            from ..engine.backend import EngineUnavailable
        except ImportError as exc:
            raise HttpError(503, f"render engine unavailable: {exc}")

        def progress(fraction, message):
            st.set_render_status(fraction=round(float(fraction), 3), message=str(message))

        t0 = time.perf_counter()
        st.render_waiting(+1)
        claimed = False
        try:
            def started():  # this render's turn has come
                nonlocal claimed
                claimed = True
                st.render_waiting(-1)
                st.set_render_status(active=True, started=time.time(), fraction=0.0, message="starting",
                                     engine=engine or "auto", quality=quality, start_beat=start, end_beat=end)

            try:
                data, report, res = render_wav(project, start, end, track_ids, engine, quality,
                                               progress=progress, started=started)
            except EngineUnavailable as exc:
                raise HttpError(409, str(exc), install_hint=engines_payload().get("install_hint") or "",
                                engine_unavailable=True)
            except (ValueError, KeyError, LookupError) as exc:
                raise HttpError(400, f"render failed: {exc}")
        finally:
            if not claimed:
                st.render_waiting(-1)
            else:
                st.set_render_status(active=False, fraction=None, message="")
        report.setdefault("render_time_s", round(time.perf_counter() - t0, 3))
        report["start_beat"] = start
        report["end_beat"] = end
        add_render_notes(project, report, quality)
        with st.lock:
            st.last_wav = data
            st.last_report = report
            _record_render(st.session, project, res, start, end, engine, quality)
        return data, report

    @staticmethod
    def _header_report(report: dict) -> str:
        slim = copy.deepcopy(report)
        for tr in slim.get("tracks", []) or []:
            if isinstance(tr, dict):
                tr.pop("phonemes", None)  # large; the studio gets phonemes from /api/phonemes
                if isinstance(tr.get("neural"), dict):
                    for k in ("alignment", "hits", "preset", "seconds"):
                        tr["neural"].pop(k, None)
        return report_header(slim)

    def api_render(self):
        body = self.read_json()
        st = self.state
        with st.lock:
            project = st.project.copy()
        try:
            start, end, track_ids = render_args(project, body.get("start_beat"), body.get("end_beat"), body.get("track_ids"))
            engine, quality = engine_args(body.get("engine"), body.get("quality"))
        except ValueError as exc:
            raise HttpError(400, str(exc))
        data, report = self._render(project, start, end, track_ids, engine, quality)
        self._send(200, data, "audio/wav", {
            "X-Render-Report": self._header_report(report),
            "Content-Disposition": 'inline; filename="render.wav"',
        })

    def api_render_status(self):
        self.send_json(self.state.get_render_status())

    def api_render_report(self):
        with self.state.lock:
            report = self.state.last_report
        if report is None:
            raise HttpError(404, "nothing rendered yet")
        slim = copy.deepcopy(report)
        for tr in slim.get("tracks", []) or []:
            if isinstance(tr, dict):
                tr.pop("phonemes", None)
        self.send_json(slim)

    # ---- API: engines and sounds ---------------------------------------
    def api_engines(self):
        try:
            out = engines_payload()
        except ImportError as exc:
            raise HttpError(501, f"tool layer unavailable: {exc}")
        with self.state.lock:
            project = self.state.project.copy()
        try:
            from ..ai.tools import resolved_engines

            out["project"] = {"engine": project.engine or "auto", "tracks": resolved_engines(project)}
        except Exception:  # pragma: no cover - never fail the status call over the project
            pass
        self.send_json(out)

    def api_sounds(self):
        """Installed sound sets (``?set=<id>`` lists one set's presets, filtered by ``search``), where they
        live, how to install more, and which sounds the project's backing band plays."""
        args: dict = {}
        if self.query.get("set"):
            args["set"] = self.query["set"]
        if self.query.get("search"):
            args["search"] = self.query["search"]
        if self.query.get("limit"):
            try:
                args["limit"] = max(1, min(500, int(self.query["limit"])))
            except ValueError:
                raise HttpError(400, "limit must be a number") from None
        try:
            from .. import sounds
            from ..ai.tools import Session, call_tool
        except ImportError as exc:
            raise HttpError(501, f"sound sets unavailable: {exc}")
        with self.state.lock:
            project = self.state.project.copy()
        errors = _tool_error_types()
        try:
            out = call_tool(Session(project, getattr(self.state.session, "path", None)), "list_sounds", args)
        except errors as exc:  # type: ignore[misc]
            raise HttpError(404 if "set" in args else 400, str(exc))
        out["sounds_dir"] = str(sounds.sounds_dir())
        out["install_hint"] = "synth2 sounds list   (what is offered)   synth2 sounds install <id>"
        self.send_json(out)

    def api_render_last(self):
        with self.state.lock:
            data = self.state.last_wav
            report = self.state.last_report
        if not data:
            raise HttpError(404, "nothing rendered yet")
        self._send(200, data, "audio/wav", {
            "X-Render-Report": self._header_report(report or {}),
            "Content-Disposition": 'attachment; filename="render.wav"',
        })

    # ---- API: AI director ----------------------------------------------
    def api_ai(self):
        body = self.read_json()
        message = str(body.get("message") or "").strip()
        status = director_status()
        if not status["available"]:
            self.send_error_json(501, status["reason"], available=False)
            return
        if not message:
            raise HttpError(400, "missing 'message'")
        st = self.state
        if not st.ai_lock.acquire(blocking=False):
            raise HttpError(409, "the AI Director is still working on the previous request")
        try:
            with st.lock:
                try:
                    if st.director is None or getattr(st.director, "session", st.session) is not st.session:
                        from ..ai.director import Director

                        st.director = Director(st.session)
                    elif body.get("reset") and hasattr(st.director, "reset"):
                        st.director.reset()
                    director = st.director
                except Exception as exc:  # DirectorUnavailable, missing SDK, bad credentials...
                    if exc.__class__.__name__ == "DirectorUnavailable" or isinstance(exc, ImportError):
                        self.send_error_json(501, str(exc), available=False)
                        return
                    raise HttpError(500, f"AI Director could not start: {exc.__class__.__name__}: {exc}")
            # chat() may take minutes of Claude round trips: it runs without the studio lock (each of its
            # tool calls locks the session on its own), so the studio keeps answering meanwhile.
            try:
                out = director.chat(message)
            except Exception as exc:
                traceback.print_exc()
                raise HttpError(500, f"AI Director error: {exc.__class__.__name__}: {exc}")
            with st.lock:
                st.touch()
                result = {
                    "reply": (out or {}).get("reply", ""),
                    "actions": (out or {}).get("actions", []),
                    "project": st.project.to_dict(),
                    "history": st.history(),
                }
        finally:
            st.ai_lock.release()
        self.send_json(result)

    # ---- API: files ----------------------------------------------------
    def api_import(self):
        filename = os.path.basename(self.query.get("filename") or "")
        if not filename:
            raise HttpError(400, "missing ?filename=")
        lower = filename.lower()
        suffix = ".s2p.json" if lower.endswith(".s2p.json") else Path(lower).suffix
        if suffix not in IMPORT_EXTENSIONS and suffix != ".s2p.json":
            raise HttpError(400, f"unsupported file type {suffix!r}")
        raw = self.read_body()
        if not raw:
            raise HttpError(400, "empty file")
        safe = re.sub(r"[^A-Za-z0-9 _.()-]+", "_", filename).strip() or ("import" + suffix)
        with tempfile.TemporaryDirectory(prefix="synth2-import-") as tmpdir:
            tmp = Path(tmpdir) / safe
            tmp.write_bytes(raw)
            try:
                project = _load_project_file(tmp)
                _check_finite(project)
            except Exception as exc:
                raise HttpError(400, f"could not import {filename}: {exc.__class__.__name__}: {exc}")
        if not project.title or project.title == "Untitled":
            project.title = re.sub(r"(\.s2p)?\.[A-Za-z0-9]+$", "", filename) or "Imported"
        st = self.state
        with st.lock:
            push_undo(st.session, f"import {filename}", new_path=None)
            st.session.project = project
            if hasattr(st.session, "path"):
                st.session.path = None  # the upload came from the browser; Save asks where to write
            st.touch()
            out = {"ok": True, "project": project.to_dict(), "history": st.history()}
        self.send_json(out)

    def api_export(self):
        fmt = self.query.get("format") or "json"
        with self.state.lock:
            project = self.state.project.copy()
        if fmt.lower().lstrip(".") == "wav":  # the audio itself: a full render with the chosen engine and quality
            try:
                engine, quality = engine_args(self.query.get("engine"), self.query.get("quality"))
            except ValueError as exc:
                raise HttpError(400, str(exc))
            data, report = self._render(project, None, None, None, engine, quality)
            base = re.sub(r"[^A-Za-z0-9 _.-]+", "", project.title or "").strip() or "untitled"
            self._send(200, data, "audio/wav", {
                "X-Render-Report": self._header_report(report),
                "Content-Disposition": f'attachment; filename="{_ascii(base.replace(" ", "_"))}.wav"'})
            return
        try:
            data, filename, mime = export_project(project, fmt)
        except ValueError as exc:
            raise HttpError(400, str(exc))
        except (ImportError, RuntimeError) as exc:
            raise HttpError(501, f"export to {fmt} unavailable: {exc}")
        self._send(200, data, mime, {"Content-Disposition": f'attachment; filename="{_ascii(filename)}"'})

    def api_save(self):
        """Save to ``path`` (default: the session's project path).

        Only a native project file (``.s2p.json``/``.json``/``.s2p``/``.s2s``) becomes the
        session's project path; saving to ``.mid``/``.svp``/``.musicxml``/``.ust`` is an export
        (``"exported": true``) and the next plain Save still writes the lossless project file.
        """
        body = self.read_json()
        path = body.get("path")
        if path is not None and not isinstance(path, str):
            raise HttpError(400, "'path' must be a string")
        if path and any(ord(c) < 0x20 or ord(c) == 0x7F for c in path):
            raise HttpError(400, "'path' must not contain control characters")
        st = self.state
        with st.lock:
            path = path or getattr(st.session, "path", None)
            if not path:
                raise HttpError(400, "no project file yet: pass {\"path\": \"<name>.s2p.json\"} (Save as…); "
                                     "exports such as .mid or .svp do not become the project file")
            try:
                saved = save_project(st.project, str(path))
            except ImportError as exc:
                raise HttpError(501, f"cannot save that format yet: {exc}")
            except (OSError, ValueError) as exc:
                raise HttpError(400, f"could not save: {exc}")
            native = is_project_file(saved)
            if native and hasattr(st.session, "path"):
                st.session.path = saved
            project_path = getattr(st.session, "path", None)
        self.send_json({"ok": True, "path": saved, "format": save_format(saved) or "json", "exported": not native,
                        "project_path": project_path})

    def api_health(self):
        self.send_json({"ok": True, "revision": self.state.revision})


# printable ASCII except '%' (the escape character) and '"' (ends a quoted header parameter)
_HEADER_SAFE = "".join(chr(c) for c in range(0x20, 0x7F) if chr(c) not in '%"')


def _ascii(s: str) -> str:
    """Make ``s`` safe as an HTTP header value: CR/LF and other control characters, non-ASCII,
    ``%`` and ``"`` are percent-encoded, so a value (e.g. a file name) can never split the header."""
    return urllib.parse.quote(s, safe=_HEADER_SAFE)


def _media_type(content_type: str | None) -> str:
    return (content_type or "").split(";", 1)[0].strip().lower()


def _host_name(host: str) -> str | None:
    """The lower-case host name of a ``Host`` header value (``None`` if malformed)."""
    m = _HOST_RE.fullmatch(host.strip())
    if not m:
        return None
    return (m["v6"] or m["name"]).lower().rstrip(".")


def _same_origin(origin: str, host: str | None) -> bool:
    """Whether an ``Origin`` header names the origin the request was sent to (``host`` = its Host header)."""
    if not host:
        return False
    o = origin.strip().lower()
    return o in (f"http://{host.lower()}", f"https://{host.lower()}")


def _allowed_host_names(bind_host: str | None, extra=()) -> set[str]:
    """Host names (besides IP addresses) the server answers to; see :meth:`StudioServer.host_allowed`."""
    names = {"localhost"}
    bind = (bind_host or "").strip().strip("[]").lower().rstrip(".")
    if bind in WILDCARD_HOSTS:  # reachable from the network: also accept this machine's own name
        try:
            name = socket.gethostname().lower().rstrip(".")
        except OSError:  # pragma: no cover
            name = ""
        if name:
            short = name.split(".")[0]
            names |= {name, short, short + ".local"}
    else:
        names.add(bind)
    names |= {h.strip().lower().rstrip(".") for h in extra if h and h.strip()}
    return names


def _reject_json_constant(name: str):
    raise ValueError(f"{name} is not allowed (numbers must be finite)")


def _finite_float(text: str) -> float:
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"number {text} is out of range")
    return value


def _sanitize_nonfinite(obj):
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_nonfinite(v) for v in obj]
    return obj


def dumps_json(obj: Any, **kw) -> str:
    """``json.dumps`` that always yields strict JSON: NaN/Infinity (which the browser's
    ``JSON.parse`` rejects) are written as ``null`` instead of breaking the whole document."""
    try:
        return json.dumps(obj, allow_nan=False, default=_json_default, **kw)
    except ValueError:
        loose = json.loads(json.dumps(obj, default=_json_default))
        return json.dumps(_sanitize_nonfinite(loose), allow_nan=False, **kw)


def _check_finite(project: Project) -> None:
    """Raise ``ValueError`` if ``project`` holds NaN/Infinity (e.g. ``"bpm": "inf"``)."""
    try:
        json.dumps(project.to_dict(), allow_nan=False, default=_json_default)
    except ValueError:
        raise ValueError("the project contains NaN or infinite numbers") from None


def _json_default(obj):
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if isinstance(obj, (set, tuple)):
        return list(obj)
    return str(obj)


ROUTES = {
    ("GET", "/api/project"): StudioHandler.api_get_project,
    ("PUT", "/api/project"): StudioHandler.api_put_project,
    # POST alias for the studio's navigator.sendBeacon() flush on page unload (beacons cannot PUT); it gets the
    # same Origin and Content-Type: application/json checks, so other sites cannot use it.
    ("POST", "/api/project"): StudioHandler.api_put_project,
    ("GET", "/api/history"): StudioHandler.api_history,
    ("POST", "/api/new"): StudioHandler.api_new,
    ("POST", "/api/undo"): StudioHandler.api_undo,
    ("POST", "/api/redo"): StudioHandler.api_redo,
    ("GET", "/api/tools"): StudioHandler.api_tools,
    ("POST", "/api/tool"): StudioHandler.api_tool,
    ("GET", "/api/meta"): StudioHandler.api_meta,
    ("GET", "/api/phonemes"): StudioHandler.api_phonemes,
    ("POST", "/api/split_lyrics"): StudioHandler.api_split_lyrics,
    ("POST", "/api/render"): StudioHandler.api_render,
    ("GET", "/api/render/last"): StudioHandler.api_render_last,
    ("GET", "/api/render/status"): StudioHandler.api_render_status,
    ("GET", "/api/render/report"): StudioHandler.api_render_report,
    ("GET", "/api/engines"): StudioHandler.api_engines,
    ("GET", "/api/sounds"): StudioHandler.api_sounds,
    ("POST", "/api/ai"): StudioHandler.api_ai,
    ("POST", "/api/import"): StudioHandler.api_import,
    ("GET", "/api/export"): StudioHandler.api_export,
    ("POST", "/api/save"): StudioHandler.api_save,
    ("GET", "/api/health"): StudioHandler.api_health,
}


def _warm_up() -> None:
    try:
        from ..phonemes import lyric_phonemes

        lyric_phonemes("hello", "en")
    except Exception:  # pragma: no cover - best effort
        pass


class StudioServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr, state: StudioState, quiet: bool = False, allowed_hosts=()):
        super().__init__(addr, StudioHandler)
        self.state = state
        self.quiet = quiet
        self.allowed_hosts = _allowed_host_names(addr[0], allowed_hosts)
        # Load the pronunciation dictionary in the background so the first lyric edit is instant.
        threading.Thread(target=_warm_up, name="synth2-warmup", daemon=True).start()

    def host_allowed(self, host: str) -> bool:
        """Whether a request's ``Host`` header names this server rather than some other site.

        Accepted: ``localhost``, any IP address, the bound ``--host`` name (with a wildcard
        bind, this machine's host name) and ``--allow-host`` names, on any port (so ssh
        port forwards work).  A DNS-rebinding page always sends its own domain name, so it
        is refused before it can read anything.
        """
        name = _host_name(host)
        if not name:
            return False
        if name in self.allowed_hosts:
            return True
        try:
            ipaddress.ip_address(name)
        except ValueError:
            return False
        return True

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        if host in ("0.0.0.0", "", "::"):
            host = "127.0.0.1"
        return f"http://{host}:{port}/"


def make_server(host: str = "127.0.0.1", port: int = 8765, project_path: str | None = None, quiet: bool = False,
                allowed_hosts=()) -> StudioServer:
    """Create (but do not start) the studio server. ``port=0`` picks a free port.

    ``allowed_hosts`` lists extra host names the studio may be opened at (e.g. behind a
    reverse proxy); see :meth:`StudioServer.host_allowed`.
    """
    return StudioServer((host, port), StudioState(project_path), quiet=quiet, allowed_hosts=allowed_hosts)


def run(host: str = "127.0.0.1", port: int = 8765, project_path: str | None = None, open_browser: bool = False,
        allowed_hosts=()) -> None:
    """Start the studio and serve until interrupted."""
    server = make_server(host, port, project_path, allowed_hosts=allowed_hosts)
    _log(f"Synth2 Vocal Studio running at {server.url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.4, lambda: webbrowser.open(server.url)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        _log("shutting down")
    finally:
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m synth2.server.app", description="Synth2 Vocal Studio web server")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--project", default=None, help="project file to open (any supported format)")
    ap.add_argument("--open", action="store_true", help="open the studio in a web browser")
    ap.add_argument("--allow-host", action="append", default=[], metavar="NAME",
                    help="extra host name the studio may be opened at (repeatable; localhost and IP addresses "
                         "always work)")
    args = ap.parse_args(argv)
    run(args.host, args.port, args.project, args.open, allowed_hosts=args.allow_host)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
