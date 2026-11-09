"""Voice backends and engine selection.

The renderer asks a backend to turn one track into dry mono audio. Two engines ship with
Synth2:

``formant``
    The built-in parametric engine in this package (numpy only, always available).
``neural``
    A DiffSinger voicebank and neural vocoder running on ``onnxruntime``
    (:mod:`synth2.neural`; an optional extra with models that are installed separately).

Which one sings a track is an *engine spec*: ``auto``, ``formant``, ``neural`` or
``neural:<voicebank id>``. :func:`resolve_engine` looks, in this order, at the explicit
render argument, ``track.engine``, ``project.engine`` and the ``SYNTH2_ENGINE`` environment
variable; the first one that is not empty or ``auto`` wins. With nothing set the neural
engine is used when it is installed, the formant engine otherwise.

Other backends implement :class:`VoiceBackend` and register the voice ids they provide with
:func:`register_backend`; the project model, tools, GUI and mixer stay unchanged.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Protocol

import numpy as np

ENGINE_CHOICES = ("auto", "formant", "neural")
QUALITY_CHOICES = ("draft", "normal", "high")
ENGINE_ENV = "SYNTH2_ENGINE"
QUALITY_ENV = "SYNTH2_QUALITY"


class EngineUnavailable(RuntimeError):
    """An engine was asked for by name but cannot run here (not installed, no models)."""


class VoiceBackend(Protocol):
    name: str

    def voices(self) -> list[str]:
        """Voice ids this backend can render."""
        ...

    def render_track(self, project, track, sample_rate: int, t0: float, t1: float,
                     note_range: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
        """Return (mono float32 audio covering [t0, t1) seconds, report dict).

        Sample ``i`` of the audio is at time ``t0 + i / sample_rate``. ``note_range`` (beats) is
        set for a partial render: the audio must still be what the full render has over
        ``[t0, t1)`` (so plan with the whole track as context, not only the notes in the window),
        and the report's ``notes``, ``phonemes`` and ``warnings`` cover only the notes that
        overlap ``note_range``.
        """
        ...


_BACKENDS: dict[str, VoiceBackend] = {}
_VOICE_BACKEND: dict[str, str] = {}


def register_backend(backend: VoiceBackend, voice_ids: list[str] | None = None):
    _BACKENDS[backend.name] = backend
    for v in voice_ids if voice_ids is not None else backend.voices():
        _VOICE_BACKEND[v] = backend.name


def backend_for_voice(voice_id: str) -> VoiceBackend:
    """The registered backend of a voice id (the formant engine unless a plug-in claimed it)."""
    name = _VOICE_BACKEND.get(voice_id, "formant")
    if name not in _BACKENDS:
        from .renderer import FormantBackend  # registers itself

        register_backend(FormantBackend(), [])
    return _BACKENDS[name]


def list_backends() -> list[dict]:
    return [{"name": n, "voices": [v for v, b in _VOICE_BACKEND.items() if b == n]} for n in _BACKENDS]


# ---- engine selection ---------------------------------------------------------------------------

def parse_engine(spec) -> tuple[str, str | None]:
    """``(name, voicebank id or None)`` of an engine spec.

    ``None``, ``""`` and ``"auto"`` give ``("auto", None)``; ``"neural:qixuan"`` gives
    ``("neural", "qixuan")``. Raises :class:`ValueError` for anything else.
    """
    if spec is None:
        return "auto", None
    if not isinstance(spec, str):
        raise ValueError(f"engine must be a string, got {type(spec).__name__}")
    s = spec.strip()
    name, sep, bank = s.partition(":")
    name = name.strip().lower()
    bank = bank.strip()
    if not name and not sep:
        return "auto", None
    if name not in ENGINE_CHOICES or (sep and (name != "neural" or not bank)):
        raise ValueError(f"unknown engine {spec!r}; use auto, formant, neural or neural:<voicebank id>")
    return name, (bank or None)


def normalize_engine(spec) -> str:
    """The canonical spelling of an engine spec for storing in a project (``""`` for auto)."""
    name, bank = parse_engine(spec)
    if name == "auto":
        return ""
    return f"{name}:{bank}" if bank else name


def parse_quality(q) -> str | None:
    """``draft`` / ``normal`` / ``high``, or None when unset. Raises :class:`ValueError` otherwise."""
    if q is None or (isinstance(q, str) and not q.strip()):
        return None
    s = str(q).strip().lower()
    if s not in QUALITY_CHOICES:
        raise ValueError(f"unknown quality {q!r}; use one of {', '.join(QUALITY_CHOICES)}")
    return s


def _neural_available(bank: str | None = None) -> tuple[bool, str]:
    try:
        from .. import neural
    except Exception as exc:  # a broken optional package must never break the formant engine
        return False, f"the neural engine failed to import ({exc})"
    try:
        return neural.available(bank)
    except Exception as exc:
        return False, f"the neural engine check failed ({exc})"


def resolve_engine(project=None, track=None, override=None) -> tuple[str, str | None]:
    """Decide which engine sings ``track``: ``("formant", None)`` or ``("neural", bank or None)``.

    The first spec that is not empty or ``auto`` wins: ``override`` (a render argument),
    ``track.engine``, ``project.engine``, then the ``SYNTH2_ENGINE`` environment variable.
    Without one, a voice that a plug-in backend registered keeps that backend, and any other
    voice is sung by the neural engine when it is available and by the formant engine when not.
    The name of a plug-in backend is returned as the engine name in that case.

    Raises :class:`ValueError` for a malformed spec and :class:`EngineUnavailable` when the
    neural engine is asked for by name but is not installed.
    """
    specs = (override, getattr(track, "engine", None), getattr(project, "engine", None),
             os.environ.get(ENGINE_ENV))
    for spec in specs:
        name, bank = parse_engine(spec)
        if name == "auto":
            continue
        if name == "neural":
            ok, reason = _neural_available(bank)
            if not ok:
                raise EngineUnavailable(
                    f"the neural engine was requested but is not available: {reason}. "
                    'Install it with: pip install "synth2[neural]" && synth2 models install '
                    "(or use engine=formant)")
        return name, bank
    registered = _VOICE_BACKEND.get(getattr(track, "voice", None) or "", "formant")
    if registered != "formant" and registered in _BACKENDS:
        return registered, None
    ok, _ = _neural_available(None)
    return ("neural", None) if ok else ("formant", None)


def backend_for_track(project, track, override=None) -> VoiceBackend:
    """The backend that renders ``track`` (see :func:`resolve_engine`)."""
    name, bank = resolve_engine(project, track, override)
    if name == "neural":
        from .. import neural

        return neural.backend(bank)
    if name == "formant" and "formant" not in _BACKENDS:
        from .renderer import FormantBackend  # registers itself

        register_backend(FormantBackend(), [])
    return _BACKENDS[name]


def engine_signature(project, override=None) -> list[str]:
    """The resolved engine of every track, as strings, for cache keys. Never raises."""
    out = []
    for t in getattr(project, "tracks", []):
        try:
            name, bank = resolve_engine(project, t, override)
            out.append(f"{name}:{bank}" if bank else name)
        except Exception:
            out.append("unavailable")
    return out


# ---- per-render options -------------------------------------------------------------------------

_RENDER_OPTIONS: ContextVar[dict] = ContextVar("synth2_render_options", default={})


@contextmanager
def render_options(**opts):
    """Set options for the renders inside the ``with`` block (``quality=...``; None values are
    ignored). Backends read them with :func:`current_render_options`."""
    merged = {**_RENDER_OPTIONS.get(), **{k: v for k, v in opts.items() if v is not None}}
    token = _RENDER_OPTIONS.set(merged)
    try:
        yield merged
    finally:
        _RENDER_OPTIONS.reset(token)


def current_render_options() -> dict:
    return dict(_RENDER_OPTIONS.get())


def current_quality(default: str = "normal") -> str:
    """The quality preset of the current render: the ``quality`` render option, else the
    ``SYNTH2_QUALITY`` environment variable, else ``default``."""
    q = _RENDER_OPTIONS.get().get("quality")
    if q is None:
        try:
            q = parse_quality(os.environ.get(QUALITY_ENV))
        except ValueError:
            q = None
    return q or default
