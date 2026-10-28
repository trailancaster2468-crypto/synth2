"""Voice backend interface.

The renderer asks a backend to turn one track into dry mono audio.  The
built-in ``formant`` backend is the parametric engine in this package.  Other
backends (for example a neural acoustic model + vocoder running on
``onnxruntime``) implement the same interface and register the voices they
provide; the project model, tools, GUI and mixer stay unchanged.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class VoiceBackend(Protocol):
    name: str

    def voices(self) -> list[str]:
        """Voice ids this backend can render."""
        ...

    def render_track(self, project, track, sample_rate: int, t0: float, t1: float,
                     note_range: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
        """Return (mono float32 audio covering [t0, t1) seconds, report dict)."""
        ...


_BACKENDS: dict[str, VoiceBackend] = {}
_VOICE_BACKEND: dict[str, str] = {}


def register_backend(backend: VoiceBackend, voice_ids: list[str] | None = None):
    _BACKENDS[backend.name] = backend
    for v in voice_ids if voice_ids is not None else backend.voices():
        _VOICE_BACKEND[v] = backend.name


def backend_for_voice(voice_id: str) -> VoiceBackend:
    name = _VOICE_BACKEND.get(voice_id, "formant")
    if name not in _BACKENDS:
        from .renderer import FormantBackend  # registers itself

        register_backend(FormantBackend(), [])
    return _BACKENDS[name]


def list_backends() -> list[dict]:
    return [{"name": n, "voices": [v for v, b in _VOICE_BACKEND.items() if b == n]} for n in _BACKENDS]
