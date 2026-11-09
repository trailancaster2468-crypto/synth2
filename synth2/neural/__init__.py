"""The neural singing engine: DiffSinger voicebanks on ``onnxruntime``.

This package is optional. Importing it is cheap and never fails: ``onnxruntime`` and the
models are only touched when a track is actually rendered. The contract is in
``docs/NEURAL.md``.

    available(bank=None) -> (ok, reason)    can the engine run here?
    backend(bank=None)   -> VoiceBackend    the backend the renderer calls
    status()             -> dict            what is installed (for tools, CLI and the studio)
    credits(bank=None)   -> list[str]       attribution lines the models ask for
    refresh()                               forget cached answers (after installing a model)
"""
from __future__ import annotations

import importlib.metadata
import importlib.util
import os

from . import models

INSTALL_HINT = 'pip install "synth2[neural]" && synth2 models install'
_cache: dict = {}


def _has_module(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _key(bank):
    base = models.models_dir()
    try:
        stamp = base.stat().st_mtime_ns
    except OSError:
        stamp = None
    return (bank, str(base), stamp, os.environ.get(models.BANK_ENV))


def available(bank: str | None = None) -> tuple[bool, str]:
    """``(True, "")`` when the neural engine can render with ``bank`` (default: the default
    voicebank), else ``(False, reason)``. Cheap, cached, and it never raises."""
    key = _key(bank)
    hit = _cache.get(key)
    if hit is None:
        if len(_cache) > 64:
            _cache.clear()
        hit = _cache[key] = _check(bank)
    return hit


def _check(bank: str | None) -> tuple[bool, str]:
    try:
        if not _has_module("onnxruntime"):
            return False, "onnxruntime is not installed"
        if not _has_module("yaml"):
            return False, "PyYAML is not installed"
        if not _has_module(__name__ + ".backend"):
            return False, "this build of synth2 has no neural engine"
        vb = models.find("voicebank", bank)
        if vb is None:
            if bank:
                return False, f"voicebank {bank!r} is not installed in {models.models_dir()}"
            return False, f"no voicebank is installed in {models.models_dir()}"
        if not (vb.root / "dsvocoder").is_dir() and models.vocoder_for(vb) is None:
            return False, f"no vocoder is installed for voicebank {vb.id!r}"
        return True, ""
    except Exception as exc:  # never let a broken install break the formant engine
        return False, f"the neural engine check failed ({exc})"


def refresh() -> None:
    """Forget cached availability (call after installing or removing a model)."""
    _cache.clear()
    try:
        from . import backend as _backend
    except Exception:
        return
    reset = getattr(_backend, "reset", None)
    if callable(reset):
        reset()


def backend(bank: str | None = None):
    """The :class:`~synth2.engine.backend.VoiceBackend` of the neural engine. ``bank`` pins a
    voicebank id; with None each track uses its voice preset's bank or the default bank."""
    from .backend import get_backend

    return get_backend(bank)


def credits(bank: str | None = None) -> list[str]:
    """The attribution lines of a voicebank and its vocoder (empty when nothing is installed)."""
    vb = models.find("voicebank", bank)
    if vb is None:
        return []
    out = list(vb.credits)
    voc = models.vocoder_for(vb)
    if voc is not None:
        out += [c for c in voc.credits if c not in out]
    return out


def status() -> dict:
    """What the neural engine has to work with. Never raises."""
    ok, reason = available()
    try:
        ort = importlib.metadata.version("onnxruntime")
    except Exception:
        ort = None
    default = models.find("voicebank")
    banks = models.installed("voicebank")
    vocoders = models.installed("vocoder")
    commercial = None
    if default is not None:
        voc = models.vocoder_for(default)
        flags = [default.commercial_use] + ([voc.commercial_use] if voc is not None else [])
        commercial = False if False in flags else (True if all(f is True for f in flags) else None)
    return {
        "available": ok,
        "reason": reason,
        "install_hint": "" if ok else INSTALL_HINT,
        "onnxruntime": ort,
        "models_dir": str(models.models_dir()),
        "default_bank": default.id if default is not None else None,
        "banks": [m.summary() for m in banks],
        "vocoders": [m.summary() for m in vocoders],
        "commercial_use": commercial,
        "credits": credits(),
    }
