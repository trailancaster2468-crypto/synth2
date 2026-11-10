"""Installed neural models (voicebanks and vocoders) and the installer that puts them there.

Model weights are never part of Synth2. They are installed into a per-user directory
(:func:`models_dir`), one folder per model::

    <models_dir>/<id>/synth2-model.json     manifest (see docs/NEURAL.md)
    <models_dir>/<id>/files/...             the unmodified contents of the downloaded archive

The read side (:func:`installed`, :func:`find`, :func:`vocoder_for`) says where the models live
and which ones are there. The write side (:data:`REGISTRY`, :func:`install`, :func:`add_local`,
:func:`remove`) downloads a model's official archive, checks it against a pinned SHA-256 and
extracts it without trusting anything in it.

Importing this module stays cheap, so the rest of Synth2 can ask whether the neural engine could
run: everything the installer needs (``urllib``, ``ssl``, ``zipfile``, ``hashlib``) is imported
inside the functions that use it.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = "synth2-model.json"
MANIFEST_FORMAT = "synth2-model"
MODELS_ENV = "SYNTH2_MODELS"
BANK_ENV = "SYNTH2_NEURAL_BANK"
KINDS = ("voicebank", "vocoder")


def models_dir() -> Path:
    """The directory installed models live in (it may not exist yet).

    ``$SYNTH2_MODELS`` if set, else ``%LOCALAPPDATA%\\synth2\\models`` on Windows and
    ``$XDG_DATA_HOME/synth2/models`` (default ``~/.local/share/synth2/models``) elsewhere.
    """
    env = os.environ.get(MODELS_ENV)
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "synth2" / "models"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "synth2" / "models"


@dataclass(frozen=True)
class InstalledModel:
    """One installed model. ``path`` is its folder, ``root`` the folder that holds the model's
    own files (for a voicebank the one with ``dsconfig.yaml``, for a vocoder the one with
    ``vocoder.yaml``), ``manifest`` the parsed ``synth2-model.json``."""

    id: str
    kind: str
    name: str
    version: str
    path: Path
    root: Path
    manifest: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def license(self) -> str:
        return str(self.manifest.get("license") or "")

    @property
    def license_file(self) -> Path | None:
        rel = self.manifest.get("license_file")
        return (self.path / rel) if rel else None

    @property
    def commercial_use(self) -> bool | None:
        v = self.manifest.get("commercial_use")
        return v if isinstance(v, bool) else None

    @property
    def terms(self) -> list[str]:
        return [str(t) for t in self.manifest.get("terms") or []]

    @property
    def credits(self) -> list[str]:
        return [str(c) for c in self.manifest.get("credits") or []]

    @property
    def languages(self) -> list[str]:
        return [str(x) for x in self.manifest.get("languages") or []]

    @property
    def vocoder(self) -> str | None:
        """(voicebank) id of the installed vocoder model this bank is meant to be used with."""
        v = self.manifest.get("vocoder")
        return str(v) if v else None

    @property
    def default(self) -> bool:
        return bool(self.manifest.get("default"))

    @property
    def external(self) -> bool:
        """True when the files live outside the models directory (a bank added in place)."""
        return bool(self.manifest.get("external"))

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "version": self.version,
                "license": self.license, "commercial_use": self.commercial_use,
                "languages": self.languages, "credits": self.credits, "terms": self.terms,
                "default": self.default, "path": str(self.path),
                **({"vocoder": self.vocoder} if self.vocoder else {})}


def read_manifest(folder: Path) -> InstalledModel | None:
    """The model installed in ``folder``, or None when it has no usable manifest."""
    try:
        data = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != MANIFEST_FORMAT:
        return None
    kind, mid = data.get("kind"), data.get("id")
    if kind not in KINDS or not isinstance(mid, str) or not mid:
        return None
    root = Path(str(data.get("root") or "files"))
    root = root if root.is_absolute() else folder / root
    if not root.is_dir():
        return None
    return InstalledModel(id=mid, kind=kind, name=str(data.get("name") or mid),
                          version=str(data.get("model_version") or ""), path=folder, root=root,
                          manifest=data)


def installed(kind: str | None = None) -> list[InstalledModel]:
    """Installed models, sorted by id; ``kind`` is ``"voicebank"`` or ``"vocoder"`` to filter.
    Folders without a valid manifest are skipped."""
    base = models_dir()
    try:
        folders = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []
    out = []
    for folder in folders:
        m = read_manifest(folder)
        if m is not None and (kind is None or m.kind == kind):
            out.append(m)
    return sorted(out, key=lambda m: m.id)


def find(kind: str, model_id: str | None = None) -> InstalledModel | None:
    """The installed model of ``kind`` with this id. Without an id: the default one, which is
    ``$SYNTH2_NEURAL_BANK`` (voicebanks only) if that is installed, else the first model whose
    manifest says ``"default": true``, else the first by id."""
    models = installed(kind)
    if model_id:
        return next((m for m in models if m.id == model_id), None)
    if kind == "voicebank":
        env = os.environ.get(BANK_ENV)
        hit = next((m for m in models if m.id == env), None) if env else None
        if hit is not None:
            return hit
    return next((m for m in models if m.default), models[0] if models else None)


def vocoder_for(bank: InstalledModel) -> InstalledModel | None:
    """The installed vocoder model a voicebank should use: the one its manifest names, else the
    default vocoder. (A bank may also carry its own vocoder in a ``dsvocoder`` folder; the engine
    looks there first.)"""
    if bank.vocoder:
        hit = find("vocoder", bank.vocoder)
        if hit is not None:
            return hit
    return find("vocoder")
