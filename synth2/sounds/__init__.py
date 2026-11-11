"""Installed sound sets: soundfonts, SFZ instruments, drum kits, sample folders, impulse responses.

Samples are never part of Synth2. They are installed into a per-user directory
(:func:`sounds_dir`), one folder per set::

    <sounds_dir>/<id>/synth2-sound.json     manifest (see docs/SOUNDS.md)
    <sounds_dir>/<id>/files/...             the unmodified contents of the downloaded archive

The read side (:func:`installed`, :func:`find`) says where the sets live and which ones are
there; it needs nothing but ``json``, ``os`` and ``pathlib``. The write side
(:data:`REGISTRY`, :func:`install`, :func:`add_local`, :func:`remove`) lives in
:mod:`synth2.sounds.installer` and is loaded on first use.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

MANIFEST = "synth2-sound.json"
MANIFEST_FORMAT = "synth2-sound"
SOUNDS_ENV = "SYNTH2_SOUNDS"
KINDS = ("soundfont", "sfz", "drumkit", "samples", "impulse")

_LAZY = ("REGISTRY", "SoundSpec", "install", "add_local", "remove", "available", "credits")


def sounds_dir() -> Path:
    """The directory installed sound sets live in (it may not exist yet).

    ``$SYNTH2_SOUNDS`` if set, else ``%LOCALAPPDATA%\\synth2\\sounds`` on Windows and
    ``$XDG_DATA_HOME/synth2/sounds`` (default ``~/.local/share/synth2/sounds``) elsewhere:
    the folder next to the models directory.
    """
    env = os.environ.get(SOUNDS_ENV)
    if env:
        return Path(env).expanduser()
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "synth2" / "sounds"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "synth2" / "sounds"


@dataclass(frozen=True)
class InstalledSound:
    """One installed sound set. ``path`` is its folder, ``root`` the folder its files are
    relative to (``<path>/files`` unless the set was added in place), ``manifest`` the parsed
    ``synth2-sound.json``."""

    id: str
    kind: str
    name: str
    version: str
    path: Path
    root: Path
    manifest: dict = field(default_factory=dict, compare=False, hash=False)

    @property
    def files(self) -> list[Path]:
        """The files that matter (``.sf2`` files, ``.sfz`` files, or folders of samples), as
        absolute paths in the order the manifest lists them. Entries that are missing on disk
        are left out."""
        out = []
        for rel in self.manifest.get("files") or []:
            p = Path(str(rel))
            p = p if p.is_absolute() else self.root / p
            if p.exists():
                out.append(p)
        return out

    @property
    def presets(self) -> list[dict]:
        """Presets the manifest points out: dicts with ``name`` and ``file`` (relative to
        ``root``) and where known ``category``, ``bank`` and ``program``. An SFZ preset may
        carry ``defines`` (values for ``$variables`` the ``.sfz`` file uses without defining
        them) and ``cc`` (controller number -> starting value, to pick a variant of the
        program). A soundfont's full preset list is in the file itself; this is only a guide."""
        return [dict(p) for p in self.manifest.get("presets") or [] if isinstance(p, dict)]

    def file(self, rel: str) -> Path:
        """Absolute path of a file named relative to ``root`` (as in ``files`` and ``presets``)."""
        p = Path(str(rel))
        return p if p.is_absolute() else self.root / p

    @property
    def tags(self) -> list[str]:
        """What the set covers, e.g. ``["gm"]``, ``["drums", "acoustic"]``, ``["piano"]``."""
        return [str(t) for t in self.manifest.get("tags") or []]

    @property
    def gm(self) -> bool:
        """True for a complete General MIDI set (every program and the percussion bank)."""
        return "gm" in self.tags

    @property
    def size(self) -> int:
        """Installed size in bytes as recorded at install time (0 when unknown)."""
        v = self.manifest.get("installed_size")
        return int(v) if isinstance(v, (int, float)) else 0

    @property
    def license(self) -> str:
        return str(self.manifest.get("license") or "")

    @property
    def license_file(self) -> Path | None:
        rel = self.manifest.get("license_file")
        return (self.path / rel) if rel else None

    @property
    def commercial_use(self) -> bool | None:
        """True or False when the publisher states it, None when nothing is stated."""
        v = self.manifest.get("commercial_use")
        return v if isinstance(v, bool) else None

    @property
    def commercial_use_statement(self) -> str:
        """The publisher's own words on commercial use (may be empty)."""
        return str(self.manifest.get("commercial_use_statement") or "")

    @property
    def share_alike(self) -> bool:
        return bool(self.manifest.get("share_alike"))

    @property
    def terms(self) -> list[str]:
        return [str(t) for t in self.manifest.get("terms") or []]

    @property
    def credits(self) -> list[str]:
        """The lines a published work that uses this set should carry."""
        return [str(c) for c in self.manifest.get("credits") or []]

    @property
    def external(self) -> bool:
        """True when the files live outside the sounds directory (a set added in place)."""
        return bool(self.manifest.get("external"))

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "version": self.version,
                "tags": self.tags, "size": self.size, "license": self.license,
                "commercial_use": self.commercial_use,
                "commercial_use_statement": self.commercial_use_statement,
                "share_alike": self.share_alike, "credits": self.credits, "terms": self.terms,
                "files": [str(p) for p in self.files], "path": str(self.path),
                "root": str(self.root), "external": self.external}


def read_manifest(folder: Path) -> InstalledSound | None:
    """The sound set installed in ``folder``, or None when it has no usable manifest."""
    folder = Path(folder)
    try:
        data = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or data.get("format") != MANIFEST_FORMAT:
        return None
    kind, sid = data.get("kind"), data.get("id")
    if kind not in KINDS or not isinstance(sid, str) or not sid:
        return None
    root = Path(str(data.get("root") or "files"))
    root = root if root.is_absolute() else folder / root
    if not root.exists():
        return None
    return InstalledSound(id=sid, kind=kind, name=str(data.get("name") or sid),
                          version=str(data.get("sound_version") or ""), path=folder, root=root,
                          manifest=data)


def installed(kind: str | None = None) -> list[InstalledSound]:
    """Installed sound sets, sorted by id; ``kind`` (one of :data:`KINDS`) filters. Folders
    without a valid manifest are skipped."""
    base = sounds_dir()
    try:
        folders = sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith("."))
    except OSError:
        return []
    out = []
    for folder in folders:
        s = read_manifest(folder)
        if s is not None and (kind is None or s.kind == kind):
            out.append(s)
    return sorted(out, key=lambda s: s.id)


def find(sound_id: str) -> InstalledSound | None:
    """The installed sound set with this id, or None."""
    if not sound_id or not isinstance(sound_id, str):
        return None
    if "/" in sound_id or "\\" in sound_id or sound_id.startswith("."):
        return None
    hit = read_manifest(sounds_dir() / sound_id)
    if hit is not None and hit.id == sound_id:
        return hit
    return next((s for s in installed() if s.id == sound_id), None)


def __getattr__(name: str):
    if name in _LAZY:
        from . import installer

        return getattr(installer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
