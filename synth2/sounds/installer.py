"""The write side of :mod:`synth2.sounds`: install registry sets, add local ones, remove.

Built on :mod:`synth2.assets`. A registry set is fetched from its publisher's own release
(a release archive, files of a pinned commit, or a distribution's package pool), every
download is checked against a pinned size and SHA-256, and the licence is shown and must
be accepted before anything is fetched.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .. import assets
from ..assets import (AssetError, DownloadError, LicenseNotAccepted, Source,  # noqa: F401
                      UnsafeArchive, VerificationError)
from . import (KINDS, MANIFEST, MANIFEST_FORMAT, InstalledSound, find, installed, read_manifest,
               sounds_dir)

LOCAL_TERMS = ("Added from local files. Synth2 knows nothing about the licence of these sounds: "
               "you are responsible for knowing and following it.")
SAMPLE_EXT = (".wav", ".flac", ".ogg", ".aif", ".aiff")
_DRUM_WORDS = ("kick", "snare", "hat", "hihat", "tom", "crash", "ride", "clap", "cymbal", "bd",
               "sd", "hh", "rim", "cowbell", "perc")


@dataclass(frozen=True)
class SoundSpec:
    """A sound set the installer knows. ``sources`` are the downloads; ``listing`` names a
    module in :mod:`synth2.sounds.lists` for a set that is fetched file by file from a
    pinned commit. ``license`` and ``commercial_use_statement`` are the publisher's own
    words, ``license_source`` is where they were read."""

    id: str
    kind: str
    name: str
    version: str
    publisher: str
    homepage: str
    license: str
    license_source: str
    commercial_use: bool | None
    commercial_use_statement: str
    credits: tuple[str, ...]
    files: tuple[str, ...]
    size: int                               # bytes downloaded
    sources: tuple[Source, ...] = ()
    listing: str = ""
    root: str = "files"
    license_file: str = ""
    license_url: str = ""
    share_alike: bool = False
    terms: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    presets: tuple[dict, ...] = ()
    description: str = ""
    unclear: str = ""                       # what the publisher leaves open, if anything

    def get_sources(self) -> list[Source]:
        if not self.listing:
            return list(self.sources)
        from . import lists

        return list(self.sources) + lists.sources(self.listing)

    def manifest(self) -> dict:
        data: dict = {"format": MANIFEST_FORMAT, "version": 1, "id": self.id, "kind": self.kind,
                      "name": self.name, "sound_version": self.version,
                      "publisher": self.publisher, "root": self.root,
                      "files": list(self.files)}
        if self.presets:
            data["presets"] = [dict(p) for p in self.presets]
        data["tags"] = list(self.tags)
        if self.description:
            data["description"] = self.description
        data.update({"license": self.license, "license_file": self.license_file,
                     "license_url": self.license_url, "license_source": self.license_source})
        if self.commercial_use is not None:
            data["commercial_use"] = self.commercial_use
        data.update({"commercial_use_statement": self.commercial_use_statement,
                     "share_alike": self.share_alike, "terms": list(self.terms),
                     "credits": list(self.credits)})
        if self.unclear:
            data["unclear"] = self.unclear
        srcs = self.sources
        data.update({"homepage": self.homepage,
                     "source": srcs[0].url if srcs else self._listing_base(),
                     "size": self.size})
        if len(srcs) == 1 and not self.listing:
            data["sha256"] = srcs[0].sha256
        else:
            data["sha256"] = self.digest()
        data["external"] = False
        return data

    def _listing_base(self) -> str:
        from . import lists

        return lists.base_url(self.listing)

    def digest(self) -> str:
        """One SHA-256 for a set with several sources: the hash of the sorted lines
        ``<sha256>  <size>  <dest>/<name>`` of all of them."""
        import hashlib

        rows = sorted(f"{s.sha256}  {s.size}  {(s.dest + '/' if s.dest else '') + s.name}"
                      for s in self.get_sources())
        return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name, "version": self.version,
                "publisher": self.publisher, "tags": list(self.tags), "size": self.size,
                "license": self.license, "commercial_use": self.commercial_use,
                "commercial_use_statement": self.commercial_use_statement,
                "share_alike": self.share_alike, "credits": list(self.credits),
                "terms": list(self.terms), "homepage": self.homepage,
                "license_source": self.license_source, "description": self.description,
                **({"unclear": self.unclear} if self.unclear else {})}


def __getattr__(name: str):
    if name == "REGISTRY":
        from .registry import REGISTRY

        return REGISTRY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _registry() -> dict[str, SoundSpec]:
    from . import registry

    return registry.REGISTRY


def available() -> list[dict]:
    """The registry as rows for a listing, each with ``installed`` (bool)."""
    have = {s.id for s in installed()}
    return [{**spec.summary(), "installed": spec.id in have} for spec in _registry().values()]


def credits(ids=None) -> list[str]:
    """Credit lines of the installed sets with these ids (default: all installed sets), in
    order and without repeats."""
    sets = installed() if ids is None else [s for s in (find(i) for i in ids) if s is not None]
    out: list[str] = []
    for s in sets:
        for line in s.credits:
            if line not in out:
                out.append(line)
    return out


def _gate(spec: SoundSpec, accepted: bool) -> None:
    assets.require_license(accepted, spec.id, name=spec.name, license=spec.license,
                           terms=spec.terms, credits=spec.credits,
                           url=spec.license_url or spec.license_source,
                           commercial_use=spec.commercial_use,
                           statement=spec.commercial_use_statement)


def install(sound_id: str, *, accept_license: bool = False, source=None, progress=None,
            force: bool = False) -> InstalledSound:
    """Install a registry sound set.

    Without ``accept_license`` this raises :class:`LicenseNotAccepted` (name, licence, the
    publisher's statement, terms, credits, URL) before anything is fetched. ``source`` is a
    local copy of the set's archive, or a folder holding its source files, to use instead
    of downloading; it is verified the same way. An installed set is returned as it is
    unless ``force``.
    """
    registry = _registry()
    if sound_id not in registry:
        raise AssetError(f"unknown sound set {sound_id!r} (see: synth2 sounds list)")
    spec = registry[sound_id]
    if not force:
        have = read_manifest(sounds_dir() / sound_id)
        if have is not None:
            return have
    _gate(spec, accept_license)

    def check(files: Path):
        root = files.parent / spec.root
        missing = [f for f in spec.files if not (root / f).exists()]
        if missing:
            raise AssetError(f"{spec.id}: the download has no {missing[0]}")
        return None

    sources = spec.get_sources()
    folder = assets.install_item(sounds_dir(), spec.id, sources, spec.manifest(), MANIFEST,
                                 local=source, progress=progress, check=check,
                                 workers=4 if len(sources) > 8 else 1, record_size=True)
    sound = read_manifest(folder)
    if sound is None:
        raise AssetError(f"{spec.id}: the install did not produce a usable sound set")
    return sound


def _scan(folder: Path, limit: int = 20000) -> dict:
    """What a folder holds: relative paths of its ``.sf2``/``.sf3``, ``.sfz`` and sample files."""
    found = {"sf2": [], "sfz": [], "samples": []}
    count = 0
    for root, dirs, names in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for n in sorted(names):
            count += 1
            if count > limit:
                return found
            low = n.lower()
            rel = (Path(root) / n).relative_to(folder).as_posix()
            if low.endswith((".sf2", ".sf3")):
                found["sf2"].append(rel)
            elif low.endswith(".sfz"):
                found["sfz"].append(rel)
            elif low.endswith(SAMPLE_EXT):
                found["samples"].append(rel)
    return found


def _looks_like_drums(names) -> bool:
    import re

    hits = 0
    for n in names:
        words = re.split(r"[^a-z]+", Path(n).stem.lower())
        if any(w in _DRUM_WORDS for w in words):
            hits += 1
    return hits >= max(2, len(names) // 3)


def detect(folder: Path, only: str | None = None) -> tuple[str, list[str]]:
    """``(kind, files)`` for a folder of sounds. ``only`` names one file in it."""
    if only:
        low = only.lower()
        if low.endswith((".sf2", ".sf3")):
            return "soundfont", [only]
        if low.endswith(".sfz"):
            return "sfz", [only]
        if low.endswith(SAMPLE_EXT):
            return "samples", [only]
        raise AssetError(f"{only}: not a soundfont, an SFZ instrument or an audio file")
    found = _scan(folder)
    if found["sf2"] and not found["sfz"]:
        return "soundfont", found["sf2"]
    if found["sfz"]:
        return "sfz", found["sfz"]
    if found["samples"]:
        kind = "drumkit" if _looks_like_drums(found["samples"]) else "samples"
        return kind, ["."]
    raise AssetError(f"{folder} holds no .sf2, .sfz or audio files")


def add_local(path, sound_id: str | None = None, kind: str | None = None, *,
              name: str | None = None, copy: bool = False, license: str = "",
              commercial_use: bool | None = None, credits=(), tags=(),
              force: bool = False) -> InstalledSound:
    """Register sounds the user already has: a ``.sf2`` or ``.sfz`` file, a folder of them,
    a folder of audio files, or an archive of any of these.

    A file or folder is registered in place (``external: true``; nothing is copied) unless
    ``copy``; an archive is always unpacked into the sounds directory. ``kind`` overrides
    the detected kind (a folder of audio files is ``samples``, or ``drumkit`` when the file
    names look like drums; say ``impulse`` for impulse responses). ``license``,
    ``commercial_use`` and ``credits`` record what the user knows about the sounds; left
    out, the manifest says the licence is the user's responsibility.
    """
    path = Path(path).expanduser()
    if kind is not None and kind not in KINDS:
        raise AssetError(f"kind must be one of {', '.join(KINDS)}")
    if not path.exists():
        raise AssetError(f"{path} does not exist")
    is_archive = path.is_file() and assets.archive_kind(path) is not None
    stem = path.name
    if path.is_file():
        for ext in (".tar.gz", ".tar.xz", ".tar.bz2"):
            if stem.lower().endswith(ext):
                stem = stem[:-len(ext)]
                break
        else:
            stem = Path(stem).stem
    sid = assets.check_id(sound_id or assets.slug(stem))
    if sid in _registry():
        raise AssetError(f"{sid!r} is the id of a registry sound set; choose another id")
    if not force and (sounds_dir() / sid).exists():
        raise AssetError(f"a sound set with the id {sid!r} is already installed")
    single = path.name if path.is_file() and not is_archive else None

    def check(folder: Path, external: bool) -> dict:
        found, files = detect(folder, single)
        return {"kind": kind or found, "files": files}

    manifest = {"format": MANIFEST_FORMAT, "version": 1, "id": sid, "kind": kind or "",
                "name": name or stem, "sound_version": "", "root": "", "files": [],
                "tags": [str(t) for t in tags],
                "license": license or "unknown (added by the user)",
                **({"commercial_use": commercial_use} if isinstance(commercial_use, bool) else {}),
                "commercial_use_statement": "", "share_alike": False,
                "terms": [] if license else [LOCAL_TERMS],
                "credits": [str(c) for c in credits], "source": str(path.resolve())}
    folder = assets.add_item(sounds_dir(), sid, path, manifest, MANIFEST, copy=copy, check=check,
                             record_size=True)
    sound = read_manifest(folder)
    if sound is None:
        assets.remove_item(sounds_dir(), sid, MANIFEST)
        raise AssetError(f"{sid}: the sounds could not be registered")
    return sound


def remove(sound_id: str) -> bool:
    """Delete an installed sound set. For a set registered in place only its manifest folder
    is deleted, never the user's files. False when no such set is installed."""
    return assets.remove_item(sounds_dir(), sound_id, MANIFEST)
