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
inside the functions of :mod:`synth2.assets` that use it.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .. import assets
from ..assets import (AssetError, DownloadError, LicenseNotAccepted, UnsafeArchive,  # noqa: F401
                      VerificationError)

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


# ----------------------------------------------------------------------------- the installer

@dataclass(frozen=True)
class ModelSpec:
    """A model the installer knows: where its official archive is, what it must hash to, and
    what its licence asks. :meth:`manifest` is what an install writes."""

    id: str
    kind: str
    name: str
    model_version: str
    root: str                       # relative to the model folder, starts with "files"
    source: str                     # URL of the official release archive
    sha256: str
    size: int
    license: str
    license_file: str
    license_url: str
    commercial_use: bool | None
    terms: tuple[str, ...]
    credits: tuple[str, ...]
    homepage: str
    engine: str = "diffsinger"
    vocoder: str | None = None
    languages: tuple[str, ...] = ()
    default: bool = False

    def manifest(self) -> dict:
        data: dict = {"format": MANIFEST_FORMAT, "version": 1, "id": self.id, "kind": self.kind,
                      "name": self.name, "model_version": self.model_version,
                      "engine": self.engine, "root": self.root}
        if self.vocoder:
            data["vocoder"] = self.vocoder
        if self.languages:
            data["languages"] = list(self.languages)
        data.update({"license": self.license, "license_file": self.license_file,
                     "license_url": self.license_url})
        if self.commercial_use is not None:
            data["commercial_use"] = self.commercial_use
        data.update({"terms": list(self.terms), "credits": list(self.credits),
                     "homepage": self.homepage, "source": self.source, "sha256": self.sha256,
                     "default": self.default, "external": False})
        return data

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "name": self.name,
                "version": self.model_version, "license": self.license,
                "commercial_use": self.commercial_use, "languages": list(self.languages),
                "credits": list(self.credits), "terms": list(self.terms), "size": self.size,
                "source": self.source, "license_url": self.license_url,
                **({"vocoder": self.vocoder} if self.vocoder else {})}


REGISTRY: dict[str, ModelSpec] = {
    "qixuan": ModelSpec(
        id="qixuan", kind="voicebank", name="绮萱 (Qixuan)", model_version="2.7.0",
        root="files/Qixuan_v2.7.0_DiffSinger_OpenUtau", vocoder="pc-nsf-hifigan",
        languages=("zh", "ja", "en"),
        source="https://github.com/yqzhishen/qixuan-diffsinger/releases/download/v2.7.0/"
               "Qixuan_v2.7.0_DiffSinger_OpenUtau.zip",
        sha256="fe8ee7c95883d327a2b7facbbe832a7a53e1344918dc791ca83fc23b08034110",
        size=420_707_627,
        license="Qixuan DiffSinger voicebank Terms of Use",
        license_file="files/Qixuan_v2.7.0_DiffSinger_OpenUtau/Terms_of_Use.zh-CN.pdf",
        license_url="https://github.com/yqzhishen/qixuan-diffsinger/tree/main/terms_of_use",
        commercial_use=True,
        terms=(
            "Works made with this voicebank may be used commercially or non-commercially.",
            "When you publish a work, state that the vocals are AI-synthesised and credit the "
            "voicebank 绮萱 (Qixuan) and DiffSinger.",
            "No offensive, violent, pornographic, political-propaganda or religious content.",
            "Do not use the outputs to train voice synthesis or voice conversion models.",
            "Do not convert or modify the model weights; redistribute only as the complete, "
            "unmodified, free package with its terms.",
        ),
        credits=(
            "Vocals: AI-synthesised singing voice, voicebank 绮萱 (Qixuan) v2.7.0 (voice provider "
            "颜绮萱, DiffSinger Community), synthesised with DiffSinger (OpenVPI).",
        ),
        homepage="https://github.com/yqzhishen/qixuan-diffsinger",
        default=True,
    ),
    "pc-nsf-hifigan": ModelSpec(
        id="pc-nsf-hifigan", kind="vocoder",
        name="PC-NSF-HiFiGAN 44.1 kHz (DiffSinger Community Vocoder 2025.02)",
        model_version="2025.02", root="files",
        source="https://github.com/openvpi/vocoders/releases/download/"
               "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/"
               "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep",
        sha256="ba7d43142d41f6900c8264b5662ca7125a50feb8760bb8b9615c61a8f5e6902e",
        size=52_760_610,
        license="CC BY-NC-SA 4.0", license_file="files/NOTICE.txt",
        license_url="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        commercial_use=False,
        terms=(
            "The vocoder weights are provided by the OpenVPI Team under CC BY-NC-SA 4.0: "
            "attribution required, non-commercial use only, share alike.",
        ),
        credits=(
            "Vocoder: PC-NSF-HiFiGAN (DiffSinger Community Vocoder Project, OpenVPI Team), "
            "CC BY-NC-SA 4.0.",
        ),
        homepage="https://github.com/openvpi/vocoders",
        default=True,
    ),
}

_MARKER = {"voicebank": "dsconfig.yaml", "vocoder": "vocoder.yaml"}
LOCAL_TERMS = ("Added from local files. Synth2 knows nothing about this model's licence: you are "
               "responsible for knowing and following it.")


def default_set() -> list[str]:
    """Ids of the models a plain ``install()`` installs: the default voicebank of the registry
    and the vocoder it names."""
    bank = next((s for s in REGISTRY.values() if s.kind == "voicebank" and s.default), None)
    if bank is None:
        return []
    return [bank.id] + ([bank.vocoder] if bank.vocoder in REGISTRY else [])


def _refresh() -> None:
    pkg = sys.modules.get(__package__)
    if pkg is not None and hasattr(pkg, "refresh"):
        pkg.refresh()


def _gate(spec: ModelSpec, accepted: bool) -> None:
    assets.require_license(accepted, spec.id, name=spec.name, license=spec.license,
                           terms=spec.terms, credits=spec.credits, url=spec.license_url,
                           commercial_use=spec.commercial_use)


def _check_spec(spec: ModelSpec):
    def check(files: Path):
        rel = Path(spec.root).relative_to("files")
        if not (files / rel / _MARKER[spec.kind]).is_file():
            raise AssetError(f"{spec.id}: the archive has no {rel / _MARKER[spec.kind]}")
        return None

    return check


def spec_for_archive(path) -> ModelSpec | None:
    """The registry model whose official archive this file is (by SHA-256), or None."""
    try:
        digest, size = assets.sha256_file(path)
    except OSError:
        return None
    return next((s for s in REGISTRY.values() if s.sha256 == digest and s.size == size), None)


def install(model_id: str | None = None, *, accept_license: bool = False, source=None,
            progress=None, force: bool = False) -> list[InstalledModel]:
    """Install a registry model, or with no id the default voicebank and its vocoder.

    Without ``accept_license`` this raises :class:`LicenseNotAccepted` (name, licence, terms,
    credits, URL) before anything is fetched. ``source`` is a local copy of the model's
    archive to use instead of downloading; it is verified the same way. A model that is
    already installed is returned as it is unless ``force``. ``progress(done, total, label)``
    is called while downloading.
    """
    if model_id is None and source is not None:
        spec = spec_for_archive(source)
        if spec is None:
            raise AssetError(f"{source} is not the archive of a model in the registry; "
                             "register it with add_local (synth2 models add)")
        ids = [spec.id]
    else:
        ids = [model_id] if model_id else default_set()
    for mid in ids:
        if mid not in REGISTRY:
            known = ", ".join(sorted(REGISTRY))
            raise AssetError(f"unknown model {mid!r} (the registry has: {known})")
    todo = []
    done: dict[str, InstalledModel] = {}
    for mid in ids:
        spec = REGISTRY[mid]
        have = None if force else read_manifest(models_dir() / mid)
        if have is not None and have.kind == spec.kind:
            done[mid] = have
        else:
            todo.append(spec)
    for spec in todo:
        _gate(spec, accept_license)
    for spec in todo:
        src = assets.Source(url=spec.source, sha256=spec.sha256, size=spec.size, unpack="zip")
        folder = assets.install_item(models_dir(), spec.id, [src], spec.manifest(), MANIFEST,
                                     local=source, progress=progress, check=_check_spec(spec))
        model = read_manifest(folder)
        if model is None:
            raise AssetError(f"{spec.id}: the install did not produce a usable model")
        done[spec.id] = model
    if todo:
        _refresh()
    return [done[mid] for mid in ids]


def _find_root(folder: Path, kind: str | None) -> tuple[Path, str] | None:
    """The folder with ``dsconfig.yaml`` / ``vocoder.yaml`` at or up to two levels below
    ``folder``, and the kind it makes."""
    kinds = [kind] if kind else list(KINDS)
    level = [folder]
    for _ in range(3):
        for cand in level:
            for k in kinds:
                if (cand / _MARKER[k]).is_file():
                    return cand, k
        nxt = []
        for cand in level:
            try:
                nxt += sorted(p for p in cand.iterdir() if p.is_dir() and not p.is_symlink())
            except OSError:
                pass
        level = nxt
    return None


def _local_name(root: Path, kind: str) -> str:
    """The display name a model folder gives itself, or its folder name."""
    import re

    try:
        if kind == "voicebank":
            for line in (root / "character.txt").read_text(encoding="utf-8",
                                                           errors="replace").splitlines():
                if line.lower().startswith("name="):
                    return line.split("=", 1)[1].strip() or root.name
        else:
            text = (root / "vocoder.yaml").read_text(encoding="utf-8", errors="replace")
            m = re.search(r"^name:\s*['\"]?([^'\"\n#]+)", text, re.M)
            if m:
                return m.group(1).strip()
    except OSError:
        pass
    return root.name


def add_local(path, model_id: str | None = None, kind: str | None = None, *,
              name: str | None = None, copy: bool = False, force: bool = False) -> InstalledModel:
    """Register a voicebank or vocoder that is not in the registry.

    ``path`` is the model's folder (the one with ``dsconfig.yaml`` or ``vocoder.yaml``, or a
    folder above it) or a zip of it. A folder is registered in place (``external: true``, an
    absolute ``root``) unless ``copy``; a zip is always unpacked into the models directory.
    The manifest carries no ``commercial_use`` and says in ``terms`` that the licence is the
    user's responsibility.
    """
    path = Path(path).expanduser()
    if kind is not None and kind not in KINDS:
        raise AssetError(f"kind must be one of {', '.join(KINDS)}")
    if not path.exists():
        raise AssetError(f"{path} does not exist")
    stem = path.name
    for ext in (".zip", ".oudep", ".tar.gz", ".tgz", ".tar.xz"):
        if path.is_file() and stem.lower().endswith(ext):
            stem = stem[:-len(ext)]
            break
    mid = assets.check_id(model_id or assets.slug(stem))
    if mid in REGISTRY:
        raise AssetError(f"{mid!r} is the id of a registry model; choose another id")
    if not force and (models_dir() / mid).exists():
        raise AssetError(f"a model with the id {mid!r} is already installed")
    if path.is_file() and assets.archive_kind(path) is None:
        raise AssetError(f"{path} is neither a model folder nor an archive")

    def check(folder: Path, external: bool) -> dict:
        hit = _find_root(folder, kind)
        if hit is None:
            what = _MARKER[kind] if kind else "dsconfig.yaml or vocoder.yaml"
            raise AssetError(f"{path} holds no {what}: not a DiffSinger voicebank or vocoder")
        root, found = hit
        rel = str(root.resolve()) if external else \
            (Path("files") / root.relative_to(folder)).as_posix()
        return {"kind": found, "name": name or _local_name(root, found), "root": rel}

    manifest = {"format": MANIFEST_FORMAT, "version": 1, "id": mid, "kind": kind or "",
                "name": name or mid, "model_version": "", "engine": "diffsinger", "root": "",
                "license": "unknown (added by the user)", "terms": [LOCAL_TERMS], "credits": [],
                "source": str(path.resolve()), "default": False}
    folder = assets.add_item(models_dir(), mid, path, manifest, MANIFEST, copy=copy, check=check)
    model = read_manifest(folder)
    if model is None:
        assets.remove_item(models_dir(), mid, MANIFEST)
        raise AssetError(f"{mid}: the model could not be registered")
    _refresh()
    return model


def remove(model_id: str) -> bool:
    """Delete an installed model. For a model registered in place only its manifest folder is
    deleted, never the user's files. False when no such model is installed."""
    ok = assets.remove_item(models_dir(), model_id, MANIFEST)
    if ok:
        _refresh()
    return ok
