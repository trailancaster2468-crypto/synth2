"""File lists of sound sets that are fetched file by file from a pinned commit.

One module per set. Each has ``BASE`` (the URL the paths are relative to, ending in ``/``)
and ``FILES``, a text with one line per file: ``<sha256> <size> <path>``. ``DEST`` (optional)
is the folder below ``files/`` the paths are stored in.
"""
from __future__ import annotations

import importlib

from ...assets import AssetError, Source


def load(name: str):
    try:
        return importlib.import_module(f"{__name__}.{name}")
    except ImportError:
        raise AssetError(f"the file list {name!r} is missing from this build of synth2") from None


def base_url(name: str) -> str:
    return load(name).BASE


def entries(name: str) -> list[tuple[str, int, str]]:
    """``(sha256, size, path)`` for every file of the list."""
    out = []
    for line in load(name).FILES.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        digest, size, path = line.split(" ", 2)
        out.append((digest, int(size), path))
    return out


def sources(name: str) -> list[Source]:
    from urllib.parse import quote

    mod = load(name)
    dest = getattr(mod, "DEST", "").strip("/")
    out = []
    for digest, size, path in entries(name):
        folder, _, fname = path.rpartition("/")
        out.append(Source(url=mod.BASE + quote(path), sha256=digest, size=size, filename=fname,
                          unpack="none", dest="/".join(p for p in (dest, folder) if p)))
    return out
