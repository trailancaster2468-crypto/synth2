"""File format import/export.

Every reader returns a :class:`synth2.project.Project` (times in beats) and
every writer takes one:

=================================  =========================  ==============================
Format                             Extensions                 Module
=================================  =========================  ==============================
Synth2 project JSON                ``.json`` ``.s2p``         :mod:`synth2.project`
Synthesizer V Studio               ``.svp``                   :mod:`synth2.io.svp`
Standard MIDI File                 ``.mid`` ``.midi``         :mod:`synth2.io.midi`
MusicXML                           ``.musicxml`` ``.xml``     :mod:`synth2.io.musicxml`
                                   ``.mxl`` (compressed)
UTAU sequence                      ``.ust``                   :mod:`synth2.io.ust`
Synth2 text score                  ``.s2s`` ``.txt``          :mod:`synth2.score`
WAV audio (render output only)     ``.wav``                   :mod:`synth2.io.wav`
=================================  =========================  ==============================

:func:`load_any` and :func:`export_any` dispatch on the file extension (or an
explicit format name) so callers never need to know which module handles what.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the project model imports this package lazily; avoid a cycle at runtime
    from ..project import Project

FORMATS: dict[str, tuple[str, ...]] = {
    "json": (".json", ".s2p"),
    "svp": (".svp",),
    "midi": (".mid", ".midi"),
    "musicxml": (".musicxml", ".xml", ".mxl"),
    "ust": (".ust",),
    "score": (".s2s", ".txt"),
}
_ALIASES = {
    "s2p": "json", "project": "json", "mid": "midi", "xml": "musicxml", "mxl": "musicxml",
    "s2s": "score", "txt": "score", "text": "score", "synthv": "svp", "utau": "ust",
}


def normalize_format(fmt: str) -> str:
    """Map a format name or extension (``"mid"``, ``".xml"``, ``"score"`` ...) to a key of :data:`FORMATS`."""
    f = str(fmt).strip().lower().lstrip(".")
    f = _ALIASES.get(f, f)
    if f not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; valid: {', '.join(FORMATS)}")
    return f


def detect_format(path: str | Path, data: bytes | None = None) -> str:
    """Guess the format of ``path`` from its extension, falling back to sniffing ``data``."""
    suffix = Path(path).suffix.lower()
    for name, exts in FORMATS.items():
        if suffix in exts:
            if name == "json" and data is not None and _looks_like_svp(data):
                return "svp"
            return name
    if data is None:
        data = Path(path).read_bytes()
    head = data[:64].lstrip(b"\xef\xbb\xbf \t\r\n")
    if head.startswith(b"MThd") or (head.startswith(b"RIFF") and b"RMID" in data[:16]):
        return "midi"
    if head.startswith(b"PK") or head.startswith(b"<"):
        return "musicxml"
    if head.startswith(b"["):
        return "ust"
    if head.startswith(b"{"):
        return "svp" if _looks_like_svp(data) else "json"
    return "score"


def _looks_like_svp(data: bytes) -> bool:
    try:
        d = json.loads(data.decode("utf-8-sig", errors="replace").strip().rstrip("\x00"))
    except ValueError:
        return False
    return isinstance(d, dict) and "time" in d and "tracks" in d and d.get("format") != "synth2-project"


def load_any(path: str | Path) -> "Project":
    """Load a project from any supported file format (chosen by extension, then by content)."""
    from ..project import Project

    path = Path(path)
    data = path.read_bytes()
    fmt = detect_format(path, data)
    if fmt == "svp":
        from .svp import loads_svp
        return loads_svp(data, title=path.stem)
    if fmt == "midi":
        from .midi import loads_midi
        return loads_midi(data, title=path.stem)
    if fmt == "musicxml":
        from .musicxml import loads_musicxml
        return loads_musicxml(data, title=path.stem)
    if fmt == "ust":
        from .ust import loads_ust
        return loads_ust(data, title=path.stem)
    if fmt == "score":
        from ..score import parse_score
        return parse_score(data.decode("utf-8-sig"))
    return Project.from_json(data.decode("utf-8-sig"))


def export_any(project: "Project", path: str | Path, fmt: str | None = None, **options) -> Path:
    """Write ``project`` to ``path`` in ``fmt`` (default: from the extension).

    When ``fmt`` is given and ``path`` has no extension, the format's default
    extension is appended.  ``options`` are passed to the writer (e.g.
    ``track=1`` for UST, which holds a single track).  Returns the written path.
    """
    path = Path(path)
    if fmt is None:
        fmt = detect_format(path, b"")
        if path.suffix.lower() not in {e for exts in FORMATS.values() for e in exts}:
            raise ValueError(f"cannot infer the export format from {path.name!r}; pass fmt=")
    requested = str(fmt).strip().lower().lstrip(".")
    fmt = normalize_format(fmt)
    if not path.suffix:
        ext = f".{requested}" if f".{requested}" in FORMATS[fmt] else FORMATS[fmt][0]  # "mxl" -> .mxl
        path = path.parent / (path.name + (".s2p.json" if fmt == "json" else ext))
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "svp":
        from .svp import write_svp
        return write_svp(project, path)
    if fmt == "midi":
        from .midi import write_midi
        return write_midi(project, path)
    if fmt == "musicxml":
        from .musicxml import write_musicxml
        return write_musicxml(project, path)
    if fmt == "ust":
        from .ust import write_ust
        return write_ust(project, path, track=options.get("track", 0))
    if fmt == "score":
        from ..score import project_to_score
        path.write_text(project_to_score(project), encoding="utf-8")
        return path
    path.write_text(project.to_json(), encoding="utf-8")
    return path


__all__ = ["FORMATS", "normalize_format", "detect_format", "load_any", "export_any"]
