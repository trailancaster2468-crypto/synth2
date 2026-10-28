"""UTAU sequence text (``.ust``) import and export.

A UST file is an INI-like list of sections::

    [#VERSION]
    UST Version1.2
    Charset=UTF-8
    [#SETTING]
    Tempo=120.00
    ProjectName=song
    VoiceDir=%VOICE%uta
    [#0000]
    Length=480          # ticks, 480 per quarter note
    Lyric=あ
    NoteNum=60
    Intensity=100
    VBR=65,180,35,20,20,0,0,0
    [#0001]
    Length=480
    Lyric=R             # rest
    NoteNum=60
    [#TRACKEND]

Notes are strictly sequential.  A ``Tempo=`` entry inside a note section
changes the tempo from that note on.  Files are decoded as UTF-8 and fall
back to Shift-JIS (cp932), the historical UTAU encoding.  VCV lyrics such as
``"a か"`` / ``"- か"`` are reduced to the kana.  ``Intensity`` maps to the
``velocity`` attribute (``Intensity / 100``) and the ``VBR`` vibrato fields
(length %, cycle ms, depth cents) map to ``vibrato_start`` / ``vibrato_rate`` /
``vibrato_depth``.  The track language is ``"ja"`` when the lyrics contain kana.

:func:`write_ust` writes one project track as UTF-8 UST 1.2.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..project import Note, Project, Track
from ..timing import TempoMap, TempoPoint

TICKS = 480
REST_LYRICS = {"r", "rest", ""}
_VCV = re.compile(r"^(?:-|[aiueonN])\s+(\S.*)$")  # VCV prefix: "a か", "- か"


def _has_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def _decode(data: bytes) -> str:
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
    head = data[:400].lower()
    order = ("cp932", "utf-8") if b"charset=shift_jis" in head or b"charset=sjis" in head else ("utf-8", "cp932")
    for enc in order:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("latin-1")


def _float(s: str | None, default: float | None = None) -> float | None:
    if s is None:
        return default
    try:
        return float(s.strip().replace(",", "."))
    except ValueError:
        return default


def parse_sections(text: str) -> list[tuple[str, dict[str, str]]]:
    """Split UST text into ``[(section name, {key: value})]`` in file order."""
    sections: list[tuple[str, dict[str, str]]] = []
    current: dict[str, str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            current = {}
            sections.append((line[1:-1].lstrip("#").strip(), current))
            continue
        if current is None:
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            current[k.strip()] = v.strip()
        else:
            current.setdefault("_lines", "")
            current["_lines"] += line + "\n"
    return sections


def _clean_lyric(lyric: str) -> str:
    s = lyric.strip()
    m = _VCV.match(s)
    if m and _has_kana(m.group(1)):
        s = m.group(1).strip()
    return s


def loads_ust(data: bytes | str, title: str | None = None) -> Project:
    """Parse UST content (bytes are decoded as UTF-8, falling back to Shift-JIS)."""
    text = _decode(data) if isinstance(data, bytes) else data.lstrip("﻿")
    sections = parse_sections(text)
    setting = next((d for name, d in sections if name.upper() == "SETTING"), {})
    version = next((d for name, d in sections if name.upper() == "VERSION"), {})
    bpm0 = _float(setting.get("Tempo"), 120.0) or 120.0
    name = setting.get("ProjectName") or ""
    project = Project(title=name or title or "Untitled")  # ProjectName wins over the file name
    track = project.add_track(name or "UTAU")
    tempo_pts: list[TempoPoint] = [TempoPoint(0.0, bpm0)]
    tick = 0
    pending_vbr: list[tuple[Note, list[float]]] = []
    for sec, d in sections:
        if not sec.isdigit():
            continue
        length = int(round(_float(d.get("Length"), 0.0) or 0.0))
        bpm = _float(d.get("Tempo"))
        if bpm and bpm > 0:
            tempo_pts = [p for p in tempo_pts if abs(p.beat - tick / TICKS) > 1e-9] + [TempoPoint(tick / TICKS, bpm)]
        lyric = _clean_lyric(d.get("Lyric", ""))
        if length > 0 and lyric.lower() not in REST_LYRICS:
            note = Note(tick / TICKS, length / TICKS, int(round(_float(d.get("NoteNum"), 60.0))), lyric)
            intensity = _float(d.get("Intensity"))
            if intensity is not None and abs(intensity - 100.0) > 1e-9:
                note.attrs["velocity"] = round(max(0.0, min(2.0, intensity / 100.0)), 4)
            vbr = [_float(x, 0.0) for x in d.get("VBR", "").split(",") if x.strip()]
            if len(vbr) >= 3:
                pending_vbr.append((note, vbr))
            track.notes.append(note)
        tick += max(0, length)
    # drop redundant tempo points (UTAU repeats the tempo on the first note)
    pts: list[TempoPoint] = []
    for p in sorted(tempo_pts, key=lambda p: p.beat):
        if not pts or abs(p.bpm - pts[-1].bpm) > 1e-9:
            if pts and abs(p.beat - pts[-1].beat) < 1e-9:
                pts[-1] = p
            else:
                pts.append(p)
    project.tempo = TempoMap(pts)
    for note, vbr in pending_vbr:
        length_pct, cycle_ms, depth_cents = vbr[0], vbr[1], vbr[2]
        if depth_cents <= 0 or length_pct <= 0:
            note.attrs["vibrato_depth"] = 0.0
            continue
        secs = project.tempo.beat_to_sec(note.end) - project.tempo.beat_to_sec(note.onset)
        note.attrs["vibrato_depth"] = round(depth_cents / 100.0, 4)
        if cycle_ms > 0:
            note.attrs["vibrato_rate"] = round(1000.0 / cycle_ms, 4)
        note.attrs["vibrato_start"] = round(max(0.0, secs * (1.0 - min(100.0, length_pct) / 100.0)), 4)
    if any(_has_kana(n.lyric) for n in track.notes):
        track.language = "ja"
    track.sort()
    project.metadata["ust"] = {k: v for k, v in {
        "version": (version.get("_lines") or "").strip() or None,
        "voice_dir": setting.get("VoiceDir"),
        "project_name": name or None,
    }.items() if v}
    return project


def read_ust(path: str | Path) -> Project:
    """Read an UTAU ``.ust`` file."""
    path = Path(path)
    return loads_ust(path.read_bytes(), title=path.stem)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _fmt(x: float) -> str:
    return f"{float(x):.2f}"


def dumps_ust(project: Project, track: int | str = 0) -> str:
    """Serialise one track of ``project`` as UST 1.2 text (CRLF line endings)."""
    t: Track = project.track(track)
    # monophonic tick grid
    mono: list[list] = []
    for n in sorted(t.notes, key=lambda n: (n.onset, -n.pitch)):
        on, off = max(0, int(round(n.onset * TICKS))), int(round(n.end * TICKS))
        if mono and mono[-1][0] == on:
            continue
        if mono and mono[-1][1] > on:
            mono[-1][1] = on
        mono.append([on, off, n])
    mono = [m for m in mono if m[1] > m[0]]

    # entries: [start, end, note or None]; rests fill the gaps
    entries: list[list] = []
    cursor = 0
    for on, off, n in mono:
        if on > cursor:
            entries.append([cursor, on, None])
        entries.append([on, off, n])
        cursor = off
    changes = [(int(round(p.beat * TICKS)), p.bpm) for p in project.tempo.points[1:]]
    if changes and max(c[0] for c in changes) >= cursor:  # trailing rest so late changes land on a boundary
        entries.append([cursor, max(c[0] for c in changes) + TICKS, None])
    # rests can be split at tempo changes; changes inside notes move to the nearest boundary
    for tick, _ in changes:
        for k, (s, e, n) in enumerate(entries):
            if s < tick < e and n is None:
                entries[k:k + 1] = [[s, tick, None], [tick, e, None]]
                break
    starts = [e[0] for e in entries]
    tempo_at: dict[int, float] = {}
    for tick, bpm in changes:
        best = min(range(len(starts)), key=lambda k: abs(starts[k] - tick)) if starts else None
        if best is not None:
            tempo_at[best] = bpm

    nl = "\r\n"
    lines = ["[#VERSION]", "UST Version1.2", "Charset=UTF-8", "[#SETTING]", f"Tempo={_fmt(project.tempo.bpm_at(0.0))}",
             "Tracks=1", f"ProjectName={project.title}", "VoiceDir=%VOICE%uta", "OutFile=", "CacheDir=",
             "Tool1=wavtool.exe", "Tool2=resampler.exe", "Mode2=True"]
    for k, (s, e, n) in enumerate(entries):
        lines.append(f"[#{k:04d}]")
        lines.append(f"Length={e - s}")
        if n is None:
            nxt = next((x[2] for x in entries[k + 1:] if x[2] is not None), None)
            lines += ["Lyric=R", f"NoteNum={nxt.pitch if nxt else 60}"]
        else:
            lines += [f"Lyric={n.lyric}", f"NoteNum={max(0, min(127, int(n.pitch)))}"]
        lines.append("PreUtterance=")
        if n is not None:
            vel = n.attrs.get("velocity")
            lines.append(f"Intensity={int(round(float(vel) * 100)) if vel is not None else 100}")
            lines.append("Modulation=0")
        if k == 0:
            lines.append(f"Tempo={_fmt(project.tempo.bpm_at(0.0))}")
        elif k in tempo_at:
            lines.append(f"Tempo={_fmt(tempo_at[k])}")
        if n is not None and n.attrs.get("vibrato_depth") is not None:
            depth = float(n.attrs["vibrato_depth"])
            if depth <= 0:
                lines.append("VBR=0,180,0,20,20,0,0,0")
            else:
                rate = float(n.attrs.get("vibrato_rate") or 5.5)
                secs = project.tempo.beat_to_sec(n.end) - project.tempo.beat_to_sec(n.onset)
                start = n.attrs.get("vibrato_start")
                pct = 65.0 if start is None or secs <= 0 else max(0.0, min(100.0, 100.0 * (1 - float(start) / secs)))
                lines.append(f"VBR={pct:.6g},{1000.0 / max(rate, 0.1):.6g},{depth * 100:.6g},20,20,0,0,0")
    lines.append("[#TRACKEND]")
    return nl.join(lines) + nl


def write_ust(project: Project, path: str | Path, track: int | str = 0) -> Path:
    """Write ``project.track(track)`` to ``path`` as a UTF-8 UST file."""
    path = Path(path)
    path.write_bytes(dumps_ust(project, track).encode("utf-8"))
    return path
