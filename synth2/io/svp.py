"""Synthesizer V Studio project (``.svp``) import and export.

An ``.svp`` file is JSON (Synthesizer V writes a trailing NUL byte) with time
in *blicks*: 705 600 000 per quarter note.  The parts used here::

    {"version": 113,
     "time": {"meter": [{"index": 0, "numerator": 4, "denominator": 4}],
              "tempo": [{"position": 0, "bpm": 120.0}]},
     "library": [group, ...],            # reusable note groups
     "tracks": [{"name", "dispColor", "dispOrder", "renderEnabled",
                 "mixer": {"gainDecibel", "pan", "mute", "solo", "display"},
                 "mainGroup": group,
                 "mainRef": {"groupID", "blickOffset", "pitchOffset",
                             "isInstrumental", "database": {...}, "voice": {...}},
                 "groups": [ref, ...]}],  # references into "library"
     "renderConfig": {...}}

    group = {"name", "uuid", "parameters": {curve: {"mode", "points": [pos, val, ...]}},
             "notes": [{"onset", "duration", "lyrics", "phonemes", "pitch",
                        "detune", "attributes": {...}}]}

Import flattens the main group and every referenced library group (applying
``blickOffset`` / ``pitchOffset``) into one Synth2 track, maps parameter
curves onto :data:`~synth2.project.CURVE_SPECS` (clamped to our ranges) and maps
the note attributes whose meaning is unambiguous:

========================  ====================================================
Synthesizer V             Synth2
========================  ====================================================
``fF0Vbr`` (Hz)           ``vibrato_rate``
``tF0VbrStart`` (s)       ``vibrato_start``
``dF0Vbr == 0``           ``vibrato_depth = 0`` (other depths keep the default:
                          the SynthV depth scale is not documented)
``tF0Left + tF0Right``    ``transition`` (seconds, SynthV default 0.07 + 0.07)
note ``detune`` (cents)   ``detune``
``phonemes``              ``Note.phonemes`` when every symbol exists in our
                          inventory
========================  ====================================================

Everything else that matters for a lossless SynthV -> Synth2 -> SynthV round
trip (raw note attributes, unsupported phoneme strings, singer database and
voice settings, track colour, render config) is kept in
``project.metadata["svp"]`` and reused by :func:`write_svp`.  The original
singer name is recorded per track (``metadata["svp"]["tracks"][track_id]["singer"]``);
the Synth2 voice stays the default ``"aurora"``.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from ..project import CURVE_SPECS, Curve, Note, Project, Track
from ..timing import TempoMap, TempoPoint

BLICKS_PER_BEAT = 705_600_000
SVP_VERSION = 113
SVP_CURVES = ("pitchDelta", "vibratoEnv", "loudness", "tension", "breathiness", "voicing", "gender", "toneShift")

# SynthV defaults for the attributes we map
_SV_TRANSITION_SIDE = 0.07  # tF0Left / tF0Right, seconds
_MAPPED_ATTRS = ("fF0Vbr", "tF0VbrStart", "dF0Vbr", "tF0Left", "tF0Right")

_LANG_FROM_SV = {"japanese": "ja", "english": "en"}
_LANG_TO_SV = {"ja": ("japanese", "romaji"), "en": ("english", "arpabet")}
_COLORS = ["ff7db235", "ff15a7c4", "ffd96c4f", "ffb57ed9", "ffe6c13d", "ff4f8ad9", "ffd94f8a", "ff4fd9b5"]


def _beats(blicks: float) -> float:
    return float(blicks) / BLICKS_PER_BEAT


def _blicks(beats: float) -> int:
    return int(round(float(beats) * BLICKS_PER_BEAT))


def _num(x: float):
    """Write integral floats as ints (as SynthV itself does)."""
    x = float(x)
    return int(x) if x.is_integer() else x


def _language(value: Any) -> str | None:
    return _LANG_FROM_SV.get(str(value or "").strip().lower())


def _has_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def _valid_phonemes(s: str) -> bool:
    from ..phonemes.inventory import is_valid

    toks = s.split()
    return bool(toks) and all(is_valid(t) for t in toks)


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------
def _pairs(points: Any) -> list[tuple[float, float]]:
    if not points:
        return []
    if isinstance(points[0], (list, tuple)):
        return [(float(p[0]), float(p[1])) for p in points if len(p) >= 2]
    return [(float(points[i]), float(points[i + 1])) for i in range(0, len(points) - 1, 2)]


def _merge_curves(track: Track, params: dict, blick_offset: float):
    for name in SVP_CURVES:
        param = params.get(name)
        if not isinstance(param, dict):
            continue
        pts = _pairs(param.get("points"))
        if not pts:
            continue
        _, lo, hi, _, _ = CURVE_SPECS[name]
        curve = track.curves.get(name)
        if curve is None:
            mode = str(param.get("mode", "cubic")).lower()
            curve = track.curves[name] = Curve([], "linear" if mode == "linear" else "cubic")
        merged = {round(b, 9): (b, v) for b, v in curve.points}  # bulk merge: hand-drawn curves are long
        for pos, val in pts:
            b = _beats(pos + blick_offset)
            merged[round(b, 9)] = (b, min(hi, max(lo, val)))
        curve.points = sorted(merged.values())


def _note_from_svp(d: dict, blick_offset: float, pitch_offset: int, meta: dict) -> Note:
    lyric = str(d.get("lyrics", "") or "").strip() or "la"
    note = Note(
        onset=_beats(float(d.get("onset", 0)) + blick_offset),
        duration=_beats(d.get("duration", BLICKS_PER_BEAT)),
        pitch=int(round(float(d.get("pitch", 60)))) + int(pitch_offset),
        lyric=lyric,
    )
    detune = float(d.get("detune") or 0.0)
    if detune:
        note.attrs["detune"] = detune
    lang = _language(d.get("languageOverride"))
    if lang:
        note.attrs["language"] = lang
    attrs = d.get("attributes") or {}
    if isinstance(attrs, dict):
        if isinstance(attrs.get("fF0Vbr"), (int, float)):
            note.attrs["vibrato_rate"] = float(attrs["fF0Vbr"])
        if isinstance(attrs.get("tF0VbrStart"), (int, float)):
            note.attrs["vibrato_start"] = max(0.0, float(attrs["tF0VbrStart"]))
        if isinstance(attrs.get("dF0Vbr"), (int, float)) and float(attrs["dF0Vbr"]) == 0.0:
            note.attrs["vibrato_depth"] = 0.0
        left, right = attrs.get("tF0Left"), attrs.get("tF0Right")
        if isinstance(left, (int, float)) or isinstance(right, (int, float)):
            lv = float(left) if isinstance(left, (int, float)) else _SV_TRANSITION_SIDE
            rv = float(right) if isinstance(right, (int, float)) else _SV_TRANSITION_SIDE
            note.attrs["transition"] = round(max(0.0, lv + rv), 6)
        if attrs:
            meta.setdefault("note_attributes", {})[note.id] = dict(attrs)
    phonemes = str(d.get("phonemes") or "").strip()
    if phonemes:
        if _valid_phonemes(phonemes):
            note.phonemes = phonemes
        else:
            meta.setdefault("note_phonemes", {})[note.id] = phonemes
    return note


def _add_group(track: Track, group: dict, blick_offset: float, pitch_offset: int, meta: dict,
               begin: float | None = None, end: float | None = None):
    for nd in group.get("notes") or []:
        if not isinstance(nd, dict):
            continue
        onset = float(nd.get("onset", 0)) + blick_offset
        if begin is not None and end is not None and not (begin <= onset < end):
            continue
        track.notes.append(_note_from_svp(nd, blick_offset, pitch_offset, meta))
    _merge_curves(track, group.get("parameters") or {}, blick_offset)


def project_from_svp_dict(d: dict, title: str | None = None) -> Project:
    """Convert a parsed ``.svp`` JSON document into a :class:`Project`."""
    time = d.get("time") or {}
    tempos = sorted((t for t in time.get("tempo") or [] if isinstance(t, dict)), key=lambda t: float(t.get("position", 0)))
    points = [TempoPoint(_beats(t.get("position", 0)), float(t.get("bpm", 120.0))) for t in tempos if float(t.get("bpm", 0)) > 0]
    meters = sorted((m for m in time.get("meter") or [] if isinstance(m, dict)), key=lambda m: int(m.get("index", 0)))
    render = d.get("renderConfig") or {}
    project = Project(title=title or str(render.get("filename") or "") or "Untitled", tempo=TempoMap(points or [TempoPoint(0.0, 120.0)]))
    if meters:
        project.time_signature = (int(meters[0].get("numerator", 4)), int(meters[0].get("denominator", 4)))
    if isinstance(render.get("sampleRate"), int) and render["sampleRate"] > 0:
        project.sample_rate = int(render["sampleRate"])

    meta: dict[str, Any] = {"version": d.get("version", SVP_VERSION), "tracks": {}}
    if render:
        meta["renderConfig"] = render
    if len(meters) > 1:
        meta["meters"] = meters
    library = {str(g.get("uuid")): g for g in d.get("library") or [] if isinstance(g, dict)}

    svp_tracks = [t for t in d.get("tracks") or [] if isinstance(t, dict)]
    order = sorted(range(len(svp_tracks)), key=lambda i: (svp_tracks[i].get("dispOrder", i), i))
    skipped = []
    for i in order:
        st = svp_tracks[i]
        main_ref = st.get("mainRef") or {}
        name = str(st.get("name") or f"Track {i + 1}")
        if main_ref.get("isInstrumental"):
            skipped.append(name)
            continue
        track = project.add_track(name)
        mixer = st.get("mixer") or {}
        track.gain_db = float(mixer.get("gainDecibel", 0.0))
        track.pan = max(-1.0, min(1.0, float(mixer.get("pan", 0.0))))
        track.mute = bool(mixer.get("mute", False))
        track.solo = bool(mixer.get("solo", False))
        db = main_ref.get("database") or {}
        lang = _language(db.get("languageOverride")) or _language(db.get("language"))
        _add_group(track, st.get("mainGroup") or {}, float(main_ref.get("blickOffset", 0)),
                   int(main_ref.get("pitchOffset", 0)), meta)
        for ref in st.get("groups") or []:
            if not isinstance(ref, dict) or ref.get("isInstrumental"):
                continue
            group = library.get(str(ref.get("groupID")))
            if group is None:
                meta.setdefault("missing_groups", []).append(str(ref.get("groupID")))
                continue
            begin, end = ref.get("blickAbsoluteBegin"), ref.get("blickAbsoluteEnd")
            window = (float(begin), float(end)) if isinstance(begin, (int, float)) and isinstance(end, (int, float)) and end > begin else (None, None)
            _add_group(track, group, float(ref.get("blickOffset", 0)), int(ref.get("pitchOffset", 0)), meta, *window)
        track.sort()
        if lang is None:
            lang = "ja" if any(_has_kana(n.lyric) for n in track.notes) else "en"
        track.language = lang
        meta["tracks"][track.id] = {
            "name": name,
            "singer": str(db.get("name") or ""),
            "language": lang,
            "database": db,
            "voice": main_ref.get("voice") or {},
            "dictionary": main_ref.get("dictionary", ""),
            "dispColor": st.get("dispColor"),
            "renderEnabled": st.get("renderEnabled", False),
        }
    if skipped:
        meta["skipped_instrumental_tracks"] = skipped
    project.metadata["svp"] = meta
    return project


def loads_svp(data: bytes | str, title: str | None = None) -> Project:
    """Parse ``.svp`` content (bytes or text; trailing NULs are tolerated)."""
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig", errors="replace")
    text = data.lstrip("﻿").strip().rstrip("\x00").strip()
    return project_from_svp_dict(json.loads(text), title=title)


def read_svp(path: str | Path) -> Project:
    """Read a Synthesizer V Studio ``.svp`` project."""
    path = Path(path)
    return loads_svp(path.read_bytes(), title=path.stem)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def _sv_curve(curve: Curve | None) -> dict:
    if curve is None or not curve.points:
        return {"mode": "cubic", "points": []}
    pts = sorted(curve.points)
    flat: list = []
    if curve.interpolation == "step":
        eps = BLICKS_PER_BEAT // 480
        for k, (b, v) in enumerate(pts):
            if k:
                flat += [max(_blicks(b) - eps, flat[-2] + 1), flat[-1]]
            flat += [_blicks(b), _num(v)]
        return {"mode": "linear", "points": flat}
    for b, v in pts:
        flat += [_blicks(b), _num(v)]
    return {"mode": "linear" if curve.interpolation == "linear" else "cubic", "points": flat}


def _sv_attributes(note: Note, raw: dict) -> dict:
    out = {k: v for k, v in raw.items() if k not in _MAPPED_ATTRS}
    a = note.attrs
    if a.get("vibrato_rate") is not None:
        out["fF0Vbr"] = float(a["vibrato_rate"])
    if a.get("vibrato_start") is not None:
        out["tF0VbrStart"] = float(a["vibrato_start"])
    depth = a.get("vibrato_depth")
    if depth is not None and float(depth) <= 0.0:
        out["dF0Vbr"] = 0.0
    elif isinstance(raw.get("dF0Vbr"), (int, float)) and raw["dF0Vbr"] != 0:
        out["dF0Vbr"] = raw["dF0Vbr"]
    t = a.get("transition")
    if t is not None:
        left, right = raw.get("tF0Left"), raw.get("tF0Right")
        lv = float(left) if isinstance(left, (int, float)) else _SV_TRANSITION_SIDE
        rv = float(right) if isinstance(right, (int, float)) else _SV_TRANSITION_SIDE
        if abs(lv + rv - float(t)) < 1e-6 and (left is not None or right is not None):
            out.update({k: raw[k] for k in ("tF0Left", "tF0Right") if k in raw})
        else:
            out["tF0Left"] = out["tF0Right"] = float(t) / 2.0
    return out


def _take() -> dict:
    return {"activeTakeId": 0, "takes": [{"id": 0, "expr": 0.0, "liked": False}]}


def _sv_note(note: Note, meta: dict) -> dict:
    raw_attrs = (meta.get("note_attributes") or {}).get(note.id) or {}
    phonemes = note.phonemes or (meta.get("note_phonemes") or {}).get(note.id) or ""
    d = {
        "musicalType": "singing",
        "onset": _blicks(note.onset),
        "duration": max(1, _blicks(note.duration)),
        "lyrics": note.lyric,
        "phonemes": phonemes,
        "accent": "",
        "pitch": int(note.pitch),
        "detune": _num(note.attrs.get("detune") or 0),
        "instantMode": False,
        "attributes": _sv_attributes(note, raw_attrs),
        "systemAttributes": {},
        "pitchTakes": _take(),
        "timbreTakes": _take(),
    }
    lang = note.attrs.get("language")
    if lang in _LANG_TO_SV:
        d["languageOverride"] = _LANG_TO_SV[lang][0]
    return d


def project_to_svp_dict(project: Project) -> dict:
    """Convert ``project`` to a Synthesizer V Studio JSON document (a dict)."""
    meta = project.metadata.get("svp") or {}
    version = meta.get("version") if isinstance(meta.get("version"), int) else SVP_VERSION
    num, den = project.time_signature
    tracks = []
    for i, t in enumerate(project.tracks):
        tmeta = (meta.get("tracks") or {}).get(t.id) or {}
        gid = str(uuid.uuid4())
        db = {"name": "", "language": "", "phoneset": "", "languageOverride": "", "phonesetOverride": "", "backendType": ""}
        db.update(tmeta.get("database") or {})
        if not (tmeta.get("database") and tmeta.get("language") == t.language):
            # new track, or the language changed since import: describe it in SynthV terms
            lang, phoneset = _LANG_TO_SV.get(t.language, _LANG_TO_SV["en"])
            db["language"], db["phoneset"] = lang, phoneset
            db["languageOverride"] = db["phonesetOverride"] = ""
        tracks.append({
            "name": t.name,
            "dispColor": tmeta.get("dispColor") or _COLORS[i % len(_COLORS)],
            "dispOrder": i,
            "renderEnabled": bool(tmeta.get("renderEnabled", False)),
            "mixer": {"gainDecibel": float(t.gain_db), "pan": float(t.pan), "mute": bool(t.mute),
                      "solo": bool(t.solo), "display": True},
            "mainGroup": {
                "name": "main",
                "uuid": gid,
                "parameters": {name: _sv_curve(t.curves.get(name)) for name in SVP_CURVES},
                "vocalModes": {},
                "notes": [_sv_note(n, meta) for n in sorted(t.notes, key=lambda n: (n.onset, n.pitch))],
            },
            "mainRef": {
                "groupID": gid,
                "blickAbsoluteBegin": 0,
                "blickAbsoluteEnd": -1,
                "blickOffset": 0,
                "pitchOffset": 0,
                "isInstrumental": False,
                "systemPitchDelta": {"mode": "cubic", "points": []},
                "database": db,
                "dictionary": tmeta.get("dictionary", ""),
                "voice": tmeta.get("voice") or {"vocalModeInherited": True, "vocalModePreset": "", "vocalModeParams": {}},
            },
            "groups": [],
        })
    render = {"destination": "", "filename": project.title, "numChannels": 1, "aspirationFormat": "noAspiration",
              "bitDepth": 16, "sampleRate": int(project.sample_rate), "exportMixDown": True, "exportPitch": False}
    render.update(meta.get("renderConfig") or {})
    return {
        "version": version,
        "time": {
            "meter": [{"index": 0, "numerator": int(num), "denominator": int(den)}],
            "tempo": [{"position": _blicks(p.beat), "bpm": _num(p.bpm)} for p in project.tempo.points],
        },
        "library": [],
        "tracks": tracks,
        "renderConfig": render,
    }


def dumps_svp(project: Project) -> str:
    """Serialise ``project`` as ``.svp`` text (JSON plus the trailing NUL SynthV writes)."""
    return json.dumps(project_to_svp_dict(project), ensure_ascii=False, separators=(",", ":")) + "\x00"


def write_svp(project: Project, path: str | Path) -> Path:
    """Write ``project`` as a Synthesizer V Studio ``.svp`` file."""
    path = Path(path)
    path.write_bytes(dumps_svp(project).encode("utf-8"))
    return path
