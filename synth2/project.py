"""Project data model: Project -> Tracks -> Notes + parameter curves.

The model mirrors what a professional vocal synthesizer studio exposes:

* notes with pitch, timing, lyrics and optional phoneme overrides
* per-note expression attributes (vibrato, transition, consonant length, ...)
* per-track automation curves (pitch deviation, loudness, tension,
  breathiness, gender, voicing, tone shift, vibrato envelope)
* per-track voice selection, language, mixer settings and expression style

Everything serialises to plain JSON (``*.s2p.json``) so AI models can read and
write projects directly.
"""
from __future__ import annotations

import copy
import json
import uuid
from dataclasses import dataclass, field, fields, asdict
from pathlib import Path
from typing import Any, Iterable

from .timing import TempoMap, midi_to_note, note_to_midi

FORMAT_VERSION = 1

# name -> (default value, min, max, unit, description)
CURVE_SPECS: dict[str, tuple[float, float, float, str, str]] = {
    "pitchDelta": (0.0, -1200.0, 1200.0, "cents", "Pitch deviation added on top of the generated pitch curve"),
    "loudness": (0.0, -48.0, 12.0, "dB", "Loudness offset"),
    "tension": (0.0, -1.0, 1.0, "", "Vocal tension: negative = soft/lax, positive = pressed/bright"),
    "breathiness": (0.0, -1.0, 1.0, "", "Amount of breath noise"),
    "gender": (0.0, -1.0, 1.0, "", "Formant character: negative = more feminine, positive = more masculine"),
    "voicing": (1.0, 0.0, 1.0, "", "1 = fully voiced, 0 = whisper"),
    "toneShift": (0.0, -1200.0, 1200.0, "cents", "Shift formants independently from pitch"),
    "vibratoEnv": (1.0, 0.0, 2.0, "x", "Multiplier for the automatic vibrato depth"),
}

NOTE_ATTRIBUTES: dict[str, tuple[Any, str]] = {
    "vibrato_depth": (None, "Vibrato depth in semitones (peak), default from style/voice"),
    "vibrato_rate": (None, "Vibrato rate in Hz"),
    "vibrato_start": (None, "Seconds after note onset before vibrato begins"),
    "transition": (None, "Pitch transition (portamento) time in seconds into this note"),
    "overshoot": (None, "Pitch overshoot amount 0..1 when arriving at this note"),
    "consonant_length": (1.0, "Multiplier for onset consonant durations"),
    "tension": (None, "Per-note tension offset -1..1"),
    "breathiness": (None, "Per-note breathiness offset -1..1"),
    "detune": (0.0, "Static pitch offset in cents"),
    "velocity": (1.0, "Note dynamics 0..2 (1 = normal)"),
    "language": (None, "Override track language for this note ('en', 'ja')"),
}


def _new_id() -> str:
    while True:
        h = uuid.uuid4().hex[:12]
        if not h.isdigit():  # an all-digit id would be taken for a track index by lookups
            return h


@dataclass
class Note:
    onset: float  # beats
    duration: float  # beats
    pitch: int  # MIDI note number
    lyric: str = "la"
    phonemes: str | None = None  # explicit phoneme override, space separated
    attrs: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)

    @property
    def end(self) -> float:
        return self.onset + self.duration

    @property
    def note_name(self) -> str:
        return midi_to_note(self.pitch)

    def attr(self, name: str, default=None):
        v = self.attrs.get(name)
        if v is None:
            spec = NOTE_ATTRIBUTES.get(name)
            return default if spec is None or spec[0] is None else spec[0]
        return v

    def to_dict(self):
        # round the end (not the duration) so back-to-back notes still touch exactly after a round trip
        onset = round(self.onset, 6)
        d = {
            "id": self.id,
            "onset": onset,
            "duration": round(round(self.end, 6) - onset, 6),
            "pitch": self.pitch,
            "lyric": self.lyric,
        }
        if self.phonemes:
            d["phonemes"] = self.phonemes
        if self.attrs:
            d["attrs"] = dict(self.attrs)
        return d

    @classmethod
    def from_dict(cls, d):
        pitch = d.get("pitch", 60)
        return cls(
            onset=float(d["onset"]),
            duration=float(d["duration"]),
            pitch=note_to_midi(pitch),
            lyric=str(d.get("lyric", "la")),
            phonemes=d.get("phonemes") or None,
            attrs=dict(d.get("attrs") or {}),
            id=str(d.get("id") or _new_id()),
        )


@dataclass
class Curve:
    """Automation curve: sorted control points (beat, value) with interpolation."""

    points: list[tuple[float, float]] = field(default_factory=list)
    interpolation: str = "cubic"  # "linear" | "cubic" | "step"

    def sort(self):
        self.points.sort(key=lambda p: p[0])

    def add(self, beat: float, value: float):
        self.points = [p for p in self.points if abs(p[0] - beat) > 1e-9]
        self.points.append((float(beat), float(value)))
        self.sort()

    def clear_range(self, start: float, end: float):
        self.points = [p for p in self.points if not (start <= p[0] <= end)]

    def to_dict(self):
        return {"interpolation": self.interpolation, "points": [[round(b, 6), round(v, 6)] for b, v in self.points]}

    @classmethod
    def from_dict(cls, d):
        if isinstance(d, list):
            c = cls([(float(b), float(v)) for b, v in d])
        else:
            c = cls([(float(b), float(v)) for b, v in d.get("points", [])], d.get("interpolation", "cubic"))
        c.sort()
        return c


@dataclass
class Track:
    name: str = "Vocal"
    voice: str = "aurora"
    language: str = "en"
    style: str = "pop"  # expression style used by the automatic expression engine
    notes: list[Note] = field(default_factory=list)
    curves: dict[str, Curve] = field(default_factory=dict)
    gain_db: float = 0.0
    pan: float = 0.0  # -1 left .. 1 right
    mute: bool = False
    solo: bool = False
    reverb_send: float = 0.25
    seed: int = 0
    voice_params: dict[str, float] = field(default_factory=dict)  # global tweaks (tension, breathiness...)
    id: str = field(default_factory=_new_id)
    # engine spec for this track ("" = inherit the project's): auto, formant, neural or neural:<voicebank>
    engine: str = ""
    # the track's role in the mix ("" = none): lead, double, harmony, answer, adlib or crowd; a double may name
    # the track it doubles ("double:Harmony"). The neural engine and its vocal chain read it (engine/vocal_chain.py)
    role: str = ""

    # ---- note editing -------------------------------------------------
    def sort(self):
        self.notes.sort(key=lambda n: (n.onset, n.pitch))

    def add_note(self, onset, duration, pitch, lyric="la", **attrs) -> Note:
        phonemes = attrs.pop("phonemes", None)
        n = Note(float(onset), float(duration), note_to_midi(pitch), lyric, phonemes, {k: v for k, v in attrs.items() if v is not None})
        self.notes.append(n)
        self.sort()
        return n

    def find_note(self, note_id: str) -> Note | None:
        return next((n for n in self.notes if n.id == note_id), None)

    def remove_note(self, note_id: str) -> bool:
        before = len(self.notes)
        self.notes = [n for n in self.notes if n.id != note_id]
        return len(self.notes) != before

    def notes_in_range(self, start: float = float("-inf"), end: float = float("inf")) -> list[Note]:
        return [n for n in self.notes if n.onset < end and n.end > start]

    def curve(self, name: str, create: bool = True) -> Curve | None:
        if name not in CURVE_SPECS:
            raise KeyError(f"unknown curve {name!r}; valid: {', '.join(CURVE_SPECS)}")
        if name not in self.curves and create:
            self.curves[name] = Curve()
        return self.curves.get(name)

    @property
    def end(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    def overlaps(self) -> list[tuple[Note, Note]]:
        """Return pairs of overlapping notes (a monophonic voice cannot sing them)."""
        self.sort()
        out = []
        for a, b in zip(self.notes, self.notes[1:]):
            if b.onset < a.end - 1e-5:
                out.append((a, b))
        return out

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "voice": self.voice,
            "language": self.language,
            "style": self.style,
            "gain_db": self.gain_db,
            "pan": self.pan,
            "mute": self.mute,
            "solo": self.solo,
            "reverb_send": self.reverb_send,
            "seed": self.seed,
            "voice_params": dict(self.voice_params),
            "notes": [n.to_dict() for n in sorted(self.notes, key=lambda n: n.onset)],
            "curves": {k: c.to_dict() for k, c in self.curves.items() if c.points},
            **({"engine": self.engine} if self.engine else {}),
            **({"role": self.role} if self.role else {}),
        }

    @classmethod
    def from_dict(cls, d):
        t = cls(
            name=d.get("name", "Vocal"),
            voice=d.get("voice", "aurora"),
            language=d.get("language", "en"),
            style=d.get("style", "pop"),
            gain_db=float(d.get("gain_db", 0.0)),
            pan=float(d.get("pan", 0.0)),
            mute=bool(d.get("mute", False)),
            solo=bool(d.get("solo", False)),
            reverb_send=float(d.get("reverb_send", 0.25)),
            seed=int(d.get("seed", 0)),
            voice_params={k: float(v) for k, v in (d.get("voice_params") or {}).items()},
            id=str(d.get("id") or _new_id()),
            engine=str(d.get("engine") or ""),
            role=str(d.get("role") or ""),
        )
        t.notes = [Note.from_dict(n) for n in d.get("notes", [])]
        t.curves = {k: Curve.from_dict(v) for k, v in (d.get("curves") or {}).items() if k in CURVE_SPECS}
        t.sort()
        return t


@dataclass
class MasterSettings:
    """Master / mix settings. The fields after ``limiter`` drive the mixing and mastering chain
    (``synth2.engine.mixing``); their defaults suit both vocal-only projects and full songs."""

    gain_db: float = 0.0  # trim after normalisation (or after the loudness target)
    reverb_room: float = 0.55  # 0..1 room size
    reverb_wet: float = 1.0  # scales every track's send
    normalize: bool = True  # vocal-only: peak-normalise to -1 dBFS; songs: hit the loudness target
    limiter: bool = True
    # ---- mastering
    loudness_target: float | None = None  # integrated LUFS; None = auto (-11 with a backing band, else peak-normalise)
    ceiling_db: float = -1.0  # true-peak ceiling (dBTP) of the mastering limiter
    glue: float = 0.5  # master glue compression 0..1 (songs with a backing band)
    master_eq: list | None = None  # EQ band specs folded into every source; None = preset (subsonic HPF, +1 dB air)
    # ---- vocals
    vocal_chain: bool = True  # channel strip on every vocal: HPF, de-mud, presence, air, de-esser, compressor
    vocal_delay: float = 0.0  # ping-pong delay return level for the vocals (0 = off, 0.1-0.3 typical)
    delay_beats: float = 0.5  # delay time in beats (0.5 = eighth notes)
    delay_feedback: float = 0.35
    delay_duck_db: float = 6.0  # the delay return dips this much while the vocals sing (repeats bloom in the gaps)
    # ---- band (backing stems from synth2.arranger)
    vocal_balance_db: float | None = 4.0  # band gain is set so vocals sit this far above it in 1-5 kHz; None = off
    backing_gain_db: float = 0.0  # band trim after the vocal balance
    band_duck_db: float = 1.5  # band dips this much (x role factor) while the vocals sing
    vocal_carve_db: float = 3.0  # and its 1.5-4 kHz band is carved by this much (x role factor)
    sidechain_db: float = 6.0  # kick ducking depth for stems flagged sidechain (x role factor)
    sidechain_release: float = 0.16  # s
    band_width: float = 1.0  # stereo width multiplier for the band (bass stays mono)
    band_glue: float = 0.5  # instrument bus compression 0..1
    band_reverb: float = 0.5  # scales the role-based sends of the band into the song reverb (stems bring their own)
    sub_max_db: float | None = 4.0  # band's 30-60 Hz octave may sit this far above its 120-240 Hz one (else low shelf, max -4 dB)
    # per-track / per-stem overrides, keyed by track id or name, backing stem name or role:
    # {gain_db, pan, reverb, delay, eq_add: [bands], chain: bool, comp: {...} | None, sidechain_db,
    #  duck, carve, width, mono_below, crest_db (transient limiter: peaks within this many dB of the loud level;
    #  drums / perc / fx by default, opt-in for vocals)}
    mix_overrides: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        valid = {f.name for f in fields(cls)}
        return cls(**{k: copy.deepcopy(v) for k, v in (d or {}).items() if k in valid})


@dataclass
class Project:
    title: str = "Untitled"
    tempo: TempoMap = field(default_factory=TempoMap)
    time_signature: tuple[int, int] = (4, 4)
    key: str = "C major"
    sample_rate: int = 44100
    tracks: list[Track] = field(default_factory=list)
    master: MasterSettings = field(default_factory=MasterSettings)
    metadata: dict[str, Any] = field(default_factory=dict)
    # backing-track arrangement for synth2.arranger (JSON-serialisable dict, see synth2/arranger/FORMAT.md)
    backing: dict | None = None
    # default engine spec of the tracks ("" = auto): auto, formant, neural or neural:<voicebank>
    engine: str = ""
    # the file the project was loaded from or last saved to (not stored in it): relative sound paths
    # of the backing are looked up next to it
    path: str | None = field(default=None, compare=False, repr=False)

    # ---- convenience -----------------------------------------------------
    @classmethod
    def new(cls, title="Untitled", bpm=120.0, voice="aurora", language="en", **kw) -> "Project":
        p = cls(title=title, **kw)
        p.tempo.set(0.0, bpm)
        p.add_track("Lead", voice=voice, language=language)
        return p

    def add_track(self, name="Vocal", voice="aurora", language="en", **kw) -> Track:
        t = Track(name=name, voice=voice, language=language, **kw)
        if "seed" not in kw:
            t.seed = len(self.tracks) * 7919 + 17
        self.tracks.append(t)
        return t

    def track(self, ref: str | int | None = None) -> Track:
        """Look up a track by index, id or (case-insensitive) name. ``None`` = first track.

        An exact id match wins over a digit string read as an index (ids in older projects may be all digits).
        """
        if not self.tracks:
            raise LookupError("project has no tracks")
        if ref is None:
            return self.tracks[0]
        if isinstance(ref, int):
            return self.tracks[ref]
        ref = str(ref)
        for t in self.tracks:
            if t.id == ref:
                return t
        if ref.isdigit():
            return self.tracks[int(ref)]
        for t in self.tracks:
            if t.name.lower() == ref.lower():
                return t
        raise LookupError(f"no track {ref!r}")

    @property
    def bpm(self) -> float:
        return self.tempo.bpm_at(0.0)

    def beats_per_bar(self) -> float:
        num, den = self.time_signature
        return num * 4.0 / den

    def duration_beats(self) -> float:
        return max((t.end for t in self.tracks), default=0.0)

    def duration_seconds(self, tail: float = 0.0) -> float:
        return self.tempo.beat_to_sec(self.duration_beats()) + tail

    def copy(self) -> "Project":
        p = Project.from_dict(copy.deepcopy(self.to_dict()))
        p.path = self.path
        return p

    # ---- serialisation ---------------------------------------------------
    def to_dict(self):
        return {
            "format": "synth2-project",
            "version": FORMAT_VERSION,
            "title": self.title,
            "tempo": self.tempo.to_list(),
            "time_signature": list(self.time_signature),
            "key": self.key,
            "sample_rate": self.sample_rate,
            "master": self.master.to_dict(),
            "metadata": dict(self.metadata),
            "tracks": [t.to_dict() for t in self.tracks],
            **({"backing": copy.deepcopy(self.backing)} if self.backing else {}),
            **({"engine": self.engine} if self.engine else {}),
        }

    @classmethod
    def from_dict(cls, d) -> "Project":
        if "bpm" in d and "tempo" not in d:
            d = dict(d, tempo=[{"beat": 0, "bpm": d["bpm"]}])
        p = cls(
            title=d.get("title", "Untitled"),
            tempo=TempoMap.from_list(d.get("tempo")),
            time_signature=tuple(d.get("time_signature", (4, 4))),
            key=d.get("key", "C major"),
            sample_rate=int(d.get("sample_rate", 44100)),
            master=MasterSettings.from_dict(d.get("master")),
            metadata=dict(d.get("metadata") or {}),
            engine=str(d.get("engine") or ""),
        )
        p.tracks = [Track.from_dict(t) for t in d.get("tracks", [])]
        backing = d.get("backing")
        p.backing = copy.deepcopy(backing) if isinstance(backing, dict) and backing else None
        return p

    def to_json(self, indent=2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    @classmethod
    def from_json(cls, s: str) -> "Project":
        return cls.from_dict(json.loads(s))

    def save(self, path: str | Path):
        Path(path).write_text(self.to_json(), encoding="utf-8")
        self.path = str(Path(path).resolve())

    @classmethod
    def load(cls, path: str | Path) -> "Project":
        """Load any supported format: .json/.s2p, .s2s score, .svp, .mid, .musicxml/.mxl, .ust."""
        from .io import load_any

        project = load_any(path)
        project.path = str(Path(path).resolve())
        return project

    def summary(self) -> dict:
        """Compact description of the project, useful for AI agents."""
        out = {
            "title": self.title,
            "bpm": self.bpm,
            "tempo_changes": len(self.tempo.points) - 1,
            "time_signature": f"{self.time_signature[0]}/{self.time_signature[1]}",
            "key": self.key,
            "length_beats": round(self.duration_beats(), 3),
            "length_seconds": round(self.duration_seconds(), 2),
            "tracks": [],
        }
        for i, t in enumerate(self.tracks):
            pitches = [n.pitch for n in t.notes]
            out["tracks"].append({
                "index": i,
                "id": t.id,
                "name": t.name,
                "voice": t.voice,
                "language": t.language,
                "style": t.style,
                "notes": len(t.notes),
                "range": f"{midi_to_note(min(pitches))}-{midi_to_note(max(pitches))}" if pitches else None,
                "curves": sorted(k for k, c in t.curves.items() if c.points),
                "lyrics": " ".join(n.lyric for n in t.notes)[:400],
                "mute": t.mute,
                "gain_db": t.gain_db,
                "pan": t.pan,
                **({"engine": t.engine} if t.engine else {}),
                **({"role": t.role} if t.role else {}),
            })
        if self.engine:
            out["engine"] = self.engine
        if self.backing:
            secs = self.backing.get("sections") or []
            out["backing"] = {"instruments": sorted(self.backing.get("instruments") or {}),
                              "sections": [s.get("name") for s in secs if isinstance(s, dict)]}
        return out


def iter_active_tracks(project: Project) -> Iterable[Track]:
    soloed = [t for t in project.tracks if t.solo]
    for t in project.tracks:
        if t.mute:
            continue
        if soloed and not t.solo:
            continue
        yield t
