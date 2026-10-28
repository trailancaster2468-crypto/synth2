"""Render orchestration: project -> per-track vocals -> mix -> master."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..phonemes import english
from ..project import CURVE_SPECS, Project, Track, iter_active_tracks
from ..timing import midi_to_note
from ..voices import get_voice
from . import effects
from .backend import backend_for_voice, register_backend
from .expression import build_dynamics, build_pitch, evaluate_curve, get_style
from .params import build_params
from .planner import plan_track
from .vocoder import synthesize

VOICE_TAIL = 0.6  # seconds rendered after the last note (release)


def frame_hop(sr: int) -> int:
    return max(32, int(round(sr * 0.0029)))


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    stems: dict[str, np.ndarray] = field(default_factory=dict)
    report: dict = field(default_factory=dict)
    start_seconds: float = 0.0

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sample_rate

    def save(self, path: str | Path, bits: int = 16) -> Path:
        from ..io.wav import write_wav

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        return write_wav(path, self.audio, self.sample_rate, bits)

    def save_stems(self, directory: str | Path, bits: int = 16) -> list[Path]:
        from ..io.wav import write_wav

        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        names = {t["id"]: t["name"] for t in self.report.get("tracks", [])}
        out = []
        for tid, audio in self.stems.items():
            safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", names.get(tid, tid)) or tid
            out.append(write_wav(d / f"{safe}.wav", audio, self.sample_rate, bits))
        return out

    def mono(self) -> np.ndarray:
        return self.audio.mean(axis=1)


class FormantBackend:
    """The built-in parametric singing engine."""

    name = "formant"

    def voices(self) -> list[str]:
        from ..voices import VOICES

        return list(VOICES)

    def render_track(self, project: Project, track: Track, sample_rate: int, t0: float, t1: float,
                     note_range: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
        sr = sample_rate
        voice = get_voice(track.voice).with_params(track.voice_params)
        style = get_style(track.style or voice.default_style)
        rng = np.random.default_rng(track.seed)
        notes = track.notes
        if note_range is not None:
            b0, b1 = note_range
            notes = [n for n in notes if n.end > b0 - 4 and n.onset < b1 + 1]
        plan = plan_track(notes, project.tempo, track.language, style, rng)

        hop = frame_hop(sr)
        fp = hop / sr
        n_frames = max(2, int(np.ceil((t1 - t0) / fp)) + 1)
        t = t0 + np.arange(n_frames) * fp
        b2s = project.tempo.beat_to_sec
        curves = {name: evaluate_curve(track.curves.get(name), t, b2s, spec[0]) for name, spec in CURVE_SPECS.items()}
        for name, spec in CURVE_SPECS.items():
            curves[name] = np.clip(curves[name], spec[1], spec[2])

        pitch = build_pitch(t, plan.timings, style, voice, rng, vibrato_env=curves["vibratoEnv"])
        pitch = pitch + curves["pitchDelta"] / 100.0
        loud, tension, breath = build_dynamics(t, plan.timings, style, voice, pitch)
        loud = loud + curves["loudness"] + 1.5 * (voice.tension + curves["tension"])
        tension = np.clip(tension + voice.tension + curves["tension"], -1, 1)
        breath = np.clip(breath + voice.breathiness + curves["breathiness"], -1, 1)
        params = build_params(plan, t, voice, pitch, loud, tension, breath, curves["gender"],
                              curves["toneShift"], curves["voicing"])
        audio = synthesize(params, voice, sr, rng)
        n_out = int(round((t1 - t0) * sr))
        audio = audio[:n_out]
        if len(audio) < n_out:
            audio = np.pad(audio, (0, n_out - len(audio)))

        warnings = list(plan.warnings)
        for n in notes:
            if n.pitch < voice.range_low - 2 or n.pitch > voice.range_high + 2:
                warnings.append(
                    f"note {n.id} ({midi_to_note(n.pitch)} at beat {n.onset:g}) is outside {voice.name}'s "
                    f"range {midi_to_note(voice.range_low)}-{midi_to_note(voice.range_high)}")
        if track.language == "en":
            guessed = sorted({
                w for n in notes for w in re.findall(r"[A-Za-z']+", n.lyric or "")
                if not n.phonemes and not (n.lyric or "").startswith((".", "[")) and w.lower() not in ("br",)
                and english.lookup(w) is None and len(w) > 1
            })
            if guessed:
                warnings.append("pronunciation guessed (not in dictionary): " + ", ".join(guessed[:20])
                                + " - use '+' continuation notes for split words or set explicit phonemes")
        if plan.start < 0:
            warnings.append("the first consonants start before beat 0 and are cut; start the song at beat 1 or later")
        report = {
            "id": track.id, "name": track.name, "voice": voice.id, "style": style.name,
            "notes": len(notes), "warnings": warnings, "phonemes": plan.note_phonemes,
            "peak_db": _db(np.max(np.abs(audio)) if len(audio) else 0.0),
        }
        return audio.astype(np.float32), report


register_backend(FormantBackend())


def _db(x: float) -> float:
    return float(round(20 * np.log10(max(x, 1e-9)), 2))


def render_track(project: Project, track: Track | str | int, sample_rate: int | None = None,
                 tail: float = VOICE_TAIL) -> tuple[np.ndarray, dict]:
    """Render one track dry (no pan / reverb), from beat 0 to the end of the track."""
    if not isinstance(track, Track):
        track = project.track(track)
    sr = int(sample_rate or project.sample_rate)
    t1 = project.tempo.beat_to_sec(track.end) + tail
    return backend_for_voice(track.voice).render_track(project, track, sr, 0.0, t1)


def render_project(project: Project, *, sample_rate: int | None = None, track_ids=None,
                   start_beat: float | None = None, end_beat: float | None = None,
                   progress: Callable[[float, str], None] | None = None) -> RenderResult:
    """Render and mix a whole project (or a beat range / subset of tracks)."""
    t_start = time.perf_counter()
    sr = int(sample_rate or project.sample_rate)
    tracks = list(iter_active_tracks(project))
    if track_ids:
        wanted = {str(x) for x in track_ids}
        tracks = [t for i, t in enumerate(project.tracks)
                  if t.id in wanted or t.name in wanted or str(i) in wanted]
    b2s = project.tempo.beat_to_sec
    start_sec = b2s(start_beat) if start_beat is not None else 0.0
    end_b = end_beat if end_beat is not None else max((t.end for t in tracks), default=0.0)
    end_sec = max(b2s(end_b), start_sec + 0.1)
    room = project.master.reverb_room
    reverb_tail = 0.3 + 2.2 * room if any(t.reverb_send > 0 for t in tracks) else 0.0
    voice_end = end_sec + VOICE_TAIL
    total = int(round((voice_end - start_sec + reverb_tail) * sr))
    n_voice = int(round((voice_end - start_sec) * sr))

    dry = np.zeros((total, 2))
    bus = np.zeros((total, 2))
    stems: dict[str, np.ndarray] = {}
    track_reports = []
    note_range = (start_beat if start_beat is not None else -1e9, end_b) if (start_beat is not None or end_beat is not None) else None
    for k, tr in enumerate(tracks):
        if progress:
            progress(k / max(len(tracks), 1), f"rendering {tr.name}")
        if not tr.notes:
            continue
        mono, rep = backend_for_voice(tr.voice).render_track(project, tr, sr, start_sec, voice_end, note_range)
        mono = mono[:n_voice]
        st = effects.pan_stereo(mono.astype(np.float64), tr.pan, tr.gain_db)
        stem = np.zeros((total, 2))
        stem[: len(st)] = st
        stems[tr.id] = stem.astype(np.float32)
        dry += stem
        if tr.reverb_send > 0:
            bus += stem * tr.reverb_send
        rep.update({"gain_db": tr.gain_db, "pan": tr.pan})
        track_reports.append(rep)
    if progress:
        progress(0.9, "mixing")
    master = dry
    if reverb_tail > 0 and project.master.reverb_wet > 0 and np.any(bus):
        wet = effects.apply_reverb(bus, sr, room)[:total] * project.master.reverb_wet
        master = master + wet
    master *= 10 ** (project.master.gain_db / 20.0)
    peak = float(np.max(np.abs(master))) if master.size else 0.0
    if project.master.normalize and peak > 1e-6:
        master *= 10 ** (-1.0 / 20.0) / peak
    if project.master.limiter:
        master = effects.limiter(master, sr, -0.3)
    # trim the silent tail
    if master.size:
        lvl = np.max(np.abs(master), axis=1)
        nz = np.nonzero(lvl > 10 ** (-72 / 20.0))[0]
        keep = min(total, (nz[-1] + int(0.15 * sr)) if len(nz) else int(0.5 * sr))
        master = master[:keep]
        stems = {k: v[:keep] for k, v in stems.items()}
    elapsed = time.perf_counter() - t_start
    dur = len(master) / sr
    report = {
        "duration_s": round(dur, 3),
        "sample_rate": sr,
        "start_seconds": round(start_sec, 3),
        "peak_db": _db(np.max(np.abs(master)) if master.size else 0.0),
        "rms_db": _db(float(np.sqrt(np.mean(master ** 2))) if master.size else 0.0),
        "render_time_s": round(elapsed, 3),
        "realtime_factor": round(dur / elapsed, 1) if elapsed > 0 else None,
        "tracks": track_reports,
        "warnings": [w for r in track_reports for w in r["warnings"]],
    }
    if progress:
        progress(1.0, "done")
    return RenderResult(master.astype(np.float32), sr, stems, report, start_sec)
