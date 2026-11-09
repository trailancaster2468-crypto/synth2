"""Render orchestration: project -> per-track vocals -> mix -> master."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from ..phonemes import english, japanese
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
REGION_FADE = 0.05  # s: a beat-range render fades out whatever still sounds at end_beat over this time
REGION_FADE_IN = 0.005  # s: and fades in at start_beat, which may cut into a sounding note
PRE_ROLL = 0.1  # s: frames synthesised before a partial window (smoothing, open phases, pulse overlap)
POST_ROLL = 0.05  # s: and after it
NORMALIZE_PEAK_DB = -1.0  # normalisation target for the master peak
NORMALIZE_MAX_GAIN_DB = 20.0  # normalisation never raises the level more than this (near-silence, breaths)
TRIM_BELOW_PEAK_DB = 72.0  # the silent tail is trimmed where the master stays this far below its peak


def frame_hop(sr: int) -> int:
    return max(32, int(round(sr * 0.0029)))


@dataclass
class RenderResult:
    audio: np.ndarray
    sample_rate: int
    # per track: post-fader and pre-reverb, with the master's normalisation and gain (so they sum to
    # the dry master), turned down together by report["stems_gain_db"] if one would clip
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
        used: set[str] = set()
        for tid, audio in self.stems.items():
            out.append(write_wav(d / f"{_stem_name(names.get(tid, tid), tid, used)}.wav", audio,
                                 self.sample_rate, bits))
        return out

    def mono(self) -> np.ndarray:
        return self.audio.mean(axis=1)


def _stem_name(name: str, tid: str, used: set[str]) -> str:
    """A file name for a stem: the track name with unsafe characters replaced (Unicode letters
    are kept), the track id when nothing is left, and a numeric suffix when the name is taken
    (compared case-insensitively, as on macOS and Windows file systems). ``used`` is updated."""
    safe = re.sub(r"[^\w.-]+", "_", str(name)).strip("._")
    if not re.search(r"[^\W_]", safe):
        safe = re.sub(r"[^\w.-]+", "_", str(tid)).strip("._") or "track"
    base, k = safe, 2
    while safe.lower() in used:
        safe = f"{base}_{k}"
        k += 1
    used.add(safe.lower())
    return safe


class FormantBackend:
    """The built-in parametric singing engine."""

    name = "formant"

    def voices(self) -> list[str]:
        from ..voices import VOICES

        return list(VOICES)

    def render_track(self, project: Project, track: Track, sample_rate: int, t0: float, t1: float,
                     note_range: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
        """Render ``track`` dry over ``[t0, t1)`` seconds.

        The whole track is always planned, so a partial window keeps the full render's phrase
        structure (phrase starts and ends, arcs, tapers), word heads of ``+`` / ``-`` notes and
        random draws. Its frames lie on the full render's frame grid, with some pre- and post-roll
        so the edges of the window match the full render. ``note_range`` (beats) only limits what
        the report lists (notes, phonemes, warnings) to the notes that overlap it.
        """
        sr = sample_rate
        voice = get_voice(track.voice).with_params(track.voice_params)
        overrides = {k[6:]: v for k, v in track.voice_params.items() if k.startswith("style.")}
        style = get_style(track.style or voice.default_style, overrides)
        rng = np.random.default_rng(track.seed)
        plan = plan_track(track.notes, project.tempo, track.language, style, rng)

        hop = frame_hop(sr)
        fp = hop / sr
        k0 = 0
        if note_range is None:
            n_frames = max(2, int(np.ceil((t1 - t0) / fp)) + 1)
            t = t0 + np.arange(n_frames) * fp
        else:
            k0 = int(np.floor(min(t0, max(0.0, t0 - PRE_ROLL)) / fp))
            n_frames = max(2, int(np.ceil((t1 + POST_ROLL) / fp)) - k0 + 1)
            t = (k0 + np.arange(n_frames)) * fp
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
        # the vocoder's sample 0 is frame k0 (absolute sample k0 * hop): crop to [t0, t1)
        off = 0 if note_range is None else int(round(t0 * sr)) - k0 * hop
        n_out = int(round((t1 - t0) * sr))
        audio = audio[off:off + n_out]
        if len(audio) < n_out:
            audio = np.pad(audio, (0, n_out - len(audio)))

        # the report covers the notes in the requested range only (all of them for a full render)
        notes = track.notes if note_range is None else track.notes_in_range(*note_range)
        shown = {n.id for n in notes}
        all_ids = {n.id for n in track.notes}
        warnings = []
        for w in plan.warnings:  # keep warnings about shown notes, and those about no note at all
            ids = set(re.findall(r"[^\s,;:()'\"]+", w)) & all_ids
            if not ids or ids & shown:
                warnings.append(w)
        for n in notes:
            if n.pitch < voice.range_low - 2 or n.pitch > voice.range_high + 2:
                warnings.append(
                    f"note {n.id} ({midi_to_note(n.pitch)} at beat {n.onset:g}) is outside {voice.name}'s "
                    f"range {midi_to_note(voice.range_low)}-{midi_to_note(voice.range_high)}")
        guessed = sorted({
            w for n in notes for w in re.findall(r"[A-Za-z']+", n.lyric or "")
            if (n.attrs.get("language") or track.language) == "en" and not japanese.is_kana(n.lyric or "")
            and not n.phonemes and not (n.lyric or "").startswith((".", "[")) and w.lower() not in ("br",)
            and english.lookup(w) is None and len(w) > 1
        })
        if guessed:
            warnings.append("pronunciation guessed (not in dictionary): " + ", ".join(guessed[:20])
                            + " - use '+' continuation notes for split words or set explicit phonemes")
        if plan.start < 0 and (note_range is None or note_range[0] <= 0):
            warnings.append("the first consonants start before beat 0 and are cut; start the song at beat 1 or later")
        report = {
            "id": track.id, "name": track.name, "voice": voice.id, "style": style.name,
            "notes": len(notes), "warnings": warnings,
            "phonemes": {k: v for k, v in plan.note_phonemes.items() if k in shown},
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


def _resolve_tracks(project: Project, track_ids) -> list[Track]:
    """The tracks named by ``track_ids`` (ids, case-insensitive names or indexes, negative ones
    included), in project order. Raises :class:`LookupError` for an entry that matches no track."""
    if isinstance(track_ids, (str, int, np.integer)):
        track_ids = [track_ids]
    picked: set[int] = set()
    for ref in track_ids:
        try:
            if isinstance(ref, (int, np.integer)) and not isinstance(ref, bool):
                tr = project.tracks[int(ref)]
            else:
                key = str(ref).strip()
                try:
                    tr = project.track(key)
                except LookupError:
                    if not re.fullmatch(r"-\d+", key):
                        raise
                    tr = project.tracks[int(key)]
        except LookupError:  # includes IndexError
            names = ", ".join(f"{i}: {t.name!r} ({t.id})" for i, t in enumerate(project.tracks)) or "none"
            raise LookupError(f"no track {ref!r} to render; tracks are {names}") from None
        picked.add(id(tr))
    return [t for t in project.tracks if id(t) in picked]


def render_project(project: Project, *, sample_rate: int | None = None, track_ids=None,
                   start_beat: float | None = None, end_beat: float | None = None,
                   progress: Callable[[float, str], None] | None = None) -> RenderResult:
    """Render and mix a whole project (or a beat range / subset of tracks).

    ``track_ids`` entries are track ids, names (case-insensitive) or indexes; an unknown entry
    raises :class:`LookupError`. A beat-range render sings the region the way the full render
    does: it is planned with the whole song around it, the reverb of earlier notes carries into
    it, and notes still sounding at ``end_beat`` fade out over :data:`REGION_FADE` seconds. It is
    not normalised on its own (that would play quiet passages at full scale); only a full render
    is peak-normalised, by at most :data:`NORMALIZE_MAX_GAIN_DB`. The master gain applies after
    normalisation, and the stems get the same gain, so they sum to the dry master.
    """
    t_start = time.perf_counter()
    sr = int(sample_rate or project.sample_rate)
    tracks = _resolve_tracks(project, track_ids) if track_ids else list(iter_active_tracks(project))
    b2s = project.tempo.beat_to_sec
    partial = start_beat is not None or end_beat is not None
    start_sec = b2s(start_beat) if start_beat is not None else 0.0
    end_b = end_beat if end_beat is not None else max((t.end for t in tracks), default=0.0)
    end_sec = max(b2s(end_b), start_sec + 0.1)
    room = project.master.reverb_room
    reverb_tail = 0.3 + 2.2 * room if any(t.reverb_send > 0 for t in tracks) else 0.0
    # a region that ends mid-phrase fades out right after end_beat instead of singing on
    voice_end = end_sec + (REGION_FADE if end_beat is not None else VOICE_TAIL)
    total = int(round((voice_end - start_sec + reverb_tail) * sr))
    n_voice = int(round((voice_end - start_sec) * sr))
    # a region starting later is rendered from earlier on (pre-roll) so the reverb of the notes before
    # it carries into it; the pre-roll is dropped after the reverb
    render0 = min(start_sec, max(0.0, start_sec - reverb_tail))
    n_pre = int(round(start_sec * sr)) - int(round(render0 * sr))
    n_fade = min(int(round(REGION_FADE * sr)), n_voice)
    fade_out = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, n_fade))

    dry = np.zeros((n_pre + total, 2))
    bus = np.zeros((n_pre + total, 2))
    stems: dict[str, np.ndarray] = {}
    track_reports = []
    note_range = (start_beat if start_beat is not None else -1e9, end_b) if partial else None
    last_end = start_sec  # end (s) of the last note that starts before end_beat
    for k, tr in enumerate(tracks):
        if progress:
            progress(k / max(len(tracks), 1), f"rendering {tr.name}")
        if not tr.notes:
            continue
        mono, rep = backend_for_voice(tr.voice).render_track(project, tr, sr, render0, voice_end, note_range)
        mono = np.asarray(mono, dtype=np.float64)[: n_pre + n_voice]
        if end_beat is not None and n_fade:
            a = n_pre + n_voice - n_fade
            mono = mono.copy()
            mono[a:a + n_fade] *= fade_out[: max(0, len(mono) - a)]
        st = effects.pan_stereo(mono, tr.pan, tr.gain_db)
        stem = np.zeros((n_pre + total, 2))
        stem[: len(st)] = st
        stems[tr.id] = stem
        dry += stem
        if tr.reverb_send > 0:
            bus += stem * tr.reverb_send
        rep.update({"gain_db": tr.gain_db, "pan": tr.pan})
        track_reports.append(rep)
        ends = [n.end for n in tr.notes if n.onset < end_b]
        if ends:
            last_end = max(last_end, b2s(max(ends)))
    if progress:
        progress(0.9, "mixing")
    master = dry
    if reverb_tail > 0 and project.master.reverb_wet > 0 and np.any(bus):
        wet = effects.apply_reverb(bus, sr, room)[: len(dry)] * project.master.reverb_wet
        master = master + wet
    master = master[n_pre:]
    stems = {k: v[n_pre:] for k, v in stems.items()}
    if start_beat is not None and start_sec > 0:  # the region may start inside a note: no click
        n_in = min(int(round(REGION_FADE_IN * sr)), len(master))
        ramp = np.linspace(0.0, 1.0, n_in, endpoint=False)[:, None]
        for x in (master, *stems.values()):
            x[:n_in] *= ramp
    # normalisation sets the reference level (full renders only, with a gain limit), then the master
    # gain trims it; the stems get the same linear gain so they sum to the dry master
    peak = float(np.max(np.abs(master))) if master.size else 0.0
    gain = 1.0
    if project.master.normalize and not partial and peak > 0:
        gain = min(10 ** (NORMALIZE_PEAK_DB / 20.0) / peak, 10 ** (NORMALIZE_MAX_GAIN_DB / 20.0))
    norm_db = 20 * np.log10(gain)
    gain *= 10 ** (project.master.gain_db / 20.0)
    master = master * gain
    stems = {k: v * gain for k, v in stems.items()}
    if project.master.limiter:
        master = effects.limiter(master, sr, -0.3)
    # stems are not limited: if one would clip, all of them are turned down by the same amount
    stem_peak = max((float(np.max(np.abs(v))) for v in stems.values() if v.size), default=0.0)
    stem_gain = 1.0 / stem_peak if stem_peak > 1.0 else 1.0
    if stem_gain != 1.0:
        stems = {k: v * stem_gain for k, v in stems.items()}
    # trim the silent tail (relative to the peak, so a quiet mix keeps its reverb), but never before
    # the release of the last note
    if master.size:
        lvl = np.max(np.abs(master), axis=1)
        pk = float(lvl.max())
        nz = np.nonzero(lvl > pk * 10 ** (-TRIM_BELOW_PEAK_DB / 20.0))[0] if pk > 0 else []
        keep = (nz[-1] + int(0.15 * sr)) if len(nz) else int(0.5 * sr)
        floor = int(round((min(last_end + 0.1, voice_end) - start_sec) * sr))
        keep = min(len(master), max(keep, floor))
        master = master[:keep]
        stems = {k: v[:keep] for k, v in stems.items()}
    stems = {k: v.astype(np.float32) for k, v in stems.items()}
    elapsed = time.perf_counter() - t_start
    dur = len(master) / sr
    report = {
        "duration_s": round(dur, 3),
        "sample_rate": sr,
        "start_seconds": round(start_sec, 3),
        "peak_db": _db(np.max(np.abs(master)) if master.size else 0.0),
        "rms_db": _db(float(np.sqrt(np.mean(master ** 2))) if master.size else 0.0),
        "normalize_gain_db": round(float(norm_db), 2),
        "stems_gain_db": round(float(20 * np.log10(stem_gain)), 2),
        "render_time_s": round(elapsed, 3),
        "realtime_factor": round(dur / elapsed, 1) if elapsed > 0 else None,
        "tracks": track_reports,
        "warnings": [w for r in track_reports for w in r["warnings"]],
    }
    if progress:
        progress(1.0, "done")
    return RenderResult(master.astype(np.float32), sr, stems, report, start_sec)
