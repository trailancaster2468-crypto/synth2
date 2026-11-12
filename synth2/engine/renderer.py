"""Render orchestration: project -> per-track vocals -> mix -> master."""
from __future__ import annotations

import hashlib
import json
import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

from ..phonemes import english, japanese
from ..project import CURVE_SPECS, Project, Track, iter_active_tracks
from ..timing import midi_to_note
from ..voices import get_voice
from .backend import (backend_for_track, backend_for_voice, engine_signature, parse_quality,  # noqa: F401
                      register_backend, render_options)
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
    # the dry master), turned down together by report["stems_gain_db"] if one would clip. A mapping
    # key -> (n, 2) float32; a render returns a mixing.StemStore, which builds a stem's array when
    # it is read
    stems: Mapping[str, np.ndarray] = field(default_factory=dict)
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


def track_report(project: Project, track: Track, plan, voice, style, audio: np.ndarray,
                 note_range: tuple[float, float] | None = None, *, engine: str = "formant",
                 check_range: bool = True) -> dict:
    """The per-track render report every backend returns.

    It covers the notes in ``note_range`` (beats) only, all of them for a full render: planner
    warnings about those notes, notes outside the voice's range (``check_range``), English words
    whose pronunciation was guessed, and the phonemes each note was sung with. ``plan`` is the
    :func:`.planner.plan_track` result, ``voice`` and ``style`` the resolved voice and style, and
    ``audio`` the dry mono render (for the peak).
    """
    notes = track.notes if note_range is None else track.notes_in_range(*note_range)
    shown = {n.id for n in notes}
    all_ids = {n.id for n in track.notes}
    warnings = []
    for w in plan.warnings:  # keep warnings about shown notes, and those about no note at all
        ids = set(re.findall(r"[^\s,;:()'\"]+", w)) & all_ids
        if not ids or ids & shown:
            warnings.append(w)
    if check_range:
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
    return {
        "id": track.id, "name": track.name, "voice": voice.id, "style": style.name, "engine": engine,
        "notes": len(notes), "warnings": warnings,
        "phonemes": {k: v for k, v in plan.note_phonemes.items() if k in shown},
        "peak_db": _db(np.max(np.abs(audio)) if len(audio) else 0.0),
    }


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

        report = track_report(project, track, plan, voice, style, audio, note_range, engine=self.name)
        return audio.astype(np.float32), report


register_backend(FormantBackend())


def _db(x: float) -> float:
    return float(round(20 * np.log10(max(x, 1e-9)), 2))


def render_track(project: Project, track: Track | str | int, sample_rate: int | None = None,
                 tail: float = VOICE_TAIL, *, engine: str | None = None,
                 quality: str | None = None) -> tuple[np.ndarray, dict]:
    """Render one track dry (no pan / reverb), from beat 0 to the end of the track.

    ``engine`` (``auto`` / ``formant`` / ``neural`` / ``neural:<voicebank>``) overrides the
    track's and the project's engine; ``quality`` is ``draft`` / ``normal`` / ``high``.
    """
    if not isinstance(track, Track):
        track = project.track(track)
    sr = int(sample_rate or project.sample_rate)
    t1 = project.tempo.beat_to_sec(track.end) + tail
    backend = backend_for_track(project, track, engine)
    with render_options(quality=parse_quality(quality)):
        audio, rep = backend.render_track(project, track, sr, 0.0, t1)
    rep.setdefault("engine", backend.name)
    return audio, rep


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


SONG_LUFS = -11.0  # default integrated loudness of a song with a backing band
LEGACY_CEILING_DB = -0.3  # limiter ceiling (dBTP) of vocal-only renders, as before the mastering chain
BACKING_TAIL = 2.0  # s rendered after the backing's estimated end (cymbals, synth releases, its own reverb)


def _backing_api(project: Project) -> tuple[Callable | None, Callable | None, Callable | None, str | None]:
    """``(render_backing, kick_times, song_length_seconds, warning)`` from :mod:`synth2.arranger`,
    imported lazily. All None when the project has no backing or the arranger is not available
    (``song_length_seconds`` is optional)."""
    if not getattr(project, "backing", None):
        return None, None, None, None
    try:
        from ..arranger import kick_times, render_backing
    except ImportError as exc:
        return None, None, None, f"backing ignored: the arranger engine is not available ({exc})"
    try:
        from ..arranger import song_length_seconds
    except ImportError:
        song_length_seconds = None
    return render_backing, kick_times, song_length_seconds, None


def _rms(x: np.ndarray) -> float:
    if not x.size:
        return 0.0
    f = x.reshape(-1)
    return float(np.sqrt(np.einsum("i,i->", f, f, dtype=np.float64) / f.size))


def _num(v) -> float | None:
    return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) and np.isfinite(v) else None


def _backing_end_beat(project: Project, spec, kicks_beats: np.ndarray) -> float | None:
    """Best guess of where the backing arrangement ends (beats): explicit length keys of the spec,
    its sections' bar ranges (1-based, inclusive), or a bar after the last kick."""
    bpb = project.beats_per_bar()
    cands: list[float] = []
    if isinstance(spec, dict):
        for k in ("length_beats", "end_beat", "end"):
            if _num(spec.get(k)) is not None:
                cands.append(_num(spec[k]))
        for k in ("length_bars", "bars"):
            if _num(spec.get(k)) is not None:
                cands.append(_num(spec[k]) * bpb)
        for coll in ("sections", "parts", "tracks", "patterns"):
            items = spec.get(coll)
            items = items.values() if isinstance(items, dict) else items
            for sec in items if isinstance(items, (list, tuple, type({}.values()))) else []:
                if not isinstance(sec, dict):
                    continue
                for k in ("end_beat", "end"):
                    if _num(sec.get(k)) is not None:
                        cands.append(_num(sec[k]))
                bars = sec.get("bars")
                if isinstance(bars, (list, tuple)) and len(bars) == 2 and _num(bars[1]) is not None:
                    cands.append(_num(bars[1]) * bpb)
                elif isinstance(bars, str) and re.fullmatch(r"\s*\d+\s*-\s*\d+\s*", bars):
                    cands.append(float(bars.split("-")[1]) * bpb)
                if _num(sec.get("start_bar")) is not None and _num(sec.get("length_bars")) is not None:
                    cands.append((_num(sec["start_bar"]) - 1 + _num(sec["length_bars"])) * bpb)
    if len(kicks_beats):
        cands.append(float(np.max(kicks_beats)) + bpb)
    return max(cands) if cands else None


# level decisions of recent full song renders (band gain, compressor thresholds, master gain), so a
# section preview of the same mix (same master settings, backing and track mixer settings) is
# mastered like the full song instead of on its own
_LEVEL_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_LEVEL_KEYS = ("band_gain_db", "bus_threshold_db", "low_end_trim_db", "glue_threshold_db", "master_gain_db")


def _level_key(project: Project, sr: int, engine: str | None = None) -> str:
    mixer = [(t.id, t.voice, t.gain_db, t.pan, t.mute, t.solo, t.reverb_send) for t in project.tracks]
    blob = json.dumps({"master": project.master.to_dict(), "backing": getattr(project, "backing", None),
                       "mixer": mixer, "sr": sr, "key": project.key, "tempo": project.tempo.to_list(),
                       "engines": engine_signature(project, engine)},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def _overrides(mo: dict, *refs) -> dict:
    """Merged ``MasterSettings.mix_overrides`` entries for ``refs`` (later refs win); keys are
    matched exactly or case-insensitively."""
    out: dict = {}
    if not mo:
        return out
    low = {str(k).lower(): v for k, v in mo.items()}
    for r in refs:
        if r is None:
            continue
        v = mo.get(r) if r in mo else low.get(str(r).lower())
        if isinstance(v, dict):
            out.update(v)
    return out


def render_project(project: Project, *, sample_rate: int | None = None, track_ids=None,
                   start_beat: float | None = None, end_beat: float | None = None,
                   progress: Callable[[float, str], None] | None = None,
                   engine: str | None = None, quality: str | None = None) -> RenderResult:
    """Render, mix and master a whole project (or a beat range / subset of tracks).

    ``engine`` picks the singing engine for this render (``auto``, ``formant``, ``neural`` or
    ``neural:<voicebank>``) and overrides ``track.engine`` and ``project.engine``; see
    :func:`.backend.resolve_engine`. ``quality`` (``draft`` / ``normal`` / ``high``) trades render
    time for quality in engines that support it. The report lists the engine of every track, the
    ``engines`` used and, for the neural engine, the ``credits`` its models ask for.

    ``track_ids`` entries are track ids, names (case-insensitive) or indexes; an unknown entry
    raises :class:`LookupError`. A beat-range render sings the region the way the full render
    does: it is planned with the whole song around it, the reverb of earlier notes carries into
    it, and notes still sounding at ``end_beat`` fade out over :data:`REGION_FADE` seconds.

    Every vocal goes through the channel strip of :mod:`.mixing` (``master.vocal_chain``). When
    the project has a backing arrangement (``project.backing``, rendered by
    :mod:`synth2.arranger`; skipped for a ``track_ids`` subset or while a track is soloed), the
    band stems are mixed under the vocals and the master is glued, limited and brought to an
    integrated loudness (``master.loudness_target``, default :data:`SONG_LUFS`). Vocal-only
    projects keep the classic behaviour: a full render is peak-normalised to -1 dBFS (by at most
    :data:`NORMALIZE_MAX_GAIN_DB`), a region is not normalised on its own. ``master.gain_db``
    applies after the normalisation (or the loudness target), and the stems (vocal tracks by id,
    backing stems by name) get the same gains, so they sum to the dry part of the master.
    """
    with render_options(quality=parse_quality(quality)):
        return _render_project(project, sample_rate, track_ids, start_beat, end_beat, progress, engine)


def _render_project(project: Project, sample_rate, track_ids, start_beat, end_beat, progress,
                    engine) -> RenderResult:
    from . import mixing

    t_start = time.perf_counter()
    sr = int(sample_rate or project.sample_rate)
    ms = project.master
    tracks = _resolve_tracks(project, track_ids) if track_ids else list(iter_active_tracks(project))
    # resolve every engine first: a missing engine fails before any audio is rendered
    backends = {tr.id: backend_for_track(project, tr, engine) for tr in tracks if tr.notes}
    credits: list[str] = []
    b2s = project.tempo.beat_to_sec
    partial = start_beat is not None or end_beat is not None
    start_sec = b2s(start_beat) if start_beat is not None else 0.0
    warnings_extra: list[str] = []

    # ---- the backing band (full mix only: not for a track subset or while a track is soloed)
    render_backing = kick_fn = length_fn = None
    if not track_ids and not any(t.solo for t in project.tracks):
        render_backing, kick_fn, length_fn, warn = _backing_api(project)
        if warn:
            warnings_extra.append(warn)
    kicks_abs = np.zeros(0)
    backing_end_b = None
    if render_backing is not None:
        kicks_abs = np.sort(np.asarray(kick_fn(project) if kick_fn else [], dtype=np.float64).ravel())
        kicks_abs = kicks_abs[np.isfinite(kicks_abs)]
        length_s = _num(length_fn(project)) if length_fn is not None else None
        if length_s:
            backing_end_b = project.tempo.sec_to_beat(length_s)
        else:
            kb = np.array([project.tempo.sec_to_beat(float(t)) for t in kicks_abs[-1:]]) if len(kicks_abs) else np.zeros(0)
            backing_end_b = _backing_end_beat(project, project.backing, kb)

    vocal_end_b = max((t.end for t in tracks), default=0.0)
    end_b = end_beat if end_beat is not None else max(vocal_end_b, backing_end_b or 0.0)
    end_sec = max(b2s(end_b), start_sec + 0.1)
    room = project.master.reverb_room
    band_sends = render_backing is not None and ms.band_reverb > 0
    reverb_tail = 0.3 + 2.2 * room if (any(t.reverb_send > 0 for t in tracks) or band_sends) else 0.0
    if ms.vocal_delay > 0 and any(t.notes for t in tracks):
        reverb_tail = max(reverb_tail, 1.5)
    # a region that ends mid-phrase fades out right after end_beat instead of singing on
    voice_end = end_sec + (REGION_FADE if end_beat is not None else VOICE_TAIL)
    tail = reverb_tail
    if render_backing is not None and end_beat is None:
        tail = max(tail, BACKING_TAIL)
    total = int(round((voice_end - start_sec + tail) * sr))
    n_voice = int(round((voice_end - start_sec) * sr))
    # a region starting later is rendered from earlier on (pre-roll) so the reverb of the notes before
    # it carries into it; the pre-roll is dropped after the reverb
    render0 = min(start_sec, max(0.0, start_sec - reverb_tail))
    n_pre = int(round(start_sec * sr)) - int(round(render0 * sr))
    n_all = n_pre + total
    n_fade = min(int(round(REGION_FADE * sr)), n_voice)
    fade_out = 0.5 + 0.5 * np.cos(np.linspace(0.0, np.pi, n_fade))

    sources: list[mixing.Source] = []
    track_reports = []
    note_range = (start_beat if start_beat is not None else -1e9, end_b) if partial else None
    last_end = start_sec  # end (s) of the last note that starts before end_beat
    mo = ms.mix_overrides or {}
    for k, tr in enumerate(tracks):
        if progress:
            progress(0.8 * k / max(len(tracks), 1), f"rendering {tr.name}")
        if not tr.notes:
            continue
        backend = backends[tr.id]
        mono, rep = backend.render_track(project, tr, sr, render0, voice_end, note_range)
        rep.setdefault("engine", backend.name)
        for c in rep.get("credits") or []:
            if c not in credits:
                credits.append(c)
        mono = np.asarray(mono, dtype=np.float32)[: n_pre + n_voice]
        if end_beat is not None and n_fade:
            a = n_pre + n_voice - n_fade
            mono = mono.copy()
            mono[a:a + n_fade] = mono[a:a + n_fade] * fade_out[: max(0, len(mono) - a)]
        low = min(n.pitch + float(n.attrs.get("detune", 0) or 0) / 100 for n in tr.notes)
        neural = rep.get("neural") if isinstance(rep.get("neural"), dict) else {}
        sources.append(mixing.Source(
            key=tr.id, name=tr.name, kind="vocal", audio=mixing.Sparse.from_dense(mono, length=n_all),
            gain_db=tr.gain_db, pan=tr.pan, reverb_send=tr.reverb_send, delay_send=1.0,
            lowest_hz=440.0 * 2 ** ((low - 69) / 12.0), overrides=_overrides(mo, tr.id, tr.name),
            engine=str(rep["engine"]), vocal_role=str(getattr(tr, "role", "") or ""),
            dial=str(neural.get("dial") or ""), copy_of=neural.get("copy_of") or None))
        del mono
        rep.update({"gain_db": tr.gain_db, "pan": tr.pan})
        track_reports.append(rep)
        ends = [n.end for n in tr.notes if n.onset < end_b]
        if ends:
            last_end = max(last_end, b2s(max(ends)))

    backing_reports = []
    if render_backing is not None:
        if progress:
            progress(0.8, "rendering backing")
        t1 = render0 + n_all / sr
        used = {s.key for s in sources}
        band = list(render_backing(project, sr, render0, t1) or [])
        band.reverse()
        while band:  # each stem is let go once the mixer's copy is made
            st = band.pop()
            audio = np.asarray(getattr(st, "audio"), dtype=np.float32)
            if audio.ndim == 1:
                audio = np.stack([audio, audio], axis=1)
            elif audio.shape[1] != 2:
                audio = audio[:, :2] if audio.shape[1] > 2 else np.repeat(audio[:, :1], 2, axis=1)
            audio = audio[:n_all]
            if end_beat is not None:  # the band stops with the region, like the vocals
                a = n_pre + n_voice - n_fade
                audio = audio[: n_pre + n_voice].copy()
                if n_fade:
                    audio[a:a + n_fade] = audio[a:a + n_fade] * fade_out[:, None][: max(0, len(audio) - a)]
            name = str(getattr(st, "name", "") or "backing")
            role = str(getattr(st, "role", "keys") or "keys")
            key, j = name, 2
            while key in used:
                key, j = f"{name}_{j}", j + 1
            used.add(key)
            src = mixing.Source(
                key=key, name=name, kind="backing", audio=mixing.Sparse.from_dense(audio, length=n_all), role=role,
                gain_db=float(getattr(st, "gain_db", 0.0) or 0.0), pan=float(getattr(st, "pan", 0.0) or 0.0),
                sidechain=bool(getattr(st, "sidechain", False)), overrides=_overrides(mo, role, name))
            sources.append(src)
            entry = {"name": key, "role": role, "gain_db": src.gain_db, "pan": src.pan, "sidechain": src.sidechain}
            if getattr(st, "sound", ""):
                entry["sound"] = str(st.sound)
            backing_reports.append(entry)
            for w in getattr(st, "warnings", None) or ():
                if w not in warnings_extra:
                    warnings_extra.append(str(w))
            for c in getattr(st, "credits", None) or ():
                if c not in credits:
                    credits.append(str(c))
            del st, audio
        if backing_end_b is not None and end_beat is None:
            last_end = max(last_end, b2s(backing_end_b))
    band_mode = any(s.kind == "backing" for s in sources)

    if progress:
        progress(0.85, "mixing")
    t_mix = time.perf_counter()
    bpm = project.tempo.bpm_at(start_beat or 0.0) if hasattr(project.tempo, "bpm_at") else project.bpm
    mastered = band_mode or ms.loudness_target is not None
    level_key = _level_key(project, sr, engine) if mastered else None
    fixed = _LEVEL_CACHE.get(level_key) if (mastered and partial) else None
    mix = mixing.mix_sources(sources, sr, band_mode=band_mode, settings=ms, kicks=kicks_abs - render0, bpm=bpm,
                             reverb_room=room, reverb_wet=ms.reverb_wet if reverb_tail > 0 else 0.0, fixed=fixed,
                             consume=True)
    stems = mix.stems
    master = (mix.dry + mix.wet)[n_pre:]
    stems.crop(n_pre)
    del sources
    if start_beat is not None and start_sec > 0:  # the region may start inside a note: no click
        n_in = min(int(round(REGION_FADE_IN * sr)), len(master))
        ramp = np.linspace(0.0, 1.0, n_in, endpoint=False)[:, None]
        master[:n_in] *= ramp
        stems.scale(np.concatenate([ramp[:, 0], np.ones(len(master) - n_in)]).astype(np.float32))

    if progress:
        progress(0.92, "mastering")
    mix_info = dict(mix.info)
    norm_db = 0.0
    if mastered:
        target = (ms.loudness_target if ms.loudness_target is not None else SONG_LUFS) if ms.normalize else None
        master, stem_gain, minfo = mixing.master_bus(
            master, np.zeros_like(master), sr, glue=ms.glue if band_mode else 0.0, target_lufs=target,
            ceiling_db=ms.ceiling_db, limiter=ms.limiter, gain_db=ms.gain_db, fixed=fixed)
        mix_info.update(minfo)
        levels = {k: mix_info.get(k) for k in _LEVEL_KEYS}
        if not partial and level_key:  # remember the full song's levels for section previews
            _LEVEL_CACHE[level_key] = levels
            _LEVEL_CACHE.move_to_end(level_key)
            while len(_LEVEL_CACHE) > 8:
                _LEVEL_CACHE.popitem(last=False)
        mix_info["levels_from_full_render"] = fixed is not None
        for k in _LEVEL_KEYS:
            mix_info[k] = mixing._r(mix_info.get(k))
        stems.scale(stem_gain.astype(np.float32))
        norm_db = minfo.get("master_gain_db") or 0.0
    else:
        # normalisation sets the reference level (full renders only, with a gain limit), then the master
        # gain trims it; the stems get the same linear gain so they sum to the dry master
        peak = float(np.max(np.abs(master))) if master.size else 0.0
        gain = 1.0
        if ms.normalize and not partial and peak > 0:
            gain = min(10 ** (NORMALIZE_PEAK_DB / 20.0) / peak, 10 ** (NORMALIZE_MAX_GAIN_DB / 20.0))
        norm_db = 20 * np.log10(gain)
        gain *= 10 ** (ms.gain_db / 20.0)
        master = master * gain
        stems.scale(gain)
        if ms.limiter:
            master = mixing.limit(master, sr, LEGACY_CEILING_DB)
    # stems are not limited: if one would clip, all of them are turned down by the same amount
    stem_peak = stems.peak()
    stem_gain = 1.0 / stem_peak if stem_peak > 1.0 else 1.0
    if stem_gain != 1.0:
        stems.scale(stem_gain)
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
        stems.crop(0, keep)
    master = master.astype(np.float32)

    # ---- metering
    loud = mixing.loudness_stats(master, sr) if master.size else {}
    tp = mix_info.pop("true_peak", None)
    if tp is None:
        tp = mixing.true_peak(master) if master.size else 0.0
    names = {r["id"]: r["name"] for r in track_reports}
    stem_levels = []
    for k in stems:
        stem_levels.append({"key": k, "name": names.get(k, k), "group": mix.groups.get(k, "vocal"),
                            "peak_db": _db(stems.peak(k)), "rms_db": _db(stems.rms(k))})
    elapsed = time.perf_counter() - t_start
    dur = len(master) / sr
    report = {
        "duration_s": round(dur, 3),
        "sample_rate": sr,
        "start_seconds": round(start_sec, 3),
        "peak_db": _db(np.max(np.abs(master)) if master.size else 0.0),
        "rms_db": _db(_rms(master)),
        "true_peak_db": _db(tp),
        "loudness_lufs": mixing._r(loud.get("integrated")),
        "short_term_max_lufs": mixing._r(loud.get("short_term_max")),
        "loudness_range_lu": mixing._r(loud.get("range")),
        "vocal_to_band_db": mix_info.get("vocal_to_band_db"),
        "normalize_gain_db": round(float(norm_db), 2),
        "stems_gain_db": round(float(20 * np.log10(stem_gain)), 2),
        "stem_levels": stem_levels,
        "mix": {"mode": "song" if band_mode else ("mastered" if mastered else "vocal"), **mix_info},
        "render_time_s": round(elapsed, 3),
        "mix_time_s": round(time.perf_counter() - t_mix, 3),
        "realtime_factor": round(dur / elapsed, 1) if elapsed > 0 else None,
        "engines": sorted({r["engine"] for r in track_reports}),
        "tracks": track_reports,
        "backing": backing_reports,
        "warnings": [w for r in track_reports for w in r["warnings"]] + warnings_extra,
    }
    if credits:
        report["credits"] = credits
    if progress:
        progress(1.0, "done")
    return RenderResult(master, sr, stems, report, start_sec)
