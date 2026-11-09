"""Render a project's backing arrangement into mix-ready stereo stems."""
from __future__ import annotations

import bisect
import json
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from . import dsp
from .arrangement import Compiled, Instrument, compile_arrangement
from .drums import GROUP_OF, drum_params, one_shot, velocity_gain
from .fx import apply_mute, process, render_fx
from .synth import Hit, hit_length, preset_params, render_hit

PRE_ROLL = 4.0  # s rendered before t0 so reverb/delay tails and long notes are right at the window start
DRUM_SEND = {"kick": 0.0, "snare": 1.0, "hats": 0.45, "cymbals": 0.6}
HUMANIZE_DEFAULTS = {"drums": {"timing_ms": 1.5, "velocity": 0.06}, "synth": {"timing_ms": 1.5, "velocity": 0.04},
                     "fx": {"timing_ms": 0.0, "velocity": 0.0}}


@dataclass
class BackingStem:
    """One rendered backing stem. ``audio`` is float32 (n, 2) covering [t0, t1) exactly.

    ``gain_db`` and ``pan`` are the instrument's mix defaults (not applied to ``audio``);
    ``sidechain`` asks the mixer to duck this stem under the kick (see :func:`kick_times`).
    """

    name: str
    role: str
    audio: np.ndarray
    gain_db: float = 0.0
    pan: float = 0.0
    sidechain: bool = False


_CACHE: "OrderedDict[str, Compiled]" = OrderedDict()
_CACHE_LOCK = threading.Lock()


def compiled_for(project) -> Compiled | None:
    """Compile (and cache) ``project.backing``. Raises ArrangementError when it is invalid."""
    arr = getattr(project, "backing", None)
    if not arr:
        return None
    key = json.dumps(arr, sort_keys=True, default=str) + "|" + str(project.key) + "|" + str(project.beats_per_bar())
    with _CACHE_LOCK:
        comp = _CACHE.get(key)
        if comp is not None:
            _CACHE.move_to_end(key)
            return comp
    comp = compile_arrangement(arr, project.key, project.beats_per_bar())
    with _CACHE_LOCK:
        _CACHE[key] = comp
        while len(_CACHE) > 8:
            _CACHE.popitem(last=False)
    return comp


def _rand2(*parts) -> tuple[float, float]:
    h = zlib.crc32("|".join(str(p) for p in parts).encode())
    return (h & 0xFFFF) / 65535.0, ((h >> 16) & 0xFFFF) / 65535.0


def _humanize(inst: Instrument) -> dict:
    d = dict(HUMANIZE_DEFAULTS[inst.kind])
    d.update(inst.humanize or {})
    return d


def _drum_hits(comp: Compiled, name: str, inst: Instrument, b2s) -> list[tuple]:
    """(time_s, voice, velocity, variant, cut_s | None) for every hit of a drum kit."""
    hz = _humanize(inst)
    tms, vj = hz.get("timing_ms", 0.0) / 1000.0, hz.get("velocity", 0.0)
    out = []
    for e in comp.drum_events.get(name, []):
        u1, u2 = _rand2(name, e.voice, f"{e.beat:.5f}")
        t = b2s(max(e.beat, 0.0)) + (u1 + u2 - 1.0) * tms * (0.5 if e.voice == "kick" else 1.0)
        v = max(0.05, min(1.5, e.velocity * (1.0 + (u2 - 0.5) * 2 * vj)))
        variant = int(u1 * 1000) % 4
        out.append([max(0.0, t), e.voice, v, variant, None])
    out.sort(key=lambda x: x[0])
    # open hats are choked by the next closed/open hat
    hat_times = [h[0] for h in out if h[1] in ("hat", "ohat")]
    for h in out:
        if h[1] == "ohat":
            i = bisect.bisect_right(hat_times, h[0] + 1e-4)
            if i < len(hat_times):
                h[4] = hat_times[i] - h[0]
    return [tuple(h) for h in out]


def kick_times(project) -> np.ndarray:
    """Seconds of every kick-drum hit in the backing (after stops and humanising), sorted."""
    comp = compiled_for(project)
    if comp is None:
        return np.zeros(0)
    b2s = project.tempo.beat_to_sec
    times = []
    for name, inst in comp.instruments.items():
        if inst.kind != "drums" or inst.mute:
            continue
        times += [h[0] for h in _drum_hits(comp, name, inst, b2s) if h[1] == "kick"]
    return np.array(sorted(times), dtype=np.float64)


def _synth_hits(comp: Compiled, name: str, inst: Instrument, p: dict, b2s) -> list[Hit]:
    hz = _humanize(inst)
    tms, vj = hz.get("timing_ms", 0.0) / 1000.0, hz.get("velocity", 0.0)
    gate_frac = float(p.get("gate", 0.85))
    hits = []
    for e in comp.note_events.get(name, []):
        u1, u2 = _rand2(name, f"{e.beat:.5f}", e.pitches)
        start = b2s(max(e.beat, 0.0))
        end = b2s(max(e.beat + e.beats * e.gate_scale, 0.0))
        t = max(0.0, start + (u1 + u2 - 1.0) * tms)
        v = max(0.05, min(1.5, e.velocity * (1.0 + (u2 - 0.5) * 2 * vj)))
        # the pattern's gate already shapes e.beats; the preset gate only trims sustained notes a little
        gate = max(0.012, (end - start) * (0.5 + 0.5 * gate_frac) if e.beats > 1.01 else (end - start))
        hits.append(Hit(t, gate, tuple(int(x) for x in e.pitches), v, bright=e.bright,
                        seed=zlib.crc32(f"{name}|{e.beat:.5f}|{e.pitches}".encode()), strum=e.strum))
    hits.sort(key=lambda h: h.start)
    if p.get("mono"):
        for a, b in zip(hits, hits[1:]):
            a.cut = max(0.005, b.start - a.start)
            if p.get("glide") and b.start - (a.start + a.gate) < 0.06:
                b.glide_from = float(a.pitches[-1])
    return hits


def _synth_params(inst: Instrument) -> dict:
    p = preset_params(inst.preset, inst.params)
    if inst.eq:
        eq = dict(p.get("eq") or {})
        for k, v in inst.eq.items():
            if k == "bands":
                eq["bands"] = list(eq.get("bands", [])) + list(v)
            elif k == "low_cut":
                eq[k] = max(v, eq.get(k, 0) or 0)
            elif k == "high_cut":
                eq[k] = min(v, eq.get(k) or v)
            else:
                eq[k] = v
        p["eq"] = eq
    return p


def _stop_samples(comp: Compiled, name: str, b2s, sr: int, S0: int) -> list[tuple[int, int]]:
    """Stop ranges of an instrument as buffer sample ranges, sorted."""
    return sorted((int(round(b2s(b0) * sr)) - S0, int(round(b2s(b1) * sr)) - S0) for b0, b1 in comp.stops_for(name))


def _cut_at_stop(y: np.ndarray, s: int, stops: list[tuple[int, int]], sr: int) -> np.ndarray:
    """Silence a sound placed at sample s from the first stop that starts after its onset (sudden stop:
    nothing rings through the stop or comes back after it)."""
    end = s + len(y)
    for a, _b in stops:
        if s < a < end:
            c = a - s
            f = min(c, int(0.004 * sr))
            y = y.copy() if not y.flags.writeable else y
            y[c:] = 0.0
            if f > 1:
                y[c - f:c] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
            break
    return y


def _segments(n: int, stops: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of [0, n) between stop ranges."""
    out, pos = [], 0
    for a, b in stops:
        if b <= 0 or a >= n:
            continue
        if a > pos:
            out.append((pos, min(a, n)))
        pos = max(pos, b)
    if pos < n:
        out.append((pos, n))
    return out


def _post(inst: Instrument, buf: np.ndarray, comp: Compiled, sr: int, S0: int, b2s, beat_s: float,
          send: float = 1.0, eq: bool = False) -> np.ndarray:
    if eq and inst.eq:
        curve = inst.eq
        buf = dsp.fft_filter(buf, sr, lambda f: dsp.eq_curve(f, curve))
    if inst.drive:
        buf = dsp.saturate(buf, inst.drive)
    if inst.width != 1.0:
        buf = dsp.set_width(buf, inst.width)
    if inst.delay or inst.reverb * send > 0:
        # echoes and reverb tails never ring into (or through) a stop: process each stretch between stops alone
        stops = _stop_samples(comp, inst.name, b2s, sr, S0)
        for a, b in _segments(len(buf), stops):
            seg = buf[a:b]
            if not np.any(seg):
                continue
            if inst.delay:
                seg = dsp.apply_delay(seg, sr, inst.delay["beats"] * beat_s, inst.delay["feedback"],
                                      inst.delay["mix"], inst.delay.get("pingpong", True))
            if inst.reverb * send > 0:
                seg = dsp.apply_reverb(seg, sr, inst.reverb * send, inst.reverb_decay)
            buf[a:b] = seg
    for pe in comp.processes_for(inst.name):
        a = int(round(b2s(pe.beat0) * sr)) - S0
        b = int(round(b2s(pe.beat1) * sr)) - S0
        if b <= 0 or a >= len(buf):
            continue
        process(pe.effect, buf, a, b, sr, beat_s, pe.params)
    for b0, b1 in comp.stops_for(inst.name):
        a = int(round(b2s(b0) * sr)) - S0
        b = int(round(b2s(b1) * sr)) - S0
        apply_mute(buf, a, b, sr)
    return buf


def render_backing(project, sample_rate: int, t0: float, t1: float) -> list[BackingStem]:
    """Render ``project.backing`` to stems covering [t0, t1) seconds (``round((t1-t0)*sr)`` samples each).

    Returns [] when the project has no backing. Stems for instruments that are silent in the window are
    omitted. Raises :class:`~synth2.arranger.arrangement.ArrangementError` for an invalid arrangement.
    """
    comp = compiled_for(project)
    if comp is None:
        return []
    sr = int(sample_rate)
    n = int(round((t1 - t0) * sr))
    if n <= 0:
        return []
    b2s = project.tempo.beat_to_sec
    beat_s = 60.0 / project.tempo.bpm_at(0.0)
    pre_s = PRE_ROLL
    for pe in comp.processes:  # effects that started before t0 need their source audio
        if b2s(pe.beat0) < t0 < b2s(pe.beat1):
            pre_s = max(pre_s, t0 - b2s(pe.beat0) + 0.1)
    pre = int(round(pre_s * sr))
    S0 = int(round(t0 * sr)) - pre
    N = pre + n
    def job(item):
        name, inst = item
        if inst.mute:
            return []
        if inst.kind == "drums":
            return _render_drums(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s)
        fn = _render_synth if inst.kind == "synth" else _render_fx
        st = fn(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s)
        return [st] if st is not None else []

    items = list(comp.instruments.items())
    workers = min(len(items), max(1, min(4, os.cpu_count() or 1)))
    if workers > 1 and n > sr * 20:
        # numpy releases the GIL in its array kernels and FFTs: instruments render in parallel
        with ThreadPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(job, items))
    else:
        results = [job(it) for it in items]
    return [st for group in results for st in group]


def _audible(audio: np.ndarray) -> bool:
    """False for a stem that is silent in the window (below -100 dBFS: only stray tails)."""
    return bool(len(audio)) and float(np.abs(audio).max()) > 1e-5


def _finish(buf: np.ndarray, pre: int, n: int) -> np.ndarray:
    out = np.ascontiguousarray(buf[pre:pre + n], dtype=np.float32)
    if out.shape[0] < n:
        out = np.concatenate([out, np.zeros((n - out.shape[0], 2), np.float32)])
    if not np.isfinite(out).all():
        out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return out


def _render_drums(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s) -> list[BackingStem]:
    hits = _drum_hits(comp, name, inst, b2s)
    stops = _stop_samples(comp, name, b2s, sr, S0)
    bufs: dict[str, np.ndarray] = {}
    for (t, voice, vel, variant, cut) in hits:
        s = int(round(t * sr)) - S0
        if s >= N:
            break
        params = drum_params(voice, inst.params.get(voice))
        shot = one_shot(voice, params, vel, variant, sr)
        if s + len(shot) <= 0:
            continue
        g = velocity_gain(vel) * float(params.get("level", 1.0))
        gl, gr = dsp.pan_gains(float(params.get("pan", 0.0)))
        if cut is not None and cut * sr < len(shot):
            c = max(8, int(cut * sr))
            y = shot[:c] * np.array([g * gl, g * gr], np.float32)
            f = min(c, int(0.006 * sr))
            y[c - f:] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
        else:
            y = shot * np.array([g * gl, g * gr], np.float32)
        y = _cut_at_stop(y, s, stops, sr)
        group = GROUP_OF[voice] if inst.split else "kit"
        if group not in bufs:
            bufs[group] = np.zeros((N, 2), np.float32)
        dsp.add_at(bufs[group], y, s)
    out = []
    order = ["kick", "snare", "hats", "cymbals", "kit"]
    for group in sorted(bufs, key=order.index):
        buf = _post(inst, bufs[group], comp, sr, S0, b2s, beat_s, DRUM_SEND.get(group, 0.6), eq=True)
        audio = _finish(buf, pre, n)
        if not _audible(audio):
            continue
        out.append(BackingStem(f"{name} {group}" if inst.split else name, inst.role, audio, inst.gain_db, inst.pan,
                               inst.sidechain))
    return out


def _render_synth(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s) -> BackingStem | None:
    p = _synth_params(inst)
    hits = _synth_hits(comp, name, inst, p, b2s)
    stops = _stop_samples(comp, name, b2s, sr, S0)
    buf = None
    cache: dict = {}
    vs = float(p.get("vel_sens", 0.7))
    for h in hits:
        s = int(round(h.start * sr)) - S0
        if s >= N:
            break
        length = hit_length(p, h.gate) if h.cut is None else min(hit_length(p, h.gate), h.cut + 0.004)
        if s + int(length * sr) + 1 <= 0:
            continue
        # identical hits (same pitches, gate, velocity bucket, brightness ...) are rendered once: patterns repeat
        vq = round(h.velocity * 20) / 20
        key = (h.pitches, int(round(h.gate * 400)), vq, round(h.bright, 2), round(h.strum, 4),
               None if h.glide_from is None else round(h.glide_from, 2),
               None if h.cut is None else int(round(h.cut * 400)))
        y = cache.get(key)
        if y is None:
            hq = Hit(0.0, round(h.gate * 400) / 400, h.pitches, vq, h.glide_from,
                     None if h.cut is None else round(h.cut * 400) / 400, round(h.bright, 2),
                     zlib.crc32(repr(key).encode()), round(h.strum, 4))
            y = render_hit(p, hq, sr)
            y.setflags(write=False)
            cache[key] = y
        scale = ((1 - vs) + vs * h.velocity) / max((1 - vs) + vs * vq, 1e-3)
        y = _cut_at_stop(y * np.float32(scale), s, stops, sr)
        if buf is None:
            buf = np.zeros((N, 2), np.float32)
        dsp.add_at(buf, y, s)
    if buf is None:
        return None
    buf = _post(inst, buf, comp, sr, S0, b2s, beat_s)
    audio = _finish(buf, pre, n)
    if not _audible(audio):
        return None
    return BackingStem(name, inst.role, audio, inst.gain_db, inst.pan, inst.sidechain)


def _render_fx(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s) -> BackingStem | None:
    buf = None
    stops = _stop_samples(comp, name, b2s, sr, S0)
    for e in comp.fx_events.get(name, []):
        t = b2s(max(e.beat, 0.0))
        s = int(round(t * sr)) - S0
        if s >= N:
            break
        dur = b2s(e.beat + e.beats) - t
        if s + int((dur + 3.0) * sr) <= 0:
            continue
        rng = np.random.default_rng(zlib.crc32(f"{name}|{e.sound}|{e.beat:.5f}".encode()))
        y = render_fx(e.sound, dur, e.pitch, sr, rng, e.params, beat_s)
        g = e.velocity * dsp.undb(e.gain_db)
        gl, gr = dsp.pan_gains(e.pan)
        y = _cut_at_stop(y * np.array([g * gl, g * gr], np.float32), s, stops, sr)
        if buf is None:
            buf = np.zeros((N, 2), np.float32)
        dsp.add_at(buf, y, s)
    if buf is None:
        return None
    buf = _post(inst, buf, comp, sr, S0, b2s, beat_s)
    audio = _finish(buf, pre, n)
    if not _audible(audio):
        return None
    return BackingStem(name, inst.role, audio, inst.gain_db, inst.pan, inst.sidechain)


def song_length_seconds(project) -> float:
    """End of the last section/event of the backing in seconds (0 without backing)."""
    comp = compiled_for(project)
    if comp is None:
        return 0.0
    return float(project.tempo.beat_to_sec(comp.end_beat))

