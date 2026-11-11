"""Render a project's backing arrangement into mix-ready stereo stems."""
from __future__ import annotations

import bisect
import json
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np

from . import dsp, gm, soundbank
from .arrangement import Compiled, Instrument, compile_arrangement
from .drums import GROUP_OF, drum_params, one_shot, synth_voice, velocity_gain
from .fx import apply_mute, process, render_fx
from .sampler import loudness, midi_velocity, note_groups, render_note
from .synth import Hit, hit_length, preset_params, render_hit

PRE_ROLL = 4.0  # s rendered before t0 so reverb/delay tails and long notes are right at the window start
DRUM_SEND = {"kick": 0.0, "snare": 1.0, "hats": 0.45, "cymbals": 0.6}
# reverb send of the stems of a kit split per kind of drum (others 0.6)
DRUM_SEND_EACH = {"kick": 0.0, "snare": 1.0, "clap": 1.0, "rim": 1.0, "snap": 1.0, "toms": 1.0, "hat": 0.45,
                  "ohat": 0.45, "shaker": 0.45, "tamb": 0.45, "cowbell": 0.45, "crash": 0.6, "ride": 0.6}
SAMPLED_TAIL = 12.0  # s a sampled note may ring after its key is released (for notes before the render window)
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
    warnings: tuple = ()  # what did not go as the arrangement asked (a sound that is not installed ...)
    credits: tuple = ()  # credit lines of the sound sets this stem plays
    sound: str = ""  # the sampled sound this stem plays ("" = synthesised)


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
        times += [h[0] for h in _drum_hits(comp, name, inst, b2s) if h[1] in ("kick", "kick2")]
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
    dirs = soundbank.project_dirs(project, project.backing)

    def job(item):
        name, inst = item
        if inst.mute:
            return []
        if inst.kind == "drums":
            return _render_drums(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s, dirs)
        if inst.kind == "synth" and inst.sound:
            st = _render_sampled(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s, dirs)
            return [st] if st is not None else []
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


def _kit_voice(kit: soundbank.Resolved, inst: Instrument, voice: str, dirs: list[str]):
    """(instrument, key) that plays a voice of a sampled kit with the voice's own options
    (tune, pan, gain_db, another sample, another key), or None when the kit has no such drum."""
    opts = inst.voice_sound.get(voice) or {}
    ck = (voice, json.dumps([opts, inst.params.get(voice)], sort_keys=True))
    if ck in kit.voices:
        return kit.voices[ck]
    out = None
    base = kit.instrument
    key = None
    normalize = kit.options.get("normalize", True)
    gate = float(kit.options.get("gate") or soundbank.DRUM_GATE)
    if opts.get("sample"):
        res = soundbank.resolve({"path": opts["sample"], "normalize": False}, drums=True, dirs=dirs)
        if res.instrument is not None and res.instrument.keys():
            base, key = res.instrument, res.instrument.keys()[0]
            normalize = True  # a drum from another file has no place in the kit's own balance
        else:
            msg = f"{voice}: {res.error or 'no playable sample'}; the kit's own drum plays"
            if msg not in kit.warnings:
                kit.warnings.append(msg)
    if key is None:
        keys = set(base.keys())
        want = [int(opts["note"])] if opts.get("note") is not None else list(gm.drum_keys(voice))
        key = next((k for k in want if k in keys), None)
    if key is not None:
        regs = [r for r in base.regions if r.lokey <= key <= r.hikey]
        tw = {}
        if opts.get("tune"):
            tw["tune"] = 100.0 * float(opts["tune"])
        if tw:
            regs = [soundbank.tweak_region(r, tw) if r.resolve is None else
                    replace(soundbank.tweak_region(r, tw), resolve=soundbank._tweaked_resolver(r.resolve, tw))
                    for r in regs]
        if opts.get("pan") is not None:
            regs = [replace(r, pan=float(opts["pan"])) for r in regs]
        vinst = base.with_regions(regs, gain=1.0)
        gain = base.gain
        if normalize:  # as loud as the synthesised drum it replaces (which the mix was balanced with)
            sv, sp = synth_voice(voice, inst.params.get(voice))
            gain = soundbank.match_voice_level(vinst, key, sv, sp, gate) * 10.0 ** (
                float(kit.options.get("gain_db") or 0.0) / 20.0)
        gain *= 10.0 ** (float(opts.get("gain_db") or 0.0) / 20.0)
        out = (vinst.with_regions(regs, gain=gain), key)
    kit.voices[ck] = out
    return out


def _stem_of(inst: Instrument, voice: str) -> str:
    if inst.stems == "kit":
        return "kit"
    if inst.stems == "each":
        return gm.stem_of(voice)
    return GROUP_OF[gm.synth_voice(voice)[0]]


def _render_drums(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s, dirs=()) -> list[BackingStem]:
    hits = _drum_hits(comp, name, inst, b2s)
    stops = _stop_samples(comp, name, b2s, sr, S0)
    kit, notes, label, credits = None, [], "", ()
    if inst.sound:
        res = soundbank.resolve(inst.sound[0], drums=True, kit=inst.kit, kit_params=inst.params, dirs=list(dirs))
        if res.instrument is None:
            notes.append(f"{name}: {res.error}; the synthesised kit plays instead")
        else:
            kit, label, credits = res, res.label, tuple(res.credits)
    plan: list = [None] * len(hits)
    if kit is not None:
        # which regions every hit of the song plays (round robin counts hits per voice from the start), and
        # where the kit's own choke groups cut them
        counts: dict[str, int] = {}
        in_group: dict[int, list[float]] = {}
        gate = float(inst.sound[0].get("gate") or soundbank.DRUM_GATE)
        for i, (t, voice, vel, _variant, _cut) in enumerate(hits):
            seq = counts.get(voice, 0)
            counts[voice] = seq + 1
            kv = _kit_voice(kit, inst, voice, list(dirs))
            if kv is None:
                continue
            vq = float(min(127, max(1, int(round(midi_velocity(vel) / 2.0)) * 2)))
            rand = _rand2(name, voice, f"{t:.5f}", "layer")[0] * 0.999
            regs = kv[0].select(kv[1], int(vq), seq, rand)
            if not regs:
                continue
            groups, off_by = note_groups(regs)
            for g in groups:
                in_group.setdefault(g, []).append(t)
            plan[i] = (kv[0], kv[1], vq, seq, rand, regs, off_by)
        missing = sorted({h[1] for h, pl in zip(hits, plan) if pl is None})
        if missing:
            notes.append(f"{name}: {label} has no {', '.join(missing)}; synthesised")
    bufs: dict[str, np.ndarray] = {}
    cache: dict = {}
    for i, (t, voice, vel, variant, cut) in enumerate(hits):
        s = int(round(t * sr)) - S0
        if s >= N:
            break
        if plan[i] is not None:
            vinst, key, vq, seq, rand, regs, off_by = plan[i]
            for g in off_by:  # choked by the next hit of the group
                times = in_group.get(g, ())
                j = bisect.bisect_right(times, t + 1e-4)
                if j < len(times):
                    cut = times[j] - t if cut is None else min(cut, times[j] - t)
            length = (inst.voice_sound.get(voice) or {}).get("length")
            if length:
                cut = float(length) if cut is None else min(cut, float(length))
            ck = (voice, tuple(id(r) for r in regs), vq, None if cut is None else int(round(cut * 1000)))
            y = cache.get(ck)
            if y is None:
                y = render_note(vinst, key, vq, gate, sr, seq=seq, rand=rand, cut=cut, regions=regs)
                y.setflags(write=False)
                cache[ck] = y
            if s + len(y) <= 0:
                continue
            g = (midi_velocity(vel) / vq) ** 2
            y = y * np.float32(g)
        else:
            sv, params = (voice, drum_params(voice, inst.params.get(voice))) if voice in GROUP_OF else \
                synth_voice(voice, inst.params.get(voice))
            shot = one_shot(sv, params, vel, variant, sr)
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
        group = _stem_of(inst, voice)
        if group not in bufs:
            bufs[group] = np.zeros((N, 2), np.float32)
        dsp.add_at(bufs[group], y, s)
    out = []
    if inst.stems == "each":
        order, sends = list(gm.STEM_ORDER), DRUM_SEND_EACH
    else:
        order, sends = ["kick", "snare", "hats", "cymbals", "kit"], DRUM_SEND
    if kit is not None:
        notes += [f"{name}: {w}" for w in kit.warnings[:4]]
    for group in sorted(bufs, key=order.index):
        buf = _post(inst, bufs[group], comp, sr, S0, b2s, beat_s, sends.get(group, 0.6), eq=True)
        audio = _finish(buf, pre, n)
        if not _audible(audio):
            continue
        out.append(BackingStem(f"{name} {group}" if inst.split else name, inst.role, audio, inst.gain_db, inst.pan,
                               inst.sidechain, tuple(notes), credits, label))
    return out


def _render_sampled(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s, dirs=()) -> BackingStem | None:
    """A pitched instrument played by its sampled sound layers; the synthesised preset when
    none of them can be loaded."""
    layers, notes, credits, labels = [], [], [], []
    for layer in inst.sound:
        res = soundbank.resolve(layer, synth_preset=inst.preset, low=inst.low, dirs=list(dirs),
                                chords=inst.role not in ("bass", "lead"))
        if res.instrument is None:
            notes.append(f"{name}: {res.error}")
        else:
            layers.append((layer, res.instrument))
            labels.append(res.label)
            credits += [c for c in res.credits if c not in credits]
            notes += [f"{name}: {w}" for w in res.warnings[:4]]
    if not layers:
        st = _render_synth(comp, name, inst, sr, S0, N, pre, n, b2s, beat_s)
        if st is not None:
            st.warnings = tuple(f"{m}; the synthesised preset {inst.preset!r} plays instead" for m in notes)
        return st
    p = _synth_params(inst)
    stops = _stop_samples(comp, name, b2s, sr, S0)
    buf = None
    for layer, sinst in layers:
        pl = p
        if layer.get("mono") is not None or layer.get("glide") is not None:
            pl = dict(p)
            pl["mono"] = bool(layer.get("mono", p.get("mono")))
            pl["glide"] = float(layer.get("glide", p.get("glide") or 0.0))
        hits = _synth_hits(comp, name, inst, pl, b2s)
        glide_s = float(pl.get("glide") or 0.0)
        transpose = int(layer.get("transpose") or 0)
        drift = float(layer.get("drift") or 0.0)
        keys = set(sinst.keys())
        counts: dict[int, int] = {}
        cache: dict = {}
        for h in hits:
            s = int(round(h.start * sr)) - S0
            pitches = sorted(h.pitches, reverse=h.strum < 0)
            late = s >= N
            vq = float(min(127, max(1, int(round(midi_velocity(h.velocity) / 2.0)) * 2)))
            for i, pitch in enumerate(pitches):
                key = int(pitch) + transpose
                seq = counts.get(key, 0)
                counts[key] = seq + 1  # counted for every note of the song, so a window renders the same takes
                if late or key not in keys:
                    continue
                if s + int((h.gate + SAMPLED_TAIL) * sr) <= 0:
                    continue
                u1, u2 = _rand2(h.seed, key, "layer")
                regs = sinst.select(key, int(vq), seq, u1 * 0.999)
                if not regs:
                    continue
                cents = int(round((u2 - 0.5) * 2.0 * drift)) if drift else 0
                glide = None
                if h.glide_from is not None and glide_s > 0 and i == 0:
                    glide = round(float(h.glide_from) + transpose - key, 2)
                ck = (tuple(id(r) for r in regs), key, vq, int(round(h.gate * 200)),
                      None if h.cut is None else int(round(h.cut * 200)), cents, glide)
                y = cache.get(ck)
                if y is None:
                    bend = None
                    if glide:
                        m = max(2, int(glide_s * sr))
                        bend = glide * np.linspace(1.0, 0.0, m)
                    y = render_note(sinst, key, vq, round(h.gate * 200) / 200.0, sr, pitch=key + cents / 100.0,
                                    seq=seq, rand=u1 * 0.999, regions=regs, bend=bend,
                                    cut=None if h.cut is None else round(h.cut * 200) / 200.0)
                    y.setflags(write=False)
                    cache[ck] = y
                at = s + int(round(i * abs(h.strum) * sr))
                if at + len(y) <= 0 or at >= N:
                    continue
                g = (midi_velocity(h.velocity) / vq) ** 2
                y = _cut_at_stop(y * np.float32(g), at, stops, sr)
                if buf is None:
                    buf = np.zeros((N, 2), np.float32)
                dsp.add_at(buf, y, at)
            if late:
                break
    if buf is None:
        return None
    buf = _post(inst, buf, comp, sr, S0, b2s, beat_s, eq=True)
    audio = _finish(buf, pre, n)
    if not _audible(audio):
        return None
    return BackingStem(name, inst.role, audio, inst.gain_db, inst.pan, inst.sidechain, tuple(notes), tuple(credits),
                       " + ".join(labels))


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


def sound_report(project) -> dict:
    """Which sound every instrument of the backing plays on this machine.

    ``{"instruments": {name: {"type", "sound", "plays": "sampled" | "synth", "label", "problem"}},
    "warnings": [...], "credits": [...]}``: ``warnings`` lists every sound that falls back to the
    synthesised instrument and why, ``credits`` the credit lines of the sound sets in use.
    """
    comp = compiled_for(project)
    out = {"instruments": {}, "warnings": [], "credits": []}
    if comp is None:
        return out
    dirs = soundbank.project_dirs(project, project.backing)
    for name, inst in comp.instruments.items():
        entry = {"type": inst.kind, "sound": None, "plays": "synth"}
        if inst.sound:
            entry["sound"] = [soundbank.describe(layer) for layer in inst.sound]
            labels, problems = [], []
            for layer in inst.sound:
                if inst.kind == "drums":
                    res = soundbank.resolve(layer, drums=True, kit=inst.kit, kit_params=inst.params, dirs=dirs)
                else:
                    res = soundbank.resolve(layer, synth_preset=inst.preset, low=inst.low, dirs=dirs,
                                            chords=inst.role not in ("bass", "lead"))
                if res.instrument is None:
                    problems.append(res.error)
                else:
                    labels.append(res.label)
                    out["credits"] += [c for c in res.credits if c not in out["credits"]]
            if labels:
                entry.update(plays="sampled", label=" + ".join(labels))
            fallback = "the synthesised kit" if inst.kind == "drums" else f"the synthesised preset {inst.preset!r}"
            for msg in problems:
                out["warnings"].append(f"{name}: {msg}" + ("" if labels else f"; {fallback} plays instead"))
            if problems:
                entry["problem"] = "; ".join(problems)
        out["instruments"][name] = entry
    return out


def song_length_seconds(project) -> float:
    """End of the last section/event of the backing in seconds (0 without backing)."""
    comp = compiled_for(project)
    if comp is None:
        return 0.0
    return float(project.tempo.beat_to_sec(comp.end_beat))

