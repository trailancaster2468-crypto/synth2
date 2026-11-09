"""Automatic expression ("AI pitch / dynamics") engine.

Turns a flat sequence of notes into a human-like performance:

* note-to-note pitch transitions (portamento) with preparation and overshoot
* scoops into phrase starts and falls at phrase ends
* delayed, fading vibrato with natural rate/depth wobble
* slow pitch drift and pitch-dependent dynamics / tension
* phrase-level dynamics arcs

The behaviour is controlled by named :class:`Style` presets (``pop``,
``ballad``, ``rock``, ``jazz``, ``robotic``, ``soft``, ``classical``,
``rnb``) and can be overridden per note via note attributes.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace, fields

import numpy as np


@dataclass(frozen=True)
class Style:
    name: str
    description: str
    transition: float = 0.09  # s, base portamento time
    transition_offset: float = -0.35  # centre of transition relative to note onset, in units of transition time
    overshoot: float = 0.08  # fraction of the interval
    preparation: float = 0.03  # fraction of the interval (dip before moving)
    scoop: float = 0.45  # semitones below target at phrase start
    scoop_time: float = 0.09
    fall: float = 0.5  # semitones drop at phrase end
    fall_time: float = 0.14
    vibrato_depth: float = 1.0  # multiplier on voice vibrato depth
    vibrato_rate: float = 1.0  # multiplier on voice vibrato rate
    vibrato_start: float = 0.28  # s after onset
    vibrato_fade: float = 0.25  # s fade-in
    vibrato_min_note: float = 0.38  # s, shorter notes get no vibrato
    drift_cents: float = 7.0
    phrase_arc_db: float = 1.5  # crescendo into the phrase centre, decrescendo out
    note_swell_db: float = 0.8  # swell on long notes
    end_taper_db: float = 2.5  # loudness taper on phrase-final notes
    pitch_loudness: float = 0.25  # dB per semitone above the voice's mid range
    pitch_tension: float = 0.025  # tension per semitone above mid range
    tension: float = 0.0
    breathiness: float = 0.0
    end_breathiness: float = 0.25  # extra breath on phrase endings
    auto_breath: bool = True
    timing_humanize: float = 0.0  # random timing offset std (s)

    def to_dict(self):
        return asdict(self)


STYLES: dict[str, Style] = {s.name: s for s in [
    Style("pop", "Modern pop: clean transitions, moderate vibrato on held notes, light scoops."),
    Style("ballad", "Emotional ballad: slower glides, deeper delayed vibrato, breathy phrase ends.",
          transition=0.13, overshoot=0.05, scoop=0.6, scoop_time=0.12, fall=0.35, vibrato_depth=1.15,
          vibrato_start=0.35, vibrato_fade=0.4, phrase_arc_db=2.5, note_swell_db=1.5, breathiness=0.15,
          end_breathiness=0.45, tension=-0.15),
    Style("rock", "Rock: punchy attacks, strong scoops, pressed tone, fast wide vibrato.",
          transition=0.07, overshoot=0.12, preparation=0.05, scoop=0.9, scoop_time=0.07, fall=0.9,
          fall_time=0.12, vibrato_depth=1.2, vibrato_rate=1.08, vibrato_start=0.3, tension=0.35,
          pitch_tension=0.04, pitch_loudness=0.35, end_breathiness=0.1),
    Style("jazz", "Jazz/soul: bluesy scoops and falls, lazy timing, warm relaxed tone.",
          transition=0.12, overshoot=0.06, scoop=1.0, scoop_time=0.13, fall=1.2, fall_time=0.2,
          vibrato_depth=0.9, vibrato_rate=0.92, vibrato_start=0.4, tension=-0.1, breathiness=0.1,
          timing_humanize=0.008),
    Style("rnb", "R&B: smooth wide glides, expressive scoops and controlled vibrato.",
          transition=0.14, overshoot=0.1, preparation=0.05, scoop=0.8, scoop_time=0.11, fall=0.8,
          vibrato_depth=1.05, vibrato_start=0.32, breathiness=0.1, end_breathiness=0.35),
    Style("robotic", "Classic vocaloid / machine tuning: instant pitch snaps, no drift, steady tone.",
          transition=0.025, transition_offset=-0.5, overshoot=0.0, preparation=0.0, scoop=0.0, fall=0.0,
          vibrato_depth=0.6, vibrato_start=0.45, vibrato_fade=0.15, vibrato_min_note=0.6, drift_cents=0.0,
          phrase_arc_db=0.0, note_swell_db=0.0, end_taper_db=0.5, pitch_loudness=0.1, pitch_tension=0.0,
          end_breathiness=0.0, auto_breath=False),
    Style("soft", "Soft/intimate: gentle onsets, little vibrato, airy breathy tone.",
          transition=0.12, overshoot=0.03, scoop=0.3, scoop_time=0.12, fall=0.25, vibrato_depth=0.6,
          vibrato_start=0.45, phrase_arc_db=1.0, note_swell_db=0.6, tension=-0.45, breathiness=0.4,
          end_breathiness=0.5, pitch_loudness=0.15),
    Style("classical", "Classical/operatic: legato lines, early rich vibrato, bright projected tone.",
          transition=0.11, overshoot=0.04, preparation=0.02, scoop=0.15, fall=0.1, vibrato_depth=1.6,
          vibrato_rate=1.0, vibrato_start=0.12, vibrato_fade=0.2, vibrato_min_note=0.3, tension=0.25,
          phrase_arc_db=2.0, note_swell_db=1.2, end_taper_db=1.5, end_breathiness=0.05),
    Style("flat", "No automatic expression: pitch exactly follows the notes (for manual tuning).",
          transition=0.04, overshoot=0.0, preparation=0.0, scoop=0.0, fall=0.0, vibrato_depth=0.0,
          drift_cents=0.0, phrase_arc_db=0.0, note_swell_db=0.0, end_taper_db=0.0, pitch_loudness=0.0,
          pitch_tension=0.0, end_breathiness=0.0, auto_breath=False),
]}


def get_style(name: str, overrides: dict | None = None) -> Style:
    s = STYLES.get(name) or STYLES["pop"]
    if overrides:
        valid = {f.name for f in fields(Style)} - {"name", "description"}
        s = replace(s, **{k: type(getattr(s, k))(v) for k, v in overrides.items() if k in valid})
    return s


def list_styles() -> list[dict]:
    return [{"name": s.name, "description": s.description} for s in STYLES.values()]


@dataclass
class NoteTiming:
    """A note in seconds, annotated with phrase context by the planner."""

    index: int
    start: float
    end: float
    pitch: float  # MIDI (float, includes detune)
    attrs: dict
    phrase_start: bool = False
    phrase_end: bool = False
    lead: float = 0.0  # seconds of onset consonants before ``start``
    melisma: bool = False


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def smooth_noise(n: int, frame_rate: float, rng: np.random.Generator, cutoff_hz: float = 1.5, n_sines: int = 6,
                 t0: float = 0.0):
    """Low-frequency band-limited noise in [-1, 1] built from random sinusoids.

    ``t0`` is the absolute time of the first frame, so the same ``rng`` state gives the same
    noise at the same time whatever window is rendered.
    """
    t = t0 + np.arange(n) / frame_rate
    out = np.zeros(n)
    for _ in range(n_sines):
        f = rng.uniform(0.15, cutoff_hz)
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(n_sines / 2.0)


def _keyed_rng(rng: np.random.Generator, *key: int) -> np.random.Generator:
    """A generator that depends only on the seed ``rng`` was made from and on ``key``.

    It does not depend on how much of ``rng`` has been drawn (by the planner, or for other notes),
    so a partial render gets the same randomness for a note as the full render of the song.
    """
    bg = rng.bit_generator
    ss = getattr(bg, "seed_seq", None) or getattr(bg, "_seed_seq", None)
    if not isinstance(ss, np.random.SeedSequence):
        return rng
    spawn = tuple(ss.spawn_key) + tuple(int(k) % (1 << 32) for k in key)
    return np.random.default_rng(np.random.SeedSequence(ss.entropy, spawn_key=spawn))


def build_pitch(t: np.ndarray, notes: list[NoteTiming], style: Style, voice, rng: np.random.Generator,
                vibrato_env: np.ndarray | None = None) -> np.ndarray:
    """Return the sung pitch (MIDI semitones, float) at frame times ``t``."""
    n = len(t)
    if not notes:
        return np.full(n, 60.0)
    frame_rate = 1.0 / (t[1] - t[0]) if n > 1 else 200.0
    robotic = float(voice.robotic)

    # 1) base curve, built note by note in time order so each note owns the frames
    #    from its transition onwards (later notes overwrite earlier ones)
    pitch = np.full(n, notes[0].pitch, dtype=float)
    for i in range(1, len(notes)):
        nt, prev = notes[i], notes[i - 1]
        if (nt.start - nt.lead) - prev.end >= 0.08:
            # separated by a rest that is still silent where the onset consonants begin: switch
            # while silent, just before them. (Onset consonants may borrow the whole rest and part
            # of the previous note; then the previous vowel is still sounding and the note is
            # joined with a normal transition below instead of a jump inside that vowel.)
            pitch[t >= nt.start - nt.lead - 0.015] = nt.pitch
            continue
        interval = nt.pitch - prev.pitch
        T = nt.attrs.get("transition")
        if T is None:
            T = style.transition * (1.0 + 0.035 * min(abs(interval), 12))
        T = float(np.clip(T, 0.01, 0.6))
        T = min(T, 0.6 * (prev.end - prev.start) + 0.02, 0.8 * (nt.end - nt.start) + 0.02)
        centre = nt.start + style.transition_offset * T
        a, b = centre - T / 2, centre + T / 2
        mask = t >= a
        pitch[mask] = prev.pitch + interval * _smoothstep((t[mask] - a) / max(T, 1e-4))
        if interval == 0:
            continue
        # preparation: a small movement away from the target just before the transition
        prep = style.preparation * (1 - robotic)
        if prep > 0:
            pw = T * 0.8
            px = (t - (a - pw)) / pw
            pm = (px > 0) & (px < 1)
            pitch[pm] -= np.sign(interval) * min(abs(interval), 7) * prep * np.sin(np.pi * px[pm])
        # overshoot: damped oscillation after arriving
        ov = nt.attrs.get("overshoot")
        ov = (style.overshoot if ov is None else float(ov)) * (1 - robotic)
        if ov > 0:
            dt = t - b
            om = (dt > 0) & (dt < 0.4)
            amp = np.sign(interval) * min(abs(interval), 7) * ov
            pitch[om] += amp * np.sin(2 * np.pi * 5.5 * dt[om]) * np.exp(-dt[om] / 0.07)

    # 2) scoops into phrase starts, falls at phrase ends
    for i, nt in enumerate(notes):
        dur = nt.end - nt.start
        if nt.phrase_start and style.scoop > 0 and not nt.melisma:
            st = min(style.scoop_time, 0.4 * dur)
            lead = min(nt.lead, 0.12)
            x = (t - (nt.start - lead)) / max(st + lead, 1e-3)
            # hold the full scoop depth back to where the pitch switches to this note (before the
            # onset consonants), so a voiced onset (l, m, w...) never hears a pitch step
            t_switch = nt.start - nt.lead - 0.015
            if i > 0:
                t_switch = max(t_switch, notes[i - 1].end + 0.02)
            m = (t >= t_switch) & (x < 1) & (t < nt.end)
            pitch[m] -= style.scoop * (1 - _smoothstep(x[m])) * (1 - robotic)
        if nt.phrase_end and style.fall > 0 and dur > 0.3:
            ft = min(style.fall_time, 0.35 * dur)
            x = (t - (nt.end - ft)) / ft
            m = (x >= 0) & (x <= 1.2)
            pitch[m] -= style.fall * np.clip(x[m], 0, 1.2) ** 2 * (1 - robotic)

    # 3) vibrato. Each note's randomness comes from its own generator keyed by its onset time, and
    #    everything is a function of absolute time, so a partial render sings the same vibrato.
    vib = np.zeros(n)
    for nt in notes:
        dur = nt.end - nt.start
        depth = nt.attrs.get("vibrato_depth")
        depth = voice.vibrato_depth * style.vibrato_depth if depth is None else float(depth)
        rate = nt.attrs.get("vibrato_rate")
        rate = voice.vibrato_rate * style.vibrato_rate if rate is None else float(rate)
        if depth <= 0 or rate <= 0 or dur < style.vibrato_min_note:
            continue
        vstart = nt.attrs.get("vibrato_start")
        vstart = min(style.vibrato_start, 0.45 * dur) if vstart is None else float(vstart)
        a = nt.start + vstart
        m = (t >= a) & (t < nt.end + 0.05)
        if not m.any():
            continue
        nrng = _keyed_rng(rng, 1, int(round(nt.start * 1000)))
        p_wob, p0, p_dwob = nrng.uniform(0, 6.28), nrng.uniform(0, 0.5), nrng.uniform(0, 6.28)
        tt = t[m] - a
        fade_in = _smoothstep(tt / max(style.vibrato_fade, 1e-3))
        fade_out = _smoothstep((nt.end + 0.05 - t[m]) / 0.12)
        # phase = 2 pi * integral from the vibrato start of rate * (1 + w sin(2 pi 0.8 s + p_wob)) ds
        w, wf = (1 - robotic) * 0.06, 2 * np.pi * 0.8
        phase = 2 * np.pi * rate * (tt + w / wf * (np.cos(p_wob) - np.cos(wf * tt + p_wob))) + p0
        dwob = 1.0 + (1 - robotic) * 0.15 * np.sin(2 * np.pi * 0.55 * tt + p_dwob)
        vib[m] += depth * dwob * fade_in * fade_out * np.sin(phase)
    if vibrato_env is not None:
        vib *= vibrato_env
    pitch += vib

    # 4) drift (on absolute time, from a generator that only depends on the seed)
    if style.drift_cents > 0 and robotic < 1:
        drift = smooth_noise(n, frame_rate, _keyed_rng(rng, 0), 2.0, t0=float(t[0]) if n else 0.0)
        pitch += drift * style.drift_cents / 100.0 * (1 - robotic)
    return pitch


def build_dynamics(t: np.ndarray, notes: list[NoteTiming], style: Style, voice, pitch: np.ndarray):
    """Return (loudness_dB, tension, breathiness) frame arrays generated by the style."""
    n = len(t)
    frame_rate = 1.0 / (t[1] - t[0]) if n > 1 else 200.0
    loud = np.zeros(n)
    tension = np.full(n, style.tension)
    breath = np.full(n, style.breathiness)
    mid = 0.5 * (voice.range_low + voice.range_high)
    above = np.clip(pitch - mid, -12, 12)
    loud += style.pitch_loudness * above
    tension += style.pitch_tension * above

    # phrase arcs
    phrases: list[list[NoteTiming]] = []
    for nt in notes:
        if nt.phrase_start or not phrases:
            phrases.append([])
        phrases[-1].append(nt)
    for ph in phrases:
        a, b = ph[0].start, ph[-1].end
        if b - a < 0.3:
            continue
        m = (t >= a - 0.3) & (t <= b + 0.3)
        x = np.clip((t[m] - a) / (b - a), 0, 1)
        loud[m] += style.phrase_arc_db * (np.sin(np.pi * np.clip(x * 1.2, 0, 1)) - 0.5)

    # per-note velocity / tension / breathiness hold over the note's whole sung span: from its onset
    # consonants (start - lead) to where the next note's consonants begin, or through the release of
    # a phrase-final note (short of an automatic breath before the next phrase). The spans are
    # assigned, not added, so they can neither overlap nor leave gaps when note times are humanized.
    vel_db, ten_off, br_off = np.zeros(n), np.zeros(n), np.zeros(n)
    for i, nt in enumerate(notes):
        nxt = notes[i + 1].start - notes[i + 1].lead if i + 1 < len(notes) else np.inf
        hi = min(nxt, nt.end + 0.1) if nt.phrase_end or i + 1 == len(notes) else nxt
        m = (t >= nt.start - nt.lead) & (t < hi)
        v = nt.attrs.get("velocity")
        vel_db[m] = 20 * np.log10(max(1.0 if v is None else float(v), 0.05))
        v = nt.attrs.get("tension")
        ten_off[m] = 0.0 if v is None else float(v)
        v = nt.attrs.get("breathiness")
        br_off[m] = 0.0 if v is None else float(v)
    loud += vel_db
    tension += ten_off
    breath += br_off

    for i, nt in enumerate(notes):
        dur = nt.end - nt.start
        m = (t >= nt.start) & (t < nt.end)
        if not m.any():
            continue
        x = (t[m] - nt.start) / max(dur, 1e-3)
        if dur > 0.5:
            loud[m] += style.note_swell_db * np.sin(np.pi * x) * min(1.0, (dur - 0.5) / 0.5)
        if nt.phrase_end:
            # the taper (and the breathier tone) holds through the release instead of snapping
            # back while the voice is still fading out
            nxt = notes[i + 1].start - notes[i + 1].lead if i + 1 < len(notes) else np.inf
            mt = (t >= nt.start) & (t < min(nt.end + 0.3, nxt - 0.05))
            tail = np.clip(((t[mt] - nt.start) / max(dur, 1e-3) - 0.55) / 0.45, 0, 1)
            loud[mt] -= style.end_taper_db * tail
            breath[mt] += style.end_breathiness * tail
    # the generated dynamics never step faster than ~40 ms (per-note velocity / tension /
    # breathiness attributes, phrase-arc edges); user curves are added later and stay exact
    w = max(1, int(round(0.04 * frame_rate)))
    return _smooth(loud, w), _smooth(tension, w), _smooth(breath, w)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    """Hann smoothing over ``width`` frames with edge padding."""
    if width < 2 or len(x) < 2:
        return x
    k = np.hanning(width + 2)[1:-1]
    xp = np.pad(x, width, mode="edge")
    return np.convolve(xp, k / k.sum(), mode="same")[width:-width]


def _times_to_beats(t: np.ndarray, beat_to_sec, beats: np.ndarray) -> np.ndarray:
    """Map times (s) to beats between ``beats[0]`` and ``beats[-1]`` (sorted).

    ``beat_to_sec`` is linear between tempo changes, so its inverse is interpolated through
    ``beats`` plus every tempo change between them, which is exact for a :class:`TempoMap` (passed
    itself or as its bound ``beat_to_sec``). Another callable is sampled finely instead. Times
    outside that span only keep their side of it (a curve is constant there).
    """
    tempo = beat_to_sec if hasattr(beat_to_sec, "beat_to_sec") else getattr(beat_to_sec, "__self__", None)
    b2s = getattr(beat_to_sec, "beat_to_sec", beat_to_sec)
    b0, b1 = beats[0], beats[-1]
    tempo_points = getattr(tempo, "points", None)
    if tempo_points is not None:
        breaks = np.array([float(p.beat) for p in tempo_points], dtype=float)
    else:
        breaks = np.linspace(b0, b1, 2049)
    grid = np.unique(np.concatenate([beats, breaks[(breaks > b0) & (breaks < b1)]]))
    secs = np.array([b2s(float(b)) for b in grid])
    tb = np.interp(t, secs, grid)
    tb[t < secs[0]] = b0 - 1.0
    tb[t > secs[-1]] = b1 + 1.0
    return tb


def evaluate_curve(curve, t: np.ndarray, beat_to_sec, default: float) -> np.ndarray:
    """Evaluate a project :class:`~synth2.project.Curve` at times ``t`` (seconds).

    The curve is interpolated in beat space, the way the studio's parameter lanes draw it, so a
    tempo change between two points does not bend it. ``beat_to_sec`` is the tempo map's
    ``beat_to_sec`` (or the :class:`TempoMap` itself).
    """
    if curve is None or not curve.points:
        return np.full(len(t), default, dtype=float)
    pts = sorted(curve.points, key=lambda p: p[0])  # stable: same-beat jumps keep their order
    ys = np.array([v for _, v in pts], dtype=float)
    if len(pts) == 1:
        return np.full(len(t), ys[0])
    xs = np.array([b for b, _ in pts], dtype=float)
    t = _times_to_beats(np.asarray(t, dtype=float), beat_to_sec, xs)  # from here on t is in beats
    if curve.interpolation == "step":
        idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 1)
        return ys[idx]
    if curve.interpolation == "linear":
        return np.interp(t, xs, ys)
    # "cubic": cosine-eased segments (smooth, never overshoots the control points)
    idx = np.clip(np.searchsorted(xs, t, side="right") - 1, 0, len(xs) - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    y0, y1 = ys[idx], ys[idx + 1]
    u = np.clip((t - x0) / np.maximum(x1 - x0, 1e-9), 0, 1)
    u = 0.5 - 0.5 * np.cos(np.pi * u)
    out = y0 + (y1 - y0) * u
    out[t <= xs[0]] = ys[0]
    out[t >= xs[-1]] = ys[-1]
    return out
