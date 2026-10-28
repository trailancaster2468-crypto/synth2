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


def smooth_noise(n: int, frame_rate: float, rng: np.random.Generator, cutoff_hz: float = 1.5, n_sines: int = 6):
    """Low-frequency band-limited noise in [-1, 1] built from random sinusoids."""
    t = np.arange(n) / frame_rate
    out = np.zeros(n)
    for _ in range(n_sines):
        f = rng.uniform(0.15, cutoff_hz)
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 2 * np.pi))
    return out / np.sqrt(n_sines / 2.0)


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
        if nt.start - prev.end >= 0.08:
            # separated by a rest: switch while silent, just before the onset consonants
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
        ov = nt.attrs.get("overshoot", style.overshoot) * (1 - robotic)
        if ov > 0:
            dt = t - b
            om = (dt > 0) & (dt < 0.4)
            amp = np.sign(interval) * min(abs(interval), 7) * ov
            pitch[om] += amp * np.sin(2 * np.pi * 5.5 * dt[om]) * np.exp(-dt[om] / 0.07)

    # 2) scoops into phrase starts, falls at phrase ends
    for nt in notes:
        dur = nt.end - nt.start
        if nt.phrase_start and style.scoop > 0 and not nt.melisma:
            st = min(style.scoop_time, 0.4 * dur)
            lead = min(nt.lead, 0.12)
            x = (t - (nt.start - lead)) / max(st + lead, 1e-3)
            m = (x >= 0) & (x < 1) & (t < nt.end)
            pitch[m] -= style.scoop * (1 - _smoothstep(x[m])) * (1 - robotic)
        if nt.phrase_end and style.fall > 0 and dur > 0.3:
            ft = min(style.fall_time, 0.35 * dur)
            x = (t - (nt.end - ft)) / ft
            m = (x >= 0) & (x <= 1.2)
            pitch[m] -= style.fall * np.clip(x[m], 0, 1.2) ** 2 * (1 - robotic)

    # 3) vibrato
    vib = np.zeros(n)
    for nt in notes:
        dur = nt.end - nt.start
        depth = nt.attrs.get("vibrato_depth")
        depth = voice.vibrato_depth * style.vibrato_depth if depth is None else float(depth)
        if depth <= 0 or dur < style.vibrato_min_note:
            continue
        rate = nt.attrs.get("vibrato_rate")
        rate = voice.vibrato_rate * style.vibrato_rate if rate is None else float(rate)
        vstart = nt.attrs.get("vibrato_start")
        vstart = min(style.vibrato_start, 0.45 * dur) if vstart is None else float(vstart)
        a = nt.start + vstart
        m = (t >= a) & (t < nt.end + 0.05)
        if not m.any():
            continue
        tt = t[m] - a
        fade_in = _smoothstep(tt / max(style.vibrato_fade, 1e-3))
        fade_out = _smoothstep((nt.end + 0.05 - t[m]) / 0.12)
        wobble = 1.0 + (1 - robotic) * 0.06 * np.sin(2 * np.pi * 0.8 * tt + rng.uniform(0, 6.28))
        phase = 2 * np.pi * np.cumsum(rate * wobble) / frame_rate + rng.uniform(0, 0.5)
        dwob = 1.0 + (1 - robotic) * 0.15 * np.sin(2 * np.pi * 0.55 * tt + rng.uniform(0, 6.28))
        vib[m] += depth * dwob * fade_in * fade_out * np.sin(phase)
    if vibrato_env is not None:
        vib *= vibrato_env
    pitch += vib

    # 4) drift
    if style.drift_cents > 0 and robotic < 1:
        pitch += smooth_noise(n, frame_rate, rng, 2.0) * style.drift_cents / 100.0 * (1 - robotic)
    return pitch


def build_dynamics(t: np.ndarray, notes: list[NoteTiming], style: Style, voice, pitch: np.ndarray):
    """Return (loudness_dB, tension, breathiness) frame arrays generated by the style."""
    n = len(t)
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
        m = (t >= a - 0.3) & (t <= b + 0.1)
        x = np.clip((t[m] - a) / (b - a), 0, 1)
        loud[m] += style.phrase_arc_db * (np.sin(np.pi * np.clip(x * 1.2, 0, 1)) - 0.5)

    for nt in notes:
        dur = nt.end - nt.start
        m = (t >= nt.start) & (t < nt.end)
        if not m.any():
            continue
        x = (t[m] - nt.start) / max(dur, 1e-3)
        vel = float(nt.attrs.get("velocity", 1.0) or 1.0)
        loud[m] += 20 * np.log10(max(vel, 0.05))
        if dur > 0.5:
            loud[m] += style.note_swell_db * np.sin(np.pi * x) * min(1.0, (dur - 0.5) / 0.5)
        if nt.attrs.get("tension") is not None:
            tension[m] += float(nt.attrs["tension"])
        if nt.attrs.get("breathiness") is not None:
            breath[m] += float(nt.attrs["breathiness"])
        if nt.phrase_end:
            tail = np.clip((x - 0.55) / 0.45, 0, 1)
            loud[m] -= style.end_taper_db * tail
            breath[m] += style.end_breathiness * tail
    return loud, tension, breath


def evaluate_curve(curve, t: np.ndarray, beat_to_sec, default: float) -> np.ndarray:
    """Evaluate a project :class:`~synth2.project.Curve` at times ``t`` (seconds)."""
    if curve is None or not curve.points:
        return np.full(len(t), default, dtype=float)
    xs = np.array([beat_to_sec(b) for b, _ in curve.points])
    ys = np.array([v for _, v in curve.points], dtype=float)
    if len(xs) == 1:
        return np.full(len(t), ys[0])
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
