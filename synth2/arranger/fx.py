"""Sound effects: one-shot FX sounds and processing effects applied to stems.

One-shots (``{"type": "fx", "sound": ...}`` events): laser, zap, ufo, riser,
downlifter, whoosh, impact, boom, radio, bleeps, glitch, modem, alarm, powerdown,
coin, error, reverse_cymbal, sweep.

Processing effects (``{"type": "stutter" | "tapestop" | "crush" | "filter" | "gate" |
"reverse" | "telephone", ...}`` events) rewrite a time range of the affected stems:
glitch stutter/retrigger, tape stop, bit crush, filter sweep, rhythmic gate,
reverse and a lo-fi telephone/radio band.
"""
from __future__ import annotations

import math

import numpy as np

from .dsp import (bandpass_noise, convolve_stereo, fade_edges, fft_filter, filter_response, noise, osc, pan_gains,
                  reverb_ir, saturate, sine, stft_filter)
from ..timing import midi_to_hz

# sound -> (default length in beats, default pitch (MIDI) or None, description)
FX_SOUNDS: dict[str, tuple[float, int | None, str]] = {
    "laser": (0.5, 96, "pew! fast falling pitch sweep (a space-gun shot); pitch = start note"),
    "zap": (0.25, 100, "short electric zap / ray-gun blip"),
    "ufo": (4.0, 76, "warbling flying-saucer theremin hover (wobbly vibrato, slow glide, autopan)"),
    "riser": (8.0, 48, "noise + saw build-up rising over its length (end it on the downbeat of the drop)"),
    "downlifter": (4.0, 72, "falling noise sweep after a drop / into a breakdown"),
    "whoosh": (1.0, None, "quick noise swoosh up and down (transitions)"),
    "sweep": (4.0, None, "filtered white-noise sweep up (params: up=false for down)"),
    "impact": (4.0, 33, "cinematic hit: sub drop + noise burst + reverb tail (section starts, final hit)"),
    "boom": (2.0, 31, "808-style sub boom / drop"),
    "radio": (4.0, None, "alien transmission: band-limited static, crackle, carrier whistle and beeps"),
    "bleeps": (2.0, 84, "random computer bleeps (deterministic per event); params: rate (steps per beat)"),
    "glitch": (1.0, 72, "granular digital glitch burst: random blips, noise grains and crushed bits"),
    "modem": (2.0, None, "dial-up modem handshake chirps (the internet itself attacking)"),
    "alarm": (4.0, 81, "two-tone siren / red-alert alarm"),
    "powerdown": (2.0, 72, "machine powering down: pitch falls 3 octaves while the filter closes"),
    "coin": (0.5, 83, "chiptune coin / point pickup (two quick squares)"),
    "error": (1.0, 45, "buzzy computer error 'eh-eh' (two square buzzes)"),
    "reverse_cymbal": (2.0, None, "reversed crash swelling into the next downbeat (end it on the downbeat)"),
}
FX_RMS = 0.063  # -24 dBFS
# sub-heavy sounds read as quieter than their RMS: lift them
FX_TRIM_DB = {"boom": 9.0, "impact": 4.0, "powerdown": 2.0, "coin": 2.0, "error": 2.0}
FX_PEAK = 0.63  # -4 dBFS
PROCESS_EFFECTS = {
    "stutter": "retrigger the first `slice` beats of the range over and over (glitch stutter); params: slice "
               "(beats, default 0.25), decay (0..1 level drop per repeat), pitch (semitones per repeat), "
               "gate (0..1 portion of each slice that sounds)",
    "tapestop": "tape/turntable stop: playback slows to a halt over the range (then the audio resumes after it); "
                "params: curve (1 = linear slowdown, 2 = holds speed longer)",
    "crush": "bit crusher + sample-rate reduction over the range; params: bits (default 6), downsample "
             "(default 8), mix (0..1)",
    "filter": "filter sweep over the range; params: mode lp|hp|bp, from_hz, to_hz, resonance (0..1)",
    "gate": "rhythmic volume gate (trance gate); params: steps (string like 'x.x.xx.x', one char per 16th), "
            "grid (default 16)",
    "reverse": "reverse the audio of the range (each `slice` beats, default the whole range)",
    "telephone": "lo-fi telephone/radio band-pass with a little crush over the range (alien transmissions)",
}


def _t(n, sr):
    return np.arange(n, dtype=np.float32) / sr


def _env_ad(n, sr, attack, decay_tau):
    t = _t(n, sr)
    return (np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-np.maximum(t - attack, 0) / decay_tau)).astype(np.float32)


def render_fx(sound: str, dur: float, pitch: int | None, sr: int, rng: np.random.Generator,
              params: dict | None = None, beat_s: float = 0.357) -> np.ndarray:
    """Render an FX one-shot of about ``dur`` seconds (tails may run longer) as stereo float32.

    ``params`` may hold ``fade_in`` (s, default 0.001): a longer fade-in for a sound that starts the song
    from digital silence (a 1 ms edge on noise reads as a click)."""
    p = params or {}
    if pitch is None:
        pitch = FX_SOUNDS[sound][1] or 72
    f0 = float(midi_to_hz(pitch))
    dur = max(dur, 0.03)
    fn = _FX[sound]
    out = fn(dur, f0, sr, rng, p, beat_s)
    if out.ndim == 1:
        out = np.stack([out, out], axis=1)
    out = out.astype(np.float32)
    pk = float(np.abs(out).max())
    rms = float(np.sqrt(np.mean(out.astype(np.float64) ** 2)))
    if pk > 0:  # calibrated by loudness (with a peak cap): an FX at gain_db 0 sits just under the band
        out *= min(FX_RMS / max(rms, 1e-9), FX_PEAK / pk) * 10 ** (FX_TRIM_DB.get(sound, 0.0) / 20)
    fade_in = float(np.clip(float(p.get("fade_in", 0.001) or 0.001), 0.001, 0.5 * dur))
    return fade_edges(out, sr, fade_in, 0.01)


def _laser(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    f_end = f0 * float(p.get("drop", 0.08))
    f = f_end + (f0 - f_end) * np.exp(-t / (dur * 0.22))
    f = f * (1 + 0.05 * np.sin(2 * math.pi * 31 * t))
    s = osc("square", f, n, sr) * 0.6 + osc("saw", f * 1.005, n, sr) * 0.4
    env = _env_ad(n, sr, 0.002, dur * 0.35)
    y = s * env
    pan = np.linspace(-0.4, 0.4, n, dtype=np.float32) * float(p.get("pan_sweep", 1.0))
    return np.stack([y * np.cos((pan + 1) * math.pi / 4) * 1.41, y * np.sin((pan + 1) * math.pi / 4) * 1.41], 1)


def _zap(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    f = f0 * np.exp(-t / (dur * 0.3)) + 80
    carrier = osc("saw", f, n, sr)
    ring = np.sin(2 * math.pi * 1700 * t)
    y = (carrier * (0.6 + 0.4 * ring)) * _env_ad(n, sr, 0.001, dur * 0.25)
    y += 0.3 * noise(n, rng) * _env_ad(n, sr, 0.0005, 0.01)
    return y


def _ufo(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    rate = float(p.get("rate", 9.0))
    depth = float(p.get("depth", 1.6))  # semitones
    glide = float(p.get("glide", 5.0))  # semitones over the whole length (+ up, - down)
    semis = depth * np.sin(2 * math.pi * rate * t) + glide * (t / dur) + 0.7 * np.sin(2 * math.pi * 0.7 * t)
    f = f0 * np.exp2(semis / 12)
    s = sine(f, n, sr) * 0.75 + osc("triangle", f * 2.003, n, sr) * 0.25
    ring = 0.75 + 0.25 * np.sin(2 * math.pi * 33 * t)
    env = np.minimum(1.0, t / (dur * 0.15)) * np.minimum(1.0, (dur - t) / (dur * 0.2))
    y = s * ring * env
    pan = 0.7 * np.sin(2 * math.pi * 0.5 * t)
    return np.stack([y * np.cos((pan + 1) * math.pi / 4) * 1.41, y * np.sin((pan + 1) * math.pi / 4) * 1.41], 1)


def _riser(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    x = t / dur
    nz = noise(n, rng, 2)
    lo, hi = float(p.get("from_hz", 300)), float(p.get("to_hz", 9000))
    centre = lo * (hi / lo) ** x

    def mask(times, freqs):
        c = np.interp(np.clip(times, 0, dur), t[::256], centre[::256])[:, None]
        return filter_response(freqs, c, 0.45, "bp")

    band = stft_filter(nz, sr, mask) * 1.5
    semis = 24 * x ** 1.5
    tone = osc("saw", f0 * np.exp2(semis / 12), n, sr) * 0.25 + osc("saw", f0 * 1.01 * np.exp2(semis / 12), n, sr) * 0.25
    tone = np.stack([tone, tone], 1)
    env = (x ** 2.2)[:, None]
    return (band * 0.8 + tone * float(p.get("tone", 0.6))) * env


def _downlifter(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    x = t / dur
    nz = noise(n, rng, 2)
    centre = 9000 * (200 / 9000) ** x

    def mask(times, freqs):
        c = np.interp(np.clip(times, 0, dur), t[::256], centre[::256])[:, None]
        return filter_response(freqs, c, 0.35, "bp")

    band = stft_filter(nz, sr, mask) * 1.5
    tone = osc("saw", f0 * np.exp2(-24 * x / 12), n, sr) * 0.3
    env = ((1 - x) ** 1.6)[:, None]
    return (band + np.stack([tone, tone], 1) * 0.5) * env


def _whoosh(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    x = t / dur
    nz = noise(n, rng, 2)
    centre = 400 * (8000 / 400) ** np.sin(math.pi * x)

    def mask(times, freqs):
        c = np.interp(np.clip(times, 0, dur), t[::256], centre[::256])[:, None]
        return filter_response(freqs, c, 0.3, "bp")

    band = stft_filter(nz, sr, mask)
    env = np.sin(math.pi * x) ** 2
    return band * env[:, None] * 1.5


def _sweep(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    x = t / dur
    if not p.get("up", True):
        x = 1 - x
    nz = noise(n, rng, 2)
    cut = 200 * (14000 / 200) ** x

    def mask(times, freqs):
        c = np.interp(np.clip(times, 0, dur), t[::256], cut[::256])[:, None]
        return filter_response(freqs, c, 0.5, "lp", 12)

    y = stft_filter(nz, sr, mask) * 0.6
    return y * (0.3 + 0.7 * x)[:, None]


def _impact(dur, f0, sr, rng, p, beat_s):
    n = int(sr * max(dur, 1.5))
    t = _t(n, sr)
    f = f0 * (1 + 2.5 * np.exp(-t / 0.05)) * np.exp(-t / (dur * 2.0))
    sub = sine(np.maximum(f, 20), n, sr) * np.exp(-t / (dur * 0.45))
    burst = bandpass_noise(n, sr, rng, 60, 3500) * np.exp(-t / 0.08)
    dry = saturate(sub * 0.9 + burst * 0.5, 0.3)
    st = np.stack([dry, dry], 1)
    ir = reverb_ir(sr, 2.5, 0.8, 0.6)
    wet = convolve_stereo(st * 0.5, ir)
    return st + wet * 0.7


def _boom(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    f = f0 * (1 + 1.5 * np.exp(-t / 0.04))
    y = sine(f, n, sr) * np.exp(-t / (dur * 0.4)) * np.minimum(1.0, t / 0.002)
    return saturate(y, 0.25)


def _radio(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    static = bandpass_noise(n, sr, rng, 400, 3200) * 0.5
    # crackle: sparse impulses
    crack = np.zeros(n, np.float32)
    k = max(1, int(dur * 40))
    idx = rng.integers(0, n, k)
    crack[idx] = rng.uniform(-1, 1, k).astype(np.float32) * 3
    crack = fft_filter(crack, sr, lambda f: 1.0 / np.sqrt(1.0 + (f / 5000.0) ** 2))
    # fading in and out of signal (AM 'tuning' wobble)
    wob = 0.55 + 0.45 * np.sin(2 * math.pi * (0.6 + 0.4 * rng.uniform()) * t + rng.uniform(0, 6))
    whistle = sine(1300 * (1 + 0.04 * np.sin(2 * math.pi * 0.3 * t)), n, sr) * 0.12
    beeps = np.zeros(n, np.float32)
    step = beat_s / 2
    for i in range(int(dur / step)):
        if rng.uniform() < 0.35:
            a = int(i * step * sr)
            m = int(step * 0.6 * sr)
            if a + m < n:
                beeps[a:a + m] += sine(float(rng.choice([1000, 1500, 2000])), m, sr) * 0.25
    y = static * wob + crack * 0.4 + whistle * wob + beeps
    return np.stack([y, y * 0.92], 1)


def _bleeps(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    rate = float(p.get("rate", 4))  # notes per beat
    step = beat_s / rate
    y = np.zeros((n, 2), np.float32)
    scale = [0, 3, 5, 7, 10, 12, 15, 19, 24]
    for i in range(int(dur / step)):
        if rng.uniform() < float(p.get("density", 0.8)):
            a = int(i * step * sr)
            m = int(step * sr * rng.uniform(0.3, 0.8))
            if a + m > n or m < 8:
                continue
            f = f0 * 2 ** (rng.choice(scale) / 12)
            b = osc(str(rng.choice(["square", "pulse25", "triangle"])), f, m, sr) * _env_ad(m, sr, 0.001, step * 0.5)
            gl, gr = pan_gains(float(rng.uniform(-0.6, 0.6)))
            y[a:a + m, 0] += b * gl * 0.5
            y[a:a + m, 1] += b * gr * 0.5
    return y


def _glitch(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    y = np.zeros((n, 2), np.float32)
    pos = 0
    while pos < n:
        m = int(sr * rng.choice([0.008, 0.015, 0.03, 0.045]))
        kind = rng.integers(0, 4)
        if kind == 0:
            g = noise(m, rng) * 0.6
        elif kind == 1:
            g = osc("square", f0 * 2 ** (rng.integers(-12, 24) / 12), m, sr) * 0.6
        elif kind == 2:
            g = np.zeros(m, np.float32)
        else:
            g = sine(f0 * 2 ** (rng.integers(0, 36) / 12), m, sr)
        g = np.round(g * 6) / 6  # crushed
        g = fade_edges(g.astype(np.float32), sr, 0.0005, 0.001)
        gl, gr = pan_gains(float(rng.uniform(-0.8, 0.8)))
        e = min(n, pos + m)
        y[pos:e, 0] += g[:e - pos] * gl
        y[pos:e, 1] += g[:e - pos] * gr
        pos += m
    return y


def _modem(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    y = np.zeros(n, np.float32)
    # dial tone, then FSK warble, then a noisy handshake
    a = int(n * 0.2)
    y[:a] = (sine(350, a, sr) + sine(440, a, sr)) * 0.3
    b = int(n * 0.55)
    bits = rng.integers(0, 2, int((b - a) / (sr * 0.012)) + 2)
    fs = np.repeat(np.where(bits, 1270.0, 1070.0), int(sr * 0.012))[: b - a]
    fs2 = np.repeat(np.where(rng.integers(0, 2, len(bits)), 2225.0, 2025.0), int(sr * 0.012))[: b - a]
    y[a:b] = sine(fs, b - a, sr) * 0.3 + sine(fs2, b - a, sr) * 0.25
    y[b:] = bandpass_noise(n - b, sr, rng, 600, 3500) * 0.5 * (0.6 + 0.4 * np.sin(2 * math.pi * 23 * t[b:]))
    return fade_edges(y, sr, 0.002, 0.02)


def _alarm(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    rate = float(p.get("rate", 2.0)) / beat_s / 2  # alternations per second (2 per beat by default)
    f = np.where(np.floor(t * rate * 2) % 2 == 0, f0, f0 * 0.75)
    s = osc("square", f, n, sr) * 0.5 + osc("saw", f, n, sr) * 0.3
    s = fft_filter(s, sr, lambda fr: 1.0 / np.sqrt(1.0 + (fr / 3000.0) ** 4))
    return s * np.minimum(1.0, t / 0.01)


def _powerdown(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    t = _t(n, sr)
    x = t / dur
    f = f0 * np.exp2(-36 * x ** 1.3 / 12)
    s = osc("saw", f, n, sr) * 0.5 + osc("square", f * 0.5, n, sr) * 0.4
    st = np.stack([s, s], 1)
    cut = 6000 * (80 / 6000) ** x

    def mask(times, freqs):
        c = np.interp(np.clip(times, 0, dur), t[::256], cut[::256])[:, None]
        return filter_response(freqs, c, 0.4, "lp")

    return stft_filter(st, sr, mask) * (1 - x ** 3)[:, None]


def _coin(dur, f0, sr, rng, p, beat_s):
    n = int(sr * max(dur, 0.3))
    a = int(0.07 * sr)
    y = np.zeros(n, np.float32)
    y[:a] = osc("square", f0, a, sr) * 0.5
    y[a:] = osc("square", f0 * 2 ** (5 / 12), n - a, sr) * 0.5 * np.exp(-_t(n - a, sr) / 0.15)
    return y


def _error(dur, f0, sr, rng, p, beat_s):
    n = int(sr * dur)
    y = np.zeros(n, np.float32)
    seg = int(n * 0.38)
    for k in range(2):
        a = int(k * n * 0.5)
        s = osc("square", f0, seg, sr) * 0.5 + osc("saw", f0 * 1.5, seg, sr) * 0.25
        y[a:a + seg] += fade_edges(s.astype(np.float32), sr, 0.002, 0.01)
    return y


def _reverse_cymbal(dur, f0, sr, rng, p, beat_s):
    from .drums import DRUM_DEFAULTS, one_shot

    params = dict(DRUM_DEFAULTS["crash"])
    params["decay"] = max(dur * 1.1, 0.5)
    c = np.array(one_shot("crash", params, 1.0, 1, sr))
    n = int(dur * sr)
    c = c[:n][::-1].copy()
    if len(c) < n:
        c = np.concatenate([np.zeros((n - len(c), 2), np.float32), c])
    return c


_FX = {
    "laser": _laser, "zap": _zap, "ufo": _ufo, "riser": _riser, "downlifter": _downlifter, "whoosh": _whoosh,
    "sweep": _sweep, "impact": _impact, "boom": _boom, "radio": _radio, "bleeps": _bleeps, "glitch": _glitch,
    "modem": _modem, "alarm": _alarm, "powerdown": _powerdown, "coin": _coin, "error": _error,
    "reverse_cymbal": _reverse_cymbal,
}


# ------------------------------------------------------------------------------- processing effects

def _xfade(n_total: int, sr: int, ms: float = 4.0):
    f = max(2, min(n_total // 2, int(sr * ms / 1000)))
    return f


def process(effect: str, buf: np.ndarray, a: int, b: int, sr: int, beat_s: float, params: dict,
            rng: np.random.Generator | None = None) -> None:
    """Apply a processing effect in place to ``buf[a:b]`` (stereo). Edges are crossfaded."""
    n = len(buf)
    a, b = max(0, a), min(n, b)
    if b - a < 16:
        return
    seg = buf[a:b].copy()
    L = b - a
    p = params or {}
    if effect == "stutter":
        sl = max(16, int(float(p.get("slice", 0.25)) * beat_s * sr))
        src = seg[:sl].copy()
        decay = float(p.get("decay", 0.0))
        pitch = float(p.get("pitch", 0.0))
        gate = float(p.get("gate", 1.0))
        out = np.zeros_like(seg)
        k = 0
        pos = 0
        while pos < L:
            piece = src
            if pitch:
                ratio = 2 ** (pitch * k / 12)
                idx = np.arange(0, sl, ratio)
                piece = np.stack([np.interp(idx, np.arange(sl), src[:, c]) for c in range(2)], 1).astype(np.float32)
            m = min(len(piece), L - pos, max(16, int(len(piece) * gate)))
            g = (1 - decay) ** k
            out[pos:pos + m] = fade_edges(piece[:m].copy(), sr, 0.001, 0.003) * g
            pos += len(piece) if gate >= 1 else max(int(sl), 16)
            k += 1
        new = out
    elif effect == "tapestop":
        curve = float(p.get("curve", 1.5))
        x = np.arange(L, dtype=np.float64) / L
        rate = (1 - x) ** curve
        pos = np.cumsum(rate)
        pos -= pos[0]
        new = np.stack([np.interp(pos, np.arange(L), seg[:, c]) for c in range(2)], 1).astype(np.float32)
        # darken as it slows
        new *= np.minimum(1.0, rate * 4)[:, None].astype(np.float32)
    elif effect == "crush":
        bits = float(p.get("bits", 6))
        ds = max(1, int(p.get("downsample", 8)))
        q = 2 ** (bits - 1)
        held = np.repeat(seg[::ds], ds, axis=0)[:L]
        crushed = np.round(held * q) / q
        mix = float(p.get("mix", 1.0))
        new = (seg * (1 - mix) + crushed * mix).astype(np.float32)
    elif effect == "filter":
        ftype = p.get("mode", p.get("type", "lp"))
        f_from = float(p.get("from_hz", 200 if ftype != "hp" else 20))
        f_to = float(p.get("to_hz", 12000 if ftype != "hp" else 2000))
        res = float(p.get("resonance", 0.3))
        dur = L / sr

        def mask(times, freqs):
            x = np.clip(times / dur, 0, 1)[:, None]
            c = f_from * (f_to / f_from) ** x
            return filter_response(freqs, np.clip(c, 20, 0.47 * sr), res, ftype, 12 if ftype == "hp" else 24)

        new = stft_filter(seg, sr, mask)
    elif effect == "gate":
        steps = str(p.get("steps", "x.x.x.x.x.x.x.x.")).replace("|", "").replace(" ", "")
        grid = float(p.get("grid", 16))
        step_len = beat_s * 4 / grid
        t = np.arange(L) / sr
        idx = (t / step_len).astype(np.int64) % max(1, len(steps))
        on = np.array([c not in ".-_" for c in steps], dtype=np.float32)[idx]
        # smooth the gate edges (3 ms)
        k = max(1, int(0.003 * sr))
        on = np.convolve(on, np.ones(k) / k, mode="same").astype(np.float32)
        new = seg * on[:, None]
    elif effect == "reverse":
        sl = p.get("slice")
        if sl:
            m = max(16, int(float(sl) * beat_s * sr))
            new = seg.copy()
            for s in range(0, L, m):
                new[s:s + m] = fade_edges(seg[s:s + m][::-1].copy(), sr, 0.002, 0.002)
        else:
            new = seg[::-1].copy()
    elif effect == "telephone":
        band = fft_filter(seg, sr, lambda f: 1.0 / np.sqrt(1.0 + (380.0 / np.maximum(f, 1.0)) ** 4)
                          / np.sqrt(1.0 + (f / 3200.0) ** 4) * (1 + 0.8 * np.exp(-0.5 * ((f - 1500) / 500) ** 2)))
        new = saturate(band * 1.5, 0.3)
        q = 2 ** 7
        new = (np.round(new * q) / q).astype(np.float32)
    else:
        raise ValueError(f"unknown effect {effect!r}")
    # crossfade at the edges
    f = _xfade(L, sr)
    ramp = np.linspace(0, 1, f, dtype=np.float32)[:, None]
    new[:f] = seg[:f] * (1 - ramp) + new[:f] * ramp
    new[L - f:] = new[L - f:] * (1 - ramp) + seg[L - f:] * ramp
    buf[a:b] = new


def apply_mute(buf: np.ndarray, a: int, b: int, sr: int, fade_out_ms: float = 4.0, fade_in_ms: float = 2.0):
    """Silence buf[a:b] with a short fade into the gap (sudden stop) and a short fade back in."""
    n = len(buf)
    if b <= 0 or a >= n:
        return
    fo = max(1, int(sr * fade_out_ms / 1000))
    fi = max(1, int(sr * fade_in_ms / 1000))
    s0 = max(0, a)
    s1 = min(n, b)
    if s0 - fo >= 0 and a >= 0:
        buf[s0 - fo:s0] *= np.linspace(1, 0, fo, dtype=np.float32)[:, None]
    buf[s0:s1] = 0.0
    if s1 < n and b >= 0:
        e = min(n, s1 + fi)
        buf[s1:e] *= np.linspace(0, 1, e - s1, dtype=np.float32)[:, None]

