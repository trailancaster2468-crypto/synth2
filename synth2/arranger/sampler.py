"""Sample player: regions of sampled audio played back at any pitch, polyphonic and offline.

A :class:`SampleInstrument` is a list of :class:`Region` objects (one sample each, with a
key and velocity range, tuning, loop, envelopes, filter, pan ...), however it was loaded
(SoundFont 2, SFZ or a folder of WAV files).  :func:`render_note` returns the stereo
audio of one note; the arranger adds notes into its stem buffers.

Everything is vectorised numpy and deterministic: the same call returns the same samples.

Resampling: the part of a sample a note needs is prepared once per (sample, loop,
transposition band) and cached: low-passed below the new Nyquist when the note plays
above the sample's own pitch or rate (no aliasing), oversampled 2x, and then read with
4-point Hermite interpolation.  A note at the sample's own pitch and rate is copied.
Loops stay continuous through the filter (the prepared audio carries one extra pass of
the loop in its steady state), and every tap of the interpolator wraps on its own.
"""
from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from typing import Callable

import numpy as np

from . import dsp
from .samples import Sample

# oversample: 1 or 2 (2x band-limited oversampling before the 4-point interpolator);
# cache_mb: memory for prepared sample audio (least recently used entries are dropped)
QUALITY = {"oversample": 2, "cache_mb": 512}
LEVELS_PER_OCTAVE = 4  # transposition bands of the anti-aliasing filter
MIN_RELEASE = 0.004  # s; shorter releases click
OFF_TIME = 0.006  # s; fade of a choked or stolen voice
SILENCE_DB = -90.0
LOOP_MODES = ("no_loop", "one_shot", "loop_continuous", "loop_sustain")
FILTER_TYPES = ("lpf_1p", "lpf_2p", "lpf_4p", "lpf_6p", "hpf_1p", "hpf_2p", "hpf_4p", "bpf_2p")


@dataclass
class Envelope:
    """Delay, attack, hold, decay, sustain, release. Times in seconds, ``sustain`` a linear
    level 0..1.

    ``shape`` selects the curves after the linear attack: ``"exp"`` (SFZ: exponential decay
    to the sustain level and exponential release, about -78 dB after the stated time),
    ``"db"`` (SoundFont volume envelope: decay and release fall linearly in dB, the stated
    time being a fall of 100 dB) or ``"lin"`` (SoundFont modulation envelope: linear, the
    stated time being a fall from 1 to 0)."""

    delay: float = 0.0
    attack: float = 0.0
    hold: float = 0.0
    decay: float = 0.0
    sustain: float = 1.0
    release: float = 0.01
    shape: str = "exp"

    def at(self, t: np.ndarray, gate: float | None) -> np.ndarray:
        """Envelope level at times ``t`` (s) for a key released at ``gate`` (None = held)."""
        t = np.asarray(t, dtype=np.float64)
        if gate is None:
            return self._held(t)
        y = self._held(np.minimum(t, gate))
        m = t > gate
        if m.any():
            level = float(self._held(np.array([gate]))[0])
            u = t[m] - gate
            r = max(self.release, 1e-4)
            if self.shape == "db":
                y[m] = level * 10.0 ** (-5.0 * u / r)
            elif self.shape == "lin":
                y[m] = np.maximum(level - u / r, 0.0)
            else:
                y[m] = level * np.exp(-9.0 * u / r)
        return y

    def _held(self, x: np.ndarray) -> np.ndarray:
        d, a = self.delay, self.attack
        t1 = d + a
        t2 = t1 + self.hold
        s = min(max(self.sustain, 0.0), 1.0)
        y = np.zeros_like(x)
        if a > 0:
            m = (x >= d) & (x < t1)
            y[m] = (x[m] - d) / a
        y[x >= t1] = 1.0
        m = x > t2
        if m.any():
            u = x[m] - t2
            dec = self.decay
            if dec <= 1e-5:
                y[m] = s
            elif self.shape == "db":
                y[m] = np.maximum(10.0 ** (-5.0 * u / dec), s)
            elif self.shape == "lin":
                y[m] = np.maximum(1.0 - u / dec, s)
            else:
                y[m] = s + (1.0 - s) * np.exp(-9.0 * u / dec)
        return y

    def end(self, gate: float | None) -> float | None:
        """Seconds until the level has fallen to silence, None when it never does (held with
        a sustain level above zero)."""
        floor = 10.0 ** (SILENCE_DB / 20.0)
        t2 = self.delay + self.attack + self.hold
        natural = None
        if self.sustain <= floor:
            if self.decay <= 1e-5:
                natural = t2
            elif self.shape == "db":
                natural = t2 + self.decay * (-SILENCE_DB / 100.0)
            elif self.shape == "lin":
                natural = t2 + self.decay
            else:
                natural = t2 + self.decay * 1.15
        if gate is None:
            return natural
        level = float(self._held(np.array([gate]))[0])
        r = max(self.release, 1e-4)
        if level <= floor:
            rel = gate
        elif self.shape == "db":
            rel = gate + r * max(0.0, (20.0 * math.log10(level) - SILENCE_DB) / 100.0)
        elif self.shape == "lin":
            rel = gate + r * level
        else:
            rel = gate + r
        return rel if natural is None else min(rel, natural)


@dataclass
class Region:
    """One sample and how it plays. Units follow SFZ: cents, dB, Hz, seconds; pan -1..1."""

    sample: Sample
    lokey: int = 0
    hikey: int = 127
    lovel: int = 0
    hivel: int = 127
    root: int = 60  # the key the sample sounds at its recorded pitch
    tune: float = 0.0  # cents, added at every key
    keytrack: float = 100.0  # cents of pitch per key (0 = the same pitch on every key)
    start: int = 0  # first frame played
    end: int = -1  # one past the last frame played (-1 = the whole sample)
    loop_mode: str = "no_loop"
    loop_start: int = 0
    loop_end: int = 0  # exclusive
    volume: float = 0.0  # dB
    pan: float = 0.0
    amp_veltrack: float = 1.0  # 0 = velocity does not change loudness, 1 = full
    vel_curve: float = 2.0  # loudness = (velocity / 127) ** vel_curve
    vel_points: tuple | None = None  # ((velocity, gain 0..1), ...) to interpolate instead of the power curve
    amp_keytrack: float = 0.0  # dB per key above amp_keycenter
    amp_keycenter: int = 60
    ampeg: Envelope = field(default_factory=Envelope)
    ampeg_vel2attack: float = 0.0  # s added to the attack at full velocity (negative = faster)
    ampeg_key2hold: float = 0.0  # timecents of hold per key below 60 (SoundFont)
    ampeg_key2decay: float = 0.0  # timecents of decay per key below 60 (SoundFont)
    cutoff: float | None = None  # Hz; None = no filter
    resonance: float = 0.0  # dB of gain at the cutoff frequency (0 = no peak to speak of)
    fil_type: str = "lpf_2p"
    fil_keytrack: float = 0.0  # cents of cutoff per key above fil_keycenter
    fil_keycenter: int = 60
    fil_veltrack: float = 0.0  # cents of cutoff at full velocity
    fileg: Envelope | None = None
    fileg_depth: float = 0.0  # cents at the envelope's peak
    pitcheg: Envelope | None = None
    pitcheg_depth: float = 0.0  # cents
    vib_depth: float = 0.0  # cents
    vib_freq: float = 5.0  # Hz
    vib_delay: float = 0.0  # s
    vib_fade: float = 0.0  # s
    trem_depth: float = 0.0  # dB
    trem_freq: float = 5.0
    trem_delay: float = 0.0
    group: int = 0  # choke group this region belongs to
    off_by: int = 0  # the group whose notes silence this region
    off_time: float = OFF_TIME
    seq_length: int = 1  # round robin: plays on every seq_length-th note ...
    seq_position: int = 1  # ... at this position (1-based)
    lorand: float = 0.0  # random layers: plays when the note's random number is in [lorand, hirand)
    hirand: float = 1.0
    trigger: str = "attack"  # "attack" | "release" (a sample played when the key is released)
    rt_decay: float = 0.0  # dB per second a release sample loses for the time the key was held
    delay: float = 0.0  # s before the region starts
    # crossfades (SFZ): silent up to the first value, full from the second, full up to the third, silent from
    # the fourth; over velocities and over keys
    xf_vel: tuple | None = None
    xf_key: tuple | None = None
    fixed_key: int | None = None  # SoundFont keynum: the region always plays as this key
    fixed_vel: int | None = None
    # (key, velocity) -> the region with its per-note modulation applied (SoundFont modulators)
    resolve: Callable[[int, int], "Region"] | None = field(default=None, repr=False, compare=False)

    def frames(self) -> int:
        return self.sample.frames if self.end < 0 else min(self.end, self.sample.frames)

    def looped(self) -> bool:
        return (self.loop_mode in ("loop_continuous", "loop_sustain")
                and 0 <= self.loop_start < self.loop_end - 1 and self.loop_end <= self.frames())


@dataclass
class SampleInstrument:
    """A playable set of regions. ``gain`` is a linear output trim (set by level matching),
    ``info`` says where it came from (format, file, preset, bank, program), ``warnings`` what
    the loader could not honour."""

    name: str
    regions: list[Region]
    gain: float = 1.0
    drums: bool = False
    info: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    _index: dict = field(default_factory=dict, repr=False, compare=False)

    def select(self, key: int, vel: int, seq: int = 0, rand: float = 0.0, trigger: str = "attack") -> list[Region]:
        """The regions that sound for this key and velocity. ``seq`` counts earlier notes on
        the same key (round robin); ``rand`` in [0, 1) picks among random layers."""
        key, vel = int(key), int(min(127, max(1, vel)))
        cands = self._index.get((key, trigger))
        if cands is None:
            cands = [r for r in self.regions if r.trigger == trigger and r.lokey <= key <= r.hikey]
            self._index[(key, trigger)] = cands
        out = []
        for r in cands:
            if not r.lovel <= vel <= r.hivel:
                continue
            if r.seq_length > 1 and (seq % r.seq_length) + 1 != r.seq_position:
                continue
            if not (r.lorand <= rand < r.hirand or (rand >= r.hirand >= 1.0)):
                continue
            out.append(r)
        return out

    def keys(self) -> list[int]:
        """Keys that have at least one attack region."""
        have = np.zeros(128, bool)
        for r in self.regions:
            if r.trigger == "attack":
                have[max(0, r.lokey):min(127, r.hikey) + 1] = True
        return [int(k) for k in np.flatnonzero(have)]

    def with_regions(self, regions: list[Region], **changes) -> "SampleInstrument":
        return replace(self, regions=regions, _index={}, **changes)


# ----------------------------------------------------------------------------- prepared audio

@dataclass
class _Prepared:
    audio: np.ndarray  # float32 (m, channels), in oversampled frames
    over: int  # oversampling factor of ``audio``
    covered: int  # sample frames [0, covered) are present
    wrap_base: int  # sample frame the loop wraps to (in the layout of ``audio``)
    tail_shift: int  # frames between the loop end and the audio after the loop


class _Cache:
    """Prepared sample audio, least recently used first out, bounded in bytes."""

    def __init__(self):
        self._d: "OrderedDict[tuple, _Prepared]" = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()

    def get(self, key):
        with self._lock:
            p = self._d.get(key)
            if p is not None:
                self._d.move_to_end(key)
            return p

    def put(self, key, p: _Prepared):
        limit = int(QUALITY.get("cache_mb", 512)) << 20
        with self._lock:
            old = self._d.pop(key, None)
            if old is not None:
                self._bytes -= old.audio.nbytes
            self._d[key] = p
            self._bytes += p.audio.nbytes
            while self._bytes > limit and len(self._d) > 1:
                _k, v = self._d.popitem(last=False)
                self._bytes -= v.audio.nbytes

    def clear(self):
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def nbytes(self) -> int:
        return self._bytes


_CACHE = _Cache()


def clear_cache() -> None:
    """Drop all prepared sample audio."""
    _CACHE.clear()


def cache_bytes() -> int:
    return _CACHE.nbytes()


def _fast_len(n: int) -> int:
    """The next FFT length >= n of the form 2^a * {1, 3, 5, 9, 15}."""
    best = 1 << max(1, (n - 1).bit_length())
    for m in (3, 5, 9, 15):
        p = m
        while p < n:
            p <<= 1
        best = min(best, p)
    return best


def _fft_process(x: np.ndarray, level: int, over: int) -> np.ndarray:
    """Low-pass ``x`` (frames, channels) below the Nyquist of its transposition band and
    oversample it by ``over``; returns ``len(x) * over`` frames."""
    n = len(x)
    size = _fast_len(n + 96)
    spec = np.fft.rfft(x, size, axis=0)
    f = np.arange(spec.shape[0], dtype=np.float64) / size  # cycles per sample
    if level > 0:
        fc = 0.5 / 2.0 ** (level / LEVELS_PER_OCTAVE)
        lo = 0.88 * fc
    else:
        fc, lo = 0.5, 0.485
    g = np.clip((fc - f) / (fc - lo), 0.0, 1.0)
    g = 0.5 - 0.5 * np.cos(np.pi * g)
    spec *= g[:, None]
    if over > 1:
        big = np.zeros((size * over // 2 + 1, x.shape[1]), dtype=spec.dtype)
        big[: spec.shape[0]] = spec
        y = np.fft.irfft(big, size * over, axis=0) * over
    else:
        y = np.fft.irfft(spec, size, axis=0)
    return np.ascontiguousarray(y[: n * over], dtype=np.float32)


_PAD = 4  # zero frames after the audio so interpolator taps past the end read silence
_MAX_UNROLL = 1 << 18  # loops longer than this are filtered in place
_MARGIN = 256  # extra frames read past a prefix so the filter's edge stays outside it


def _prepare(reg: Region, need: int, level: int, over: int) -> _Prepared:
    """Prepared audio of a region's sample covering frames [0, need) (all of it when the
    region loops)."""
    smp = reg.sample
    end = reg.frames()
    looped = reg.looped()
    ls, le = reg.loop_start, reg.loop_end
    stop = end if reg.loop_mode != "loop_continuous" or not looped else le
    processed = level > 0 or over > 1
    key = (smp.key, ls if looped else -1, le if looped else -1, stop if looped else -1, level, over)
    hit = _CACHE.get(key)
    if hit is not None and (looped or hit.covered >= min(need, stop)):
        return hit
    if looped:
        covered = stop
    else:
        covered = min(stop, max(need, 2 * (hit.covered if hit is not None else 0), 16384))
    pad = np.zeros((_PAD * over, smp.channels), np.float32)
    if not processed:
        audio = np.concatenate([smp.read(0, covered), pad])
        prep = _Prepared(audio, 1, covered, ls, 0)
    elif looped and le - ls <= _MAX_UNROLL:
        length = le - ls
        guard = min(length, _MARGIN)
        head = smp.read(0, le)
        loop = head[ls:le]
        y = _fft_process(np.concatenate([head, loop, loop[:guard]]), level, over)[: (le + length) * over]
        parts = [y]
        if stop > le:
            lead = min(le, _MARGIN)
            tail = _fft_process(smp.read(le - lead, stop), level, over)[lead * over:]
            parts.append(tail)
        parts.append(pad)
        prep = _Prepared(np.concatenate(parts), over, covered, le, length)
    else:
        extra = min(_MARGIN, smp.frames - covered) if covered < smp.frames else 0
        y = _fft_process(smp.read(0, covered + extra), level, over)[: covered * over]
        prep = _Prepared(np.concatenate([y, pad]), over, covered, ls, 0)
    prep.audio.setflags(write=False)
    _CACHE.put(key, prep)
    return prep


def _hermite(audio: np.ndarray, idx: list[np.ndarray], frac: np.ndarray) -> np.ndarray:
    """4-point, third-order Hermite interpolation; ``idx`` are the tap indices for -1, 0, 1, 2."""
    ym, y0, y1, y2 = (audio[i] for i in idx)
    f = frac[:, None]
    c1 = 0.5 * (y1 - ym)
    c2 = ym - 2.5 * y0 + 2.0 * y1 - 0.5 * y2
    c3 = 0.5 * (y2 - ym) + 1.5 * (y0 - y1)
    return ((c3 * f + c2) * f + c1) * f + y0


# ------------------------------------------------------------------------------------ filters

def _q_of(resonance_db: float) -> float:
    return 10.0 ** (max(-6.0, min(resonance_db, 40.0)) / 20.0)


def _biquad_response(kind: str, fc: float, q: float, w: np.ndarray) -> np.ndarray:
    """Complex response of the filter at angular frequencies ``w`` (rad/sample)."""
    w0 = 2.0 * math.pi * fc
    z1 = np.exp(-1j * w)
    poles = {"1p": 1, "2p": 2, "4p": 4, "6p": 6}[kind[-2:]]
    base = kind[:3]
    if poles == 1:
        p = math.exp(-w0)
        if base == "lpf":
            return (1.0 - p) / (1.0 - p * z1)
        return 0.5 * (1.0 + p) * (1.0 - z1) / (1.0 - p * z1)
    cs, sn = math.cos(w0), math.sin(w0)
    alpha = sn / (2.0 * q)
    if base == "lpf":
        b = ((1 - cs) / 2, 1 - cs, (1 - cs) / 2)
    elif base == "hpf":
        b = ((1 + cs) / 2, -(1 + cs), (1 + cs) / 2)
    else:
        b = (alpha, 0.0, -alpha)
    z2 = z1 * z1
    h = (b[0] + b[1] * z1 + b[2] * z2) / ((1 + alpha) - 2 * cs * z1 + (1 - alpha) * z2)
    if poles > 2:
        plain = _biquad_response(base + "_2p", fc, 0.7071, w)
        h = h * plain ** (poles // 2 - 1)
    return h


def static_filter(y: np.ndarray, sr: int, kind: str, cutoff: float, resonance_db: float) -> np.ndarray:
    """The causal recursive filter of ``kind`` applied through one FFT (same output as the
    difference equation; the note is zero-padded for the filter's ringing)."""
    n = len(y)
    fc = min(max(cutoff, 16.0), 0.49 * sr) / sr
    q = _q_of(resonance_db)
    ring = int(min(8192, max(256, 14.0 * q / (2 * math.pi * fc))))
    size = _fast_len(n + ring)
    spec = np.fft.rfft(y, size, axis=0)
    w = 2.0 * math.pi * np.arange(spec.shape[0], dtype=np.float64) / size
    h = _biquad_response(kind, fc, q, w)
    spec *= h[:, None] if y.ndim == 2 else h
    return np.fft.irfft(spec, size, axis=0)[:n].astype(np.float32)


def _magnitude(kind: str, freqs: np.ndarray, cutoff: np.ndarray, q: float) -> np.ndarray:
    """Magnitude (frames x bins) of the filter for a cutoff per frame (shape (F, 1))."""
    r = np.maximum(freqs[None, :], 1e-3) / cutoff
    poles = {"1p": 1, "2p": 2, "4p": 4, "6p": 6}[kind[-2:]]
    base = kind[:3]
    if base == "hpf":
        r = 1.0 / r
    if base == "bpf":
        return 1.0 / np.sqrt(1.0 + (q * (r - 1.0 / r)) ** 2)
    if poles == 1:
        return 1.0 / np.sqrt(1.0 + r * r)
    r2 = r * r
    h = 1.0 / np.sqrt((1.0 - r2) ** 2 + r2 / (q * q))
    if poles > 2:
        h = h * (1.0 / np.sqrt((1.0 - r2) ** 2 + r2 / 0.5)) ** (poles // 2 - 1)
    return h


# -------------------------------------------------------------------------------------- notes

def midi_velocity(velocity: float) -> float:
    """The arranger's velocity (0.8 = a normal hit, 1.0 = an accent) as a MIDI velocity 1..127."""
    return float(min(127.0, max(1.0, velocity * 115.0)))


def _lfo(t: np.ndarray, freq: float, delay: float, fade: float) -> np.ndarray:
    u = t - delay
    y = np.sin(2.0 * math.pi * freq * np.maximum(u, 0.0))
    if fade > 0:
        y *= np.clip(u / fade, 0.0, 1.0)
    return y


def _region_gain(reg: Region, key: int, vel: float) -> float:
    v = min(max(vel, 1.0), 127.0) / 127.0
    track = min(max(reg.amp_veltrack, -1.0), 1.0)
    if reg.vel_points is not None:
        xs, ys = zip(*reg.vel_points)
        g = (1.0 - abs(track)) + abs(track) * float(np.interp(v * 127.0, xs, ys))
    elif track >= 0:
        g = (1.0 - track) + track * v ** reg.vel_curve
    else:
        g = (1.0 + track) - track * (1.0 - v) ** reg.vel_curve
    db = reg.volume + reg.amp_keytrack * (key - reg.amp_keycenter)
    for xf, x in ((reg.xf_vel, vel), (reg.xf_key, float(key))):
        if xf is not None:
            in_lo, in_hi, out_lo, out_hi = xf
            if in_hi > in_lo:
                g *= math.sqrt(min(1.0, max(0.0, (x - in_lo) / (in_hi - in_lo))))
            if out_hi > out_lo:
                g *= math.sqrt(min(1.0, max(0.0, (out_hi - x) / (out_hi - out_lo))))
    return g * 10.0 ** (db / 20.0)


def render_region(reg: Region, key: int, vel: float, gate: float, sr: int, *, pitch: float | None = None,
                  cut: float | None = None, bend: np.ndarray | None = None, max_seconds: float = 60.0
                  ) -> np.ndarray:
    """Stereo float32 audio of one region for one note.

    ``key`` selects tracking (pitch, filter, level), ``pitch`` is the sounding pitch as a
    MIDI number when it differs from the key, ``vel`` the MIDI velocity 1..127, ``gate`` the
    seconds the key is held, ``cut`` the seconds after which the voice is choked, ``bend`` an
    array of semitone offsets per output sample (shorter arrays hold their last value).
    """
    if reg.resolve is not None:
        reg = reg.resolve(int(key), int(round(min(127.0, max(1.0, vel)))))
    if reg.fixed_key is not None:
        key = reg.fixed_key
        pitch = None
    if reg.fixed_vel is not None:
        vel = float(reg.fixed_vel)
    pitch = float(key if pitch is None else pitch)
    smp = reg.sample
    end = reg.frames()
    start = min(max(0, reg.start), max(0, end - 1))
    if end - start < 2:
        return np.zeros((0, 2), np.float32)
    one_shot = reg.loop_mode == "one_shot"
    looped = reg.looped()
    gate = max(float(gate), 0.0)
    env = reg.ampeg
    if not one_shot and env.release < MIN_RELEASE:
        env = replace(env, release=MIN_RELEASE)
    if reg.ampeg_vel2attack:
        env = replace(env, attack=max(0.0, env.attack + reg.ampeg_vel2attack * vel / 127.0))
    if reg.ampeg_key2hold or reg.ampeg_key2decay:
        env = replace(env, hold=env.hold * 2.0 ** (reg.ampeg_key2hold * (60 - key) / 1200.0),
                      decay=env.decay * 2.0 ** (reg.ampeg_key2decay * (60 - key) / 1200.0))
    env_gate = None if one_shot else gate
    cents0 = (pitch - reg.root) * reg.keytrack + reg.tune
    ratio0 = 2.0 ** (cents0 / 1200.0) * smp.rate / sr

    # length of the voice in output samples
    t_end = env.end(env_gate)
    if t_end is None:
        t_end = max_seconds
    if cut is not None:
        t_end = min(t_end, max(cut, 0.0) + reg.off_time)
    if not looped or (reg.loop_mode == "loop_sustain" and env_gate is not None):
        # a region that runs to the end of its sample: an upper bound from the slowest playback
        low = ratio0 * 2.0 ** (-(abs(reg.vib_depth) + abs(reg.pitcheg_depth) + 1200.0 * (bend is not None)) / 1200.0)
        avail = (end - start) / max(low, 1e-6) / sr
        if looped:
            avail += gate
        t_end = min(t_end, avail)
    n = int(min(t_end, max_seconds) * sr) + 1
    if n < 2:
        return np.zeros((0, 2), np.float32)
    t = np.arange(n, dtype=np.float64) / sr

    # pitch curve
    cents = None
    if reg.vib_depth:
        cents = reg.vib_depth * _lfo(t, reg.vib_freq, reg.vib_delay, reg.vib_fade)
    if reg.pitcheg is not None and reg.pitcheg_depth:
        c = reg.pitcheg_depth * reg.pitcheg.at(t, env_gate)
        cents = c if cents is None else cents + c
    if bend is not None and len(bend):
        b = np.asarray(bend, dtype=np.float64)
        b = b[:n] if len(b) >= n else np.concatenate([b, np.full(n - len(b), b[-1])])
        cents = 100.0 * b if cents is None else cents + 100.0 * b
    if cents is None:
        pos = start + ratio0 * np.arange(n, dtype=np.float64)
        ratio_max = ratio0
    else:
        ratio = ratio0 * np.exp2(cents / 1200.0)
        pos = np.empty(n, dtype=np.float64)
        pos[0] = start
        np.cumsum(ratio[:-1], out=pos[1:])
        pos[1:] += start
        ratio_max = float(ratio.max())

    # positions through the loop
    ls, le = reg.loop_start, reg.loop_end
    rel_i = None
    if looped:
        length = le - ls
        if reg.loop_mode == "loop_sustain" and env_gate is not None:
            rel_i = min(n, int(round(gate * sr)))
            if rel_i < n:
                p = pos[rel_i]
                first = ls + (p - le) % length if p >= le else p
                pos[rel_i:] = first + (pos[rel_i:] - p)
    if not looped or rel_i is not None:
        live = int(np.searchsorted(pos, end - 1e-9)) if rel_i is None else \
            (rel_i + int(np.searchsorted(pos[rel_i:], end - 1e-9)))
        if live < n:
            n = live
            if n < 2:
                return np.zeros((0, 2), np.float32)
            pos, t = pos[:n], t[:n]
            if rel_i is not None:
                rel_i = min(rel_i, n)

    # read the sample
    exact = (cents is None and abs(ratio0 - 1.0) < 1e-9 and not looped and float(start).is_integer())
    if exact:
        raw = smp.read(start, start + n)
        y = raw if raw.shape[1] == 2 else np.repeat(raw, 2, axis=1)
        mono = raw.shape[1] == 1
    else:
        level = 0
        if ratio_max > 1.003:
            level = int(math.ceil(LEVELS_PER_OCTAVE * math.log2(ratio_max) - 1e-9))
        over = 2 if int(QUALITY.get("oversample", 2)) >= 2 else 1
        need = end if looped else min(end, int(pos[-1]) + 3)
        prep = _prepare(reg, need, level, over)
        audio = prep.audio
        u = pos * prep.over
        k = np.floor(u).astype(np.int64)
        frac = (u - k).astype(np.float32)
        top = len(audio) - 1
        taps = []
        for j in (-1, 0, 1, 2):
            idx = k + j
            if looped:
                lo, ln = le * prep.over, (le - ls) * prep.over
                base, shift = prep.wrap_base * prep.over, prep.tail_shift * prep.over
                if rel_i is None:
                    idx = np.where(idx >= lo, base + (idx - lo) % ln, idx)
                else:
                    a = idx[:rel_i]
                    idx[:rel_i] = np.where(a >= lo, base + (a - lo) % ln, a)
                    b = idx[rel_i:]
                    idx[rel_i:] = np.where(b >= lo, b + shift, b)
            taps.append(np.clip(idx, 0, top))
        raw = _hermite(audio, taps, frac)
        mono = raw.shape[1] == 1
        y = raw if not mono else np.repeat(raw, 2, axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)

    # filter
    if reg.cutoff is not None and reg.fil_type in FILTER_TYPES:
        off = reg.fil_keytrack * (key - reg.fil_keycenter) + reg.fil_veltrack * min(vel, 127.0) / 127.0
        moving = reg.fileg is not None and abs(reg.fileg_depth) > 1.0
        top_hz = min(0.45 * sr, 19000.0)
        if not moving:
            fc = reg.cutoff * 2.0 ** (off / 1200.0)
            if not (reg.fil_type.startswith("lpf") and fc >= top_hz and reg.resonance < 1.0):
                y = static_filter(y, sr, reg.fil_type, fc, reg.resonance)
        else:
            q = _q_of(reg.resonance)
            fileg, depth, base, kind = reg.fileg, reg.fileg_depth, reg.cutoff, reg.fil_type

            def mask(frame_t, freqs):
                c = base * np.exp2((off + depth * fileg.at(np.maximum(frame_t, 0.0), env_gate)) / 1200.0)
                c = np.clip(c, 16.0, 0.49 * sr)
                return _magnitude(kind, freqs, c[:, None], q).astype(np.float32)

            y = dsp.stft_filter(y, sr, mask)

    # level
    amp = env.at(t, env_gate)
    if reg.trem_depth:
        amp = amp * 10.0 ** (reg.trem_depth * _lfo(t, reg.trem_freq, reg.trem_delay, 0.0) / 20.0)
    g = _region_gain(reg, key, vel)
    pan = min(max(reg.pan, -1.0), 1.0)
    if mono:
        gl, gr = dsp.pan_gains(pan)
    else:
        gl, gr = min(1.0, 1.0 - pan), min(1.0, 1.0 + pan)
    y = y[:n] * (amp.astype(np.float32)[:, None] * np.array([g * gl, g * gr], np.float32))
    if cut is not None:
        c = int(max(cut, 0.0) * sr)
        if c < n:
            f = max(2, min(n - c, int(reg.off_time * sr)))
            y[c:c + f] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
            y = y[:c + f]
            n = len(y)
    f = min(n // 2, max(2, int(0.001 * sr)))
    if f > 1 and float(np.abs(y[-1]).max()) > 1e-5:
        y[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)[:, None]
    if reg.delay > 0:
        y = np.concatenate([np.zeros((int(reg.delay * sr), 2), np.float32), y])
    return y


def _mix(parts: list[tuple[int, np.ndarray]]) -> np.ndarray:
    n = max((s + len(y) for s, y in parts), default=0)
    out = np.zeros((n, 2), np.float32)
    for s, y in parts:
        out[s:s + len(y)] += y
    return out


def render_note(inst: SampleInstrument, key: int, velocity: float, gate: float, sr: int, *,
                pitch: float | None = None, seq: int = 0, rand: float = 0.0, cut: float | None = None,
                bend: np.ndarray | None = None, regions: list[Region] | None = None) -> np.ndarray:
    """Stereo float32 audio of one note: every region the key and velocity select, mixed.

    ``velocity`` is a MIDI velocity 1..127 (fractions scale loudness smoothly; the layer is
    chosen by the rounded value), ``gate`` the seconds the key is held. ``seq`` and ``rand``
    choose round-robin and random layers. The audio runs until the release has died away.
    """
    vel = float(min(127.0, max(1.0, velocity)))
    layer = int(round(vel))
    regs = inst.select(key, layer, seq, rand) if regions is None else regions
    parts = []
    for r in regs:
        y = render_region(r, key, vel, gate, sr, pitch=pitch, cut=cut, bend=bend)
        if len(y):
            parts.append((0, y))
    if cut is None or cut > gate:
        for r in inst.select(key, layer, seq, rand, trigger="release"):
            rr = replace(r, loop_mode="one_shot", volume=r.volume - r.rt_decay * gate, trigger="attack")
            y = render_region(rr, key, vel, 0.0, sr, pitch=pitch)
            if len(y):
                parts.append((int(round(gate * sr)), y))
    out = _mix(parts)
    if inst.gain != 1.0:
        out *= np.float32(inst.gain)
    return out


def note_groups(regions: list[Region]) -> tuple[set[int], set[int]]:
    """(groups the regions belong to, groups that silence them)."""
    return {r.group for r in regions if r.group}, {r.off_by for r in regions if r.off_by}


def loudness(audio: np.ndarray, sr: int, window: float = 0.4) -> float:
    """RMS of the loudest ``window`` seconds of ``audio`` (0 for silence). A sound shorter than
    the window counts with the silence around it, so short and long sounds compare by what a
    listener integrates."""
    if len(audio) == 0:
        return 0.0
    p = (audio.astype(np.float64) ** 2).sum(axis=1) if audio.ndim == 2 else audio.astype(np.float64) ** 2
    w = max(1, int(window * sr))
    if len(p) <= w:
        return float(np.sqrt(p.sum() / w))
    c = np.concatenate([[0.0], np.cumsum(p)])
    return float(np.sqrt(max((c[w:] - c[:-w]).max() / w, 0.0)))
