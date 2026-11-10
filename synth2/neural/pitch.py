"""The bank's pitch predictor under Synth2's pitch controls.

A voicebank's ``dspitch/`` holds a linguistic encoder and a diffusion model that turn phonemes
with durations plus the score's notes into a sung pitch curve (MIDI semitones per frame). This
module holds the model-free half, a port of OpenUtau's ``DiffSingerPitch.cs``: the note-level
inputs (:func:`build_notes`), checks of a prediction (:func:`range_excess`, :func:`bridge`)
and :class:`BankPitch`, which lays a track's pitch controls over the predicted curve phrase by
phrase. The engine runs the model (``engine.Runner.predict_pitch``).

What the model's frame inputs mean (DiffSinger ``forward_pitch_preprocess``):

* The model builds its own base pitch from the notes (the steps smoothed over a few frames,
  :func:`base_pitch`) and predicts the deviation from it.
* ``retake`` true = generate this frame. False = keep it: the model is told the deviation
  ``pitch - base`` there and generates the other frames to join it.
* ``expr`` (0..1 per frame): 0 returns the base pitch; up to 0.5 the curve stays close to it;
  overshoot, preparation, drift and vibrato arrive between 0.5 and 1.

Frame layout: phonemes and lengths cover the whole phrase, head ``SP`` and tail ``SP``
included. Notes are given on that grid; :func:`build_notes` adds the rest notes the model
expects (before the first note, in gaps, after the last note).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

REST_SYMBOLS = ("SP", "AP")
PITCH_STEPS = 10  # OpenUtau's default number of diffusion steps for pitch

# How much of the learned pitch expression each Synth2 style asks for. Below 0.5 the predictor
# returns plain note steps; vibrato and drift appear from about 0.75.
STYLE_EXPR = {"pop": 1.0, "ballad": 1.0, "rock": 1.0, "jazz": 1.0, "rnb": 1.0, "classical": 1.0, "soft": 0.75,
              "robotic": 0.5}
MIN_EXPR = 0.5
SYNTHETIC_VIBRATO_STYLES = ("robotic",)
VIB_ATTRS = ("vibrato_depth", "vibrato_rate", "vibrato_start")
TRANS_ATTRS = ("transition", "overshoot")
MAX_DEV = 6.0  # semitones a predicted sung frame may lie outside the notes within 0.25 s
MAX_OUTLIERS = 0.1  # share of a phrase's sung frames that may be outliers before it is rejected
SUSTAIN = dict(min_note=0.35, head=0.12, tail=0.05, ramp=0.06)


@dataclass
class PhraseNote:
    """A sung note on the phrase's frame grid (frame 0 = first frame of the head ``SP``)."""

    start: int
    end: int
    midi: float  # MIDI note number, may be fractional (detune)
    slur: bool = False  # continues the previous note's syllable (Synth2 ``-``, OpenUtau ``+``)
    rest: bool | None = None  # None: decided from the phonemes under the note


@dataclass
class NoteInputs:
    midi: np.ndarray  # float32 [N]
    dur: np.ndarray  # int64 [N], sums to the phrase's frame count
    rest: np.ndarray  # bool [N]
    owner: np.ndarray  # int [N]: index into ``notes`` each padded note follows (retake masks)

    def frame_owner(self) -> np.ndarray:
        """Index of the real note each frame belongs to."""
        return np.repeat(self.owner, self.dur)

    def frame_midi(self) -> np.ndarray:
        """The unsmoothed note steps per frame (rests carry their neighbour's pitch)."""
        return np.repeat(self.midi, self.dur).astype(np.float64)

    def frame_rest(self) -> np.ndarray:
        return np.repeat(self.rest, self.dur)


def notes_from_timings(timings, f_start: int, frame_s: float) -> list[PhraseNote]:
    """Phrase notes from note timings in seconds.

    ``timings``: the phrase's pitched notes in order, objects with ``start`` and ``end`` (seconds
    on the same clock as the frame grid), ``pitch`` (MIDI) and optionally ``melisma`` (Synth2's
    planner ``NoteTiming``). ``f_start``: global frame index of the phrase's first frame (the
    head ``SP``); frame ``k`` is at ``k * frame_s`` seconds.
    """
    out = []
    for tm in timings:
        a = int(round(tm.start / frame_s)) - int(f_start)
        b = int(round(tm.end / frame_s)) - int(f_start)
        out.append(PhraseNote(a, max(b, a + 1), float(tm.pitch), slur=bool(getattr(tm, "melisma", False))))
    return out


def fit_duration_sum(durs: list[int], total: int) -> list[int]:
    """OpenUtau ``FitDurationSum``: the last entry absorbs the difference to ``total``."""
    out = list(durs)
    if not out:
        return out
    out[-1] += total - sum(out)
    if out[-1] < 0:
        deficit, out[-1] = -out[-1], 0
        for i in range(len(out) - 2, -1, -1):
            take = min(out[i], deficit)
            out[i] -= take
            deficit -= take
            if deficit <= 0:
                break
        if deficit > 0:
            raise ValueError(f"cannot fit note durations to {total} frames")
    return out


def build_notes(notes: list[PhraseNote], phones: list[str], ph_dur: list[int], is_vowel, tail: int | None = None
                ) -> NoteInputs:
    """Note-level model inputs for one phrase (OpenUtau ``DsPitch.Process``, note section).

    * a rest note from frame 0 to the first note (head ``SP`` and the first onset consonants);
    * a rest note for every gap between notes, owned by the note before it;
    * each note; a slur keeps the rest status of the note before it, any other note is a rest
      when no vowel sounds under it (``SP`` / ``AP`` / consonants only);
    * a rest note for the tail (``tail`` frames; the last entry is fitted to the frame count);
    * every run of rests takes the pitch of the nearest sung note (a run between two sung notes
      is split in the middle), or 60 when nothing is sung.
    """
    if not notes:
        raise ValueError("a phrase needs at least one note")
    total = int(sum(ph_dur))
    edges = np.concatenate([[0], np.cumsum(ph_dur)])
    midi, dur, rest, owner = [float(notes[0].midi)], [max(0, int(notes[0].start))], [True], [0]
    prev_end = int(notes[0].start)
    for i, nt in enumerate(notes):
        a, b = int(nt.start), int(nt.end)
        if a > prev_end:
            midi.append(float(nt.midi))
            dur.append(a - prev_end)
            rest.append(True)
            owner.append(max(i - 1, 0))
        elif a < prev_end:
            a = prev_end  # overlapping notes: a monophonic line keeps the earlier note's frames
        midi.append(float(nt.midi))
        dur.append(max(0, b - a))
        owner.append(i)
        if nt.rest is not None:
            rest.append(bool(nt.rest))
        elif nt.slur:
            rest.append(rest[-1])
        else:
            sung = [p for p, s, e in zip(phones, edges[:-1], edges[1:]) if e > a and s < b]
            rest.append(not any(p not in REST_SYMBOLS and is_vowel(p) for p in sung))
        prev_end = max(prev_end, b)
    midi.append(float(notes[-1].midi))
    dur.append(max(0, total - prev_end) if tail is None else int(tail))
    rest.append(True)
    owner.append(len(notes) - 1)

    n = len(rest)
    if all(rest):
        midi = [60.0] * n
    else:
        i = 0
        while i < n:
            if not rest[i]:
                i += 1
                continue
            j = i + 1
            while j < n and rest[j]:
                j += 1
            if i == 0:
                midi[:j] = [midi[j]] * j
            elif j == n:
                midi[i:] = [midi[i - 1]] * (n - i)
            else:
                mid = (i + j + 1) // 2
                midi[i:mid] = [midi[i - 1]] * (mid - i)
                midi[mid:j] = [midi[j]] * (j - mid)
            i = j
    dur = fit_duration_sum(dur, total)
    return NoteInputs(np.array(midi, np.float32), np.array(dur, np.int64), np.array(rest, bool),
                      np.array(owner, int))


def base_pitch(inputs: NoteInputs, width: int = 4) -> np.ndarray:
    """The model's base pitch: the frame-level note steps under a sine window (DiffSinger
    ``SinusoidalSmoothingConv1d``). The graph computes its own; this copy is for analysis and
    for building ``keep_pitch`` curves.

    ``width`` 4 frames is what Qixuan v2.7.0 does (measured: with ``expr`` 0 the model returns
    this curve within 4 cents). The window's first tap is zero, which leaves the three taps
    0.29, 0.41, 0.29: a step between frames ``k - 1`` and ``k`` reads 0.29 and 0.71 of the way
    there.
    """
    x = inputs.frame_midi()
    w = max(1, int(width))
    if w < 2 or len(x) < 2:
        return x
    k = np.sin(np.pi * np.arange(w) / w)
    k = k / k.sum()
    pad = w // 2
    xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.correlate(xp, k, mode="valid")


def range_excess(pred: np.ndarray, inputs: NoteInputs, frames: int = 22) -> np.ndarray:
    """Semitones by which each frame of ``pred`` lies outside the range of the note pitches within
    ``frames`` frames of it (0 inside the range). For a sanity check of a prediction: over five
    scores and three seeds Qixuan stays within 3 semitones (scoops into phrase starts)."""
    fm = inputs.frame_midi()
    w = max(0, int(frames))
    win = np.lib.stride_tricks.sliding_window_view(np.pad(fm, (w, w), mode="edge"), 2 * w + 1)
    return np.maximum(np.maximum(np.asarray(pred, float) - win.max(axis=1), win.min(axis=1) - pred), 0.0)


def slow_trend(x: np.ndarray, frames: int) -> np.ndarray:
    """Zero-phase Hann low-pass, reflect padded. A 0.4 s window (34 frames at hop 512 / 44.1 kHz)
    removes 5 Hz and faster, keeps what is slower than about 1.5 Hz."""
    w = int(frames) | 1
    if w < 3 or len(x) < w:
        return np.array(x, float)
    k = np.hanning(w + 2)[1:-1]
    k /= k.sum()
    return np.convolve(np.pad(np.asarray(x, float), (w // 2, w // 2), mode="reflect"), k, mode="valid")


def scale_fast(pred: np.ndarray, base: np.ndarray, gain, weight: np.ndarray, hop_s: float, trend_s: float = 0.4
               ) -> np.ndarray:
    """Scale the vibrato of a predicted curve: ``pred + (gain - 1) * weight * fast``.

    ``fast`` is ``pred - base`` minus its slow trend (vibrato and flutter, no drift, no note
    steps). ``gain`` scalar or per frame: 0 removes the vibrato, 1 returns ``pred`` unchanged, 2
    doubles it. ``weight`` (0..1 per frame) limits the change to the sustained part of notes.
    """
    dev = np.asarray(pred, float) - np.asarray(base, float)
    fast = dev - slow_trend(dev, int(round(trend_s / hop_s)))
    g = np.broadcast_to(np.asarray(gain, float), dev.shape)
    return np.asarray(pred, float) + (g - 1.0) * np.asarray(weight, float) * fast


def soft_window(mask: np.ndarray, ramp: int = 3) -> np.ndarray:
    """A 0/1 frame mask as a weight with cosine ramps of ``ramp`` frames inside each run of ones
    (0 outside the mask, 1 from ``ramp`` frames inside it)."""
    w = np.asarray(mask, float)
    if ramp < 1 or not w.any():
        return w
    k = np.hanning(2 * ramp + 1)
    k /= k.sum()
    lo = np.convolve(np.pad(w, (ramp, ramp), mode="edge"), k, mode="valid")
    return np.minimum(w, np.clip(2.0 * lo - 1.0, 0.0, 1.0))


# ---- Synth2's pitch controls over a predicted curve -----------------------------------------------
def default_expr(style, voice) -> float:
    """The ``expr`` a style asks for, lowered by the voice's ``robotic`` character."""
    e = STYLE_EXPR.get(style.name, 1.0) * (1.0 - 0.5 * float(voice.robotic))
    return float(np.clip(e, MIN_EXPR, 1.0))


def default_vibrato(style, track) -> str:
    """``"synthetic"`` when the vibrato is the user's design (robotic style, vibrato parameters
    on the track), else ``"auto"`` (the predicted vibrato, except on notes with vibrato attributes)."""
    params = getattr(track, "voice_params", {}) or {}
    if style.name in SYNTHETIC_VIBRATO_STYLES or any("vibrato" in k for k in params):
        return "synthetic"
    return "auto"


def _ramp(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def sustain_weight(t, timings, min_note, head, tail, ramp, pick=None) -> np.ndarray:
    """1 on the sustained part of notes at least ``min_note`` s long (and chosen by ``pick``)."""
    w = np.zeros(len(t))
    for nt in timings:
        if nt.end - nt.start < min_note or (pick is not None and not pick(nt)):
            continue
        a, b = nt.start + head, nt.end - tail
        if b - a >= 2 * ramp:
            w = np.maximum(w, _ramp((t - a) / ramp) * _ramp((b - t) / ramp))
    return w


def transition_windows(t, timings, style) -> np.ndarray:
    """Frames owned by explicit transition attributes: preparation, transition, 0.25 s after it."""
    m = np.zeros(len(t), bool)
    for i in range(1, len(timings)):
        nt = timings[i]
        if not any(k in nt.attrs for k in TRANS_ATTRS):
            continue
        T = float(np.clip(nt.attrs.get("transition", style.transition), 0.01, 0.6))
        c = nt.start + style.transition_offset * T
        m |= (t >= c - T / 2 - 0.8 * T) & (t <= min(c + T / 2 + 0.25, nt.end))
    return m


def bridge(x, bad, margin: int = 2) -> np.ndarray:
    """``x`` with every run of bad frames (widened by ``margin`` frames on both sides) replaced
    by a straight line between the frames next to the run; a run at an end of ``x`` holds the
    nearest good value."""
    m = np.convolve(np.asarray(bad, float), np.ones(2 * margin + 1), mode="same") > 0
    if m.all():
        return x
    good = np.flatnonzero(~m)
    out = np.array(x, float)
    out[m] = np.interp(np.flatnonzero(m), good, out[good])
    return out


class BankPitch:
    """The bank's predictor laid under a track's pitch controls, phrase by phrase.

    ``t`` is the track's frame grid, ``s2`` Synth2's own curve on it (``build_pitch``) and
    ``s2_flat`` the same curve without vibrato. Notes with a ``transition`` / ``overshoot``
    attribute keep Synth2's curve in a window around their start; the predicted vibrato is
    scaled by ``style.vibrato_depth * vibratoEnv`` on sustained frames; notes whose vibrato is
    the user's design (vibrato attributes, or every note under ``vibrato="synthetic"``) get
    Synth2's vibrato instead. A few sung frames far outside the notes around them (the
    predictor now and then drops an octave on a consonant) are bridged; a phrase the predictor
    cannot sing keeps Synth2's curve. ``pitchDelta`` is added by the caller.
    """

    def __init__(self, t, timings, style, s2, s2_flat, env, expr: float, steps: int, vibrato: str,
                 is_vowel, frame: float):
        self.expr, self.steps = float(expr), int(steps)
        self.timings = timings
        self.is_vowel, self.frame = is_vowel, float(frame)
        self.s2, self.s2_flat, self.s2_vib = s2, s2_flat, s2 - s2_flat
        self.keep = transition_windows(t, timings, style)
        has_attr = lambda nt: any(k in nt.attrs for k in VIB_ATTRS)
        synth = has_attr
        if vibrato in ("synthetic", "predicted"):
            synth = lambda nt: vibrato == "synthetic"
        self.w_syn = sustain_weight(t, timings, **{**SUSTAIN, "min_note": 0.0}, pick=synth)
        self.syn_frames = np.zeros(len(t))
        for nt in timings:
            if synth(nt):
                self.syn_frames[(t >= nt.start) & (t < nt.end + 0.05)] = 1.0
        self.w_sus = sustain_weight(t, timings, **SUSTAIN)
        self.gain = float(style.vibrato_depth) * env

    def notes(self, units, seq, durs, f_start: int) -> NoteInputs | None:
        """The phrase's note inputs, or None when it has no pitched note."""
        ids = {i for u in units for i in u.notes}
        tm = [x for x in self.timings if x.index in ids]
        if not tm:
            return None
        try:
            return build_notes(notes_from_timings(tm, f_start, self.frame), seq, durs, self.is_vowel)
        except ValueError:
            return None

    def key(self, ni: NoteInputs | None, sl: slice) -> list:
        """Everything a phrase's curve depends on besides the model and its noise."""
        if ni is None:
            return ["expression", self.s2[sl]]
        return ["bank", self.expr, self.steps, ni.midi, ni.dur, ni.rest, self.keep[sl], self.s2[sl],
                self.s2_flat[sl], self.w_syn[sl], self.syn_frames[sl], self.w_sus[sl], self.gain[sl]]

    def phrase(self, predict, ni: NoteInputs | None, seq, durs, sl: slice) -> tuple[np.ndarray, str, int]:
        """``(pitch, fallback, repaired)``: pitch in MIDI semitones for the phrase's frames
        (without ``pitchDelta``), why Synth2's curve was used instead ("" when it was not), and
        the number of outlier frames bridged. ``predict(seq, durs, ni, expr, steps, keep_pitch,
        retake)`` runs the model."""
        k = self.keep[sl]
        repaired = 0
        try:
            if ni is None:
                raise ValueError("no pitched note")
            kw = {"keep_pitch": self.s2_flat[sl], "retake": ~k} if k.any() else {}
            pred = predict(seq, durs, ni, self.expr, self.steps, **kw)
            if not np.isfinite(pred).all():
                raise ValueError("prediction not finite")
            bad = ~ni.frame_rest() & ~k & (range_excess(pred, ni) > MAX_DEV)
            if bad.any():
                if bad.sum() > MAX_OUTLIERS * max(1, int((~ni.frame_rest()).sum())):
                    raise ValueError("prediction out of range")
                pred = bridge(pred, bad)
                repaired = int(bad.sum())
        except Exception as err:  # the phrase keeps Synth2's curve
            return self.s2[sl], f"{type(err).__name__}: {err}", 0
        if k.any():
            ws = soft_window(k, 3)
            pred = ws * self.s2_flat[sl] + (1.0 - ws) * pred
        weight = np.maximum(self.w_sus[sl], self.w_syn[sl]) * (1.0 - soft_window(k, 3))
        p = scale_fast(pred, base_pitch(ni), self.gain[sl] * (1.0 - self.w_syn[sl]), weight, self.frame)
        return p + self.s2_vib[sl] * self.syn_frames[sl], "", repaired
