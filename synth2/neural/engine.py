"""Runs a voicebank's models for one phrase: durations, pitch, variance, acoustic, vocoder.

:class:`Runner` holds the sessions one phrase uses and builds every feed from the model's
own input list, so banks with speaker embeddings, a tension variance, no variance model or
no ``depth`` input all go through the same code. :func:`synth_phrase` is the phrase
pipeline, a port of OpenUtau's DiffSinger renderer plus three measures from this engine's
research notes (``docs/research/neural-engine``):

* slow motion: notes shorter than ``slow_note`` seconds are sung up to ``slow_max`` times
  slower by the variance and acoustic models and the mel is squeezed back onto the real
  frame grid; the vocoder gets the real f0, so timing and pitch are as written;
* a floor under the predicted voicing of vowels (a collapsed prediction is a whispered
  syllable);
* reproducible noise: the diffusion models draw their noise inside the graph, so in
  reproducible mode every phrase runs them in sessions made right after
  ``onnxruntime.set_seed`` with a seed from the track seed and the phrase's place.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

from . import align as al
from .voicebank import Voicebank, VoicebankError, feed, speedup_for

ENGINE_VERSION = "1"  # part of every cache key: raise it when the same inputs give other audio
QUALITY_STEPS = {"draft": 5, "normal": 10, "high": 20}
VARIANCE_STEPS = 10
DEPTH = 0.6  # shallow-diffusion depth asked for; clipped to the bank's max_depth
STEPS_ENV = "SYNTH2_NEURAL_STEPS"
REPRODUCIBLE_ENV = "SYNTH2_NEURAL_REPRODUCIBLE"
# limits OpenUtau clamps the variance curves to before the acoustic model reads them
VARIANCE_LIMITS = {"breathiness": (-96.0, 0.0), "voicing": (-96.0, 0.0), "energy": (-96.0, 0.0),
                   "tension": (-10.0, 10.0)}


def reproducible() -> bool:
    """False when ``SYNTH2_NEURAL_REPRODUCIBLE`` is ``0`` / ``false`` / ``off`` / ``no``."""
    return (os.environ.get(REPRODUCIBLE_ENV) or "1").strip().lower() not in ("0", "false", "off", "no")


def steps_for(quality: str | None = None) -> int:
    """Acoustic diffusion steps: ``SYNTH2_NEURAL_STEPS`` if set, else the quality preset's."""
    try:
        n = int(os.environ.get(STEPS_ENV) or 0)
    except ValueError:
        n = 0
    if n > 0:
        return n
    if quality is None:
        from ..engine.backend import current_quality

        quality = current_quality()
    return QUALITY_STEPS.get(quality, QUALITY_STEPS["normal"])


@dataclass(frozen=True)
class Options:
    """Per-render settings of the phrase pipeline (scalars only: they go into cache keys)."""

    steps: int = QUALITY_STEPS["normal"]
    var_steps: int = VARIANCE_STEPS
    depth: float = DEPTH
    velocity: float = 1.0
    slow_note: float | None = al.SLOW_NOTE  # None or 0: no slow motion
    slow_max: float = al.SLOW_MAX
    voicing_floor: float | None = al.VOICING_FLOOR  # dB; None or 0: no floor
    min_vowel: float | None = None

    def key(self) -> list:
        return [self.steps, self.var_steps, float(self.depth), float(self.velocity), self.slow_note or 0.0,
                float(self.slow_max), self.voicing_floor or 0.0, self.min_vowel or 0.0]


@dataclass
class PhraseInput:
    """One phrase ready for the models, on its own frames (``n = sum(durs)``)."""

    seq: list[str]
    durs: list[int]
    is_vowel: list[bool]
    slow: np.ndarray  # per phoneme of seq: slow-motion factor
    pitch: np.ndarray  # MIDI semitones per frame, as sung
    tone: np.ndarray | float = 0.0  # semitones added to the pitch the variance and acoustic models see
    gender: np.ndarray | float = 0.0  # the acoustic model's formant shift, positive = brighter
    offsets: dict = field(default_factory=dict)  # variance name -> offset per frame or scalar


class Timer:
    def __init__(self):
        self.t: dict[str, float] = {}

    def add(self, k: str, dt: float) -> None:
        self.t[k] = self.t.get(k, 0.0) + dt


class Runner:
    """The models of one bank for one phrase.

    With ``seed`` every model that draws noise runs in a session of its own, created right
    after ``onnxruntime.set_seed(seed)``; without it the bank's shared sessions are used and
    each run is another take. ``blend`` is the speaker blend (``Voicebank.blend``).
    """

    def __init__(self, bank: Voicebank, seed: int | None = None, blend=None, timer: Timer | None = None):
        self.bank, self.seed, self.blend = bank, seed, blend
        self.timer = timer or Timer()
        self._fresh: dict[str, object] = {}

    def session(self, key: str):
        if self.seed is None or not self.bank.draws_noise(key):
            return self.bank.session(key)
        sess = self._fresh.get(key)
        if sess is None:
            t = time.perf_counter()
            sess = self._fresh[key] = self.bank.fresh_session(key, self.seed)
            self.timer.add("sessions", time.perf_counter() - t)
        return sess

    def close(self) -> None:
        self._fresh.clear()

    def _run(self, key: str, values: dict, sub, length: int) -> dict:
        """Run model ``key`` and return ``{output name: array}``. A ``spk_embed`` input is
        filled with the blend's embedding repeated ``length`` times."""
        sess = self.session(key)
        io = self.bank.io(key)
        if io.takes("spk_embed") and "spk_embed" not in values:
            e = self.bank.embed(sub, int(io.shapes["spk_embed"][-1]), self.blend)
            values["spk_embed"] = np.tile(e[None, None], (1, length, 1)).astype(np.float32)
        res = sess.run(None, feed(io, values, f"{key} model of voicebank {self.bank.id!r}"))
        return dict(zip(io.outputs, res))

    # ---- the models
    def predict_dur(self, phones, word_div, word_dur, ph_midi) -> np.ndarray:
        """The duration model's length (frames, float64) of every phoneme."""
        t = time.perf_counter()
        sub = self.bank.dur
        d = {"tokens": sub.tokens(phones), "languages": sub.language_ids(phones),
             "word_div": np.array([word_div], np.int64), "word_dur": np.array([word_dur], np.int64)}
        d.update(self._run("dur.linguistic", d, sub, len(phones)))
        d["ph_midi"] = np.array([ph_midi], np.int64)
        d.pop("spk_embed", None)
        out = self._run("dur", d, sub, len(phones))["ph_dur_pred"][0]
        self.timer.add("dur", time.perf_counter() - t)
        return out.astype(np.float64)

    def predict_pitch(self, seq, durs, ni, expr, steps: int, keep_pitch=None, retake=None) -> np.ndarray:
        """The pitch predictor's curve in MIDI semitones for every frame (float64).

        Frames with ``retake`` false are conditioned on ``keep_pitch`` and the rest is
        generated around them.
        """
        sub = self.bank.pitch
        if sub is None:
            raise VoicebankError(f"voicebank {self.bank.id!r} has no pitch predictor")
        t = time.perf_counter()
        durs = [int(d) for d in durs]
        n = sum(durs)
        if len(seq) != len(durs) or n <= 0 or min(durs) < 0 or int(ni.dur.sum()) != n:
            raise ValueError("phonemes, lengths and notes must describe the same frames")
        if retake is None or keep_pitch is None:
            mask = np.ones(n, bool)
            pitch = np.full(n, 60.0, np.float32)
        else:
            mask = np.asarray(retake, bool)
            pitch = np.where(mask, 60.0, np.asarray(keep_pitch, np.float64)).astype(np.float32)
            if mask.shape != (n,) or pitch.shape != (n,):
                raise ValueError("retake and keep_pitch must have one value per frame")
        ph_dur = np.array([durs], np.int64)
        d = {"tokens": sub.tokens(seq), "languages": sub.language_ids(seq), "ph_dur": ph_dur}
        enc = self._run("pitch.linguistic", d, sub, len(seq))["encoder_out"]
        e = np.clip(np.broadcast_to(np.asarray(expr, np.float32), (n,)), 0.0, 1.0)
        d = {"encoder_out": enc, "ph_dur": ph_dur, "note_midi": ni.midi[None], "note_dur": ni.dur[None],
             "note_rest": ni.rest[None], "pitch": pitch[None], "retake": mask[None],
             "expr": np.ascontiguousarray(e, np.float32)[None], "steps": np.array(max(1, int(steps)), np.int64)}
        out = self._run("pitch", d, sub, n)["pitch_pred"][0]
        self.timer.add("pitch", time.perf_counter() - t)
        return out.astype(np.float64)

    def predict_variance(self, seq, durs, pitch_semi, steps: int) -> dict:
        """``{name: curve}`` for every variance the bank predicts (empty without ``dsvariance/``)."""
        sub = self.bank.variance
        if sub is None:
            return {}
        t = time.perf_counter()
        n = len(pitch_semi)
        ph_dur = np.array([durs], np.int64)
        d = {"tokens": sub.tokens(seq), "languages": sub.language_ids(seq), "ph_dur": ph_dur}
        enc = self._run("variance.linguistic", d, sub, len(seq))["encoder_out"]
        self.session("variance")
        outs = self.bank.io("variance").outputs
        z = np.zeros((1, n), np.float32)
        d = {"encoder_out": enc, "ph_dur": ph_dur, "pitch": pitch_semi[None].astype(np.float32), "energy": z,
             "breathiness": z, "voicing": z, "tension": z, "retake": np.ones((1, n, len(outs)), bool),
             "steps": np.array(steps, np.int64), "speedup": np.array(speedup_for(steps), np.int64)}
        res = self._run("variance", d, sub, n)
        self.timer.add("variance", time.perf_counter() - t)
        return {(o[:-5] if o.endswith("_pred") else o): r[0] for o, r in res.items()}

    def mel(self, seq, durs, f0, var: dict, gender=0.0, velocity=1.0, steps: int = 10, depth: float = DEPTH,
            offsets: dict | None = None) -> np.ndarray:
        """The acoustic model's mel, ``float32 [frames, bins]`` for ``sum(durs)`` frames."""
        t = time.perf_counter()
        bank, sub = self.bank, self.bank.acoustic
        n = len(f0)
        offsets = offsets or {}
        self.session("acoustic")
        io = bank.io("acoustic")
        g = np.broadcast_to(np.asarray(gender, np.float32), (n,))[None].astype(np.float32)
        v = np.broadcast_to(np.asarray(velocity, np.float32), (n,))[None].astype(np.float32)
        d = {"tokens": sub.tokens(seq), "languages": sub.language_ids(seq), "durations": np.array([durs], np.int64),
             "f0": f0[None].astype(np.float32), "gender": g, "velocity": v,
             "depth": np.array(min(depth, bank.max_depth), np.float32), "steps": np.array(steps, np.int64),
             "speedup": np.array(speedup_for(steps), np.int64)}
        for k, (lo, hi) in VARIANCE_LIMITS.items():
            if io.takes(k):
                c = var.get(k)
                if c is None:
                    raise VoicebankError(f"voicebank {bank.id!r}: the acoustic model takes {k!r} and the bank has "
                                         f"no variance model that predicts it")
                d[k] = np.clip(c + offsets.get(k, 0.0), lo, hi)[None].astype(np.float32)
        mel = self._run("acoustic", d, sub, n)["mel"]
        self.timer.add("acoustic", time.perf_counter() - t)
        return mel[0]

    def vocode(self, mel, f0) -> np.ndarray:
        t = time.perf_counter()
        d = {"mel": mel[None].astype(np.float32), "f0": f0[None].astype(np.float32)}
        wav = self._run("vocoder", d, None, len(f0))["waveform"][0]
        self.timer.add("vocoder", time.perf_counter() - t)
        return wav


def hz(semitones) -> np.ndarray:
    return 440.0 * 2 ** ((semitones - 69) / 12)


def _on(grid, n: int, value):
    """A per-frame value on the slow grid (scalars pass through)."""
    if np.ndim(value) == 0:
        return value
    return np.interp(grid, np.arange(n), value)


def synth_phrase(run: Runner, ph: PhraseInput, opts: Options) -> np.ndarray:
    """The phrase's waveform at the bank's sample rate: ``sum(durs) * hop`` samples, float32,
    exactly as the vocoder returns it."""
    frame = run.bank.frame
    n = int(sum(ph.durs))
    p_semi = ph.pitch
    if len(p_semi) != n:
        raise ValueError(f"the phrase has {n} frames and {len(p_semi)} pitch values")
    f0 = hz(p_semi)
    tone, gender, offsets = ph.tone, ph.gender, ph.offsets
    if ph.slow.max() > al.SLOW_MIN:
        # fast notes: the variance and acoustic models sing them in slow motion and the mel is
        # squeezed back onto the real frame grid
        durs_ac, to_true, to_slow = al.slow_grid(ph.durs, ph.slow)
        p_ac = np.interp(to_true, np.arange(n), p_semi) + _on(to_true, n, tone)
        gender = _on(to_true, n, gender)
        offsets = {k: _on(to_true, n, v) for k, v in offsets.items()}
    else:
        durs_ac, to_slow, p_ac = ph.durs, None, p_semi + tone
    var = run.predict_variance(ph.seq, durs_ac, p_ac, steps=opts.var_steps)
    if opts.voicing_floor and "voicing" in var:
        var["voicing"] = al.floor_voicing(np.clip(var["voicing"], -96, 0), ph.seq, durs_ac, ph.is_vowel, frame,
                                          opts.voicing_floor)
    mel = run.mel(ph.seq, durs_ac, hz(p_ac), var, gender=gender, velocity=opts.velocity, steps=opts.steps,
                  depth=opts.depth, offsets=offsets)
    if to_slow is not None:
        mel = al.warp_rows(mel, to_slow)
    return run.vocode(mel, f0)
