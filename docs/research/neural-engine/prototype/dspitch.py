"""DiffSinger pitch predictor on onnxruntime: a port of OpenUtau's ``DiffSingerPitch.cs``.

A voicebank's ``dspitch/`` folder holds a linguistic encoder and a diffusion pitch model that turn
phonemes with durations plus the score's notes into a sung pitch curve (MIDI semitones per frame).

    bank = PitchBank(folder)                      # dspitch/ of a voicebank
    notes = [PhraseNote(start, end, midi, slur)]  # frames from the phrase's first frame
    notes = notes_from_timings(timings, f_start, hop / sr)  # or from notes in seconds
    curve = bank.predict(phones, ph_dur, notes, expr=1.0, steps=10, seed=1234)

Frame layout (the same one the acoustic model gets): ``phones`` and ``ph_dur`` cover the whole
phrase, head ``SP`` and tail ``SP`` included, and ``sum(ph_dur)`` is the number of frames. Notes
are given on that frame grid; :func:`build_notes` adds the rest notes the model expects (before the
first note, in gaps, after the last note).

Banks differ: a multi-speaker bank's pitch model also takes ``spk_embed float[1,F,H]`` (one of the
embeddings listed under ``speakers:`` in ``dspitch/dsconfig.yaml``), and every sub-model has its own
phoneme and language tables. The feed is built from the session's own input list.

Model inputs (Qixuan v2.7.0, read from the sessions, not assumed):

    linguistic  tokens int64[1,P]  languages int64[1,P]  ph_dur int64[1,P]
                -> encoder_out float[1,P,256]  x_masks bool[1,P]
    pitch       encoder_out  ph_dur int64[1,P]
                note_midi float[1,N]  note_rest bool[1,N]  note_dur int64[1,N]
                pitch float[1,F]  expr float[1,F]  retake bool[1,F]  steps int64[]
                -> pitch_pred float[1,F]

What the frame inputs mean (DiffSinger ``forward_pitch_preprocess``, checked against the model):

* The model builds its own base pitch from ``note_midi`` / ``note_dur`` (the note steps smoothed
  over three frames, see :func:`base_pitch`) and predicts the deviation from it.
* ``retake`` true = generate this frame. False = keep this frame: the model is told the deviation
  ``pitch - base`` there and generates the other frames to join it; its output on kept frames is
  within a few cents of ``pitch``. ``pitch`` is not read where ``retake`` is true.
* ``expr`` (0..1, per frame) blends the "generate" embedding with the "keep, deviation 0"
  embedding. 0 returns the base pitch; up to 0.5 the curve stays close to it; overshoot,
  preparation, drift and vibrato arrive between 0.5 and 1.

The diffusion noise is drawn inside the graph. A session created right after
``onnxruntime.set_seed(n)`` gives the same curve on its first run every time (also across thread
counts and processes), so ``seed`` makes a fresh pitch session (0.15 to 0.3 s); without a seed one
session is reused and each run is another take. ``set_seed`` is process-global.

Helpers for laying Synth2's pitch controls over a predicted curve: :func:`retake_mask` and
:func:`soft_window` (keep a given curve on some frames), :func:`scale_fast` (vibrato depth),
:func:`range_excess` (sanity check of a prediction).

numpy and the standard library only; ``onnxruntime`` and ``yaml`` are imported on first use.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REST_SYMBOLS = ("SP", "AP")
_SYMBOL = re.compile(r"\bsymbol:\s*([^,}\s]+)")
_TYPE = re.compile(r"\btype:\s*([A-Za-z_]+)")
DEFAULT_STEPS = 10  # OpenUtau's DiffSingerStepsPitch default


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


def retake_mask(inputs: NoteInputs, retake_notes) -> np.ndarray:
    """Frame mask that regenerates the notes in ``retake_notes`` (OpenUtau ``BuildRetakeFrameMask``).

    A gap rest follows the note before it, the head rest the first note, the tail the last note.
    """
    keep = set(int(i) for i in retake_notes)
    return np.repeat(np.array([o in keep for o in inputs.owner], bool), inputs.dur)


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


class PitchBank:
    """A voicebank's pitch predictor (``<bank>/dspitch``)."""

    def __init__(self, folder, threads: int | None = None, speaker: str | None = None):
        self.folder = Path(folder)
        self.threads = int(threads or os.environ.get("SYNTH2_NEURAL_THREADS") or 2)
        import yaml

        cfg = yaml.safe_load((self.folder / "dsconfig.yaml").read_text(encoding="utf-8-sig"))
        if not cfg.get("pitch"):
            raise ValueError(f"{self.folder} has no pitch model")
        if cfg.get("predict_dur"):
            # word mode (word_div / word_dur into the linguistic encoder) is not ported
            raise ValueError("pitch models with predict_dur: true are not supported")
        if not cfg.get("use_continuous_acceleration", False):
            raise ValueError("pitch models that take 'speedup' instead of 'steps' are not supported")
        self.config = cfg
        self.hop_s = float(cfg.get("hop_size", 512)) / float(cfg.get("sample_rate", 44100))
        self.use_expr = bool(cfg.get("use_expr", False))
        self.use_note_rest = bool(cfg.get("use_note_rest", False))
        self.phonemes = self._phoneme_table(self.folder / cfg["phonemes"])
        self.languages = {}
        if cfg.get("use_lang_id"):
            self.languages = json.loads((self.folder / cfg["languages"]).read_text(encoding="utf-8-sig"))
        self.vowels = self._vowels(self.folder / "dsdict.yaml")
        self.speaker = self._speaker(cfg.get("speakers") or [], speaker)
        self._embed = None
        self._ling = None
        self._pitch = None

    @staticmethod
    def _speaker(speakers: list[str], want: str | None) -> str | None:
        """The ``speakers`` entry for a voice mode: exact match, then trailing path segments
        (OpenUtau ``getSpeakerIndexBySuffix``); the first entry when no mode is asked for."""
        if not speakers:
            return None
        if not want:
            return speakers[0]
        parts = want.split("/")
        for sp in speakers:
            if sp == want or sp.split("/")[-len(parts):] == parts:
                return sp
        raise KeyError(f"speaker {want!r} not in {speakers}")

    @staticmethod
    def _phoneme_table(path: Path) -> dict[str, int]:
        if path.suffix.lower() == ".json":
            return json.loads(path.read_text(encoding="utf-8-sig"))
        return {p: i for i, p in enumerate(path.read_text(encoding="utf-8-sig").splitlines())}

    @staticmethod
    def _vowels(path: Path) -> set[str]:
        """Symbols typed ``vowel`` in ``dsdict.yaml`` (OpenUtau's ``g2p.IsVowel``), plus SP / AP.

        The file is scanned line by line (a bank's dictionary can run to millions of lines and
        ``yaml.safe_load`` then takes minutes). Both layouts are read: ``- {symbol: a, type: vowel}``
        and ``- symbol: a`` followed by ``type: vowel``.
        """
        out = set(REST_SYMBOLS)
        if not path.exists():
            return out
        sym = None
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                m = _SYMBOL.search(line)
                if m:
                    sym = m.group(1).strip("'\"")
                m = _TYPE.search(line)
                if m and sym is not None:
                    if m.group(1) == "vowel":
                        out.add(sym)
                    sym = None
        return out

    def is_vowel(self, phoneme: str) -> bool:
        return phoneme in self.vowels

    def has(self, phoneme: str) -> bool:
        return phoneme in self.phonemes

    # ---- sessions
    def _session(self, name: str):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.log_severity_level = 3
        return ort.InferenceSession(str(self.folder / self.config[name]), so, providers=["CPUExecutionProvider"])

    def linguistic_session(self):
        if self._ling is None:
            self._ling = self._session("linguistic")
        return self._ling

    def pitch_session(self, seed: int | None = None):
        """The diffusion model. With ``seed``: a fresh session whose first run is reproducible."""
        if seed is not None:
            import onnxruntime as ort

            ort.set_seed(int(seed) & 0x7FFFFFFF)
            return self._session("pitch")
        if self._pitch is None:
            self._pitch = self._session("pitch")
        return self._pitch

    # ---- inference
    def encode(self, phones: list[str], ph_dur: list[int]) -> np.ndarray:
        """Linguistic encoder (deterministic): ``encoder_out`` float[1, P, hidden]."""
        missing = [p for p in phones if p not in self.phonemes]
        if missing:
            raise KeyError(f"phonemes not in the pitch model: {sorted(set(missing))}")
        feed = {"tokens": np.array([[self.phonemes[p] for p in phones]], np.int64),
                "ph_dur": np.array([ph_dur], np.int64)}
        if self.languages:
            feed["languages"] = np.array(
                [[self.languages.get(p.split("/")[0], 0) if "/" in p else 0 for p in phones]], np.int64)
        return self.linguistic_session().run(["encoder_out"], feed)[0]

    def predict(self, phones: list[str], ph_dur: list[int], notes: list[PhraseNote], expr=1.0,
                steps: int = DEFAULT_STEPS, seed: int | None = None, keep_pitch: np.ndarray | None = None,
                retake: np.ndarray | None = None, inputs: NoteInputs | None = None) -> np.ndarray:
        """Predicted pitch in MIDI semitones for every frame of the phrase (float64 [F]).

        ``expr``: scalar or per-frame array, 0 (plain note steps) to 1 (full expression).
        ``keep_pitch`` + ``retake``: frames with ``retake`` false are conditioned on ``keep_pitch``
        (semitones) and the rest is generated around them. The model returns its own values for
        the kept frames too (close to ``keep_pitch``, not equal); callers that need the kept
        frames exact write them back, as OpenUtau does.
        """
        ph_dur = [int(d) for d in ph_dur]
        n = sum(ph_dur)
        if len(phones) != len(ph_dur) or n <= 0 or min(ph_dur) < 0:
            raise ValueError("phones and ph_dur must describe at least one frame")
        ni = inputs or build_notes(notes, phones, ph_dur, self.is_vowel)
        if int(ni.dur.sum()) != n:
            raise ValueError("note durations do not cover the phrase")
        if retake is None or keep_pitch is None:
            mask = np.ones(n, bool)
            pitch = np.full(n, 60.0, np.float32)
        else:
            mask = np.asarray(retake, bool)
            pitch = np.where(mask, 60.0, np.asarray(keep_pitch, np.float64)).astype(np.float32)
            if mask.shape != (n,) or pitch.shape != (n,):
                raise ValueError("retake and keep_pitch must have one value per frame")
        feed = {"encoder_out": self.encode(phones, ph_dur), "ph_dur": np.array([ph_dur], np.int64),
                "note_midi": ni.midi[None], "note_dur": ni.dur[None], "pitch": pitch[None], "retake": mask[None],
                "steps": np.array(max(1, int(steps)), np.int64)}
        if self.use_note_rest:
            feed["note_rest"] = ni.rest[None]
        if self.use_expr:
            e = np.clip(np.broadcast_to(np.asarray(expr, np.float32), (n,)), 0.0, 1.0)
            feed["expr"] = np.ascontiguousarray(e, np.float32)[None]
        sess = self.pitch_session(seed)
        dim = next((i.shape[-1] for i in sess.get_inputs() if i.name == "spk_embed"), None)
        if dim is not None:
            feed["spk_embed"] = np.tile(self.embed(dim)[None, None], (1, n, 1)).astype(np.float32)
        names = {i.name for i in sess.get_inputs()}
        out = sess.run(["pitch_pred"], {k: v for k, v in feed.items() if k in names})[0][0]
        return out.astype(np.float64)

    def embed(self, dim: int) -> np.ndarray:
        """The speaker embedding (``<speaker>.emb``: raw little-endian float32, one per hidden unit)."""
        if self._embed is None:
            if self.speaker is None:
                raise ValueError("the pitch model takes spk_embed and dsconfig.yaml lists no speakers")
            e = np.fromfile(str(self.folder / (self.speaker + ".emb")), "<f4")
            if e.size != dim:
                raise ValueError(f"{self.speaker}.emb has {e.size} values, the model takes {dim}")
            self._embed = e
        return self._embed
