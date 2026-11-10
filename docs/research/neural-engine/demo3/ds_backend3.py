"""Demo-3 VoiceBackend: the combined DiffSinger engine (demo3/dsinfer.py) behind Synth2's renderer.

One singer (Qixuan) plays every vocal track. A track's character comes from CAST: the model's `gender`
(positive = formants up), `shift` (semitones added to the written notes before singing: the register the
character sings in), `tone` (semitones the acoustic model sings above the real pitch), engine options and
`doubles` (extra takes: detuned, delayed, other noise seed). No ring modulator, no effect.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import numpy as np

import dsinfer
from synth2.engine.expression import evaluate_curve
from synth2.engine.renderer import FormantBackend
from synth2.voices import VOICES

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
QIXUAN = S / "models/installed/qixuan/files/Qixuan_v2.7.0_DiffSinger_OpenUtau"
VOCODER = S / "models/installed/pc-nsf-hifigan/files/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"

CACHE = S / "demo3" / "cache"
ENGINE_SHA = hashlib.sha1(b"".join((S / "demo3" / f).read_bytes() for f in ("dsinfer.py", "dspitch.py"))).hexdigest()

# v67/config.json, banks.qixuan.options (seed: the track's seed)
OPTIONS = {"slow_note": 0.25, "slow_max": 2.0, "voicing_floor": 6.0, "pitch_source": "bank", "expr": None,
           "pitch_steps": 10, "steps": 10, "var_steps": 10, "depth": 0.6, "velocity": 1.0}

# Track-name prefix -> character (the longest matching prefix wins; split channels such as "AI Verses" or
# "Aliens Hook" inherit their source's entry). Overridden by the JSON file named in DEMO3_CAST.
# measured in runs/cast.jsonl (cast_spec1.py, cast_spec2.py); NOTES.md has the table
# -3 dB of breathiness on vowel frames; the bank's pitch predictor sings 6-9 cents flat at expr 0.5 (checks.py)
LEAD_OPTS = {"vowel_breath_off": -3.0, "detune": 7.0}
CAST: dict[str, dict] = {
    # the AI: the bank's own voice in its comfortable register
    "AI": {"gender": 0.0, "opts": LEAD_OPTS},
    "AI Double": {"gender": 0.45, "opts": LEAD_OPTS},
    "Harmony": {"gender": 0.35, "opts": LEAD_OPTS},
    # the low alien: the written bass register (the bank breaks there on its own), the acoustic model an octave up
    "Aliens": {"gender": -0.6, "tone": 12},
    # the big alien: formants down, shadowed an octave below by a second take
    "Aliens 2": {"gender": -1.0, "opts": {"detune": 10.0},
                 "doubles": [{"shift": -12, "tone": 12, "gender": 0.4, "delay": 0.0, "detune": 10.0, "gain_db": -4.0}]},
    # the crowd: two low voices and a high one, each doubled by a detuned, late second take
    "Crowd": {"gender": -0.3, "tone": 12, "doubles": [{"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5}]},
    "Crowd 2": {"gender": -0.7, "tone": 12, "doubles": [{"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5}]},
    "Crowd 3": {"gender": 0.2, "doubles": [{"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5}]},
}


def load_cast() -> dict[str, dict]:
    cast = copy.deepcopy(CAST)
    path = os.environ.get("DEMO3_CAST")
    if path:
        for k, v in json.loads(Path(path).read_text()).items():
            cast[k] = v
    return cast


def active_rms(x, sr, win=0.02):
    n = int(win * sr)
    if len(x) < n:
        return 0.0
    f = x[: len(x) // n * n].reshape(-1, n)
    e = np.mean(f.astype(np.float64) ** 2, axis=1)
    if e.max() <= 0:
        return 0.0
    return float(np.sqrt(np.mean(e[e > e.max() * 1e-4])))


def shifted(track, semitones):
    """A copy of the track with every note moved by `semitones`."""
    if not semitones:
        return track
    tr = copy.copy(track)
    tr.notes = [copy.copy(n) for n in track.notes]
    for n in tr.notes:
        n.pitch += int(semitones)
    return tr


class DSBackend:
    name = "diffsinger"

    def __init__(self, ds=None, cast=None, options=None, boost_db=None, threads=2, log=print):
        self.ds = ds or dsinfer.DS(QIXUAN, VOCODER, threads=threads, seed=1)
        self.cast = cast if cast is not None else load_cast()
        self.options = dict(OPTIONS, **(options or {}))
        self.boost_db = boost_db or {}
        self.formant = FormantBackend()
        self.log = log
        self.rows = []  # one dict per rendered track, for the notes
        self.use_cache = os.environ.get("DEMO3_NO_CACHE") is None

    def voices(self):
        return list(VOICES)

    def character(self, track) -> dict:
        best = ""
        for k in self.cast:
            if track.name == k or track.name.startswith(k + " "):
                if len(k) > len(best):
                    best = k
        return dict(self.cast.get(best, {}), character=best or "(default)")

    def _key(self, project, track, window, c, opts) -> str:
        """Fingerprint of everything a dry take depends on (track ids and mixer settings left out)."""
        d = track.to_dict()
        for k in ("id", "name", "gain_db", "pan", "mute", "solo", "reverb_send", "engine"):
            d.pop(k, None)
        for n in d.get("notes", []):
            n.pop("id", None)
        blob = json.dumps({"track": d, "tempo": project.tempo.to_list(), "window": window, "char": c, "opts": opts,
                           "engine": ENGINE_SHA}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()

    def sing(self, project, track, window=None, **over):
        """Dry take of a track with its character's settings. Returns (audio from time 0, info).
        Takes are kept in demo3/cache (the engine repeats a take bit for bit for the same seed)."""
        c = dict(self.character(track), **over)
        opts = dict(self.options, **c.get("opts", {}))
        opts.setdefault("seed", track.seed if track.seed is not None else 1)
        key = self._key(project, track, window, c, opts)
        f = CACHE / f"{key}.npy"
        if self.use_cache and f.exists():
            info = json.loads(f.with_suffix(".json").read_text())
            info["character"] = c
            return np.load(f), info
        audio, info = self._sing(project, track, window, c, opts)
        if self.use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(f, audio)
            f.with_suffix(".json").write_text(json.dumps({k: v for k, v in info.items() if k not in ("phones", "character", "dump")},
                                                          default=str))
        return audio, info

    def _sing(self, project, track, window, c, opts):
        tr = shifted(track, c.get("shift", 0))
        audio, info = dsinfer.render_track_ds(self.ds, project, tr, gender=float(c.get("gender", 0.0)),
                                              tone=float(c.get("tone", 0.0)), window=window, **opts)
        audio = audio.astype(np.float32) * 10 ** (c.get("gain_db", 0.0) / 20)
        for i, d in enumerate(c.get("doubles", []) or []):
            # a second take: other noise seed, detuned, delayed; `shift` / `tone` / `gender` are added to the main take's
            o = dict(opts, seed=int(opts["seed"]) + 101 * (i + 1), detune=float(d.get("detune", 0.0)))
            tr2 = shifted(track, c.get("shift", 0) + d.get("shift", 0))
            a2, _ = dsinfer.render_track_ds(self.ds, project, tr2, gender=float(c.get("gender", 0.0)) + d.get("gender", 0.0),
                                            tone=float(c.get("tone", 0.0)) + d.get("tone", 0.0), window=window, **o)
            k = int(round(d.get("delay", 0.0) * dsinfer.SR))
            a2 = np.concatenate([np.zeros(k, np.float32), a2.astype(np.float32)])[: len(audio)]
            # level of the second take relative to the main take's active level
            ra, rb = active_rms(audio, dsinfer.SR), active_rms(a2, dsinfer.SR)
            audio[: len(a2)] += a2 * (ra / rb if ra > 0 and rb > 0 and i == 0 else 1.0) * 10 ** (d.get("gain_db", 0.0) / 20)
        info["character"] = c
        return audio, info

    def render_track(self, project, track, sample_rate, t0, t1, note_range=None):
        assert sample_rate == dsinfer.SR, "this backend renders at 44.1 kHz only"
        sr = sample_rate
        ref, report = self.formant.render_track(project, track, sr, t0, t1, note_range)
        audio, info = self.sing(project, track, window=(t0, t1))
        c = info["character"]
        i0 = int(round(t0 * sr))
        n = int(round((t1 - t0) * sr))
        seg = audio[max(0, i0): i0 + n]
        out = np.zeros(n, np.float32)
        out[: len(seg)] = seg
        # loudness lane as a gain ride
        tc = t0 + np.arange(0, n, 441) / sr
        ld = evaluate_curve(track.curves.get("loudness"), tc, project.tempo.beat_to_sec, 0.0)
        out *= np.interp(np.arange(n), np.arange(0, n, 441), 10 ** (np.clip(ld, -48, 12) / 20)).astype(np.float32)
        # same active level as the formant engine's take, so the song's faders keep their meaning
        a, b = active_rms(ref, sr), active_rms(out, sr)
        g = 1.0
        if a > 0 and b > 0:
            g = a / b * 10 ** (self.boost_db.get(track.name, self.boost_db.get(c["character"], 0.0)) / 20)
            out *= g
        report = dict(report, engine="diffsinger", peak_db=float(20 * np.log10(max(1e-9, np.max(np.abs(out))))))
        row = {"track": track.name, "voice": track.voice, "style": track.style, "character": c["character"],
               "gender": c.get("gender", 0.0), "shift": c.get("shift", 0), "tone": c.get("tone", 0.0),
               "doubles": len(c.get("doubles", []) or []), "phrases": info.get("phrases", 0),
               "sung_s": round(info.get("audio_s", 0.0), 1), "gain_db": round(20 * np.log10(max(g, 1e-9)), 1),
               "pitch_source": info.get("pitch_source"), "fallbacks": len(info.get("pitch_fallbacks", []) or []),
               "repaired": info.get("pitch_repaired", 0), "slow_max": max(info.get("slow", [1.0]) or [1.0])}
        self.rows.append(row)
        self.log(f"    [ds] {track.name}: {row['phrases']} phrases, {row['sung_s']}s sung, gender {row['gender']:+.2f} "
                 f"shift {row['shift']:+d} tone {row['tone']:+.0f} doubles {row['doubles']} gain {row['gain_db']:+.1f} dB "
                 f"pitch {row['pitch_source']} fallbacks {row['fallbacks']}")
        return out, report
