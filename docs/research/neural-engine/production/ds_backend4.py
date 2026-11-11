"""Production backend (copy of demo3/ds_backend3.py): Qixuan behind Synth2's renderer, with the vocal production.

Differences from the demo-3 backend:
* the engine is `$S/flow` (the colleague's copy, lead options from flow/config.json) when that config exists, else
  `$S/demo3` with the demo's options; PROD_ENGINE=demo3|flow forces one,
* the lead channels (LEAD) go through chain4.LEAD after the level match, the harmony through chain4.HARMONY; the
  renderer's own vocal strip is switched off for them in build4.py,
* "AI Double" is not a second take: it is the lead's take over the double's notes, through chain4.adt (delayed,
  detuned, darker, consonants pushed back). PROD_DOUBLE=take gives the demo's second take back,
* takes are cached in prod/cache; takes of the demo-3 engine are also read from demo3/cache.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
P = S / "prod"
_eng = os.environ.get("PROD_ENGINE") or ("flow" if (S / "flow" / "config.json").exists() else "demo3")
ENGINE_DIR = S / _eng
sys.path.insert(0, str(ENGINE_DIR))
sys.path.insert(0, str(P))

import dsinfer  # noqa: E402  (the engine copy in ENGINE_DIR)

import chain4  # noqa: E402
from synth2.engine.expression import evaluate_curve  # noqa: E402
from synth2.engine.renderer import FormantBackend  # noqa: E402
from synth2.voices import VOICES  # noqa: E402

QIXUAN = S / "models/installed/qixuan/files/Qixuan_v2.7.0_DiffSinger_OpenUtau"
VOCODER = S / "models/installed/pc-nsf-hifigan/files/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
CACHE = P / "cache"
READ_CACHES = [CACHE] + ([S / "demo3" / "cache"] if _eng == "demo3" else [])
ENGINE_SHA = hashlib.sha1(b"".join((ENGINE_DIR / f).read_bytes() for f in ("dsinfer.py", "dspitch.py"))).hexdigest()

# demo-3 options (v67/config.json, banks.qixuan.options)
OPTIONS = {"slow_note": 0.25, "slow_max": 2.0, "voicing_floor": 6.0, "pitch_source": "bank", "expr": None,
           "pitch_steps": 10, "steps": 10, "var_steps": 10, "depth": 0.6, "velocity": 1.0}
LEAD_OPTS = {"vowel_breath_off": -3.0, "detune": 7.0}
SUPPORT_OPTS = LEAD_OPTS  # the double (as a second take) and the harmony
if _eng == "flow":
    # flow/config.json: "before" = the demo's lead options, "flow" = the options added on top (render_track_ds
    # keyword arguments). The lead gets all of them; the double and the harmony sing no breaths of their own.
    _cfg = json.loads((S / "flow" / "config.json").read_text())
    LEAD_OPTS = dict(LEAD_OPTS, **_cfg.get("flow", {}))
    SUPPORT_OPTS = {k: v for k, v in LEAD_OPTS.items() if k != "breath"}

LEAD = ("AI", "AI Verses", "AI Pre", "AI Bridge", "AI Breakdown", "AI Final")
CAST: dict[str, dict] = {
    "AI": {"gender": 0.0, "opts": LEAD_OPTS},
    "AI Double": {"gender": 0.45, "opts": SUPPORT_OPTS},
    "Harmony": {"gender": 0.35, "opts": SUPPORT_OPTS},
    "Aliens": {"gender": -0.6, "tone": 12},
    "Aliens 2": {"gender": -1.0, "opts": {"detune": 10.0},
                 "doubles": [{"shift": -12, "tone": 12, "gender": 0.4, "delay": 0.0, "detune": 10.0, "gain_db": -4.0}]},
    "Crowd": {"gender": -0.3, "tone": 12, "doubles": [{"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5}]},
    "Crowd 2": {"gender": -0.7, "tone": 12, "doubles": [{"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5}]},
    "Crowd 3": {"gender": 0.2, "doubles": [{"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5}]},
}
# the double made from the lead's take (chain4.adt) and its tone
DOUBLE = {"delay": 0.018, "depth": 0.003, "rate": 0.31, "detune_cents": 6.0,
          "dark": [["highshelf", 4500, -5.0, 0.7], ["peak", 3000, -2.0, 1.0]]}


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
    if not semitones:
        return track
    tr = copy.copy(track)
    tr.notes = [copy.copy(n) for n in track.notes]
    for n in tr.notes:
        n.pitch += int(semitones)
    return tr


class DSBackend:
    name = "diffsinger"

    def __init__(self, ds=None, cast=None, options=None, boost_db=None, threads=2, log=print, chain=True,
                 double=None, lead_chain=None, harmony_chain=None):
        self.ds = ds or dsinfer.DS(QIXUAN, VOCODER, threads=threads, seed=1)
        self.cast = cast if cast is not None else copy.deepcopy(CAST)
        self.options = dict(OPTIONS, **(options or {}))
        self.boost_db = boost_db or {}
        self.formant = FormantBackend()
        self.log = log
        self.rows = []
        self.chain = chain  # False: every channel leaves the backend dry (the renderer's strip is then left on)
        self.double = double or os.environ.get("PROD_DOUBLE", "adt")  # "adt" | "take"
        self.lead_chain = lead_chain or chain4.LEAD
        self.harmony_chain = harmony_chain or chain4.HARMONY
        self.use_cache = os.environ.get("PROD_NO_CACHE") is None
        self.info = {}  # track name -> chain info of the last render

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
        d = track.to_dict()
        for k in ("id", "name", "gain_db", "pan", "mute", "solo", "reverb_send", "engine"):
            d.pop(k, None)
        for n in d.get("notes", []):
            n.pop("id", None)
        blob = json.dumps({"track": d, "tempo": project.tempo.to_list(), "window": window, "char": c, "opts": opts,
                           "engine": ENGINE_SHA}, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()

    def sing(self, project, track, window=None, **over):
        """Dry take of a track with its character's settings: (audio from time 0, info).
        The window is widened to whole 4 s steps before singing, so that renders whose pre-roll differs by a
        fraction of a second (another reverb length) share their takes; a take stored for the exact window is used
        when there is one."""
        c = dict(self.character(track), **over)
        opts = dict(self.options, **c.get("opts", {}))
        opts.setdefault("seed", track.seed if track.seed is not None else 1)
        wide = None if window is None else (max(0.0, float(np.floor((window[0] - 0.01) / 4.0) * 4.0)),
                                            float(np.ceil(window[1] / 4.0) * 4.0))
        key = self._key(project, track, wide, c, opts)
        if self.use_cache:
            for k in (self._key(project, track, window, c, opts), key):
                for d in READ_CACHES:
                    f = d / f"{k}.npy"
                    if f.exists():
                        info = json.loads(f.with_suffix(".json").read_text())
                        info["character"] = c
                        return np.load(f), info
        audio, info = self._sing(project, track, wide, c, opts)
        if self.use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            np.save(CACHE / f"{key}.npy", audio)
            (CACHE / f"{key}.json").write_text(json.dumps(
                {k: v for k, v in info.items() if k not in ("phones", "character", "dump")}, default=str))
        return audio, info

    def _sing(self, project, track, window, c, opts):
        tr = shifted(track, c.get("shift", 0))
        audio, info = dsinfer.render_track_ds(self.ds, project, tr, gender=float(c.get("gender", 0.0)),
                                              tone=float(c.get("tone", 0.0)), window=window, **opts)
        audio = audio.astype(np.float32) * 10 ** (c.get("gain_db", 0.0) / 20)
        for i, d in enumerate(c.get("doubles", []) or []):
            o = dict(opts, seed=int(opts["seed"]) + 101 * (i + 1), detune=float(d.get("detune", 0.0)))
            tr2 = shifted(track, c.get("shift", 0) + d.get("shift", 0))
            a2, _ = dsinfer.render_track_ds(self.ds, project, tr2, gender=float(c.get("gender", 0.0)) + d.get("gender", 0.0),
                                            tone=float(c.get("tone", 0.0)) + d.get("tone", 0.0), window=window, **o)
            k = int(round(d.get("delay", 0.0) * dsinfer.SR))
            a2 = np.concatenate([np.zeros(k, np.float32), a2.astype(np.float32)])[: len(audio)]
            ra, rb = active_rms(audio, dsinfer.SR), active_rms(a2, dsinfer.SR)
            audio[: len(a2)] += a2 * (ra / rb if ra > 0 and rb > 0 and i == 0 else 1.0) * 10 ** (d.get("gain_db", 0.0) / 20)
        info["character"] = c
        return audio, info

    def _window(self, audio, sr, t0, t1):
        i0 = int(round(t0 * sr))
        n = int(round((t1 - t0) * sr))
        seg = audio[max(0, i0): i0 + n]
        out = np.zeros(n, np.float32)
        out[: len(seg)] = seg
        return out

    def _lead_copy(self, project, track, sr, t0, t1):
        """The lead's take (all lead channels, dry) over the notes of `track`, 60 ms before to 120 ms after each."""
        n = int(round((t1 - t0) * sr))
        out = np.zeros(n, np.float32)
        for tr in project.tracks:
            if tr.name in LEAD and tr.notes:
                a, _ = self.sing(project, tr, window=(t0, t1))
                out += self._window(a, sr, t0, t1)
        b2s = project.tempo.beat_to_sec
        gate = np.zeros(n, np.float32)
        for nt in track.notes:
            a = int((b2s(nt.onset) - 0.06 - t0) * sr)
            b = int((b2s(nt.onset + nt.duration) + 0.12 - t0) * sr)
            gate[max(0, a): max(0, min(n, b))] = 1.0
        k = int(0.02 * sr)  # 20 ms fades
        gate = np.convolve(gate, np.hanning(2 * k + 1) / np.hanning(2 * k + 1).sum(), mode="same")
        return out * gate.astype(np.float32)

    def render_track(self, project, track, sample_rate, t0, t1, note_range=None):
        assert sample_rate == dsinfer.SR, "this backend renders at 44.1 kHz only"
        sr = sample_rate
        ref, report = self.formant.render_track(project, track, sr, t0, t1, note_range)
        n = int(round((t1 - t0) * sr))
        lead_copy = self.chain and self.double == "adt" and track.name == "AI Double"
        if lead_copy:
            c = dict(self.character(track), doubles=[])
            info = {"phrases": 0, "audio_s": 0.0, "pitch_source": "lead take"}
            out = self._lead_copy(project, track, sr, t0, t1)
        else:
            audio, info = self.sing(project, track, window=(t0, t1))
            c = info["character"]
            out = self._window(audio, sr, t0, t1)
        # loudness lane as a gain ride
        tc = t0 + np.arange(0, n, 441) / sr
        ld = evaluate_curve(track.curves.get("loudness"), tc, project.tempo.beat_to_sec, 0.0)
        out *= np.interp(np.arange(n), np.arange(0, n, 441), 10 ** (np.clip(ld, -48, 12) / 20)).astype(np.float32)
        cinfo = {}
        if self.chain and np.any(out):
            if track.name in LEAD:
                out, cinfo = chain4.process(out, sr, self.lead_chain)
            elif track.name.startswith("Harmony"):
                out, cinfo = chain4.process(out, sr, self.harmony_chain)
            elif lead_copy:
                out, cinfo = chain4.process(out, sr, self.harmony_chain)
                out = chain4.adt(out, sr, **DOUBLE)
        self.info[track.name] = cinfo
        # same active level as the formant engine's take, so the song's faders keep their meaning
        a, b = active_rms(ref, sr), active_rms(out, sr)
        g = 1.0
        if a > 0 and b > 0:
            g = a / b * 10 ** (self.boost_db.get(track.name, self.boost_db.get(c["character"], 0.0)) / 20)
            out = out * np.float32(g)
        report = dict(report, engine="diffsinger", peak_db=float(20 * np.log10(max(1e-9, np.max(np.abs(out))))))
        row = {"track": track.name, "voice": track.voice, "style": track.style, "character": c["character"],
               "gender": c.get("gender", 0.0), "shift": c.get("shift", 0), "tone": c.get("tone", 0.0),
               "doubles": len(c.get("doubles", []) or []), "phrases": info.get("phrases", 0),
               "sung_s": round(info.get("audio_s", 0.0), 1), "gain_db": round(20 * np.log10(max(g, 1e-9)), 1),
               "pitch_source": info.get("pitch_source"), "chain": cinfo}
        self.rows.append(row)
        self.log(f"    [ds] {track.name}: {row['phrases']} phrases, {row['sung_s']}s sung, gender {row['gender']:+.2f} "
                 f"doubles {row['doubles']} gain {row['gain_db']:+.1f} dB pitch {row['pitch_source']}"
                 + (f" chain {json.dumps({k: v for k, v in cinfo.items() if k in ('rider', 'comp1', 'comp2', 'bursts2', 'gr_speed')})}"
                    if cinfo else ""))
        return out.astype(np.float32), report
