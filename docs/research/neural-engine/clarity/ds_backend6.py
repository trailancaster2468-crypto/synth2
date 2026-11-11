"""Backend of the clarity study (copy of mix5/ds_backend5.py). What differs from the fourth demo's backend:
* the engine is clar/dsinfer.py (mix5's engine plus the clarity options, all off by default) and the chain is
  clar/chain5.py (prod/chain4.py plus the stages `lift` and `trans`),
* the dial: CLAR_DIAL=A|B|C (default A = the fourth demo) picks the lead's engine options and chain from
  clar/config.json ("dial"); the doubles are made from the lead's take as before, the harmony and the lead's answer
  lines sing with the lead's consonant options, without breaths,
* takes are written to clar/cache (read from clar/cache, mix5/cache, sep/cache); a lead take is stored with its
  alignment (KEY.dump.json), and a lead take found in a cache without one is sung again (the engine reproduces a
  take bit for bit), so every lead take can be measured against its phoneme times.

The fourth demo's backend: the engine of sep/ (flow engine + the `adlib` option;
dsinfer.py, dspitch.py and config.json are copies), the cast of the one-kind-per-track arrangement of
mix5/build5.py, and the roles: what a channel gets inside the channel for its role (ROLE_FX: tone, a slap, a far
room, the duck under the lead) and the copies made from another channel's take (COPIES: the two doubles, the
harmony's left side).

The rest is prod/ds_backend4.py:

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
P = S / "clar"
_eng = "flow"
ENGINE_DIR = P
sys.path.insert(0, str(S / "prod"))
sys.path.insert(0, str(ENGINE_DIR))

import dsinfer  # noqa: E402  (the engine copy in ENGINE_DIR)

assert dsinfer.__file__ == str(P / "dsinfer.py"), dsinfer.__file__
import chain5 as chain4  # noqa: E402
from synth2.engine import mixing as mx  # noqa: E402
from synth2.engine.expression import evaluate_curve  # noqa: E402
from synth2.engine.renderer import FormantBackend  # noqa: E402
from synth2.voices import VOICES  # noqa: E402

QIXUAN = S / "models/installed/qixuan/files/Qixuan_v2.7.0_DiffSinger_OpenUtau"
VOCODER = S / "models/installed/pc-nsf-hifigan/files/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
CACHE = P / "cache"
READ_CACHES = [CACHE, S / "mix5" / "cache", S / "sep" / "cache"]  # takes are written to clar/cache only
# Takes are keyed by the hash of sep/'s engine files: clar/dsinfer.py is that engine plus options that are off
# unless a channel's options set them (the options are part of the key), so the other channels' takes are reused.
ENGINE_SHA = hashlib.sha1(b"".join((S / "sep" / f).read_bytes() for f in ("dsinfer.py", "dspitch.py"))).hexdigest()

# demo-3 options (v67/config.json, banks.qixuan.options)
OPTIONS = {"slow_note": 0.25, "slow_max": 2.0, "voicing_floor": 6.0, "pitch_source": "bank", "expr": None,
           "pitch_steps": 10, "steps": 10, "var_steps": 10, "depth": 0.6, "velocity": 1.0}
LEAD_OPTS = {"vowel_breath_off": -3.0, "detune": 7.0}
SUPPORT_OPTS = LEAD_OPTS  # the double (as a second take) and the harmony
if _eng == "flow":
    # flow/config.json: "before" = the demo's lead options, "flow" = the options added on top (render_track_ds
    # keyword arguments). The lead gets all of them; the double and the harmony sing no breaths of their own.
    _cfg = json.loads((P / "config.json").read_text())
    LEAD_OPTS = dict(LEAD_OPTS, **_cfg.get("flow", {}))
    # the dial: engine options merged over the fourth demo's (None removes a key), chain stages merged over
    # chain5.LEAD4 (None switches a stage off; a stage the fourth demo has not is added before "breaths")
    DIAL = os.environ.get("CLAR_DIAL", "A")
    _dial = (_cfg.get("dial") or {}).get(DIAL) or {}
    if os.environ.get("CLAR_DIAL_JSON"):  # a dial setting given directly (experiments)
        _dial = json.loads(os.environ["CLAR_DIAL_JSON"])
    LEAD_OPTS = {k: v for k, v in dict(LEAD_OPTS, **(_dial.get("engine") or {})).items() if v is not None}
    SUPPORT_OPTS = {k: v for k, v in LEAD_OPTS.items() if k != "breath"}


LEAD_CHAIN = chain4.dial_chain(_dial.get("chain"))

LEAD = ("AI", "AI Verses", "AI Pre", "AI Bridge", "AI Breakdown", "AI Final")
_AD = _cfg.get("adlib", {})  # engine options of the ad-lib kinds (render_track_ds `adlib`)
# Engine options of a short answer line: the lead's, without breaths.
ANSWER_OPTS = SUPPORT_OPTS
CAST: dict[str, dict] = {
    "AI": {"gender": 0.0, "opts": LEAD_OPTS},
    "AI Double": {"gender": 0.45, "opts": SUPPORT_OPTS},
    "Harmony": {"gender": 0.35, "opts": SUPPORT_OPTS},
    "AI Answer": {"gender": 0.2, "opts": ANSWER_OPTS},
    "AI Adlib": {"gender": 0.3, "opts": {"adlib": _AD.get("AI Adlib")}},
    "Aliens": {"gender": -0.6, "tone": 12},
    "Alien Adlib": {"gender": -0.75, "tone": 12, "opts": {"adlib": _AD.get("Alien Adlib")}},
    "Alien Chant": {"gender": -0.6, "tone": 12, "opts": {"adlib": _AD.get("Alien Chant")}},
    "Alien Chant 2": {"gender": -1.0, "opts": {"detune": 10.0, "adlib": _AD.get("Alien Chant")},
                      "doubles": [{"shift": -12, "tone": 12, "gender": 0.4, "delay": 0.0, "detune": 10.0, "gain_db": -4.0}]},
    "Aliens 2": {"gender": -1.0, "opts": {"detune": 10.0},
                 "doubles": [{"shift": -12, "tone": 12, "gender": 0.4, "delay": 0.0, "detune": 10.0, "gain_db": -4.0}]},
    "Crowd": {"gender": -0.3, "tone": 12, "doubles": [{"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5}]},
    "Crowd 2": {"gender": -0.7, "tone": 12, "doubles": [{"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5}]},
    "Crowd 3": {"gender": 0.2, "doubles": [{"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5}]},
    "Crowd Shout": {"gender": -0.3, "tone": 12, "opts": {"adlib": _AD.get("Crowd Shout")},
                    "doubles": [{"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5}]},
    "Crowd Shout 2": {"gender": -0.7, "tone": 12, "opts": {"adlib": _AD.get("Crowd Shout")},
                      "doubles": [{"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5}]},
    "Crowd Shout 3": {"gender": 0.2, "opts": {"adlib": _AD.get("Crowd Shout")},
                      "doubles": [{"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5}]},
}
# the one-kind channels of build5.MOVES6: each keeps the voice setting of the channel its notes came from
_AL = {"tone": 12, "opts": {"adlib": _AD.get("Alien Adlib")}}
CAST.update({
    "AI Click": CAST["AI Adlib"], "AI Hai": CAST["AI Adlib"], "AI Stutter": dict(CAST["AI Adlib"], gender=0.0),
    "Alien Answer": CAST["Aliens"], "Alien Unison": CAST["Aliens"],
    "Alien Click": dict(_AL, gender=-0.75, opts={"adlib": dict(_AD.get("Alien Adlib") or {}, salt={"click": 2})}),
    "Alien Ouch": dict(_AL, gender=-0.75), "Alien Laugh": dict(_AL, gender=-0.75),
    "Alien Nande": dict(_AL, gender=-0.6), "Alien Stutter": dict(_AL, gender=-0.6),
    "Crowd Click": CAST["Crowd Shout"], "Crowd Click 2": CAST["Crowd Shout 2"], "Crowd Click 3": CAST["Crowd Shout 3"],
})
# Takes of the stamped "click" chosen by audition (audition.py, audition.json: salts 0-3 sung dry, "click" counted
# in three recognisers' transcripts of bar 52 and of the crowd's pairs in bars 49-51). A stamped hit repeats one
# take, so the take is worth choosing: the aliens' take 2 is heard as "click" x4 by all three (take 0 by none),
# the crowd's first and third voice take 1 (pairs 9 / 6 / 6 of 6 against 7 / 6 / 0). The second crowd voice and
# the lead keep take 0 (the lead's click is heard as "click" in none of ten takes).
for _k, _n in (("Crowd Click", 1), ("Crowd Click 3", 1)):
    CAST[_k] = dict(CAST[_k], opts={"adlib": dict(CAST[_k]["opts"]["adlib"] or {}, salt={"click": _n})})
if os.environ.get("SEP_CAST"):  # JSON merged into the cast, e.g. '{"AI Adlib": {"opts": {"adlib": {"stamp": false}}}}'
    for _k, _v in json.loads(os.environ["SEP_CAST"]).items():
        _c = CAST.setdefault(_k, {})
        for _a, _b in _v.items():
            if _a == "opts":
                _o = dict(_c.get("opts") or {})
                for _x, _y in _b.items():
                    _o[_x] = dict(_o.get(_x) or {}, **_y) if isinstance(_y, dict) and isinstance(_o.get(_x), dict) else _y
                _c["opts"] = _o
            else:
                _c[_a] = _b
# the split channels of the ad-lib tracks (build5.SPLITS5): the stutter before "reboot" is the lead's own word and
# keeps the lead's voice; the aliens' ad-libs in the verse and the bridge keep the aliens' voice of those lines
CAST["AI Adlib Bridge"] = dict(CAST["AI Adlib"], gender=0.0)
CAST["Alien Adlib Bridge"] = dict(CAST["Alien Adlib"], gender=-0.6)
# static part of the lead's chain for the lead's ad-libs (EQ only: a gain that moves would make the copies differ)
ADLIB_CHAIN = {"eq": chain4.LEAD["eq"]}
# the double made from the lead's take (chain4.adt) and its tone
DOUBLE = {"delay": 0.018, "depth": 0.003, "rate": 0.31, "detune_cents": 6.0,
          "dark": [["highshelf", 4500, -5.0, 0.7], ["peak", 3000, -2.0, 1.0]]}
DOUBLE_R = dict(DOUBLE, delay=0.027, rate=0.23, detune_cents=-5.0, phase=1.6)
HARMONY_L = {"delay": 0.022, "depth": 0.003, "rate": 0.27, "detune_cents": -7.0, "phase": 0.8, "dark": None}
# channels made from other channels' takes: name -> (source channels, chain, adt settings)
COPIES = {
    "AI Double": (LEAD, "harmony", DOUBLE), "AI Double R": (LEAD, "harmony", DOUBLE_R),
    "Harmony L": (("Harmony",), "harmony", HARMONY_L), "Harmony Final L": (("Harmony Final",), "harmony", HARMONY_L),
}
# What a role gets inside its channel, after the level match (so the dry level keeps the fader's meaning), in this
# order: eq (tone), duck_db (gain reduction while the lead sings), slap (one repeat), room (a small mono room
# with a pre-delay, wet level under the dry). Pan, fader and sends are build5's.
_BAND = [["hp", 330, 0, 0.7071, 2], ["lp", 5200, 0, 0.7071, 2]]
ROLE_FX = {
    "lead": {}, "double": {}, "stutter": {}, "alien stutter": {},
    "harmony": {"eq": [["peak", 2800, -3.0, 0.9]], "room": {"decay": 0.28, "predelay": 0.030, "level_db": -13.0}},
    "answer": {"eq": [["highshelf", 6000, -1.5, 0.7]], "room": {"decay": 0.22, "predelay": 0.020, "level_db": -16.0}},
    "adlib": {"eq": _BAND, "duck_db": 2.0, "slap": {"time": 0.095, "level_db": -9.0, "lp": 3200}},
    # no slap on the aliens' lines and answers: a repeat 125 ms later at -14 dB cost words on their verse-2 lines
    # (tiny.en / base.en / small.en WER 0.30 / 0.26 / 0.12 without it, 0.33 / 0.33 / 0.16 with it; 43 words)
    "alien lead": {},
    "alien answer": {"duck_db": 1.5},
    "alien backing": {"eq": [["highshelf", 3800, -4.0, 0.7]], "duck_db": 3.0,
                      "room": {"decay": 0.30, "predelay": 0.030, "level_db": -12.0}},
    "alien adlib": {"eq": [["hp", 260, 0, 0.7071, 2], ["lp", 4600, 0, 0.7071, 2]], "duck_db": 2.0,
                    "slap": {"time": 0.110, "level_db": -8.0, "lp": 2600}},
    # runs of one sound on eighth notes: a repeat 95 to 110 ms later would land between the hits and blur them,
    # so the clicks and the laugh get no slap (the laugh is thrown into the song's delay after it instead)
    "click": {"eq": _BAND, "duck_db": 2.0},
    "alien click": {"eq": [["hp", 260, 0, 0.7071, 2], ["lp", 4600, 0, 0.7071, 2]], "duck_db": 2.0},
    "alien laugh": {"eq": [["hp", 260, 0, 0.7071, 2]], "duck_db": 2.0},
    "alien chant": {"eq": [["highshelf", 4200, -3.0, 0.7]], "duck_db": 2.5,
                    "room": {"decay": 0.30, "predelay": 0.030, "level_db": -12.0}},
    "crowd": {"eq": [["hp", 170, 0, 0.7071, 2], ["highshelf", 3800, -4.0, 0.7]], "duck_db": 3.0,
              "room": {"decay": 0.38, "predelay": 0.038, "level_db": -9.0}},
    "crowd shout": {"eq": [["hp", 220, 0, 0.7071, 2], ["highshelf", 5000, -2.0, 0.7]], "duck_db": 1.5,
                    "room": {"decay": 0.38, "predelay": 0.038, "level_db": -11.0}},
}
if os.environ.get("MIX5_FX"):  # JSON merged into ROLE_FX, e.g. '{"crowd": {"duck_db": 0}}'; "off" = no role effects
    if os.environ["MIX5_FX"] == "off":
        ROLE_FX = {}
    else:
        for _k, _v in json.loads(os.environ["MIX5_FX"]).items():
            ROLE_FX[_k] = dict(ROLE_FX.get(_k, {}), **_v)


def slap(x, sr, time=0.1, level_db=-10.0, lp=3000.0):
    """x plus one band-limited repeat `time` s later, `level_db` under it."""
    k = int(round(time * sr))
    y = np.zeros(len(x), np.float64)
    y[k:] = x[: len(x) - k]
    y = mx.eq(y, [["hp", 300, 0, 0.7071, 2], ["lp", lp, 0, 0.7071, 2]], sr)
    return (x + y * 10 ** (level_db / 20)).astype(np.float32)


def save_take(path, audio):
    """A take as its sounding span in 16 bit with its place and peak (a take is an array from time 0 and mostly
    silence: 33 MB as float32 for a window at the end of the song)."""
    nz = np.flatnonzero(audio)
    a, b = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)
    pk = float(np.max(np.abs(audio[a:b]))) if b > a else 1.0
    np.savez(path, a=a, n=len(audio), pk=pk, x=np.round(audio[a:b] / pk * 32767).astype(np.int16))


def load_take(path):
    if path.suffix == ".npy":
        return np.load(path)
    d = np.load(path)
    out = np.zeros(int(d["n"]), np.float32)
    x = d["x"]
    out[int(d["a"]): int(d["a"]) + len(x)] = x.astype(np.float32) * (float(d["pk"]) / 32767.0)
    return out


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
        self.lead_chain = lead_chain or LEAD_CHAIN
        self.dumps = {}  # lead track name -> alignment dumps of its takes (file paths)
        self.harmony_chain = harmony_chain or chain4.HARMONY
        self.use_cache = os.environ.get("PROD_NO_CACHE") is None
        self.info = {}  # track name -> chain info of the last render
        self._env = {}  # window -> where the lead sings

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
        if opts.get("breath") and track.name in LEAD:
            # the lead takes no breath where an ad-lib of its own voice sounds (0.12 s before each ad-lib note to its end)
            b2s = project.tempo.beat_to_sec
            busy = sorted((round(b2s(n.onset) - 0.12, 3), round(b2s(n.onset + n.duration), 3))
                          for t in project.tracks if t.name.startswith(("AI Adlib", "AI Click", "AI Hai", "AI Stutter"))
                          for n in t.notes)
            if busy:
                opts["breath"] = dict(opts["breath"], avoid=[list(x) for x in busy])
        wide = None if window is None else (max(0.0, float(np.floor((window[0] - 0.01) / 4.0) * 4.0)),
                                            float(np.ceil(window[1] / 4.0) * 4.0))
        key = self._key(project, track, wide, c, opts)
        if self.use_cache:
            for k in (self._key(project, track, window, c, opts), key):
                for d in READ_CACHES:
                    for f in (d / f"{k}.npz", d / f"{k}.npy"):
                        if f.exists() and f.with_suffix(".json").exists():
                            dj = d / f"{k}.dump.json"
                            if track.name in LEAD and not dj.exists():
                                continue  # a lead take without its alignment: sung again below
                            info = json.loads(f.with_suffix(".json").read_text())
                            info["character"] = c
                            if track.name in LEAD:
                                self.dumps.setdefault(track.name, set()).add(str(dj))
                            return load_take(f), info
        audio, info = self._sing(project, track, wide, c, opts)
        if self.use_cache:
            CACHE.mkdir(parents=True, exist_ok=True)
            save_take(CACHE / f"{key}.npz", audio)
            if info.get("dump"):
                r = lambda q: [round(float(v), 2) for v in q]
                (CACHE / f"{key}.dump.json").write_text(json.dumps([
                    {"f_start": int(ph["f_start"]), "seq": ph["seq"], "durs": [int(v) for v in ph["durs"]],
                     "vowel": [bool(v) for v in ph["vowel"]]} for ph in info["dump"]]))
                self.dumps.setdefault(track.name, set()).add(str(CACHE / f"{key}.dump.json"))
            (CACHE / f"{key}.json").write_text(json.dumps(
                {k: v for k, v in info.items() if k not in ("phones", "character", "dump", "adlib")}, default=str))
        return audio, info

    def _sing(self, project, track, window, c, opts):
        tr = shifted(track, c.get("shift", 0))
        audio, info = dsinfer.render_track_ds(self.ds, project, tr, gender=float(c.get("gender", 0.0)),
                                              tone=float(c.get("tone", 0.0)), window=window,
                                              dump=track.name in LEAD, **opts)
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

    def _lead_env(self, project, sr, t0, t1):
        """0..1 per sample: where the lead sings (its dry takes within 25 dB of their loud level; 60 ms hold,
        30 ms attack, 180 ms release)."""
        key = (round(t0, 4), round(t1, 4))
        if key not in self._env:
            n = int(round((t1 - t0) * sr))
            x = np.zeros(n, np.float32)
            for tr in project.tracks:
                if tr.name in LEAD and tr.notes:
                    a, _ = self.sing(project, tr, window=(t0, t1))
                    x += self._window(a, sr, t0, t1)
            hop = 441
            nb = max(1, n // hop)
            e = np.sqrt(np.mean(x[: nb * hop].reshape(nb, hop).astype(np.float64) ** 2, axis=1) + 1e-12)
            lv = 20 * np.log10(e)
            on = np.zeros(nb)
            if np.any(lv > -80):
                on = (lv > np.percentile(lv[lv > -80], 90) - 25.0).astype(np.float64)
            hold = 6
            on = np.max(np.pad(on, (hold, 0))[np.arange(hold + 1)[None, :] + np.arange(nb)[:, None]], axis=1)
            g = np.zeros(nb)
            v = 0.0
            for i in range(nb):
                c = 1 - np.exp(-0.01 / (0.03 if on[i] > v else 0.18))
                v += (on[i] - v) * c
                g[i] = v
            self._env[key] = np.interp(np.arange(n), np.arange(nb) * hop + hop / 2, g).astype(np.float32)
        return self._env[key]

    def _role_fx(self, project, track, out, sr, t0, t1):
        """The role's in-channel effects (ROLE_FX) on the level-matched channel."""
        import build5
        role = build5.role_of(track.name)
        fx = ROLE_FX.get(role) or {}
        if role == "lead" and track.name not in LEAD:
            fx = {}
        if not fx or not np.any(out):
            return out, role
        if fx.get("eq"):
            out = mx.eq(out.astype(np.float64), fx["eq"], sr).astype(np.float32)
        if fx.get("duck_db"):
            out = out * (10 ** (-fx["duck_db"] * self._lead_env(project, sr, t0, t1) / 20)).astype(np.float32)
        if fx.get("slap"):
            out = slap(out, sr, **fx["slap"])
        if fx.get("room"):
            r = fx["room"]
            wet = chain4.room(out, sr, decay=r["decay"], predelay=r["predelay"], level_db_=r["level_db"])
            if r.get("duck_db"):  # the room steps back while the lead sings and blooms in the gaps
                wet = wet * (10 ** (-r["duck_db"] * self._lead_env(project, sr, t0, t1) / 20)).astype(np.float32)
            out = out + wet
        return out.astype(np.float32), role

    def _lead_copy(self, project, track, sr, t0, t1, sources=LEAD):
        """The take of the `sources` channels (dry) over the notes of `track`, 60 ms before to 120 ms after each."""
        n = int(round((t1 - t0) * sr))
        out = np.zeros(n, np.float32)
        for tr in project.tracks:
            if tr.name in sources and tr.notes:
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
        lead_copy = self.chain and self.double == "adt" and track.name in COPIES
        if lead_copy:
            c = dict(self.character(track), doubles=[])
            info = {"phrases": 0, "audio_s": 0.0, "pitch_source": "take of " + COPIES[track.name][0][0]}
            out = self._lead_copy(project, track, sr, t0, t1, COPIES[track.name][0])
        else:
            audio, info = self.sing(project, track, window=(t0, t1))
            c = info["character"]
            out = self._window(audio, sr, t0, t1)
        # loudness lane as a gain ride
        tc = t0 + np.arange(0, n, 441) / sr
        ld = evaluate_curve(track.curves.get("loudness"), tc, project.tempo.beat_to_sec, 0.0)
        if ((c.get("opts") or {}).get("adlib") or {}).get("chop"):
            # the engine chops every take itself, locked to its start: of the loudness lane (the fan chop of
            # vocals.py on top of any fade) only the upper envelope is applied, a running maximum over 0.13 s
            k = 6
            pad = np.pad(ld, k, mode="edge")
            ld = np.max(pad[np.arange(2 * k + 1)[None, :] + np.arange(len(ld))[:, None]], axis=1)
        out *= np.interp(np.arange(n), np.arange(0, n, 441), 10 ** (np.clip(ld, -48, 12) / 20)).astype(np.float32)
        cinfo = {}
        if self.chain and np.any(out):
            if track.name in LEAD:
                out, cinfo = chain4.process(out, sr, self.lead_chain)
            elif track.name.startswith("Harmony") and not lead_copy:
                out, cinfo = chain4.process(out, sr, self.harmony_chain)
            elif track.name.startswith("AI Answer"):
                out, cinfo = chain4.process(out, sr, self.lead_chain)
            elif track.name.startswith(("AI Adlib", "AI Click", "AI Hai", "AI Stutter")):
                out, cinfo = chain4.process(out, sr, ADLIB_CHAIN)
            elif lead_copy:
                out, cinfo = chain4.process(out, sr, self.harmony_chain)
                out = chain4.adt(out, sr, **COPIES[track.name][2])
        self.info[track.name] = cinfo
        # same active level as the formant engine's take, so the song's faders keep their meaning
        a, b = active_rms(ref, sr), active_rms(out, sr)
        g = 1.0
        if a > 0 and b > 0:
            g = a / b * 10 ** (self.boost_db.get(track.name, self.boost_db.get(c["character"], 0.0)) / 20)
            out = out * np.float32(g)
        out, role = self._role_fx(project, track, out, sr, t0, t1)
        report = dict(report, engine="diffsinger", peak_db=float(20 * np.log10(max(1e-9, np.max(np.abs(out))))))
        row = {"track": track.name, "role": role, "voice": track.voice, "style": track.style, "character": c["character"],
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
