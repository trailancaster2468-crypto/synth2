"""DiffSinger (OpenVPI) ONNX inference for any bank in OpenUtau layout: Synth2 plan -> phonemes/durations -> mel -> wav.

`q4/dsinfer4.py` (speaker embeds, tension, missing variance, bundled vocoder) plus four options of
`render_track_ds`, each off by default. With all four off the models are fed exactly what `dsinfer4.py` feeds them.

    slow_note=0.25       slow-motion rendering: a note shorter than this many seconds is sung up to `slow_max` times
                         slower by the variance and acoustic models and the mel is squeezed back onto the real frame
                         grid; the vocoder gets the real f0, so timing and pitch are as written
    voicing_floor=6.0    a vowel frame's predicted voicing is held at or above the running median of the phrase's
                         vowel voicing minus this many dB (collapsed predictions are whispered syllables)
    pitch_source="bank"  the bank's own pitch predictor (`dspitch/`) at `expr` (None: by style, robotic 0.5, soft
                         0.75, others 1.0); Synth2's `build_pitch` is used for a bank without `dspitch/` and for any
                         phrase the predictor cannot sing
    seed=N               reproducible noise: every phrase gets fresh variance, acoustic, vocoder and pitch sessions
                         made right after `onnxruntime.set_seed` with a seed derived from N and the phrase's place
                         and length

A fifth option, not from the four studies and also off by default: `min_vowel=0.04` keeps a vowel that the
duration model squeezed to one or two frames between consonant clusters at 0.04 s (see `align`).

The flow options (all off by default; `render_track_ds` lists them, config.json has the values in use) make the
line continuous: `cons_cap` / `cons_cap_keep` / `onset_keep` shorten the consonants of fast notes in favour of the
vowels, `cons_voicing_floor` keeps the voice on through voiced consonants, `breath` puts a breath (AP) into the gaps
between lines and before a phrase. Tried and left off: `cons_scale`, `min_vowel_share`, `sp_gap`, `stop_breath`,
`fric_breath`, `level`.

The ad-lib option (`adlib`, off by default; ADLIB has the defaults, `render_adlib` the description) sings a track
of interjections, shouts, sound words or chant syllables hit by hit: every word is its own take with silence
around it, a designed pitch, no breath and no glide from its neighbours, and (with `stamp`) one take per distinct
sound pasted at every repetition.

Follows OpenUtau's DiffSinger pipeline (phonemizer alignment, pitch, variance, acoustic, vocoder).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/user/synth2")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import onnxruntime as ort
import yaml

import dspitch
from synth2.engine.expression import build_pitch, evaluate_curve, get_style
from synth2.engine.planner import plan_track
from synth2.project import CURVE_SPECS
from synth2.voices import get_voice

SR = 44100
HOP = 512
FRAME = HOP / SR
HEAD = 8  # frames of SP before a phrase
TAIL = 10  # and after it
PHRASE_GAP = 0.30  # s of silence that splits phrases
SP_GAP = 0.045  # s: shorter gaps inside a phrase are sung through (legato)
PAD = 0.5  # s given to the duration model before the first note

SLOW_NOTE = 0.25  # s: recommended `slow_note`
SLOW_MAX = 2.0  # largest slow-motion factor
VOICING_FLOOR = 6.0  # dB: recommended `voicing_floor`
VOICING_WINDOW = 1.5  # s of vowel time the floor's running median covers
MIN_VOWEL = 0.04  # s: `min_vowel` value tried (3 to 4 frames)
MIN_VOWEL_SHARE = 0.5  # of its note, the most `min_vowel` gives a vowel

# How much of the learned pitch expression each Synth2 style asks for. Below 0.5 the predictor returns plain note
# steps; vibrato and drift appear from about 0.75.
STYLE_EXPR = {"pop": 1.0, "ballad": 1.0, "rock": 1.0, "jazz": 1.0, "rnb": 1.0, "classical": 1.0, "soft": 0.75,
              "robotic": 0.5}
MIN_EXPR = 0.5
SYNTHETIC_VIBRATO_STYLES = ("robotic",)
VIB_ATTRS = ("vibrato_depth", "vibrato_rate", "vibrato_start")
TRANS_ATTRS = ("transition", "overshoot")
MAX_DEV = 6.0  # semitones a predicted sung frame may lie outside the notes within 0.25 s: further out is an outlier
MAX_OUTLIERS = 0.1  # share of a phrase's sung frames that may be outliers; with more the phrase is rejected
SUSTAIN = dict(min_note=0.35, head=0.12, tail=0.05, ramp=0.06)

# Consonant classes by phoneme name without the language prefix (English and Japanese sets of the banks).
CONS_CLASS = {}
for _k, _v in {"stop": "p t k ky kw py ty cl", "vstop": "b d g dx jh by dy gy gw", "aff": "ch ts",
               "fric": "s sh f th hh h hy", "vfric": "z zh v dh j", "nasal": "m n ng my ny",
               "glide": "l r w y _r ry"}.items():
    for _p in _v.split():
        CONS_CLASS[_p] = _k
UNVOICED = ("stop", "aff", "fric")
VOICED_OBSTRUENT = ("vstop", "vfric")

# Recommended values of the flow options (the same as config.json's "flow").
CONS_CAP = {"stop": 0.035, "vstop": 0.035, "aff": 0.06, "fric": 0.05, "vfric": 0.05, "nasal": 0.06, "glide": 0.05}
CONS_CAP_KEEP = 0.5
CONS_VOICING_FLOOR = {"vfric": 8.0, "nasal": 8.0, "glide": 8.0, "vstop": 11.0}
BREATH = {"min_gap": 0.12, "fill": 0.45, "length": 0.30, "min_len": 0.10, "breathiness": -44.0, "first": True}


def cons_class(p):
    """Class of a bank phoneme ("en/s" -> "fric"); None for vowels, SP and AP."""
    if p in ("SP", "AP"):
        return None
    return CONS_CLASS.get(p.split("/")[-1])


JA_VOWELS = {"a", "i", "u", "e", "o", "N"}
JA_MAP = {"c": "h", "ph": "f", "dz": "z", "hh": "h", "jh": "j", "q": "cl", "l": "r", "dx": "r", "v": "v"}


def to_ds(ph: str, ja: bool) -> str:
    if ph == "br":
        return "AP"
    if ph in ("sil", "q") and not ja:
        return "SP"
    if ph == "sil":
        return "SP"
    if ja:
        return "ja/" + JA_MAP.get(ph, ph)
    return "en/" + ph


class Timer:
    def __init__(self):
        self.t = {}

    def add(self, k, dt):
        self.t[k] = self.t.get(k, 0.0) + dt


class DS:
    """Generic DiffSinger bank in OpenUtau layout. Feeds are built from each session's own input list."""

    def __init__(self, root, vocoder=None, threads=2, speaker=None, seed=1):
        root = Path(root)
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.log_severity_level = 3
        self.so = so
        self.threads = threads
        self.speaker = speaker
        self.timer = Timer()
        self.root = root
        ort.set_seed(seed)

        def load(d):
            cfg = yaml.safe_load((d / "dsconfig.yaml").read_text(encoding="utf-8-sig"))
            ph = json.loads((d / cfg["phonemes"]).read_text(encoding="utf-8-sig"))
            lg = json.loads((d / cfg["languages"]).read_text(encoding="utf-8-sig")) if cfg.get("languages") else {}
            return cfg, ph, lg

        def embed(d, cfg, s):
            """Speaker embedding for session s (None when the model takes no spk_embed)."""
            dim = next((i.shape[-1] for i in s.get_inputs() if i.name == "spk_embed"), None)
            if dim is None:
                return None
            spks = cfg.get("speakers") or []
            pick = spks[0]
            if speaker:
                want = speaker.split("/")
                for sp in spks:
                    if sp == speaker or sp.split("/")[-len(want):] == want:
                        pick = sp
                        break
                else:
                    raise KeyError(f"speaker {speaker!r} not in {spks}")
            e = np.fromfile(str(d / (pick + ".emb")), "<f4")
            assert e.size == dim, (pick, e.size, dim)
            return e

        self.ac_cfg, self.ac_ph, self.ac_lg = load(root)
        self.ac_path = root / self.ac_cfg["acoustic"]
        self.acoustic = self._sess(self.ac_path)
        self.ac_emb = embed(root, self.ac_cfg, self.acoustic)
        d = root / "dsdur"
        self.du_cfg, self.du_ph, self.du_lg = load(d)
        self.du_ling = self._sess(d / self.du_cfg["linguistic"])
        self.du_dur = self._sess(d / self.du_cfg["dur"])
        self.du_emb = embed(d, self.du_cfg, self.du_dur)
        d = root / "dsvariance"
        self.va_var = None
        if (d / "dsconfig.yaml").exists():
            self.va_cfg, self.va_ph, self.va_lg = load(d)
            self.va_ling = self._sess(d / self.va_cfg["linguistic"])
            self.va_path = d / self.va_cfg["variance"]
            self.va_var = self._sess(self.va_path)
            self.va_emb = embed(d, self.va_cfg, self.va_var)
        if vocoder is None:
            vd = root / "dsvocoder"
            vcfg = yaml.safe_load((vd / "vocoder.yaml").read_text(encoding="utf-8-sig"))
            vocoder = vd / vcfg["model"]
        self.vocoder_path = str(vocoder)
        self.vocoder = self._sess(vocoder)
        self.max_depth = float(self.ac_cfg.get("max_depth", 1.0))
        self.ac_in = [i.name for i in self.acoustic.get_inputs()]
        self._pitch = False  # dspitch.PitchBank, None when the bank has none; loaded on first use

    def _sess(self, p):
        return ort.InferenceSession(str(p), self.so, providers=["CPUExecutionProvider"])

    @property
    def pitch(self):
        """The bank's pitch predictor, or None when the bank has no usable `dspitch/`."""
        if self._pitch is False:
            self._pitch = None
            if (self.root / "dspitch" / "dsconfig.yaml").exists():
                try:
                    self._pitch = dspitch.PitchBank(self.root / "dspitch", threads=self.threads, speaker=self.speaker)
                except (ValueError, KeyError, OSError):
                    self._pitch = None
        return self._pitch

    def reseed(self, seed):
        """Fresh sessions for the models that can draw noise inside the graph (variance, acoustic, vocoder).

        A session made right after `onnxruntime.set_seed(n)` gives the same output on its first run every time, so
        one call per phrase makes the phrase reproducible whatever was rendered before it. Some bundled vocoders
        draw noise too (Aurelian's `tgm_hifigan_v107`); a deterministic vocoder gives the same output either way.
        """
        t = time.time()
        seed = int(seed) & 0x7FFFFFFF
        if self.va_var is not None:
            ort.set_seed(seed)
            self.va_var = self._sess(self.va_path)
        ort.set_seed(seed)
        self.acoustic = self._sess(self.ac_path)
        ort.set_seed(seed)
        self.vocoder = self._sess(self.vocoder_path)
        self.timer.add("sessions", time.time() - t)

    @staticmethod
    def _tok(m, phones):
        return np.array([[m[p] for p in phones]], np.int64)

    @staticmethod
    def _lang(m, phones):
        return np.array([[m.get(p.split("/")[0], 0) if "/" in p else 0 for p in phones]], np.int64)

    @staticmethod
    def _feed(s, d):
        names = [i.name for i in s.get_inputs()]
        miss = [n for n in names if n not in d]
        assert not miss, f"no value for inputs {miss}"
        return {n: d[n] for n in names}

    def has(self, ph):
        return ph in self.ac_ph and ph in self.du_ph and (self.va_var is None or ph in self.va_ph)

    def predict_dur(self, phones, word_div, word_dur, ph_midi):
        t = time.time()
        d = {"tokens": self._tok(self.du_ph, phones), "languages": self._lang(self.du_lg, phones),
             "word_div": np.array([word_div], np.int64), "word_dur": np.array([word_dur], np.int64)}
        enc, mask = self.du_ling.run(None, self._feed(self.du_ling, d))
        d.update({"encoder_out": enc, "x_masks": mask, "ph_midi": np.array([ph_midi], np.int64)})
        if self.du_emb is not None:
            d["spk_embed"] = np.tile(self.du_emb[None, None], (1, len(phones), 1)).astype(np.float32)
        out = self.du_dur.run(None, self._feed(self.du_dur, d))[0][0]
        self.timer.add("dur", time.time() - t)
        return out.astype(np.float64)

    def predict_variance(self, phones, durs, pitch_semi, steps=10):
        """Returns {name: curve} for every variance the bank predicts (empty without dsvariance)."""
        if self.va_var is None:
            return {}
        t = time.time()
        n = len(pitch_semi)
        ph_dur = np.array([durs], np.int64)
        d = {"tokens": self._tok(self.va_ph, phones), "languages": self._lang(self.va_lg, phones), "ph_dur": ph_dur}
        enc = self.va_ling.run(None, self._feed(self.va_ling, d))[0]
        z = np.zeros((1, n), np.float32)
        outs = [o.name for o in self.va_var.get_outputs()]
        d.update({"encoder_out": enc, "pitch": pitch_semi[None].astype(np.float32), "energy": z, "breathiness": z,
                  "voicing": z, "tension": z, "retake": np.ones((1, n, len(outs)), bool),
                  "steps": np.array(steps, np.int64), "speedup": np.array(max(1, 1000 // steps), np.int64)})
        if self.va_emb is not None:
            d["spk_embed"] = np.tile(self.va_emb[None, None], (1, n, 1)).astype(np.float32)
        res = self.va_var.run(None, self._feed(self.va_var, d))
        self.timer.add("variance", time.time() - t)
        return {o[:-5] if o.endswith("_pred") else o: r[0] for o, r in zip(outs, res)}

    def mel(self, phones, durs, f0, var, gender=0.0, velocity=1.0, steps=10, depth=0.6, offs=None):
        """Acoustic model: mel float32 [frames, bins] for sum(durs) frames."""
        n = len(f0)
        offs = offs or {}
        t = time.time()
        g = np.broadcast_to(np.asarray(gender, np.float32), (n,))[None].astype(np.float32)
        v = np.broadcast_to(np.asarray(velocity, np.float32), (n,))[None].astype(np.float32)
        d = {"tokens": self._tok(self.ac_ph, phones), "languages": self._lang(self.ac_lg, phones),
             "durations": np.array([durs], np.int64), "f0": f0[None].astype(np.float32),
             "gender": g, "velocity": v, "depth": np.array(min(depth, self.max_depth), np.float32),
             "steps": np.array(steps, np.int64), "speedup": np.array(max(1, 1000 // steps), np.int64)}
        lim = {"breathiness": (-96, 0), "voicing": (-96, 0), "energy": (-96, 0), "tension": (-10, 10)}
        for k, (lo, hi) in lim.items():
            if k in self.ac_in:
                c = var.get(k)
                if c is None:
                    raise KeyError(f"acoustic model needs {k!r}, the variance model does not predict it")
                d[k] = np.clip(c + offs.get(k, 0.0), lo, hi)[None].astype(np.float32)
        if self.ac_emb is not None:
            d["spk_embed"] = np.tile(self.ac_emb[None, None], (1, n, 1)).astype(np.float32)
        mel = self.acoustic.run(None, self._feed(self.acoustic, d))[0]
        self.timer.add("acoustic", time.time() - t)
        return mel[0]

    def vocode(self, mel, f0):
        t = time.time()
        wav = self.vocoder.run(None, {"mel": mel[None].astype(np.float32), "f0": f0[None].astype(np.float32)})[0][0]
        self.timer.add("vocoder", time.time() - t)
        return wav

    def synth(self, phones, durs, f0_ac, f0_voc, var, gender=0.0, velocity=1.0, steps=10, depth=0.6, offs=None):
        mel = self.mel(phones, durs, f0_ac, var, gender=gender, velocity=velocity, steps=steps, depth=depth, offs=offs)
        return self.vocode(mel, f0_voc), mel


# ---------------------------------------------------------------------------------------------
def build_phrases(plan):
    """Split plan.units into phrases (lists of units) at silences >= PHRASE_GAP."""
    phrases, cur, prev_end = [], None, None
    for u in plan.units:
        st = u.start - (u.lead or 0.0)
        if cur is None or st - prev_end >= PHRASE_GAP:
            cur = []
            phrases.append(cur)
        cur.append(u)
        prev_end = u.end
    return phrases


def phrase_groups(units, plan, ds, sp_gap=SP_GAP, breath=None, prev_end=None):
    """OpenUtau-style word groups: each starts at a vowel / SP / AP; consonants trail the previous starter.

    Returns (groups, end) where a group is dict(pos, phones, midi) and `vowel=True` on groups a nucleus starts.
    sp_gap (s): a gap between two notes shorter than this is sung through.
    breath: None, or dict(min_gap, fill, length, first, min_len, breathiness). A gap of at least min_gap gets a
    breath (AP) that ends where the next note's consonants start: the whole gap up to `fill` s, the last `length`
    s of a longer one (the renderer turns a breath left shorter than min_len into a rest).
    The phrase itself starts with a breath of `length` s when the silence before it (prev_end = end of the
    phrase before, None = start of the track, breathed when `first`) has room for it.
    breath["avoid"]: [(start s, end s), ...] where no breath is put (an ad-lib of the same voice on another track
    sounds there): a breath that would overlap one of them is a rest.
    """
    by_index = {tm.index: tm.pitch for tm in plan.timings}

    def midi(u):
        # u.notes holds track note indices; plan.timings skips breaths and closures, so the note is looked up by
        # index (a breath takes the pitch of the nearest pitched note)
        if not u.notes or not by_index:
            return 60
        k = min(by_index, key=lambda i: (min(abs(i - j) for j in u.notes), i))
        return int(round(by_index[k]))

    def conv(ph, ja):
        d = to_ds(ph, ja)
        if not ds.has(d):
            alt = ("en/" if ja else "ja/") + ph
            d = alt if ds.has(alt) else "SP"
        return d

    def clear(a, b):
        return not any(a < y and x < b for x, y in (breath or {}).get("avoid", ()))

    first = units[0]
    groups = [dict(pos=None, phones=["SP"], midi=midi(first))]
    if breath and not first.breath:
        room = 1e9 if prev_end is None else first.start - (first.lead or 0.0) - prev_end - breath.get("release", 0.1)
        on = first.start - (first.lead or 0.0)
        if (prev_end is not None or breath.get("first", True)) and room >= breath["min_gap"] \
                and clear(on - min(breath["length"], room), on):
            groups.append(dict(pos=first.start - (first.lead or 0.0) - min(breath["length"], room), phones=["AP"],
                               midi=midi(first), rest=True))
    for k, u in enumerate(units):
        ja = any(n[-1] in JA_VOWELS for n in u.nuclei) if u.nuclei else False
        gap = u.start - units[k - 1].end if k > 0 else 0.0
        if k > 0 and gap >= sp_gap:
            if breath and not u.breath and not units[k - 1].breath and gap >= breath["min_gap"] \
                    and clear(max(units[k - 1].end, u.start - (u.lead or 0.0) - breath["length"]), u.start - (u.lead or 0.0)):
                if gap > breath["fill"]:
                    groups.append(dict(pos=units[k - 1].end, phones=["SP"], midi=midi(u)))
                    groups.append(dict(pos=u.start - (u.lead or 0.0) - breath["length"], phones=["AP"], midi=midi(u),
                                       rest=True))
                else:
                    groups.append(dict(pos=units[k - 1].end, phones=["AP"], midi=midi(u), rest=True))
            else:
                groups.append(dict(pos=units[k - 1].end, phones=["SP"], midi=midi(u), rest=True))
        if u.breath:
            groups.append(dict(pos=u.start, phones=["AP"], midi=midi(u)))
            continue
        if not u.nuclei:
            if u.closure:
                groups.append(dict(pos=u.start, phones=["SP"], midi=midi(u)))
            else:
                groups[-1]["phones"] += [conv(p, ja) for p in u.phones]
            continue
        groups[-1]["phones"] += [conv(p, ja) for p in u.onset]
        span = u.end - u.start
        for j, nuc in enumerate(u.nuclei):
            groups[-1]["phones"] += [conv(p, ja) for p in nuc[:-1]]
            groups.append(dict(pos=u.start + span * j / len(u.nuclei), phones=[conv(nuc[-1], ja)], midi=midi(u),
                               vowel=True))
        groups[-1]["phones"] += [conv(p, ja) for p in u.coda]
    first_pos = groups[1]["pos"]
    groups[0]["pos"] = first_pos - PAD
    for a, b in zip(groups, groups[1:]):  # a breath placed from the end never starts before the group before it
        b["pos"] = max(b["pos"], a["pos"] + FRAME)
    return groups, units[-1].end


def align(groups, end, pred, min_vowel=None, cons_scale=1.0, cons_cap=None, min_vowel_share=None, onset_keep=False,
          cons_cap_keep=0.0):
    """OpenUtau alignment: returns start time (s) of every phoneme after the leading SP, in order.

    min_vowel (s): a vowel the predicted lengths leave shorter than this takes the difference from the consonants
    that share its note, up to MIN_VOWEL_SHARE of the note.
    cons_scale: the predicted lengths of consonants are multiplied by this (a number, or {class: number}) before
    a note's time is shared out, so the vowel takes what they give up.
    cons_cap ({class: s}): no consonant that shares a note with a vowel is longer than its class's cap; the vowel
    takes the rest. The consonants after a rest or a breath are never capped (nothing competes for their time).
    cons_cap_keep: no cap takes a consonant under this share of the length the duration model gave it, so the
    consonants the model wants long (the p of "apocalypse", 116 ms) are trimmed, not cut to the cap.
    min_vowel_share: every vowel gets at least this share of its note, taken from the consonants after it.
    onset_keep: the consonants after a rest or breath keep their predicted (scaled) length, as those at the start
    of the phrase do, instead of sharing the rest's time in proportion.
    """
    pred = np.array(pred, float)
    classes = [cons_class(p) for g in groups for p in g["phones"]]
    if cons_scale != 1.0:
        for i, c in enumerate(classes):
            if c is not None:
                pred[i] *= cons_scale.get(c, 1.0) if isinstance(cons_scale, dict) else cons_scale
    cap = np.array([(cons_cap or {}).get(c, np.inf) if c is not None else np.inf for c in classes])
    counts = [len(g["phones"]) for g in groups]
    idx = np.concatenate([[0], np.cumsum(counts)])
    pos = [g["pos"] for g in groups] + [end]
    starts = []
    # first group: SP + onset consonants keep their predicted length, ending at the first vowel
    cons = pred[1:idx[1]] * FRAME
    lim = max(0.0, pos[1] - pos[0] - 2 * FRAME)
    if cons.sum() > lim and cons.sum() > 0:
        cons = cons * lim / cons.sum()
    c = pos[1] - cons.sum()
    for d in cons:
        starts.append(c)
        c += d
    for g in range(1, len(groups)):
        seg = np.maximum(pred[idx[g]:idx[g + 1]], 0.5)
        span = pos[g + 1] - pos[g]
        if onset_keep and len(seg) > 1 and groups[g]["phones"][0] in ("SP", "AP"):
            c = seg[1:] * FRAME
            room = max(span - 2 * FRAME, 0.5 * span)
            if c.sum() > room:
                c *= room / c.sum()
            seg = np.concatenate([[span - c.sum()], c])
        else:
            seg = seg * span / seg.sum()
        over = np.zeros(len(seg) - 1)
        if groups[g].get("vowel"):
            over = np.maximum(seg[1:] - np.maximum(cap[idx[g] + 1:idx[g + 1]], cons_cap_keep * seg[1:]), 0.0)
        if over.any():
            seg[1:] -= over
            seg[0] += over.sum()
        if min_vowel_share and groups[g].get("vowel") and len(seg) > 1 and seg[0] < min_vowel_share * span:
            seg[1:] *= (1.0 - min_vowel_share) * span / seg[1:].sum()
            seg[0] = min_vowel_share * span
        if min_vowel and groups[g].get("vowel") and len(seg) > 1:
            want = min(float(min_vowel), MIN_VOWEL_SHARE * span)
            if seg[0] < want:
                seg[1:] *= (span - want) / seg[1:].sum()
                seg[0] = want
        c = pos[g]
        for d in seg:
            starts.append(c)
            c += d
    return starts


# ------------------------------------------------------------------------------- slow-motion rendering
def slow_factors(groups, end, slow_note=SLOW_NOTE, slow_max=SLOW_MAX):
    """Per phoneme (leading SP included): how many times slower it is sung than written.

    A note shorter than slow_note is stretched to slow_note (at most slow_max times); silence is never stretched and
    the consonants before a note follow that note.
    """
    pos = [g["pos"] for g in groups] + [end]
    kg = [float(np.clip(slow_note / max(b - a, 1e-3), 1.0, slow_max)) for a, b in zip(pos, pos[1:])]
    k = []
    for g, grp in enumerate(groups):
        rest = grp["phones"][0] in ("SP", "AP")
        own = kg[min(g + 1, len(groups) - 1)] if rest else kg[g]
        k += [1.0 if p in ("SP", "AP") else own for p in grp["phones"]]
    return np.array(k)


def slow_grid(durs, k):
    """Phoneme lengths (frames) stretched by k, and the frame positions that map each grid onto the other.

    Phoneme edges of the two grids coincide; inside a phoneme the map is linear.
    """
    edges = np.concatenate([[0.0], np.cumsum(durs)])
    slow = np.maximum(np.diff(np.round(np.concatenate([[0.0], np.cumsum(np.asarray(durs) * k)]))), 1).astype(int)
    edges2 = np.concatenate([[0.0], np.cumsum(slow)])
    to_true = np.interp(np.arange(int(edges2[-1])) + 0.5, edges2, edges) - 0.5
    to_slow = np.interp(np.arange(int(edges[-1])) + 0.5, edges, edges2) - 0.5
    return [int(d) for d in slow], to_true, to_slow


def warp_rows(x, at):
    """Rows of x (frames, bins) at fractional frame positions, linearly interpolated."""
    at = np.clip(at, 0, len(x) - 1)
    i = np.minimum(at.astype(int), len(x) - 2)
    f = (at - i)[:, None]
    return (x[i] * (1 - f) + x[i + 1] * f).astype(np.float32)


# ------------------------------------------------------------------------------- vowel voicing floor
def _running_median(x, win):
    """Median of the win samples centred on each sample (reflected at the ends)."""
    n = len(x)
    if n <= win:
        return np.full(n, np.median(x))
    h = win // 2
    xp = np.concatenate([x[h:0:-1], x, x[-2:-h - 2:-1]])
    return np.median(xp[np.arange(win)[None, :] + np.arange(n)[:, None]], axis=1)


def floor_voicing(voicing, seq, durs, is_vowel, floor_db=VOICING_FLOOR, window=VOICING_WINDOW):
    """Lift the vowel frames whose predicted voicing collapsed.

    In fast lines the variance model predicts some short vowels 10 to 60 dB under the vowels around them while
    their breathiness stays put; the acoustic model then whispers the syllable. Frames of vowels (is_vowel, one
    flag per phoneme of seq) are held at or above the running median of the phrase's vowel voicing minus floor_db.
    A vowel that dies away into a rest keeps its predicted decay (its trailing low frames are left alone, unless
    the whole vowel is low). Consonants and rests keep the prediction. floor_db=None returns it unchanged.
    """
    if floor_db is None:
        return voicing
    idx = np.concatenate([[0], np.cumsum(durs)])
    vmask = np.repeat(np.asarray(is_vowel, bool), durs)
    if vmask.sum() < 4:
        return voicing
    out = voicing.copy()
    floor = np.full(len(out), -np.inf)
    floor[vmask] = _running_median(out[vmask], int(round(window / FRAME))) - floor_db
    for i in np.flatnonzero(is_vowel):
        a, b = idx[i], idx[i + 1]
        low = out[a:b] < floor[a:b]
        if not low.any():
            continue
        if i + 1 < len(seq) and seq[i + 1] in ("SP", "AP") and not low.all():
            k = b - a
            while low[k - 1]:
                k -= 1
            low[k:] = False
        out[a:b] = np.where(low, floor[a:b], out[a:b])
    return out


BREATH_FALL = 0.09  # s over which a breath's noise dies away before the consonant after it


def breath_shape(n):
    """dB under its peak for the n frames of a breath: in from -6, loudest after a third, a slow fall to -4 and
    out to -22 over the last BREATH_FALL s, as the breaths of the bank's sample songs run (loudest early, 15 to
    20 dB down in their last quarter)."""
    t = (np.arange(n) + 0.5) * FRAME
    end = n * FRAME
    fall = max(end - BREATH_FALL, 0.5 * end)
    return np.interp(t, [0.0, 0.35 * fall, fall, end], [-6.0, 0.0, -4.0, -22.0])


def shape_consonants(var, seq, durs, is_vowel, voicing_floor=None, stop_breath=None, fric_breath=0.0,
                     ap_breath=None, ap_offset=0.0):
    """The variance curves on consonant and breath frames (var is changed in place).

    voicing_floor (dB, or {class: dB}): frames of voiced consonants (nasals, glides, voiced stops and fricatives)
    are held at or above the vowel voicing around them (running median over the phrase's vowels, interpolated
    across the consonants) minus this, so the voice does not stop on them.
    stop_breath (dB): the breathiness of unvoiced stops and affricates is held at or above this (the burst).
    fric_breath (dB): added to the breathiness of unvoiced fricatives.
    ap_breath (dB): the breathiness of a breath (AP) peaks at this (`breath_shape`), its voicing is -96.
    ap_offset (dB): with ap_breath None, added to the breathiness the variance model predicts for a breath.
    """
    idx = np.concatenate([[0], np.cumsum(durs)]).astype(int)
    cls = [cons_class(p) for p in seq]
    for k in ("voicing", "breathiness"):
        if k in var:
            var[k] = np.clip(var[k], -96, 0)
    if voicing_floor is not None and "voicing" in var:
        v = var["voicing"]
        vmask = np.repeat(np.asarray(is_vowel, bool), durs)
        if vmask.sum() >= 4:
            at = np.flatnonzero(vmask)
            ref = np.interp(np.arange(len(v)), at, _running_median(v[vmask], int(round(VOICING_WINDOW / FRAME))))
            for i, c in enumerate(cls):
                if c in ("nasal", "glide") or c in VOICED_OBSTRUENT:
                    fl = voicing_floor.get(c) if isinstance(voicing_floor, dict) else voicing_floor
                    if fl is not None:
                        a, b = idx[i], idx[i + 1]
                        v[a:b] = np.maximum(v[a:b], ref[a:b] - fl)
        var["voicing"] = v
    if "breathiness" in var:
        br = var["breathiness"]
        for i, c in enumerate(cls):
            a, b = idx[i], idx[i + 1]
            if stop_breath is not None and c in ("stop", "aff"):
                br[a:b] = np.maximum(br[a:b], stop_breath)
            elif fric_breath and c == "fric":
                br[a:b] += fric_breath
            elif seq[i] == "AP" and (ap_breath is not None or ap_offset):
                br[a:b] = br[a:b] + ap_offset if ap_breath is None else ap_breath + breath_shape(b - a)
                if "voicing" in var:
                    var["voicing"][a:b] = -96.0
        var["breathiness"] = br
    return var


def vowel_weight(durs, is_vowel):
    """One weight per frame: 1 inside vowels, 0.5 on a vowel's first and last frame, 0 elsewhere."""
    w = np.zeros(int(sum(durs)), np.float32)
    idx = np.concatenate([[0], np.cumsum(durs)]).astype(int)
    for i in np.flatnonzero(is_vowel):
        a, b = idx[i], idx[i + 1]
        w[a:b] = 1.0
        w[a] = w[b - 1] = 0.5
    return w


# ------------------------------------------------------------------------------- level after the vocoder
def even_level(wav, seq, durs, is_vowel, lift=None, limit=None, lift_max=9.0, classes=("fric", "vfric"),
               limit_classes=("fric", "stop", "aff", "vfric"), win=0.010):
    """Bring consonants that the model sang very quietly or very loudly toward the vowels of their phrase.

    The reference level is the median level of the phrase's vowel frames (RMS in `win` s windows). A consonant of
    `classes` whose loudest window is more than `lift` dB under it is raised to that (by at most lift_max dB); a
    consonant of `limit_classes` whose loudest window is less than `limit` dB under it is lowered to that. One gain
    per consonant, ramped over 5 ms at its edges; vowels are untouched.
    """
    n = int(win * SR)
    k = len(wav) // n
    lv = 10 * np.log10((wav[: k * n].reshape(k, n).astype(np.float64) ** 2).mean(axis=1) + 1e-12)
    idx = (np.concatenate([[0], np.cumsum(durs)]) * HOP).astype(int)
    vow = [lv[idx[i] // n:max(idx[i + 1] // n, idx[i] // n + 1)] for i in np.flatnonzero(is_vowel)]
    if not vow:
        return wav
    ref = float(np.median(np.concatenate(vow)))
    gain = np.zeros(len(wav))
    for i, p in enumerate(seq):
        c = cons_class(p)
        if c is None:
            continue
        a, b = idx[i], min(idx[i + 1], len(wav))
        seg = lv[a // n:max(b // n, a // n + 1)]
        if not len(seg):
            continue
        peak = float(seg.max()) - ref
        g = 0.0
        if lift is not None and c in classes and peak < -lift:
            g = min(-lift - peak, lift_max)
        if limit is not None and c in limit_classes and peak > -limit:
            g = -limit - peak
        gain[a:b] = g
    if not gain.any():
        return wav
    r = int(0.005 * SR)
    gain = np.convolve(np.pad(gain, r, mode="edge"), np.hanning(2 * r + 1) / np.hanning(2 * r + 1).sum(), mode="valid")
    return (wav * 10 ** (gain[: len(wav)] / 20)).astype(np.float32)


# ------------------------------------------------------------------------------- the bank's pitch predictor
def phrase_seed(seed, f_start, n) -> int:
    """31-bit seed from the render seed and the phrase's place and length, not from its notes or lyrics: with the
    noise unchanged, editing a note leaves the rest of the phrase's take in place."""
    key = [int(seed), int(f_start), int(n)]
    return int.from_bytes(hashlib.sha256(json.dumps(key).encode()).digest()[:4], "big") & 0x7FFFFFFF


def default_expr(style, voice) -> float:
    e = STYLE_EXPR.get(style.name, 1.0) * (1.0 - 0.5 * float(voice.robotic))
    return float(np.clip(e, MIN_EXPR, 1.0))


def default_vibrato(style, track) -> str:
    """"synthetic" when the vibrato is the user's design (robotic style, vibrato parameters on the track)."""
    params = getattr(track, "voice_params", {}) or {}
    if style.name in SYNTHETIC_VIBRATO_STYLES or any("vibrato" in k for k in params):
        return "synthetic"
    return "auto"


def _ramp(x):
    return 0.5 - 0.5 * np.cos(np.pi * np.clip(x, 0.0, 1.0))


def sustain_weight(t, timings, min_note, head, tail, ramp, pick=None):
    """1 on the sustained part of notes at least min_note s long (and chosen by pick)."""
    w = np.zeros(len(t))
    for nt in timings:
        if nt.end - nt.start < min_note or (pick is not None and not pick(nt)):
            continue
        a, b = nt.start + head, nt.end - tail
        if b - a >= 2 * ramp:
            w = np.maximum(w, _ramp((t - a) / ramp) * _ramp((b - t) / ramp))
    return w


def transition_windows(t, timings, style):
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


def bridge(x, bad, margin=2):
    """x with every run of bad frames (widened by margin frames on both sides) replaced by a straight line between
    the frames next to the run; a run at an end of x holds the nearest good value."""
    m = np.convolve(np.asarray(bad, float), np.ones(2 * margin + 1), mode="same") > 0
    if m.all():
        return x
    good = np.flatnonzero(~m)
    out = np.array(x, float)
    out[m] = np.interp(np.flatnonzero(m), good, out[good])
    return out


class BankPitch:
    """The bank's predictor laid under Synth2's pitch controls, phrase by phrase.

    Notes with a `transition` / `overshoot` attribute keep `build_pitch`'s curve in a window around their start; the
    predicted vibrato is scaled by style.vibrato_depth * vibratoEnv on sustained frames; notes whose vibrato is the
    user's design (vibrato attributes, or every note under vibrato="synthetic") get `build_pitch`'s vibrato instead.
    A few sung frames far outside the notes around them (the predictor now and then drops an octave on a consonant)
    are bridged; a phrase the predictor cannot sing keeps `build_pitch`'s curve. pitchDelta is added by the caller.
    """

    def __init__(self, bank, t, timings, style, voice, s2, s2_flat, env, expr, steps, vibrato):
        self.bank, self.expr, self.steps = bank, float(expr), int(steps)
        self.timings = timings
        self.s2, self.s2_flat, self.s2_vib = s2, s2_flat, s2 - s2_flat
        self.keep = transition_windows(t, timings, style)
        has_attr = lambda nt: any(k in nt.attrs for k in VIB_ATTRS)
        synth = (lambda nt: True) if vibrato == "synthetic" else (lambda nt: False) if vibrato == "predicted" else has_attr
        self.w_syn = sustain_weight(t, timings, **{**SUSTAIN, "min_note": 0.0}, pick=synth)
        self.syn_frames = np.zeros(len(t))
        for nt in timings:
            if synth(nt):
                self.syn_frames[(t >= nt.start) & (t < nt.end + 0.05)] = 1.0
        self.w_sus = sustain_weight(t, timings, **SUSTAIN)
        self.gain = float(style.vibrato_depth) * env
        self.fallbacks = []
        self.repaired = 0  # outlier frames bridged

    def phrase(self, units, seq, durs, f_start, sl, seed=None):
        """Pitch in MIDI semitones for the phrase's frames (without pitchDelta)."""
        ids = {i for u in units for i in u.notes}
        tm = [x for x in self.timings if x.index in ids]
        bank, k = self.bank, self.keep[sl]
        try:
            if not tm:
                raise ValueError("no pitched note")
            notes = dspitch.notes_from_timings(tm, f_start, FRAME)
            ni = dspitch.build_notes(notes, seq, durs, bank.is_vowel)
            kw = {"keep_pitch": self.s2_flat[sl], "retake": ~k} if k.any() else {}
            pred = bank.predict(seq, durs, notes, expr=self.expr, steps=self.steps, seed=seed, inputs=ni, **kw)
            if not np.isfinite(pred).all():
                raise ValueError("prediction not finite")
            bad = ~ni.frame_rest() & ~k & (dspitch.range_excess(pred, ni) > MAX_DEV)
            if bad.any():
                if bad.sum() > MAX_OUTLIERS * max(1, int((~ni.frame_rest()).sum())):
                    raise ValueError("prediction out of range")
                pred = bridge(pred, bad)
                self.repaired += int(bad.sum())
        except Exception as err:  # the phrase keeps build_pitch's curve
            self.fallbacks.append(f"{type(err).__name__}: {err}")
            return self.s2[sl]
        if k.any():
            ws = dspitch.soft_window(k, 3)
            pred = ws * self.s2_flat[sl] + (1.0 - ws) * pred
        weight = np.maximum(self.w_sus[sl], self.w_syn[sl]) * (1.0 - dspitch.soft_window(k, 3))
        p = dspitch.scale_fast(pred, dspitch.base_pitch(ni), self.gain[sl] * (1.0 - self.w_syn[sl]), weight, bank.hop_s)
        return p + self.s2_vib[sl] * self.syn_frames[sl]


# ------------------------------------------------------------------------------- ad-libs, hit by hit
# Defaults of the `adlib` option of render_track_ds.
ADLIB = {
    "split": "word",     # a hit is one word (a note and its "+" notes) | "syllable" (every note) | "run" (touching notes)
    "pitch": "flat",     # the pitch shape of a hit: "flat" | "fall" | "rise", or {lyric: shape, "default": shape}
    "bend": 3.0,         # semitones a "fall" drops (a "rise" climbs) from `bend_from` of the hit to its end
    "bend_from": 0.3,
    "glide": 0.04,       # s of pitch transition between the notes of one hit
    "vibrato": False,    # True: the voice's designed vibrato (rate and depth of the voice / the note), phase 0 at the note
    "stamp": True,       # one take per distinct sound (phonemes, pitches, lengths), pasted at every repetition
    "tight": True,       # a hit that touches the next hit ends where that hit's onset consonants start
    "tight_onset": 0.4,  # the most of its first note the onset consonants of a hit in a touching run may take
    "onset_max": None,   # s: the most the onset consonants of a free-standing hit may take (None: the duration
                         # model's lengths; with 0.085 s every recogniser heard the crowd's "click" as "tick" / "kick")
    # s per consonant class, in a touching run: the most one onset consonant may take before the onset is fitted
    # into its room. After a rest the duration model gives an initial stop most of the onset (its closure, which is
    # silence there) and squeezes what follows it: the l of "click" got 12 to 23 ms.
    "onset_cap": {"stop": 0.025, "vstop": 0.03, "aff": 0.07, "fric": 0.07, "vfric": 0.06, "nasal": 0.06, "glide": 0.06},
    "salt": None,        # {lyric: n}: another take of that sound (n is mixed into its seed); for auditioning takes
    "gap": 0.02,         # s of silence a tight hit that ends in a vowel leaves before the next hit's onset
    "hold": 0.07,        # s a hit that ends in a vowel is sung past its end (the model's own decay), then cut
    "run_gap": 0.25,     # s: hits closer than this belong to one run (accent pattern)
    "accent": None,      # dB per position in a run, e.g. [0, -2.5, -1, -3.5] (cycled)
    "pre": 0.012,        # s kept before the first phoneme of a take
    "release": 0.05,     # s kept after a take that ends in a consonant (0.01 after a held vowel), then a 30 ms fade
    "chop": None,        # {"hz": 11, "depth_db": -9}: a loudness chop locked to the start of every take
    "glottal": False,    # True: a hit that starts with a vowel (not the syllabic N) gets a closure (cl) before it
    "cons_cap": None,    # consonant caps as in `align` (None: the duration model's lengths)
}


def adlib_hits(plan, notes, split="word"):
    """The plan's sung units grouped into hits (lists of units)."""
    hits = []
    for u in plan.units:
        if u.breath or u.closure or not u.nuclei:
            continue
        lyric = (notes[u.notes[0]].lyric or "") if u.notes else ""
        touch = bool(hits) and u.start - hits[-1][-1].end < 0.03
        if split == "syllable" or not hits:
            new = True
        elif split == "run":
            new = not touch
        else:
            new = lyric != "+" or not touch
        if new:
            hits.append([u])
        else:
            hits[-1].append(u)
    return hits


def _adlib_shape(spec, lyric):
    if isinstance(spec, dict):
        return spec.get(lyric, spec.get(lyric.lower(), spec.get("default", "flat")))
    return spec


def render_adlib(ds, project, track, plan, voice, style, ad, steps=10, depth=0.6, gender=0.0, tone=0.0,
                 velocity=1.0, var_steps=10, window=None, slow_note=None, slow_max=SLOW_MAX, voicing_floor=None,
                 seed=None, detune=0.0, vowel_breath_off=0.0, vowel_voicing_off=0.0, cons_voicing_floor=None,
                 dump=False, verbose=False):
    """A track of ad-libs: every hit (`adlib_hits`) is sung as a take of its own and pasted at its place.

    A take is the hit alone between rests: no breath, nothing of the notes before or after it in the duration,
    variance and acoustic models' inputs, and a designed pitch (the written pitch held flat from before the first
    consonant to after the last, a fall or a rise by `bend` semitones, a short glide between the notes of one
    hit, the voice's designed vibrato when asked for, the track's pitchDelta curve on top). Its length is the
    written length of its notes (the planner's humanised times place the hit, they do not change the sound); its
    onset consonants lie before the note with the lengths the duration model gives them (at most `onset_max` s
    when set; in a touching run at most `tight_onset` of the note, each consonant under its `onset_cap`).

    With `tight`, a hit that touches the next one ends where that hit's onset consonants start, so the onset
    falls into silence and not onto this hit's vowel: a hit that ends in a consonant is sung that much shorter
    (its closure is the silence), a hit that ends in a vowel is sung to its written end plus `hold` s and cut
    `gap` s before the next onset with a fade (the acoustic model lets the last 60 to 80 ms before a rest die
    away; the cut keeps the vowel at full level to the end). The last hit of a run of equal hits is cut like the
    ones before it, so all of them are one sound.

    With `stamp`, the take of each distinct sound (phonemes, frame lengths, pitches, pitch shape, vibrato,
    pitchDelta) is made once, with a seed derived from the sound and not from its place, and every repetition is
    a copy of the same samples (cut where its own place asks for it); without it every hit is a fresh take.
    `accent` scales the copies by their position in a run. Returns (audio, info) as render_track_ds.
    """
    notes = track.notes
    b2s = project.tempo.beat_to_sec
    by_index = {tm.index: tm.pitch for tm in plan.timings}
    hits = adlib_hits(plan, notes, ad["split"])
    t_end = max(0.0, plan.end) + 1.0
    out = np.zeros(int(np.ceil(t_end * SR)) + SR, np.float32)
    info = {"phrases": 0, "phones": [], "audio_s": 0.0, "pitch_source": "adlib", "adlib": {"hits": [], "takes": 0}}
    T0 = int(round((PAD + 0.25) / FRAME)) * FRAME  # the first note of a take starts here on the take's own grid
    touches = [h + 1 < len(hits) and hits[h + 1][0].start - hits[h][-1].end < 0.03 for h in range(len(hits))]
    run_pos, pos = [], 0
    for h, units in enumerate(hits):
        pos = 0 if h == 0 or units[0].start - hits[h - 1][-1].end >= ad["run_gap"] else pos + 1
        run_pos.append(pos)
    dur_cache = {}

    def lay(g, end_):
        """Phoneme start times of geometry g sung to end_."""
        k = (tuple(g["phones"]), tuple(g["lens"]), round(g["groups"][0]["pos"] / FRAME), round(end_ / FRAME), bool(g["cap"]))
        if k not in dur_cache:
            ps = [x["pos"] for x in g["groups"]] + [end_]
            fr = [int(round(x / FRAME)) for x in ps]
            word_dur = [max(1, y - x) for x, y in zip(fr, fr[1:])]
            pred = np.array(ds.predict_dur(g["phones"], g["word_div"], word_dur, g["ph_midi"]), float)
            for i in range(1, g["word_div"][0]):
                cap = g["cap"].get(cons_class(g["phones"][i]))
                if cap:
                    pred[i] = min(pred[i], cap / FRAME)
            dur_cache[k] = align(g["groups"], end_, pred, cons_cap=ad["cons_cap"], onset_keep=True)
        return dur_cache[k]

    def geometry(h):
        """The take of hit h on its own time grid, before any cut for the hit after it."""
        units = hits[h]
        lens = []
        for u in units:
            a = b2s(notes[u.notes[0]].onset)
            b = b2s(notes[u.notes[-1]].onset + notes[u.notes[-1]].duration)
            lens.append(max(2, int(round((b - a) / FRAME))))
        rel, c = [], T0
        for n_fr in lens:
            rel.append((c, c + n_fr * FRAME))
            c += n_fr * FRAME
        local = [dataclasses.replace(u, start=a, end=b, breath=False) for u, (a, b) in zip(units, rel)]
        groups, end = phrase_groups(local, plan, ds, breath=None)
        if ad["glottal"] and len(groups[0]["phones"]) == 1 and ds.has("ja/cl") and groups[1]["phones"][0] != "ja/N":
            groups[0]["phones"].append("ja/cl")
        room = [ad["onset_max"]] if ad["onset_max"] else []
        in_run = bool(ad["tight"]) and (touches[h] or (h > 0 and touches[h - 1]))
        if in_run:
            room.append(ad["tight_onset"] * lens[0] * FRAME)
        if room:
            groups[0]["pos"] = max(groups[0]["pos"], T0 - min(room) - 2 * FRAME)
        g = {"units": units, "lens": lens, "rel": rel, "groups": groups, "end": end,
             "phones": [q for x in groups for q in x["phones"]], "word_div": [len(x["phones"]) for x in groups],
             "ph_midi": [x["midi"] for x in groups for _ in x["phones"]],
             "vowel_end": cons_class(groups[-1]["phones"][-1]) is None, "cap": (ad["onset_cap"] or {}) if in_run else {}}
        g["hold"] = int(round(ad["hold"] / FRAME)) * FRAME if g["vowel_end"] else 0.0
        starts = lay(g, end + g["hold"])
        g["onset_len"] = T0 - starts[0] if len(starts) else 0.0
        return g

    in_window = [window is None or not (u[-1].end + 0.5 < window[0] or u[0].start - 1.0 > window[1]) for u in hits]
    geo = {}

    def geom(h):
        if h not in geo:
            geo[h] = geometry(h)
        return geo[h]

    raw, takes = {}, {}
    prev = None  # (sound, cut, touches) of the hit before
    for h, units in enumerate(hits):
        if not (in_window[h] or (h + 1 < len(hits) and in_window[h + 1])):  # the hit before a wanted one sets its cut
            prev = None
            continue
        g, u0 = geom(h), units[0]
        end, hold, rel = g["end"], g["hold"], g["rel"]
        midis = [float(by_index.get(u.notes[0], 60.0)) if u.notes else 60.0 for u in units]
        sound = (tuple(g["phones"]), tuple(g["lens"]), tuple(midis))
        cut = 0.0
        if ad["tight"] and touches[h]:
            cut = min(geom(h + 1)["onset_len"] + (ad["gap"] if g["vowel_end"] else 0.0), 0.5 * g["lens"][-1] * FRAME)
        elif ad["tight"] and prev is not None and prev[2] and prev[0] == sound:
            cut = prev[1]
        prev = (sound, cut, touches[h])
        if not in_window[h]:
            continue
        if not g["vowel_end"]:  # a final consonant is sung earlier; a final vowel is cut after the vocoder
            end = end - int(round(cut / FRAME)) * FRAME
        starts = lay(g, end + hold)
        b = [int(round(x / FRAME)) for x in starts] + [int(round((end + hold) / FRAME))]
        for j in range(1, len(b)):
            b[j] = max(b[j], b[j - 1] + 1)
        durs = [HEAD] + [y - x for x, y in zip(b, b[1:])] + [TAIL]
        seq = ["SP"] + g["phones"][1:] + ["SP"]
        f_start = b[0] - HEAD
        n = sum(durs)
        tt = (f_start + np.arange(n)) * FRAME  # take time of every frame
        lyric = (notes[u0.notes[0]].lyric or "") if u0.notes else ""
        shape = _adlib_shape(ad["pitch"], lyric)
        bend_end = g["end"] - cut  # a fall or a rise is complete where the hit stops sounding
        delta = np.clip(evaluate_curve(track.curves.get("pitchDelta"), u0.start + (tt - T0), b2s, 0.0),
                        *CURVE_SPECS["pitchDelta"][1:3]) / 100.0
        dkey = tuple(np.round(delta, 2)) if np.any(np.abs(delta) > 0.005) else ()
        vib = []
        if ad["vibrato"]:
            for u in units:
                at = notes[u.notes[0]].attrs if u.notes else {}
                vd = at.get("vibrato_depth")
                vd = voice.vibrato_depth * style.vibrato_depth if vd is None else float(vd)
                vr = at.get("vibrato_rate")
                vr = voice.vibrato_rate * style.vibrato_rate if vr is None else float(vr)
                vib.append((round(float(vd), 3), round(float(vr), 3)))
        key = (tuple(seq), tuple(durs), tuple(midis), shape, dkey, tuple(vib),
               round(bend_end / FRAME) if shape != "flat" else 0)
        kid = hashlib.sha1(json.dumps(key, default=str).encode()).hexdigest()[:10]
        if not ad["stamp"]:
            kid = f"{kid}@{h}"
        take = raw.get(kid)
        if take is None:
            # designed pitch: the written notes as steps with a short glide, flat through consonants and rests
            p = np.full(n, midis[0])
            for (a_, _), m_prev, m_ in zip(rel[1:], midis, midis[1:]):
                x = np.clip((tt - (a_ - ad["glide"] / 2)) / max(ad["glide"], 1e-3), 0.0, 1.0)
                p += (m_ - m_prev) * (3 * x ** 2 - 2 * x ** 3)
            if shape in ("fall", "rise"):
                a_ = T0 + ad["bend_from"] * (bend_end - T0)
                x = np.clip((tt - a_) / max(bend_end - a_, FRAME), 0.0, 1.2)
                p += (-1.0 if shape == "fall" else 1.0) * float(ad["bend"]) * x
            elif shape != "flat":
                raise ValueError(f"adlib pitch shape {shape!r}")
            for (a_, b_), (vd, vr) in zip(rel, vib):
                m_ = (tt >= a_) & (tt < b_ + 0.1)
                if vd > 0 and vr > 0:
                    p[m_] += vd * np.sin(2 * np.pi * vr * (tt[m_] - a_))
            p_semi = p + delta + detune / 100.0
            f0 = 440.0 * 2 ** ((p_semi - 69) / 12)
            is_vowel = [False] + [j == 0 and x.get("vowel", False) and q not in ("SP", "AP")
                                  for x in g["groups"] for j, q in enumerate(x["phones"])][1:] + [False]
            k = np.append(slow_factors(g["groups"], end + hold, slow_note, slow_max), 1.0) if slow_note else np.ones(len(seq))
            if k.max() > 1.02:
                durs_ac, to_true, to_slow = slow_grid(durs, k)
                p_ac = np.interp(to_true, np.arange(n), p_semi) + tone
            else:
                durs_ac, to_slow, p_ac = durs, None, p_semi + tone
            if seed is not None:
                place = int(kid[:7], 16) if ad["stamp"] else int(round(u0.start / FRAME))
                place += int((ad["salt"] or {}).get(lyric, 0)) * 7919
                ds.reseed(phrase_seed(seed, place, n))
            var = ds.predict_variance(seq, durs_ac, p_ac, steps=var_steps)
            if voicing_floor is not None and "voicing" in var:
                var["voicing"] = floor_voicing(np.clip(var["voicing"], -96, 0), seq, durs_ac, is_vowel, voicing_floor)
            if cons_voicing_floor is not None:
                shape_consonants(var, seq, durs_ac, is_vowel, cons_voicing_floor)
            offs = {"breathiness": 0.0, "voicing": 0.0, "tension": 0.0}
            if vowel_breath_off or vowel_voicing_off:
                vm = vowel_weight(durs_ac, is_vowel)
                offs["breathiness"] = vowel_breath_off * vm
                offs["voicing"] = vowel_voicing_off * vm
            mel = ds.mel(seq, durs_ac, 440.0 * 2 ** ((p_ac - 69) / 12), var, gender=gender, velocity=velocity,
                         steps=steps, depth=depth, offs=offs)
            if to_slow is not None:
                mel = warp_rows(mel, to_slow)
            wav = ds.vocode(mel, f0).astype(np.float32).copy()
            # silence before the first phoneme
            i0 = max(0, int((HEAD * FRAME - ad["pre"]) * SR))
            fi = int(0.005 * SR)
            wav[:i0] = 0.0
            wav[i0:i0 + fi] *= np.linspace(0, 1, fi, dtype=np.float32)
            if ad["chop"]:
                tw = np.arange(len(wav)) / SR - (T0 / FRAME - f_start) * FRAME
                ph = np.clip(tw, 0.0, None) * float(ad["chop"]["hz"])
                wav *= 10 ** (float(ad["chop"]["depth_db"]) * (0.5 - 0.5 * np.cos(2 * np.pi * ph)) / 20).astype(np.float32)
            take = {"wav": wav, "at": int(round((T0 / FRAME - f_start) * HOP)), "f_start": f_start, "seq": seq,
                    "durs": durs, "pitch": p_semi if dump else None}
            raw[kid] = take
            info["phrases"] += 1
            info["audio_s"] += n * FRAME
            info["phones"].append(list(zip(seq, durs)))
            if verbose:
                print(lyric, kid, " ".join(f"{q}:{d}" for q, d in zip(seq, durs)))
        # the copy for this place: cut after the release (after `cut` before the end for a final vowel)
        stop = end + 0.01 - cut if g["vowel_end"] else end + ad["release"]
        i1 = int((stop - take["f_start"] * FRAME) * SR)
        gk = (kid, i1)
        if gk not in takes:
            w = take["wav"].copy()
            i1 = max(0, min(len(w), i1))
            fo = min(int(0.03 * SR), len(w) - i1)
            if fo > 0:
                w[i1:i1 + fo] *= np.linspace(1, 0, fo, dtype=np.float32)
            w[i1 + max(fo, 0):] = 0.0
            takes[gk] = w
        w = takes[gk]
        gain = 10 ** (float(ad["accent"][run_pos[h] % len(ad["accent"])]) / 20) if ad["accent"] else 1.0
        s0 = int(round(u0.start * SR)) - take["at"]
        a = max(0, -s0)
        seg = w[a:len(w) - max(0, s0 + len(w) - len(out))]
        out[s0 + a:s0 + a + len(seg)] += seg * np.float32(gain)
        info["adlib"]["hits"].append({"t": round(float(u0.start), 4), "end": round(float(units[-1].end), 4),
                                      "lyric": lyric, "take": kid, "run_pos": run_pos[h], "touches": bool(touches[h]),
                                      "gain_db": round(20 * np.log10(gain), 2), "onset_len": round(g["onset_len"], 4),
                                      "cut": round(float(cut), 4), "seq": take["seq"], "durs": take["durs"], "shape": shape})
    info["adlib"]["takes"] = len(raw)
    if dump:
        info["adlib"]["pitch"] = {k_: t_["pitch"] for k_, t_ in raw.items()}
    return out[: int(np.ceil(t_end * SR))], info


# ---------------------------------------------------------------------------------------------
def render_track_ds(ds, project, track, steps=10, depth=0.6, gender=0.0, tone=0.0, velocity=1.0, var_steps=10,
                    breath_off=0.0, voicing_off=0.0, tension_off=0.0, verbose=False, window=None,
                    slow_note=None, slow_max=SLOW_MAX, voicing_floor=None, pitch_source="synth2", expr=None,
                    pitch_steps=10, vibrato=None, seed=None, min_vowel=None, dump=False,
                    vowel_breath_off=0.0, vowel_voicing_off=0.0, detune=0.0,
                    cons_scale=1.0, cons_cap=None, cons_cap_keep=0.0, min_vowel_share=None, onset_keep=False,
                    sp_gap=SP_GAP,
                    breath=None, cons_voicing_floor=None, stop_breath=None, fric_breath=0.0, level=None,
                    adlib=None):
    """Render a Synth2 track with DiffSinger. Returns (mono float32 at 44.1 kHz, info).

    adlib (dict, off by default): the track is a track of ad-libs and is sung hit by hit, see `render_adlib`.

    The four options (module docstring) are off by default: slow_note, voicing_floor, pitch_source="bank", seed.
    min_vowel (s, off by default) is a fifth, experimental one: see `align`.
    vowel_breath_off / vowel_voicing_off (dB, 0 = off): offsets on the breathiness and voicing curves of vowel
    frames only (half the offset on a vowel's first and last frame); consonants and rests keep the prediction.
    detune (cents, 0 = off): a constant added to the sung pitch (acoustic model and vocoder), for doubles.
    dump=True adds info["dump"]: per phrase the phonemes, frames, vowel flags, pitch and variance curves (real grid).

    Flow options (all off by default; with them off the render is the one above):
    cons_scale, cons_cap, cons_cap_keep, min_vowel_share, onset_keep: consonant lengths, see `align`.
    sp_gap (s): gaps between notes shorter than this are sung through. breath: breaths in the gaps between lines
    and before a phrase, see `phrase_groups` (BREATH has the recommended values; its "breathiness" is what the
    breath's frames are fed).
    cons_voicing_floor, stop_breath, fric_breath: the variance curves on consonants, see `shape_consonants`.
    level: the level of consonants after the vocoder, see `even_level`.
    """
    voice = get_voice(track.voice).with_params(track.voice_params)
    overrides = {k[6:]: v for k, v in track.voice_params.items() if k.startswith("style.")}
    style = get_style(track.style or voice.default_style, overrides)
    rng = np.random.default_rng(track.seed)
    plan = plan_track(track.notes, project.tempo, track.language, style, rng)
    if not plan.units:
        return np.zeros(SR, np.float32), {"phrases": 0}
    if adlib:
        return render_adlib(ds, project, track, plan, voice, style, dict(ADLIB, **adlib), steps=steps, depth=depth,
                            gender=gender, tone=tone, velocity=velocity, var_steps=var_steps, window=window,
                            slow_note=slow_note, slow_max=slow_max, voicing_floor=voicing_floor, seed=seed,
                            detune=detune, vowel_breath_off=vowel_breath_off, vowel_voicing_off=vowel_voicing_off,
                            cons_voicing_floor=cons_voicing_floor, dump=dump, verbose=verbose)
    k_off = int(np.floor((min(0.0, plan.start) - 1.0) / FRAME))  # global frame grid origin
    n_frames = int(np.ceil((plan.end + 1.0) / FRAME)) - k_off + 1
    t = (k_off + np.arange(n_frames)) * FRAME
    b2s = project.tempo.beat_to_sec
    curves = {name: np.clip(evaluate_curve(track.curves.get(name), t, b2s, spec[0]), spec[1], spec[2])
              for name, spec in CURVE_SPECS.items()}
    rng_state = rng.bit_generator.state
    s2 = build_pitch(t, plan.timings, style, voice, rng, vibrato_env=curves["vibratoEnv"])
    delta = curves["pitchDelta"] / 100.0 + detune / 100.0
    pitch = s2 + delta
    info = {"phrases": 0, "phones": [], "audio_s": 0.0, "pitch_source": "synth2"}
    bp = None
    if pitch_source == "bank" and ds.pitch is not None:
        r2 = np.random.default_rng()
        r2.bit_generator.state = rng_state
        s2_flat = build_pitch(t, plan.timings, style, voice, r2, vibrato_env=np.zeros(len(t)))
        e = default_expr(style, voice) if expr is None else float(expr)
        bp = BankPitch(ds.pitch, t, plan.timings, style, voice, s2, s2_flat, curves["vibratoEnv"], e, pitch_steps,
                       default_vibrato(style, track) if vibrato is None else vibrato)
        info.update({"pitch_source": "bank", "expr": e})
    elif pitch_source not in ("synth2", "bank"):
        raise ValueError(f"pitch_source {pitch_source!r}")

    t_end = max(0.0, plan.end) + 1.0
    out = np.zeros(int(np.ceil(t_end * SR)) + SR, np.float32)
    prev_end = None
    for units in build_phrases(plan):
        this_prev, prev_end = prev_end, units[-1].end
        if window is not None and (units[-1].end + 0.5 < window[0] or units[0].start - units[0].lead - 1.0 > window[1]):
            continue
        groups, end = phrase_groups(units, plan, ds, sp_gap=sp_gap, breath=breath, prev_end=this_prev)
        phones = [p for g in groups for p in g["phones"]]
        word_div = [len(g["phones"]) for g in groups]
        pos = [g["pos"] for g in groups] + [end]
        fr = [int(round(p / FRAME)) for p in pos]
        word_dur = [max(1, b - a) for a, b in zip(fr, fr[1:])]
        ph_midi = [g["midi"] for g in groups for _ in g["phones"]]
        pred = ds.predict_dur(phones, word_div, word_dur, ph_midi)
        starts = align(groups, end, pred, min_vowel, cons_scale=cons_scale, cons_cap=cons_cap,
                       min_vowel_share=min_vowel_share, onset_keep=onset_keep, cons_cap_keep=cons_cap_keep)
        # frame boundaries on the global grid: [head SP][phonemes...][tail SP]
        b = [int(round(s / FRAME)) for s in starts] + [int(round(end / FRAME))]
        for i in range(1, len(b)):
            b[i] = max(b[i], b[i - 1] + 1)
        durs = [HEAD] + [y - x for x, y in zip(b, b[1:])] + [TAIL]
        seq = ["SP"] + phones[1:] + ["SP"]
        if breath:  # a breath the consonants left too little room for is a rest
            seq = ["SP" if p == "AP" and d * FRAME < breath.get("min_len", 0.0) else p for p, d in zip(seq, durs)]
        f_start = b[0] - HEAD
        n = sum(durs)
        sl = slice(f_start - k_off, f_start - k_off + n)
        ph_seed = None if seed is None else phrase_seed(seed, f_start, n)
        if bp is not None:
            tp = time.time()
            p_semi = bp.phrase(units, seq, durs, f_start, sl, seed=ph_seed) + delta[sl]
            ds.timer.add("pitch", time.time() - tp)
        else:
            p_semi = pitch[sl]
        assert len(p_semi) == n, (len(p_semi), n)
        f0 = 440.0 * 2 ** ((p_semi - 69) / 12)
        # the phonemes that start a note (the vowels), one flag per entry of seq
        is_vowel = [False] + [j == 0 and g.get("vowel", False) and p not in ("SP", "AP")
                              for g in groups for j, p in enumerate(g["phones"])][1:] + [False]
        k = np.append(slow_factors(groups, end, slow_note, slow_max), 1.0) if slow_note else np.ones(len(seq))
        if k.max() > 1.02:
            # Fast notes: the variance and acoustic models sing them in slow motion and the mel is squeezed back
            # onto the real frame grid. The vocoder gets the real f0, so pitch and timing are as written.
            durs_ac, to_true, to_slow = slow_grid(durs, k)
            p_ac = np.interp(to_true, np.arange(n), p_semi) + tone
        else:
            durs_ac, to_slow, p_ac = durs, None, p_semi + tone
        if ph_seed is not None:
            ds.reseed(ph_seed)
        var = ds.predict_variance(seq, durs_ac, p_ac, steps=var_steps)
        raw = dict(var) if dump else None
        if voicing_floor is not None and "voicing" in var:
            var["voicing"] = floor_voicing(np.clip(var["voicing"], -96, 0), seq, durs_ac, is_vowel, voicing_floor)
        if cons_voicing_floor is not None or stop_breath is not None or fric_breath or breath:
            shape_consonants(var, seq, durs_ac, is_vowel, cons_voicing_floor, stop_breath, fric_breath,
                             breath.get("breathiness") if breath else None, breath.get("offset", 0.0) if breath else 0.0)
        offs = {"breathiness": breath_off, "voicing": voicing_off, "tension": tension_off}
        if vowel_breath_off or vowel_voicing_off:
            vm = vowel_weight(durs_ac, is_vowel)
            offs["breathiness"] = breath_off + vowel_breath_off * vm
            offs["voicing"] = voicing_off + vowel_voicing_off * vm
        mel = ds.mel(seq, durs_ac, 440.0 * 2 ** ((p_ac - 69) / 12), var, gender=gender, velocity=velocity, steps=steps,
                     depth=depth, offs=offs)
        if to_slow is not None:
            mel = warp_rows(mel, to_slow)
        wav = ds.vocode(mel, f0)
        fade = int(0.01 * SR)
        wav = wav.copy()
        if level:
            wav = even_level(wav, seq, durs, is_vowel, **level)
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
        s0 = f_start * HOP
        a = max(0, -s0)
        seg = wav[a:]
        s0 = max(0, s0)
        seg = seg[: len(out) - s0]
        out[s0:s0 + len(seg)] += seg
        info["phrases"] += 1
        info["audio_s"] += n * FRAME
        info["phones"].append(list(zip(seq, durs)))
        info.setdefault("slow", []).append(round(float(k.max()), 2))
        if dump:
            back = (lambda c: c) if to_slow is None else (lambda c: warp_rows(np.asarray(c)[:, None], to_slow)[:, 0])
            info.setdefault("dump", []).append({
                "f_start": f_start, "seq": seq, "durs": durs, "vowel": is_vowel, "pitch": p_semi.copy(),
                "frames_ac": int(sum(durs_ac)), "raw": {x: back(c) for x, c in raw.items()},
                "fed": {x: back(c) for x, c in var.items()}})
        if verbose:
            print(" ".join(f"{p}:{d}" for p, d in zip(seq, durs)))
    if bp is not None:
        info["pitch_fallbacks"] = bp.fallbacks
        info["pitch_repaired"] = bp.repaired
    return out[: int(np.ceil(t_end * SR))], info
