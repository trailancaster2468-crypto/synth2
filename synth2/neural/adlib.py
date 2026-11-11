"""Ad-lib mode: a track of interjections, shouts, sound words or chant syllables sung hit by hit.

Sung as lines, such a track comes out as one legato phrase: the words glide into each other,
a breath is drawn before them and the pitch predictor sings a melody over them. With
``neural.adlib=1`` every hit (a word, a syllable or a run of touching notes) is a take of its
own between rests: nothing of the notes around it reaches the duration, variance and acoustic
models, and its pitch is designed (the written pitch held flat from before the first consonant
to after the last, a fall or a rise, a short glide between the notes of one hit, the voice's
designed vibrato when asked for, the track's ``pitchDelta`` curve on top). A take is as long
as its notes are written; the planner's humanised times place it, they do not change its
sound. With ``stamp`` the take of each distinct sound is made once, with a seed derived from
the sound and not from its place, and every repetition is a copy of the same samples.

Options (``neural.adlib.<name>`` voice parameters, all numbers):

=====================  ======================================================================
``split``              what a hit is: 0 a word (a note and its ``+`` notes), 1 every note,
                       2 a run of touching notes
``bend``               semitones the pitch falls (negative) or rises (positive) from
                       ``bend_from`` of the hit to its end; ``bend.<lyric>`` for one lyric
``bend_from``          where the bend starts, as a share of the hit (0.3)
``glide``              s of pitch transition between the notes of one hit (0.04)
``vibrato``            1: the voice's designed vibrato (rate and depth of the voice or the note)
``stamp``              1 (default): one take per distinct sound; 0: every hit a fresh take
``salt``               another take of every sound (mixed into its seed); ``salt.<lyric>`` for
                       one lyric: a stamped hit repeats one take, so the take is worth choosing
``tight``              1 (default): a hit that touches the next one ends where that hit's onset
                       consonants start
``tight_onset``        the most of its first note the onset of a hit in a touching run may take
``onset_max``          s: the most the onset of a free-standing hit may take (0: as predicted)
``onset_cap.<class>``  s: the most one onset consonant of a class takes in a touching run
``onset_floor.<class>``  s: the least it gets there
``cons_cap.<class>``   s: consonant caps inside a hit, as for lines (none by default)
``gap``                s of silence a tight hit that ends in a vowel leaves before the next onset
``hold``               s a hit that ends in a vowel is sung past its end, then cut
``run_gap``            s: hits closer than this belong to one run (for ``accent``)
``accent.<n>``         dB of the n-th hit of a run (0, 1, ...; the list is cycled)
``pre``, ``release``   s kept before the first phoneme and after a final consonant
``chop_hz``, ``chop_db``  a loudness chop locked to the start of every take
``glottal``            1: a hit that starts with a vowel gets a closure before it
=====================  ======================================================================

The timbre settings of a hit (formant shift, timbre shift, breathiness, voicing, tension) are
the track's curves read at the hit's start; the loudness curve and note velocities are gain,
as for lines.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json

import numpy as np

from . import align as al
from .cache import stable_hash
from .engine import Options, PhraseInput, Runner, synth_phrase

TOUCH = 0.03  # s: hits closer than this touch
FADE_IN = 0.005  # s faded in after the silence before a take's first phoneme
FADE_OUT = 0.03  # s faded out after a take's cut
VOWEL_RELEASE = 0.01  # s kept after a take that ends in a held vowel
SALT_STEP = 7919
SPLITS = ("word", "syllable", "run")
DEFAULTS = {"split": 0.0, "bend": 0.0, "bend_from": 0.3, "glide": 0.04, "vibrato": 0.0, "stamp": 1.0, "tight": 1.0,
            "tight_onset": 0.4, "onset_max": 0.0, "salt": 0.0, "gap": 0.02, "hold": 0.07, "run_gap": 0.25,
            "pre": 0.012, "release": 0.05, "chop_hz": 0.0, "chop_db": 0.0, "glottal": 0.0}
# After a rest the duration model gives an initial stop most of the onset (its closure, which
# is silence there) and squeezes what follows it; in a touching run each onset consonant is
# held under these lengths before the onset is fitted into its room.
ONSET_CAP = {"stop": 0.025, "vstop": 0.03, "aff": 0.07, "fric": 0.07, "vfric": 0.06, "nasal": 0.06, "glide": 0.06}
TABLES = ("bend", "salt", "onset_cap", "onset_floor", "cons_cap", "accent")


def settings(pairs) -> dict:
    """The ad-lib options from a preset's ``adlib`` pairs: the scalars of :data:`DEFAULTS` and
    the tables ``bends`` / ``salts`` (by lyric), ``onset_cap`` / ``onset_floor`` / ``cons_cap``
    (by consonant class) and ``accent`` (a list of dB, None without one). Unknown names are
    ignored."""
    ad: dict = dict(DEFAULTS)
    tables: dict = {"bend": {}, "salt": {}, "onset_cap": dict(ONSET_CAP), "onset_floor": {}, "cons_cap": {},
                    "accent": {}}
    for name, value in pairs or ():
        head, _, tail = name.partition(".")
        if tail and head in tables:
            tables[head][tail] = float(value)
        elif name in ad:
            ad[name] = float(value)
    ad["split"] = SPLITS[int(min(len(SPLITS) - 1, max(0, round(ad["split"]))))]
    ad["bends"], ad["salts"] = tables["bend"], tables["salt"]
    for k in ("onset_cap", "onset_floor", "cons_cap"):
        ad[k] = {c: v for c, v in tables[k].items() if c in al.CONS_CLASSES and v > 0}
    steps = sorted((int(k), v) for k, v in tables["accent"].items() if k.isdigit())
    ad["accent"] = [v for _, v in steps] or None
    return ad


def by_lyric(table: dict, lyric: str, default: float) -> float:
    return table.get(lyric, table.get(lyric.lower(), default))


def hits_of(plan, notes, split: str = "word") -> list[list]:
    """The plan's sung units grouped into hits (lists of units)."""
    hits: list[list] = []
    for u in plan.units:
        if u.breath or u.closure or not u.nuclei:
            continue
        lyric = (notes[u.notes[0]].lyric or "") if u.notes else ""
        touch = bool(hits) and u.start - hits[-1][-1].end < TOUCH
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


def busy_spans(project, track) -> list[tuple[float, float]]:
    """Where ad-lib tracks of ``track``'s voice sound, in seconds (from 0.12 s before each of
    their notes to its end): a line of the same voice takes no breath there."""
    from .presets import track_preset

    b2s = project.tempo.beat_to_sec
    spans = set()
    for other in project.tracks:
        if other is track or other.voice != track.voice or track_preset(other).adlib is None:
            continue
        for n in other.notes:
            spans.add((round(b2s(n.onset) - 0.12, 3), round(b2s(n.onset + n.duration), 3)))
    return sorted(spans)


def take_id(seq, durs, midis, shape: str, delta_key, vib, bend_frame: int) -> str:
    """Names a distinct sound: what a stamped take is made once for and seeded by."""
    key = (tuple(seq), tuple(durs), tuple(midis), shape, delta_key, tuple(vib), bend_frame)
    return hashlib.sha1(json.dumps(key, default=str).encode()).hexdigest()[:10]


class Hits:
    """The hits of one ad-lib track and their takes.

    ``sing(buf, buf0)`` adds every hit that reaches the buffer (absolute bank-rate samples from
    ``buf0``). A hit's audio does not depend on the buffer: its geometry reads the hit before
    and after it only.
    """

    def __init__(self, project, track, plan, voice, style, ad: dict, bank, convert, shared: Runner, blend,
                 opts: Options, ident, cache, seeded: bool, detune: float, timbre, timer):
        self.project, self.track, self.plan, self.voice, self.style = project, track, plan, voice, style
        self.ad, self.bank, self.convert, self.shared, self.blend = ad, bank, convert, shared, blend
        self.opts, self.ident, self.cache, self.seeded, self.detune = opts, ident, cache, seeded, float(detune)
        self.timbre, self.timer = timbre, timer  # timbre(t) -> (tone, gender, offsets) at time t
        self.frame, self.hop, self.sr = bank.frame, bank.hop, bank.sample_rate
        self.notes = track.notes
        self.pitch_of = {tm.index: tm.pitch for tm in plan.timings}
        self.hits = hits_of(plan, self.notes, ad["split"])
        # the first note of a take starts here on the take's own grid
        self.t0 = int(round((al.PAD + 0.25) / self.frame)) * self.frame
        n = len(self.hits)
        self.touches = [h + 1 < n and self.hits[h + 1][0].start - self.hits[h][-1].end < TOUCH for h in range(n)]
        self.run_pos, pos = [], 0
        for h, units in enumerate(self.hits):
            pos = 0 if h == 0 or units[0].start - self.hits[h - 1][-1].end >= ad["run_gap"] else pos + 1
            self.run_pos.append(pos)
        b2s = project.tempo.beat_to_sec
        # frames per unit of every hit: the written lengths of its notes
        self.lens = [[max(2, int(round((b2s(self.notes[u.notes[-1]].onset + self.notes[u.notes[-1]].duration)
                                        - b2s(self.notes[u.notes[0]].onset)) / self.frame))) for u in units]
                     for units in self.hits]
        self._geo: dict[int, dict] = {}
        self._lay: dict[tuple, list[float]] = {}
        self._raw: dict[str, dict] = {}
        self._copies: dict[tuple, np.ndarray] = {}
        self.stats = {"phrases": 0, "cached": 0, "sung_s": 0.0}
        self.rows: list[dict] = []

    # ---- geometry
    def _starts(self, g: dict, end: float) -> list[float]:
        """Phoneme start times of geometry ``g`` sung to ``end``."""
        frame = self.frame
        k = (tuple(g["phones"]), tuple(g["lens"]), round(g["groups"][0]["pos"] / frame), round(end / frame),
             bool(g["cap"]), bool(g["floor"]))
        if k not in self._lay:
            ps = [x["pos"] for x in g["groups"]] + [end]
            fr = [int(round(x / frame)) for x in ps]
            word_dur = [max(1, y - x) for x, y in zip(fr, fr[1:])]
            pred = np.array(self.shared.predict_dur(g["phones"], g["word_div"], word_dur, g["ph_midi"]), float)
            for i in range(1, g["word_div"][0]):
                c = al.cons_class(g["phones"][i])
                if g["cap"].get(c):
                    pred[i] = min(pred[i], g["cap"][c] / frame)
                if g["floor"].get(c):
                    pred[i] = max(pred[i], g["floor"][c] / frame)
            self._lay[k] = al.align(g["groups"], end, pred, frame, cons_cap=self.ad["cons_cap"] or None,
                                    onset_keep=True)
        return self._lay[k]

    def geometry(self, h: int) -> dict:
        """The take of hit ``h`` on its own time grid, before any cut for the hit after it."""
        g = self._geo.get(h)
        if g is not None:
            return g
        ad, frame, t0 = self.ad, self.frame, self.t0
        units, lens = self.hits[h], self.lens[h]
        rel, c = [], t0
        for n in lens:
            rel.append((c, c + n * frame))
            c += n * frame
        local = [dataclasses.replace(u, start=a, end=b, breath=False) for u, (a, b) in zip(units, rel)]
        groups, end = al.phrase_groups(local, self.plan, self.convert, frame)
        if ad["glottal"] and len(groups[0]["phones"]) == 1 and self.bank.has("ja/cl") \
                and groups[1]["phones"][0] != "ja/N":
            groups[0]["phones"].append("ja/cl")
        room = [ad["onset_max"]] if ad["onset_max"] else []
        in_run = bool(ad["tight"]) and (self.touches[h] or (h > 0 and self.touches[h - 1]))
        if in_run:
            room.append(ad["tight_onset"] * lens[0] * frame)
        if room:
            groups[0]["pos"] = max(groups[0]["pos"], t0 - min(room) - 2 * frame)
        g = {"lens": lens, "rel": rel, "groups": groups, "end": end,
             "phones": [q for x in groups for q in x["phones"]], "word_div": [len(x["phones"]) for x in groups],
             "ph_midi": [x["midi"] for x in groups for _ in x["phones"]],
             "vowel_end": al.cons_class(groups[-1]["phones"][-1]) is None,
             "cap": ad["onset_cap"] if in_run else {}, "floor": ad["onset_floor"] if in_run else {},
             "midis": [float(self.pitch_of.get(u.notes[0], 60.0)) if u.notes else 60.0 for u in units]}
        g["hold"] = int(round(ad["hold"] / frame)) * frame if g["vowel_end"] else 0.0
        starts = self._starts(g, end + g["hold"])
        g["onset_len"] = t0 - starts[0] if len(starts) else 0.0
        g["sound"] = (tuple(g["phones"]), tuple(lens), tuple(g["midis"]))
        self._geo[h] = g
        return g

    def cut(self, h: int) -> float:
        """Seconds hit ``h`` stops before its written end, to leave room for the hit after it.
        The last hit of a run of equal hits is cut like the one before it."""
        if not self.ad["tight"]:
            return 0.0
        g = self.geometry(h)
        if self.touches[h]:
            return min(self.geometry(h + 1)["onset_len"] + (self.ad["gap"] if g["vowel_end"] else 0.0),
                       0.5 * g["lens"][-1] * self.frame)
        if h > 0 and self.touches[h - 1] and self.geometry(h - 1)["sound"] == g["sound"]:
            return self.cut(h - 1)
        return 0.0

    def span(self, h: int) -> tuple[float, float]:
        """Seconds outside which hit ``h`` has no sound, whatever the duration model predicts."""
        start = self.hits[h][0].start
        return (start - self.t0 - 2 * self.frame,
                start + sum(self.lens[h]) * self.frame + max(self.ad["release"], VOWEL_RELEASE) + FADE_OUT
                + 2 * self.frame)

    # ---- takes
    def _take(self, h: int, g: dict, end: float, cut: float) -> tuple[str, dict]:
        """``(id, take)`` of hit ``h`` sung to ``end``: from this render's takes, the cache or
        the models."""
        ad, frame, t0, sr = self.ad, self.frame, self.t0, self.sr
        units = self.hits[h]
        u0 = units[0]
        hold, rel, midis = g["hold"], g["rel"], g["midis"]
        starts = self._starts(g, end + hold)
        b = [int(round(x / frame)) for x in starts] + [int(round((end + hold) / frame))]
        for j in range(1, len(b)):
            b[j] = max(b[j], b[j - 1] + 1)
        durs = [al.HEAD] + [y - x for x, y in zip(b, b[1:])] + [al.TAIL]
        seq = ["SP"] + g["phones"][1:] + ["SP"]
        f_start = b[0] - al.HEAD
        n = sum(durs)
        tt = (f_start + np.arange(n)) * frame  # take time of every frame
        lyric = (self.notes[u0.notes[0]].lyric or "") if u0.notes else ""
        bend = float(by_lyric(ad["bends"], lyric, ad["bend"]))
        shape = "flat" if bend == 0 else ("fall" if bend < 0 else "rise")
        bend_end = g["end"] - cut  # a fall or a rise is complete where the hit stops sounding
        from ..engine.expression import evaluate_curve
        from ..project import CURVE_SPECS

        b2s = self.project.tempo.beat_to_sec
        delta = np.clip(evaluate_curve(self.track.curves.get("pitchDelta"), u0.start + (tt - t0), b2s, 0.0),
                        *CURVE_SPECS["pitchDelta"][1:3]) / 100.0
        dkey = tuple(np.round(delta, 2)) if np.any(np.abs(delta) > 0.005) else ()
        vib = []
        if ad["vibrato"]:
            for u in units:
                at = self.notes[u.notes[0]].attrs if u.notes else {}
                vd = at.get("vibrato_depth")
                vd = self.voice.vibrato_depth * self.style.vibrato_depth if vd is None else float(vd)
                vr = at.get("vibrato_rate")
                vr = self.voice.vibrato_rate * self.style.vibrato_rate if vr is None else float(vr)
                vib.append((round(float(vd), 3), round(float(vr), 3)))
        kid = take_id(seq, durs, midis, shape, dkey, vib, round(bend_end / frame) if shape != "flat" else 0)
        if not ad["stamp"]:
            kid = f"{kid}@{h}"
        tone, gender, offsets = self.timbre(u0.start)
        # designed pitch: the written notes as steps with a short glide, flat through consonants and rests
        p = np.full(n, midis[0])
        for (a_, _), m_prev, m_ in zip(rel[1:], midis, midis[1:]):
            x = np.clip((tt - (a_ - ad["glide"] / 2)) / max(ad["glide"], 1e-3), 0.0, 1.0)
            p += (m_ - m_prev) * (3 * x ** 2 - 2 * x ** 3)
        if bend:
            a_ = t0 + ad["bend_from"] * (bend_end - t0)
            p += bend * np.clip((tt - a_) / max(bend_end - a_, frame), 0.0, 1.2)
        for (a_, b_), (vd, vr) in zip(rel, vib):
            m_ = (tt >= a_) & (tt < b_ + 0.1)
            if vd > 0 and vr > 0:
                p[m_] += vd * np.sin(2 * np.pi * vr * (tt[m_] - a_))
        p_semi = p + delta + self.detune / 100.0
        is_vowel = [False] + [j == 0 and bool(x.get("vowel", False)) and q not in al.REST_SYMBOLS
                              for x in g["groups"] for j, q in enumerate(x["phones"])][1:] + [False]
        opts = self.opts
        slow = np.append(al.slow_factors(g["groups"], end + hold, opts.slow_note, opts.slow_max), 1.0) \
            if opts.slow_note else np.ones(len(seq))
        seed = None
        if self.seeded:
            place = int(kid[:7], 16) if ad["stamp"] else int(round(u0.start / frame))
            place += int(by_lyric(ad["salts"], lyric, ad["salt"])) * SALT_STEP
            seed = al.phrase_seed(int(self.track.seed or 0), place, n)
        key = stable_hash(self.ident, "adlib", seq, durs, is_vowel, slow, p_semi, tone, gender, offsets, seed)
        take = self._raw.get(key)
        if take is not None:
            return key, take
        wav = self.cache.get(key) if self.cache is not None else None
        if wav is not None and len(wav) == n * self.hop:
            self.stats["cached"] += 1
        else:
            run = Runner(self.bank, seed, self.blend, self.timer)
            try:
                wav = synth_phrase(run, PhraseInput(seq, durs, is_vowel, slow, p_semi, tone, gender, offsets), opts)
            finally:
                run.close()
            if self.cache is not None:
                self.cache.put(key, wav)
        wav = np.array(wav, np.float32)
        # silence before the first phoneme
        i0 = max(0, int((al.HEAD * frame - ad["pre"]) * sr))
        fi = int(FADE_IN * sr)
        wav[:i0] = 0.0
        wav[i0:i0 + fi] *= np.linspace(0, 1, fi, dtype=np.float32)
        if ad["chop_hz"] and ad["chop_db"]:
            tw = np.arange(len(wav)) / sr - (t0 / frame - f_start) * frame
            ph = np.clip(tw, 0.0, None) * float(ad["chop_hz"])
            wav *= 10 ** (float(ad["chop_db"]) * (0.5 - 0.5 * np.cos(2 * np.pi * ph)) / 20).astype(np.float32)
        take = {"wav": wav, "at": int(round((t0 / frame - f_start) * self.hop)), "f_start": f_start, "seq": seq,
                "durs": durs, "is_vowel": is_vowel, "id": kid, "shape": shape, "lyric": lyric}
        self._raw[key] = take
        self.stats["phrases"] += 1
        self.stats["sung_s"] += n * frame
        return key, take

    def sing(self, buf: np.ndarray, buf0: int, warnings: list[str] | None = None) -> None:
        """Add every hit that reaches ``buf`` (whose first sample is absolute sample ``buf0``).
        A hit that cannot be sung is left out and named in ``warnings``."""
        from .voicebank import VoicebankError

        sr = self.sr
        lo, hi = buf0 / sr, (buf0 + len(buf)) / sr
        self.stats.setdefault("failed", 0)
        for h, units in enumerate(self.hits):
            a, b = self.span(h)
            if b < lo or a > hi:
                continue
            try:
                w, s0, row = self._copy(h)
            except VoicebankError:
                raise
            except Exception as exc:  # one hit that cannot be sung must not lose the others
                self.stats["failed"] += 1
                if warnings is not None:
                    beat = self.project.tempo.sec_to_beat(max(0.0, units[0].start))
                    warnings.append(f"the hit at beat {beat:.2f} could not be sung ({type(exc).__name__}: {exc})")
                continue
            x, y = max(s0, buf0), min(s0 + len(w), buf0 + len(buf))
            if y > x:
                buf[x - buf0:y - buf0] += w[x - s0:y - s0]
            self.rows.append(row)

    def _copy(self, h: int) -> tuple[np.ndarray, int, dict]:
        """``(samples, first absolute sample, report row)`` of hit ``h`` at its place."""
        ad, frame, sr = self.ad, self.frame, self.sr
        units = self.hits[h]
        g, u0 = self.geometry(h), units[0]
        cut = self.cut(h)
        end = g["end"]
        if not g["vowel_end"]:  # a final consonant is sung earlier; a final vowel is cut after the vocoder
            end = end - int(round(cut / frame)) * frame
        key, take = self._take(h, g, end, cut)
        # the copy for this place: cut after the release (after `cut` before the end for a final vowel)
        stop = end + VOWEL_RELEASE - cut if g["vowel_end"] else end + ad["release"]
        at = int((stop - take["f_start"] * frame) * sr)
        w = self._copies.get((key, at))
        if w is None:
            w = take["wav"].copy()
            i1 = max(0, min(len(w), at))
            fo = min(int(FADE_OUT * sr), len(w) - i1)
            if fo > 0:
                w[i1:i1 + fo] *= np.linspace(1, 0, fo, dtype=np.float32)
            w[i1 + max(fo, 0):] = 0.0
            self._copies[(key, at)] = w
        gain = 10 ** (float(ad["accent"][self.run_pos[h] % len(ad["accent"])]) / 20) if ad["accent"] else 1.0
        row = {"t": round(float(u0.start), 4), "end": round(float(units[-1].end), 4), "lyric": take["lyric"],
               "take": take["id"], "run_pos": self.run_pos[h], "touches": bool(self.touches[h]),
               "gain_db": round(20 * np.log10(gain), 2), "onset_s": round(float(g["onset_len"]), 4),
               "cut_s": round(float(cut), 4), "shape": take["shape"],
               "sample": int(round(u0.start * sr)) - take["at"], "phones": list(take["seq"]),
               "frames": [int(d) for d in take["durs"]], "vowels": [bool(v) for v in take["is_vowel"]]}
        return w * np.float32(gain), row["sample"], row
