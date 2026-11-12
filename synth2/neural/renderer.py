"""The neural engine as a :class:`~synth2.engine.backend.VoiceBackend`.

One track is rendered like this:

1. ``plan_track`` plans the whole track, exactly as for the formant engine.
2. ``align`` cuts the plan into phrases; the phrases that can reach the render window are
   laid out on the track's frame grid (the duration model runs for those only), with a
   breath in the gaps between lines and the consonant options of the track's dial setting.
3. Every phrase's waveform comes from the phrase cache or from ``engine.synth_phrase``. The
   cache holds what the phrase pipeline returned, before level, gain rides and effects, under
   a key made of everything the waveform depends on.
4. The phrases are overlap-added onto the absolute sample grid of the bank's rate, gain is
   applied as a function of absolute time, then effects, then the resampler puts the
   window on the project's sample grid.

A phrase's audio does not depend on the window, so a partial render equals the same span of
the full render.

A track in ad-lib mode is sung hit by hit instead of phrase by phrase (``adlib.py``). A
track with the role ``double`` is not sung when the project has a track for it to double:
its audio is that track's take over the double's own notes. The report names the track's
role; the mixer (``engine/vocal_chain.py``) gives it the role's chain.
"""
from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import replace

import numpy as np

from . import adlib
from . import align as al
from . import cache as phrase_cache
from . import fx, models, resample
from .cache import stable_hash
from .engine import ENGINE_VERSION, Options, PhraseInput, Runner, Timer, reproducible, steps_for, synth_phrase
from .phonemes import PhonemeMap
from .pitch import PITCH_STEPS, BankPitch, default_expr, default_vibrato
from .presets import BREATH_DB, DETUNE, GENDER_LIMIT, VoicePreset, track_preset
from .voicebank import Voicebank, VoicebankError, load_installed

MAX_BANKS = 2  # voicebanks kept loaded
FADE = 0.01  # s faded in and out on every phrase before it is added
PITCH_ENV = "SYNTH2_NEURAL_PITCH"
VOICING_DB = 40.0  # dB taken off the predicted voicing when the voicing curve is 0
TENSION_SCALE = 4.0  # the bank's tension units per unit of the tension curve
TENSION_GAIN_DB = 1.5  # dB per unit of the tension curve for a bank without a tension input
COMFORT_RANGE = (50, 72)  # MIDI notes a bank sings well when its manifest gives no "range"
VELOCITY_SMOOTH = 5  # frames of smoothing on the steps of note velocities
COPY_LEAD, COPY_TAIL, COPY_FADE = 0.06, 0.12, 0.02  # s around a double's notes that the doubled take is copied

_lock = threading.Lock()
_banks: "OrderedDict[tuple, Voicebank]" = OrderedDict()
_backends: dict = {}


def get_backend(bank: str | None = None) -> "NeuralBackend":
    """The backend that sings with voicebank ``bank`` (None: each track's preset or the
    default bank). One instance per bank id."""
    with _lock:
        hit = _backends.get(bank)
        if hit is None:
            hit = _backends[bank] = NeuralBackend(bank)
        return hit


def reset() -> None:
    """Forget loaded voicebanks and their sessions (after installing or removing a model)."""
    with _lock:
        _banks.clear()
        _backends.clear()


def load_bank(bank_id: str | None = None) -> Voicebank:
    """The loaded voicebank with this id (None: the default one). At most ``MAX_BANKS`` stay
    loaded; the least recently used one is dropped."""
    model = models.find("voicebank", bank_id)
    if model is None:
        from ..engine.backend import EngineUnavailable

        what = f"voicebank {bank_id!r} is not installed" if bank_id else "no voicebank is installed"
        raise EngineUnavailable(f"{what} in {models.models_dir()}")
    key = (model.id, str(model.root))
    with _lock:
        bank = _banks.get(key)
        if bank is None:
            bank = _banks[key] = load_installed(model)
            bank.comfort_range = _comfort_range(model)  # MIDI notes the bank sings well
        _banks.move_to_end(key)
        while len(_banks) > MAX_BANKS:
            _banks.popitem(last=False)
        return bank


def _comfort_range(model) -> tuple[float, float]:
    r = model.manifest.get("range")
    try:
        return float(r[0]), float(r[1])
    except (TypeError, ValueError, IndexError):
        return COMFORT_RANGE


def window_span(t0: float, t1: float, sr: int, bank_sr: int, pre_roll: float = 0.0) -> tuple[int, int, int, int]:
    """``(out_start, n_out, buf_start, buf_end)``: the project samples of ``[t0, t1)`` and the
    span of bank-rate samples (absolute indices) they are made from, with the resampler's
    margin and ``pre_roll`` seconds for effects in front."""
    out_start = int(round(t0 * sr))
    n_out = max(0, int(round((t1 - t0) * sr)))
    if sr == bank_sr:
        lo, hi = out_start, out_start + n_out
    else:
        m = int(resample.margin(bank_sr, sr)) + 2
        lo = (out_start * bank_sr) // sr - m
        hi = -((-(out_start + n_out) * bank_sr) // sr) + m
    return out_start, n_out, lo - int(np.ceil(pre_roll * bank_sr)), hi


def place(buf: np.ndarray, buf_start: int, wav: np.ndarray, start: int, sr: int) -> None:
    """Add a phrase's waveform, faded over ``FADE`` seconds at both ends, into ``buf`` (whose
    first sample is absolute sample ``buf_start``) at absolute sample ``start``."""
    fade = int(FADE * sr)
    wav = np.array(wav, np.float32)
    if 0 < fade <= len(wav) // 2:
        wav[:fade] *= np.linspace(0, 1, fade)
        wav[-fade:] *= np.linspace(1, 0, fade)
    a = max(start, buf_start)
    b = min(start + len(wav), buf_start + len(buf))
    if b > a:
        buf[a - buf_start:b - buf_start] += wav[a - start:b - start]


def velocity_db(t: np.ndarray, timings) -> np.ndarray:
    """dB per frame from the notes' ``velocity`` attributes (0 where a note has none). A note's
    value holds from its onset consonants to the next note's; steps are smoothed."""
    out = np.zeros(len(t))
    for i, nt in enumerate(timings):
        v = nt.attrs.get("velocity")
        if v is None:
            continue
        nxt = timings[i + 1].start - timings[i + 1].lead if i + 1 < len(timings) else np.inf
        hi = min(nxt, nt.end + 0.1) if nt.phrase_end or i + 1 == len(timings) else nxt
        out[(t >= nt.start - nt.lead) & (t < hi)] = 20 * np.log10(max(float(v), 0.05))
    if out.any() and len(out) > VELOCITY_SMOOTH:
        k = np.hanning(VELOCITY_SMOOTH + 2)[1:-1]
        out = np.convolve(out, k / k.sum(), mode="same")
    return out


class NeuralBackend:
    """Sings tracks with a DiffSinger voicebank."""

    name = "neural"

    def __init__(self, bank: str | None = None):
        self.bank_id = bank

    def voices(self) -> list[str]:
        from ..voices import VOICES

        return list(VOICES)

    def bank_for(self, preset: VoicePreset) -> tuple[Voicebank, list[str]]:
        """The voicebank a track sings with: this backend's, else the preset's, else the default."""
        if self.bank_id:
            return load_bank(self.bank_id), []
        if preset.bank and models.find("voicebank", preset.bank) is not None:
            return load_bank(preset.bank), []
        notes = [f"voicebank {preset.bank!r} is not installed; the default voicebank sings"] if preset.bank else []
        return load_bank(None), notes

    def render_track(self, project, track, sample_rate: int, t0: float, t1: float,
                     note_range: tuple[float, float] | None = None) -> tuple[np.ndarray, dict]:
        """Render ``track`` dry over ``[t0, t1)`` seconds: ``int(round((t1 - t0) * sample_rate))``
        samples, the first one at absolute project sample ``int(round(t0 * sample_rate))``.

        The report carries the track's ``role`` (and, for a double, ``neural.copy_of``); the
        mixer gives a track with a role that role's chain (``engine/vocal_chain.py``). A
        ``double`` whose project has a track to double is not sung: its audio is that track's
        take over the double's notes."""
        return self._render(project, track, int(sample_rate), t0, t1, note_range)

    def _render(self, project, track, sr: int, t0: float, t1: float, note_range=None, doubling=()):
        from ..engine import vocal_chain
        from ..engine.expression import get_style
        from ..engine.planner import plan_track
        from ..engine.renderer import track_report
        from ..voices import get_voice

        voice = get_voice(track.voice).with_params(track.voice_params)
        overrides = {k[6:]: v for k, v in track.voice_params.items() if k.startswith("style.")}
        style = get_style(track.style or voice.default_style, overrides)
        rng = np.random.default_rng(track.seed)
        plan = plan_track(track.notes, project.tempo, track.language, style, rng)
        preset = track_preset(track)
        bank, warnings = self.bank_for(preset)
        role, source = vocal_chain.parse_role(getattr(track, "role", ""))
        if getattr(track, "role", "") and not role:
            warnings.append(f"unknown role {track.role!r} ignored (roles: {', '.join(vocal_chain.ROLES)})")
        n_out = max(0, int(round((t1 - t0) * sr)))
        info: dict = {}
        sources = self._doubled(project, track, source, doubling) if role == "double" else []
        if sources and n_out:
            audio = self._copy(project, track, sources, sr, t0, t1, doubling, info)
        elif plan.units and n_out:
            audio = self._sing(project, track, plan, voice, style, rng, preset, bank, sr, t0, t1, warnings, info)
            self._range_warning(track.notes if note_range is None else track.notes_in_range(*note_range), bank,
                                preset, warnings)
        else:
            audio = np.zeros(n_out, np.float32)
            if not plan.units:
                warnings.append("the track has nothing to sing")
        report = track_report(project, track, plan, voice, style, audio, note_range, engine=self.name,
                              check_range=False)
        report["warnings"] = report["warnings"] + [w for w in warnings if w not in report["warnings"]]
        report["credits"] = list(bank.credits)
        report["bank"] = bank.id
        report["role"] = role
        info.setdefault("dial", preset.dial)
        report["neural"] = info
        return audio, report

    @staticmethod
    def _doubled(project, track, source: str | None, doubling=()) -> list:
        """The tracks a double copies: the one it names, else the leads of its voice."""
        from ..engine import vocal_chain

        if source:
            try:
                named = project.track(source)
            except LookupError:
                named = None
            found = [named] if named is not None else []
        else:
            found = [t for t in project.tracks
                     if vocal_chain.parse_role(getattr(t, "role", ""))[0] == "lead" and t.voice == track.voice]
        return [t for t in found if t is not track and t.notes and t.id not in doubling]

    def _copy(self, project, track, sources, sr: int, t0: float, t1: float, doubling, info: dict) -> np.ndarray:
        """The takes of ``sources`` over the notes of ``track`` (from ``COPY_LEAD`` s before to
        ``COPY_TAIL`` s after each, ``COPY_FADE`` s fades), with the track's loudness curve."""
        from ..engine import vocal_chain
        from ..engine.expression import evaluate_curve

        n = max(0, int(round((t1 - t0) * sr)))
        out = np.zeros(n, np.float32)
        for src in sources:
            a, _ = self._render(project, src, sr, t0, t1, None, (*doubling, track.id))
            out[: len(a)] += a[:n]
        b2s = project.tempo.beat_to_sec
        k = int(COPY_FADE * sr)
        first = int(round(t0 * sr)) - 2 * k  # the gate is built with a margin, so a window's edge does not shape it
        gate = np.zeros(n + 4 * k)
        for nt in track.notes:
            a = int((b2s(nt.onset) - COPY_LEAD) * sr) - first
            b = int((b2s(nt.onset + nt.duration) + COPY_TAIL) * sr) - first
            gate[max(0, a):max(0, min(len(gate), b))] = 1.0
        win = np.hanning(2 * k + 1)
        gate = np.convolve(gate, win / win.sum(), mode="same")[2 * k:2 * k + n]
        hop = max(1, int(round(0.01 * sr)))
        s0 = int(round(t0 * sr))
        grid = np.arange((s0 // hop) * hop, s0 + n + hop, hop)  # absolute samples: the same in every window
        ld = np.clip(evaluate_curve(track.curves.get("loudness"), grid / sr, b2s, 0.0), -48.0, 12.0)
        gain = np.interp(s0 + np.arange(n), grid, 10 ** (ld / 20.0))
        info.update(pitch="copy", copy=[s.name for s in sources], phrases=0, cached=0, sung_s=0.0, alignment=[],
                    copy_of=vocal_chain.parse_role(getattr(sources[0], "role", ""))[0] or "lead")
        return (out * gate * gain).astype(np.float32)

    # ---- one track
    def _sing(self, project, track, plan, voice, style, rng, preset: VoicePreset, bank: Voicebank, sr: int,
              t0: float, t1: float, warnings: list[str], info: dict) -> np.ndarray:
        from ..engine.expression import build_pitch, evaluate_curve
        from ..project import CURVE_SPECS

        frame, bsr, hop = bank.frame, bank.sample_rate, bank.hop
        fx_spec = []
        for item in preset.fx_spec():
            try:
                fx.parse_fx(item)
            except ValueError as exc:
                warnings.append(f"effect {item.split(':')[0]!r} ignored: {exc}")
            else:
                fx_spec.append(item)
        pre_roll = float(fx.fx_tail(fx_spec)) if fx_spec else 0.0
        out_start, n_out, buf0, buf1 = window_span(t0, t1, sr, bsr, pre_roll)

        # the track's frame grid and everything that is a function of time on it
        k_off = int(np.floor((min(0.0, plan.start) - 1.0) / frame))
        n_frames = int(np.ceil((plan.end + 1.0) / frame)) - k_off + 1
        t = (k_off + np.arange(n_frames)) * frame
        b2s = project.tempo.beat_to_sec
        curves = {name: np.clip(evaluate_curve(track.curves.get(name), t, b2s, spec[0]), spec[1], spec[2])
                  for name, spec in CURVE_SPECS.items()}
        rng_state = rng.bit_generator.state
        s2 = build_pitch(t, plan.timings, style, voice, rng, vibrato_env=curves["vibratoEnv"])

        hits_mode = preset.adlib is not None  # detached hits with a designed pitch instead of lines
        source = (os.environ.get(PITCH_ENV) or "").strip().lower()
        if source not in ("bank", "expression"):
            source = "bank"
        if preset.expr is not None and preset.expr <= 0:
            source = "expression"
        if hits_mode:
            source = "adlib"
        elif source == "bank" and bank.pitch is None:
            source = "expression"
            if bank.pitch_note:
                warnings.append(f"the voicebank's pitch predictor is not used ({bank.pitch_note})")
        detune = float(preset.detune) if preset.detune is not None else (DETUNE if source == "bank" else 0.0)
        delta = curves["pitchDelta"] / 100.0 + detune / 100.0
        pitch = s2 + delta
        bp = None
        if source == "bank":
            # the same stream and the same seed sequence as rng (build_pitch keys per-note draws on
            # the seed sequence), so the two curves differ in the vibrato only
            r2 = np.random.default_rng(track.seed)
            r2.bit_generator.state = rng_state
            s2_flat = build_pitch(t, plan.timings, style, voice, r2, vibrato_env=np.zeros(len(t)))
            expr = default_expr(style, voice) if preset.expr is None else float(preset.expr)
            bp = BankPitch(t, plan.timings, style, s2, s2_flat, curves["vibratoEnv"], expr, PITCH_STEPS,
                           default_vibrato(style, track), bank.is_vowel, frame)
            info["expr"] = expr
        info["pitch"] = source

        acoustic_takes_tension = bool(bank.acoustic.config.get("use_tension_embed"))
        tone = preset.tone + curves["toneShift"] / 100.0
        gender = np.clip(preset.gender - curves["gender"], -GENDER_LIMIT, GENDER_LIMIT)
        off = {"breathiness": preset.breath_db + BREATH_DB * curves["breathiness"],
               "voicing": preset.voicing_db + VOICING_DB * (curves["voicing"] - 1.0)}
        gain_db = preset.level_db + float(voice.loudness_db) + curves["loudness"] + velocity_db(t, plan.timings)
        if acoustic_takes_tension:
            off["tension"] = preset.tension + TENSION_SCALE * curves["tension"]
        else:
            gain_db = gain_db + TENSION_GAIN_DB * curves["tension"]

        seeded = reproducible()
        breath = None if hits_mode else preset.breath()
        if breath:
            busy = adlib.busy_spans(project, track)
            if busy:  # no breath where an ad-lib of the same voice sounds on another track
                breath["avoid"] = busy
        opts = Options(steps=steps_for(), velocity=preset.velocity,
                       slow_note=al.SLOW_NOTE if preset.slow is None else (preset.slow if preset.slow > 0 else None),
                       voicing_floor=al.VOICING_FLOOR if preset.voicing_floor is None else
                       (preset.voicing_floor if preset.voicing_floor > 0 else None),
                       cons_floor=preset.cons_floor, vowel_breath_db=preset.vowel_breath_db,
                       cons_gain=preset.cons_gain, cons_gain_hf=preset.cons_gain_hf)
        if not hits_mode:  # a hit has its own consonant options (adlib.py) and no breath
            opts = replace(opts, cons_cap=preset.cons_cap, cons_cap_keep=preset.cons_cap_keep,
                           cons_cap_share=preset.cons_cap_share, onset_keep=preset.onset_keep, burst=preset.burst,
                           breath_db=breath["breathiness"] if breath else None,
                           min_breath=breath["min_len"] if breath else 0.0)
        weights = preset.mode_weights()
        unknown = sorted(m for m in weights if m not in bank.modes)
        if unknown:
            warnings.append(f"voicebank {bank.id!r} has no voice mode {', '.join(unknown)} (it has: "
                            f"{', '.join(sorted(bank.modes)) or 'none'}); ignored")
        blend = bank.blend({m: w for m, w in weights.items() if m in bank.modes})
        ident = [ENGINE_VERSION, bank.identity(), [list(b) for b in blend] if blend else None, opts.key(),
                 "seeded" if seeded else "shared"]
        cache = phrase_cache.default_cache() if phrase_cache.enabled() else None
        pm = PhonemeMap(bank.has)
        timer = Timer()
        shared = Runner(bank, None, blend, timer)

        buf = np.zeros(max(0, buf1 - buf0), np.float32)
        lo_s, hi_s = buf0 / bsr, buf1 / bsr
        alignment: list[dict] = []  # per phrase: first frame, phonemes, their lengths in frames, note starters
        stats = {"phrases": 0, "cached": 0, "sung_s": 0.0, "pitch_fallbacks": 0, "pitch_repaired": 0, "failed": 0}
        if hits_mode:
            def timbre(at: float):
                """Timbre shift, formant shift and variance offsets at ``at`` seconds."""
                c = {name: float(np.clip(evaluate_curve(track.curves.get(name), np.array([at]), b2s, spec[0]),
                                         spec[1], spec[2])[0]) for name, spec in CURVE_SPECS.items()}
                o = {"breathiness": preset.breath_db + BREATH_DB * c["breathiness"],
                     "voicing": preset.voicing_db + VOICING_DB * (c["voicing"] - 1.0)}
                if acoustic_takes_tension:
                    o["tension"] = preset.tension + TENSION_SCALE * c["tension"]
                return (preset.tone + c["toneShift"] / 100.0,
                        float(np.clip(preset.gender - c["gender"], -GENDER_LIMIT, GENDER_LIMIT)), o)

            hits = adlib.Hits(project, track, plan, voice, style, adlib.settings(preset.adlib), bank, pm.convert,
                              shared, blend, opts, ident, cache, seeded, detune, timbre, timer)
            hits.sing(buf, buf0, warnings)
            stats.update(hits.stats)
            info["hits"] = hits.rows
            phrases = []
        else:
            phrases, cuts = al.split_long(al.build_phrases(plan))
            if cuts:
                warnings.append(f"{cuts} phrase(s) longer than {al.MAX_PHRASE:g} s were split at their longest gap; "
                                "add a rest or a breath to choose where")
        before = breath["length"] if breath else 0.0
        prev_end = None  # where the phrase before ended: a phrase may start with a breath in the silence
        for units in phrases:
            this_prev, prev_end = prev_end, units[-1].end
            a, b = al.phrase_bounds(units, frame, before)
            if b < lo_s or a > hi_s:
                continue
            try:
                got = self._phrase(bank, plan, units, pm, shared, bp, pitch, delta, tone, gender, off, opts, blend,
                                   ident, cache, track, seeded, k_off, n_frames, timer, (buf0, buf1), breath,
                                   this_prev)
                if got is None:
                    continue
                wav, f_start, meta = got
            except VoicebankError:
                raise
            except Exception as exc:  # one phrase that cannot be sung must not lose the others
                stats["failed"] += 1
                beat = project.tempo.sec_to_beat(max(0.0, units[0].start))
                warnings.append(f"the phrase at beat {beat:.2f} could not be sung ({type(exc).__name__}: {exc})")
                continue
            place(buf, buf0, wav, f_start * hop, bsr)
            stats["phrases"] += 1
            stats["sung_s"] += len(wav) / bsr
            for k in ("cached", "pitch_fallbacks", "pitch_repaired"):
                stats[k] += meta.get(k, 0)
            alignment.append(meta["layout"])
        if stats["pitch_fallbacks"]:
            warnings.append(f"the voicebank's pitch predictor could not sing {stats['pitch_fallbacks']} phrase(s); "
                            "Synth2's own pitch curve was used there")
        warnings.extend(pm.warnings())

        if np.any(gain_db != 0.0) and len(buf):
            ts = (buf0 + np.arange(len(buf))) / bsr
            buf = (buf * 10.0 ** (np.interp(ts, t, gain_db) / 20.0)).astype(np.float32)
        if fx_spec and len(buf):
            buf = np.asarray(fx.apply_fx(buf, bsr, fx_spec, t0=buf0 / bsr, seed=int(track.seed or 0)), np.float32)
        if sr == bsr:
            out = buf[out_start - buf0:out_start - buf0 + n_out]
        else:
            out = resample.resample(buf, bsr, sr, in_start=buf0, out_start=out_start, n_out=n_out)
        if len(out) < n_out:
            out = np.pad(out, (0, n_out - len(out)))
        info.update(stats, steps=opts.steps, seeded=seeded, preset=preset.summary(), dial=preset.dial,
                    detune=detune, frame_s=frame, alignment=alignment,
                    seconds={k: round(v, 3) for k, v in timer.t.items()})
        return np.ascontiguousarray(out, np.float32)

    def _phrase(self, bank, plan, units, pm, shared: Runner, bp, pitch, delta, tone, gender, off, opts: Options,
                blend, ident, cache, track, seeded: bool, k_off: int, n_frames: int, timer: Timer, span,
                breath=None, prev_end=None):
        """``(waveform, first frame, meta)`` of one phrase, from the cache or the models; None
        when the phrase has no sample inside ``span`` (absolute bank-rate samples)."""
        frame = bank.frame
        groups, end = al.phrase_groups(units, plan, pm.convert, frame, breath, prev_end)
        phones, word_div, word_dur, ph_midi = al.dur_inputs(groups, end, frame)
        pred = shared.predict_dur(phones, word_div, word_dur, ph_midi)
        seq, durs, f_start, is_vowel = al.phrase_frames(groups, end, pred, frame, opts.min_vowel, opts.min_breath,
                                                        **opts.caps())
        n = int(sum(durs))
        if (f_start + n) * bank.hop <= span[0] or f_start * bank.hop >= span[1]:
            return None
        layout = {"frame": int(f_start), "phones": list(seq), "frames": [int(d) for d in durs],
                  "vowels": [bool(v) for v in is_vowel]}
        a = f_start - k_off
        # the phrase's frames on the track grid (an index array when it reaches past the grid)
        sl = slice(a, a + n) if a >= 0 and a + n <= n_frames else np.clip(np.arange(a, a + n), 0, n_frames - 1)
        slow = np.append(al.slow_factors(groups, end, opts.slow_note, opts.slow_max), 1.0) if opts.slow_note \
            else np.ones(len(seq))
        if opts.burst:
            slow = al.slow_consonants(slow, seq, durs, frame, dict(opts.burst))
        seed = al.phrase_seed(int(track.seed or 0), f_start, n) if seeded else None
        ni = bp.notes(units, seq, durs, f_start) if bp is not None else None
        pitch_key = bp.key(ni, sl) if bp is not None else ["expression", pitch[sl]]
        offsets = {k: v[sl] for k, v in off.items()}
        key = stable_hash(ident, seq, durs, is_vowel, slow, pitch_key, delta[sl], tone[sl], gender[sl], offsets, seed)
        meta_key = stable_hash(key, "meta")
        if cache is not None:
            wav = cache.get(key)
            if wav is not None and len(wav) == n * bank.hop:
                m = cache.get(meta_key)
                fb, rep = (int(m[0]), int(m[1])) if m is not None and len(m) == 2 else (0, 0)
                return wav, f_start, {"cached": 1, "pitch_fallbacks": fb, "pitch_repaired": rep, "layout": layout}
        run = Runner(bank, seed, blend, timer)
        try:
            fallback, repaired = "", 0
            if bp is not None:
                p, fallback, repaired = bp.phrase(run.predict_pitch, ni, seq, durs, sl)
                p_semi = p + delta[sl]
            else:
                p_semi = pitch[sl]
            ph = PhraseInput(seq, durs, is_vowel, slow, p_semi, tone[sl], gender[sl], offsets)
            wav = synth_phrase(run, ph, opts)
        finally:
            run.close()
        if cache is not None:
            cache.put(key, wav)
            cache.put(meta_key, np.array([1.0 if fallback else 0.0, float(repaired)], np.float32))
        return wav, f_start, {"pitch_fallbacks": 1 if fallback else 0, "pitch_repaired": repaired, "layout": layout}

    @staticmethod
    def _range_warning(notes, bank, preset: VoicePreset, warnings: list[str]) -> None:
        from ..timing import midi_to_note

        lo, hi = bank.comfort_range or COMFORT_RANGE
        lo, hi = lo - preset.tone, hi - preset.tone
        high = sum(1 for n in notes if n.pitch > hi + 2)
        low = sum(1 for n in notes if n.pitch < lo - 2)
        if high:
            warnings.append(f"{high} note(s) lie above the voicebank's comfortable range (up to "
                            f"{midi_to_note(int(round(hi)))}); high notes sound thin, consider transposing the "
                            "part down")
        if low:
            warnings.append(f"{low} note(s) lie below the voicebank's comfortable range (from "
                            f"{midi_to_note(int(round(lo)))}); consider transposing the part up")
