"""How Synth2's voices map onto a voicebank, and how clearly the bank articulates.

A voicebank has one singer (and perhaps a few voice modes); Synth2 has a cast of voices. A
:class:`VoicePreset` turns a voice id into settings of the neural engine: a formant shift, a
timbre shift, variance offsets, a level, optional character effects and optionally another
bank. ``Track.voice_params`` overrides any of it with numeric ``neural.*`` parameters:

==============================  ===============================================================
``neural.gender``               the acoustic model's formant shift: +1 is the top of the bank's
                                key-shift range (brighter, smaller), -1 the bottom (darker)
``neural.tone``                 semitones the timbre is shifted while the pitch stays as written
                                (12: the acoustic model sings an octave above the rendered pitch)
``neural.breath_db``            dB added to the predicted breathiness
``neural.vowel_breath_db``      dB added to the breathiness of vowel frames only
``neural.voicing_db``           dB added to the predicted voicing
``neural.tension``              offset on the predicted tension (banks that have one)
``neural.velocity``             articulation speed of consonants (1 = as trained)
``neural.level_db``             fixed gain of the voice
``neural.detune``               cents added to the sung pitch
``neural.expr``                 how much of the bank's learned pitch expression is used (0.5..1;
                                0 = Synth2's own pitch curve)
``neural.slow``                 notes shorter than this many seconds are sung in slow motion
                                (0 = off)
``neural.voicing_floor``        dB a vowel's voicing may fall under the phrase's median (0 = off)
``neural.dial``                 the articulation dial: 0 smooth, 1 balanced, 2 crisp
``neural.cons_cap[.<class>]``   s: the longest a consonant (of a class) sharing a note with a
                                vowel may be (0 = no cap)
``neural.cons_cap_share``       the caps bind only where a note's consonants take more than this
                                share of the note (0 = everywhere)
``neural.cons_cap_keep``        no cap takes a consonant under this share of its predicted length
``neural.onset_keep``           1: consonants after a rest keep their predicted length
``neural.cons_floor[.<class>]`` dB under the vowels a voiced consonant's voicing may fall (0 = off)
``neural.burst[.<class>]``      s a stop or affricate is sung in slow motion before it is cut out
                                around its release (0 = off; ``neural.burst=1`` = the crisp values)
``neural.cons_gain.<class>``    dB on a consonant class after the vocoder; ``neural.cons_gain.hf``
                                (Hz) limits it to the band above
``neural.breaths``              breaths between lines: 0 = none, 1 = as designed, other values
                                scale their level
``neural.adlib``                1: the track is sung hit by hit (see ``adlib.py``);
                                ``neural.adlib.<option>`` sets an option of that mode
``neural.mode.<name>``          weight of a voice mode of the bank (several = a blend)
``neural.fx.<preset>``          amount (0..1) of a character effect (several = a chain)
==============================  ===============================================================

Consonant classes: ``stop`` (p t k), ``vstop`` (b d g), ``aff`` (ch ts), ``fric`` (s sh f th
h), ``vfric`` (z zh v dh), ``nasal``, ``glide``.

The dial (:data:`DIALS`) is one named choice of the consonant settings, from ``smooth`` (short
consonants, a continuous line) to ``crisp`` (consonants at their natural length where a note
has room, stops cut around their release); the individual parameters override parts of it.
``SYNTH2_NEURAL_DIAL`` sets the dial of tracks that do not name one.

The voice parameters of the formant engine keep a meaning: ``formant_scale`` moves the
formant shift, ``breathiness`` the breathiness offset, and ``robotic`` lowers the pitch
expression (through the voice, see ``pitch.default_expr``).
"""
from __future__ import annotations

import math
import os
from dataclasses import asdict, dataclass, replace

from .align import CONS_CLASSES, VOICED_CONS

GENDER_LIMIT = 1.2  # the formant shift is kept within this many key-shift ranges
BREATH_DB = 12.0  # dB of breathiness per unit of the voice's ``breathiness``
KEY_SHIFT = 5.0  # semitones of formant shift one unit of gender stands for
FX_ORDER = ("double", "robot", "alien", "crush", "radio")  # chain order; other presets follow by name
PREFIX = "neural."

DIAL_ENV = "SYNTH2_NEURAL_DIAL"
DIAL_NAMES = ("smooth", "balanced", "crisp")  # ``neural.dial`` 0, 1, 2
DEFAULT_DIAL = "crisp"
# Consonant settings per dial position: caps (s per class), the share of a note above which the
# caps bind (None: everywhere), voicing floors of voiced consonants (dB under the vowels) and
# the stops sung in slow motion (s) and cut out around their release.
DIALS: dict[str, dict] = {
    "smooth": {"cons_cap": {"stop": 0.035, "vstop": 0.035, "aff": 0.06, "fric": 0.05, "vfric": 0.05, "nasal": 0.06,
                            "glide": 0.05},
               "cons_cap_share": None,
               "cons_floor": {"vfric": 8.0, "nasal": 8.0, "glide": 8.0, "vstop": 11.0},
               "burst": {}},
    "balanced": {"cons_cap": {"stop": 0.045, "vstop": 0.04, "aff": 0.07, "fric": 0.065, "vfric": 0.05, "nasal": 0.06,
                              "glide": 0.05},
                 "cons_cap_share": 0.35,
                 "cons_floor": {"vfric": 8.0, "nasal": 8.0, "glide": 8.0, "vstop": 14.0},
                 "burst": {}},
    "crisp": {"cons_cap": {"stop": 0.05, "vstop": 0.045, "aff": 0.08, "fric": 0.08, "vfric": 0.06, "nasal": 0.07,
                           "glide": 0.06},
              "cons_cap_share": 0.4,
              "cons_floor": {"vfric": 11.0, "nasal": 8.0, "glide": 8.0, "vstop": 16.0},
              "burst": {"stop": 0.09, "aff": 0.12}},
}
BURST = DIALS["crisp"]["burst"]  # what ``neural.burst=1`` switches on
BURST_CLASSES = ("stop", "aff")
CONS_CAP_KEEP = 0.5
VOWEL_BREATH_DB = -3.0  # dB on the breathiness of vowels (the bank's vowels are airy)
DETUNE = 7.0  # cents: the bank's pitch predictor sings a little flat
# Breaths between lines: a gap of at least min_gap s gets one, the whole gap up to fill s, the
# last length s of a longer one; a breath left shorter than min_len s is a rest; breathiness is
# the dB the breath's frames peak at.
BREATH = {"min_gap": 0.12, "fill": 0.45, "length": 0.30, "min_len": 0.10, "breathiness": -44.0, "first": True}


def dial_name(value) -> str | None:
    """The dial position a name or a number (0, 1, 2) stands for; None when it is neither."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v in DIAL_NAMES:
            return v
    x = _num(value)
    if x is None:
        return None
    return DIAL_NAMES[int(min(len(DIAL_NAMES) - 1, max(0, round(x))))]


def default_dial() -> str:
    """``SYNTH2_NEURAL_DIAL`` when it names a dial position, else :data:`DEFAULT_DIAL`."""
    return dial_name(os.environ.get(DIAL_ENV) or "") or DEFAULT_DIAL


@dataclass(frozen=True)
class VoicePreset:
    gender: float = 0.0
    tone: float = 0.0
    breath_db: float = 0.0
    voicing_db: float = 0.0
    tension: float = 0.0
    velocity: float = 1.0
    level_db: float = 0.0
    fx: tuple[tuple[str, float], ...] = ()
    bank: str | None = None
    modes: tuple[tuple[str, float], ...] = ()
    expr: float | None = None  # None: by style
    slow: float | None = None  # None: the engine's default
    voicing_floor: float | None = None  # None: the engine's default
    vowel_breath_db: float = VOWEL_BREATH_DB
    detune: float | None = None  # None: DETUNE where the bank's pitch predictor sings, else 0
    dial: str = DEFAULT_DIAL
    # the dial's consonant settings with the track's overrides applied (pairs sorted by class)
    cons_cap: tuple[tuple[str, float], ...] = tuple(sorted(DIALS[DEFAULT_DIAL]["cons_cap"].items()))
    cons_cap_share: float | None = DIALS[DEFAULT_DIAL]["cons_cap_share"]
    cons_cap_keep: float = CONS_CAP_KEEP
    onset_keep: bool = True
    cons_floor: tuple[tuple[str, float], ...] = tuple(sorted(DIALS[DEFAULT_DIAL]["cons_floor"].items()))
    burst: tuple[tuple[str, float], ...] = tuple(sorted(DIALS[DEFAULT_DIAL]["burst"].items()))
    cons_gain: tuple[tuple[str, float], ...] = ()
    cons_gain_hf: float | None = None
    breaths: float = 1.0  # 0: no breaths between lines; else a factor on their level
    adlib: tuple[tuple[str, float], ...] | None = None  # None: sung as lines; else the ad-lib options set

    def fx_spec(self) -> list[str]:
        """The effect chain as ``fx.parse_fx`` reads it (empty: no effect)."""
        return [f"{name}:{amount:g}" for name, amount in self.fx]

    def mode_weights(self) -> dict[str, float]:
        return dict(self.modes)

    def breath(self) -> dict | None:
        """The breath settings of ``align.phrase_groups`` (None: no breaths)."""
        if self.breaths <= 0:
            return None
        return dict(BREATH, breathiness=BREATH["breathiness"] + 20.0 * math.log10(self.breaths))

    def summary(self) -> dict:
        d = asdict(self)
        d["fx"] = [{"preset": n, "amount": a} for n, a in self.fx]
        d["modes"] = dict(self.modes)
        for k in ("cons_cap", "cons_floor", "burst", "cons_gain"):
            d[k] = dict(getattr(self, k))
        d["adlib"] = None if self.adlib is None else dict(self.adlib)
        return d


# Formant shift, timbre shift and level per built-in voice. The low voices sing with the
# acoustic model an octave above the rendered pitch (``tone=12``): a soprano bank has no voice
# below about A3, and sung this way it keeps its words down to E2. The levels bring a line in
# the middle of each voice's range to the level the formant engine sings it at (active RMS,
# default voicebank), so a project's faders mean the same with either engine.
PRESETS: dict[str, VoicePreset] = {
    "nova": VoicePreset(level_db=1.5),
    "aurora": VoicePreset(level_db=2.0),
    "seraph": VoicePreset(gender=0.1, level_db=0.5),
    "pixel": VoicePreset(gender=0.35),
    "atlas": VoicePreset(gender=-0.3, tone=12.0, level_db=4.5),
    "ember": VoicePreset(gender=-0.7, tone=12.0, level_db=6.0),
    "basso": VoicePreset(gender=-0.6, tone=12.0, level_db=8.5),
}

_FLOATS = ("gender", "tone", "breath_db", "voicing_db", "tension", "velocity", "level_db", "expr", "slow",
           "voicing_floor", "vowel_breath_db", "detune", "cons_cap_keep")
_CLASS_TABLES = {"cons_cap": CONS_CLASSES, "cons_floor": VOICED_CONS, "burst": BURST_CLASSES, "cons_gain": CONS_CLASSES}


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fx_rank(name: str):
    return (FX_ORDER.index(name) if name in FX_ORDER else len(FX_ORDER), name)


def _pairs(table: dict, signed: bool = False) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((k, float(v)) for k, v in table.items() if v > 0 or (signed and v)))


def preset_for(voice_id: str, voice_params: dict | None = None) -> VoicePreset:
    """The preset of a voice id with the track's ``voice_params`` applied. Unknown voices get
    the neutral preset; values that are not numbers are ignored."""
    base = PRESETS.get(voice_id, VoicePreset())
    params = voice_params or {}
    changes: dict = {}
    gender, breath = base.gender, base.breath_db
    try:
        from ..voices import get_voice

        voice = get_voice(voice_id)
    except Exception:
        voice = None
    scale = _num(params.get("formant_scale"))
    if voice is not None and scale and scale > 0 and voice.formant_scale > 0:
        gender += 12.0 * math.log2(scale / voice.formant_scale) / KEY_SHIFT
    b = _num(params.get("breathiness"))
    if voice is not None and b is not None:
        breath += BREATH_DB * (b - voice.breathiness)
    dial = dial_name(params.get(PREFIX + "dial")) or default_dial()
    tables = {k: dict(DIALS[dial][k]) for k in ("cons_cap", "cons_floor", "burst")}
    tables["cons_gain"] = dict(base.cons_gain)
    changes["dial"] = dial
    changes["cons_cap_share"] = DIALS[dial]["cons_cap_share"]
    fx = dict(base.fx)
    modes = dict(base.modes)
    adlib_on, adlib = base.adlib is not None, dict(base.adlib or ())
    floors_set = False  # the track names one floor for every class
    each: list[tuple[str, str, float]] = []  # per-class values, applied after the ones for every class
    for key, raw in params.items():
        if not isinstance(key, str) or not key.startswith(PREFIX):
            continue
        v = _num(raw)
        if v is None:
            continue
        name = key[len(PREFIX):]
        head, _, tail = name.partition(".")
        if name == "gender":
            gender = v + (gender - base.gender)
        elif name == "breath_db":
            breath = v + (breath - base.breath_db)
        elif name in _FLOATS:
            changes[name] = v
        elif name == "cons_cap_share":
            changes[name] = v if v > 0 else None
        elif name == "onset_keep":
            changes[name] = v > 0
        elif name == "breaths":
            changes[name] = max(0.0, v)
        elif name == "cons_gain.hf":
            changes["cons_gain_hf"] = v if v > 0 else None
        elif name in ("cons_cap", "cons_floor"):
            tables[name] = {c: v for c in _CLASS_TABLES[name]} if v > 0 else {}
            floors_set |= name == "cons_floor"
        elif name == "burst":
            tables[name] = (dict(BURST) if v == 1 else {c: v for c in BURST_CLASSES}) if v > 0 else {}
        elif head in _CLASS_TABLES and tail in _CLASS_TABLES[head]:
            each.append((head, tail, v))
        elif name == "adlib":
            adlib_on = v > 0
        elif head == "adlib" and tail:
            adlib[tail] = v
        elif head == "fx" and tail:
            fx[tail.lower()] = min(1.0, max(0.0, v))
        elif head == "mode" and tail:
            modes[tail.lower()] = max(0.0, v)
    if adlib_on:
        # detached hits are sung with the bank's own breathiness and consonant voicing unless the track says otherwise
        changes.setdefault("vowel_breath_db", 0.0)
        if not floors_set:
            tables["cons_floor"] = {}
    for head, tail, v in each:
        tables[head][tail] = v
    changes["gender"] = min(GENDER_LIMIT, max(-GENDER_LIMIT, gender))
    changes["breath_db"] = breath
    chain = [(n, a) for n, a in fx.items() if a > 0 and n != "none"]
    changes["fx"] = tuple(sorted(chain, key=lambda x: _fx_rank(x[0])))
    changes["modes"] = tuple(sorted((n, w) for n, w in modes.items() if w > 0))
    for k, t in tables.items():
        changes[k] = _pairs(t, signed=k == "cons_gain")
    changes["adlib"] = tuple(sorted((k, float(x)) for k, x in adlib.items())) if adlib_on else None
    if "velocity" in changes:
        changes["velocity"] = min(2.0, max(0.5, changes["velocity"]))
    if "cons_cap_keep" in changes:
        changes["cons_cap_keep"] = min(1.0, max(0.0, changes["cons_cap_keep"]))
    return replace(base, **changes)


# What a role in the mix (``Track.role``, see ``engine/vocal_chain.py``) changes in how a track
# is sung, unless the track sets the parameter itself: backing parts draw no breaths of their
# own, and an ad-lib track is sung hit by hit.
ROLE_PARAMS = {"double": {"neural.breaths": 0.0}, "harmony": {"neural.breaths": 0.0}, "answer": {"neural.breaths": 0.0},
               "adlib": {"neural.adlib": 1.0}}


def track_preset(track) -> VoicePreset:
    """The preset a track is sung with: its voice's, its role's defaults, its own parameters."""
    from ..engine.vocal_chain import parse_role

    role = parse_role(getattr(track, "role", ""))[0]
    return preset_for(track.voice, {**ROLE_PARAMS.get(role, {}), **(track.voice_params or {})})
