"""How Synth2's voices map onto a voicebank.

A voicebank has one singer (and perhaps a few voice modes); Synth2 has a cast of voices. A
:class:`VoicePreset` turns a voice id into settings of the neural engine: a formant shift, a
timbre shift, variance offsets, a level, optional character effects and optionally another
bank. ``Track.voice_params`` overrides any of it with numeric ``neural.*`` parameters:

==========================  ===================================================================
``neural.gender``           the acoustic model's formant shift: +1 is the top of the bank's
                            key-shift range (brighter, smaller), -1 the bottom (darker, larger)
``neural.tone``             semitones the timbre is shifted while the pitch stays as written
``neural.breath_db``        dB added to the predicted breathiness
``neural.voicing_db``       dB added to the predicted voicing
``neural.tension``          offset on the predicted tension (banks that have one)
``neural.velocity``         articulation speed of consonants (1 = as trained)
``neural.level_db``         fixed gain of the voice
``neural.expr``             how much of the bank's learned pitch expression is used (0.5..1;
                            0 = Synth2's own pitch curve)
``neural.slow``             notes shorter than this many seconds are sung in slow motion
                            (0 = off)
``neural.voicing_floor``    dB a vowel's voicing may fall under the phrase's median (0 = off)
``neural.mode.<name>``      weight of a voice mode of the bank (several = a blend)
``neural.fx.<preset>``      amount (0..1) of a character effect (several = a chain)
==========================  ===================================================================

The voice parameters of the formant engine keep a meaning: ``formant_scale`` moves the
formant shift, ``breathiness`` the breathiness offset, and ``robotic`` lowers the pitch
expression (through the voice, see ``pitch.default_expr``).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace

GENDER_LIMIT = 1.2  # the formant shift is kept within this many key-shift ranges
BREATH_DB = 12.0  # dB of breathiness per unit of the voice's ``breathiness``
KEY_SHIFT = 5.0  # semitones of formant shift one unit of gender stands for
FX_ORDER = ("double", "robot", "alien", "crush", "radio")  # chain order; other presets follow by name
PREFIX = "neural."


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

    def fx_spec(self) -> list[str]:
        """The effect chain as ``fx.parse_fx`` reads it (empty: no effect)."""
        return [f"{name}:{amount:g}" for name, amount in self.fx]

    def mode_weights(self) -> dict[str, float]:
        return dict(self.modes)

    def summary(self) -> dict:
        d = asdict(self)
        d["fx"] = [{"preset": n, "amount": a} for n, a in self.fx]
        d["modes"] = dict(self.modes)
        return d


# provisional values (formant shift, timbre shift) per built-in voice
PRESETS: dict[str, VoicePreset] = {
    "nova": VoicePreset(),
    "aurora": VoicePreset(),
    "seraph": VoicePreset(gender=0.1),
    "pixel": VoicePreset(gender=0.35),
    "atlas": VoicePreset(gender=-0.7),
    "ember": VoicePreset(gender=-0.85),
    "basso": VoicePreset(gender=-1.0, tone=12.0),
}

_FLOATS = ("gender", "tone", "breath_db", "voicing_db", "tension", "velocity", "level_db", "expr", "slow",
           "voicing_floor")


def _num(v) -> float | None:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _fx_rank(name: str):
    return (FX_ORDER.index(name) if name in FX_ORDER else len(FX_ORDER), name)


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
    fx = dict(base.fx)
    modes = dict(base.modes)
    for key, raw in params.items():
        if not isinstance(key, str) or not key.startswith(PREFIX):
            continue
        v = _num(raw)
        if v is None:
            continue
        name = key[len(PREFIX):]
        if name == "gender":
            gender = v + (gender - base.gender)
        elif name == "breath_db":
            breath = v + (breath - base.breath_db)
        elif name in _FLOATS:
            changes[name] = v
        elif name.startswith("fx."):
            fx[name[3:].lower()] = min(1.0, max(0.0, v))
        elif name.startswith("mode."):
            modes[name[5:].lower()] = max(0.0, v)
    changes["gender"] = min(GENDER_LIMIT, max(-GENDER_LIMIT, gender))
    changes["breath_db"] = breath
    chain = [(n, a) for n, a in fx.items() if a > 0 and n != "none"]
    changes["fx"] = tuple(sorted(chain, key=lambda x: _fx_rank(x[0])))
    changes["modes"] = tuple(sorted((n, w) for n, w in modes.items() if w > 0))
    if "velocity" in changes:
        changes["velocity"] = min(2.0, max(0.5, changes["velocity"]))
    return replace(base, **changes)
