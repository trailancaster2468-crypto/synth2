"""Voice library.

A voice is a parametric singer model for the built-in engine: vocal-tract size
(formant scaling), glottal source character, noise levels, natural pitch range
and default vibrato.  Voices can be tweaked per track via ``track.voice_params``
and new voices can be registered at runtime or loaded from JSON files.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, replace, fields
from pathlib import Path


@dataclass(frozen=True)
class Voice:
    id: str
    name: str
    description: str
    gender: str = "female"
    formant_scale: float = 1.17  # vocal tract length factor relative to adult male
    f4: float = 4100.0
    f5: float = 5000.0
    bandwidth_scale: float = 1.0
    range_low: int = 55  # comfortable MIDI range
    range_high: int = 84
    source_corner: float = 450.0  # Hz, glottal source spectral roll-off corner
    source_tilt: float = 11.0  # dB/octave roll-off of the glottal source above the corner
    aspiration_db: float = -30.0  # breath-noise level relative to voiced
    jitter: float = 0.004  # cycle-to-cycle f0 perturbation (fraction)
    shimmer_db: float = 0.4
    tension: float = 0.0
    breathiness: float = 0.0
    singer_formant_db: float = 4.0  # boost around ~3 kHz
    brightness_db: float = 0.0  # high-shelf above ~5 kHz
    vibrato_rate: float = 5.6
    vibrato_depth: float = 0.32  # semitones
    robotic: float = 0.0  # 0 natural .. 1 classic machine-like vocaloid character
    loudness_db: float = 0.0
    default_style: str = "pop"
    tags: tuple[str, ...] = ()

    def with_params(self, params: dict) -> "Voice":
        valid = {f.name for f in fields(self)}
        clean = {}
        for k, v in params.items():
            if k in valid and k not in ("id", "name", "description", "tags"):
                cur = getattr(self, k)
                clean[k] = type(cur)(v) if not isinstance(cur, tuple) else v
        return replace(self, **clean) if clean else self

    def to_dict(self):
        d = asdict(self)
        d["tags"] = list(self.tags)
        return d


VOICES: dict[str, Voice] = {}


def register(voice: Voice) -> Voice:
    VOICES[voice.id] = voice
    return voice


register(Voice(
    "aurora", "Aurora", "Bright, clear female pop lead with a modern polished tone.",
    gender="female", formant_scale=1.17, range_low=55, range_high=86, source_tilt=10.0,
    tension=0.1, singer_formant_db=5.0, brightness_db=1.5, vibrato_rate=5.7, vibrato_depth=0.3,
    default_style="pop", tags=("female", "pop", "bright", "lead"),
))
register(Voice(
    "seraph", "Seraph", "Soft, airy female voice for ballads, lullabies and intimate tracks.",
    gender="female", formant_scale=1.15, range_low=53, range_high=84, source_corner=380.0,
    source_tilt=13.5, aspiration_db=-24.0, tension=-0.35, breathiness=0.35, singer_formant_db=1.5,
    brightness_db=-1.0, vibrato_rate=5.3, vibrato_depth=0.26, default_style="ballad",
    tags=("female", "soft", "breathy", "ballad"),
))
register(Voice(
    "atlas", "Atlas", "Powerful male tenor for pop and rock, strong mid-range presence.",
    gender="male", formant_scale=1.0, f4=3500.0, f5=4500.0, range_low=45, range_high=72,
    source_corner=300.0, source_tilt=10.0, tension=0.2, singer_formant_db=6.0, vibrato_rate=5.5,
    vibrato_depth=0.3, default_style="rock", tags=("male", "tenor", "rock", "pop"),
))
register(Voice(
    "ember", "Ember", "Warm, dark male baritone suited to soul, jazz and low ballads.",
    gender="male", formant_scale=0.95, f4=3300.0, f5=4300.0, range_low=40, range_high=67,
    source_corner=260.0, source_tilt=12.0, aspiration_db=-27.0, tension=-0.1, breathiness=0.1,
    singer_formant_db=4.0, brightness_db=-1.5, vibrato_rate=5.2, vibrato_depth=0.35,
    default_style="jazz", tags=("male", "baritone", "warm", "jazz"),
))
register(Voice(
    "nova", "Nova", "Crisp synthetic idol voice with the classic machine-tuned vocaloid character.",
    gender="female", formant_scale=1.22, f4=4300.0, f5=5300.0, range_low=57, range_high=89,
    source_tilt=9.0, aspiration_db=-36.0, jitter=0.0005, shimmer_db=0.05, tension=0.25,
    singer_formant_db=5.0, brightness_db=3.0, vibrato_rate=6.2, vibrato_depth=0.22, robotic=0.85,
    default_style="robotic", tags=("female", "synthetic", "vocaloid", "edm", "jpop"),
))
register(Voice(
    "pixel", "Pixel", "Light, youthful high voice with a small vocal tract; cute and agile.",
    gender="neutral", formant_scale=1.3, f4=4600.0, f5=5600.0, range_low=60, range_high=91,
    source_corner=520.0, source_tilt=11.0, aspiration_db=-29.0, tension=0.0, breathiness=0.1,
    singer_formant_db=2.0, brightness_db=1.0, vibrato_rate=6.0, vibrato_depth=0.25, robotic=0.2,
    default_style="pop", tags=("young", "high", "cute", "jpop"),
))
register(Voice(
    "basso", "Basso", "Deep, resonant bass voice for low harmonies and choir foundations.",
    gender="male", formant_scale=0.9, f4=3100.0, f5=4100.0, range_low=36, range_high=62,
    source_corner=220.0, source_tilt=12.5, aspiration_db=-28.0, tension=0.0,
    singer_formant_db=5.0, brightness_db=-2.0, vibrato_rate=5.0, vibrato_depth=0.3,
    default_style="classical", tags=("male", "bass", "choir", "low"),
))


def get_voice(voice_id: str) -> Voice:
    if voice_id in VOICES:
        return VOICES[voice_id]
    low = voice_id.lower()
    for v in VOICES.values():
        if v.id == low or v.name.lower() == low:
            return v
    raise KeyError(f"unknown voice {voice_id!r}; available: {', '.join(VOICES)}")


def list_voices() -> list[dict]:
    return [
        {
            "id": v.id, "name": v.name, "description": v.description, "gender": v.gender,
            "range": f"{_n(v.range_low)}-{_n(v.range_high)}", "default_style": v.default_style,
            "tags": list(v.tags),
        }
        for v in VOICES.values()
    ]


def _n(m):
    from ..timing import midi_to_note

    return midi_to_note(m)


def load_voice_file(path: str | Path) -> Voice:
    """Load a custom voice from JSON (fields of :class:`Voice`; ``base`` selects a preset to extend)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    base = get_voice(data.pop("base", "aurora"))
    vid = data.pop("id")
    name = data.pop("name", vid)
    desc = data.pop("description", f"Custom voice based on {base.name}")
    tags = tuple(data.pop("tags", ()))
    v = replace(base.with_params(data), id=vid, name=name, description=desc, tags=tags)
    return register(v)
