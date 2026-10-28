"""Phoneme inventory with acoustic targets for the formant/harmonic engine.

Formant values are adult-male reference targets (Hz), based on classic
measurements (Peterson & Barney 1952; Hillenbrand et al. 1995) and adjusted
for sung vowels.  Voices scale them (see :mod:`synth2.voices`).

Every phoneme carries enough information for the segment planner to build
frame-level parameter tracks: formant targets, voicing, frication spectrum,
burst/aspiration timing, nasal anti-resonance and nominal duration.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Phoneme:
    name: str
    kind: str  # vowel diphthong nasal liquid glide fricative stop affricate aspirate flap silence breath
    formants: tuple[float, float, float] = (500.0, 1500.0, 2500.0)
    voiced: bool = True
    amp_db: float = 0.0  # voiced level relative to a vowel
    fric: tuple[float, float, float] | None = None  # (center Hz, bandwidth Hz, gain dB) of frication noise
    dur: float = 0.08  # nominal duration in seconds (consonants)
    nasal_zero: float | None = None  # anti-resonance (Hz) for nasals
    parts: tuple[str, ...] = ()  # component phonemes for diphthongs / affricates
    place: str | None = None  # labial / alveolar / velar / palatal (for stops)
    aspiration: float = 0.0  # aspiration duration (s) after release (voiceless stops)
    coarticulate: bool = False  # take formants from the neighbouring vowel (h, breath)
    description: str = ""

    @property
    def is_vowel(self) -> bool:
        return self.kind in ("vowel", "diphthong")

    @property
    def is_sonorant(self) -> bool:
        return self.kind in ("vowel", "diphthong", "nasal", "liquid", "glide", "flap")


_P = Phoneme
PHONEMES: dict[str, Phoneme] = {}


def _add(*ps: Phoneme):
    for p in ps:
        PHONEMES[p.name] = p


# --- English (ARPABET-derived, lower case) vowels ---------------------------
_add(
    _P("iy", "vowel", (280, 2250, 3000), description="beet"),
    _P("ih", "vowel", (400, 1950, 2550), description="bit"),
    _P("eh", "vowel", (540, 1800, 2500), description="bet"),
    _P("ae", "vowel", (680, 1700, 2420), description="bat"),
    _P("aa", "vowel", (740, 1100, 2450), description="father"),
    _P("ao", "vowel", (590, 880, 2420), description="thought"),
    _P("uh", "vowel", (450, 1050, 2250), description="book"),
    _P("uw", "vowel", (310, 870, 2250), description="boot"),
    _P("ah", "vowel", (650, 1200, 2400), description="but"),
    _P("ax", "vowel", (520, 1400, 2450), amp_db=-2, description="schwa (about)"),
    _P("er", "vowel", (480, 1330, 1680), description="bird"),
    _P("ey", "diphthong", parts=("eh", "iy"), description="bait"),
    _P("ay", "diphthong", parts=("aa", "iy"), description="bite"),
    _P("ow", "diphthong", parts=("ao", "uw"), description="boat"),
    _P("aw", "diphthong", parts=("aa", "uw"), description="bout"),
    _P("oy", "diphthong", parts=("ao", "iy"), description="boy"),
)

# --- Japanese vowels -------------------------------------------------------------
_add(
    _P("a", "vowel", (720, 1220, 2550), description="JA a"),
    _P("i", "vowel", (290, 2250, 3000), description="JA i"),
    _P("u", "vowel", (330, 1350, 2300), description="JA u (unrounded)"),
    _P("e", "vowel", (470, 1900, 2550), description="JA e"),
    _P("o", "vowel", (470, 860, 2450), description="JA o"),
    _P("N", "nasal", (260, 1500, 2550), amp_db=-6, dur=0.12, nasal_zero=1600, description="JA moraic nasal"),
)

# --- nasals, liquids, glides ------------------------------------------------------
_add(
    _P("m", "nasal", (260, 1050, 2250), amp_db=-7, dur=0.07, nasal_zero=900, place="labial"),
    _P("n", "nasal", (260, 1600, 2600), amp_db=-7, dur=0.065, nasal_zero=1700, place="alveolar"),
    _P("ng", "nasal", (260, 2050, 2700), amp_db=-8, dur=0.075, nasal_zero=3000, place="velar"),
    _P("l", "liquid", (360, 1050, 2800), amp_db=-4, dur=0.06),
    _P("r", "liquid", (360, 1250, 1650), amp_db=-3, dur=0.065),
    _P("dx", "flap", (330, 1600, 2600), amp_db=-9, dur=0.028, place="alveolar", description="flap / JA r"),
    _P("w", "glide", (300, 680, 2200), amp_db=-3, dur=0.055),
    _P("y", "glide", (260, 2250, 3000), amp_db=-3, dur=0.05),
)

# --- fricatives ------------------------------------------------------------------
_add(
    _P("s", "fricative", (400, 1700, 2600), voiced=False, fric=(6800, 3500, 0), dur=0.10),
    _P("z", "fricative", (300, 1700, 2600), voiced=True, amp_db=-14, fric=(6500, 3500, -8), dur=0.08),
    _P("sh", "fricative", (400, 1900, 2500), voiced=False, fric=(3300, 2400, 1), dur=0.11),
    _P("zh", "fricative", (300, 1900, 2500), voiced=True, amp_db=-14, fric=(3100, 2400, -8), dur=0.08),
    _P("f", "fricative", (400, 1100, 2400), voiced=False, fric=(4500, 7000, -12), dur=0.09),
    _P("v", "fricative", (300, 1100, 2400), voiced=True, amp_db=-12, fric=(4500, 7000, -18), dur=0.06),
    _P("th", "fricative", (400, 1500, 2600), voiced=False, fric=(5500, 7000, -14), dur=0.09),
    _P("dh", "fricative", (300, 1500, 2600), voiced=True, amp_db=-12, fric=(5000, 7000, -20), dur=0.05),
    _P("hh", "aspirate", (500, 1500, 2500), voiced=False, dur=0.07, coarticulate=True, description="h"),
    _P("ph", "fricative", (400, 900, 2300), voiced=False, fric=(2500, 6000, -16), dur=0.08, description="JA f (bilabial)"),
    _P("c", "fricative", (300, 2200, 3000), voiced=False, fric=(4200, 3000, -6), dur=0.09, description="JA hi (palatal)"),
)

# --- stops and affricates -------------------------------------------------------------
_add(
    _P("p", "stop", (250, 900, 2200), voiced=False, fric=(1200, 4000, -12), dur=0.075, place="labial", aspiration=0.035),
    _P("b", "stop", (250, 900, 2200), voiced=True, amp_db=-24, fric=(1200, 4000, -18), dur=0.06, place="labial"),
    _P("t", "stop", (250, 1700, 2600), voiced=False, fric=(4800, 3500, -4), dur=0.07, place="alveolar", aspiration=0.04),
    _P("d", "stop", (250, 1700, 2600), voiced=True, amp_db=-24, fric=(4500, 3500, -10), dur=0.055, place="alveolar"),
    _P("k", "stop", (250, 1900, 2500), voiced=False, fric=(2300, 1600, -5), dur=0.075, place="velar", aspiration=0.045),
    _P("g", "stop", (250, 1900, 2500), voiced=True, amp_db=-24, fric=(2300, 1600, -11), dur=0.06, place="velar"),
    _P("ch", "affricate", parts=("t", "sh"), voiced=False, dur=0.11),
    _P("jh", "affricate", parts=("d", "zh"), voiced=True, dur=0.09),
    _P("ts", "affricate", parts=("t", "s"), voiced=False, dur=0.09),
    _P("dz", "affricate", parts=("d", "z"), voiced=True, dur=0.08),
    _P("q", "stop", (300, 1500, 2500), voiced=False, fric=None, dur=0.06, place="glottal", description="glottal stop / JA small tsu"),
)

# --- silence / breath ----------------------------------------------------------------
_add(
    _P("sil", "silence", voiced=False, dur=0.0),
    _P("br", "breath", (550, 1500, 2500), voiced=False, dur=0.25, coarticulate=True, description="breath intake"),
)

VOWELS = frozenset(n for n, p in PHONEMES.items() if p.is_vowel)

ALIASES = {
    "hh": "hh", "h": "hh", "j": "jh", "pau": "sil", "sp": "sil", "cl": "q", "R": "N",
    "aa0": "aa", "ah0": "ax", "er0": "er", "ih0": "ih", "iy0": "iy", "uw0": "uw",
}


def get(name: str) -> Phoneme:
    """Look up a phoneme by name (case sensitive for Japanese ``N``)."""
    key = name if name in PHONEMES else name.lower()
    key = ALIASES.get(key, key)
    if key not in PHONEMES:
        raise KeyError(f"unknown phoneme {name!r}")
    return PHONEMES[key]


def is_valid(name: str) -> bool:
    try:
        get(name)
        return True
    except KeyError:
        return False


def expand(names: list[str]) -> list[Phoneme]:
    """Resolve names to Phoneme objects (diphthongs/affricates kept intact)."""
    return [get(n) for n in names]


def inventory_table() -> list[dict]:
    """Return a JSON-friendly description of the inventory (for AI agents / docs)."""
    out = []
    for p in PHONEMES.values():
        out.append({"name": p.name, "kind": p.kind, "voiced": p.voiced, "parts": list(p.parts), "description": p.description})
    return out
