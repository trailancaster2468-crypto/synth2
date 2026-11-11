"""General MIDI tables: program names, the percussion key map, and which program stands in
for each synth preset (and which synth preset stands in for a program)."""
from __future__ import annotations

import re

GM_PROGRAMS = (
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano", "Honky-tonk Piano", "Electric Piano 1",
    "Electric Piano 2", "Harpsichord", "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ", "Percussive Organ", "Rock Organ", "Church Organ",
    "Reed Organ", "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)", "Acoustic Guitar (steel)",
    "Electric Guitar (jazz)", "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)", "Electric Bass (pick)",
    "Fretless Bass", "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello",
    "Contrabass", "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani", "String Ensemble 1",
    "String Ensemble 2", "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
    "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1",
    "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe", "English Horn", "Bassoon",
    "Clarinet", "Piccolo", "Flute", "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)", "Lead 4 (chiff)", "Lead 5 (charang)",
    "Lead 6 (voice)", "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)",
    "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)",
    "FX 1 (rain)", "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)", "FX 6 (goblins)",
    "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter", "Applause",
    "Gunshot",
)

# short names an arrangement may use for a program ("preset": "strings")
GM_ALIASES = {
    "piano": 0, "grand piano": 0, "bright piano": 1, "honky tonk": 3, "epiano": 4, "electric piano": 4, "rhodes": 4,
    "fm piano": 5, "dx piano": 5, "harpsichord": 6, "clav": 7, "clavinet": 7, "celesta": 8, "glock": 9,
    "glockenspiel": 9, "music box": 10, "vibes": 11, "vibraphone": 11, "marimba": 12, "xylophone": 13, "bell": 14,
    "bells": 14, "tubular bells": 14, "dulcimer": 15, "organ": 16, "drawbar organ": 16, "rock organ": 18,
    "church organ": 19, "accordion": 21, "harmonica": 22, "guitar": 25, "nylon guitar": 24, "acoustic guitar": 25,
    "steel guitar": 25, "jazz guitar": 26, "clean guitar": 27, "electric guitar": 27, "muted guitar": 28,
    "overdrive guitar": 29, "overdriven guitar": 29, "distortion guitar": 30, "distorted guitar": 30, "bass": 33,
    "upright bass": 32, "acoustic bass": 32, "finger bass": 33, "electric bass": 33, "pick bass": 34,
    "picked bass": 34, "fretless bass": 35, "slap bass": 36, "synth bass": 38, "violin": 40, "viola": 41, "cello": 42,
    "contrabass": 43, "tremolo strings": 44, "pizzicato": 45, "harp": 46, "timpani": 47, "strings": 48,
    "slow strings": 49, "synth strings": 50, "choir": 52, "voice": 53, "orchestra hit": 55, "trumpet": 56,
    "trombone": 57, "tuba": 58, "muted trumpet": 59, "french horn": 60, "horn": 60, "brass": 61, "brass section": 61,
    "synth brass": 62, "soprano sax": 64, "alto sax": 65, "sax": 65, "tenor sax": 66, "baritone sax": 67, "oboe": 68,
    "english horn": 69, "bassoon": 70, "clarinet": 71, "piccolo": 72, "flute": 73, "recorder": 74, "pan flute": 75,
    "shakuhachi": 77, "whistle": 78, "ocarina": 79, "square lead": 80, "lead": 81, "saw lead": 81, "pad": 89,
    "warm pad": 89, "polysynth": 90, "choir pad": 91, "glass pad": 92, "halo pad": 94, "sweep pad": 95, "sitar": 104,
    "banjo": 105, "shamisen": 106, "koto": 107, "kalimba": 108, "bagpipe": 109, "fiddle": 110, "steel drums": 114,
    "woodblock": 115, "taiko": 116, "melodic tom": 117, "synth drum": 118,
}

# synth preset -> programs to try, best first
PRESET_PROGRAMS = {
    "bass_pick": (34, 33), "bass_slap": (36, 37, 33), "bass_sub": (38, 39), "bass_growl": (34, 33),
    "bass_chip": (38, 39), "power_saw": (30, 29), "gtr_mute": (28, 30), "gtr_clean": (27, 26), "supersaw": (81, 90),
    "brass": (62, 61), "chip_lead": (80, 81), "chip_square": (80, 81), "chip_arp": (80, 81), "saw_lead": (81, 80),
    "sync_lead": (84, 80), "pluck": (45, 46), "organ": (16, 18), "pad": (89, 88), "pad_glass": (92, 88),
    "bell": (14, 112), "glock": (9, 11), "music_box": (10, 9), "epiano": (4, 5), "acoustic": (25, 24), "piano": (0, 1),
}

# synth kit -> General MIDI / GS drum kit programs (bank 128) to try, best first
KIT_PROGRAMS = {"pop": (0,), "rock": (16, 0), "electro": (24, 25, 0), "chip": (25, 24, 0)}

_PROGRAM_FALLBACK = {4: "epiano", 5: "epiano", 9: "glock", 10: "music_box", 24: "acoustic", 25: "acoustic",
                     26: "gtr_clean", 27: "gtr_clean", 28: "gtr_mute", 29: "power_saw", 30: "power_saw",
                     31: "gtr_clean", 36: "bass_slap", 37: "bass_slap", 38: "bass_sub", 39: "bass_sub", 45: "pluck",
                     46: "pluck", 80: "chip_lead", 92: "pad_glass", 98: "pad_glass"}
_FAMILY_FALLBACK = ("piano", "bell", "organ", "acoustic", "bass_pick", "pad", "pad", "brass", "saw_lead", "saw_lead",
                    "saw_lead", "pad", "pad_glass", "pluck", "bell", "pad")


def synth_preset_for(program: int) -> str:
    """The synth preset that stands in for a General MIDI program (0..127)."""
    program = int(program) & 127
    return _PROGRAM_FALLBACK.get(program, _FAMILY_FALLBACK[program // 8])


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(name).lower()).strip()


_BY_NAME = {_norm(n): i for i, n in enumerate(GM_PROGRAMS)}
_BY_NAME.update({_norm(k): v for k, v in GM_ALIASES.items()})


def program_of(preset) -> tuple[int, int] | None:
    """(bank, program) named by ``preset`` without looking at any sound set: a number, a
    ``"bank:program"`` string, a General MIDI program name or one of the short aliases
    (``"strings"``, ``"slap bass"`` ...). None when it is none of those."""
    if isinstance(preset, bool) or preset is None:
        return None
    if isinstance(preset, (int, float)):
        return (0, int(preset)) if 0 <= int(preset) <= 127 else None
    if isinstance(preset, (tuple, list)) and len(preset) == 2:
        try:
            return int(preset[0]), int(preset[1])
        except (TypeError, ValueError):
            return None
    s = str(preset).strip()
    parts = s.split(":")
    if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
        return int(parts[0]), int(parts[1])
    if s.isdigit():
        return (0, int(s)) if int(s) <= 127 else None
    hit = _BY_NAME.get(_norm(s))
    return None if hit is None else (0, hit)


# ------------------------------------------------------------------------------------ percussion

# kit voice -> General MIDI percussion keys to try, best first
DRUM_KEYS = {
    "kick": (36, 35), "snare": (38, 40), "clap": (39,), "rim": (37,), "hat": (42,), "ohat": (46,),
    "crash": (49, 57), "ride": (51, 59), "tom1": (50, 48), "tom2": (47, 45), "tom3": (43, 41),
    "shaker": (82, 70, 69), "tamb": (54,), "cowbell": (56,), "snap": (),
}

# further voices a sampled kit can play: voice -> (keys, stem, synth stand-in voice, its parameter changes)
EXTRA_VOICES = {
    "kick2": ((35, 36), "kick", "kick", {}),
    "snare2": ((40, 38), "snare", "snare", {"tone": 215.0}),
    "phat": ((44, 42), "hat", "hat", {"decay": 0.03}),
    "crash2": ((57, 49), "crash", "crash", {"tone": 1.12}),
    "splash": ((55, 49), "crash", "crash", {"decay": 0.8, "tone": 1.3}),
    "china": ((52, 57), "crash", "crash", {"decay": 1.4, "tone": 0.85}),
    "ride2": ((59, 51), "ride", "ride", {"tone": 1.1}),
    "ridebell": ((53, 51), "ride", "ride", {"bell": 1.0}),
    "tom1b": ((48, 50), "toms", "tom1", {"pitch": 165.0}),
    "tom2b": ((45, 47), "toms", "tom2", {"pitch": 120.0}),
    "tom3b": ((41, 43), "toms", "tom3", {"pitch": 85.0}),
    "bongo_hi": ((60,), "bongos", "tom1", {"pitch": 400.0, "decay": 0.14}),
    "bongo_lo": ((61,), "bongos", "tom1", {"pitch": 300.0, "decay": 0.16}),
    "conga_mute": ((62,), "congas", "tom1", {"pitch": 250.0, "decay": 0.08}),
    "conga_hi": ((63,), "congas", "tom1", {"pitch": 250.0, "decay": 0.2}),
    "conga_lo": ((64,), "congas", "tom2", {"pitch": 185.0, "decay": 0.24}),
    "timbale_hi": ((65,), "timbales", "tom1", {"pitch": 330.0, "decay": 0.22}),
    "timbale_lo": ((66,), "timbales", "tom2", {"pitch": 230.0, "decay": 0.26}),
    "agogo_hi": ((67,), "agogo", "cowbell", {"tone": 1.7}),
    "agogo_lo": ((68,), "agogo", "cowbell", {"tone": 1.3}),
    "cabasa": ((69,), "shaker", "shaker", {"decay": 0.035}),
    "maracas": ((70,), "shaker", "shaker", {"decay": 0.03, "brightness": 0.8}),
    "whistle": ((71, 72), "whistle", "rim", {"tone": 2400.0, "decay": 0.12}),
    "guiro": ((73, 74), "guiro", "shaker", {"decay": 0.09, "brightness": 0.3}),
    "claves": ((75,), "claves", "rim", {"tone": 2500.0, "decay": 0.06}),
    "woodblock_hi": ((76,), "woodblock", "rim", {"tone": 1900.0, "decay": 0.045}),
    "woodblock_lo": ((77,), "woodblock", "rim", {"tone": 1300.0, "decay": 0.055}),
    "cuica": ((79, 78), "cuica", "tom1", {"pitch": 500.0, "decay": 0.12}),
    "triangle_mute": ((80,), "triangle", "hat", {"decay": 0.03, "brightness": 1.0}),
    "triangle": ((81,), "triangle", "ohat", {"decay": 0.5, "brightness": 1.0}),
    "vibraslap": ((58,), "vibraslap", "tamb", {"decay": 0.5}),
}
EXTRA_ALIASES = {
    "pedal_hat": "phat", "hat_pedal": "phat", "ph": "phat", "ride_bell": "ridebell", "bell_ride": "ridebell",
    "bongo": "bongo_hi", "conga": "conga_hi", "timbale": "timbale_hi", "agogo": "agogo_hi", "woodblock": "woodblock_hi",
    "wood": "woodblock_hi", "clave": "claves", "maraca": "maracas", "tri": "triangle",
}

# stem of each kind of drum when a kit is split per drum
STEM_OF = {"kick": "kick", "snare": "snare", "clap": "clap", "rim": "rim", "hat": "hat", "ohat": "ohat",
           "crash": "crash", "ride": "ride", "tom1": "toms", "tom2": "toms", "tom3": "toms", "shaker": "shaker",
           "tamb": "tamb", "cowbell": "cowbell", "snap": "snap"}
STEM_OF.update({v: spec[1] for v, spec in EXTRA_VOICES.items()})
STEM_ORDER = ("kick", "snare", "clap", "rim", "snap", "toms", "hat", "ohat", "shaker", "tamb", "cowbell", "crash",
              "ride", "bongos", "congas", "timbales", "agogo", "claves", "woodblock", "triangle", "guiro", "cuica",
              "whistle", "vibraslap", "perc")

_KEY_VOICE = {}
for _v, _keys in DRUM_KEYS.items():
    for _k in _keys:
        _KEY_VOICE.setdefault(_k, _v)
for _v, _spec in EXTRA_VOICES.items():
    _KEY_VOICE[_spec[0][0]] = _v
_KEY_VOICE.update({72: "whistle", 74: "guiro", 78: "cuica"})

GM_DRUM_NAMES = {
    35: "Acoustic Bass Drum", 36: "Bass Drum 1", 37: "Side Stick", 38: "Acoustic Snare", 39: "Hand Clap",
    40: "Electric Snare", 41: "Low Floor Tom", 42: "Closed Hi-Hat", 43: "High Floor Tom", 44: "Pedal Hi-Hat",
    45: "Low Tom", 46: "Open Hi-Hat", 47: "Low-Mid Tom", 48: "Hi-Mid Tom", 49: "Crash Cymbal 1", 50: "High Tom",
    51: "Ride Cymbal 1", 52: "Chinese Cymbal", 53: "Ride Bell", 54: "Tambourine", 55: "Splash Cymbal", 56: "Cowbell",
    57: "Crash Cymbal 2", 58: "Vibraslap", 59: "Ride Cymbal 2", 60: "Hi Bongo", 61: "Low Bongo", 62: "Mute Hi Conga",
    63: "Open Hi Conga", 64: "Low Conga", 65: "High Timbale", 66: "Low Timbale", 67: "High Agogo", 68: "Low Agogo",
    69: "Cabasa", 70: "Maracas", 71: "Short Whistle", 72: "Long Whistle", 73: "Short Guiro", 74: "Long Guiro",
    75: "Claves", 76: "Hi Wood Block", 77: "Low Wood Block", 78: "Mute Cuica", 79: "Open Cuica", 80: "Mute Triangle",
    81: "Open Triangle", 82: "Shaker",
}


def key_voice(voice: str) -> int | None:
    """The kit key of a numbered voice (``"n36"``), else None."""
    m = re.fullmatch(r"n(\d{1,3})", voice)
    return int(m.group(1)) if m and int(m.group(1)) <= 127 else None


def drum_keys(voice: str) -> tuple[int, ...]:
    """Kit keys to try for a voice, best first."""
    k = key_voice(voice)
    if k is not None:
        return (k,)
    if voice in DRUM_KEYS:
        return DRUM_KEYS[voice]
    return EXTRA_VOICES[voice][0] if voice in EXTRA_VOICES else ()


def voice_of_key(key: int) -> str:
    """The voice name that plays a kit key (``"kick"`` for 36; ``"n27"`` for keys without a name)."""
    return _KEY_VOICE.get(int(key), f"n{int(key)}")


def stem_of(voice: str) -> str:
    """The stem a voice goes to when a kit is split per kind of drum."""
    k = key_voice(voice)
    if k is not None:
        named = _KEY_VOICE.get(k)
        return STEM_OF[named] if named else "perc"
    return STEM_OF.get(voice, "perc")


def synth_voice(voice: str) -> tuple[str, dict]:
    """(synthesised voice, parameter changes) that stands in for any voice name."""
    k = key_voice(voice)
    if k is not None:
        voice = _KEY_VOICE.get(k, "rim")
    if voice in EXTRA_VOICES:
        _keys, _stem, base, changes = EXTRA_VOICES[voice]
        return base, dict(changes)
    return voice, {}
