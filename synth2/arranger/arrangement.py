"""Arrangement format: validation and sequencing (pattern expansion into timed events).

The arrangement is a plain JSON-serialisable dict (see ``FORMAT.md``).  This module
turns it into per-instrument event lists in *beats*:

* drum hits ``(beat, voice, velocity)``
* pitched notes ``(beat, beats, pitches, velocity, ...)``
* FX one-shots, processing effects and stops (silence ranges)

Bars are 1-based and absolute (bar 1 starts at beat 0); beats inside a bar are
1-based (``beat: 3`` = the third quarter note, ``2.5`` = the "and" of 2).  Pattern
steps are 16ths by default.
"""
from __future__ import annotations

import copy
import difflib
import math
import re
from dataclasses import dataclass, field

from ..timing import note_to_midi, parse_duration
from . import gm, soundbank, theory
from .drums import ALL_VOICES, DRUM_DEFAULTS, DRUM_VOICES, canonical_voice
from .fx import FX_SOUNDS, PROCESS_EFFECTS
from .synth import PRESETS, SYNTH_DEFAULTS

FORMAT_NAME = "synth2-arrangement"
ROLES = ("drums", "bass", "keys", "lead", "pad", "fx", "perc")
PATTERN_KINDS = ("drums", "line", "chords", "arp", "notes")
KIND_ALIASES = {"bass": "line", "steps": "line", "riff": "notes", "melody": "notes", "stab": "chords",
                "strum": "chords", "drum": "drums", "beat": "drums"}
ARP_MODES = ("up", "down", "updown", "downup", "random", "order", "pingpong")
KIT_PRESETS: dict[str, dict] = {
    "pop": {},
    "rock": {"kick": {"pitch": 55, "decay": 0.38, "click": 0.8, "drive": 0.45},
             "snare": {"tone": 185, "snappy": 0.85, "noise_decay": 0.15, "drive": 0.35},
             "crash": {"decay": 2.6}},
    "electro": {"kick": {"pitch": 50, "sweep": 4.0, "decay": 0.5, "click": 0.5, "drive": 0.3},
                "snare": {"tone": 210, "snappy": 0.65, "noise_decay": 0.1, "brightness": 0.7},
                "clap": {"decay": 0.2}, "hat": {"decay": 0.035, "brightness": 0.75}},
    "chip": {"kick": {"pitch": 60, "sweep": 5.0, "sweep_time": 0.02, "decay": 0.25, "click": 0.2, "drive": 0.6},
             "snare": {"tone": 240, "snappy": 1.0, "noise_decay": 0.08, "crack": 0.2, "brightness": 0.3},
             "hat": {"decay": 0.03, "brightness": 0.3}},
}
DELAY_TIMES = {"1": 4.0, "2": 2.0, "2d": 3.0, "4": 1.0, "4d": 1.5, "4t": 2 / 3, "8": 0.5, "8d": 0.75,
               "8t": 1 / 3, "16": 0.25, "16d": 0.375, "16t": 1 / 6, "32": 0.125}

TOP_KEYS = {"format", "version", "title", "key", "swing", "humanize", "energy_response", "instruments", "patterns",
            "sections", "fills", "events", "stops", "comment", "notes", "description", "length_bars", "sample_dir"}
INSTRUMENT_KEYS = {"type", "preset", "kit", "params", "voices", "gain_db", "pan", "role", "sidechain", "width",
                   "reverb", "reverb_decay", "delay", "eq", "drive", "low", "center", "voicing", "transpose", "octave",
                   "humanize", "strum", "mute", "split", "comment", "description", "sound", "fallback"}
# per-drum options of a sampled kit (next to the synthesised voice's own parameters)
SAMPLED_VOICE_KEYS = {"note", "sample", "tune", "gain_db", "pan", "length"}
_SOUND_NUMBERS = {"tune": (-1200, 1200), "transpose": (-48, 48), "gain_db": (-40, 24), "pan": (-1, 1),
                  "attack": (0, 20), "hold": (0, 20), "decay": (0, 60), "sustain": (0, 1), "release": (0, 30),
                  "cutoff": (20, 20000), "resonance": (0, 24), "keytrack": (0, 1), "vel_to_cutoff": (0, 6),
                  "velocity_curve": (0.1, 4), "vel_sens": (0, 1), "vibrato": (0, 2), "vibrato_rate": (0.1, 12),
                  "vibrato_delay": (0, 5), "glide": (0, 1), "drift": (0, 50), "gate": (0.02, 10)}
_SOUND_BOOLS = ("normalize", "mono", "round_robin")
PATTERN_KEYS = {"kind", "type", "grid", "bars", "swing", "lanes", "steps", "velocity", "gate", "rhythm", "voicing",
                "center", "low", "octave", "transpose", "strum", "mode", "order", "octaves", "notes", "score",
                "relative", "comment", "description", "accent"}
SECTION_KEYS = {"name", "start_bar", "bars", "chords", "energy", "parts", "fills", "events", "stops", "key",
                "transpose", "like", "crash", "comment", "description", "swing"}
SEGMENT_KEYS = {"pattern", "bars", "from_bar", "to_bar", "transpose", "octave", "velocity", "offset", "comment"}
EVENT_TYPES = ("fx", "hit", "note", "chord") + tuple(PROCESS_EFFECTS)
EVENT_KEYS = {"type", "bar", "beat", "beats", "part", "parts", "except", "sound", "pitch", "velocity", "voice",
              "gain_db", "pan", "params", "comment", "voicing"} | {
    "slice", "decay", "gate", "bits", "downsample", "mix", "from_hz", "to_hz", "resonance", "steps", "grid", "curve",
    "rate", "depth", "glide", "drop", "tone", "density", "up", "pan_sweep", "mode"}
STOP_KEYS = {"bar", "beat", "beats", "parts", "except", "comment", "to_bar", "to_beat"}


class ArrangementError(ValueError):
    """The arrangement has errors; ``.messages`` lists them."""

    def __init__(self, messages: list[str]):
        self.messages = messages
        super().__init__("invalid backing arrangement:\n" + "\n".join(messages))


# --------------------------------------------------------------------------------------- data

@dataclass
class Instrument:
    name: str
    kind: str  # "drums" | "synth" | "fx"
    preset: str | None = None
    params: dict = field(default_factory=dict)  # synth: overrides of the preset; drums: per-voice overrides
    gain_db: float = 0.0
    pan: float = 0.0
    role: str = "keys"
    sidechain: bool = False
    width: float = 1.0
    reverb: float = 0.0
    reverb_decay: float = 1.2
    delay: dict | None = None
    eq: dict | None = None
    drive: float = 0.0
    low: int = 48
    center: int = 64
    voicing: str = "triad"
    transpose: int = 0
    humanize: dict = field(default_factory=dict)
    strum: float = 0.0
    mute: bool = False
    split: bool = True
    stems: str = "groups"  # drum kits: "groups" (kick / snare / hats / cymbals), "each" (one per kind of drum), "kit"
    kit: str = "pop"
    sound: list | None = None  # sampled sound layers (see FORMAT.md); None = synthesised
    voice_sound: dict = field(default_factory=dict)  # sampled kits: per-voice options (tune, gain_db, sample ...)


@dataclass
class DrumEvent:
    beat: float
    voice: str
    velocity: float
    swing_beats: float = 0.0


@dataclass
class NoteEvent:
    beat: float
    beats: float
    pitches: tuple[int, ...]
    velocity: float
    bright: float = 0.0
    strum: float = 0.0  # seconds between strummed notes (negative = up-strum)
    gate_scale: float = 1.0


@dataclass
class FxEvent:
    beat: float
    beats: float
    sound: str
    pitch: int | None
    velocity: float
    gain_db: float
    pan: float
    params: dict


@dataclass
class ProcessEvent:
    effect: str
    beat0: float
    beat1: float
    parts: set | None  # None = every part
    exclude: set
    params: dict


@dataclass
class StopEvent:
    beat0: float
    beat1: float
    parts: set | None
    exclude: set


@dataclass
class Section:
    name: str
    start_bar: int
    bars: int
    chords: list  # per bar: list of (offset_beats, Chord | None)
    energy: float
    parts: dict  # instrument -> list of segment dicts
    key: tuple
    transpose: int
    swing: float | None


@dataclass
class Compiled:
    instruments: dict
    sections: list
    drum_events: dict  # inst -> [DrumEvent]
    note_events: dict  # inst -> [NoteEvent]
    fx_events: dict  # inst -> [FxEvent]
    processes: list
    stops: list
    end_beat: float
    beats_per_bar: float
    warnings: list
    chord_track: list  # (beat, Chord|None) change points

    def stops_for(self, inst: str) -> list[tuple[float, float]]:
        return [(s.beat0, s.beat1) for s in self.stops if _applies(s.parts, s.exclude, inst)]

    def processes_for(self, inst: str) -> list[ProcessEvent]:
        return [p for p in self.processes if _applies(p.parts, p.exclude, inst)]


def _applies(parts: set | None, exclude: set, inst: str) -> bool:
    if inst in exclude:
        return False
    return parts is None or inst in parts


# ------------------------------------------------------------------------------------ context

class _Ctx:
    def __init__(self):
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def err(self, where: str, msg: str):
        self.errors.append(f"error: {where}: {msg}")

    def warn(self, where: str, msg: str):
        self.warnings.append(f"warning: {where}: {msg}")

    def unknown_keys(self, where: str, d: dict, allowed: set):
        for k in d:
            if k not in allowed:
                close = difflib.get_close_matches(str(k), sorted(allowed), n=1)
                self.warn(where, f"unknown key {k!r} ignored" + (f" (did you mean {close[0]!r}?)" if close else ""))


def _num(ctx: _Ctx, where: str, v, default=None, lo=None, hi=None):
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        try:
            v = float(v)
        except (TypeError, ValueError):
            ctx.err(where, f"expected a number, got {v!r}")
            return default
    v = float(v)
    if not math.isfinite(v):
        ctx.err(where, f"expected a finite number, got {v!r}")
        return default
    if lo is not None and v < lo:
        ctx.warn(where, f"{v:g} is below the minimum {lo:g}; clamped")
        v = lo
    if hi is not None and v > hi:
        ctx.warn(where, f"{v:g} is above the maximum {hi:g}; clamped")
        v = hi
    return v


def _pitch(ctx: _Ctx, where: str, v, default=None):
    if v is None:
        return default
    try:
        return note_to_midi(v)
    except (ValueError, TypeError):
        ctx.err(where, f"invalid note {v!r}; use a name like 'A4', 'F#3', 'Bb2' or a MIDI number")
        return default


def _suggest(name, options) -> str:
    close = difflib.get_close_matches(str(name), list(options), n=2)
    return f" (did you mean {' or '.join(repr(c) for c in close)}?)" if close else ""


# ------------------------------------------------------------------------------ instruments

def _instrument(ctx: _Ctx, name: str, d) -> Instrument | None:
    where = f"instruments.{name}"
    if not isinstance(d, dict):
        ctx.err(where, "must be an object like {\"preset\": \"bass_pick\", \"gain_db\": -3}")
        return None
    ctx.unknown_keys(where, d, INSTRUMENT_KEYS)
    typ = d.get("type")
    preset = d.get("preset")
    if preset is not None and not isinstance(preset, str):
        ctx.err(f"{where}.preset", f"must be a preset name string, got {preset!r}")
        return None
    if typ is not None and not isinstance(typ, str):
        ctx.err(f"{where}.type", "must be 'drums', 'synth' or 'fx'")
        return None
    sound_given = d.get("sound") not in (None, False, "", [], {})
    sound_preset = None
    kit_like = preset in ("drums", "kit") or bool(d.get("kit")) or name.lower() in ("drums", "kit", "drumkit")
    if sound_given and preset not in PRESETS and (typ == "synth" or (typ in (None, "") and not kit_like)):
        # 'preset' names the sampled preset; the synth preset that stands in for it is 'fallback', or the one
        # that fits its General MIDI program
        sound_preset = preset
        fb = d.get("fallback")
        if fb is not None and fb not in PRESETS:
            ctx.err(f"{where}.fallback", f"unknown preset {fb!r}{_suggest(fb, PRESETS)}; presets: {', '.join(PRESETS)}")
            fb = None
        if fb is None:
            bp = gm.program_of(preset)
            fb = gm.synth_preset_for(bp[1]) if bp is not None and bp[0] != 128 else "piano"
        preset, typ = fb, "synth"
    elif sound_given and typ in ("drums", "kit", "drum") or (sound_given and typ in (None, "") and kit_like):
        if preset not in (None, "drums", "kit"):
            sound_preset = preset
    if typ in (None, ""):
        if preset in PRESETS:
            typ = "synth"
        elif preset in ("drums", "kit") or d.get("kit") or name.lower() in ("drums", "kit", "drumkit"):
            typ = "drums"
        elif preset == "fx" or name.lower().startswith("fx") or name.lower() == "sfx":
            typ = "fx"
        elif preset is None:
            ctx.err(where, f"needs a 'preset' (one of {', '.join(PRESETS)}) or type 'drums'/'fx'")
            return None
        else:
            ctx.err(where, f"unknown preset {preset!r}{_suggest(preset, PRESETS)}; presets: {', '.join(PRESETS)}")
            return None
    if typ in ("kit", "drum"):
        typ = "drums"
    if typ not in ("drums", "synth", "fx"):
        ctx.err(where, f"type must be 'drums', 'synth' or 'fx', got {typ!r}")
        return None
    inst = Instrument(name=name, kind=typ)
    if typ == "synth":
        if preset not in PRESETS:
            ctx.err(where, f"unknown preset {preset!r}{_suggest(preset, PRESETS)}; presets: {', '.join(PRESETS)}")
            return None
        inst.preset = preset
        pr = PRESETS[preset]
        params = d.get("params") or {}
        if not isinstance(params, dict):
            ctx.err(f"{where}.params", "must be an object of synth parameters")
            params = {}
        for k in params:
            if k not in SYNTH_DEFAULTS:
                ctx.warn(f"{where}.params", f"unknown synth parameter {k!r} ignored{_suggest(k, SYNTH_DEFAULTS)}")
        inst.params = {k: v for k, v in params.items() if k in SYNTH_DEFAULTS}
        _check_synth_params(ctx, f"{where}.params", inst.params)
        if "osc" in inst.params:
            layers = inst.params["osc"]
            if not isinstance(layers, list) or not all(isinstance(x, dict) for x in layers):
                ctx.err(f"{where}.params.osc", "must be a list of oscillator layers like "
                                               "[{\"wave\": \"saw\", \"voices\": 3, \"detune\": 15}]")
                inst.params.pop("osc")
            else:
                from .dsp import WAVES

                for i, layer in enumerate(layers):
                    if layer.get("wave", "saw") not in WAVES:
                        ctx.err(f"{where}.params.osc[{i}]", f"unknown wave {layer.get('wave')!r}; waves: "
                                                            f"{', '.join(WAVES)}")
        if "filter" in inst.params and inst.params["filter"] not in ("lp", "hp", "bp", "none", None):
            ctx.err(f"{where}.params.filter", "must be 'lp', 'hp', 'bp' or 'none'")
        inst.role = pr.get("role", "keys")
        inst.sidechain = bool(pr.get("sidechain", False))
        inst.width = float(pr.get("width", 1.0))
        inst.low = note_to_midi(pr.get("low", "C3"))
        inst.center = note_to_midi(pr.get("center", pr.get("low", "C4"))) if pr.get("center") else inst.low + 16
        inst.voicing = pr.get("voicing", "triad")
        if preset in ("power_saw", "gtr_mute"):
            inst.strum = 0.006
        if preset in ("gtr_clean", "acoustic"):
            inst.strum = 0.012
    elif typ == "drums":
        kit = d.get("kit", "pop")
        if not isinstance(kit, str) or kit not in KIT_PRESETS:
            ctx.err(f"{where}.kit", f"unknown kit {kit!r}; kits: {', '.join(KIT_PRESETS)}")
            kit = "pop"
        voices = copy.deepcopy(KIT_PRESETS[kit])
        given = d.get("voices") or d.get("params") or {}
        if not isinstance(given, dict):
            ctx.err(f"{where}.voices", "must be an object {voice: {param: value}}")
            given = {}
        for v, ov in given.items():
            cv = canonical_voice(v)
            if cv is None:
                ctx.err(f"{where}.voices", f"unknown drum voice {v!r}{_suggest(v, ALL_VOICES)}; voices: "
                                           f"{', '.join(ALL_VOICES)}, or n<key> for any key of a sampled kit")
                continue
            if not isinstance(ov, dict):
                ctx.err(f"{where}.voices.{v}", "must be an object of parameters")
                continue
            synth_keys = DRUM_DEFAULTS[gm.synth_voice(cv)[0]]
            good, sampled = {}, {}
            for k, v_ in ov.items():
                if k == "sample":
                    if isinstance(v_, str) and v_:
                        sampled[k] = v_
                    else:
                        ctx.err(f"{where}.voices.{v}.sample", "must be the path of a sample file")
                    continue
                if k not in synth_keys and k != "bell" and k not in SAMPLED_VOICE_KEYS:
                    ctx.warn(f"{where}.voices.{v}", f"unknown parameter {k!r} ignored; {cv} parameters: "
                                                    f"{', '.join(synth_keys)}; sampled kits also take "
                                                    f"{', '.join(sorted(SAMPLED_VOICE_KEYS))}")
                    continue
                if isinstance(v_, bool) or not isinstance(v_, (int, float)) or not math.isfinite(v_):
                    ctx.err(f"{where}.voices.{v}.{k}", f"must be a number, got {v_!r}")
                    continue
                if k in synth_keys or k == "bell":
                    good[k] = float(v_)
                if k in SAMPLED_VOICE_KEYS:
                    sampled[k] = float(v_)
            voices.setdefault(cv, {}).update(good)
            if sampled:
                inst.voice_sound.setdefault(cv, {}).update(sampled)
        inst.params = voices
        inst.kit = kit
        inst.role = "drums"
        inst.reverb_decay = 0.9
        split = d.get("split", "each" if sound_given else True)
        if isinstance(split, str):
            if split not in ("each", "groups", "kit"):
                ctx.err(f"{where}.split", "must be true, false, 'groups' (kick / snare / hats / cymbals stems), "
                                          "'each' (one stem per kind of drum) or 'kit' (one stem)")
                split = "groups"
            inst.stems = split
        else:
            inst.stems = "groups" if split else "kit"
        inst.split = inst.stems != "kit"
    else:
        inst.role = "fx"
        inst.reverb_decay = 1.6
    inst.gain_db = _num(ctx, f"{where}.gain_db", d.get("gain_db"), 0.0, -60, 24)
    inst.pan = _num(ctx, f"{where}.pan", d.get("pan"), 0.0, -1, 1)
    if "role" in d:
        if d["role"] not in ROLES:
            ctx.err(f"{where}.role", f"must be one of {', '.join(ROLES)}")
        else:
            inst.role = d["role"]
    if "sidechain" in d:
        inst.sidechain = bool(d["sidechain"])
    inst.width = _num(ctx, f"{where}.width", d.get("width"), inst.width, 0, 2)
    inst.reverb = _num(ctx, f"{where}.reverb", d.get("reverb"), inst.reverb, 0, 1)
    inst.reverb_decay = _num(ctx, f"{where}.reverb_decay", d.get("reverb_decay"), inst.reverb_decay, 0.2, 6)
    inst.drive = _num(ctx, f"{where}.drive", d.get("drive"), 0.0, 0, 1)
    inst.strum = _num(ctx, f"{where}.strum", d.get("strum"), inst.strum, 0, 0.1)
    inst.transpose = int(_num(ctx, f"{where}.transpose", d.get("transpose"), 0, -48, 48)) + 12 * int(
        _num(ctx, f"{where}.octave", d.get("octave"), 0, -4, 4))
    inst.mute = bool(d.get("mute", False))
    if "low" in d:
        inst.low = _pitch(ctx, f"{where}.low", d["low"], inst.low)
        if "center" not in d and inst.kind == "synth":
            inst.center = inst.low + 16
    if "center" in d:
        inst.center = _pitch(ctx, f"{where}.center", d["center"], inst.center)
    if "voicing" in d:
        if d["voicing"] not in theory.VOICINGS:
            ctx.err(f"{where}.voicing", f"unknown voicing {d['voicing']!r}; voicings: {', '.join(theory.VOICINGS)}")
        else:
            inst.voicing = d["voicing"]
    if "eq" in d:
        inst.eq = _eq(ctx, f"{where}.eq", d["eq"])
    if "delay" in d and d["delay"]:
        dl = d["delay"]
        if not isinstance(dl, dict):
            ctx.err(f"{where}.delay", "must be an object like {\"time\": \"8d\", \"feedback\": 0.35, \"mix\": 0.2}")
        else:
            t = dl.get("time", "8d")
            if isinstance(t, str) and t in DELAY_TIMES:
                beats = DELAY_TIMES[t]
            else:
                beats = _num(ctx, f"{where}.delay.time", t, 0.75, 0.05, 8)
                if isinstance(t, str) and t not in DELAY_TIMES:
                    ctx.warn(f"{where}.delay.time", f"{t!r}: use a note value ({', '.join(DELAY_TIMES)}) or beats")
            inst.delay = {"beats": beats, "feedback": _num(ctx, f"{where}.delay.feedback", dl.get("feedback"), 0.35, 0, 0.9),
                          "mix": _num(ctx, f"{where}.delay.mix", dl.get("mix"), 0.2, 0, 1),
                          "pingpong": bool(dl.get("pingpong", True))}
    if sound_given:
        if typ == "fx":
            ctx.warn(f"{where}.sound", "an FX rack has no sampled sounds; ignored")
        else:
            inst.sound = _sound_layers(ctx, f"{where}.sound", d["sound"], typ == "drums", sound_preset)
    hz = d.get("humanize")
    if hz is not None:
        if not isinstance(hz, dict):
            ctx.err(f"{where}.humanize", "must be an object {\"timing_ms\": 3, \"velocity\": 0.05}")
        else:
            inst.humanize = {"timing_ms": _num(ctx, f"{where}.humanize.timing_ms", hz.get("timing_ms"), None, 0, 40),
                             "velocity": _num(ctx, f"{where}.humanize.velocity", hz.get("velocity"), None, 0, 0.5)}
            inst.humanize = {k: v for k, v in inst.humanize.items() if v is not None}
    return inst


def _sound_layers(ctx: _Ctx, where: str, v, drums: bool, preset=None) -> list[dict] | None:
    """The validated layers of an instrument's ``sound``: each a dict with ``set`` or ``path``
    and options. Problems are reported; a layer that names no sound is dropped."""
    items = v if isinstance(v, list) else [v]
    out = []
    for i, item in enumerate(items):
        w = f"{where}[{i}]" if isinstance(v, list) else where
        if isinstance(item, str) and item.strip():
            item = item.strip()
            layer = {"path": item} if soundbank.looks_like_path(item) else {"set": item}
        elif isinstance(item, dict):
            ctx.unknown_keys(w, item, soundbank.SOUND_KEYS)
            layer = {}
            for k in ("set", "path", "file"):
                if item.get(k) is not None:
                    if isinstance(item[k], str) and item[k].strip():
                        layer[k] = item[k].strip()
                    else:
                        ctx.err(f"{w}.{k}", f"must be a string, got {item[k]!r}")
            if "set" not in layer and "path" not in layer:
                ctx.err(w, "needs 'set' (the id of an installed sound set) or 'path' (a .sf2 or .sfz file, a WAV "
                           "file or a folder of samples)")
                continue
            pr = item.get("preset")
            if pr is not None:
                ok = (isinstance(pr, str) and pr.strip()) or (isinstance(pr, int) and not isinstance(pr, bool)) or (
                    isinstance(pr, list) and len(pr) == 2 and all(isinstance(x, int) for x in pr))
                if ok:
                    layer["preset"] = pr.strip() if isinstance(pr, str) else pr
                else:
                    ctx.err(f"{w}.preset", "must be a preset name, 'bank:program', a program number or "
                                           "[bank, program]")
            for k, (lo, hi) in _SOUND_NUMBERS.items():
                if item.get(k) is not None:
                    x = _num(ctx, f"{w}.{k}", item[k], None, lo, hi)
                    if x is not None:
                        layer[k] = int(x) if k == "transpose" else x
            for k in _SOUND_BOOLS:
                if item.get(k) is not None:
                    layer[k] = bool(item[k])
            if item.get("cc") is not None:
                c = item["cc"]
                if isinstance(c, dict) and all(str(k).isdigit() and isinstance(x, (int, float)) and not isinstance(
                        x, bool) for k, x in c.items()):
                    layer["cc"] = {str(int(k)): float(min(127, max(0, x))) for k, x in c.items()}
                else:
                    ctx.err(f"{w}.cc", "must be an object of controller numbers and values 0..127, like {\"100\": 127}")
            if item.get("keyswitch") is not None:
                ks = _pitch(ctx, f"{w}.keyswitch", item["keyswitch"])
                if ks is not None:
                    layer["keyswitch"] = ks
            kr = item.get("key_range")
            if kr is not None:
                if isinstance(kr, (list, tuple)) and len(kr) == 2:
                    a, b = _pitch(ctx, f"{w}.key_range", kr[0]), _pitch(ctx, f"{w}.key_range", kr[1])
                    if a is not None and b is not None:
                        layer["key_range"] = [min(a, b), max(a, b)]
                else:
                    ctx.err(f"{w}.key_range", "must be [low, high] notes like [\"C2\", \"C5\"]")
            vr = item.get("vel_range")
            if vr is not None:
                if isinstance(vr, (list, tuple)) and len(vr) == 2 and all(
                        isinstance(x, (int, float)) and not isinstance(x, bool) for x in vr):
                    layer["vel_range"] = [int(max(1, min(vr))), int(min(127, max(vr)))]
                else:
                    ctx.err(f"{w}.vel_range", "must be [low, high] MIDI velocities 1..127")
        else:
            ctx.err(w, "must be the id of an installed sound set, a path, or an object like "
                       "{\"set\": \"<id>\", \"preset\": \"0:33\"}")
            continue
        if preset is not None and "preset" not in layer:
            layer["preset"] = preset
        out.append(layer)
    if drums and len(out) > 1:
        ctx.warn(where, "a drum kit plays one sound; the first layer is used")
        out = out[:1]
    return out or None


_BOOL_PARAMS = {"mono"}
_STR_PARAMS = {"engine": ("synth", "fm", "piano"), "filter": ("lp", "hp", "bp", "none")}


def _check_synth_params(ctx: _Ctx, where: str, params: dict):
    """Type-check synth parameter overrides in place (bad entries are reported and dropped)."""
    for k in list(params):
        v = params[k]
        if k == "osc":
            continue
        if k in _BOOL_PARAMS:
            if not isinstance(v, bool):
                ctx.err(f"{where}.{k}", f"must be true or false, got {v!r}")
                params.pop(k)
        elif k in _STR_PARAMS:
            if v is None and k == "filter":
                params[k] = "none"
            elif v not in _STR_PARAMS[k]:
                ctx.err(f"{where}.{k}", f"must be one of {', '.join(_STR_PARAMS[k])}, got {v!r}")
                params.pop(k)
        elif k == "chip_arp":
            if v is not None and (not isinstance(v, list) or not v or not all(
                    isinstance(x, (int, float)) and not isinstance(x, bool) for x in v)):
                ctx.err(f"{where}.chip_arp", "must be a list of semitone offsets like [0, 4, 7] (or null)")
                params.pop(k)
        elif k == "eq":
            params[k] = _eq(ctx, f"{where}.eq", v)
        elif isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            ctx.err(f"{where}.{k}", f"must be a number, got {v!r}")
            params.pop(k)
        elif k in ("attack", "decay", "release", "f_attack", "f_decay", "f_release", "pitch_decay", "glide",
                   "index_decay", "index2_decay", "vibrato_delay") and v < 0:
            ctx.err(f"{where}.{k}", "must be >= 0 (seconds)")
            params.pop(k)
        elif k == "cutoff" and not 20 <= v <= 22000:
            ctx.err(f"{where}.cutoff", "must be 20..22000 Hz")
            params.pop(k)
        elif k == "partials" and not 1 <= v <= 64:
            ctx.err(f"{where}.partials", "must be 1..64")
            params.pop(k)
    layers = params.get("osc")
    if isinstance(layers, list):
        for i, layer in enumerate(layers):
            for lk, lv in layer.items():
                if lk == "wave":
                    continue
                if lk not in ("level", "octave", "semi", "cents", "voices", "detune", "spread"):
                    ctx.warn(f"{where}.osc[{i}]", f"unknown oscillator key {lk!r} ignored; keys: wave, level, "
                                                  f"octave, semi, cents, voices, detune, spread")
                elif isinstance(lv, bool) or not isinstance(lv, (int, float)) or not math.isfinite(lv):
                    ctx.err(f"{where}.osc[{i}].{lk}", f"must be a number, got {lv!r}")
            if not 1 <= int(layer.get("voices", 1) if isinstance(layer.get("voices", 1), (int, float)) else 1) <= 16:
                ctx.err(f"{where}.osc[{i}].voices", "must be 1..16")


def _eq(ctx: _Ctx, where: str, eq):
    if not eq:
        return None
    if not isinstance(eq, dict):
        ctx.err(where, "must be an object like {\"low_cut\": 120, \"high_cut\": 9000, "
                       "\"bands\": [{\"hz\": 2500, \"db\": 3, \"q\": 1}]}")
        return None
    ctx.unknown_keys(where, eq, {"low_cut", "high_cut", "bands", "low_shelf", "high_shelf"})
    out = {}
    for k in ("low_cut", "high_cut"):
        if eq.get(k):
            out[k] = _num(ctx, f"{where}.{k}", eq[k], None, 10, 22000)
    bands = []
    for i, b in enumerate(eq.get("bands") or []):
        if not isinstance(b, dict):
            ctx.err(f"{where}.bands[{i}]", "must be {hz, db, q}")
            continue
        bands.append({"hz": _num(ctx, f"{where}.bands[{i}].hz", b.get("hz"), 1000, 20, 20000),
                      "db": _num(ctx, f"{where}.bands[{i}].db", b.get("db"), 0, -24, 24),
                      "q": _num(ctx, f"{where}.bands[{i}].q", b.get("q"), 1.0, 0.1, 10)})
    if bands:
        out["bands"] = bands
    for k in ("low_shelf", "high_shelf"):
        if isinstance(eq.get(k), dict):
            out[k] = {"hz": _num(ctx, f"{where}.{k}.hz", eq[k].get("hz"), 200 if k == "low_shelf" else 5000, 20, 20000),
                      "db": _num(ctx, f"{where}.{k}.db", eq[k].get("db"), 0, -24, 24)}
    return out or None


# --------------------------------------------------------------------------------- patterns

def split_steps(s) -> list[str]:
    """Steps of a step string: whitespace-separated tokens if it has spaces, else one char per step.
    '|' bar separators are ignored."""
    if isinstance(s, list):
        return [str(x) for x in s]
    s = str(s).replace("|", " " if " " in str(s).strip() else "")
    if re.search(r"\s", s.strip()):
        return s.split()
    return list(s.strip())


_DRUM_TOKENS = {".": None, "-": None, "_": None, "x": 0.8, "X": 1.0, "o": 0.5, "g": 0.4, "r": "roll", "f": "flam",
                "R": "roll3"}


def _drum_token(tok: str):
    if tok in _DRUM_TOKENS:
        return _DRUM_TOKENS[tok]
    if re.fullmatch(r"[1-9]", tok):
        return int(tok) / 9.0
    try:
        v = float(tok)
        return max(0.0, min(1.5, v))
    except ValueError:
        raise ValueError(f"invalid drum step {tok!r}: use . x X o g r (roll) f (flam), a digit 1-9 or a number")


_LINE_SPECIAL = {".": "rest", "-": "tie", "_": "tie", "~": "tie"}


def _parse_degree_step(tok: str):
    """-> (degree token, velocity, flags) for a line step token."""
    vel = 0.8
    flags = ""
    t = tok
    while t and t[-1] in "!?^":
        if t[-1] == "!":
            vel = 1.0
        elif t[-1] == "?":
            vel = 0.5
        else:
            flags += "^"
        t = t[:-1]
    if t in ("x", "X"):
        return "R", (0.55 if t == "x" else 0.7), flags + "dead"
    if t.startswith("x") and len(t) > 1:  # dead note on a degree
        return t[1:], 0.55, flags + "dead"
    return t, vel, flags


def _pattern(ctx: _Ctx, name: str, d, bpb: float):
    where = f"patterns.{name}"
    if not isinstance(d, dict):
        ctx.err(where, "must be an object with a 'kind' (drums, line, chords, arp, notes)")
        return None
    ctx.unknown_keys(where, d, PATTERN_KEYS)
    kind = d.get("kind", d.get("type"))
    if kind is not None and not isinstance(kind, str):
        ctx.err(where, f"kind must be a string ({', '.join(PATTERN_KINDS)}), got {kind!r}")
        return None
    kind = KIND_ALIASES.get(kind, kind)
    if kind is None:
        if "lanes" in d:
            kind = "drums"
        elif "notes" in d or "score" in d:
            kind = "notes"
        elif "rhythm" in d and "mode" not in d and "order" not in d:
            kind = "chords"
        elif "steps" in d:
            kind = "line"
    if kind not in PATTERN_KINDS:
        ctx.err(where, f"unknown kind {kind!r}; kinds: {', '.join(PATTERN_KINDS)}")
        return None
    pat = {"name": name, "kind": kind}
    grid = int(_num(ctx, f"{where}.grid", d.get("grid"), 16, 1, 64))
    pat["grid"] = grid
    pat["swing"] = _num(ctx, f"{where}.swing", d.get("swing"), None, 0, 0.75)
    pat["gate"] = _num(ctx, f"{where}.gate", d.get("gate"), None, 0.05, 1.5)
    pat["transpose"] = int(_num(ctx, f"{where}.transpose", d.get("transpose"), 0, -48, 48)) + 12 * int(
        _num(ctx, f"{where}.octave", d.get("octave"), 0, -4, 4))
    pat["strum"] = _num(ctx, f"{where}.strum", d.get("strum"), None, -0.1, 0.1)
    bars_given = d.get("bars")
    vel = d.get("velocity")
    if vel is not None:
        if isinstance(vel, str):
            vs = split_steps(vel)
            try:
                pat["velocity"] = [None if c in ".-_" else int(c) / 9.0 if len(c) == 1 and c.isdigit() else float(c)
                                   for c in vs]
            except ValueError:
                ctx.err(f"{where}.velocity", "use a digit string like '9.5.7.5.' (9 = full, '.' = unchanged) or a "
                                             "list of numbers")
        elif isinstance(vel, list):
            pat["velocity"] = [None if v is None else _num(ctx, f"{where}.velocity", v, 0.8, 0, 1.5) for v in vel]
        else:
            ctx.err(f"{where}.velocity", "must be a digit string or a list of numbers")
    steps_per_bar = grid * bpb / 4.0
    if kind == "drums":
        lanes = d.get("lanes")
        if not isinstance(lanes, dict) or not lanes:
            ctx.err(where, "drums pattern needs 'lanes': {\"kick\": \"x...x...x...x...\", \"snare\": "
                           "\"....x.......x...\", ...}")
            return None
        out = {}
        for lane, s in lanes.items():
            v = canonical_voice(lane)
            if v is None:
                ctx.err(f"{where}.lanes", f"unknown drum voice {lane!r}{_suggest(lane, ALL_VOICES)}; voices: "
                                          f"{', '.join(ALL_VOICES)}, or n<key> for any key of a sampled kit")
                continue
            toks = split_steps(s)
            parsed = []
            for t in toks:
                try:
                    parsed.append(_drum_token(t))
                except ValueError as exc:
                    ctx.err(f"{where}.lanes.{lane}", str(exc))
                    parsed.append(None)
            if v in out:
                ctx.warn(f"{where}.lanes", f"lane {lane!r} duplicates {v!r}; the later one wins")
            out[v] = parsed
        pat["lanes"] = out
        longest = max((len(x) for x in out.values()), default=int(steps_per_bar))
        pat["length_steps"] = longest
    elif kind in ("line", "chords", "arp"):
        key = "steps" if kind == "line" else "rhythm"
        raw = d.get(key, d.get("steps") if kind != "line" else None)
        if raw is None:
            if kind == "arp":
                raw = "x" * int(steps_per_bar)
            else:
                example = '"R.R.5.8.R.R.L.5."' if kind == "line" else '"x..x..x.x..x..x."'
                ctx.err(where, f"{kind} pattern needs '{key}' (a step string, e.g. {example})")
                return None
        toks = split_steps(raw)
        if not toks:
            ctx.err(f"{where}.{key}", "is empty")
            return None
        parsed = []
        for i, t in enumerate(toks):
            if t in _LINE_SPECIAL:
                parsed.append((_LINE_SPECIAL[t], None, 0.0, ""))
                continue
            if kind == "line":
                deg, v, flags = _parse_degree_step(t)
                if deg not in theory.COMPACT_ALIASES and not re.fullmatch(r"[b#]?(R|r|1|2|3|4|5|6|7|8|9|10|11|12|13)[',]*", deg):
                    ctx.err(f"{where}.steps", f"invalid step {t!r} at step {i + 1}: use degree tokens R 1 3 5 7 8 "
                                              f"(b7, 5', R, ...), L (root an octave down), x (dead note), '-' tie, "
                                              f"'.' rest")
                    parsed.append(("rest", None, 0.0, ""))
                    continue
                parsed.append(("note", deg, v, flags))
            else:
                v = {"x": 0.82, "X": 1.0, "o": 0.55, "d": 0.85, "u": 0.68, "m": 0.75, "M": 0.95}.get(t)
                if v is None and re.fullmatch(r"[1-9]", t):
                    v = int(t) / 9.0
                if v is None:
                    ctx.err(f"{where}.{key}", f"invalid step {t!r} at step {i + 1}: use x X o (hits), d/u "
                                              f"(down/up strum), m (palm-muted), digits 1-9, '-' tie, '.' rest")
                    parsed.append(("rest", None, 0.0, ""))
                    continue
                parsed.append(("note", None, v, t))
        pat["steps"] = parsed
        pat["length_steps"] = len(parsed)
        if kind == "chords":
            vo = d.get("voicing")
            if vo is not None and vo not in theory.VOICINGS:
                ctx.err(f"{where}.voicing", f"unknown voicing {vo!r}; voicings: {', '.join(theory.VOICINGS)}")
                vo = None
            pat["voicing"] = vo
            pat["center"] = _pitch(ctx, f"{where}.center", d.get("center"))
            pat["low"] = _pitch(ctx, f"{where}.low", d.get("low"))
        if kind == "arp":
            mode = d.get("mode", "up" if not d.get("order") else "order")
            if mode not in ARP_MODES:
                ctx.err(f"{where}.mode", f"unknown arp mode {mode!r}; modes: {', '.join(ARP_MODES)}")
                mode = "up"
            pat["mode"] = mode
            pat["octaves"] = int(_num(ctx, f"{where}.octaves", d.get("octaves"), 2, 1, 4))
            pat["low"] = _pitch(ctx, f"{where}.low", d.get("low"))
            order = d.get("order")
            if order is not None:
                otoks = split_steps(order)
                for t in otoks:
                    if t not in theory.COMPACT_ALIASES and not re.fullmatch(r"[b#]?(R|r|1|2|3|4|5|6|7|8|9|10|11|12|13)[',]*", t):
                        ctx.err(f"{where}.order", f"invalid degree {t!r}; use R 1 3 5 7 8 b7 5' ...")
                pat["order"] = otoks
            elif mode == "order":
                ctx.err(f"{where}", "mode 'order' needs an 'order' list of degrees, e.g. \"1 5 8 10 8 5\"")
    elif kind == "notes":
        notes = []
        relative = bool(d.get("relative", False))
        pat["relative"] = relative
        if d.get("score") is not None:
            try:
                notes = parse_riff(str(d["score"]), relative)
            except ValueError as exc:
                ctx.err(f"{where}.score", str(exc))
        raw_notes = d.get("notes")
        if raw_notes is not None:
            if not isinstance(raw_notes, list):
                ctx.err(f"{where}.notes", "must be a list of [beat, pitch, beats, velocity?] entries")
            else:
                for i, n in enumerate(raw_notes):
                    w = f"{where}.notes[{i}]"
                    try:
                        if isinstance(n, dict):
                            beat, pitch, dur, v = n.get("beat"), n.get("pitch"), n.get("beats", n.get("dur", 0.5)), \
                                n.get("velocity", n.get("vel", 0.85))
                        else:
                            if not isinstance(n, (list, tuple)) or len(n) < 3:
                                raise ValueError("expected [beat, pitch, beats] or [beat, pitch, beats, velocity]")
                            beat, pitch, dur = n[0], n[1], n[2]
                            v = n[3] if len(n) > 3 else 0.85
                        pitches = _riff_pitches(pitch, relative)
                        notes.append((float(beat), pitches, float(parse_duration(dur)), float(v)))
                    except (ValueError, TypeError) as exc:
                        ctx.err(w, f"{exc}")
        if not notes:
            ctx.err(where, "notes pattern needs 'notes' ([[beat, pitch, beats, velocity], ...]) or 'score' "
                           "('A4 e C5 e r q E5 q.')")
            return None
        pat["notes"] = sorted(notes, key=lambda x: x[0])
        end = max(b + l for b, _, l, _ in notes)
        pat["length_beats"] = end
    # length in bars
    if kind == "notes":
        natural = max(1, math.ceil(pat["length_beats"] / bpb - 1e-6))
    else:
        natural = max(1, math.ceil(pat["length_steps"] / steps_per_bar - 1e-6))
    if bars_given is not None:
        b = int(_num(ctx, f"{where}.bars", bars_given, natural, 1, 64))
        pat["bars"] = b
    else:
        pat["bars"] = natural
        if kind != "notes" and pat["length_steps"] % int(round(steps_per_bar)) and pat["length_steps"] > steps_per_bar:
            ctx.warn(where, f"{pat['length_steps']} steps is not a whole number of bars at grid {grid} "
                            f"({int(steps_per_bar)} steps per bar); it loops every {pat['length_steps']} steps")
    return pat


def _riff_pitches(pitch, relative: bool):
    if isinstance(pitch, (list, tuple)):
        out = []
        for p in pitch:
            out.extend(_riff_pitches(p, relative))
        return tuple(out)
    if isinstance(pitch, str) and "+" in pitch and not re.fullmatch(r"-?\d+", pitch):
        return tuple(x for part in pitch.split("+") for x in _riff_pitches(part, relative))
    if relative and isinstance(pitch, str):
        t = pitch.strip()
        if t in theory.COMPACT_ALIASES or re.fullmatch(r"[b#]?(R|r|1|2|3|4|5|6|7|8|9|10|11|12|13)[',]*", t):
            return (t,)
        raise ValueError(f"invalid relative degree {pitch!r}; use R 1 3 5 7 8 b7 5' ...")
    try:
        return (note_to_midi(pitch),)
    except (ValueError, TypeError):
        raise ValueError(f"invalid pitch {pitch!r}; use a note name like 'A4' or a MIDI number"
                         + (" (or set relative: true for degree tokens)" if isinstance(pitch, str) else ""))


def parse_riff(text: str, relative: bool = False) -> list[tuple]:
    """Parse ``"A4 e C5 e r q E5+A5 q. G4! s G4? s"`` into (beat, pitches, beats, velocity) tuples.

    Pairs of PITCH DURATION; ``r`` is a rest; chords join pitches with ``+``; a trailing ``!``
    accents (velocity 1.0) and ``?`` ghosts (0.5) the note (default 0.85); ``|`` is ignored.
    With relative=True, pitches are chord degree tokens (``R 3 5 8 b7 5'``).
    """
    toks = text.replace("|", " ").split()
    if len(toks) % 2:
        raise ValueError(f"score needs PITCH DURATION pairs; got an odd number of tokens ({len(toks)}): "
                         f"{text[:60]!r}")
    pos = 0.0
    out = []
    for i in range(0, len(toks), 2):
        p, d = toks[i], toks[i + 1]
        try:
            dur = parse_duration(d)
        except (ValueError, KeyError):
            raise ValueError(f"invalid duration {d!r} after {p!r}; use w h q e s t (dots, triplets e3, ties h+q) "
                             f"or a number of beats")
        if dur <= 0:
            raise ValueError(f"duration must be positive: {d!r}")
        if p.lower() == "r":
            pos += dur
            continue
        vel = 0.85
        while p and p[-1] in "!?":
            vel = 1.0 if p[-1] == "!" else 0.5
            p = p[:-1]
        out.append((pos, _riff_pitches(p, relative), dur, vel))
        pos += dur
    if not out and toks:
        raise ValueError("score contains only rests")
    return out


# --------------------------------------------------------------------------------- sections

def _parse_chords(ctx: _Ctx, where: str, chords, bars: int, bpb: float) -> list:
    if chords is None:
        return [[(0.0, None)] for _ in range(bars)]
    if isinstance(chords, str):
        if "|" not in chords:
            ctx.err(where, "a chords string needs '|' bar lines (e.g. \"Am | Am/G | F | E\"); or give a list with "
                           "one entry per bar")
            return [[(0.0, None)] for _ in range(bars)]
        entries = [c.strip() for c in chords.strip().strip("|").split("|")]
    elif isinstance(chords, list):
        entries = [str(c) for c in chords]
    else:
        ctx.err(where, "must be a list of chord symbols (one per bar) or a '|'-separated string")
        return [[(0.0, None)] for _ in range(bars)]
    if not entries:
        return [[(0.0, None)] for _ in range(bars)]
    parsed = []
    prev = None
    for i, e in enumerate(entries):
        syms = e.split() or ["N.C."]
        bar = []
        for j, s in enumerate(syms):
            if s in ("%", "/", "-"):
                c = prev
            else:
                try:
                    c = theory.parse_chord(s)
                except ValueError as exc:
                    ctx.err(f"{where}[{i}]", str(exc))
                    c = None
            bar.append((j * bpb / len(syms), c))
            prev = c
        parsed.append(bar)
    if len(parsed) > bars:
        ctx.warn(where, f"{len(parsed)} chord entries for {bars} bars; the extra ones are ignored")
    elif bars % len(parsed):
        ctx.warn(where, f"{len(parsed)} chord entries do not divide {bars} bars evenly; the progression cycles "
                        f"and is cut at the section end")
    return [parsed[i % len(parsed)] for i in range(bars)]


def _segments(ctx: _Ctx, where: str, v, sec_start: int, sec_end: int, patterns: dict) -> list[dict]:
    """Normalise a section's part assignment into segments {pattern, b0, b1, transpose, velocity, offset}."""
    if v is None or v == "" or v == "-" or v is False:
        return [{"pattern": None, "b0": sec_start, "b1": sec_end}]
    if isinstance(v, str):
        v = [{"pattern": v}]
    elif isinstance(v, dict):
        v = [v]
    if not isinstance(v, list):
        ctx.err(where, "must be a pattern name, null (silent), an object {pattern, bars, transpose, velocity} or a "
                       "list of such objects")
        return []
    out = []
    for i, s in enumerate(v):
        w = f"{where}[{i}]" if len(v) > 1 else where
        if isinstance(s, str) or s is None:
            s = {"pattern": s}
        if not isinstance(s, dict):
            ctx.err(w, "segment must be an object {pattern, bars: [from, to]}")
            continue
        ctx.unknown_keys(w, s, SEGMENT_KEYS)
        name = s.get("pattern")
        if name in ("", "-"):
            name = None
        if name is not None and not isinstance(name, str):
            ctx.err(w, f"pattern must be a pattern name string or null, got {name!r}")
            continue
        if name is not None and name not in patterns:
            ctx.err(w, f"unknown pattern {name!r}{_suggest(name, patterns)}; patterns: {', '.join(patterns) or '(none)'}")
            continue
        b0, b1 = sec_start, sec_end
        ok = True
        if "bars" in s:
            bb = s["bars"]

            def _isnum(x):
                return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)

            if isinstance(bb, str) and re.fullmatch(r"\s*\d+\s*(-\s*\d+\s*)?", bb):
                parts = [int(x) for x in bb.split("-")]
                b0, b1 = parts[0], parts[-1]
            elif isinstance(bb, (list, tuple)) and len(bb) == 2 and all(_isnum(x) for x in bb):
                b0, b1 = int(bb[0]), int(bb[1])
            elif _isnum(bb):
                b0 = b1 = int(bb)
            else:
                ok = False
        for k in ("from_bar", "to_bar"):
            if k in s:
                v = _num(ctx, f"{w}.{k}", s[k], None, 1, 100000)
                if v is None:
                    ok = False
                elif k == "from_bar":
                    b0 = int(v)
                else:
                    b1 = int(v)
        if not ok:
            ctx.err(w, f"bars must be [from, to] (absolute bar numbers, inclusive) or 'from-to', got "
                       f"{s.get('bars')!r}")
            continue
        if b0 < sec_start or b1 > sec_end or b1 < b0:
            ctx.warn(w, f"bars {b0}-{b1} are outside the section's bars {sec_start}-{sec_end}; clipped")
            b0, b1 = max(b0, sec_start), min(b1, sec_end)
            if b1 < b0:
                continue
        out.append({"pattern": name, "b0": b0, "b1": b1,
                    "transpose": int(_num(ctx, f"{w}.transpose", s.get("transpose"), 0, -48, 48))
                    + 12 * int(_num(ctx, f"{w}.octave", s.get("octave"), 0, -4, 4)),
                    "velocity": _num(ctx, f"{w}.velocity", s.get("velocity"), 1.0, 0, 2),
                    "offset": int(_num(ctx, f"{w}.offset", s.get("offset"), 0, 0, 64))})
    return out


def _bar_beat(ctx: _Ctx, where: str, d: dict, bpb: float, bar_key="bar", beat_key="beat") -> float | None:
    bar = d.get(bar_key)
    if bar is None:
        ctx.err(where, f"needs '{bar_key}' (absolute bar number, 1 = first bar)")
        return None
    bar = _num(ctx, f"{where}.{bar_key}", bar, None, 1, 100000)
    beat = _num(ctx, f"{where}.{beat_key}", d.get(beat_key), 1.0, 1, bpb + 0.999)
    if bar is None:
        return None
    return (bar - 1) * bpb + (beat - 1)


def _parts_set(ctx: _Ctx, where: str, v, instruments: dict) -> set | None:
    if v is None or v == "all" or v == ["all"]:
        return None
    if isinstance(v, str):
        v = [v]
    if not isinstance(v, list):
        ctx.err(where, "parts must be 'all' or a list of instrument names")
        return None
    out = set()
    for name in v:
        if not isinstance(name, str):
            ctx.err(where, f"instrument names must be strings, got {name!r}")
            continue
        if name not in instruments:
            ctx.err(where, f"unknown instrument {name!r}{_suggest(name, instruments)}; instruments: "
                           f"{', '.join(instruments)}")
        out.add(name)
    return out


# ------------------------------------------------------------------------------------ compile

def compile_arrangement(arr: dict, project_key: str | None = None, beats_per_bar: float = 4.0,
                        strict: bool = True) -> Compiled:
    """Validate and expand an arrangement. Raises :class:`ArrangementError` on errors when strict."""
    ctx = _Ctx()
    comp = _compile(ctx, arr, project_key, beats_per_bar)
    if ctx.errors and strict:
        raise ArrangementError(ctx.errors + ctx.warnings)
    comp.warnings = ctx.errors + ctx.warnings
    return comp


def validate_arrangement(arr) -> list[str]:
    """All problems with an arrangement as messages starting with 'error:' or 'warning:' ([] = perfect)."""
    ctx = _Ctx()
    try:
        comp = _compile(ctx, arr, None, 4.0)
        for name, inst in comp.instruments.items():
            for layer in inst.sound or []:
                sid = layer.get("set")
                if sid and soundbank._installed(sid) is None and not soundbank.looks_like_path(sid) \
                        and soundbank.find_path(sid, soundbank.project_dirs(None, arr)) is None:
                    stand_in = "the synthesised kit" if inst.kind == "drums" else \
                        f"the synthesised preset {inst.preset!r}"
                    ctx.warn(f"instruments.{name}.sound", f"sound set {sid!r} is not installed on this machine: "
                                                          f"{stand_in} plays instead (synth2 sounds install {sid})")
    except Exception as exc:  # pragma: no cover - defensive: never crash validation
        ctx.err("arrangement", f"could not be processed: {type(exc).__name__}: {exc}")
    return ctx.errors + ctx.warnings


def _compile(ctx: _Ctx, arr, project_key, bpb) -> Compiled:
    empty = Compiled({}, [], {}, {}, {}, [], [], 0.0, bpb, [], [])
    if not isinstance(arr, dict):
        ctx.err("arrangement", "must be a JSON object with 'instruments', 'patterns' and 'sections'")
        return empty
    ctx.unknown_keys("arrangement", arr, TOP_KEYS)
    if arr.get("format") not in (None, FORMAT_NAME):
        ctx.warn("format", f"expected {FORMAT_NAME!r}")
    key_name = arr.get("key") or project_key or "C major"
    try:
        from ..ai.composer import parse_key

        parse_key(key_name)
    except ValueError as exc:
        ctx.err("key", str(exc))
        key_name = "C major"
    song_key = theory.scale_for(key_name)
    g_swing = _num(ctx, "swing", arr.get("swing"), 0.0, 0, 0.75)
    energy_resp = _num(ctx, "energy_response", arr.get("energy_response"), 1.0, 0, 2)
    g_hum = arr.get("humanize") if isinstance(arr.get("humanize"), dict) else {}

    # instruments
    instruments: dict[str, Instrument] = {}
    raw_inst = arr.get("instruments")
    if not isinstance(raw_inst, dict) or not raw_inst:
        ctx.err("instruments", "needs at least one instrument, e.g. {\"drums\": {\"type\": \"drums\"}, "
                               "\"bass\": {\"preset\": \"bass_pick\"}}")
        raw_inst = {}
    for name, d in raw_inst.items():
        inst = _instrument(ctx, str(name), d)
        if inst is not None:
            if g_hum:
                inst.humanize = {**{k: v for k, v in g_hum.items() if k in ("timing_ms", "velocity")},
                                 **inst.humanize}
            instruments[str(name)] = inst

    # patterns
    patterns: dict[str, dict] = {}
    raw_pat = arr.get("patterns") or {}
    if not isinstance(raw_pat, dict):
        ctx.err("patterns", "must be an object {name: pattern}")
        raw_pat = {}
    for name, d in raw_pat.items():
        p = _pattern(ctx, str(name), d, bpb)
        if p is not None:
            patterns[str(name)] = p
    # keep names of broken patterns so sections referencing them do not cascade errors
    known_patterns = set(raw_pat) if isinstance(raw_pat, dict) else set()

    # sections
    sections: list[Section] = []
    raw_sec = arr.get("sections")
    if not isinstance(raw_sec, list) or not raw_sec:
        ctx.err("sections", "needs a list of sections like {\"name\": \"Intro\", \"start_bar\": 1, \"bars\": 8, "
                            "\"chords\": [\"Am\", ...], \"parts\": {\"drums\": \"beat\"}}")
        raw_sec = []
    by_name: dict[str, dict] = {}
    next_bar = 1
    fills_raw: list[tuple[str, dict]] = []
    events_raw: list[tuple[str, dict]] = []
    stops_raw: list[tuple[str, dict]] = []
    for i, s in enumerate(raw_sec):
        where = f"sections[{i}]"
        if not isinstance(s, dict):
            ctx.err(where, "must be an object")
            continue
        if s.get("like") is not None:
            base = by_name.get(s["like"]) if isinstance(s["like"], str) else None
            if base is None:
                ctx.err(f"{where}.like", f"no earlier section named {s['like']!r}")
            else:
                merged = {k: copy.deepcopy(v) for k, v in base.items() if k not in ("name", "start_bar", "fills",
                                                                                      "events", "stops", "like")}
                merged.update(s)
                s = merged
        name = str(s.get("name", f"section {i + 1}"))
        where = f"sections[{i}] '{name}'"
        ctx.unknown_keys(where, s, SECTION_KEYS)
        start = s.get("start_bar", next_bar)
        start = int(_num(ctx, f"{where}.start_bar", start, next_bar, 1, 100000))
        bars = s.get("bars")
        if bars is None:
            ch = s.get("chords")
            bars = len(ch) if isinstance(ch, list) and ch else None
            if bars is None:
                ctx.err(where, "needs 'bars' (number of bars)")
                continue
        bars = int(_num(ctx, f"{where}.bars", bars, 1, 1, 10000))
        if start != next_bar and sections:
            if start < next_bar:
                ctx.warn(where, f"starts at bar {start} but the previous section runs to bar {next_bar - 1} "
                                f"(sections overlap; the later one wins for chords)")
            else:
                ctx.warn(where, f"starts at bar {start}; bars {next_bar}-{start - 1} have no section (silence)")
        next_bar = start + bars
        chords = _parse_chords(ctx, f"{where}.chords", s.get("chords"), bars, bpb)
        energy = _num(ctx, f"{where}.energy", s.get("energy"), 0.8, 0, 1)
        sec_key = song_key
        if s.get("key"):
            try:
                from ..ai.composer import parse_key

                parse_key(s["key"])
                sec_key = theory.scale_for(s["key"])
            except ValueError as exc:
                ctx.err(f"{where}.key", str(exc))
        parts_raw = s.get("parts") or {}
        if not isinstance(parts_raw, dict):
            ctx.err(f"{where}.parts", "must be an object {instrument: pattern}")
            parts_raw = {}
        parts = {}
        for inst_name, v in parts_raw.items():
            if inst_name not in instruments:
                if inst_name not in raw_inst:
                    ctx.err(f"{where}.parts", f"unknown instrument {inst_name!r}{_suggest(inst_name, instruments)}; "
                                              f"instruments: {', '.join(instruments)}")
                continue
            segs = _segments(ctx, f"{where}.parts.{inst_name}", v, start, start + bars - 1,
                             {k: patterns.get(k) for k in known_patterns})
            segs = [g for g in segs if g["pattern"] is None or g["pattern"] in patterns]
            for g in segs:
                if g["pattern"] is None:
                    continue
                kind = patterns[g["pattern"]]["kind"]
                ik = instruments[inst_name].kind
                if (ik == "drums") != (kind == "drums") or ik == "fx":
                    ctx.err(f"{where}.parts.{inst_name}",
                            f"pattern {g['pattern']!r} is a {kind} pattern but {inst_name!r} is a "
                            f"{'drum kit' if ik == 'drums' else 'FX instrument' if ik == 'fx' else 'pitched instrument'}"
                            f"; drum kits play 'drums' patterns, pitched instruments play line/chords/arp/notes "
                            f"patterns, FX instruments are played by events")
                    g["pattern"] = None
            parts[inst_name] = segs
        sec = Section(name, start, bars, chords, energy, parts, sec_key,
                      int(_num(ctx, f"{where}.transpose", s.get("transpose"), 0, -24, 24)),
                      _num(ctx, f"{where}.swing", s.get("swing"), None, 0, 0.75))
        sections.append(sec)
        by_name[name] = s
        for key, sink in (("fills", fills_raw), ("events", events_raw), ("stops", stops_raw)):
            v = s.get(key)
            if v:
                if not isinstance(v, list):
                    ctx.err(f"{where}.{key}", "must be a list")
                else:
                    sink.extend((f"{where}.{key}[{j}]", x) for j, x in enumerate(v))
        if s.get("crash"):
            kit = next((n for n, x in instruments.items() if x.kind == "drums"), None)
            if kit:
                events_raw.append((f"{where}.crash", {"type": "hit", "part": kit, "voice": "crash", "bar": start,
                                                      "velocity": 0.95}))
    for key, sink in (("fills", fills_raw), ("events", events_raw), ("stops", stops_raw)):
        v = arr.get(key)
        if v:
            if not isinstance(v, list):
                ctx.err(key, "must be a list")
            else:
                sink.extend((f"{key}[{j}]", x) for j, x in enumerate(v))

    end_beat = max(((s.start_bar - 1 + s.bars) * bpb for s in sections), default=0.0)
    # chord track (sorted change points)
    chord_track: list[tuple[float, object, tuple]] = []
    for s in sorted(sections, key=lambda s: s.start_bar):
        for bi, bar in enumerate(s.chords):
            b0 = (s.start_bar - 1 + bi) * bpb
            for off, c in bar:
                chord_track.append((b0 + off, c, s.key))
    chord_track.sort(key=lambda x: x[0])
    ct_beats = [c[0] for c in chord_track]

    import bisect

    def chord_at(beat: float):
        i = bisect.bisect_right(ct_beats, beat + 1e-6) - 1
        if i < 0:
            return None, song_key
        return chord_track[i][1], chord_track[i][2]

    def next_change(beat: float) -> float:
        i = bisect.bisect_right(ct_beats, beat + 1e-6)
        return ct_beats[i] if i < len(ct_beats) else float("inf")

    drum_events: dict[str, list] = {n: [] for n, x in instruments.items() if x.kind == "drums"}
    note_events: dict[str, list] = {n: [] for n, x in instruments.items() if x.kind == "synth"}
    fx_events: dict[str, list] = {n: [] for n, x in instruments.items() if x.kind == "fx"}

    # fills: (inst, start_beat, end_beat, pattern)
    fills: list[tuple[str, float, float, str, dict]] = []
    for where, f in fills_raw:
        if not isinstance(f, dict):
            ctx.err(where, "fill must be {part, bar, beat, pattern}")
            continue
        ctx.unknown_keys(where, f, {"part", "bar", "beat", "pattern", "comment", "transpose", "velocity"})
        part = f.get("part")
        if not isinstance(part, str) or part not in instruments:
            ctx.err(where, f"unknown part {part!r}{_suggest(part, instruments)}; instruments: {', '.join(instruments)}")
            continue
        pname = f.get("pattern")
        if not isinstance(pname, str):
            ctx.err(where, f"pattern must be a pattern name, got {pname!r}")
            continue
        if pname not in patterns:
            if pname not in known_patterns:
                ctx.err(where, f"unknown pattern {pname!r}{_suggest(pname, patterns)}")
            continue
        b = _bar_beat(ctx, where, f, bpb)
        if b is None:
            continue
        bar0 = math.floor(b / bpb + 1e-9) * bpb
        fills.append((part, b, bar0 + patterns[pname]["bars"] * bpb, pname,
                      {"transpose": int(f.get("transpose", 0) or 0), "velocity": float(f.get("velocity", 1.0) or 1.0),
                       "bar0": bar0}))

    def section_of(beat: float) -> Section | None:
        for s in sections:
            if (s.start_bar - 1) * bpb <= beat + 1e-9 < (s.start_bar - 1 + s.bars) * bpb:
                return s
        return None

    # expand sections
    voicing_state: dict[str, list[int] | None] = {}
    for s in sections:
        for inst_name, segs in s.parts.items():
            inst = instruments[inst_name]
            if inst.mute:
                continue
            for g in segs:
                if g["pattern"] is None:
                    continue
                pat = patterns[g["pattern"]]
                b0 = (g["b0"] - 1) * bpb
                b1 = g["b1"] * bpb
                # cut away fill windows of this part
                windows = [(fs, fe) for (fp, fs, fe, _, _) in fills if fp == inst_name and fs < b1 and fe > b0]
                evs = _expand(ctx, pat, inst, b0, b1, b0 - g["offset"] * bpb, g, s, chord_at, next_change, bpb,
                              g_swing, energy_resp, voicing_state)
                evs = _cut_windows(evs, windows)
                (drum_events if inst.kind == "drums" else note_events)[inst_name].extend(evs)
    for (part, fs, fe, pname, extra) in fills:
        inst = instruments[part]
        if inst.mute:
            continue
        pat = patterns[pname]
        if (inst.kind == "drums") != (pat["kind"] == "drums") or inst.kind == "fx":
            ctx.err(f"fills ({part} @ beat {fs:g})", f"pattern {pname!r} ({pat['kind']}) does not fit instrument "
                                                     f"{part!r} ({inst.kind})")
            continue
        s = section_of(fs) or (sections[0] if sections else None)
        if s is None:
            continue
        g = {"transpose": extra["transpose"], "velocity": extra["velocity"], "pattern": pname}
        evs = _expand(ctx, pat, inst, extra["bar0"], fe, extra["bar0"], g, s, chord_at, next_change, bpb, g_swing,
                      energy_resp, voicing_state)
        evs = [e for e in evs if e.beat >= fs - 1e-9]
        (drum_events if inst.kind == "drums" else note_events)[part].extend(evs)

    # one-off events
    processes: list[ProcessEvent] = []
    fx_default = next((n for n, x in instruments.items() if x.kind == "fx"), None)
    for where, e in events_raw:
        if not isinstance(e, dict):
            ctx.err(where, "event must be an object with a 'type'")
            continue
        ctx.unknown_keys(where, e, EVENT_KEYS)
        typ = e.get("type", "fx" if "sound" in e else None)
        if not isinstance(typ, str) or typ not in EVENT_TYPES:
            ctx.err(where, f"unknown event type {typ!r}; types: {', '.join(EVENT_TYPES)}")
            continue
        beat = _bar_beat(ctx, where, e, bpb)
        if beat is None:
            continue
        beats = e.get("beats")
        if typ in PROCESS_EFFECTS:
            length = _num(ctx, f"{where}.beats", beats, bpb, 0.0625, 256)
            parts = _parts_set(ctx, f"{where}.parts", e.get("parts"), instruments)
            exc = _parts_set(ctx, f"{where}.except", e.get("except"), instruments) or set()
            params = {k: v for k, v in e.items() if k not in ("type", "bar", "beat", "beats", "parts", "except",
                                                               "comment")}
            if typ == "filter" and params.get("mode", "lp") not in ("lp", "hp", "bp"):
                ctx.err(where, "filter mode must be lp, hp or bp")
            processes.append(ProcessEvent(typ, beat, beat + length, parts, exc, params))
            continue
        part = e.get("part")
        if typ == "fx":
            sound = e.get("sound")
            if not isinstance(sound, str) or sound not in FX_SOUNDS:
                ctx.err(where, f"unknown fx sound {sound!r}{_suggest(sound, FX_SOUNDS)}; sounds: "
                               f"{', '.join(FX_SOUNDS)}")
                continue
            if part is None:
                if fx_default is None:
                    fx_default = "fx"
                    instruments["fx"] = Instrument("fx", "fx", role="fx", reverb_decay=1.6,
                                                   humanize={"timing_ms": 0.0, "velocity": 0.0})
                    fx_events["fx"] = []
                part = fx_default
            if not isinstance(part, str) or part not in instruments or instruments[part].kind != "fx":
                ctx.err(where, f"part {part!r} is not an FX instrument (type 'fx')")
                continue
            length = _num(ctx, f"{where}.beats", beats, FX_SOUNDS[sound][0], 0.0625, 256)
            pitch = _pitch(ctx, f"{where}.pitch", e.get("pitch"))
            params = dict(e.get("params") or {})
            for k in ("rate", "depth", "glide", "drop", "tone", "density", "up", "pan_sweep", "from_hz", "to_hz"):
                if k in e:
                    params[k] = e[k]
            fx_events[part].append(FxEvent(beat, length, sound, pitch,
                                           _num(ctx, f"{where}.velocity", e.get("velocity"), 1.0, 0, 1.5),
                                           _num(ctx, f"{where}.gain_db", e.get("gain_db"), 0.0, -48, 12),
                                           _num(ctx, f"{where}.pan", e.get("pan"), 0.0, -1, 1), params))
            continue
        if not isinstance(part, str) or part not in instruments:
            ctx.err(where, f"needs 'part' (an instrument name); instruments: {', '.join(instruments)}")
            continue
        inst = instruments[part]
        vel = _num(ctx, f"{where}.velocity", e.get("velocity"), 0.9, 0, 1.5)
        if typ == "hit":
            if inst.kind != "drums":
                ctx.err(where, f"'hit' events play drum voices; {part!r} is not a drum kit (use 'note' or 'chord')")
                continue
            voice = canonical_voice(e.get("voice", ""))
            if voice is None:
                ctx.err(where, f"unknown drum voice {e.get('voice')!r}; voices: {', '.join(ALL_VOICES)}")
                continue
            drum_events[part].append(DrumEvent(beat, voice, vel))
        else:
            if inst.kind != "synth":
                ctx.err(where, f"'{typ}' events need a pitched instrument; {part!r} is a {inst.kind} instrument")
                continue
            length = _num(ctx, f"{where}.beats", beats, 1.0, 0.0625, 256)
            if typ == "note":
                if e.get("pitch") is None:
                    ctx.err(where, "note event needs 'pitch' (e.g. 'A4' or ['A4', 'C5'])")
                    continue
                try:
                    pitches = tuple(int(p) + inst.transpose for p in _riff_pitches(e["pitch"], False))
                except ValueError as exc:
                    ctx.err(where, str(exc))
                    continue
            else:
                chord, key = chord_at(beat)
                if chord is None:
                    ctx.warn(where, "no chord at this position; chord event skipped")
                    continue
                vo = e.get("voicing", inst.voicing)
                if vo not in theory.VOICINGS:
                    ctx.err(where, f"unknown voicing {vo!r}")
                    continue
                pitches = tuple(p + inst.transpose for p in theory.voice_chord(chord, vo, inst.center, inst.low,
                                                                                None, key))
            note_events[part].append(NoteEvent(beat, length, pitches, vel, strum=inst.strum))

    # stops
    stops: list[StopEvent] = []
    for where, st in stops_raw:
        if not isinstance(st, dict):
            ctx.err(where, "stop must be {bar, beat, beats, parts}")
            continue
        ctx.unknown_keys(where, st, STOP_KEYS)
        b0 = _bar_beat(ctx, where, st, bpb)
        if b0 is None:
            continue
        if st.get("to_bar") is not None:
            b1 = _bar_beat(ctx, where, {"bar": st["to_bar"], "beat": st.get("to_beat", 1)}, bpb)
        elif st.get("beats") is not None:
            b1 = b0 + _num(ctx, f"{where}.beats", st["beats"], 1.0, 0.0625, 10000)
        else:
            b1 = (math.floor(b0 / bpb + 1e-9) + 1) * bpb  # to the end of the bar
        if b1 is None or b1 <= b0:
            ctx.err(where, "the stop must end after it starts")
            continue
        stops.append(StopEvent(b0, b1, _parts_set(ctx, f"{where}.parts", st.get("parts"), instruments),
                               _parts_set(ctx, f"{where}.except", st.get("except"), instruments) or set()))

    # drop events that start inside a stop of their part
    def stopped(inst_name, beat):
        return any(s.beat0 - 1e-6 <= beat < s.beat1 - 1e-6 and _applies(s.parts, s.exclude, inst_name) for s in stops)

    for coll in (drum_events, note_events, fx_events):
        for name in coll:
            coll[name] = sorted((e for e in coll[name] if not stopped(name, e.beat)), key=lambda e: e.beat)

    for name, inst in instruments.items():
        has = drum_events.get(name) or note_events.get(name) or fx_events.get(name)
        if not has and not inst.mute:
            ctx.warn(f"instruments.{name}", "never plays (no section part, fill or event uses it)")
    end_beat = max([end_beat] + [e.beat + getattr(e, "beats", 0.0) for coll in (note_events, fx_events)
                                 for evs in coll.values() for e in evs])
    if arr.get("length_bars") is not None:
        lb = _num(ctx, "length_bars", arr.get("length_bars"), None, 1, 100000)
        if lb is not None:
            end_beat = lb * bpb
    return Compiled(instruments, sections, drum_events, note_events, fx_events, processes, stops, end_beat, bpb,
                    [], [(b, c) for b, c, _ in chord_track])


def _cut_windows(evs: list, windows: list[tuple[float, float]]) -> list:
    if not windows:
        return evs
    out = []
    for e in evs:
        if any(fs - 1e-9 <= e.beat < fe - 1e-9 for fs, fe in windows):
            continue
        if isinstance(e, NoteEvent):
            for fs, fe in windows:
                if e.beat < fs < e.beat + e.beats:
                    e.beats = fs - e.beat
        out.append(e)
    return out


def _step_beats(pat: dict, bpb: float) -> float:
    return 4.0 / pat["grid"]


def _swing_offset(pat, section, g_swing, step_index_in_bar, step_beats):
    sw = pat["swing"] if pat["swing"] is not None else (section.swing if section.swing is not None else g_swing)
    if sw and step_index_in_bar % 2 == 1 and pat["grid"] in (8, 16, 32):
        return sw * step_beats
    return 0.0


def _energy(section: Section, resp: float) -> tuple[float, float]:
    e = section.energy
    vel = 1.0 + resp * 0.22 * (e - 0.8)
    bright = resp * 0.9 * (e - 0.8)
    return vel, bright


def _expand(ctx, pat, inst: Instrument, b0: float, b1: float, origin: float, seg: dict, section: Section,
            chord_at, next_change, bpb, g_swing, energy_resp, voicing_state) -> list:
    """Events of one pattern played from ``origin`` (beat where pattern step 0 falls), kept in [b0, b1)."""
    kind = pat["kind"]
    sb = _step_beats(pat, bpb)
    e_vel, e_bright = _energy(section, energy_resp)
    seg_vel = float(seg.get("velocity", 1.0)) * e_vel
    vel_over = pat.get("velocity")
    out = []
    if kind == "notes":
        length = pat["bars"] * bpb
        k0 = math.floor((b0 - origin) / length + 1e-9)
        k = max(k0, 0)
        trans = pat["transpose"] + seg.get("transpose", 0) + inst.transpose + section.transpose
        while origin + k * length < b1 - 1e-9:
            base = origin + k * length
            for (beat, pitches, dur, v) in pat["notes"]:
                if beat >= length - 1e-9:
                    continue
                t = base + beat
                if t < b0 - 1e-9 or t >= b1 - 1e-9:
                    continue
                if pat.get("relative"):
                    chord, key = chord_at(t)
                    if chord is None:
                        continue
                    ps = tuple(_token_pitch(tok, chord, key, inst.low) + pat["transpose"] + seg.get("transpose", 0)
                               + inst.transpose for tok in pitches)
                else:
                    ps = tuple(int(p) + trans for p in pitches)
                d = min(dur, b1 - t)
                out.append(NoteEvent(t, d, ps, v * seg_vel, e_bright, strum=inst.strum))
            k += 1
        return out

    steps_per_bar = int(round(pat["grid"] * bpb / 4))
    total_steps = int(round((b1 - origin) / sb))
    first_step = max(0, int(math.floor((b0 - origin) / sb + 1e-9)))
    pat_steps = pat["bars"] * steps_per_bar
    if kind == "drums":
        for voice, toks in pat["lanes"].items():
            n = len(toks)
            if n == 0:
                continue
            for i in range(first_step, total_steps):
                local = i % pat_steps
                tok = toks[local % n]
                if tok is None:
                    continue
                t = origin + i * sb
                if t < b0 - 1e-9 or t >= b1 - 1e-9:
                    continue
                t += _swing_offset(pat, section, g_swing, local % steps_per_bar, sb)
                vmul = seg_vel
                if vel_over and vel_over[local % len(vel_over)] is not None:
                    tok = vel_over[local % len(vel_over)] if not isinstance(tok, str) else tok
                if tok == "roll":
                    out.append(DrumEvent(t, voice, 0.75 * vmul))
                    out.append(DrumEvent(t + sb / 2, voice, 0.65 * vmul))
                elif tok == "roll3":
                    for j in range(3):
                        out.append(DrumEvent(t + j * sb / 3, voice, (0.6 + 0.1 * j) * vmul))
                elif tok == "flam":
                    out.append(DrumEvent(t - 0.045, voice, 0.4 * vmul))  # grace note 0.045 beats early
                    out.append(DrumEvent(t, voice, 0.9 * vmul))
                else:
                    out.append(DrumEvent(t, voice, float(tok) * vmul))
        return out

    steps = pat["steps"]
    n = len(steps)
    gate = pat["gate"]
    trans = pat["transpose"] + seg.get("transpose", 0) + inst.transpose
    vo = pat.get("voicing") or inst.voicing
    center = pat.get("center") or inst.center
    low = pat.get("low") or inst.low
    strum = pat.get("strum") if pat.get("strum") is not None else inst.strum
    arp_idx = 0
    i = first_step
    # walk the steps; a note's length = its step + following ties
    while i < total_steps:
        local = i % pat_steps
        typ, deg, v, flags = steps[local % n]
        if typ != "note":
            i += 1
            continue
        t_grid = origin + i * sb
        if t_grid >= b1 - 1e-9:
            break
        j = i + 1
        while j < total_steps and steps[(j % pat_steps) % n][0] == "tie":
            j += 1
        if t_grid < b0 - 1e-9:
            i = j
            continue
        t = t_grid + _swing_offset(pat, section, g_swing, local % steps_per_bar, sb)
        length_steps = j - i
        end = min(origin + j * sb, b1)
        # split held notes at chord changes (re-strike with the new chord)
        pieces = []
        cur = t
        while True:
            nc = next_change(cur)
            if nc < end - 1e-6 and nc > cur + 1e-6 and kind != "arp":
                pieces.append((cur, nc))
                cur = nc
            else:
                pieces.append((cur, end))
                break
        if vel_over and vel_over[local % len(vel_over)] is not None:
            v = vel_over[local % len(vel_over)]
        for pi, (ps, pe) in enumerate(pieces):
            chord, key = chord_at(ps)
            if chord is None:
                continue
            g = gate if gate is not None else (0.92 if length_steps > 1 or pi < len(pieces) - 1 else 0.85)
            dur_beats = (pe - ps) - (1 - g) * sb if pi == len(pieces) - 1 else (pe - ps)
            dur_beats = max(dur_beats, sb * 0.15)
            vv = v * seg_vel * (0.9 if pi > 0 else 1.0)
            bright = e_bright
            gate_scale = 1.0
            if kind == "line":
                try:
                    pitch = _token_pitch(deg, chord, key, low) + trans
                except ValueError as exc:
                    ctx.err(f"patterns.{pat['name']}", str(exc))
                    return out
                if "dead" in flags:
                    dur_beats = min(dur_beats, sb * 0.35)
                    bright -= 1.5
                out.append(NoteEvent(ps, dur_beats, (pitch,), vv, bright))
            elif kind == "chords":
                prev = voicing_state.get(inst.name)
                pitches = theory.voice_chord(chord, vo, center, low, prev, key)
                voicing_state[inst.name] = pitches
                st = strum
                if flags == "u":
                    st = -strum if strum else -0.008
                elif flags == "d" and not strum:
                    st = 0.008
                if flags in ("m", "M"):
                    dur_beats = min(dur_beats, sb * 0.4)
                    bright -= 1.3
                out.append(NoteEvent(ps, dur_beats, tuple(p + trans for p in pitches), vv, bright, st, gate_scale))
            else:  # arp
                tones = _arp_tones(pat, chord, key, low, vo)
                if not tones:
                    break
                p = _arp_pick(pat, tones, arp_idx, f"{inst.name}|{ps:.4f}")
                arp_idx += 1
                out.append(NoteEvent(ps, dur_beats, (p + trans,), vv, bright))
        i = j
    return out


def _token_pitch(tok: str, chord, key, low: int) -> int:
    t = theory.COMPACT_ALIASES.get(tok, tok)
    m = re.fullmatch(r"([b#]?)(R|r)([',]*)", t)
    if m:
        acc, _, octs = m.groups()
        shift = 12 * (octs.count("'") - octs.count(","))
        return theory.place_in_register(chord.bass, low) + shift + (1 if acc == "#" else -1 if acc == "b" else 0)
    root = theory.place_in_register(chord.root, low)
    return root + theory.degree_semitones(t, chord, key)


def _arp_tones(pat, chord, key, low_default, voicing) -> list[int]:
    low = pat.get("low") or low_default
    if pat.get("order"):
        return [_token_pitch(t, chord, key, low) for t in pat["order"]]
    root = theory.place_in_register(chord.root, low)
    ivs = [0]
    iv3 = chord.tone(3)
    if iv3 is not None:
        ivs.append(iv3)
    iv5 = chord.tone(5)
    ivs.append(iv5 if iv5 is not None else 7)
    if chord.tone(7) is not None:
        ivs.append(chord.tone(7))
    for x in chord.intervals:
        if x > 12 and (x % 12) not in [i % 12 for i in ivs]:
            ivs.append(x % 12)
    ivs = sorted(set(ivs))
    tones = []
    for o in range(pat["octaves"]):
        tones += [root + 12 * o + i for i in ivs]
    return tones


def _arp_pick(pat, tones, idx, salt) -> int:
    mode = pat["mode"]
    n = len(tones)
    if mode in ("up", "order"):
        return tones[idx % n]
    if mode == "down":
        return tones[::-1][idx % n]
    if mode in ("updown", "pingpong"):
        cyc = tones + tones[-2:0:-1] if n > 2 else tones
        return cyc[idx % len(cyc)]
    if mode == "downup":
        r = tones[::-1]
        cyc = r + r[-2:0:-1] if n > 2 else r
        return cyc[idx % len(cyc)]
    # random but deterministic
    import zlib

    h = zlib.crc32(f"{idx}|{salt}".encode())
    return tones[h % n]
