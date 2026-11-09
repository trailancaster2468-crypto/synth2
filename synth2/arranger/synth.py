"""Pitched instruments: presets and per-hit rendering.

Three engines:

* ``synth``: subtractive. Band-limited oscillator layers (saw, square, pulse, triangle,
  sine, organ ...) with unison detune and stereo spread, a sine sub, noise, an ADSR
  amplifier, a resonant time-varying filter with its own ADSR, key and velocity
  tracking, pre-filter drive (amp + cab style "guitar" crunch), post drive, pitch
  envelope, vibrato, glide and NES-style chip arpeggio.
* ``fm``: two-operator FM (+ an optional second modulator) for bells, glockenspiel,
  electric piano and music box.
* ``piano``: additive piano-ish tone (stretched partials with per-partial decay).

A *hit* is one note or a chord struck together; all its notes share the filter
envelope (one STFT filter pass per hit).
"""
from __future__ import annotations

import copy
import math
import zlib
from dataclasses import dataclass

import numpy as np

from .dsp import (adsr, eq_curve, fade_edges, fft_filter, filter_response, osc, pan_gains, saturate, sine,
                  stft_filter, stft_size)
from ..timing import midi_to_hz

SYNTH_DEFAULTS: dict = {
    "engine": "synth",
    "osc": [{"wave": "saw", "level": 1.0, "octave": 0, "semi": 0, "cents": 0.0, "voices": 1, "detune": 0.0,
             "spread": 0.0}],
    "sub": 0.0, "sub_octave": -1, "noise": 0.0,
    "attack": 0.003, "decay": 0.4, "sustain": 0.75, "release": 0.12, "attack_curve": 0.0,
    "filter": "lp", "slope": 24, "cutoff": 3000.0, "resonance": 0.1, "keytrack": 0.5, "env": 0.0,
    "f_attack": 0.001, "f_decay": 0.25, "f_sustain": 0.0, "f_release": 0.2,
    "vel_to_cutoff": 0.8, "vel_sens": 0.7,
    "drive": 0.0, "post_drive": 0.0,
    "pitch_env": 0.0, "pitch_decay": 0.03,
    "vibrato": 0.0, "vibrato_rate": 5.5, "vibrato_delay": 0.25,
    "glide": 0.0, "mono": False,
    "chip_arp": None, "chip_arp_rate": 40.0,
    "click": 0.0,
    "eq": None,
    "level": 1.0,
    "gate": 0.85,
    # fm engine
    "ratio": 1.0, "index": 1.0, "index_decay": 0.5, "index_sustain": 0.0,
    "ratio2": 0.0, "index2": 0.0, "index2_decay": 0.1, "chorus": 0.0,
    # piano engine
    "partials": 18, "inharmonicity": 0.00025, "hammer": 0.5, "brightness": 0.6, "ring": 1.0,
}

PRESETS: dict[str, dict] = {
    # ---------------------------------------------------------------- bass
    "bass_pick": {
        "role": "bass", "low": "E1", "sidechain": True,
        "description": "Picked electric-bass-like synth: saw + square, 24 dB filter pluck, a bit of drive. "
                       "Punchy and audible on small speakers.",
        "osc": [{"wave": "saw", "level": 0.8}, {"wave": "square", "level": 0.45, "octave": 0, "cents": 4}],
        "sub": 0.55, "sub_octave": 0, "attack": 0.002, "decay": 0.5, "sustain": 0.65, "release": 0.07,
        "cutoff": 550.0, "resonance": 0.18, "env": 2.6, "f_decay": 0.16, "f_sustain": 0.2, "keytrack": 0.4,
        "drive": 0.28, "post_drive": 0.15, "click": 0.25, "gate": 0.8,
        "eq": {"bands": [{"hz": 900, "db": 2.5, "q": 1.0}]},
    },
    "bass_slap": {
        "role": "bass", "low": "E1", "sidechain": True,
        "description": "Bouncy slap/pop bass: bright snap, pitch blip and a mid 'pop', short and percussive.",
        "osc": [{"wave": "saw", "level": 0.75}, {"wave": "pulse25", "level": 0.4}],
        "sub": 0.5, "sub_octave": 0, "attack": 0.001, "decay": 0.35, "sustain": 0.45, "release": 0.05,
        "cutoff": 900.0, "resonance": 0.25, "env": 3.0, "f_decay": 0.06, "f_sustain": 0.12, "keytrack": 0.35,
        "pitch_env": 0.9, "pitch_decay": 0.012, "drive": 0.35, "click": 0.6, "gate": 0.7,
        "eq": {"bands": [{"hz": 1300, "db": 4.0, "q": 1.2}, {"hz": 250, "db": -2.0, "q": 1.0}]},
    },
    "bass_sub": {
        "role": "bass", "low": "E1", "sidechain": True,
        "description": "Clean sine/triangle sub bass for the breakdown or under another bass.",
        "osc": [{"wave": "triangle", "level": 0.5}, {"wave": "sine", "level": 0.8}],
        "attack": 0.004, "decay": 0.6, "sustain": 0.85, "release": 0.09, "filter": "lp", "cutoff": 500.0,
        "env": 0.0, "keytrack": 0.0, "post_drive": 0.1, "gate": 0.9,
    },
    "bass_growl": {
        "role": "bass", "low": "E1", "sidechain": True,
        "description": "Detuned-saw rock bass with a growl (good for choruses and the chaos bridge).",
        "osc": [{"wave": "saw", "level": 0.7, "voices": 2, "detune": 14.0}, {"wave": "square", "level": 0.4,
                                                                             "cents": -6}],
        "sub": 0.35, "sub_octave": 0, "attack": 0.002, "decay": 0.6, "sustain": 0.8, "release": 0.08,
        "cutoff": 600.0, "resonance": 0.3, "env": 1.8, "f_decay": 0.22, "f_sustain": 0.35, "keytrack": 0.5,
        "drive": 0.5, "post_drive": 0.25, "gate": 0.85,
    },
    "bass_chip": {
        "role": "bass", "low": "E1", "sidechain": True,
        "description": "NES-style triangle bass with a square edge.",
        "osc": [{"wave": "triangle", "level": 1.0}, {"wave": "pulse25", "level": 0.18}],
        "attack": 0.001, "decay": 0.3, "sustain": 0.9, "release": 0.03, "filter": "none", "gate": 0.8,
    },
    # ---------------------------------------------------------------- guitars / chords
    "power_saw": {
        "role": "keys", "low": "E2", "center": "B3", "voicing": "power8", "sidechain": True, "width": 1.0,
        "description": "Crunchy distorted power-chord 'guitar' (detuned saws into a hard amp and a speaker-cab "
                       "filter). Use voicing power/power8/full; strum with chords patterns.",
        "osc": [{"wave": "saw", "level": 0.8, "voices": 3, "detune": 16.0, "spread": 0.7},
                {"wave": "square", "level": 0.35, "voices": 2, "detune": 9.0, "spread": 0.5}],
        "attack": 0.002, "decay": 1.6, "sustain": 0.7, "release": 0.07,
        "filter": "lp", "slope": 24, "cutoff": 3600.0, "resonance": 0.12, "env": 0.6, "f_decay": 0.12,
        "keytrack": 0.0, "drive": 0.8, "click": 0.15, "gate": 0.92,
        "eq": {"low_cut": 95, "high_cut": 7000, "bands": [{"hz": 380, "db": -3.5, "q": 1.0},
                                                          {"hz": 1700, "db": 3.5, "q": 1.2},
                                                          {"hz": 3200, "db": 1.5, "q": 1.5}]},
    },
    "gtr_mute": {
        "role": "keys", "low": "E2", "center": "B3", "voicing": "power", "sidechain": True,
        "description": "Palm-muted distorted chug: tight, dark and percussive (8th/16th chugs).",
        "osc": [{"wave": "saw", "level": 0.8, "voices": 3, "detune": 14.0, "spread": 0.6},
                {"wave": "square", "level": 0.3}],
        "attack": 0.001, "decay": 0.16, "sustain": 0.1, "release": 0.04,
        "filter": "lp", "cutoff": 900.0, "resonance": 0.2, "env": 1.6, "f_decay": 0.05, "keytrack": 0.2,
        "drive": 0.85, "click": 0.3, "gate": 0.6,
        "eq": {"low_cut": 90, "high_cut": 5000, "bands": [{"hz": 1500, "db": 3, "q": 1.0}]},
    },
    "gtr_clean": {
        "role": "keys", "low": "E2", "center": "E4", "voicing": "full", "sidechain": False,
        "description": "Bright clean strum 'guitar' (pluck-filtered saws, a touch of chorus).",
        "osc": [{"wave": "saw", "level": 0.7, "voices": 2, "detune": 7.0, "spread": 0.6},
                {"wave": "triangle", "level": 0.4}],
        "attack": 0.002, "decay": 0.9, "sustain": 0.15, "release": 0.15,
        "cutoff": 900.0, "resonance": 0.1, "env": 2.4, "f_decay": 0.14, "f_sustain": 0.2, "keytrack": 0.4,
        "drive": 0.1, "gate": 0.9, "eq": {"low_cut": 120, "high_cut": 9000},
    },
    "supersaw": {
        "role": "keys", "low": "C3", "center": "E4", "voicing": "triad", "sidechain": True, "width": 1.2,
        "description": "Wide 7-voice EDM supersaw for big chorus chords and stabs.",
        "osc": [{"wave": "saw", "level": 1.0, "voices": 7, "detune": 32.0, "spread": 1.0},
                {"wave": "saw", "level": 0.35, "octave": -1, "voices": 2, "detune": 10.0, "spread": 0.4}],
        "attack": 0.005, "decay": 1.0, "sustain": 0.8, "release": 0.25,
        "cutoff": 5200.0, "resonance": 0.1, "env": 0.8, "f_decay": 0.3, "keytrack": 0.2, "gate": 0.9,
        "eq": {"low_cut": 180, "high_cut": 12000},
    },
    "brass": {
        "role": "keys", "low": "C3", "center": "E4", "voicing": "triad", "sidechain": False,
        "description": "Synth brass stab: saws with a filter swell, punchy for hits and stabs.",
        "osc": [{"wave": "saw", "level": 0.9, "voices": 3, "detune": 10.0, "spread": 0.6}],
        "attack": 0.015, "decay": 0.5, "sustain": 0.7, "release": 0.14,
        "cutoff": 900.0, "resonance": 0.15, "env": 2.2, "f_attack": 0.04, "f_decay": 0.3, "f_sustain": 0.3,
        "keytrack": 0.5, "post_drive": 0.15, "gate": 0.8,
    },
    # ---------------------------------------------------------------- leads
    "chip_lead": {
        "role": "lead", "low": "A4", "sidechain": False,
        "description": "Chiptune 25 % pulse lead with delayed vibrato (the catchy 'pikopiko' riff voice).",
        "osc": [{"wave": "pulse25", "level": 0.9}, {"wave": "square", "level": 0.25, "octave": 1, "cents": 3}],
        "attack": 0.002, "decay": 0.25, "sustain": 0.8, "release": 0.06,
        "filter": "lp", "slope": 12, "cutoff": 5000.0, "resonance": 0.0, "keytrack": 0.2,
        "pitch_env": 0.4, "pitch_decay": 0.012, "vibrato": 0.18, "vibrato_rate": 6.2, "vibrato_delay": 0.18,
        "gate": 0.8,
    },
    "chip_square": {
        "role": "lead", "low": "A4", "sidechain": False,
        "description": "Hollow 50 % square lead, softer than chip_lead.",
        "osc": [{"wave": "square", "level": 0.85}],
        "attack": 0.003, "decay": 0.3, "sustain": 0.85, "release": 0.07,
        "filter": "lp", "slope": 12, "cutoff": 4200.0, "keytrack": 0.2, "vibrato": 0.15, "vibrato_rate": 5.8,
        "vibrato_delay": 0.22, "gate": 0.85,
    },
    "chip_arp": {
        "role": "keys", "low": "A4", "sidechain": False,
        "description": "Short 12.5 % pulse blips for fast 16th arpeggios (NES style).",
        "osc": [{"wave": "pulse12", "level": 0.9}],
        "attack": 0.001, "decay": 0.12, "sustain": 0.35, "release": 0.03,
        "filter": "lp", "slope": 12, "cutoff": 5500.0, "keytrack": 0.2, "gate": 0.55,
    },
    "saw_lead": {
        "role": "lead", "low": "A4", "sidechain": False,
        "description": "Bright detuned saw lead with glide and vibrato (soaring hooks).",
        "osc": [{"wave": "saw", "level": 0.8, "voices": 2, "detune": 9.0, "spread": 0.35},
                {"wave": "square", "level": 0.3, "octave": -1}],
        "attack": 0.004, "decay": 0.4, "sustain": 0.8, "release": 0.1,
        "cutoff": 2600.0, "resonance": 0.25, "env": 1.3, "f_decay": 0.25, "f_sustain": 0.3, "keytrack": 0.6,
        "glide": 0.035, "mono": True, "vibrato": 0.2, "vibrato_rate": 5.6, "vibrato_delay": 0.3,
        "drive": 0.2, "gate": 0.9,
    },
    "sync_lead": {
        "role": "lead", "low": "A4", "sidechain": False,
        "description": "Nasal, resonant square lead with a filter 'wah' on every note (quirky hooks).",
        "osc": [{"wave": "square", "level": 0.7}, {"wave": "saw", "level": 0.5, "cents": 7}],
        "attack": 0.002, "decay": 0.3, "sustain": 0.7, "release": 0.08,
        "cutoff": 1200.0, "resonance": 0.55, "env": 1.8, "f_decay": 0.12, "f_sustain": 0.25, "keytrack": 0.8,
        "slope": 12, "vibrato": 0.12, "vibrato_rate": 6.5, "gate": 0.8,
    },
    # ---------------------------------------------------------------- plucks / keys
    "pluck": {
        "role": "keys", "low": "A3", "center": "E5", "voicing": "triad", "sidechain": False,
        "description": "Snappy synth pluck (fast filter decay), great for off-beat stabs and arps.",
        "osc": [{"wave": "saw", "level": 0.8, "voices": 2, "detune": 8.0, "spread": 0.5},
                {"wave": "square", "level": 0.4, "octave": 1}],
        "attack": 0.001, "decay": 0.32, "sustain": 0.0, "release": 0.12,
        "cutoff": 500.0, "resonance": 0.22, "env": 3.8, "f_decay": 0.11, "keytrack": 0.6, "gate": 0.5,
    },
    "organ": {
        "role": "keys", "low": "C3", "center": "E4", "voicing": "triad", "sidechain": True,
        "description": "Drawbar organ (additive), slight drive, for retro chords.",
        "osc": [{"wave": "organ", "level": 1.0, "voices": 2, "detune": 4.0, "spread": 0.5}],
        "attack": 0.004, "decay": 0.2, "sustain": 1.0, "release": 0.06, "filter": "none", "post_drive": 0.15,
        "click": 0.15, "gate": 0.95,
    },
    "pad": {
        "role": "pad", "low": "C3", "center": "E4", "voicing": "triad", "sidechain": True, "width": 1.3,
        "description": "Warm wide pad: detuned saws, slow attack, gentle low-pass.",
        "osc": [{"wave": "saw", "level": 0.8, "voices": 5, "detune": 18.0, "spread": 1.0},
                {"wave": "triangle", "level": 0.4, "octave": -1}],
        "attack": 0.35, "decay": 1.5, "sustain": 0.85, "release": 0.9, "attack_curve": 0.4,
        "cutoff": 1500.0, "resonance": 0.1, "env": 0.5, "f_attack": 0.6, "f_decay": 1.2, "f_sustain": 0.5,
        "keytrack": 0.3, "gate": 1.0, "eq": {"low_cut": 150},
    },
    "pad_glass": {
        "role": "pad", "low": "C4", "center": "G4", "voicing": "triad", "sidechain": True, "width": 1.3,
        "description": "Airy, glassy pad (pulse + triangle, brighter), for breakdowns.",
        "osc": [{"wave": "pulse25", "level": 0.5, "voices": 4, "detune": 12.0, "spread": 1.0},
                {"wave": "triangle", "level": 0.7, "octave": 1, "voices": 2, "detune": 6.0, "spread": 0.8}],
        "attack": 0.5, "decay": 2.0, "sustain": 0.8, "release": 1.2, "attack_curve": 0.5,
        "cutoff": 3500.0, "resonance": 0.15, "env": 0.0, "keytrack": 0.2, "gate": 1.0,
        "eq": {"low_cut": 250},
    },
    # ---------------------------------------------------------------- fm / additive
    "bell": {
        "engine": "fm", "role": "keys", "low": "A4", "center": "E5", "voicing": "triad", "sidechain": False,
        "description": "FM bell: inharmonic, long ring. Melodies, counter-hooks, 'notification' pings.",
        "ratio": 3.5, "index": 2.4, "index_decay": 0.9, "ratio2": 1.0, "index2": 0.3, "index2_decay": 2.0,
        "attack": 0.001, "decay": 2.4, "sustain": 0.0, "release": 1.2, "gate": 0.9, "chorus": 0.25,
        "filter": "none",
    },
    "glock": {
        "engine": "fm", "role": "keys", "low": "A5", "center": "E6", "voicing": "triad", "sidechain": False,
        "description": "Glockenspiel / toy xylophone: bright, short, sparkly.",
        "ratio": 5.4, "index": 1.5, "index_decay": 0.12, "ratio2": 0.0, "index2": 0.0, "index2_decay": 0.08,
        "attack": 0.0005, "decay": 0.9, "sustain": 0.0, "release": 0.5, "gate": 0.6, "filter": "none",
    },
    "music_box": {
        "engine": "fm", "role": "keys", "low": "A5", "center": "E6", "voicing": "triad", "sidechain": False,
        "description": "Music box: delicate, slightly detuned tines.",
        "ratio": 4.0, "index": 0.9, "index_decay": 0.25, "ratio2": 7.0, "index2": 0.3, "index2_decay": 0.05,
        "attack": 0.0005, "decay": 1.3, "sustain": 0.0, "release": 0.8, "gate": 0.6, "chorus": 0.4,
        "filter": "none",
    },
    "epiano": {
        "engine": "fm", "role": "keys", "low": "C3", "center": "E4", "voicing": "seventh", "sidechain": False,
        "description": "FM electric piano (tine bark on hard hits), for verses and soft sections.",
        "ratio": 1.0, "index": 1.3, "index_decay": 0.5, "index_sustain": 0.15, "ratio2": 14.0, "index2": 0.5,
        "index2_decay": 0.03, "attack": 0.001, "decay": 1.8, "sustain": 0.25, "release": 0.25, "gate": 0.9,
        "chorus": 0.3, "filter": "none",
    },
    "acoustic": {
        "engine": "piano", "role": "keys", "low": "E2", "center": "E4", "voicing": "full", "sidechain": False,
        "description": "Strummed steel-string 'acoustic guitar' (plucked partials + body resonance). Use voicing "
                       "full/open and d/u strums in a chords pattern.",
        "attack": 0.001, "decay": 1.4, "sustain": 0.0, "release": 0.12, "gate": 0.9, "partials": 22,
        "inharmonicity": 0.00008, "hammer": 0.9, "brightness": 0.85, "filter": "none",
        "eq": {"low_cut": 85, "high_cut": 11000, "bands": [{"hz": 110, "db": 3.0, "q": 1.4},
                                                           {"hz": 220, "db": 2.0, "q": 1.5},
                                                           {"hz": 450, "db": -4.0, "q": 1.0},
                                                           {"hz": 2600, "db": 3.0, "q": 1.0}]},
    },
    "piano": {
        "engine": "piano", "role": "keys", "low": "C3", "center": "E4", "voicing": "triad", "sidechain": False,
        "description": "Additive pop piano (stretched partials, hammer brightness by velocity).",
        "attack": 0.001, "decay": 3.0, "sustain": 0.0, "release": 0.35, "gate": 0.9, "partials": 18,
        "inharmonicity": 0.00025, "hammer": 0.5, "brightness": 0.6, "filter": "none",
    },
}

# loudness calibration: a typical part of each preset at gain_db 0 already sits in a balanced pop mix
# (measured alone at 168 bpm: bass -16 LUFS, leads -16.5, rhythm guitars -17..-18, keys -18.5..-19,
# arps/bells -19.5..-20, pads -21; the default kit's kick alone is about -17.5)
CALIBRATION: dict[str, float] = {
    "bass_pick": 0.2409, "bass_slap": 0.3336, "bass_sub": 0.3187, "bass_growl": 0.2233, "bass_chip": 0.4342,
    "power_saw": 0.2033, "gtr_mute": 0.2699, "gtr_clean": 0.2451, "supersaw": 0.2392, "brass": 0.2389,
    "chip_lead": 0.1907, "chip_square": 0.1478, "chip_arp": 0.2068, "saw_lead": 0.1663, "sync_lead": 0.1321,
    "pluck": 0.3404, "organ": 0.1346, "pad": 0.1682, "pad_glass": 0.1357, "bell": 0.1207, "glock": 0.1681,
    "music_box": 0.1774, "epiano": 0.2676, "piano": 0.5674, "acoustic": 0.8385,
}

INSTRUMENT_DEFAULT_LOW = "C3"
PRESET_KEYS_NOT_SYNTH = ("role", "low", "center", "voicing", "sidechain", "width", "description")


def preset_params(preset: str, overrides: dict | None = None) -> dict:
    """Full synth parameter dict for a preset with ``overrides`` applied (``osc`` replaces the list)."""
    base = copy.deepcopy(SYNTH_DEFAULTS)
    pr = PRESETS[preset]
    for k, v in pr.items():
        if k not in PRESET_KEYS_NOT_SYNTH:
            base[k] = copy.deepcopy(v)
    if overrides:
        for k, v in overrides.items():
            base[k] = copy.deepcopy(v)
    # osc layer defaults
    layers = []
    for layer in base.get("osc") or []:
        d = {"wave": "saw", "level": 1.0, "octave": 0, "semi": 0, "cents": 0.0, "voices": 1, "detune": 0.0,
             "spread": 0.0}
        d.update(layer)
        layers.append(d)
    base["osc"] = layers
    base["_calibration"] = CALIBRATION.get(preset, 1.0)
    return base


@dataclass
class Hit:
    """One struck note or chord. Times in seconds relative to the render buffer start."""

    start: float  # seconds (absolute song time)
    gate: float  # seconds the key is held
    pitches: tuple[int, ...]
    velocity: float = 1.0
    glide_from: float | None = None  # MIDI pitch to glide from (mono legato)
    cut: float | None = None  # seconds after start where the sound is cut (mono / choke), with a short fade
    bright: float = 0.0  # extra filter octaves (section energy)
    seed: int = 0
    strum: float = 0.0  # seconds between successive chord notes (> 0 low-to-high, < 0 high-to-low)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed & 0x7FFFFFFF)


def hit_seed(*parts) -> int:
    return zlib.crc32("|".join(str(p) for p in parts).encode())


def tail_seconds(p: dict) -> float:
    """How long after the gate a note still sounds."""
    eng = p.get("engine", "synth")
    if eng in ("fm", "piano") or p.get("sustain", 1) <= 0.0:
        return max(float(p["release"]) * 1.6, 0.01)
    return float(p["release"]) * 1.6 + 0.005


def hit_length(p: dict, gate: float) -> float:
    """Seconds to render for a hit with this gate."""
    a, d, s = float(p["attack"]), float(p["decay"]), float(p["sustain"])
    natural = None
    if s <= 0.0:
        natural = a + d * 1.8  # ~ -47 dB by then
    held = gate + tail_seconds(p)
    return min(held, natural) if natural is not None else held


def _pitch_curve(p: dict, hit: Hit, n: int, sr: int, base_midi: float) -> np.ndarray | None:
    """Semitone offsets (n,) from the base pitch, or None when the pitch is static."""
    parts = []
    t = None
    if p.get("pitch_env"):
        t = np.arange(n, dtype=np.float32) / sr
        parts.append(float(p["pitch_env"]) * np.exp(-t / max(float(p["pitch_decay"]), 1e-4)))
    if p.get("vibrato"):
        t = np.arange(n, dtype=np.float32) / sr if t is None else t
        delay = float(p["vibrato_delay"])
        ramp = np.clip((t - delay) / 0.25, 0.0, 1.0)
        parts.append(float(p["vibrato"]) * ramp * np.sin(2 * math.pi * float(p["vibrato_rate"]) * t))
    if hit.glide_from is not None and p.get("glide"):
        t = np.arange(n, dtype=np.float32) / sr if t is None else t
        parts.append((hit.glide_from - base_midi) * np.exp(-t / max(float(p["glide"]) / 3.0, 1e-4)))
    arp = p.get("chip_arp")
    if arp:
        t = np.arange(n, dtype=np.float32) / sr if t is None else t
        steps = np.asarray(arp, dtype=np.float32)
        idx = (t * float(p["chip_arp_rate"])).astype(np.int64) % len(steps)
        parts.append(steps[idx])
    if not parts:
        return None
    out = parts[0].astype(np.float32)
    for x in parts[1:]:
        out = out + x
    return out


def _osc_layers(p: dict, hit: Hit, midi: float, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Stereo oscillator mix for one note (before envelope/filter)."""
    out = np.zeros((n, 2), np.float32)
    curve = _pitch_curve(p, hit, n, sr, midi)
    for layer in p["osc"]:
        voices = max(1, int(layer.get("voices", 1)))
        det = float(layer.get("detune", 0.0))
        spread = float(layer.get("spread", 0.0))
        base = midi + 12 * int(layer.get("octave", 0)) + float(layer.get("semi", 0)) + float(layer.get("cents", 0)) / 100
        lvl = float(layer.get("level", 1.0)) / math.sqrt(voices)
        wave = layer.get("wave", "saw")
        for v in range(voices):
            if voices > 1:
                x = (v / (voices - 1)) * 2 - 1  # -1..1
                cents = det * 0.5 * x * (0.75 + 0.25 * abs(x))
                pan = spread * x
            else:
                cents, pan = 0.0, 0.0
            f0 = float(midi_to_hz(base + cents / 100))
            if curve is None:
                sig = osc(wave, f0, n, sr, rng.uniform())
            else:
                sig = osc(wave, f0 * np.exp2(curve / 12.0), n, sr, rng.uniform())
            gl, gr = pan_gains(pan)
            out[:, 0] += sig * (lvl * gl)
            out[:, 1] += sig * (lvl * gr)
    if p.get("sub"):
        f0 = float(midi_to_hz(midi + 12 * int(p.get("sub_octave", -1))))
        if f0 < 38.0:  # never below ~D1: subsonic energy only eats headroom
            f0 *= 2.0 ** math.ceil(math.log2(38.0 / f0))
        sig = sine(f0 if curve is None else f0 * np.exp2(curve / 12.0), n, sr, 0.0) * float(p["sub"])
        out += sig[:, None]
    if p.get("noise"):
        out += rng.standard_normal((n, 2), dtype=np.float32) * (float(p["noise"]) * 0.3)
    return out


def _fm_note(p: dict, hit: Hit, midi: float, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    f0 = float(midi_to_hz(midi))
    vel = hit.velocity
    curve = _pitch_curve(p, hit, n, sr, midi)
    fmul = None if curve is None else np.exp2(curve / 12.0)
    idx1 = float(p["index"]) * (0.5 + 0.5 * vel) * (float(p["index_sustain"]) + (1 - float(p["index_sustain"]))
                                                     * np.exp(-t / max(float(p["index_decay"]), 1e-3)))
    out = np.zeros((n, 2), np.float32)
    chorus = float(p.get("chorus", 0.0))
    detunes = (-chorus * 0.05, chorus * 0.05) if chorus else (0.0,)
    for ch_i, dc in enumerate(detunes):
        f = f0 * (1 + dc / 12 * math.log(2))
        fa = f if fmul is None else f * fmul
        mod = sine(np.asarray(fa) * float(p["ratio"]) if fmul is not None else f * float(p["ratio"]), n, sr,
                   rng.uniform(0, 6.28))
        ph_mod = idx1 * mod
        if p.get("ratio2") and p.get("index2"):
            idx2 = float(p["index2"]) * vel * np.exp(-t / max(float(p["index2_decay"]), 1e-3))
            mod2 = sine(np.asarray(fa) * float(p["ratio2"]) if fmul is not None else f * float(p["ratio2"]), n, sr,
                        rng.uniform(0, 6.28))
            ph_mod = ph_mod + idx2 * mod2
        if fmul is None:
            carrier_ph = 2 * math.pi * f * t
        else:
            carrier_ph = np.cumsum(2 * math.pi * fa / sr).astype(np.float32)
        sig = np.sin(carrier_ph + ph_mod).astype(np.float32)
        if len(detunes) == 1:
            out += sig[:, None]
        else:
            out[:, ch_i] += sig * 0.9
            out[:, 1 - ch_i] += sig * 0.35
    if len(detunes) > 1:
        out *= 0.8
    return out


def _piano_note(p: dict, hit: Hit, midi: float, n: int, sr: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n, dtype=np.float32) / sr
    f0 = float(midi_to_hz(midi))
    B = float(p["inharmonicity"]) * (1 + max(0.0, midi - 60) / 24)
    vel = hit.velocity
    bright = float(p["brightness"]) * (0.5 + 0.7 * vel)
    out = np.zeros(n, np.float32)
    n_part = int(p["partials"])
    base_decay = float(p["decay"]) * (1.6 - min(max(midi - 36, 0), 60) / 60)  # low notes ring longer
    for k in range(1, n_part + 1):
        fk = f0 * k * math.sqrt(1 + B * k * k)
        if fk > 0.45 * sr:
            break
        amp = (1.0 / k ** (1.4 - 0.6 * bright)) * (1.0 if k == 1 else 0.9)
        # hammer position comb: weakens partials near multiples of 7
        amp *= abs(math.sin(math.pi * k / 7.3)) * 0.6 + 0.4
        dk = base_decay / (1 + 0.25 * (k - 1) ** 1.1)
        env = 0.65 * np.exp(-t / (dk * 0.18)) + 0.35 * np.exp(-t / dk)
        # two strings slightly detuned -> gentle beating
        det = 1 + (0.0007 + 0.0002 * k) * (rng.uniform() - 0.3)
        ph = rng.uniform(0, 6.28)
        s = np.sin(2 * math.pi * fk * t + ph) + np.sin(2 * math.pi * fk * det * t + ph * 1.3)
        out += (0.5 * amp) * env * s.astype(np.float32)
    # hammer thump
    m = min(n, int(0.03 * sr))
    if m > 4:
        thump = rng.standard_normal(m).astype(np.float32) * np.exp(-np.arange(m) / (0.004 * sr))
        out[:m] += thump * 0.05 * float(p["hammer"]) * vel
    return np.stack([out, out], axis=1) * 0.5


def render_hit(p: dict, hit: Hit, sr: int, energy_bright: float = 0.0) -> np.ndarray:
    """Render one hit (all its pitches) to a stereo float32 array starting at the hit onset."""
    length = hit_length(p, hit.gate)
    if hit.cut is not None:
        length = min(length, hit.cut + 0.004)
    n = max(16, int(round(length * sr)))
    rng = _rng(hit.seed)
    eng = p.get("engine", "synth")
    vel = max(0.0, min(1.5, hit.velocity))
    amp_vel = (1.0 - float(p["vel_sens"])) + float(p["vel_sens"]) * vel
    # amplitude envelope
    if eng == "synth":
        env = adsr(n, sr, float(p["attack"]), float(p["decay"]), float(p["sustain"]), float(p["release"]),
                   hit.gate, float(p.get("attack_curve", 0.0)))
    else:
        env = adsr(n, sr, float(p["attack"]), float(p["decay"]), float(p["sustain"]), float(p["release"]),
                   hit.gate)
    sig = np.zeros((n, 2), np.float32)
    order = sorted(hit.pitches, reverse=hit.strum < 0)
    for k, midi in enumerate(order):
        off = min(n - 16, int(abs(hit.strum) * k * sr)) if len(order) > 1 and hit.strum else 0
        m = n - off
        if eng == "fm":
            note = _fm_note(p, hit, float(midi), m, sr, rng)
        elif eng == "piano":
            note = _piano_note(p, hit, float(midi), m, sr, rng)
        else:
            note = _osc_layers(p, hit, float(midi), m, sr, rng)
        if off:
            fade_edges(note, sr, 0.001, 0.0)
        sig[off:] += note
    if len(hit.pitches) > 1:
        sig *= 1.0 / math.sqrt(len(hit.pitches))
    sig *= env[:, None]
    if p.get("click"):
        m = min(n, int(0.006 * sr))
        clk = rng.standard_normal(m).astype(np.float32) * np.exp(-np.arange(m, dtype=np.float32) / (0.0012 * sr))
        sig[:m] += (clk * float(p["click"]) * 0.5 * vel)[:, None]
    sig = saturate(sig, float(p.get("drive", 0.0)))
    # filter (+ static eq) in one STFT pass
    ftype = p.get("filter", "lp")
    eq = p.get("eq")
    if ftype not in (None, "none", "off") or eq:
        mean_pitch = sum(hit.pitches) / len(hit.pitches)
        base = float(p["cutoff"]) * 2 ** (float(p["keytrack"]) * (mean_pitch - 60) / 12)
        base *= 2 ** (float(p["vel_to_cutoff"]) * (vel - 1.0) + energy_bright + hit.bright)
        env_oct = float(p.get("env", 0.0))
        fa, fd, fs, fr = (float(p["f_attack"]), float(p["f_decay"]), float(p["f_sustain"]), float(p["f_release"]))
        res = float(p.get("resonance", 0.0))
        slope = int(p.get("slope", 24))
        nyq = 0.47 * sr
        half_w = stft_size(sr) / 2 / sr
        eq_cache: dict = {}

        def mask(times, freqs):
            if "eq" not in eq_cache:
                eq_cache["eq"] = eq_curve(freqs, eq)
            m = None
            if ftype not in (None, "none", "off"):
                # a frame whose window already reaches the attack gets the opened filter (max of the envelope
                # at the frame centre and half a window later), so the zero-phase STFT filter does not dull the
                # first milliseconds of a pluck; decays are unaffected (the envelope falls)
                tt = np.maximum(times, 0.0)
                if env_oct:
                    fenv = np.maximum(_adsr_at(tt, fa, fd, fs, fr, hit.gate),
                                      _adsr_at(np.maximum(times + half_w, 0.0), fa, fd, fs, fr, hit.gate))
                    fc = base * np.exp2(env_oct * fenv)
                else:
                    fc = np.full(len(tt), base)
                fc = np.clip(fc, 30.0, nyq)[:, None]
                m = filter_response(freqs, fc, res, ftype, slope)
            if eq_cache["eq"] is not None:
                m = eq_cache["eq"][None, :] * (m if m is not None else np.ones((len(times), 1)))
            return m

        if env_oct == 0.0:
            # static filter: one FFT pass over the whole hit is ~3x cheaper than the STFT
            sig = fft_filter(sig, sr, lambda freqs: mask(np.zeros(1), freqs)[0])
        else:
            sig = stft_filter(sig, sr, mask)
    sig = saturate(sig, float(p.get("post_drive", 0.0)))
    sig *= float(p.get("level", 1.0)) * float(p.get("_calibration", 1.0)) * amp_vel
    if hit.cut is not None and hit.cut * sr < n:
        c = int(hit.cut * sr)
        f = min(int(0.004 * sr), c)
        sig[c:] = 0.0
        if f > 1:
            sig[c - f:c] *= np.linspace(1, 0, f, dtype=np.float32)[:, None]
    return fade_edges(sig, sr, 0.0005 if p["attack"] < 0.002 else 0.001, 0.004)


def _adsr_at(t: np.ndarray, a: float, d: float, s: float, r: float, gate: float) -> np.ndarray:
    """ADSR value at arbitrary times (seconds) - used for the filter envelope per STFT frame."""
    a = max(a, 1e-4)
    d = max(d, 1e-3)
    out = np.where(t < a, t / a, s + (1 - s) * np.exp(-(t - a) * (3.0 / d)))
    if gate < float(t.max(initial=0.0)):
        g = gate
        lvl = g / a if g < a else s + (1 - s) * math.exp(-(g - a) * (3.0 / d))
        out = np.where(t >= g, lvl * np.exp(-(t - g) * (3.0 / max(r, 1e-3))), out)
    return out
