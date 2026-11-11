"""Backing-track arranger: a JSON arrangement -> a punchy synthesised band, as mix-ready stems.

The arrangement (``project.backing``) describes instruments (preset synths, a drum kit,
an FX rack), reusable patterns (16th-step drum grids, chord-relative bass lines, chord
strums and stabs, arpeggios, explicit riffs), sections with chords and per-section part
assignments, fills, one-off events (FX hits, glitch stutters, tape stops ...) and stops.
See ``FORMAT.md`` in this package for the full format with an example.

Public API::

    from synth2.arranger import render_backing, kick_times, validate_arrangement

    stems = render_backing(project, 44100, t0, t1)  # list[BackingStem] covering [t0, t1)
    kicks = kick_times(project)                      # np.ndarray of kick-hit times (s) for sidechain ducking
    problems = validate_arrangement(arrangement)     # ["error: ...", "warning: ..."]

Everything is pure numpy: band-limited wavetable oscillators, STFT-domain time-varying
resonant filters, synthesised drums and FX.  Any instrument can play samples instead
(``"sound"``: an installed sound set, a SoundFont 2 or SFZ file, a folder of WAV files);
see :mod:`.sampler`, :mod:`.soundbank` and "Sampled sounds" in ``FORMAT.md``.
``sound_report(project)`` says which sounds play on this machine and ``list_sounds()``
what is installed.
"""
from __future__ import annotations

from .arrangement import (ARP_MODES, DELAY_TIMES, FORMAT_NAME, KIT_PRESETS, PATTERN_KINDS, ROLES, ArrangementError,
                          compile_arrangement, parse_riff, validate_arrangement)
from . import gm, soundbank
from .drums import ALL_VOICES, DRUM_ALIASES, DRUM_DEFAULTS, DRUM_GROUPS, DRUM_VOICES
from .dsp import WAVES
from .fx import FX_SOUNDS, PROCESS_EFFECTS
from .render import BackingStem, kick_times, render_backing, song_length_seconds, sound_report
from .synth import PRESETS
from .theory import VOICINGS, parse_chord

__all__ = [
    "render_backing", "kick_times", "validate_arrangement", "BackingStem", "ArrangementError", "compile_arrangement",
    "summarize", "reference", "example_arrangement", "PRESETS", "DRUM_VOICES", "FX_SOUNDS", "PROCESS_EFFECTS",
    "parse_chord", "parse_riff", "song_length_seconds", "FORMAT_NAME", "sound_report", "list_sounds",
]

SYNTH_PARAM_DOCS: dict[str, str] = {
    "osc": "oscillator layers: list of {wave (" + ", ".join(WAVES) + "), level, octave, semi, cents, voices "
           "(unison count), detune (cents spread across the unison), spread (0..1 stereo width of the unison)}",
    "sub": "sine sub-oscillator level 0..1 (never below ~38 Hz)",
    "sub_octave": "octave of the sub sine relative to the note (default -1; bass presets use 0)",
    "noise": "white noise level through the filter",
    "attack": "amp attack, s", "decay": "amp decay (time to ~95 % of the way to sustain), s",
    "sustain": "amp sustain level 0..1 (0 = plucked/percussive)", "release": "amp release, s",
    "attack_curve": "0 = linear attack, >0 = faster start (swells)",
    "filter": "lp | hp | bp | none", "slope": "12 or 24 (dB/oct, lp/hp)", "cutoff": "filter cutoff at middle C, Hz",
    "resonance": "0..1 (0.5+ squelchy)", "keytrack": "0..1 how much the cutoff follows the note pitch",
    "env": "filter envelope amount in octaves (brightness at the attack)",
    "f_attack": "filter env attack, s", "f_decay": "filter env decay, s", "f_sustain": "filter env sustain 0..1",
    "f_release": "filter env release, s", "vel_to_cutoff": "octaves of cutoff per unit of velocity",
    "vel_sens": "0..1 how much velocity changes loudness",
    "drive": "0..1 pre-filter amp distortion (the 'guitar' crunch; the filter+eq act as the speaker cab)",
    "post_drive": "0..1 saturation after the filter (warmth/grit)",
    "pitch_env": "semitones of pitch blip at the note start (decays over pitch_decay s)",
    "pitch_decay": "s", "vibrato": "vibrato depth, semitones", "vibrato_rate": "Hz",
    "vibrato_delay": "s before the vibrato fades in", "glide": "portamento time between legato notes (mono), s",
    "mono": "true = monophonic (each note cuts the previous)", "chip_arp": "NES-style fast arpeggio: list of "
    "semitone offsets cycled within each note, e.g. [0, 4, 7]", "chip_arp_rate": "chip_arp steps per second",
    "click": "attack transient (pick/key click) level", "eq": "static EQ {low_cut, high_cut, bands: [{hz, db, q}], "
    "low_shelf: {hz, db}, high_shelf: {hz, db}}", "level": "output trim (linear)",
    "gate": "0..1 how much of a long note's length the key is held (patterns also have a gate)",
    "ratio": "fm: modulator/carrier frequency ratio (inharmonic = bell)", "index": "fm: modulation index",
    "index_decay": "fm: s", "index_sustain": "fm: 0..1", "ratio2": "fm: second modulator ratio",
    "index2": "fm: second modulator index", "index2_decay": "fm: s", "chorus": "fm: 0..1 stereo detune",
    "partials": "piano: number of partials", "inharmonicity": "piano: string stiffness",
    "hammer": "piano: hammer thump", "brightness": "piano: 0..1 tone", "ring": "piano: unused (reserved)",
    "engine": "synth | fm | piano (set by the preset)",
}


def reference() -> dict:
    """The arranger's catalogue: presets, kit voices, FX, pattern grammar (the AI's manual)."""
    presets = {}
    for name, p in PRESETS.items():
        presets[name] = {"description": p.get("description", ""), "role": p.get("role"),
                         "engine": p.get("engine", "synth"), "register_low": p.get("low"),
                         "center": p.get("center"), "voicing": p.get("voicing"),
                         "sidechain": p.get("sidechain", False)}
    return {
        "presets": presets,
        "synth_parameters": SYNTH_PARAM_DOCS,
        "drum_voices": {v: dict(p) for v, p in DRUM_DEFAULTS.items()},
        "drum_aliases": dict(DRUM_ALIASES),
        "drum_stem_groups": {g: list(v) for g, v in DRUM_GROUPS.items()},
        "drum_stems_each": {v: gm.stem_of(v) for v in ALL_VOICES},
        "drum_split": "kit 'split': true or 'groups' = kick / snare / hats / cymbals stems (default for a "
                      "synthesised kit), 'each' = one stem per kind of drum (default for a kit with a 'sound'), "
                      "false or 'kit' = one stem",
        "percussion_voices": {v: {"keys": list(spec[0]), "stem": spec[1], "synth_stand_in": spec[2]}
                              for v, spec in gm.EXTRA_VOICES.items()},
        "sampled_sounds": {
            "how": "any instrument may name a sampled sound: \"sound\": \"<installed set id>\" | \"<path to "
                   ".sf2 / .sfz / WAV / folder>\" | {set | path, preset, options...} | [layer, layer] (layers "
                   "play together). 'preset' keeps naming the synth preset, which plays when the sound is not "
                   "installed (reported as a warning, never an error); without a sound preset the General MIDI "
                   "program that fits the synth preset is used. Kits: {\"type\": \"drums\", \"sound\": ...} "
                   "with per-drum 'voices': {kick: {tune (semitones), gain_db, pan, note (kit key), sample (a "
                   "file), length (s)}}. See list_sounds for what is installed.",
            "options": dict(soundbank.SOUND_OPTION_DOCS),
            "general_midi_programs": {str(i): n for i, n in enumerate(gm.GM_PROGRAMS)},
            "preset_programs": {k: list(v) for k, v in gm.PRESET_PROGRAMS.items()},
            "drum_keys": {v: list(k) for v, k in gm.DRUM_KEYS.items()},
        },
        "kits": list(KIT_PRESETS),
        "fx_sounds": {k: {"default_beats": v[0], "default_pitch": v[1], "description": v[2]}
                      for k, v in FX_SOUNDS.items()},
        "process_effects": dict(PROCESS_EFFECTS),
        "pattern_kinds": {
            "drums": "lanes {voice: steps}; step chars: . rest, x hit (0.8), X accent (1.0), o ghost (0.5), "
                     "g soft (0.4), 1-9 velocity, r roll (two 32nds), R triplet roll, f flam. Lanes loop "
                     "independently (\"hat\": \"x.X.\" repeats every beat).",
            "line": "steps: chord-relative degree per step: R (chord bass, the slash note for Am/G), 1 (root), "
                    "2 3 4 5 6 7 (chord tones from the chord, others from the key), 8 (octave), 9 10 12, b/# "
                    "prefix, ' octave up / , octave down suffix (spaced form), compact aliases L=R, (octave down) "
                    "H=5' T=3' O=8', x = dead/ghost note, - tie, . rest; ! accent, ? ghost suffix.",
            "chords": "rhythm: x hit, X accent, o soft, d/u down/up strum, m palm-muted, - tie, . rest; "
                      "voicing (" + ", ".join(VOICINGS) + "), center, low, strum (s).",
            "arp": "rhythm (default every step), mode (" + ", ".join(ARP_MODES) + "), octaves, order (degree "
                   "list for mode 'order'), low.",
            "notes": "explicit riff: notes [[beat, pitch, beats, velocity], ...] or score 'A4 e C5 e r q E5+A5 q.' "
                     "(pitch/duration pairs, r rest, + chord, ! accent, ? ghost); relative: true for degrees.",
        },
        "pattern_common": "grid (steps per bar: 16 default, 8, 12, 24, 32), bars, swing (0..0.75 fraction of a "
                          "step that off-beats are late; 0.33 = shuffle), gate (0..1.5 of a step), velocity "
                          "(digit string '9.5.7.' or list), transpose, octave",
        "voicings": list(VOICINGS), "arp_modes": list(ARP_MODES), "delay_times": list(DELAY_TIMES),
        "roles": list(ROLES), "pattern_kind_list": list(PATTERN_KINDS),
        "timing": "bars are absolute and 1-based (bar 1 = beat 0); 'beat' inside a bar is 1-based (2.5 = the and "
                  "of 2); tempo comes from the project.",
    }


def summarize(arrangement: dict, project=None) -> dict:
    """Compact overview of an arrangement (sections with bars/time, instruments, patterns, event counts)."""
    key = project.key if project is not None else None
    bpb = project.beats_per_bar() if project is not None else 4.0
    comp = compile_arrangement(arrangement, key, bpb, strict=False)
    b2s = project.tempo.beat_to_sec if project is not None else (lambda b: b * 0.5)
    secs = []
    for s in comp.sections:
        b0 = (s.start_bar - 1) * bpb
        secs.append({
            "name": s.name, "bars": f"{s.start_bar}-{s.start_bar + s.bars - 1}", "energy": s.energy,
            "start_s": round(b2s(b0), 3), "end_s": round(b2s(b0 + s.bars * bpb), 3),
            "chords": " | ".join(" ".join(c.name if c else "N.C." for _, c in bar) for bar in s.chords[:16]),
            "parts": {k: [g["pattern"] for g in v] for k, v in s.parts.items()},
        })
    insts = {}
    for name, inst in comp.instruments.items():
        insts[name] = {"type": inst.kind, "preset": inst.preset, "role": inst.role, "gain_db": inst.gain_db,
                       "pan": inst.pan, "sidechain": inst.sidechain, "mute": inst.mute,
                       **({"sound": [soundbank.describe(layer) for layer in inst.sound]} if inst.sound else {}),
                       "events": len(comp.drum_events.get(name, [])) + len(comp.note_events.get(name, []))
                       + len(comp.fx_events.get(name, []))}
    kicks = sum(1 for evs in comp.drum_events.values() for e in evs if e.voice == "kick")
    problems = list(comp.warnings)
    if any(inst.sound for inst in comp.instruments.values()):
        dirs = soundbank.project_dirs(project, arrangement)
        for name, inst in comp.instruments.items():
            for layer in inst.sound or []:
                res = soundbank.resolve(layer, drums=inst.kind == "drums", synth_preset=inst.preset, kit=inst.kit,
                                        kit_params=inst.params if inst.kind == "drums" else None, low=inst.low,
                                        dirs=dirs, chords=inst.role not in ("bass", "lead"))
                if res.instrument is None:
                    problems.append(f"warning: instruments.{name}.sound: {res.error}; the synthesised "
                                    f"{'kit' if inst.kind == 'drums' else 'preset'} plays instead")
    return {"sections": secs, "instruments": insts, "patterns": sorted((arrangement.get("patterns") or {})),
            "length_bars": round(comp.end_beat / bpb, 3), "length_s": round(b2s(comp.end_beat), 3),
            "kick_hits": kicks, "processing_events": len(comp.processes), "stops": len(comp.stops),
            "problems": problems}


def list_sounds(detail: bool = False) -> list[dict]:
    """Installed sound sets and what each can play (presets and kits; counts only unless
    ``detail``)."""
    out = []
    for entry in soundbank.catalogue():
        if not detail:
            entry = dict(entry, presets=len(entry["presets"]), kits=[k["name"] for k in entry["kits"]][:40])
        out.append(entry)
    return out


def example_arrangement() -> dict:
    """A complete 4-bar example (A minor) using every pattern kind, an FX event, a fill and a stop."""
    return {
        "format": FORMAT_NAME, "version": 1, "key": "A minor", "swing": 0.0,
        "instruments": {
            "drums": {"type": "drums", "kit": "pop", "gain_db": 0},
            "bass": {"preset": "bass_slap", "gain_db": 0},
            "gtr": {"preset": "power_saw", "gain_db": -1, "pan": -0.3},
            "lead": {"preset": "chip_lead", "gain_db": -1, "pan": 0.1, "delay": {"time": "8d", "feedback": 0.3,
                                                                                  "mix": 0.18}},
            "arp": {"preset": "chip_arp", "gain_db": -2, "pan": 0.35},
            "fx": {"type": "fx", "gain_db": 0},
        },
        "patterns": {
            "beat": {"kind": "drums", "lanes": {
                "kick": "x.....x...x.....", "snare": "....X.......X...", "clap": "....x.......x...",
                "hat": "x.x.x.x.x.x.x.X.", "ohat": "..............x."}},
            "fill": {"kind": "drums", "lanes": {"snare": "....x.x.xxxxXXXX", "kick": "x.......x.......",
                                                "tom1": "........x.x....."}},
            "bounce": {"kind": "line", "steps": "R.R8.R5.R.Lx8.5."},
            "chug": {"kind": "chords", "rhythm": "x--.x-x-.x--x-x.", "voicing": "power8"},
            "riff": {"kind": "notes", "bars": 2, "score": "A5 e C6 e E6 e A5 e G5 q E5 e D5 e  "
                                                          "C5 e D5 e E5 q. r e E5! e G5 e"},
            "sparkle": {"kind": "arp", "mode": "updown", "octaves": 2, "rhythm": "x.xxx.xxx.xxx.xx", "gate": 0.5},
        },
        "sections": [
            {"name": "Demo", "start_bar": 1, "bars": 4, "chords": ["Am", "F", "G", "E7"], "energy": 0.9,
             "crash": True,
             "parts": {"drums": "beat", "bass": "bounce", "gtr": "chug", "lead": "riff", "arp": "sparkle"},
             "fills": [{"part": "drums", "bar": 4, "beat": 3, "pattern": "fill"}]},
        ],
        "events": [
            {"type": "fx", "sound": "laser", "bar": 2, "beat": 4.5, "pitch": "A6"},
            {"type": "stutter", "bar": 3, "beat": 4, "beats": 1, "slice": 0.25, "parts": ["gtr", "lead"]},
        ],
        "stops": [{"bar": 4, "beat": 4.5, "beats": 0.5, "parts": "all", "except": ["drums"]}],
    }
