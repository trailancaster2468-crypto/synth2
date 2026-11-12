"""Build, mix and render SELECT ALL INVADERS (セレクト・オール・インベーダー).

The project is put together through the studio's tool layer, the way an AI producer would drive it
(``synth2.ai.tools.Session`` + ``call_tool``), and rendered by ``render_project`` alone:

1. ``new_project`` from the score text in ``vocals.s2s`` (the hand-written vocal tracks, in E minor),
2. ``vocals.apply_vocal_production``: AI Double, Harmony, Crowd 2 / 3, the shouts' doubles, the aliens' fan voice,
3. ``set_backing`` with ``backing.json`` (the arranger's band), then ``set_backing(merge=true)`` for the
   section tweaks of the mix (:data:`BAND_TWEAKS`),
4. ``vocals.apply_arrangement``: every kind of sound on a track of its own (65 vocal tracks), each with its
   role, its cast and its seed,
5. ``update_track`` for every vocal fader, pan and reverb send (:data:`VOCAL_MIX`),
6. ``update_project`` for the master: loudness, delay, room, master EQ, band balance and the per-track /
   per-stem overrides (:func:`master_settings`),
7. ``save_project``, then ``render_project`` at 44.1 kHz.

Engines: the score says ``engine: auto``. Where the neural engine's models are installed (``synth2 models
install``) the song is sung by the voicebank, the lead through the chain of its role, its doubles made from
its own take, the ad-libs as detached hits. Where they are not, the formant engine sings the same tracks and
the mixer's channel strip takes the place of the roles' chains. ``--engine formant`` forces the fallback.

Band: ``backing.json`` names sampled sounds for the drums (one stem per kind of drum), the pick and slap
basses, the guitars, the plucks, bells, music box and pad (``synth2 sounds install generaluser-gs
karoryfer-growlybass``). An instrument whose set is not installed plays its synthesised preset instead; the
render report says which did, and carries the credits of the voicebank and of the sound sets used.

The balance dial (``--balance``, :data:`BALANCES`): the same song and takes at four balances of the vocals
against the band. ``A`` is the sampled band as it was first mixed, ``S`` the synthesised band, ``B`` (the
default) the sampled band with its low end, punch, section dynamics and width restored, the vocal stack
without its low mids and the vocals 2 dB lower against the band than in A, ``C`` the same with the vocals
4 dB lower. Without the sound sets, B and C play the synthesised band with the vocals lower.

The mix: the lead alone in the centre; its double close around it; the harmony on both sides; answers,
ad-libs, the aliens and the crowd each in a place of their own, further back, ducked under the lead where
they overlap it. The values are documented next to the tables below; every one of them came from a
measurement (levels against the lead, 1-5 kHz balance, section loudness, limiter activity, recognisers on
the stems), none from listening.

It writes ``out/select_all_invaders/``: ``select_all_invaders.wav`` (24-bit), ``select_all_invaders.s2p.json``,
``select_all_invaders.mp3`` (320 kbps, when ffmpeg is installed; trimmed so its decoded true peak stays
under the ceiling) and ``select_all_invaders.report.json`` (the render report with engines, credits and
warnings, plus the analysis printed at the end: loudness per section, true peak, the vocal-to-band balance
per section, the low-end balance, the long-term spectrum slope and, with ``--pitch``, the pitch accuracy of
every vocal track). With ``--stems`` it also writes ``stems/``: one 24-bit WAV per vocal track and band stem
(post-fader, pre-reverb), about a hundred files of 50 MB each for the whole song.

Usage (from the repository root)::

    python examples/select_all_invaders/build.py                          # the whole song
    python examples/select_all_invaders/build.py --start-bar 33 --end-bar 48   # a quick section render
    python examples/select_all_invaders/build.py --engine formant         # without the neural models
    python examples/select_all_invaders/build.py --stems                  # with the stems (5 GB for the song)
    python examples/select_all_invaders/build.py --balance C              # band-forward (A, S, B or C)

A section render goes to ``out/select_all_invaders/preview/``. After a full render the mix levels
(band gain, compressor thresholds, master gain) are saved in ``mix_levels.json``, and a section render
of the same mix reuses them, so it is mastered like that part of the full song.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
try:
    import synth2  # noqa: F401
except ImportError:  # run from a checkout without installing
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from synth2.ai.tools import Session, call_tool  # noqa: E402
from synth2.analysis import pitch_accuracy  # noqa: E402
from synth2.engine import mixing, render_project  # noqa: E402
from synth2.engine import renderer as _renderer  # noqa: E402
from synth2.io.wav import write_wav  # noqa: E402

import vocals  # noqa: E402  (examples/select_all_invaders/vocals.py)

NAME = "select_all_invaders"
OUT = ROOT / "out" / NAME
BEATS_PER_BAR = 4

# SPEC.md: (section, first bar, last bar)
SECTIONS = [
    ("Intro", 1, 8), ("Verse 1", 9, 24), ("Pre-chorus 1", 25, 32), ("Chorus 1", 33, 48),
    ("Post-chorus", 49, 52), ("Verse 2", 53, 68), ("Pre-chorus 2", 69, 76), ("Chorus 2", 77, 92),
    ("Bridge", 93, 108), ("Breakdown", 109, 112), ("Final chorus", 113, 128), ("Outro", 129, 134),
]

# ------------------------------------------------------------------------------------------- the vocal mix
# In-channel effects (``mix_overrides`` ``fx``, docs/NEURAL.md): tone, the duck under the lead, one repeat
# ("slap"), a small room that puts a voice further back than the song's room does.
_FAR = {"decay": 0.30, "predelay": 0.030, "level_db": -12.0}
FX = {
    # the harmony's other side: the harmony's own tone and room (a double has neither by itself)
    "harmony copy": {"eq": [["peak", 2800, -3.0, 0.9]], "room": {"decay": 0.28, "predelay": 0.030, "level_db": -13.0}},
    # runs of one sound on eighth notes: a repeat 95 to 110 ms later would land between the hits and blur them
    "click": {"slap": None},
    # the stutter is the start of the lead's own word: the lead's tone, no band limit, no duck, no repeat
    "stutter": {"eq": None, "duck_db": None, "slap": None},
    "alien answer": {"duck_db": 1.5},
    "alien backing": {"eq": [["highshelf", 3800, -4.0, 0.7]], "duck_db": 3.0, "room": _FAR},
    "alien adlib": {"eq": [["hp", 260, 0, 0.7071, 2], ["lp", 4600, 0, 0.7071, 2]], "duck_db": 2.0,
                    "slap": {"time": 0.110, "level_db": -8.0, "lp": 2600}},
    "alien click": {"eq": [["hp", 260, 0, 0.7071, 2], ["lp", 4600, 0, 0.7071, 2]], "duck_db": 2.0},
    "alien laugh": {"eq": [["hp", 260, 0, 0.7071, 2]], "duck_db": 2.0},
    "alien chant": {"eq": [["highshelf", 4200, -3.0, 0.7]], "duck_db": 2.5, "room": _FAR},
    "crowd": {"eq": [["hp", 170, 0, 0.7071, 2], ["highshelf", 3800, -4.0, 0.7]], "duck_db": 3.0,
              "room": {"decay": 0.38, "predelay": 0.038, "level_db": -9.0}},
    "crowd shout": {"eq": [["hp", 220, 0, 0.7071, 2], ["highshelf", 5000, -2.0, 0.7]], "duck_db": 1.5,
                    "room": {"decay": 0.38, "predelay": 0.038, "level_db": -11.0}},
}
# The aliens' low voice: less 280 Hz (it was the main source of 200-400 Hz mud), more 2.6 kHz for its
# consonants, and a transient limit (its peaks stand far over its loud level and set off the master limiter).
ALIEN = {"crest_db": 8.0, "eq_add": [["peak", 280, -3.0, 1.0], ["peak", 2600, 2.0, 1.0]]}
# the aliens' backing parts (the final hook two octaves under the lead, the unison): out of the lead's
# 300 Hz body and 2.8 kHz bite
ALIEN_UNDER = {"crest_db": 8.0, "eq_add": [["peak", 300, -2.5, 1.0], ["peak", 2800, -3.0, 1.0]]}
CROWD = {"crest_db": 12.0}  # the shouts are peaky

# Every vocal channel: (kind, fader dB, pan, reverb send, delay send, strip overrides, in-channel effects).
# The faders are set for the neural engine, whose voices have one level per voice: its detached hits, hooks and
# chants come out 2 to 4 dB under its lines, and their faders make that up (measured against the lead, stem by
# stem).
# The lead is the anchor at 0 dB, dead centre, and nothing else is in the centre. The reverb send of a lead
# channel is scaled by its dial setting (crisp: x 0.6), the delay send is into the ducked dotted-eighth
# ping-pong (1 = the lead's). Strip overrides shape the mixer's channel strip (the tracks without a role; with
# the formant engine, every track). Section channels of the lead: 2.5 dB lower in the thin half-time verses,
# 1 dB lower in the pre-choruses so that the hook entries lift, 2.5 dB up over the crushed bridge band and in
# the final chorus, lower and in more room for the breakdown.
VOCAL_MIX: dict[str, tuple] = {
    "AI":                  ("lead", 0.0, 0.0, 0.25, 1.0, None, None),
    "AI Verses":           ("lead", -2.5, 0.0, 0.25, 1.0, None, None),
    "AI Pre":              ("lead", -1.0, 0.0, 0.25, 1.0, None, None),
    "AI Bridge":           ("lead", 2.5, 0.0, 0.25, 1.0, None, None),
    "AI Breakdown":        ("lead", -2.5, 0.0, 0.40, 1.4, None, None),
    "AI Final":            ("lead", 2.5, 0.0, 0.25, 1.0, None, None),
    # the double: one copy each side, close to the lead, about 10 dB under it together
    "AI Double":           ("double", -12.5, -0.28, 0.40, 0.4, None, None),
    "AI Double R":         ("double", -12.5, 0.28, 0.40, 0.4, None, None),
    # the harmony: its take right, its copy left, outside the doubles, more room
    "Harmony":             ("harmony", -8.6, 0.70, 0.65, 0.5, None, None),
    "Harmony L":           ("harmony", -10.1, -0.70, 0.65, 0.5, None, FX["harmony copy"]),
    "Harmony Final":       ("harmony", -10.7, 0.70, 0.65, 0.5, None, None),
    "Harmony Final L":     ("harmony", -12.2, -0.70, 0.65, 0.5, None, FX["harmony copy"]),
    # the lead's calls in other voices' sections: its own voice, a step to the right
    "AI Answer":           ("answer", -2.0, 0.18, 0.55, 0.6, None, None),
    # the lead's thrown-in sounds: off-centre, band-limited, little room; ハイ ハイ is thrown into the delay
    "AI Click":            ("click", 1.0, -0.40, 0.15, 0.0, None, FX["click"]),
    "AI Hai":              ("ad-lib", -1.6, -0.40, 0.15, 1.3, None, None),
    "AI Stutter":          ("stutter", 2.5, 0.0, 0.50, 0.3, None, FX["stutter"]),
    # the aliens' lines right of the centre; their answers and thrown-in sounds outside them
    "Aliens":              ("alien lead", 2.1, 0.25, 0.30, 0.6, ALIEN, None),
    "Aliens Verse":        ("alien lead", -0.4, 0.25, 0.30, 0.6, ALIEN, None),
    "Aliens Bridge":       ("alien lead", 3.6, 0.30, 0.30, 0.6, ALIEN, None),
    "Alien Answer":        ("alien answer", 1.7, 0.50, 0.30, 0.6, ALIEN, FX["alien answer"]),
    "Alien Answer Bridge": ("alien answer", 4.9, 0.50, 0.30, 0.6, ALIEN, FX["alien answer"]),
    "Aliens Hook":         ("alien backing", -2.0, 0.25, 0.40, 0.3, ALIEN_UNDER, FX["alien backing"]),
    "Alien Unison":        ("alien backing", 1.6, -0.45, 0.40, 0.3, ALIEN_UNDER, FX["alien backing"]),
    "Alien Ouch":          ("alien ad-lib", 3.9, 0.70, 0.15, 0.8, ALIEN, FX["alien adlib"]),
    "Alien Nande":         ("alien ad-lib", 4.4, 0.70, 0.15, 1.3, ALIEN, FX["alien adlib"]),
    "Alien Click":         ("alien click", 4.6, 0.70, 0.15, 0.0, ALIEN, FX["alien click"]),
    "Alien Laugh":         ("alien laugh", 1.9, 0.70, 0.15, 1.0, ALIEN, FX["alien laugh"]),
    "Alien Stutter":       ("alien stutter", 5.6, 0.50, 0.30, 0.3, ALIEN, None),
    # the two chant voices left and right of everything the aliens sing
    "Alien Chant":         ("alien chant", 4.8, 0.60, 0.45, 0.6, ALIEN, FX["alien chant"]),
    "Alien Chant 2":       ("alien chant", 2.5, -0.60, 0.50, 0.8, {"eq_add": [["peak", 280, -3.0, 1.0]]},
                            FX["alien chant"]),
}
# The crowd: three voices wide apart, each kind at its own level and send, dry of the delay:
# kind -> (kind of sound, fader of voices 1 and 2, fader of voice 3 (an octave up), reverb send).
# The whisper's fader is high because a whisper is quiet.
CROWD_PAN = {"": -0.85, "2": 0.85, "3": -0.30}
CROWD_KINDS = {"": ("crowd", -4.5, -8.0, 0.55), "Answer": ("crowd", -5.0, -7.0, 0.55),
               "Whisper": ("crowd", 4.0, 4.0, 0.55), "Hook": ("crowd", -9.0, -12.0, 0.55),
               "Shout": ("crowd shout", -5.0, -7.0, 0.40), "Click": ("crowd shout", -4.5, -6.5, 0.40)}
LEAD_TRACKS = tuple(k for k, v in VOCAL_MIX.items() if v[0] == "lead")  # the lead vocal (on several mixer channels)


def vocal_mix(name: str, balance: str | None = None) -> dict | None:
    """``{kind, gain_db, pan, reverb_send, delay, strip, fx}`` of a vocal track at a balance setting
    (:data:`BALANCES`; None = the default), or None for an unknown one. A second take (vocals.SECOND_TAKES)
    sits where its first take does, at its own level against it."""
    balance = balance or DEFAULT_BALANCE
    second = vocals.second_take_of(name)
    if second:
        mix = vocal_mix(second[0], balance)
        return mix and dict(mix, gain_db=mix["gain_db"] + second[1]["gain_db"])
    mix = _vocal_mix(name)
    if mix and BALANCES[balance]["restored"] and mix["kind"] in CLEAN_EQ:
        mix["fx"] = dict(mix["fx"] or {}, eq=CLEAN_EQ[mix["kind"]])
    return mix


def _vocal_mix(name: str) -> dict | None:
    keys = ("kind", "gain_db", "pan", "reverb_send", "delay", "strip", "fx")
    if name in VOCAL_MIX:
        return dict(zip(keys, VOCAL_MIX[name]))
    w = name.split()
    if w[0] != "Crowd":
        return None
    voice = next((x for x in w[1:] if x in ("2", "3")), "")
    kind = next((x for x in w[1:] if x not in ("2", "3")), "")
    if kind not in CROWD_KINDS:
        return None
    role, gain, gain3, send = CROWD_KINDS[kind]
    return dict(zip(keys, (role, gain3 if voice == "3" else gain, CROWD_PAN[voice], send, 0.0, CROWD, FX[role])))


# ------------------------------------------------------------------------------------------- the band mix
# mix_overrides of the band's stems (by stem name; "drums snare" is the snare of the "drums" kit, which has
# a stem for every kind of drum with the mixer's own preset for it).
BAND_MIX: dict[str, dict] = {
    # the kick and the bass set off most of the master limiter's gain reduction: a little less of both, and
    # the kick's transients held within 7 dB
    "drums kick": {"gain_db": -1.0, "eq_add": [["lowshelf", 60, -1.5, 0.7], ["highshelf", 6000, -3.0, 0.7]],
                   "crest_db": 7.0},
    # the snare is the lead's main rival in the presence band: less level, a deeper vocal-keyed 2.6 kHz carve
    "drums snare": {"gain_db": -2.0, "carve": 1.0, "eq_add": [["peak", 3000, -2.0, 0.8]]},
    # hats and shakers 1.5 dB lower and ducked under the vocals
    "drums hat": {"gain_db": -1.5, "duck": 1.5}, "drums ohat": {"gain_db": -1.5, "duck": 1.5},
    "drums shaker": {"gain_db": -1.5, "duck": 1.5}, "drums tamb": {"gain_db": -1.5, "duck": 1.5},
    # the mostly-side cymbals fell away in mono: narrower
    "drums crash": {"width": 0.7}, "drums ride": {"width": 0.7},
    "bass": {"gain_db": -1.0, "crest_db": 10.0},
    # the half-time verses were thin, so their plucks, slap bass and chugs come up; the pluck's 200-800 Hz sat
    # in the lead's fundamental range and its peaks (and the slap's pops) set off the limiter
    "pluck": {"gain_db": 3.5, "crest_db": 9.0, "eq_add": [["peak", 550, -3.0, 1.0]]},
    "slap": {"gain_db": 1.5, "crest_db": 12.0},
    "mute": {"gain_db": 2.0},
    # the chip-lead riff was level with the voices in the riff sections; the chorus hook tags (chip lead, bell)
    # overlap the lead's phrase ends: they duck under the vocals and get out of the way as the vocal stops
    "lead": {"gain_db": -3.5, "duck": 3.0}, "tags": {"gain_db": -3.5, "duck": 3.0},
    "bell": {"duck": 2.5},
    # the power chords leave a little room at 1 kHz for the lead's held low notes
    "gtrL": {"eq_add": [["peak", 1000, -1.5, 1.0]]}, "gtrR": {"eq_add": [["peak", 1000, -1.5, 1.0]]},
    "fx": {"gain_db": -1.5}, "sfx": {"gain_db": -1.5},
    "mbox": {"gain_db": -2.0},
    # the final chorus is the only chorus with the pad and the low chip lead: a little more of both
    "pad": {"gain_db": 3.0, "width": 0.75},
    "leadlo": {"gain_db": 2.0}, "tagslo": {"gain_db": 2.0},
}
# What changes for a stem while its instrument plays the synthesised preset (no sound set installed): the
# synthesised snare is the hissiest element of the mix (its spectrum rises 8-10 dB from 4 to 11 kHz).
SYNTH_MIX: dict[str, dict] = {
    "drums snare": {"eq_add": [["peak", 3000, -2.0, 0.8], ["highshelf", 5500, -6.5, 0.7]]},
}
# ... and while it plays its sampled sound. The sampler matches a sound's level to the preset it replaces,
# not its spectrum: the sampled guitars, basses and kick carry 4 to 7 dB more of the 1-5 kHz band than the
# synthesised ones (pick and string noise, the beater), and the mixer balances the whole band against the
# vocals in that band, so the band came out 5 dB lower under the voice. These entries put each sampled stem
# back at the level and the 1-5 kHz level its synthesised stem has in the mix.
SAMPLED_MIX: dict[str, dict] = {
    # (the sampled kick and bass are lighter under 120 Hz than the synthesised ones: no low cut, the bass up)
    "drums kick": {"gain_db": 0.0, "eq_add": [["peak", 2500, -4.0, 0.8], ["highshelf", 6000, -3.0, 0.7]]},
    # the sampled snare, clap and toms hit 5 to 8 dB harder at their peaks than the synthesised ones
    # (and the snare has more body and less crack: its rolls carry the pre-choruses)
    "drums snare": {"gain_db": -4.0, "crest_db": 10.0, "eq_add": [["peak", 3000, 5.0, 0.7]]},
    "drums clap": {"gain_db": -3.0, "crest_db": 10.0},
    "drums toms": {"gain_db": -4.5, "crest_db": 10.0},
    # sampled hats and cymbals have a body between 1 and 5 kHz that the synthesised ones (all above 6 kHz) lack,
    # and the sampled cowbell of the bridge sits right in the voice's band
    "drums hat": {"gain_db": -3.5, "eq_add": [["peak", 2500, -6.0, 0.6]]},
    "drums ohat": {"gain_db": -3.5, "eq_add": [["peak", 2500, -6.0, 0.6]]},
    "drums cowbell": {"gain_db": -5.0},
    "drums crash": {"gain_db": -2.0, "eq_add": [["peak", 2500, -5.0, 0.6]]},
    "drums ride": {"gain_db": -2.0, "eq_add": [["peak", 2500, -5.0, 0.6]]},
    "bass": {"gain_db": 0.0, "eq_add": [["highshelf", 1200, -9.5, 0.7]]},
    "slap": {"gain_db": 0.5},
    # the sampled guitars are mono sounds (the synthesised ones are detuned stereo stacks): further apart, so
    # that the band stays as wide as it was
    "gtrL": {"gain_db": -1.0, "pan": -0.8, "eq_add": [["peak", 1000, -1.5, 1.0], ["peak", 2600, -6.0, 0.7]]},
    "gtrR": {"gain_db": -1.5, "pan": 0.8, "eq_add": [["peak", 1000, -1.5, 1.0], ["peak", 2600, -5.0, 0.7]]},
    "mute": {"gain_db": -3.0, "eq_add": [["highshelf", 1500, -8.0, 0.7]]},
    "bell": {"gain_db": -3.0},
    # the pizzicato is duller than the synthesised pluck, the music box quieter
    "pluck": {"gain_db": 2.5, "eq_add": [["peak", 550, -3.0, 1.0], ["peak", 2500, 2.0, 0.7]]},
    "mbox": {"gain_db": 1.0},
    # the sampled pad carries 13 dB more level under 700 Hz than the glassy synthesised one
    "pad": {"gain_db": 1.5, "eq_add": [["lowshelf", 700, -6.0, 0.7]]},
}

# What changes again for a sampled stem at the balance settings B and C (:data:`BALANCES`): the sampled band
# measured against the synthesised one, stem by stem, in the mix (bars 33-48 and 9-24). Entries are merged
# over BAND_MIX and SAMPLED_MIX.
RESTORED_MIX: dict[str, dict] = {
    # the sampled kick (tuned down, see restore_band) and bass have 4 dB more crest than the synthesised ones
    "drums kick": {"crest_db": 5.0},
    # the sampled bass: weak fundamentals (the sine under it is "basslo"), 5 to 8 dB more at 200-320 Hz
    "bass": {"crest_db": 7.0, "eq_add": [["peak", 280, -3.0, 0.8], ["peak", 750, -3.5, 0.8],
                                         ["highshelf", 1200, -9.5, 0.7]]},
    "basslo": {"gain_db": 5.0, "crest_db": None, "eq_add": [["lp", 110, 0, 0.7071, 2]]},
    # (an entry named after a role reaches every stem of that role: the synthesised growl and sub basses keep
    # the level and tone they have next to the synthesised bass, the slap bass the pick bass's high cut)
    "growl": {"gain_db": -1.0, "crest_db": 10.0, "eq_add": []},
    "sub": {"gain_db": -1.0, "crest_db": 10.0, "eq_add": []},
    # the slap bass carried 5 dB more at 200-500 Hz than the synthesised one
    "slap": {"gain_db": 1.5, "eq_add": [["peak", 320, -4.0, 0.9], ["highshelf", 1200, -9.5, 0.7]]},
    # the backbeat: the sampled clap sat 5.7 dB under the synthesised one; clap, snare and toms are mono samples
    # (the synthesised clap is stereo noise). The clap's peaks stood as high as the kick's, 26 dB over its
    # level: with the kick and the snare under them they were the master limiter's deepest dips.
    "drums snare": {"gain_db": -4.5, "crest_db": 8.0, "spread": 0.2, "reverb": 0.12},
    "drums clap": {"gain_db": 1.5, "crest_db": 5.0, "spread": 0.7, "reverb": 0.2},
    "drums toms": {"crest_db": 8.0, "spread": 0.35},
    # the sampled crash sat 5 dB under the synthesised one, and all the kit's metal on the right
    "drums crash": {"gain_db": 3.5, "width": 1.0, "pan": -0.7, "eq_add": [["peak", 2500, -5.0, 0.6]]},
    "drums tamb": {"pan": -0.5},
    "drums ride": {"gain_db": -0.5, "width": 1.0},
    # the sampled power chords are thinner under 500 Hz than the synthesised ones (the right one 10 dB at
    # 60-200 Hz, with 5 dB more at 500-1000 Hz) and carry 10 to 15 dB more above 5 kHz
    "gtrL": {"gain_db": 0.0, "eq_add": [["lowshelf", 250, 2.0, 0.7], ["peak", 700, 3.0, 0.9],
                                        ["peak", 1000, -1.5, 1.0], ["peak", 2600, -6.0, 0.7],
                                        ["highshelf", 6500, -5.0, 0.7]]},
    "gtrR": {"gain_db": -0.5, "eq_add": [["lowshelf", 220, 4.0, 0.7], ["peak", 750, -3.5, 0.9],
                                         ["peak", 1000, -1.5, 1.0], ["peak", 2600, -5.0, 0.7],
                                         ["highshelf", 6500, -4.0, 0.7]]},
    # the sampled pad is a warm pad where the synthesised one is glassy: 15 dB more at 200-500 Hz, which filled
    # the pre-choruses, the breakdown and the final chorus; the music box has a body the synthesised one lacks
    # and stood 4 dB under the lead in the breakdown
    "pad": {"gain_db": 0.0, "eq_add": [["hp", 450, 0, 0.7071, 2], ["lowshelf", 700, -6.0, 0.7]]},
    "mbox": {"gain_db": -2.0, "eq_add": [["hp", 350, 0, 0.7071, 2]]},
    # mono samples where the synthesised presets are stereo: the muted guitar (7 dB more at 500-1000 Hz than
    # the synthesised chug), the pizzicato
    "mute": {"gain_db": -4.0, "spread": 0.45,
             "eq_add": [["peak", 800, -4.0, 0.8], ["highshelf", 1500, -8.0, 0.7]]},
    # the pizzicato carried 5 dB more under 500 Hz than the synthesised pluck and 3 dB less of 1-5 kHz
    "pluck": {"spread": 0.45, "eq_add": [["lowshelf", 320, -6.0, 0.7], ["peak", 550, -3.0, 1.0],
                                         ["peak", 2500, 4.0, 0.7]]},
}
# the synthesised stems whose RESTORED_MIX entry holds while another instrument plays its sampled sound
RESTORED_FOR = {"basslo": "bass", "growl": "bass", "sub": "bass"}
# ... and for the vocal stack (by kind of channel): the doubles, the harmony and the crowd without their
# 200-500 Hz, which the lead and the band already fill. These replace the kind's tone (``fx`` ``eq``).
_LOW_CUT = [["hp", 190, 0, 0.7071, 2], ["peak", 360, -3.0, 1.0]]
CLEAN_EQ: dict[str, list] = {
    "double": _LOW_CUT,
    "harmony": [*_LOW_CUT, ["peak", 2800, -3.0, 0.9]],
    "crowd": [["hp", 230, 0, 0.7071, 2], ["peak", 400, -3.0, 1.0], ["highshelf", 3800, -4.0, 0.7]],
    "crowd shout": [["hp", 260, 0, 0.7071, 2], ["peak", 420, -3.0, 1.0], ["highshelf", 5000, -2.0, 0.7]],
}
# The balance dial: the same song and takes at four balances of the vocals against the band.
#   A  the sampled band as it was first mixed: every sampled stem at its synthesised stem's 1-5 kHz level
#   S  the synthesised band (no sampled sound plays, whatever is installed)
#   B  the sampled band with its low end, punch and width restored, the vocal stack cleaned, the vocals 2 dB
#      lower against the band than in A
#   C  as B, the vocals 4 dB lower than in A (band-forward)
# ``vocal_balance_db``: the vocals' 1-5 kHz level over the band that the mixer sets the band's gain by. The
# values of B and C are set on the whole song so that the loudness of the vocals' stems together against the
# band's stems together (LUFS) is 2 and 4 LU lower than at A (A: +0.5 LU); the 1-5 kHz figure moves less than
# that because the restored band carries more of that range (its backbeat, its guitars at their written level).
BALANCES: dict[str, dict] = {
    "A": {"sampled": True, "restored": False, "vocal_balance_db": 8.0},
    "S": {"sampled": False, "restored": False, "vocal_balance_db": 8.0},
    "B": {"sampled": True, "restored": True, "vocal_balance_db": 5.8},
    "C": {"sampled": True, "restored": True, "vocal_balance_db": 3.8},
}
RESTORED_BAND_WIDTH = 1.4  # "band_width" at B and C while sampled sounds play: as wide as the synthesised band
DEFAULT_BALANCE = "B"

MASTER: dict = {
    "loudness_target": -11.5,  # (-11 kept the master limiter over 1 dB for ~20 % of the final chorus)
    "ceiling_db": -1.0,
    "glue": 0.35,              # lighter master glue: at 0.45 it ate most of the final chorus' extra level (and
                               # pumped); under 0.3 the limiter took over
    "vocal_balance_db": 8.0,   # the vocals' 1-5 kHz level over the band
    "band_width": 1.3,         # the band was narrow (side 9-11 dB under mid): open the L/R guitars and synths
    # master EQ (folded into every source): the default subsonic high-pass, some 250 Hz warmth (every role
    # preset scoops 250-450 Hz), a little less of the 630 Hz honk of the vocal fundamentals, a softer air shelf
    "master_eq": [["hp", 25, 0, 0.7071, 2], ["peak", 250, 2.0, 1.2], ["peak", 630, -1.0, 1.4],
                  ["highshelf", 12000, 0.5, 0.7]],
    # a short room (0.6 s): it fills the gaps between the syllables and costs fewer words than a hall
    "reverb_room": 0.10,
    "vocal_delay": 0.3,        # dotted-8th ping-pong on the vocals: it blooms in the gaps and stops ...
    "delay_duck_db": 8.0,      # ... and steps back while they sing
    "delay_beats": 0.75,
    "delay_feedback": 0.3,
}


# Section-level band tweaks, applied with set_backing(merge=true) on top of backing.json: velocity
# multipliers for the parts of a section. (The drums and the bass have channel compressors, which undo about
# two thirds of a velocity change; the keys, pads and guitars follow it.)
# * The choruses barely lifted: the second half of the intro and of each pre-chorus were as loud and bright
#   as the chorus, with more low end, so the band steps back there and the hook entries land (with the "AI Pre"
#   channel, and the intro's drums and bass entering at bar 7, see backing.json).
# * The final chorus used the same band levels as choruses 1 and 2, so it was barely louder after the master
#   chain; its guitars, supersaws, chugs, pad, drums and bass dig in harder.
# * The breakdown's pad and music box sat over the AI's "don't close the window".
# * The joke sections (post-chorus calls, the bridge, the outro review) had their calls masked by the riff,
#   the chaos drums and the crushed guitars.
BAND_TWEAKS: dict[str, dict[str, float]] = {
    "Intro": {"lead": 0.85, "bass": 0.85, "gtrL": 0.75, "gtrR": 0.75, "saws": 0.7},
    "Pre-chorus 1": {"drums": 0.7, "bass": 0.75, "gtrL": 0.75, "gtrR": 0.75, "mute": 0.7, "pad": 0.75, "arp": 0.8},
    "Post-chorus": {"lead": 0.7, "bell": 0.7, "drums": 0.8, "gtrL": 0.8, "gtrR": 0.8},
    "Pre-chorus 2": {"drums": 0.7, "bass": 0.75, "gtrL": 0.75, "gtrR": 0.75, "mute": 0.7, "pad": 0.75, "arp": 0.8},
    "Bridge": {"drums": 0.8, "gtrL": 0.75, "gtrR": 0.75, "glitch": 0.75, "growl": 0.8, "saws": 0.8, "arp": 0.8},
    "Breakdown": {"pad": 0.75, "mbox": 0.8},
    "Final chorus": {"gtrL": 1.25, "gtrR": 1.25, "saws": 1.5, "mute": 1.15, "pad": 1.2, "drums": 1.12, "bass": 1.12},
    "Outro": {"lead": 0.7, "leadlo": 0.8, "drums": 0.85, "gtrL": 0.85, "gtrR": 0.85},
}


def tweak_sections(arrangement: dict, tweaks: dict[str, dict[str, float]]) -> list[dict]:
    """Copies of the arrangement's sections named in ``tweaks`` with the parts' velocities scaled."""
    out = []
    for sec in arrangement.get("sections", []):
        if sec.get("name") not in tweaks:
            continue
        sec = json.loads(json.dumps(sec))
        for part, mult in tweaks[sec["name"]].items():
            spec = sec.get("parts", {}).get(part)
            if spec is None:
                raise KeyError(f"section {sec['name']!r} has no part {part!r}")
            segs = spec if isinstance(spec, list) else [spec]
            segs = [{"pattern": g} if isinstance(g, str) else g for g in segs]
            for g in segs:
                g["velocity"] = round(float(g.get("velocity", 1.0)) * mult, 3)
            sec["parts"][part] = segs if isinstance(spec, list) else segs[0]
        out.append(sec)
    return out


def master_settings(sampled: set[str] | None = None, tracks=None, balance: str | None = None) -> dict:
    """MASTER with the mix_overrides of every vocal track of ``tracks`` (delay send, strip overrides, in-channel
    effects) and of the band's stems, at a balance setting (:data:`BALANCES`; None = the default).
    ``sampled``: the instruments that play a sampled sound (their stems get the SAMPLED_MIX entries and, at
    the settings B and C, the RESTORED_MIX entries; the others the SYNTH_MIX entries); None = none does."""
    bal = BALANCES[balance or DEFAULT_BALANCE]
    m = json.loads(json.dumps(MASTER))
    m["vocal_balance_db"] = bal["vocal_balance_db"]
    if bal["restored"] and sampled:
        m["band_width"] = RESTORED_BAND_WIDTH
    mo = m["mix_overrides"] = {}
    for t in tracks or []:
        mix = vocal_mix(t.name, balance)
        if mix is None:
            continue
        o = dict(mix["strip"] or {}, delay=mix["delay"])
        if mix["fx"]:
            o["fx"] = mix["fx"]
        mo[t.name] = o
    sampled = set(sampled or ())
    for stem in sorted(set(BAND_MIX) | set(SYNTH_MIX) | set(SAMPLED_MIX) | set(RESTORED_MIX)):
        which = SAMPLED_MIX if stem.split()[0] in sampled else SYNTH_MIX
        restored = bal["restored"] and RESTORED_FOR.get(stem, stem.split()[0]) in sampled
        entry = {**BAND_MIX.get(stem, {}), **which.get(stem, {}), **(RESTORED_MIX.get(stem, {}) if restored else {})}
        if entry:
            mo[stem] = entry
    return json.loads(json.dumps(m))


# The sampled sounds that ring (the synthesised presets they replace are detuned stereo stacks): at the
# settings B and C each plays as two layers of the same sound a few cents apart, left and right of the
# instrument's place, each 3 dB down. (Two layers do not part a short sound: the plucks, the muted guitar and
# the clap get the mixer's ``spread`` instead, RESTORED_MIX.)
# instrument -> (cents up / down, pan left / right of the place)
STEREO_PAIRS: dict[str, tuple[float, float]] = {"pad": (6.0, 0.8), "mbox": (4.0, 0.5), "bell": (3.0, 0.4)}


def restore_band(arrangement: dict, sampled: set[str]) -> None:
    """The arrangement as the settings B and C play it (in place), for the instruments of ``sampled`` (those
    that play a sampled sound here; a synthesised instrument is left as it is).

    * The sampled kick's fundamental sits at 75-80 Hz, the synthesised kick's at 54 Hz: five semitones down.
    * The sampled pick bass has 9 to 14 dB less under 60 Hz than the synthesised one (its fundamentals are
      weak): a sine under it ("basslo") plays every line the bass plays.
    * :data:`STEREO_PAIRS`.
    * Velocity: a sampled sound's level follows the square of the velocity (the formats' default), a
      synthesised preset's follows it linearly at 70 % (the kit: to the power 1.35), and the sampler matches
      the two at a normal hit only. The section tweaks (BAND_TWEAKS: x 0.7 to x 1.25) and the soft notes of
      the patterns therefore moved a sampled instrument 2.5 to 4.5 dB further than the synthesised one (the
      pre-choruses' and the bridge's band came out 2 dB low): every sampled sound follows velocity as the
      preset it replaces does.
    """
    inst = arrangement["instruments"]
    for name in sorted(sampled):
        sound = inst[name].get("sound")
        if isinstance(sound, str):
            sound = {"set": sound}
        kit = inst[name].get("type") == "drums"
        law = {"velocity_curve": 1.35} if kit else {"velocity_curve": 1.0, "vel_sens": 0.72}
        layers = [dict(layer, **law) for layer in (sound if isinstance(sound, list) else [sound])]
        inst[name]["sound"] = layers if isinstance(sound, list) else layers[0]
    if "drums" in sampled:
        inst["drums"].setdefault("voices", {}).setdefault("kick", {})["tune"] = -5
    if "bass" in sampled:
        inst["basslo"] = {"preset": "bass_sub", "gain_db": -9}
        for sec in arrangement["sections"]:
            if sec["parts"].get("bass") is not None:
                sec["parts"]["basslo"] = json.loads(json.dumps(sec["parts"]["bass"]))
    for name, (cents, pan) in STEREO_PAIRS.items():
        sound = inst[name].get("sound")
        if name in sampled and isinstance(sound, dict):
            inst[name]["sound"] = [dict(sound, tune=cents, pan=-pan, gain_db=-3.0),
                                   dict(sound, tune=-cents, pan=pan, gain_db=-3.0)]


def band_tweaks(arrangement: dict) -> dict[str, dict[str, float]]:
    """BAND_TWEAKS for an arrangement: the sine under the bass (restore_band) follows the bass."""
    tweaks = json.loads(json.dumps(BAND_TWEAKS))
    if "basslo" in arrangement["instruments"]:
        for parts in tweaks.values():
            if "bass" in parts:
                parts["basslo"] = parts["bass"]
    return tweaks


def _sampled(project) -> set[str]:
    """The instruments of the project's band that play a sampled sound on this machine."""
    from synth2 import arranger

    return {k for k, v in arranger.sound_report(project)["instruments"].items() if v["plays"] == "sampled"}


# ------------------------------------------------------------------------------------------- build
def build_session(log=print, balance: str | None = None) -> Session:
    """The finished project in a tool session (vocals, production, backing, arrangement, mixer, master) at a
    balance setting (:data:`BALANCES`; None = :data:`DEFAULT_BALANCE`)."""
    from synth2 import arranger

    balance = balance or DEFAULT_BALANCE
    if balance not in BALANCES:
        raise KeyError(f"no balance setting {balance!r} (one of {', '.join(BALANCES)})")
    s = Session()
    r = call_tool(s, "new_project", {"score": (HERE / "vocals.s2s").read_text(encoding="utf-8")})
    for w in r.get("warnings", []) or []:
        log(f"  score warning: {w}")
    prod = vocals.apply_vocal_production(s)
    log(f"  vocal production: {len(prod['tracks'])} tracks, AI Double {prod['ai_double_notes']} notes, "
        f"Harmony {prod['harmony_notes']} notes, fan-voice regions {prod['fan_chop_regions']}")
    arrangement = json.loads((HERE / "backing.json").read_text(encoding="utf-8"))
    if not BALANCES[balance]["sampled"]:
        for inst in arrangement["instruments"].values():
            inst.pop("sound", None)
    r = call_tool(s, "set_backing", {"arrangement": arrangement})
    sampled = _sampled(s.project)
    if BALANCES[balance]["restored"] and sampled:
        restore_band(arrangement, sampled)
        r = call_tool(s, "set_backing", {"arrangement": arrangement})
    for w in r.get("warnings", []):
        log(f"  backing {w}")
    tweaks = band_tweaks(arrangement)
    if tweaks:
        r = call_tool(s, "set_backing", {"arrangement": {"sections": tweak_sections(arrangement, tweaks)},
                                         "merge": True})
        for w in r.get("warnings", []):
            log(f"  backing {w}")
    arr = vocals.apply_arrangement(s, log=log)
    roles = sorted({t["role"].split(":")[0] for t in arr["tracks"] if t["role"]})
    log(f"  arrangement: {len(arr['tracks'])} vocal tracks, roles {', '.join(roles)}")
    for t in s.project.tracks:
        mix = vocal_mix(t.name, balance)
        if mix is None:
            raise KeyError(f"no mix for the vocal track {t.name!r}")
        call_tool(s, "update_track", {"track": t.name, "gain_db": mix["gain_db"], "pan": mix["pan"],
                                      "reverb_send": mix["reverb_send"]})
    sound = arranger.sound_report(s.project)
    log("  band: " + (f"{len(sampled)} instruments play sampled sounds ({', '.join(sorted(sampled))})" if sampled
                      else "every instrument is synthesised (no sound set installed)"))
    for w in sound["warnings"]:
        log(f"  band: {w}")
    log(f"  balance setting {balance}: vocals {BALANCES[balance]['vocal_balance_db']:g} dB over the band (1-5 kHz)")
    call_tool(s, "update_project", {"master": master_settings(sampled, s.project.tracks, balance)})
    return s


# ------------------------------------------------------------------------------------------- level cache
def _mix_key(project, sr: int, engine: str | None = None) -> str:
    """A fingerprint of the mix that is stable across builds (the renderer's own cache key uses track ids,
    which are new on every build): master settings, backing, mixer and notes by track name, the engines."""
    import hashlib

    from synth2.engine.backend import engine_signature

    tracks = [(t.name, t.voice, t.gain_db, t.pan, t.mute, t.solo, t.reverb_send, t.role, t.seed,
               sorted(t.voice_params.items()), [(round(n.onset, 4), n.pitch, n.lyric) for n in t.notes])
              for t in project.tracks]
    blob = json.dumps({"master": project.master.to_dict(), "backing": project.backing, "tracks": tracks, "sr": sr,
                       "tempo": project.tempo.to_list(), "engines": engine_signature(project, engine)},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def save_levels(out: Path, project, sr: int, report: dict, engine: str | None = None) -> None:
    """Remember the full render's level decisions, so later section renders are mastered alike."""
    levels = {k: report["mix"].get(k) for k in _renderer._LEVEL_KEYS}
    (out / "mix_levels.json").write_text(json.dumps({"key": _mix_key(project, sr, engine), "levels": levels},
                                                    indent=1))


def load_levels(out: Path, project, sr: int, engine: str | None = None) -> bool:
    """Prime the renderer's level cache from ``mix_levels.json`` when it matches this mix."""
    try:
        d = json.loads((out / "mix_levels.json").read_text())
    except (OSError, ValueError):
        return False
    if d.get("key") != _mix_key(project, sr, engine) or any(v is None for v in d.get("levels", {}).values()):
        return False
    _renderer._LEVEL_CACHE[_renderer._level_key(project, sr, engine)] = d["levels"]
    return True


# ------------------------------------------------------------------------------------------- analysis
def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _band_power(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Grid (1 ms) power of ``x`` band-limited to lo..hi Hz."""
    y = mixing.eq(_mono(x).astype(np.float32), [("hp", lo, 0, 0.7071, 4), ("lp", hi, 0, 0.7071, 4)], sr,
                  threads=mixing.mix_threads())
    return mixing.block_power(y, mixing.grid_hop(sr))


def _ratio_db(pa: np.ndarray, pb: np.ndarray, mask: np.ndarray, overlap: bool = True) -> float | None:
    """Power ratio (dB) of ``pa`` over ``pb`` on the grid blocks in ``mask`` (and, with ``overlap``,
    only where ``pb`` is not silent)."""
    m = mask & (pb > 1e-14) if overlap else mask
    if m.sum() < 200:  # under 0.2 s
        return None
    return round(float(10 * np.log10(max(pa[m].sum(), 1e-30) / max(pb[m].sum(), 1e-30))), 1)


def _act(x: np.ndarray, sr: int, below_db: float = 30.0) -> np.ndarray:
    """Grid activity (bool) of a stem: 10 ms RMS within ``below_db`` of its loud (95th pct) level."""
    lvl = mixing.detector_db(x, sr, "rms", 0.01)
    on = lvl[lvl > -90]
    if not len(on):
        return np.zeros(len(lvl), dtype=bool)
    return lvl > np.percentile(on, 95) - below_db


def ltas(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Long-term average spectrum in 1/3-octave bands (mean PSD in dB) and its slope in dB/octave
    (least squares over 100 Hz - 10 kHz)."""
    m = _mono(x).astype(np.float64)
    nfft = 8192
    win = np.hanning(nfft)
    hop = nfft // 2
    k = max(1, (len(m) - nfft) // hop)
    acc = np.zeros(nfft // 2 + 1)
    for i in range(0, k, 2):  # every other frame is plenty
        fr = m[i * hop:i * hop + nfft]
        if len(fr) < nfft:
            break
        acc += np.abs(np.fft.rfft(fr * win)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    centres = 1000.0 * 2.0 ** (np.arange(-17, 14) / 3.0)  # 20 Hz .. 20 kHz
    psd = []
    for c in centres:
        b = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        psd.append(10 * np.log10(acc[b].mean() + 1e-30) if b.any() else np.nan)
    psd = np.array(psd)
    sel = (centres >= 100) & (centres <= 10000) & np.isfinite(psd)
    slope = float(np.polyfit(np.log2(centres[sel]), psd[sel], 1)[0])
    return centres, psd, slope


def analyse(project, res, start_bar: int, end_bar: int, with_pitch: bool = True) -> dict:
    """Section loudness and balance, low end, spectrum slope and pitch accuracy of a render.

    Per section: integrated loudness, the vocal-to-band ratio in 1-5 kHz while the vocals sing (all vocals vs
    the band, as the mixer measures it), the same for the lead alone (while the lead sings), the lead against
    the sum of the other vocals, and the lead's margin over the loudest single other element (any vocal track
    or band stem) in 1-5 kHz, with that element's name.
    """
    sr = res.sample_rate
    hop = mixing.grid_hop(sr)
    rep = res.report
    groups = {s["key"]: s["group"] for s in rep["stem_levels"]}
    names = {t.id: t.name for t in project.tracks}
    n = len(res.audio)
    vocal = np.zeros((n, 2), dtype=np.float32)
    band = np.zeros((n, 2), dtype=np.float32)
    lead = np.zeros((n, 2), dtype=np.float32)
    lead_keys = {t.id for t in project.tracks if t.name in LEAD_TRACKS}
    for k, v in res.stems.items():
        (vocal if groups.get(k) == "vocal" else band)[: len(v)] += v[:n]
        if k in lead_keys:
            lead[: len(v)] += v[:n]
    lo_hz, hi_hz = mixing.VBR_BAND
    pv, pb, pl = (_band_power(x, sr, lo_hz, hi_hz) for x in (vocal, band, lead))
    each = {names.get(k, k): _band_power(v, sr, lo_hz, hi_hz) for k, v in res.stems.items() if k not in lead_keys}
    other_vox = sum(p for k, p in each.items() if k in names.values())
    act_v, act_l = _act(vocal, sr), _act(lead, sr)
    nb = min(len(pv), len(act_v), len(act_l), *(len(p) for p in each.values()))
    b2s = project.tempo.beat_to_sec
    t0 = res.start_seconds
    sections = []
    for name, a, b in SECTIONS:
        if b < start_bar or a > end_bar:
            continue
        s0 = max(b2s((a - 1) * BEATS_PER_BAR), b2s((start_bar - 1) * BEATS_PER_BAR)) - t0
        s1 = min(b2s(b * BEATS_PER_BAR), b2s(end_bar * BEATS_PER_BAR)) - t0
        i0, i1 = int(s0 * sr), min(n, int(s1 * sr))
        if i1 - i0 < int(0.5 * sr):
            continue
        loud = mixing.loudness_stats(res.audio[i0:i1], sr)
        sl = slice(i0 // hop, min(nb, i1 // hop))
        mv, ml = act_v[sl], act_l[sl]
        margin, rival = None, None
        if ml.sum() >= 200:
            lead_p = pl[sl][ml].sum()
            rival, rp = max(((k, p[sl][ml].sum()) for k, p in each.items()), key=lambda kv: kv[1])
            margin = round(float(10 * np.log10(max(lead_p, 1e-30) / max(rp, 1e-30))), 1)
        sections.append({
            "section": name, "bars": f"{a}-{b}",
            "lufs": mixing._r(loud["integrated"], 1),
            "short_term_max_lufs": mixing._r(loud["short_term_max"], 1),
            "vocal_to_band_db": _ratio_db(pv[sl], pb[sl], mv),
            "lead_to_band_db": _ratio_db(pl[sl], pb[sl], ml),
            "lead_to_other_vocals_db": (_ratio_db(pl[sl], other_vox[sl], ml, overlap=False)
                                        if np.ndim(other_vox) else None),
            "lead_margin_db": margin, "next_loudest": rival,
            "lead_sings_pct": round(100.0 * float(ml.mean()), 0),
        })
    # low end (whole render): 40-120 Hz vs 120-500 Hz energy of the master
    lo = _band_power(res.audio, sr, 40, 120).sum()
    mid = _band_power(res.audio, sr, 120, 500).sum()
    centres, psd, slope = ltas(res.audio, sr)
    out = {
        "sections": sections,
        "low_40_120_vs_120_500_db": round(float(10 * np.log10(lo / max(mid, 1e-30))), 1),
        "ltas_slope_db_per_oct": round(slope, 2),
        "ltas_third_octave_db": {f"{c:.0f}": (round(float(p), 1) if np.isfinite(p) else None)
                                 for c, p in zip(centres, psd)},
        "clipped_samples": int(np.sum(np.abs(res.audio) >= 1.0)),
    }
    if with_pitch:
        pa = pitch_accuracy(project, res, max_list=5)
        out["pitch"] = [{"track": t["name"], "notes_checked": t["notes_checked"],
                         "mean_abs_cents": t["mean_abs_cents"], "max_abs_cents": t["max_abs_cents"],
                         "within_25_cents": t["within_25_cents"], "worst": t["worst"][:3],
                         "unvoiced_notes": len(t["unvoiced_notes"])} for t in pa["tracks"]]
    return out


def print_summary(rep: dict, ana: dict, wall: float) -> None:
    m = rep["mix"]
    print(f"\nrender: {rep['duration_s']:.1f} s of audio in {rep['render_time_s']:.1f} s "
          f"(mix {rep['mix_time_s']:.1f} s, {rep['realtime_factor']}x realtime; wall {wall:.1f} s)")
    print(f"master: {rep['loudness_lufs']} LUFS integrated, true peak {rep['true_peak_db']} dBTP, "
          f"short-term max {rep['short_term_max_lufs']} LUFS, LRA {rep['loudness_range_lu']} LU, "
          f"clipped samples {ana['clipped_samples']}")
    print(f"mix: vocal-to-band {rep['vocal_to_band_db']} dB (1-5 kHz), band gain {m.get('band_gain_db')} dB, "
          f"glue GR {m.get('glue_gr_max_db')} dB, limiter max {m.get('limiter_max_db')} dB / mean "
          f"{m.get('limiter_mean_db')} dB, master gain {m.get('master_gain_db')} dB, "
          f"levels from full render: {m.get('levels_from_full_render')}")
    print(f"spectrum: LTAS slope {ana['ltas_slope_db_per_oct']} dB/oct (100 Hz-10 kHz), "
          f"40-120 Hz vs 120-500 Hz {ana['low_40_120_vs_120_500_db']} dB")
    print(f"\n{'section':<14}{'bars':>8}{'LUFS':>7}{'ST max':>8}{'voc/band':>10}{'lead/band':>11}"
          f"{'lead/vox':>10}{'lead%':>7}   lead margin over the next loudest element (1-5 kHz)")
    f = lambda v: "-" if v is None else f"{v:+.1f}" if isinstance(v, float) else str(v)  # noqa: E731
    for s in ana["sections"]:
        print(f"{s['section']:<14}{s['bars']:>8}{f(s['lufs']):>7}{f(s['short_term_max_lufs']):>8}"
              f"{f(s['vocal_to_band_db']):>10}{f(s['lead_to_band_db']):>11}{f(s['lead_to_other_vocals_db']):>10}"
              f"{s['lead_sings_pct']:>7.0f}   {f(s['lead_margin_db'])} ({s['next_loudest'] or '-'})")
    if ana.get("pitch"):
        print(f"\n{'vocal track':<12}{'notes':>7}{'mean c':>8}{'max c':>8}{'<=25c':>8}")
        for t in ana["pitch"]:
            mean = "-" if t["mean_abs_cents"] is None else f"{t['mean_abs_cents']:.1f}"
            mx = "-" if t["max_abs_cents"] is None else f"{t['max_abs_cents']:.1f}"
            w = "-" if t["within_25_cents"] is None else f"{100 * t['within_25_cents']:.0f}%"
            print(f"{t['track']:<12}{t['notes_checked']:>7}{mean:>8}{mx:>8}{w:>8}")
    sampled = sorted({b["sound"] for b in rep.get("backing", []) if b.get("sound")})
    print(f"\nengines: {', '.join(rep.get('engines', []))}; band: "
          + (f"{len(sampled)} sampled sounds ({'; '.join(sampled)})" if sampled else "synthesised"))
    for c in rep.get("credits", []):
        print(f"credit: {c}")
    if rep.get("warnings"):
        print(f"{len(rep['warnings'])} warnings, e.g.: " + "; ".join(rep["warnings"][:3]))


def _decoded_true_peak_db(path: Path, sr: int) -> float:
    """True peak (dBTP) of an encoded file, decoded by ffmpeg."""
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-f", "f32le", "-ac", "2", "-ar", str(sr),
                          "-"], check=True, capture_output=True).stdout
    return float(mixing.true_peak_db(np.frombuffer(raw, dtype=np.float32).reshape(-1, 2)))


def write_mp3(wav: Path, mp3: Path, sr: int, ceiling_db: float) -> dict:
    """Encode a 320 kbps MP3. The encoder's filtering adds inter-sample overshoot (the -1 dBTP WAV decoded at
    -0.4 dBTP), so the MP3 is re-encoded with a small trim until its decoded true peak is under the ceiling."""
    trim = 0.0
    for _ in range(3):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav)]
        if trim < 0:
            cmd += ["-af", f"volume={trim:.2f}dB"]
        subprocess.run(cmd + ["-codec:a", "libmp3lame", "-b:a", "320k", "-metadata", "title=SELECT ALL INVADERS",
                              "-metadata", "artist=Synth2 Vocal Studio", str(mp3)], check=True)
        tp = _decoded_true_peak_db(mp3, sr)
        if tp <= ceiling_db:
            break
        trim -= tp - ceiling_db + 0.1
    return {"path": str(mp3), "true_peak_db": round(tp, 2), "trim_db": round(trim, 2)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sr", type=int, default=44100, help="sample rate (default 44100)")
    ap.add_argument("--start-bar", type=int, default=None, help="first bar of a section render (1-based)")
    ap.add_argument("--end-bar", type=int, default=None, help="last bar of a section render (inclusive)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output folder (default {OUT})")
    ap.add_argument("--stems", action="store_true", help="write the stems (about 100 files of 50 MB for the song)")
    ap.add_argument("--no-stems", action="store_true", help=argparse.SUPPRESS)  # (the default now)
    ap.add_argument("--no-mp3", action="store_true", help="do not encode the mp3")
    ap.add_argument("--pitch", action="store_true", help="measure the pitch accuracy of every vocal track (slow)")
    ap.add_argument("--no-pitch", action="store_true", help=argparse.SUPPRESS)  # (the default now)
    ap.add_argument("--engine", default=None,
                    help="singing engine: auto (the score's choice), formant, neural or neural:<voicebank>")
    ap.add_argument("--quality", default=None, choices=("draft", "normal", "high"), help="neural render quality")
    ap.add_argument("--balance", default=DEFAULT_BALANCE, choices=tuple(BALANCES),
                    help="the vocals against the band: A = the first sampled mix, S = the synthesised band, "
                         "B = sampled band restored, vocals 2 dB lower, C = vocals 4 dB lower "
                         f"(default {DEFAULT_BALANCE})")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()

    print("building the project through the tool layer ...")
    s = build_session(balance=args.balance)
    project = s.project
    last_bar = len(vocals.CHORDS)
    partial = args.start_bar is not None or args.end_bar is not None
    a = max(1, args.start_bar or 1)
    b = min(last_bar, args.end_bar or last_bar)
    if b < a:
        ap.error("--end-bar must not be before --start-bar")

    if not partial:
        r = call_tool(s, "save_project", {"path": str(out / f"{NAME}.s2p.json")})
        print(f"  project: {r.get('path', out / (NAME + '.s2p.json'))}")
        start_beat = end_beat = None
    else:
        start_beat, end_beat = float((a - 1) * BEATS_PER_BAR), float(b * BEATS_PER_BAR)
        reused = load_levels(out, project, args.sr, args.engine)
        print(f"  section render: bars {a}-{b} (beats {start_beat:g}-{end_beat:g}); "
              + ("mastered with the levels of the last full render" if reused else
                 "no matching full render yet, so this section is mastered on its own"))

    print(f"rendering at {args.sr} Hz ...")
    t_render = time.perf_counter()

    def progress(frac: float, msg: str) -> None:
        print(f"  [{100 * frac:3.0f}% {time.perf_counter() - t_render:5.1f} s] {msg}", flush=True)

    res = render_project(project, sample_rate=args.sr, start_beat=start_beat, end_beat=end_beat, progress=progress,
                         engine=args.engine, quality=args.quality)
    rep = res.report

    if partial:
        dest = out / "preview"
        wav = dest / f"{NAME}_bars_{a}-{b}.wav"
    else:
        dest = out
        wav = out / f"{NAME}.wav"
        save_levels(out, project, args.sr, rep, args.engine)
    dest.mkdir(parents=True, exist_ok=True)
    write_wav(wav, res.audio, res.sample_rate, 24)
    print(f"  wrote {wav}")
    if args.stems and not args.no_stems:
        sd = dest / ("stems" if not partial else f"stems_bars_{a}-{b}")
        if sd.exists():
            shutil.rmtree(sd)
        res.save_stems(sd, bits=24)
        print(f"  wrote {len(res.stems)} stems to {sd}/")
    mp3_info = None
    if not args.no_mp3 and not partial:
        if shutil.which("ffmpeg"):
            mp3_info = write_mp3(wav, wav.with_suffix(".mp3"), res.sample_rate, project.master.ceiling_db)
            print(f"  wrote {mp3_info['path']} (decoded true peak {mp3_info['true_peak_db']} dBTP, "
                  f"trim {mp3_info['trim_db']} dB)")
        else:
            print("  ffmpeg not found: no mp3")

    print("analysing ...")
    ana = analyse(project, res, a, b, with_pitch=args.pitch)
    wall = time.perf_counter() - t_wall
    rep_out = {k: v for k, v in rep.items() if k != "tracks"}
    rep_out["tracks"] = [{k: v for k, v in t.items() if k != "phonemes"} for t in rep.get("tracks", [])]
    if mp3_info:
        rep_out["mp3"] = mp3_info
    (wav.with_suffix(".report.json")).write_text(json.dumps({"report": rep_out, "analysis": ana,
                                                             "wall_time_s": round(wall, 1)},
                                                            indent=1, ensure_ascii=False, default=str))
    print_summary(rep, ana, wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
