"""Build, mix and render SELECT ALL INVADERS (セレクト・オール・インベーダー).

The project is put together through the studio's tool layer, the way an AI producer would drive it
(``synth2.ai.tools.Session`` + ``call_tool``):

1. ``new_project`` from the score text in ``vocals.s2s`` (the four hand-written vocal tracks),
2. ``vocals.apply_vocal_production``: AI Double, Harmony, Crowd 2 / 3 and the aliens' fan voice,
3. ``set_backing`` with ``backing.json`` (the arranger's band), then ``set_backing(merge=true)`` for the
   section tweaks of the mix (:data:`BAND_TWEAKS`),
4. ``duplicate_track`` / ``add_notes`` / ``delete_notes`` to put section parts (the lead in the verses,
   pre-choruses, bridge, breakdown and final chorus, the aliens' verse, bridge and final-chorus hook, the crowd's
   final-chorus hook, the final-chorus harmony) on their own mixer channels (:data:`SPLITS`),
5. ``update_track`` for every vocal fader, pan and reverb send (:data:`TRACK_MIX`),
6. ``update_project`` for the master: loudness, delay, width, master EQ, band balance and the
   per-track / per-stem overrides (:data:`MASTER`),
7. ``save_project``, then ``render_project`` at 44.1 kHz.

The mix decisions are documented next to the settings below; every one of them came from a measurement
(section loudness, 1-5 kHz balance of each element against the lead, long-term spectrum, limiter activity,
the 1-5 kHz margin of every call-and-response joke line over the band).

It writes ``out/select_all_invaders/``: ``select_all_invaders.wav`` (24-bit), ``stems/`` (one
24-bit WAV per vocal track and band stem; post-fader, pre-reverb), ``select_all_invaders.s2p.json``,
``select_all_invaders.mp3`` (320 kbps, when ffmpeg is installed; trimmed so its decoded true peak stays
under the ceiling) and ``select_all_invaders.report.json``
(the render report plus the analysis printed at the end: loudness per section, true peak, the
vocal-to-band balance per section, the low-end balance, the long-term spectrum slope and the
pitch accuracy of every vocal track).

Usage (from the repository root)::

    python examples/select_all_invaders/build.py                          # the whole song
    python examples/select_all_invaders/build.py --start-bar 33 --end-bar 48   # a quick section render

A section render goes to ``out/select_all_invaders/preview/``. After a full render the mix levels
(band gain, compressor thresholds, master gain) are saved in ``mix_levels.json``, and a section render
of the same mix reuses them, so it is mastered like that part of the full song.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

# production copy of demo3/build3.py (itself a copy of examples/select_all_invaders/build.py): the song's files are
# read from the repo. What is new here:
# * PROD_VERSE=relaxed puts prod/relaxed_verse.s2s in place of the lead's verse 1 (bars 9-24),
# * PROD_CHAIN=off leaves the vocal production of demo 3 (the renderer's strip, the second-take double); the
#   default expects prod/ds_backend4.py, which runs prod/chain4.py on the lead, the double and the harmony, so the
#   renderer's own strip is switched off for those channels ("chain": False),
# * the drum and balance settings of the demo-3 final mix (final_mix.sh) are part of MASTER,
# * a short room at higher sends in place of the song's hall (REVERB_ROOM, CHAIN_SENDS); the song's ducked
#   dotted-eighth delay is kept.
PROD = Path(__file__).resolve().parent
VERSE = os.environ.get("PROD_VERSE", "written")
CHAIN = os.environ.get("PROD_CHAIN", "on") != "off"
ROOT = Path("/home/user/synth2")
HERE = ROOT / "examples" / "select_all_invaders"
# semitones the whole song (vocals and backing) is moved by; -5 = E minor (F# minor from bar 113)
KEY_SHIFT = int(os.environ.get("DEMO3_KEY", "-5"))
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

# ------------------------------------------------------------------------------------------- the mix
# Some parts get their own mixer channel for a section (a "track split", as in any DAW), so they can sit
# differently there without fader automation. A split channel inherits its source's channel settings
# (pan, reverb send, mix_overrides) unless TRACK_MIX / mix_overrides name it.
# * "AI Verses": the lead rides 2.5 dB lower in the two verses, where the half-time band is thin (it stuck
#   out of it by ~9 dB in 1-5 kHz), which also gives the choruses more lift. In verse 2 the AI and the
#   aliens then trade lines at about the same level ("Aliens Verse" keeps the aliens' verse-2 level).
# * "AI Pre": the pre-choruses' call-and-response sits 1 dB under the chorus lead, so the hook entries lift
#   (the second half of each pre-chorus was as loud as the chorus). The pickup "select" stays on the AI.
# * "AI Bridge": the bridge is the densest run of lyric gags over the loudest, most crushed band, so the
#   lead comes up 2.5 dB there (it sat only +2-3 dB over the band in 1-5 kHz).
# * "AI Final": the final chorus has the biggest band and the most support vocals, so the lead rides 2.5 dB
#   higher there (it was the section where the lead was least clear).
# * "AI Breakdown": the AI's unguarded "loading humanity..." moment (bars 109-112) sits lower, in more
#   reverb, so the breakdown drops away before the final chorus. The pickup "select" stays on the AI.
# * "Aliens Bridge": the aliens' half of the bridge dialogue ("have you tried turning your planet...", "no, you
#   are. Respectfully.") sits 1.5 dB over their other calls, against the crushed chaos band.
# * "Aliens Hook": in the final chorus the aliens sing the hook two octaves under the AI; on its own
#   channel it supports the lead instead of competing with it. Their answers ("click!", ワレワレハ) stay
#   on the main Aliens channel, at full level.
# * "Crowd Hook" (and its doubles): the crowd sings the final-chorus hook an octave under the AI, and
#   whispers "hurry" in the breakdown; their shouts ("click!", "hey!") stay on the main crowd channels.
# * "Harmony Final": the final chorus is harmonized throughout, so its harmony sits 2 dB lower there.
SPLITS: list[dict] = [
    {"source": "AI", "name": "AI Verses", "start": 32.0, "end": 96.0, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Verses", "start": 208.0, "end": 272.0, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Pre", "start": 96.0, "end": 127.5, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Pre", "start": 272.0, "end": 303.5, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Bridge", "start": 368.0, "end": 432.0, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Breakdown", "start": 432.0, "end": 447.25, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Final", "start": 447.25, "end": 512.0, "skip_lyrics": ()},
    {"source": "Aliens", "name": "Aliens Verse", "start": 208.0, "end": 272.0, "skip_lyrics": ()},
    {"source": "Aliens", "name": "Aliens Bridge", "start": 368.0, "end": 416.0, "skip_lyrics": ()},
    {"source": "Aliens", "name": "Aliens Hook", "start": 447.0, "end": 500.0, "skip_lyrics": ("click",)},
    {"source": "Crowd", "name": "Crowd Hook", "start": 446.0, "end": 512.0, "skip_lyrics": ("click", "hey")},
    {"source": "Crowd 2", "name": "Crowd 2 Hook", "start": 446.0, "end": 512.0, "skip_lyrics": ("click", "hey")},
    {"source": "Crowd 3", "name": "Crowd 3 Hook", "start": 446.0, "end": 512.0, "skip_lyrics": ("click", "hey")},
    {"source": "Harmony", "name": "Harmony Final", "start": 447.0, "end": 512.0, "skip_lyrics": ()},
]

# Vocal faders (dB), pans and reverb sends. The AI lead is the anchor at 0 dB, dead centre; everything
# else supports it from the sides. The aliens and the crowd carry most of the jokes in the call-and-response
# gaps (post-chorus, bridge, outro), so their main channels sit 2.5-5 dB higher than their supporting parts
# (the verse-2 dialogue, the final-chorus hook); the second, bigger alien is loud enough that its slower
# warble beats against the first alien in the ワレワレハ chants.
TRACK_MIX: dict[str, dict] = {
    "AI":            {"gain_db": 0.0, "pan": 0.0, "reverb_send": 0.22},
    "AI Verses":     {"gain_db": -2.5, "pan": 0.0, "reverb_send": 0.26},
    "AI Pre":        {"gain_db": -1.0, "pan": 0.0, "reverb_send": 0.22},
    "AI Bridge":     {"gain_db": 2.5, "pan": 0.0, "reverb_send": 0.22},
    "AI Breakdown":  {"gain_db": -2.5, "pan": 0.0, "reverb_send": 0.38},
    "AI Final":      {"gain_db": 2.5, "pan": 0.0, "reverb_send": 0.22},
    "AI Double":     {"gain_db": -6.5, "pan": -0.35, "reverb_send": 0.22},
    "Harmony":       {"gain_db": -6.5, "pan": 0.35, "reverb_send": 0.30},
    "Harmony Final": {"gain_db": -8.0, "pan": 0.35, "reverb_send": 0.30},
    "Aliens":        {"gain_db": 1.5, "pan": 0.25, "reverb_send": 0.30},
    "Aliens Verse":  {"gain_db": -1.0, "pan": 0.25, "reverb_send": 0.30},
    "Aliens Bridge": {"gain_db": 3.0, "pan": 0.3, "reverb_send": 0.30},
    "Aliens Hook":   {"gain_db": -4.0, "pan": 0.25, "reverb_send": 0.30},
    "Aliens 2":      {"gain_db": 0.5, "pan": -0.35, "reverb_send": 0.40},
    "Crowd":         {"gain_db": -2.0, "pan": -0.60, "reverb_send": 0.35},
    "Crowd 2":       {"gain_db": -2.0, "pan": 0.60, "reverb_send": 0.35},
    "Crowd 3":       {"gain_db": -4.0, "pan": 0.25, "reverb_send": 0.35},
    "Crowd Hook":    {"gain_db": -7.0, "pan": -0.60, "reverb_send": 0.35},
    "Crowd 2 Hook":  {"gain_db": -7.0, "pan": 0.60, "reverb_send": 0.35},
    "Crowd 3 Hook":  {"gain_db": -9.0, "pan": 0.25, "reverb_send": 0.35},
}

# Master / mix settings (MasterSettings fields, set with update_project). mix_overrides are keyed by
# vocal track name or backing stem name ("drums snare" is the snare sub-stem of the "drums" kit).
# lead EQ on top of the channel strip: its fundamental (A4-E5 = 440-660 Hz) piled up as a honky bump in the
# long-term spectrum, the strip's 3.2 kHz presence bell stuck out of it as a narrow peak, and the voice fell
# off a cliff above 3.8 kHz (5-8 dB under the band from 4 to 9 kHz), so: less 600 Hz, a softer 3.2 kHz, a
# broad 4.8 kHz presence lift and a little air
# neural stems (Q6, 9b): the 4.8 kHz bell and the 9 kHz shelf lifted the band where the neural voice is mostly
# aspiration noise, and the 600 Hz cut took its fundamental down: no high boosts, half the 600 Hz cut
LEAD_EQ_FORMANT = [["peak", 600, -4.0, 0.9], ["peak", 3150, -2.5, 1.4], ["peak", 4800, 3.5, 0.9], ["highshelf", 9000, 2.0, 0.7]]
LEAD_EQ = [["peak", 600, -2.0, 0.9], ["peak", 3150, -2.5, 1.4]]
if os.environ.get("DEMO3_LEAD_EQ"):
    LEAD_EQ = LEAD_EQ_FORMANT if os.environ["DEMO3_LEAD_EQ"] == "formant" else json.loads(os.environ["DEMO3_LEAD_EQ"])
# the lead's strip: the EQ above, a firmer de-harsher for the bright F5 vowels and a transient limiter for
# the needle-like 't' / 'k' release bursts of the verse patter
LEAD_STRIP = {"eq_add": LEAD_EQ, "deharsh": {"rel_db": -6.0, "max_cut_db": 5.0}, "crest_db": 12.0}
if not CHAIN:  # the demo-3 final mix: a shelf for the consonants on top
    LEAD_STRIP["eq_add"] = [*LEAD_EQ, ["highshelf", 6500, 3.0, 0.7]]
# the song's hall (room 0.55, RT60 1.7 s) at a low send left the gaps between the syllables open; a short room
# (room 0.10, RT60 0.6 s) at twice the send fills them and costs fewer words than a 0.95 s plate at the same send
# (prod/fxlab.py: lead + returns, chorus holes 8.8 % -> 5.1 %, relaxed verse 10.5 % -> 5.7 %)
REVERB_ROOM = float(os.environ.get("PROD_ROOM", "0.10")) if CHAIN else 0.55
# reverb sends of the channels that go through the chain (the song: lead 0.22-0.26, double 0.22, harmony 0.30)
CHAIN_SENDS = {"AI": 0.50, "AI Verses": 0.50, "AI Pre": 0.50, "AI Bridge": 0.50, "AI Final": 0.50,
               "AI Breakdown": 0.80, "AI Double": 0.40, "Harmony": 0.50, "Harmony Final": 0.50}
# the aliens' low voice was the main source of 200-400 Hz mud (up to 47 % of the master there) and its
# consonant band sat under the band in the bridge: less 280 Hz, more 2.6 kHz, and a tighter transient limit
# (its buzzy pulse train peaks 16 dB over its loud level and set off the master limiter)
ALIEN_EQ = [["peak", 280, -3.0, 1.0], ["peak", 2600, 2.0, 1.0]]
MASTER: dict = {
    "loudness_target": -11.5,  # (-11 kept the master limiter over 1 dB for ~20 % of the final chorus)
    "ceiling_db": -1.0,
    "glue": 0.35,              # lighter master glue: at 0.45 it ate most of the final chorus' extra level (and
                               # pumped); under 0.3 the limiter took over (over 1 dB for 13 % of the final chorus)
    "vocal_balance_db": 8.0,   # the vocals' 1-5 kHz level over the band (the song: 5.2; demo-3 final mix: 8.0)
    "band_width": 1.3,         # the band was narrow (side 9-11 dB under mid): open the L/R guitars and synths
    # master EQ (folded into every source): the default subsonic high-pass, some 250 Hz warmth (every role
    # preset scoops 250-450 Hz, which left a 3-4 dB hole in the long-term spectrum), a little less of the
    # 630 Hz honk of the vocal fundamentals and a softer air shelf
    "master_eq": [["hp", 25, 0, 0.7071, 2], ["peak", 250, 2.0, 1.2], ["peak", 630, -1.0, 1.4],
                  ["highshelf", 12000, 0.5, 0.7]],
    "vocal_delay": 0.3,        # dotted-8th ping-pong on the vocals: it blooms in the gaps and stops ...
    "delay_duck_db": 8.0,      # ... and steps back while they sing
    "delay_beats": 0.75,
    "delay_feedback": 0.3,
    "mix_overrides": {
        "AI": LEAD_STRIP,  # (the split lead channels inherit it)
        "AI Breakdown": {"eq_add": LEAD_EQ, "delay": 1.4},
        "AI Double": {"eq_add": LEAD_EQ, "delay": 0.4},
        # delay sends (1 = default): the lead gets the full send, the gang shouts stay dry and punchy.
        # crest_db: the aliens' buzzy low pulse train peaks 16 dB over its loud level (the AI: 10 dB), and those
        # spikes made the master limiter duck the whole mix by up to 5 dB; the shouts are peaky too.
        # Harmony: the 600 Hz cut of the lead EQ (its fundamentals were part of the 500-630 Hz honk)
        "Harmony": {"delay": 0.5, "eq_add": [["peak", 600, -1.5, 0.9]]},
        "Aliens": {"delay": 0.6, "crest_db": 8.0, "eq_add": ALIEN_EQ},
        "Aliens 2": {"delay": 0.8, "eq_add": [["peak", 280, -3.0, 1.0]]},
        # the final-chorus hook two octaves under the lead: out of the lead's 300 Hz body and 2.8 kHz bite
        "Aliens Hook": {"delay": 0.3, "crest_db": 8.0, "eq_add": [["peak", 300, -2.5, 1.0], ["peak", 2800, -3.0, 1.0]]},
        "Crowd": {"delay": 0.0, "crest_db": 12.0}, "Crowd 2": {"delay": 0.0, "crest_db": 12.0},
        "Crowd 3": {"delay": 0.0, "crest_db": 12.0},
        # band balance. The snare was the brightest, hissiest element of every section (45-70 % of the master's
        # 5-9 kHz) and the lead's main rival in the presence band: less level and top (its spectrum rose
        # 8-10 dB from 4 to 11 kHz), and a deeper vocal-keyed 2.6 kHz carve (3 dB instead of 1.2)
        "drums snare": {"gain_db": -2.0, "carve": 1.0,
                        "eq_add": [["peak", 3000, -2.0, 0.8], ["highshelf", 5500, -6.5, 0.7]]},
        # demo-3 final mix: hats 1.5 dB lower and ducked under the vocals
        "drums hats": {"gain_db": -1.5, "duck": 1.5},
        # the kick's 50-60 Hz was a bit boomy against the 80-250 Hz body of the mix, its beater click was the
        # loudest high-frequency event of the verses, and the kick and the bass set off most of the master
        # limiter's gain reduction: a little less of both, and the kick's transients held within 7 dB
        "drums kick": {"gain_db": -1.0, "eq_add": [["lowshelf", 60, -1.5, 0.7], ["highshelf", 6000, -3.0, 0.7]],
                       "crest_db": 7.0},
        "bass": {"gain_db": -1.0, "crest_db": 10.0},
        # the anti-phase (mostly side) cymbals and pad fell away in mono: narrower
        "drums cymbals": {"width": 0.7},
        # the half-time verses were thin, so their plucks, slap bass and chugs come up; the pluck's 200-800 Hz
        # sat in the lead's fundamental range and its peaks (and the slap's pops) set off the limiter
        "pluck": {"gain_db": 3.5, "crest_db": 9.0, "eq_add": [["peak", 550, -3.0, 1.0]]},
        "slap": {"gain_db": 1.5, "crest_db": 12.0},
        "mute": {"gain_db": 2.0},
        # the chip-lead riff was level with the voices in the riff sections (the aliens' spam in the intro, the
        # post-chorus calls, the aliens' one-star review in the outro), and the intro's radio static with it.
        # The chorus hook tags (chip lead, bell) overlapped the AI's phrase-end notes: they duck under the
        # vocals (vocal-keyed, x 1.5 dB) and get out of the way as soon as the vocal stops; the power chords
        # leave a little room at 1 kHz for the lead's held low notes
        "lead": {"gain_db": -3.5, "duck": 3.0},
        "bell": {"duck": 2.5},
        "gtrL": {"eq_add": [["peak", 1000, -1.5, 1.0]]},
        "gtrR": {"eq_add": [["peak", 1000, -1.5, 1.0]]},
        "fx": {"gain_db": -1.5},
        "mbox": {"gain_db": -2.0},
        # the final chorus is the only chorus with the pad and the low chip lead: a little more of both
        "pad": {"gain_db": 3.0, "width": 0.75},
        "leadlo": {"gain_db": 2.0},
    },
}


# Section-level band tweaks, applied with set_backing(merge=true) on top of backing.json: velocity
# multipliers for the parts of a section. (The drums and the bass have channel compressors, which undo about
# two thirds of a velocity change; the keys, pads and guitars follow it.)
# * The choruses barely lifted: the second half of the intro and of each pre-chorus were as loud and bright
#   as the chorus, with more low end, so the band steps back there and the hook entries land (with the "AI Pre"
#   split, and the intro's drums and bass now entering at bar 7, see backing.json).
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
    "Bridge": {"drums": 0.8, "gtrL": 0.75, "gtrR": 0.75, "chip": 0.75, "growl": 0.8, "saws": 0.8, "arp": 0.8},
    "Breakdown": {"pad": 0.75, "mbox": 0.8},
    "Final chorus": {"gtrL": 1.25, "gtrR": 1.25, "saws": 1.5, "mute": 1.15, "pad": 1.2, "drums": 1.12, "bass": 1.12},
    "Outro": {"lead": 0.7, "leadlo": 0.8, "drums": 0.85, "gtrL": 0.85, "gtrR": 0.85},
}


_NOTE_NAMES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "G#", "A", "Bb", "B"]
_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _tr_name(letter: str, acc: str, n: int) -> str:
    return _NOTE_NAMES[(_PC[letter] + {"#": 1, "b": -1, "": 0}[acc] + n) % 12]


def transpose_chord(sym: str, n: int) -> str:
    """'Am/G' moved by n semitones ('N.C.' and other symbols without a root are returned unchanged)."""
    m = re.match(r"^([A-G])([#b]?)([^/]*)(?:/([A-G])([#b]?))?$", sym.strip())
    if not m:
        return sym
    out = _tr_name(m.group(1), m.group(2), n) + m.group(3)
    if m.group(4):
        out += "/" + _tr_name(m.group(4), m.group(5), n)
    return out


def transpose_arrangement(arrangement: dict, n: int) -> dict:
    """A copy of the backing arrangement moved by n semitones: chord symbols, keys, the riffs ('notes' patterns,
    through each section's `transpose`) and pitched fx events. Sections with `like` inherit from their source."""
    arr = json.loads(json.dumps(arrangement))
    if not n:
        return arr

    def key(k):
        m = re.match(r"^([A-G])([#b]?)(.*)$", k)
        return _tr_name(m.group(1), m.group(2), n) + m.group(3) if m else k

    def pitch(p):
        if isinstance(p, list):
            return [pitch(x) for x in p]
        m = re.match(r"^([A-G])([#b]?)(-?\d+)$", str(p))
        if not m:
            return p
        midi = _PC[m.group(1)] + {"#": 1, "b": -1, "": 0}[m.group(2)] + 12 * (int(m.group(3)) + 1) + n
        return f"{_NOTE_NAMES[midi % 12]}{midi // 12 - 1}"

    if isinstance(arr.get("key"), str):
        arr["key"] = key(arr["key"])
    for sec in arr.get("sections", []):
        if "chords" in sec:
            sec["chords"] = [transpose_chord(c, n) for c in sec["chords"]]
            sec["transpose"] = int(sec.get("transpose", 0)) + n
        if isinstance(sec.get("key"), str):
            sec["key"] = key(sec["key"])
        for ev in sec.get("events", []) or []:
            if "pitch" in ev:
                ev["pitch"] = pitch(ev["pitch"])
    return arr


def split_sources() -> dict[str, str]:
    """Split channel name -> the track it was split from (resolved to the original track)."""
    src = {sp["name"]: sp["source"] for sp in SPLITS}
    return {name: (src[s] if s in src else s) for name, s in src.items()}


# channels processed by prod/chain4.py in the backend: the renderer's strip (EQ, de-esser, compressor) is bypassed;
# their delay sends
CHAIN_CHANNELS = {"AI": {"chain": False}, "AI Breakdown": {"chain": False, "delay": 1.4},
                  "AI Double": {"chain": False, "delay": 0.4}, "Harmony": {"chain": False, "delay": 0.5}}


def master_settings() -> dict:
    """MASTER with every split channel inheriting its source's mix_overrides entry (unless it has its own)."""
    m = json.loads(json.dumps(MASTER))
    mo = m["mix_overrides"]
    if CHAIN:
        mo.update(json.loads(json.dumps(CHAIN_CHANNELS)))
        m["reverb_room"] = REVERB_ROOM
    for name, source in split_sources().items():
        if name not in mo and source in mo:
            mo[name] = json.loads(json.dumps(mo[source]))
    return m


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


def split_track(s: Session, source: str, name: str, start: float, end: float, skip_lyrics=()) -> int:
    """Move the notes of ``source`` with onsets in [start, end) to a new track ``name`` (same voice,
    settings and curves). Returns the number of notes moved."""
    src = s.project.track(source)
    pick = lambda n: start <= n.onset < end and (n.lyric or "").lower() not in skip_lyrics  # noqa: E731
    moved = [n for n in src.notes if pick(n)]
    if not moved:
        return 0
    if any(t.name == name for t in s.project.tracks):  # a second range for an existing split channel
        call_tool(s, "add_notes", {"track": name, "notes": [
            {"onset": n.onset, "duration": n.duration, "pitch": n.pitch, "lyric": n.lyric,
             **({"phonemes": n.phonemes} if n.phonemes else {}), **({"attrs": n.attrs} if n.attrs else {})}
            for n in moved]})
    else:
        call_tool(s, "duplicate_track", {"track": source, "name": name})
        dst = s.project.track(name)
        keep = {(n.onset, n.pitch) for n in moved}
        call_tool(s, "delete_notes", {"track": name, "ids": [n.id for n in dst.notes
                                                             if (n.onset, n.pitch) not in keep]})
        tracks = s.project.tracks  # put the split channel after its source (and its earlier splits) in the mixer
        tracks.remove(dst)
        splits = {sp["name"] for sp in SPLITS}
        pos = tracks.index(src) + 1
        while pos < len(tracks) and tracks[pos].name in splits:
            pos += 1
        tracks.insert(pos, dst)
    call_tool(s, "delete_notes", {"track": source, "ids": [n.id for n in moved]})
    return len(moved)


# ------------------------------------------------------------------------------------------- build
def build_session(log=print) -> Session:
    """The finished project in a tool session (vocals, production, backing, mixer, master)."""
    s = Session()
    r = call_tool(s, "new_project", {"score": (HERE / "vocals.s2s").read_text(encoding="utf-8")})
    for w in r.get("warnings", []) or []:
        log(f"  score warning: {w}")
    if VERSE == "relaxed":
        from synth2.score import parse_notes

        ai = s.project.track("AI")
        old = [n.id for n in ai.notes if 32.0 <= n.onset < 96.0]
        new = parse_notes((PROD / "relaxed_verse.s2s").read_text(encoding="utf-8"), start_beat=32.0)
        call_tool(s, "delete_notes", {"track": "AI", "ids": old})
        call_tool(s, "add_notes", {"track": "AI", "notes": [
            {"onset": n.onset, "duration": n.duration, "pitch": n.pitch, "lyric": n.lyric,
             **({"phonemes": n.phonemes} if n.phonemes else {}), **({"attrs": n.attrs} if n.attrs else {})}
            for n in new]})
        log(f"  verse 1: {len(old)} written notes replaced by {len(new)} notes of relaxed_verse.s2s")
    prod = vocals.apply_vocal_production(s)
    log(f"  vocal production: {len(prod['tracks'])} tracks, AI Double {prod['ai_double_notes']} notes, "
        f"Harmony {prod['harmony_notes']} notes, fan-voice regions {prod['fan_chop_regions']}")
    if KEY_SHIFT:
        for t in s.project.tracks:
            for n in t.notes:
                n.pitch += KEY_SHIFT
        log(f"  key: every vocal note and the backing moved by {KEY_SHIFT:+d} semitones")
    arrangement = transpose_arrangement(json.loads((HERE / "backing.json").read_text(encoding="utf-8")), KEY_SHIFT)
    r = call_tool(s, "set_backing", {"arrangement": arrangement})
    for w in r.get("warnings", []):
        log(f"  backing {w}")
    if BAND_TWEAKS:
        r = call_tool(s, "set_backing", {"arrangement": {"sections": tweak_sections(arrangement, BAND_TWEAKS)},
                                         "merge": True})
        for w in r.get("warnings", []):
            log(f"  backing {w}")
        log(f"  band tweaks: {BAND_TWEAKS}")
    for sp in SPLITS:
        moved = split_track(s, sp["source"], sp["name"], sp["start"], sp["end"], sp["skip_lyrics"])
        log(f"  split {moved} notes of {sp['source']} (beats {sp['start']:g}-{sp['end']:g}) to {sp['name']}")
    names = {t.name for t in s.project.tracks}
    for name, mix in TRACK_MIX.items():
        if name in names:
            call_tool(s, "update_track", {"track": name, **mix})
    if CHAIN:
        for name, send in CHAIN_SENDS.items():
            if name in names:
                call_tool(s, "update_track", {"track": name, "reverb_send": send})
    call_tool(s, "update_project", {"master": master_settings()})
    return s


# ------------------------------------------------------------------------------------------- level cache
def _mix_key(project, sr: int) -> str:
    """A fingerprint of the mix that is stable across builds (the renderer's own cache key uses track ids,
    which are new on every build): master settings, backing, mixer and notes by track name."""
    import hashlib

    tracks = [(t.name, t.voice, t.gain_db, t.pan, t.mute, t.solo, t.reverb_send,
               [(round(n.onset, 4), n.pitch, n.lyric) for n in t.notes]) for t in project.tracks]
    blob = json.dumps({"master": project.master.to_dict(), "backing": project.backing, "tracks": tracks, "sr": sr,
                       "tempo": project.tempo.to_list()}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def save_levels(out: Path, project, sr: int, report: dict) -> None:
    """Remember the full render's level decisions, so later section renders are mastered alike."""
    levels = {k: report["mix"].get(k) for k in _renderer._LEVEL_KEYS}
    (out / "mix_levels.json").write_text(json.dumps({"key": _mix_key(project, sr), "levels": levels}, indent=1))


def load_levels(out: Path, project, sr: int) -> bool:
    """Prime the renderer's level cache from ``mix_levels.json`` when it matches this mix."""
    try:
        d = json.loads((out / "mix_levels.json").read_text())
    except (OSError, ValueError):
        return False
    if d.get("key") != _mix_key(project, sr) or any(v is None for v in d.get("levels", {}).values()):
        return False
    _renderer._LEVEL_CACHE[_renderer._level_key(project, sr)] = d["levels"]
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


LEAD_TRACKS = ("AI", "AI Verses", "AI Pre", "AI Bridge", "AI Breakdown", "AI Final")  # the lead vocal (on several mixer channels)


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
    if rep.get("warnings"):
        print(f"\n{len(rep['warnings'])} engine warnings, e.g.: " + "; ".join(rep["warnings"][:3]))


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
    ap.add_argument("--no-stems", action="store_true", help="do not write the stems")
    ap.add_argument("--no-mp3", action="store_true", help="do not encode the mp3")
    ap.add_argument("--no-pitch", action="store_true", help="skip the pitch-accuracy analysis")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()

    print("building the project through the tool layer ...")
    s = build_session()
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
        reused = load_levels(out, project, args.sr)
        print(f"  section render: bars {a}-{b} (beats {start_beat:g}-{end_beat:g}); "
              + ("mastered with the levels of the last full render" if reused else
                 "no matching full render yet, so this section is mastered on its own"))

    print(f"rendering at {args.sr} Hz ...")
    t_render = time.perf_counter()

    def progress(frac: float, msg: str) -> None:
        print(f"  [{100 * frac:3.0f}% {time.perf_counter() - t_render:5.1f} s] {msg}", flush=True)

    res = render_project(project, sample_rate=args.sr, start_beat=start_beat, end_beat=end_beat, progress=progress)
    rep = res.report

    if partial:
        dest = out / "preview"
        wav = dest / f"{NAME}_bars_{a}-{b}.wav"
    else:
        dest = out
        wav = out / f"{NAME}.wav"
        save_levels(out, project, args.sr, rep)
    dest.mkdir(parents=True, exist_ok=True)
    write_wav(wav, res.audio, res.sample_rate, 24)
    print(f"  wrote {wav}")
    if not args.no_stems:
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
    ana = analyse(project, res, a, b, with_pitch=not args.no_pitch)
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
