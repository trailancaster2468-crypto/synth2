"""Vocal production for SELECT ALL INVADERS: the derived vocal tracks and the one-kind-per-track arrangement.

``build.py`` loads ``vocals.s2s`` into a :class:`synth2.ai.tools.Session` and then calls
:func:`apply_vocal_production` and :func:`apply_arrangement`. The score holds the parts that were written by
hand, one job per track: the lines (``AI``, ``Aliens``, ``Crowd``), the lead's calls in other voices' sections
(``AI Answer``), the ad-libs (``AI Adlib``, ``Alien Adlib``: sound words, interjections, the laugh, the
stutters), the chants (``Alien Chant``, ``Alien Chant 2``) and the gang shouts (``Crowd Shout``).

:func:`apply_vocal_production` derives the rest from them, so a change to the lead melody carries over:

1. **AI Double** (``duplicate_track``): the lead in the three choruses, including their pickups, 0.03 beats
   (~11 ms) late. With the neural engine a double is not a second performance: the track has the role
   ``double``, so its audio is the lead's own take over these notes (docs/NEURAL.md), which the mixer delays
   and detunes. With the formant engine it sings: nova again, +8 cents, a slightly smaller formant scale.
2. **Harmony**: a chord-aware close harmony under the lead, sung by pixel. Every lead note that is a chord
   tone gets the next chord tone below it (a 3rd, or a 4th below a root). A passing note gets the diatonic
   3rd below it, in E minor (F# minor in the final chorus), with the raised leading tone over the V chord.
   Notes on the "and" of beat 4 belong to the next bar's chord, because they anticipate it. Choruses 1 and 2
   are harmonized only on the hook lines (1, 3, 5, 7) and the tag line 8. The deadpan joke lines 2, 4 and 6
   stay solo. The final chorus is harmonized throughout, except its line 6 ("Funny... neither can you").
3. **Crowd 2 / Crowd 3** and **Crowd Shout 2 / Crowd Shout 3** (``duplicate_track``): the gang in unison.
   Crowd 2 is ember at the same pitch (-9 cents, 0.035 beats late), Crowd 3 is aurora an octave up (+6 cents,
   0.018 beats late). The breakdown's whispered "hurry" is not doubled an octave up.
4. **Fan voice**: the classic ワレワレハ ウチュウジン ダ effect, a voice speaking into a fan. Over every kana
   note of ``Alien Chant`` and ``Alien Chant 2`` an 11 Hz loudness chop (0 / -9 dB) is added to the loudness
   curve, and the pitch warble deepens to +-0.6 semitones. (The neural engine chops every chant syllable
   itself, locked to its start: ``neural.adlib.chop_hz``.)

:func:`apply_arrangement` then gives every kind of sound a track of its own (65 tracks, 22 kinds):

* **section channels** (:data:`SECTION_CHANNELS`): the lead in the verses, pre-choruses, bridge, breakdown
  and final chorus, the aliens' verse-2 and bridge lines and their final-chorus hook, the crowd's final
  hook and the final chorus' harmony, so each can sit differently in the mix without fader automation;
* **one kind per track** (:data:`KIND_CHANNELS`): the lead's click, ハイ ハイ and stutter; the aliens' ouch,
  clicks, laugh, stutter and ナンデ; their short answers and the unison with the lead; the crowd's answers,
  whisper and clicks apart from its line, hook and shouts;
* **copies** (:data:`COPIES`): the second double and the harmony's other side, doubles of a take;
* **second takes** (:data:`SECOND_TAKES`): every crowd voice and the low chant sing twice (another seed,
  detuned, a moment later), which is what makes three singers a gang;
* a **role** (:data:`ROLES`: ``lead``, ``double``, ``harmony``, ``answer``, ``adlib``) where the neural
  engine has a chain for the job, and the **cast** (:data:`CAST`): how the neural engine sings each track.
  The formant engine ignores both, so the song still builds without the models.

Every channel's seed is pinned (:data:`SEEDS`), so a take does not change when tracks are added or moved.

Run it on its own to print what it does::

    python examples/select_all_invaders/vocals.py
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

try:
    import synth2  # noqa: F401
except ImportError:  # run from a checkout without installing: python examples/select_all_invaders/vocals.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from synth2.ai.tools import Session, call_tool
from synth2.phonemes import japanese
from synth2.project import Curve, Note, Track

HERE = Path(__file__).resolve().parent
SCORE = HERE / "vocals.s2s"

BPM = 168.0
# one chord per bar, bars 1-134 (SPEC.md)
CHORDS: list[str] = (
    "Em Em C D Em Em C B "                                    # 1-8 intro
    + "Em Em/D C B " * 4                                      # 9-24 verse 1
    + "Am B Em D C B C D "                                    # 25-32 pre-chorus 1
    + "C D Bm Em " * 3 + "Am B Em Em "                        # 33-48 chorus 1
    + "Em C D B "                                             # 49-52 post-chorus
    + "Em Em/D C B " * 4                                      # 53-68 verse 2
    + "Am B Em D C B C D "                                    # 69-76 pre-chorus 2
    + "C D Bm Em " * 3 + "Am B Em Em "                        # 77-92 chorus 2
    + "F Em F Em Dm E Am B C C#dim D D#dim Em F F# B "        # 93-108 bridge
    + "Em C D B "                                             # 109-112 breakdown
    + "D E C#m F#m " * 3 + "Bm C# F#m F#m "                   # 113-128 final chorus (F# minor)
    + "F#m D E C# F#m N.C."                                   # 129-134 outro
).split()
assert len(CHORDS) == 134

CHORUS_STARTS = (128.0, 304.0, 448.0)  # bars 33, 77, 113
FINAL_CHORUS = (447.5, 512.0)
FINAL_LINE_6 = (488.0, 495.5)  # bars 123-124, "Funny... neither can you": left without harmony
WHISPER = (446.0, 447.4)       # bar 112, the crowd's whispered "hurry"
# chorus-relative beat ranges harmonized in choruses 1 and 2: hook lines 1, 3, 5, 7 (with pickups) and line 8
HOOK_LINES = ((-0.5, 6.0), (15.5, 22.0), (32.0, 38.0), (47.5, 54.0), (55.0, 64.0))

_PC = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7,
       "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
_QUALITY = {"": (0, 4, 7), "m": (0, 3, 7), "dim": (0, 3, 6, 9)}

# ------------------------------------------------------------------------------- one kind of sound per track
# Section channels: (source, channel, first beat, end beat (exclusive)). Bar B beat b = (B - 1) * 4 + (b - 1).
SECTION_CHANNELS = [
    ("AI", "AI Verses", 32.0, 96.0), ("AI", "AI Verses", 208.0, 272.0),          # verses 1 and 2
    ("AI", "AI Pre", 96.0, 127.5), ("AI", "AI Pre", 272.0, 303.5),               # the pre-choruses (not the pickup)
    ("AI", "AI Bridge", 368.0, 432.0), ("AI", "AI Breakdown", 432.0, 447.25), ("AI", "AI Final", 447.25, 512.0),
    ("Aliens", "Aliens Verse", 208.0, 272.0), ("Aliens", "Aliens Bridge", 368.0, 416.0),
    ("Aliens", "Aliens Hook", 447.0, 501.0),                                     # the hook two octaves under the AI
    ("Crowd", "Crowd Hook", 446.0, 512.0), ("Crowd 2", "Crowd 2 Hook", 446.0, 512.0),
    ("Crowd 3", "Crowd 3 Hook", 446.0, 512.0),
    ("Harmony", "Harmony Final", 447.0, 512.0),
    ("AI Adlib", "AI Adlib Bridge", 368.0, 432.0),
    ("Alien Adlib", "Alien Adlib Verse", 208.0, 272.0), ("Alien Adlib", "Alien Adlib Bridge", 368.0, 416.0),
]
# One kind per track: (source, first beat, end beat (exclusive), lyrics or None = every note, channel). A channel
# keeps the voice, the curves and the seed of the track its notes were written on. Tracks left without a note
# (AI Adlib, Alien Adlib and their section channels) are removed.
KIND_CHANNELS = [
    # the lead's ad-libs: three different sounds
    ("AI Adlib", 204.0, 206.0, None, "AI Click"),                 # bar 52: click x4 (sound word)
    ("AI Adlib", 280.0, 282.0, None, "AI Hai"),                   # bar 71: ハイ ハイ (interjection)
    ("AI Adlib Bridge", 368.0, 370.0, None, "AI Stutter"),        # bar 93: re- re- re- (stutter into "reboot")
    # the aliens' ad-libs: five different sounds
    ("Alien Adlib", 198.0, 199.0, None, "Alien Ouch"),            # bar 50
    ("Alien Adlib", 204.0, 206.0, None, "Alien Click"),           # bar 52: click x4
    ("Alien Adlib", 470.0, 471.0, None, "Alien Click"),           # bar 118: "click?"
    ("Alien Adlib Verse", 262.0, 264.0, None, "Alien Laugh"),     # bar 66: ha ha ha
    ("Alien Adlib Bridge", 371.0, 373.0, None, "Alien Stutter"),  # bar 93: re- re- re- (into "reply all")
    ("Alien Adlib Bridge", 395.0, 400.0, None, "Alien Nande"),    # bars 99, 100: ナンデ
    # the aliens' lines apart from their short answers to the lead and the unison with the lead
    ("Aliens", 78.0, 80.0, None, "Alien Answer"),                 # bar 20: "cookies? delicious"
    ("Aliens", 118.0, 120.0, None, "Alien Answer"),               # bar 30: "the mothership"
    ("Aliens", 199.0, 204.0, None, "Alien Answer"),               # bars 50-51: "thank you", "did we pass"
    ("Aliens", 286.0, 288.0, None, "Alien Answer"),               # bar 72: "four minutes"
    ("Aliens", 341.5, 344.0, None, "Alien Answer"),               # bar 86: "we are not a robot"
    ("Aliens Bridge", 372.5, 376.0, None, "Alien Answer Bridge"),  # bar 94: "reply all"
    ("Aliens Bridge", 403.0, 408.0, None, "Alien Answer Bridge"),  # bars 101-102: "no, you are", "respectfully"
    ("Aliens Bridge", 408.0, 416.0, None, "Alien Unison"),        # bars 103-104: "who is who ..." with the lead
    # the crowd's line apart from its echoes and answers and (on the hook channels) the whisper
    ("Crowd", 110.0, 112.0, None, "Crowd Answer"), ("Crowd 2", 110.0, 112.0, None, "Crowd 2 Answer"),
    ("Crowd 3", 110.0, 112.0, None, "Crowd 3 Answer"),            # bar 28: "four oh four"
    ("Crowd", 518.0, 520.0, None, "Crowd Answer"), ("Crowd 2", 518.0, 520.0, None, "Crowd 2 Answer"),
    ("Crowd 3", 518.0, 520.0, None, "Crowd 3 Answer"),            # bar 130: "five stars"
    ("Crowd Hook", 446.0, 447.5, ("hurry",), "Crowd Whisper"),
    ("Crowd 2 Hook", 446.0, 447.5, ("hurry",), "Crowd 2 Whisper"),
    ("Crowd 3 Hook", 446.0, 447.5, ("hurry",), "Crowd 3 Whisper"),  # bar 112: "hurry"
    # the shouts: the sound word "click" (22 of the 32 hits) apart from the shouted words
    ("Crowd Shout", 0.0, 1e9, ("click",), "Crowd Click"), ("Crowd Shout 2", 0.0, 1e9, ("click",), "Crowd Click 2"),
    ("Crowd Shout 3", 0.0, 1e9, ("click",), "Crowd Click 3"),
]
# Copies of another track's take (source, copy, role of the copy, cents): the second double; the harmony's
# other side. The neural engine does not sing a copy (the mixer delays and detunes the take it doubles); the
# formant engine sings it, detuned by the cents given here so that it is not the same signal twice.
COPIES = [("AI Double", "AI Double R", "double", -13.0), ("Harmony", "Harmony L", "double:Harmony", -7.0),
          ("Harmony Final", "Harmony Final L", "double:Harmony Final", -7.0)]
# Seeds: a new track's default seed comes from its position in the project, which every added track would
# shift. These are the seeds the takes were chosen with.
SEEDS = {
    "AI Double": 31693, "AI Double R": 31693, "Harmony": 55450, "Harmony L": 55450,
    "Crowd 2": 47531, "Crowd 3": 55450, "Crowd Shout 2": 47545, "Crowd Shout 3": 55464,
    "AI Verses": 63369, "AI Pre": 71288, "AI Bridge": 79207, "AI Breakdown": 87126, "AI Final": 95045,
    "Aliens Verse": 102964, "Aliens Bridge": 110883, "Aliens Hook": 118802, "Crowd Hook": 126721,
    "Crowd 2 Hook": 134640, "Crowd 3 Hook": 142559, "Harmony Final": 150478, "Harmony Final L": 150478,
    "AI Adlib Bridge": 79221, "Alien Adlib Verse": 102978, "Alien Adlib Bridge": 110897,
}
# Roles in the mix (track name, longest prefix wins; docs/NEURAL.md). A role decides how the neural engine
# sings a track (a double copies a take, an answer and a harmony draw no breaths, an ad-lib is sung hit by
# hit) and which chain the mixer gives it. The aliens and the crowd have no role: they keep the mixer's
# channel strip, and their place in the mix is set per track in build.py.
ROLES = {"AI": "lead", "AI Double": "double", "Harmony": "harmony", "AI Answer": "answer",
         "AI Click": "adlib", "AI Hai": "adlib", "AI Stutter": "adlib"}
# The cast: how the neural engine sings each track (track name, longest prefix wins; ``neural.*`` voice
# parameters, docs/NEURAL.md). "gender" is the formant shift the voice ends up with, whatever the track's
# formant_scale (a formant-engine setting) would add to it.
# The lead, its double, the harmony and the answers are sung with the engine's defaults (the crisp dial).
# PLAIN: the other voices were chosen with none of the dial's consonant options and no breaths of their own.
PLAIN = {"neural.breaths": 0, "neural.cons_cap": 0, "neural.cons_cap_share": 0, "neural.cons_floor": 0,
         "neural.burst": 0, "neural.onset_keep": 0, "neural.vowel_breath_db": 0, "neural.detune": 0}
ADLIB = {"neural.adlib": 1}                                    # detached hits, one stamped take per sound
ALIEN_ADLIB = {**ADLIB, "neural.adlib.vibrato": 1, "neural.adlib.bend.ha": -3, "neural.adlib.bend.ouch": -3}
CHANT = {**ADLIB, "neural.adlib.split": 1, "neural.adlib.vibrato": 1, "neural.adlib.chop_hz": 11,
         "neural.adlib.chop_db": -9, "neural.adlib.glottal": 1}  # per syllable, the fan chop locked to each one
SHOUT = {**ADLIB, "neural.adlib.bend.what": 3, "neural.adlib.bend.huh": 3}
CAST: dict[str, dict] = {
    "AI Answer": {"gender": 0.2},
    "AI Click": {"gender": 0.3}, "AI Hai": {"gender": 0.3}, "AI Stutter": {"gender": 0.0},
    "Aliens": {"gender": -0.6, **PLAIN}, "Alien Answer": {"gender": -0.6, **PLAIN},
    "Alien Unison": {"gender": -0.6, **PLAIN},
    "Alien Ouch": {"gender": -0.75, **ALIEN_ADLIB}, "Alien Laugh": {"gender": -0.75, **ALIEN_ADLIB},
    # the take of the stamped "click" was chosen by audition: this one is heard as "click"
    "Alien Click": {"gender": -0.75, **ALIEN_ADLIB, "neural.adlib.salt.click": 2},
    "Alien Nande": {"gender": -0.6, **ALIEN_ADLIB}, "Alien Stutter": {"gender": -0.6, **ALIEN_ADLIB},
    "Alien Chant": {"gender": -0.6, **CHANT},
    # the low chant: the voice at the written pitch, as dark as it goes; its second take sings an octave down
    "Alien Chant 2": {"gender": -1.0, "neural.tone": 0, "neural.detune": 10, **CHANT},
    "Crowd": {"gender": -0.3, **PLAIN}, "Crowd 2": {"gender": -0.7, **PLAIN}, "Crowd 3": {"gender": 0.2, **PLAIN},
    "Crowd Shout": {"gender": -0.3, **SHOUT}, "Crowd Shout 2": {"gender": -0.7, **SHOUT},
    "Crowd Shout 3": {"gender": 0.2, **SHOUT},
    "Crowd Click": {"gender": -0.3, **SHOUT, "neural.adlib.salt.click": 1},
    "Crowd Click 2": {"gender": -0.7, **SHOUT},
    "Crowd Click 3": {"gender": 0.2, **SHOUT, "neural.adlib.salt.click": 1},
}
# Second takes (track name, longest prefix wins): the track is sung once more with another seed (+101), as
# "<name> B": transposed, detuned (cents), later (s), with another formant shift ("gender", added to the
# track's) and at a level against the first take (dB). The mixer treats both takes alike.
SECOND_TAKES: dict[str, dict] = {
    "Crowd": {"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5},
    "Crowd 2": {"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5},
    "Crowd 3": {"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5},
    "Crowd Shout": {"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5},
    "Crowd Shout 2": {"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5},
    "Crowd Shout 3": {"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5},
    "Crowd Click": {"detune": 14, "delay": 0.021, "gender": -0.3, "gain_db": -1.5},
    "Crowd Click 2": {"detune": -11, "delay": 0.017, "gender": 0.3, "gain_db": -1.5},
    "Crowd Click 3": {"detune": 9, "delay": 0.026, "gender": -0.15, "gain_db": -1.5},
    "Alien Chant 2": {"transpose": -12, "tone": 12, "detune": 10, "delay": 0.0, "gender": 0.4, "gain_db": -4.0},
}


def chord_tones(chord: str) -> set[int]:
    """Pitch classes of a chord symbol such as ``Em``, ``C#dim`` or ``Em/D`` (the slash bass is ignored)."""
    if chord == "N.C.":
        return set()
    name = chord.split("/")[0]
    root = name[:2] if len(name) > 1 and name[1] in "#b" else name[:1]
    r = _PC[root]
    return {(r + i) % 12 for i in _QUALITY[name[len(root):]]}


def chord_at(beat: float) -> str:
    """The chord sounding at ``beat``. A note on the "and" of beat 4 anticipates the next bar's chord."""
    if abs(beat % 4.0 - 3.5) < 1e-6:
        beat += 0.5
    return CHORDS[min(int(beat // 4.0), len(CHORDS) - 1)]


def _scale(beat: float, chord: str) -> set[int]:
    if beat >= FINAL_CHORUS[0] - 1e-6:   # F# minor, E# over C#
        pcs = {6, 8, 9, 11, 1, 2, 4}
        return pcs | {5} if chord.startswith("C#") and "m" not in chord else pcs
    pcs = {4, 6, 7, 9, 11, 0, 2}          # E minor, D# over B
    return pcs | {3} if chord == "B" else pcs


def harmony_pitch(pitch: int, beat: float) -> int:
    """Close harmony note under ``pitch`` at ``beat`` (see the module docstring)."""
    chord = chord_at(beat)
    tones = chord_tones(chord)
    if pitch % 12 in tones:
        for d in (3, 4, 5):
            if (pitch - d) % 12 in tones:
                return pitch - d
    scale = _scale(beat, chord)
    for d in (3, 4):
        if (pitch - d) % 12 in scale:
            return pitch - d
    return pitch - 3


def _in(beat: float, ranges) -> bool:
    return any(a - 1e-6 <= beat < b - 1e-6 for a, b in ranges)


def chorus_ranges() -> list[tuple[float, float]]:
    """Beat ranges of the three choruses, including the pickup note before each."""
    return [(c - 0.5, c + 64.0) for c in CHORUS_STARTS]


def harmony_ranges() -> list[tuple[float, float]]:
    out = [(c + a, c + b) for c in CHORUS_STARTS[:2] for a, b in HOOK_LINES]
    return out + [(FINAL_CHORUS[0], FINAL_LINE_6[0]), (FINAL_LINE_6[1], FINAL_CHORUS[1])]


def _track(session: Session, name: str) -> Track:
    for t in session.project.tracks:
        if t.name == name:
            return t
    raise LookupError(f"vocals.s2s has no track named {name!r}")


def _fan_chop(track: Track, hz: float = 11.0, depth_db: float = -9.0, warble: float = 0.6,
              bpm: float = BPM) -> int:
    """Fan voice on the kana notes of ``track``: an ``hz`` loudness chop plus a ``warble`` semitone vibrato.

    Returns the number of chopped regions.
    """
    kana = sorted((n for n in track.notes if japanese.is_kana(n.lyric)), key=lambda n: n.onset)
    if not kana:
        return 0
    for n in kana:
        n.attrs.setdefault("vibrato_depth", warble)
    # merge kana notes that touch into regions
    regions: list[list[float]] = []
    for n in kana:
        a, b = n.onset - 0.05, n.end + 0.05
        if regions and a <= regions[-1][1] + 0.3:
            regions[-1][1] = max(regions[-1][1], b)
        else:
            regions.append([a, b])
    old = sorted(track.curves["loudness"].points) if "loudness" in track.curves else []
    if old:
        xs = np.array([p[0] for p in old])
        ys = np.array([p[1] for p in old])
        base = lambda b: float(np.interp(b, xs, ys))  # noqa: E731
    else:
        base = lambda b: 0.0  # noqa: E731
    half = 0.5 * bpm / 60.0 / hz  # beats per half cycle
    pts = [(x, y) for x, y in old if not any(a <= x <= b for a, b in regions)]
    for a, b in regions:
        pts.append((a - 0.02, base(a - 0.02)))
        k, x = 0, a
        while x <= b:
            pts.append((x, base(x) + (0.0 if k % 2 == 0 else depth_db)))
            x += half
            k += 1
        pts.append((b + 0.02, base(b + 0.02)))
    pts.sort()
    track.curves["loudness"] = Curve(points=[(round(x, 4), round(y, 3)) for x, y in pts], interpolation="cubic")
    return len(regions)


def _remove(session: Session, names) -> None:
    session.project.tracks = [t for t in session.project.tracks if t.name not in names]


def apply_vocal_production(session: Session) -> dict:
    """Add AI Double, Harmony, the crowd's doubles (lines and shouts) and the aliens' fan chop. Safe to call twice."""
    p = session.project
    derived = ("AI Double", "Harmony", "Crowd 2", "Crowd 3", "Crowd Shout 2", "Crowd Shout 3")
    _remove(session, derived)
    lead, crowd = _track(session, "AI"), _track(session, "Crowd")

    # 1) AI Double: the choruses only
    call_tool(session, "duplicate_track", {"track": lead.id, "name": "AI Double", "detune_cents": 8.0,
                                           "delay_beats": 0.03, "gain_db": -5.0, "pan": -0.35})
    dbl = _track(session, "AI Double")
    keep = chorus_ranges()
    dbl.notes = [n for n in dbl.notes if _in(n.onset - 0.03, keep)]
    dbl.curves = {k: copy.deepcopy(v) for k, v in lead.curves.items() if k == "tension"}
    dbl.voice_params["formant_scale"] = 1.19
    dbl.reverb_send = 0.22
    dbl.seed = SEEDS["AI Double"]

    # 2) Harmony: chord-aware close harmony under the lead
    sel = [n for n in lead.notes if _in(n.onset, harmony_ranges())]
    harm = p.add_track("Harmony", voice="pixel", language=lead.language)
    harm.style = "robotic"
    harm.gain_db, harm.pan, harm.reverb_send = -6.0, 0.35, 0.3
    harm.seed = SEEDS["Harmony"]
    for n in sel:
        harm.notes.append(Note(n.onset, n.duration, harmony_pitch(n.pitch, n.onset), n.lyric, n.phonemes,
                               dict(n.attrs)))
    harm.sort()

    # 3) Crowd unison doubles
    call_tool(session, "duplicate_track", {"track": crowd.id, "name": "Crowd 2", "voice": "ember",
                                           "detune_cents": -9.0, "delay_beats": 0.035, "gain_db": -5.0,
                                           "pan": 0.6})
    call_tool(session, "duplicate_track", {"track": crowd.id, "name": "Crowd 3", "voice": "aurora",
                                           "transpose": 12, "detune_cents": 6.0, "delay_beats": 0.018,
                                           "gain_db": -7.0, "pan": 0.25})
    c3 = _track(session, "Crowd 3")
    c3.notes = [n for n in c3.notes if not WHISPER[0] <= n.onset < WHISPER[1]]
    shout = _track(session, "Crowd Shout")
    call_tool(session, "duplicate_track", {"track": shout.id, "name": "Crowd Shout 2", "voice": "ember",
                                           "detune_cents": -9.0, "delay_beats": 0.035, "gain_db": -5.0,
                                           "pan": 0.6})
    call_tool(session, "duplicate_track", {"track": shout.id, "name": "Crowd Shout 3", "voice": "aurora",
                                           "transpose": 12, "detune_cents": 6.0, "delay_beats": 0.018,
                                           "gain_db": -7.0, "pan": 0.25})

    for name in ("Crowd 2", "Crowd 3", "Crowd Shout 2", "Crowd Shout 3"):
        _track(session, name).seed = SEEDS[name]

    # 4) fan voice on the alien chants
    chops = {name: _fan_chop(_track(session, name)) for name in ("Alien Chant", "Alien Chant 2")}

    # tidy mixer order
    order = ["AI", "AI Double", "Harmony", "AI Answer", "AI Adlib", "Aliens", "Alien Adlib", "Alien Chant",
             "Alien Chant 2", "Crowd", "Crowd 2", "Crowd 3", "Crowd Shout", "Crowd Shout 2", "Crowd Shout 3"]
    p.tracks.sort(key=lambda t: order.index(t.name) if t.name in order else len(order))
    return {
        "tracks": [{"name": t.name, "voice": t.voice, "notes": len(t.notes), "gain_db": t.gain_db, "pan": t.pan}
                   for t in p.tracks],
        "ai_double_notes": len(dbl.notes),
        "harmony_notes": len(harm.notes),
        "fan_chop_regions": chops,
    }


def _best(table: dict, name: str):
    """The entry of ``table`` whose key is ``name`` or its longest prefix of whole words."""
    best = ""
    for k in table:
        if (name == k or name.startswith(k + " ")) and len(k) > len(best):
            best = k
    return table.get(best)


def _note_dicts(notes) -> list[dict]:
    return [{"onset": n.onset, "duration": n.duration, "pitch": n.pitch, "lyric": n.lyric,
             **({"phonemes": n.phonemes} if n.phonemes else {}), **({"attrs": n.attrs} if n.attrs else {})}
            for n in notes]


def move_notes(session: Session, source: str, name: str, pick, after_source: bool = False) -> int:
    """Move the notes of track ``source`` that ``pick`` accepts to track ``name``, made as a copy of the source
    (voice, curves, settings and seed) when it does not exist yet. Returns the number of notes moved."""
    p = session.project
    names = {t.name for t in p.tracks}
    if source not in names:
        return 0
    src = _track(session, source)
    moved = [n for n in src.notes if pick(n)]
    if not moved:
        return 0
    if name in names:
        call_tool(session, "add_notes", {"track": name, "notes": _note_dicts(moved)})
    else:
        call_tool(session, "duplicate_track", {"track": source, "name": name})
        dst = _track(session, name)
        keep = {(n.onset, n.pitch) for n in moved}
        drop = [n.id for n in dst.notes if (n.onset, n.pitch) not in keep]
        if drop:
            call_tool(session, "delete_notes", {"track": name, "ids": drop})
        dst.seed = SEEDS.get(name, src.seed)
        if after_source:  # a section channel sits after its source (and the source's earlier channels) in the mixer
            channels = {c[1] for c in SECTION_CHANNELS}
            p.tracks.remove(dst)
            pos = p.tracks.index(src) + 1
            while pos < len(p.tracks) and p.tracks[pos].name in channels:
                pos += 1
            p.tracks.insert(pos, dst)
    call_tool(session, "delete_notes", {"track": source, "ids": [n.id for n in moved]})
    return len(moved)


def second_take_of(name: str) -> tuple[str, dict] | None:
    """``(first take's track name, its SECOND_TAKES entry)`` for the name of a second take, else None."""
    if not name.endswith(" B"):
        return None
    take = _best(SECOND_TAKES, name[:-2])
    return (name[:-2], take) if take else None


def set_gender(track: Track, gender: float) -> None:
    """Set ``neural.gender`` so that the neural engine sings ``track`` with this formant shift: the track's
    ``formant_scale`` (which shapes the formant engine's voice and stays for it) moves the neural voice too,
    and is taken off here."""
    from synth2.neural.presets import PRESETS, VoicePreset, preset_for

    base = PRESETS.get(track.voice, VoicePreset()).gender
    own = {k: v for k, v in track.voice_params.items() if not k.startswith("neural.")}
    track.voice_params["neural.gender"] = round(gender - (preset_for(track.voice, own).gender - base), 4)


def apply_cast(track: Track, cast: dict) -> None:
    for k, v in cast.items():
        if k == "gender":
            set_gender(track, v)
        else:
            track.voice_params[k] = v


def apply_arrangement(session: Session, log=None) -> dict:
    """Section channels, one kind of sound per track, the copies, the second takes, roles and the cast (see the
    module docstring). Call it once, after :func:`apply_vocal_production`."""
    p = session.project
    done: dict[str, int] = {}
    for source, name, a, b in SECTION_CHANNELS:
        k = move_notes(session, source, name, lambda n: a <= n.onset < b, after_source=True)
        done[name] = done.get(name, 0) + k
    for source, a, b, lyrics, name in KIND_CHANNELS:
        k = move_notes(session, source, name,
                       lambda n: a <= n.onset < b and (lyrics is None or (n.lyric or "").lower() in lyrics))
        done[name] = done.get(name, 0) + k
    for t in [t for t in p.tracks if not t.notes]:
        if log:
            log(f"  removed the empty track {t.name}")
        p.tracks.remove(t)
    for source, name, _, cents in COPIES:
        call_tool(session, "duplicate_track", {"track": source, "name": name, "detune_cents": cents})
        _track(session, name).seed = SEEDS[name]
        done[name] = len(_track(session, name).notes)
    for t in p.tracks:
        t.role = _best(ROLES, t.name) or ""
        apply_cast(t, _best(CAST, t.name) or {})
    for _, name, role, _ in COPIES:
        _track(session, name).role = role
    # second takes: after the track they double
    s_per_beat = 60.0 / BPM
    for t in list(p.tracks):
        take = _best(SECOND_TAKES, t.name)
        if not take:
            continue
        name = f"{t.name} B"
        call_tool(session, "duplicate_track", {"track": t.name, "name": name,
                                               "transpose": int(take.get("transpose", 0)),
                                               "delay_beats": round(take["delay"] / s_per_beat, 4)})
        b = _track(session, name)
        b.seed = t.seed + 101
        b.voice_params["neural.detune"] = float(take["detune"])
        if "tone" in take:
            b.voice_params["neural.tone"] = float(take["tone"])
        b.voice_params.pop("neural.gender", None)
        set_gender(b, (_best(CAST, t.name) or {}).get("gender", 0.0) + take["gender"])
        b.gain_db = t.gain_db + take["gain_db"]
        p.tracks.remove(b)
        p.tracks.insert(p.tracks.index(t) + 1, b)
        done[name] = len(b.notes)
    return {"channels": done, "tracks": [{"name": t.name, "role": t.role, "voice": t.voice, "seed": t.seed,
                                          "notes": len(t.notes)} for t in p.tracks]}


def load_vocals(path: Path | str = SCORE) -> Session:
    """Load ``vocals.s2s`` and apply the production. Returns the session."""
    session = Session.open(path)
    apply_vocal_production(session)
    apply_arrangement(session)
    return session


if __name__ == "__main__":
    import json

    s = Session.open(SCORE)
    out = apply_vocal_production(s)
    out.pop("tracks")
    out.update(apply_arrangement(s))
    print(json.dumps(out, indent=2, ensure_ascii=False))
