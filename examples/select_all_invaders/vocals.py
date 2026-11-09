"""Vocal production for SELECT ALL INVADERS: the derived vocal tracks and the alien "fan voice".

``build.py`` loads ``vocals.s2s`` into a :class:`synth2.ai.tools.Session` and then calls
:func:`apply_vocal_production`. The score holds the parts that were written by hand (``AI``, ``Aliens``,
``Aliens 2`` and ``Crowd``). This module derives the rest from them, so a change to the lead melody
carries over to its double and harmony:

1. **AI Double** (``duplicate_track``): the lead in the three choruses, including their pickups.
   It is nova again, +8 cents, 0.03 beats (~11 ms) late, -5 dB, panned left (-0.35), with a slightly
   smaller formant scale (1.19 instead of 1.22) so the two takes do not phase into one voice.
2. **Harmony**: a chord-aware close harmony under the lead, sung by pixel (robotic style, -6 dB, panned
   right +0.35). Every lead note that is a chord tone gets the next chord tone below it (a 3rd, or a
   4th below a root). A passing note gets the diatonic 3rd below it, in A minor (B minor in the final
   chorus), with the raised leading tone over the V chord. Notes on the "and" of beat 4 belong to the
   next bar's chord, because they anticipate it. Choruses 1 and 2 are harmonized only on the hook
   lines (1, 3, 5, 7) and the tag line 8. The deadpan joke lines 2, 4 and 6 stay solo. The final
   chorus is harmonized throughout, except its line 6 ("Funny... neither can you"): the pronoun-arc
   payoff stays as bare as in choruses 1 and 2 (the band also stops for "can").
3. **Crowd 2 / Crowd 3** (``duplicate_track``): the gang shouts in unison.
   - Crowd 2 is ember at the same pitch: -9 cents, +0.035 beats late, panned right (+0.6).
   - Crowd 3 is aurora an octave up: +6 cents, +0.018 beats late, panned +0.25, at -7 dB.
   - The written Crowd track (atlas) sits on the left (-0.6).
   - The breakdown's whispered "hurry" is not doubled an octave up (an octave-up whisper is only hiss).
4. **Fan voice**: the classic ワレワレハ ウチュウジン ダ effect, a voice speaking into a fan. Over every
   kana note of ``Aliens`` and ``Aliens 2``, an 11 Hz loudness chop (0 / -9 dB) is added to the
   loudness curve. It goes on top of any curve already there, such as the outro fade. The pitch warble
   on those notes also deepens from the track's +-0.4 semitones (+-0.7 on Aliens 2) to +-0.6 semitones,
   unless the note sets its own ``vibrato_depth``. The English alien lines keep the lighter warble so
   they stay intelligible.

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
    "Am Am F G Am Am F E "                                    # 1-8 intro
    + "Am Am/G F E " * 4                                      # 9-24 verse 1
    + "Dm E Am G F E F G "                                    # 25-32 pre-chorus 1
    + "F G Em Am " * 3 + "Dm E Am Am "                        # 33-48 chorus 1
    + "Am F G E "                                             # 49-52 post-chorus
    + "Am Am/G F E " * 4                                      # 53-68 verse 2
    + "Dm E Am G F E F G "                                    # 69-76 pre-chorus 2
    + "F G Em Am " * 3 + "Dm E Am Am "                        # 77-92 chorus 2
    + "Bb Am Bb Am Gm A Dm E F F#dim G G#dim Am Bb B E "      # 93-108 bridge
    + "Am F G E "                                             # 109-112 breakdown
    + "G A F#m Bm " * 3 + "Em F# Bm Bm "                      # 113-128 final chorus (B minor)
    + "Bm G A F# Bm N.C."                                     # 129-134 outro
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


def chord_tones(chord: str) -> set[int]:
    """Pitch classes of a chord symbol such as ``Am``, ``F#dim`` or ``Am/G`` (the slash bass is ignored)."""
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
    if beat >= FINAL_CHORUS[0] - 1e-6:   # B minor, A# over F#
        pcs = {11, 1, 2, 4, 6, 7, 9}
        return pcs | {10} if chord.startswith("F#") and "m" not in chord else pcs
    pcs = {9, 11, 0, 2, 4, 5, 7}          # A minor, G# over E
    return pcs | {8} if chord == "E" else pcs


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
    """Add AI Double, Harmony, Crowd 2 and Crowd 3, and the aliens' fan chop. Safe to call twice."""
    p = session.project
    derived = ("AI Double", "Harmony", "Crowd 2", "Crowd 3")
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

    # 2) Harmony: chord-aware close harmony under the lead
    sel = [n for n in lead.notes if _in(n.onset, harmony_ranges())]
    harm = p.add_track("Harmony", voice="pixel", language=lead.language)
    harm.style = "robotic"
    harm.gain_db, harm.pan, harm.reverb_send = -6.0, 0.35, 0.3
    harm.seed = 7 * 7919 + 17
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

    # 4) fan voice on the alien chants
    chops = {name: _fan_chop(_track(session, name)) for name in ("Aliens", "Aliens 2")}

    # tidy mixer order
    order = ["AI", "AI Double", "Harmony", "Aliens", "Aliens 2", "Crowd", "Crowd 2", "Crowd 3"]
    p.tracks.sort(key=lambda t: order.index(t.name) if t.name in order else len(order))
    return {
        "tracks": [{"name": t.name, "voice": t.voice, "notes": len(t.notes), "gain_db": t.gain_db, "pan": t.pan}
                   for t in p.tracks],
        "ai_double_notes": len(dbl.notes),
        "harmony_notes": len(harm.notes),
        "fan_chop_regions": chops,
    }


def load_vocals(path: Path | str = SCORE) -> Session:
    """Load ``vocals.s2s`` and apply the production. Returns the session."""
    session = Session.open(path)
    apply_vocal_production(session)
    return session


if __name__ == "__main__":
    import json

    s = Session.open(SCORE)
    print(json.dumps(apply_vocal_production(s), indent=2, ensure_ascii=False))
