"""The separation as a list of moves on the written score: which notes of which track are ad-libs, answers or
chants, and the track they go to. `rearrange(session)` applies it to a session loaded from the repo's vocals.s2s
(before any derived track exists); the hand-written score sep/vocals.s2s is checked against it (check_score.py).
"""
from __future__ import annotations

import copy

from synth2.project import Note

# (source track, first beat, end beat (exclusive), lyrics or None = every note, destination track)
MOVES = [
    # the lead: calls in the crowd's post-chorus, sound words, the snooze "hai hai", the stutter
    ("AI", 31.0, 32.0, None, "AI Answer"),              # bar 8: "blocked", glued to verse 1's first word
    ("AI", 194.0, 196.0, None, "AI Answer"),            # bar 49: "accept all"
    ("AI", 204.0, 206.0, ("click",), "AI Adlib"),       # bar 52: click x4
    ("AI", 206.0, 208.0, None, "AI Answer"),            # bar 52: "try again"
    ("AI", 280.0, 282.0, None, "AI Adlib"),             # bar 71: ハイ ハイ
    ("AI", 368.0, 369.5, ("[r iy]",), "AI Adlib"),      # bar 93: re- re- re-
    # the aliens: interjections, sound words, the laugh, the stutter, "nande?"
    ("Aliens", 198.0, 199.0, ("ouch",), "Alien Adlib"),
    ("Aliens", 204.0, 206.0, ("click",), "Alien Adlib"),
    ("Aliens", 262.0, 263.5, ("ha",), "Alien Adlib"),
    ("Aliens", 371.0, 372.5, ("[r iy]",), "Alien Adlib"),
    ("Aliens", 395.0, 396.0, None, "Alien Adlib"),      # ナンデ
    ("Aliens", 399.0, 400.0, None, "Alien Adlib"),      # ナンデ
    ("Aliens", 470.0, 471.0, ("click",), "Alien Adlib"),
    # the aliens: the chants
    ("Aliens", 16.0, 32.0, None, "Alien Chant"),
    ("Aliens", 288.0, 296.0, None, "Alien Chant"),
    ("Aliens", 501.0, 503.0, None, "Alien Chant"),
    # the crowd: shouts and sound words
    ("Crowd", 0.0, 1e9, ("click", "hey", "help", "what", "huh"), "Crowd Shout"),
    ("Crowd", 420.0, 424.0, None, "Crowd Shout"),       # "here! here! over here!"
]
RENAME = {"Aliens 2": "Alien Chant 2"}
SEEDS = {"AI Answer": 31, "AI Adlib": 43, "Alien Adlib": 7950, "Alien Chant": 7964, "Crowd Shout": 23788}
# kana words written one kana per note become one word over its notes (one take, not one per kana)
WORDS = {("AI Adlib", "ハ", "イ"): "ハイ", ("Alien Adlib", "ナ", "ン", "デ"): "ナンデ"}


def _take(session, src, dst, pick):
    p = session.project
    s = p.track(src)
    moved = [n for n in s.notes if pick(n)]
    if not moved:
        return 0
    if not any(t.name == dst for t in p.tracks):
        nt = p.add_track(dst, voice=s.voice, language=s.language)
        for k in ("style", "gain_db", "pan", "reverb_send", "seed"):
            setattr(nt, k, getattr(s, k))
        nt.voice_params = dict(s.voice_params)
        nt.curves = copy.deepcopy(s.curves)
        nt.seed = SEEDS[dst]
    d = p.track(dst)
    for n in moved:
        d.notes.append(Note(n.onset, n.duration, n.pitch, n.lyric, n.phonemes, dict(n.attrs)))
    ids = {id(n) for n in moved}
    s.notes = [n for n in s.notes if id(n) not in ids]
    d.sort()
    return len(moved)


def rearrange(session) -> dict:
    """Apply MOVES, RENAME and WORDS. Returns {destination: notes moved}."""
    p = session.project
    done = {}
    for src, a, b, lyrics, dst in MOVES:
        k = _take(session, src, dst, lambda n: a <= n.onset < b and (lyrics is None or (n.lyric or "").lower() in lyrics))
        done[dst] = done.get(dst, 0) + k
    for old, new in RENAME.items():
        p.track(old).name = new
    for (tr, *kana), word in WORDS.items():
        t = p.track(tr)
        i = 0
        while i + len(kana) <= len(t.notes):
            grp = t.notes[i:i + len(kana)]
            if [n.lyric for n in grp] == kana and all(abs(a.end - b.onset) < 1e-6 for a, b in zip(grp, grp[1:])):
                grp[0].lyric = word
                for n in grp[1:]:
                    n.lyric = "+"
                i += len(kana)
            else:
                i += 1
    return done
