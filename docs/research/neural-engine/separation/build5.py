"""The song build with the separated arrangement: prod/build4.py with sep/vocals.s2s, sep/vocals.py and the
channels of the new tracks.

`install()` changes build4 in place (score folder, production function, splits, faders, channel overrides); after
it `build4.build_session()` and `build4.main()` build the new arrangement. `build_session(arr)` gives either
arrangement: "before" = build4 as it is (the third demo), "after" = installed.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
HERE = S / "sep"
REPO_SONG = Path("/home/user/synth2/examples/select_all_invaders")
os.environ.setdefault("DEMO3_KEY", "-5")
os.environ.setdefault("PROD_ENGINE", "flow")
sys.path.insert(0, "/home/user/synth2")
sys.path.insert(0, str(REPO_SONG))
sys.path.insert(0, str(S / "prod"))

import build4  # noqa: E402

# Split channels of the ad-lib tracks (as build4.SPLITS): the stutter before "reboot" is the lead's own word, so
# it sits in the centre at the bridge lead's level; the aliens' laugh (verse 2), stutter and "ナンデ?" (bridge)
# follow the level of the aliens' lines in those sections.
SPLITS5 = [
    {"source": "AI Adlib", "name": "AI Adlib Bridge", "start": 368.0, "end": 432.0, "skip_lyrics": ()},
    {"source": "Alien Adlib", "name": "Alien Adlib Verse", "start": 208.0, "end": 272.0, "skip_lyrics": ()},
    {"source": "Alien Adlib", "name": "Alien Adlib Bridge", "start": 368.0, "end": 416.0, "skip_lyrics": ()},
]
# Faders, pans and reverb sends of the new tracks (the others: build4.TRACK_MIX). The lead stays the only voice
# in the centre; its calls sit a little to the right and lower, its ad-libs to the left, lower and drier; the
# aliens' ad-libs sit outside the aliens' lines, the two chant voices left and right of them, the shouts widest.
TRACK_MIX5 = {
    "AI Answer":          {"gain_db": -2.0, "pan": 0.15, "reverb_send": 0.50},
    "AI Adlib":           {"gain_db": -3.0, "pan": -0.20, "reverb_send": 0.20},
    "AI Adlib Bridge":    {"gain_db": 2.5, "pan": 0.0, "reverb_send": 0.20},
    "Alien Adlib":        {"gain_db": 1.5, "pan": 0.50, "reverb_send": 0.20},
    "Alien Adlib Verse":  {"gain_db": -1.0, "pan": 0.50, "reverb_send": 0.20},
    "Alien Adlib Bridge": {"gain_db": 3.0, "pan": 0.50, "reverb_send": 0.20},
    "Alien Chant":        {"gain_db": 1.5, "pan": 0.30, "reverb_send": 0.30},
    "Alien Chant 2":      {"gain_db": 0.5, "pan": -0.35, "reverb_send": 0.40},
    "Crowd Shout":        {"gain_db": -2.0, "pan": -0.70, "reverb_send": 0.25},
    "Crowd Shout 2":      {"gain_db": -2.0, "pan": 0.70, "reverb_send": 0.25},
    "Crowd Shout 3":      {"gain_db": -4.0, "pan": 0.30, "reverb_send": 0.25},
}
# mix_overrides of the new tracks: what their source tracks have
OVERRIDES5 = {
    "Alien Adlib": {"delay": 0.3, "crest_db": 8.0, "eq_add": build4.ALIEN_EQ},
    "Alien Chant": {"delay": 0.6, "crest_db": 8.0, "eq_add": build4.ALIEN_EQ},
    "Alien Chant 2": {"delay": 0.8, "eq_add": [["peak", 280, -3.0, 1.0]]},
    "Crowd Shout": {"delay": 0.0, "crest_db": 12.0}, "Crowd Shout 2": {"delay": 0.0, "crest_db": 12.0},
    "Crowd Shout 3": {"delay": 0.0, "crest_db": 12.0},
}
# channels the backend processes itself (ds_backend5: AI Answer through the lead's chain, AI Adlib through its EQ)
CHAIN_CHANNELS5 = {"AI Answer": {"chain": False}, "AI Adlib": {"chain": False, "delay": 0.3},
                   "AI Adlib Bridge": {"chain": False, "delay": 0.3}}
# Seeds of the split channels: a new channel's default seed comes from its position in the project, which the
# five new tracks would shift. These are the seeds of the third demo's build (seeds_before.json), so the takes of
# every channel that existed stay what they were.
SPLIT_SEEDS = {"AI Verses": 63369, "AI Pre": 71288, "AI Bridge": 79207, "AI Breakdown": 87126, "AI Final": 95045,
               "Aliens Verse": 102964, "Aliens Bridge": 110883, "Aliens Hook": 118802, "Crowd Hook": 126721,
               "Crowd 2 Hook": 134640, "Crowd 3 Hook": 142559, "Harmony Final": 150478,
               "AI Adlib Bridge": 79221, "Alien Adlib Verse": 102978, "Alien Adlib Bridge": 110897}
_installed = False


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install():
    """Point build4 at the separated arrangement."""
    global _installed
    if _installed:
        return
    _installed = True
    song = HERE / "song"
    song.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "vocals.s2s", song / "vocals.s2s")
    shutil.copyfile(REPO_SONG / "backing.json", song / "backing.json")
    build4.HERE = song
    build4.vocals = _load(HERE / "vocals.py", "vocals_sep")
    for sp in build4.SPLITS:
        sp["skip_lyrics"] = ()
        if sp["name"] == "Aliens Hook":
            sp["end"] = 501.0  # the last syllable of "invaders" (beat 500.5) belongs to the hook
    build4.SPLITS.extend(json.loads(json.dumps(SPLITS5)))
    build4.TRACK_MIX.pop("Aliens 2", None)
    build4.TRACK_MIX.update(TRACK_MIX5)
    mo = build4.MASTER["mix_overrides"]
    mo.pop("Aliens 2", None)
    mo.update(json.loads(json.dumps(OVERRIDES5)))
    build4.CHAIN_CHANNELS.update(CHAIN_CHANNELS5)
    split = build4.split_track

    def split_seeded(s, source, name, start, end, skip_lyrics=()):
        moved = split(s, source, name, start, end, skip_lyrics)
        if moved and name in SPLIT_SEEDS:
            s.project.track(name).seed = SPLIT_SEEDS[name]
        return moved

    build4.split_track = split_seeded


def build_session(arr="after", log=print):
    if arr != "before":
        install()
    return build4.build_session(log=log)
