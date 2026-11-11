"""The song build of the fourth demo: prod/build4.py with the separated score (vocals.s2s, vocals.py), one kind of
sound per vocal track (MOVES6), a role for every track (ROLES) and the mix of main against backing (ROLE_MIX,
TRACK_MIX6).

`install()` changes build4 in place (score folder, production function, splits, faders, channel overrides);
`build_session(arr)` gives either arrangement: "before" = build4 as it is (the third demo), "after" = this one.
The role of a track decides its place: fader offset, pan, sends and tone here, the in-channel effects (band limit,
slap, far room, the duck under the lead) in ds_backend5.ROLE_FX.
"""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
HERE = S / "mix5"
REPO_SONG = Path("/home/user/synth2/examples/select_all_invaders")
os.environ.setdefault("DEMO3_KEY", "-5")
os.environ.setdefault("PROD_ENGINE", "flow")
sys.path.insert(0, "/home/user/synth2")
sys.path.insert(0, str(REPO_SONG))
sys.path.insert(0, str(S / "prod"))

import build4  # noqa: E402
from synth2.ai.tools import call_tool  # noqa: E402

# ---------------------------------------------------------------------------------------- the separation (G1)
# Split channels of the ad-lib tracks (as build4.SPLITS): section channels, so the ad-libs follow the level of
# their voice's lines in that section.
SPLITS5 = [
    {"source": "AI Adlib", "name": "AI Adlib Bridge", "start": 368.0, "end": 432.0, "skip_lyrics": ()},
    {"source": "Alien Adlib", "name": "Alien Adlib Verse", "start": 208.0, "end": 272.0, "skip_lyrics": ()},
    {"source": "Alien Adlib", "name": "Alien Adlib Bridge", "start": 368.0, "end": 416.0, "skip_lyrics": ()},
]
# Seeds of the split channels of the third demo's build: a channel's default seed comes from its position in the
# project, which new tracks would shift.
SPLIT_SEEDS = {"AI Verses": 63369, "AI Pre": 71288, "AI Bridge": 79207, "AI Breakdown": 87126, "AI Final": 95045,
               "Aliens Verse": 102964, "Aliens Bridge": 110883, "Aliens Hook": 118802, "Crowd Hook": 126721,
               "Crowd 2 Hook": 134640, "Crowd 3 Hook": 142559, "Harmony Final": 150478,
               "AI Adlib Bridge": 79221, "Alien Adlib Verse": 102978, "Alien Adlib Bridge": 110897}

# ---------------------------------------------------------------------------- one kind of sound per track
# (source channel, first beat, end beat (exclusive), lyrics or None = every note, destination). Applied to the
# finished build (after the section splits), so a destination inherits the seed, the voice and the curves of the
# channel its notes were on: the takes stay the ones measured in sep/. Bar B beat b = (B - 1) * 4 + (b - 1).
MOVES6 = [
    # the lead's ad-libs: three different sounds were on "AI Adlib"
    ("AI Adlib", 204.0, 206.0, None, "AI Click"),                 # bar 52: click x4 (sound word)
    ("AI Adlib", 280.0, 282.0, None, "AI Hai"),                   # bar 71: ハイ ハイ (interjection)
    ("AI Adlib Bridge", 368.0, 370.0, None, "AI Stutter"),        # bar 93: re- re- re- (stutter into "reboot")
    # the aliens' ad-libs: five different sounds were on "Alien Adlib" and its section channels
    ("Alien Adlib", 198.0, 199.0, None, "Alien Ouch"),            # bar 50
    ("Alien Adlib", 204.0, 206.0, None, "Alien Click"),           # bar 52: click x4
    ("Alien Adlib", 470.0, 471.0, None, "Alien Click"),           # bar 118: "click?"
    ("Alien Adlib Verse", 262.0, 264.0, None, "Alien Laugh"),     # bar 66: ha ha ha
    ("Alien Adlib Bridge", 371.0, 373.0, None, "Alien Stutter"),  # bar 93: re- re- re- (into "reply all")
    ("Alien Adlib Bridge", 395.0, 400.0, None, "Alien Nande"),    # bars 99, 100: ナンデ
    # the aliens' track carried their lines, their short answers to the lead and the unison with the lead
    ("Aliens", 78.0, 80.0, None, "Alien Answer"),                 # bar 20: "cookies? delicious"
    ("Aliens", 118.0, 120.0, None, "Alien Answer"),               # bar 30: "the mothership"
    ("Aliens", 199.0, 204.0, None, "Alien Answer"),               # bars 50-51: "thank you", "did we pass"
    ("Aliens", 286.0, 288.0, None, "Alien Answer"),               # bar 72: "four minutes"
    ("Aliens", 341.5, 344.0, None, "Alien Answer"),               # bar 86: "we are not a robot"
    ("Aliens Bridge", 372.5, 376.0, None, "Alien Answer Bridge"),  # bar 94: "reply all"
    ("Aliens Bridge", 403.0, 408.0, None, "Alien Answer Bridge"),  # bars 101-102: "no, you are", "respectfully"
    ("Aliens Bridge", 408.0, 416.0, None, "Alien Unison"),        # bars 103-104: "who is who ..." with the lead
    # the crowd's track carried its own line, echoes / answers and (on the hook channels) the whisper
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
# copies made in the backend from another channel's take (ds_backend5.COPIES): the second double and the
# harmony's other side
TWINS = [("AI Double", "AI Double R"), ("Harmony", "Harmony L"), ("Harmony Final", "Harmony Final L")]

# ------------------------------------------------------------------------------------------------ the roles
# track name (longest prefix wins) -> role
ROLES = {
    "AI": "lead", "AI Double": "double", "Harmony": "harmony",
    "AI Answer": "answer", "AI Click": "click", "AI Hai": "adlib", "AI Stutter": "stutter",
    "Aliens": "alien lead", "Aliens Hook": "alien backing", "Alien Unison": "alien backing",
    "Alien Answer": "alien answer",
    "Alien Click": "alien click", "Alien Laugh": "alien laugh", "Alien Ouch": "alien adlib",
    "Alien Nande": "alien adlib", "Alien Stutter": "alien stutter",
    "Alien Chant": "alien chant",
    "Crowd": "crowd", "Crowd 2": "crowd", "Crowd 3": "crowd",
    "Crowd Shout": "crowd shout", "Crowd Click": "crowd shout",
}


def role_of(name: str) -> str:
    best = ""
    for k in ROLES:
        if (name == k or name.startswith(k + " ")) and len(k) > len(best):
            best = k
    return ROLES.get(best, "other")


# Faders (dB), pans and reverb sends of every channel that is new or moved against the third demo (the others:
# build4.TRACK_MIX / CHAIN_SENDS). Levels are faders, not measured levels: NOTES.md has the measured level of
# every role against the lead.
TRACK_MIX6 = {
    # the double: one copy each side, close to the lead (the third demo: one double at -6.5 dB, 0.35 left)
    "AI Double":          {"gain_db": -12.5, "pan": -0.28, "reverb_send": 0.40},
    "AI Double R":        {"gain_db": -12.5, "pan": 0.28, "reverb_send": 0.40},
    # the harmony: its take right, its copy left, outside the doubles, more room
    "Harmony":            {"gain_db": -10.0, "pan": 0.70, "reverb_send": 0.65},
    "Harmony L":          {"gain_db": -11.5, "pan": -0.70, "reverb_send": 0.65},
    "Harmony Final":      {"gain_db": -11.5, "pan": 0.70, "reverb_send": 0.65},
    "Harmony Final L":    {"gain_db": -13.0, "pan": -0.70, "reverb_send": 0.65},
    # the lead's calls in other voices' sections: its own voice, a step to the right and lower
    "AI Answer":          {"gain_db": -2.0, "pan": 0.18, "reverb_send": 0.55},
    # the lead's thrown-in sounds: off-centre, band-limited, a slap, little room
    "AI Click":           {"gain_db": 0.0, "pan": -0.40, "reverb_send": 0.15},
    "AI Hai":             {"gain_db": -3.0, "pan": -0.40, "reverb_send": 0.15},
    # the stutter is the start of the lead's own word "reboot": the lead's place and level in the bridge
    "AI Stutter":         {"gain_db": 2.5, "pan": 0.0, "reverb_send": 0.50},
    # the aliens' lines keep their place right of the centre; their answers and thrown-in sounds sit outside them
    "Alien Answer":        {"gain_db": 1.0, "pan": 0.50, "reverb_send": 0.30},
    "Alien Answer Bridge": {"gain_db": 2.5, "pan": 0.50, "reverb_send": 0.30},
    "Alien Unison":        {"gain_db": -1.5, "pan": -0.45, "reverb_send": 0.40},
    "Aliens Hook":         {"gain_db": -5.0, "pan": 0.25, "reverb_send": 0.40},
    "Alien Click":         {"gain_db": 0.5, "pan": 0.70, "reverb_send": 0.15},
    "Alien Ouch":          {"gain_db": 0.5, "pan": 0.70, "reverb_send": 0.15},
    "Alien Laugh":         {"gain_db": -1.0, "pan": 0.70, "reverb_send": 0.15},
    "Alien Nande":         {"gain_db": 2.0, "pan": 0.70, "reverb_send": 0.15},
    "Alien Stutter":       {"gain_db": 2.5, "pan": 0.50, "reverb_send": 0.30},
    # the two chant voices left and right of everything the aliens sing
    "Alien Chant":         {"gain_db": 1.0, "pan": 0.60, "reverb_send": 0.45},
    "Alien Chant 2":       {"gain_db": 0.0, "pan": -0.60, "reverb_send": 0.50},
}
# the three crowd voices: (suffix pattern, pan); the third no longer near the centre
CROWD_PAN = {"": -0.85, "2": 0.85, "3": -0.30}
# crowd kinds: fader of voices 1 and 2 (voice 3 is 2 dB lower), reverb send
# The whisper's fader: on a channel of its own it is matched to the formant engine's whisper, which is quiet; at
# the hook's fader (-7) "hurry" came out 19 to 22 dB under the breakdown's lead. +3 puts it about 10 dB under.
CROWD_KINDS = {"": (-3.0, 0.55), "Answer": (-3.0, 0.55), "Whisper": (3.0, 0.55), "Hook": (-8.5, 0.55),
               "Shout": (-4.0, 0.40), "Click": (-3.5, 0.40)}
# mix_overrides of the new channels (delay = send into the song's ducked ping-pong delay, 1 = the lead's)
ALIEN_EQ = build4.ALIEN_EQ
OVERRIDES6 = {
    "AI Double R": {"chain": False, "delay": 0.4},
    "Harmony L": {"chain": False, "delay": 0.5},
    "Harmony Final L": {"chain": False, "delay": 0.5},
    "AI Answer": {"chain": False, "delay": 0.6},
    "AI Click": {"chain": False, "delay": 0.0},
    "AI Hai": {"chain": False, "delay": 1.3},          # thrown into the gap before "remind me tomorrow"
    "AI Stutter": {"chain": False, "delay": 0.3},
    "Alien Answer": {"delay": 0.6, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Answer Bridge": {"delay": 0.6, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Unison": {"delay": 0.3, "crest_db": 8.0, "eq_add": [["peak", 300, -2.5, 1.0], ["peak", 2800, -3.0, 1.0]]},
    "Alien Click": {"delay": 0.0, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Ouch": {"delay": 0.8, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Laugh": {"delay": 1.0, "crest_db": 8.0, "eq_add": ALIEN_EQ},   # thrown into the gap after it
    "Alien Nande": {"delay": 1.3, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Stutter": {"delay": 0.3, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Chant": {"delay": 0.6, "crest_db": 8.0, "eq_add": ALIEN_EQ},
    "Alien Chant 2": {"delay": 0.8, "eq_add": [["peak", 280, -3.0, 1.0]]},
}
CROWD_OVERRIDE = {"delay": 0.0, "crest_db": 12.0}

# ------------------------------------------------------------------------------- the band: one kind per stem
# Four band stems carried two kinds of sound each. The notes and the instrument settings stay what they are; each
# kind gets its own stem (MIX5_BAND=off: the band as it was). Compared on bars 33-48: same loudness, band gain and
# limiter figures; the moved part is played with another humanisation (the band sum differs by -25 dB).
BAND_SPLIT = os.environ.get("MIX5_BAND", "split") == "split"
# (instrument, patterns that move, new instrument): the chip lead's chorus hook tags apart from its riffs
# (intro, post-chorus, outro), the same for the low chip lead, the bridge's glitch percussion apart from the
# intro's tick
BAND_SPLITS = [("lead", ("hook_tags",), "tags"), ("leadlo", ("hook_tags",), "tagslo"), ("chip", ("chip_glitch",), "glitch")]
# one-shot effects that are gags of the story go to "sfx"; the transitions (riser, reverse_cymbal, impact,
# downlifter, whoosh, sweep) stay on "fx"
SFX_GAGS = ("radio", "coin", "bleeps", "ufo", "zap", "error", "laser", "alarm", "glitch", "modem", "powerdown")
BAND_OVERRIDE_COPY = {"tags": "lead", "tagslo": "leadlo", "sfx": "fx"}


def split_band(arr: dict) -> dict:
    """The arrangement with BAND_SPLITS and SFX_GAGS on stems of their own."""
    arr = json.loads(json.dumps(arr))
    inst = arr["instruments"]
    inst["sfx"] = json.loads(json.dumps(inst["fx"]))
    for sec in arr["sections"]:
        for e in sec.get("events", []):
            if e.get("type", "fx" if "sound" in e else None) == "fx" and e.get("sound") in SFX_GAGS and "part" not in e:
                e["part"] = "sfx"
        parts = sec.get("parts", {})
        for src, pats, dst in BAND_SPLITS:
            spec = parts.get(src)
            if spec is None:
                continue
            segs = spec if isinstance(spec, list) else [spec]
            names = [g if isinstance(g, str) else g.get("pattern") for g in segs]
            if all(n in pats for n in names):
                parts[dst] = parts.pop(src)
                inst.setdefault(dst, json.loads(json.dumps(inst[src])))
            elif any(n in pats for n in names):
                raise ValueError(f"{sec.get('name')}: {src} plays both kinds in one section")
    return arr
_installed = False


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def install():
    """Point build4 at the separated score (as sep/build5.install)."""
    global _installed
    if _installed:
        return
    _installed = True
    song = HERE / "song"
    song.mkdir(exist_ok=True)
    shutil.copyfile(HERE / "vocals.s2s", song / "vocals.s2s")
    arr = json.loads((REPO_SONG / "backing.json").read_text(encoding="utf-8"))
    if BAND_SPLIT:
        arr = split_band(arr)
        tw = build4.BAND_TWEAKS["Bridge"]
        tw["glitch"] = tw.pop("chip")
        mo = build4.MASTER["mix_overrides"]
        for new, old in BAND_OVERRIDE_COPY.items():
            mo[new] = json.loads(json.dumps(mo[old]))
    (song / "backing.json").write_text(json.dumps(arr, ensure_ascii=False), encoding="utf-8")
    build4.HERE = song
    build4.vocals = _load(HERE / "vocals.py", "vocals_sep")
    for sp in build4.SPLITS:
        sp["skip_lyrics"] = ()
        if sp["name"] == "Aliens Hook":
            sp["end"] = 501.0  # the last syllable of "invaders" (beat 500.5) belongs to the hook
    build4.SPLITS.extend(json.loads(json.dumps(SPLITS5)))
    build4.TRACK_MIX.pop("Aliens 2", None)
    build4.MASTER["mix_overrides"].pop("Aliens 2", None)
    split = build4.split_track

    def split_seeded(s, source, name, start, end, skip_lyrics=()):
        moved = split(s, source, name, start, end, skip_lyrics)
        if moved and name in SPLIT_SEEDS:
            s.project.track(name).seed = SPLIT_SEEDS[name]
        return moved

    build4.split_track = split_seeded


def _names(p):
    return {t.name for t in p.tracks}


def move(s, source, name, pick) -> int:
    """Move the notes of channel `source` that `pick` accepts to channel `name` (made as a copy of the source:
    same voice, seed, style and curves)."""
    p = s.project
    if source not in _names(p):
        return 0
    src = p.track(source)
    moved = [n for n in src.notes if pick(n)]
    if not moved:
        return 0
    if name in _names(p):
        call_tool(s, "add_notes", {"track": name, "notes": [
            {"onset": n.onset, "duration": n.duration, "pitch": n.pitch, "lyric": n.lyric,
             **({"phonemes": n.phonemes} if n.phonemes else {}), **({"attrs": n.attrs} if n.attrs else {})}
            for n in moved]})
    else:
        call_tool(s, "duplicate_track", {"track": source, "name": name})
        dst = p.track(name)
        keep = {(n.onset, n.pitch) for n in moved}
        drop = [n.id for n in dst.notes if (n.onset, n.pitch) not in keep]
        if drop:
            call_tool(s, "delete_notes", {"track": name, "ids": drop})
        dst.seed = src.seed
    call_tool(s, "delete_notes", {"track": source, "ids": [n.id for n in moved]})
    return len(moved)


def crowd_mix(name: str):
    """(fader, pan, send) of a crowd channel from its voice (1, 2, 3) and kind, or None."""
    w = name.split()
    if w[0] != "Crowd":
        return None
    voice = next((x for x in w[1:] if x in ("2", "3")), "")
    kind = next((x for x in w[1:] if x not in ("2", "3")), "")
    if kind not in CROWD_KINDS:
        return None
    g, send = CROWD_KINDS[kind]
    return {"gain_db": g - (2.0 if voice == "3" else 0.0), "pan": CROWD_PAN[voice], "reverb_send": send}


def separate(s, log=print) -> dict:
    """MOVES6, the twins, the faders and channel settings of the roles. Returns {channel: notes moved in}."""
    p = s.project
    done: dict[str, int] = {}
    for src, a, b, lyrics, dst in MOVES6:
        k = move(s, src, dst, lambda n: a <= n.onset < b and (lyrics is None or (n.lyric or "").lower() in lyrics))
        done[dst] = done.get(dst, 0) + k
    for t in [t for t in p.tracks if not t.notes]:  # channels left without a note
        log(f"  removed the empty channel {t.name}")
        p.tracks.remove(t)
    for src, name in TWINS:
        if src in _names(p):
            call_tool(s, "duplicate_track", {"track": src, "name": name})
            p.track(name).seed = p.track(src).seed
    mo = p.master.mix_overrides
    for t in p.tracks:
        mix = TRACK_MIX6.get(t.name) or crowd_mix(t.name)
        if mix:
            call_tool(s, "update_track", {"track": t.name, **mix})
        if t.name in OVERRIDES6:
            mo[t.name] = json.loads(json.dumps(OVERRIDES6[t.name]))
        elif t.name.split()[0] == "Crowd":
            mo[t.name] = dict(CROWD_OVERRIDE)
    for k in [k for k in mo if k.startswith(("AI Adlib", "Alien Adlib"))]:
        mo.pop(k)
    log("  one kind per channel: " + ", ".join(f"{k} {v}" for k, v in done.items() if v))
    return done


def build_session(arr="after", log=print):
    if arr == "before":
        return build4.build_session(log=log)
    install()
    s = build4.build_session(log=log)
    separate(s, log=log)
    return s
