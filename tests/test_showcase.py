"""The showcase song (examples/select_all_invaders): its score, arrangement, mix and band build without models
or sound sets, and a few bars render with the formant engine."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SONG = Path(__file__).resolve().parents[1] / "examples" / "select_all_invaders"


def _load(name):
    sys.path.insert(0, str(SONG))
    try:
        spec = importlib.util.spec_from_file_location(f"showcase_{name}", SONG / f"{name}.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(str(SONG))


@pytest.fixture(scope="module")
def song(tmp_path_factory):
    import os

    old = {k: os.environ.get(k) for k in ("SYNTH2_SOUNDS", "SYNTH2_MODELS")}
    empty = tmp_path_factory.mktemp("nothing-installed")
    os.environ["SYNTH2_SOUNDS"] = os.environ["SYNTH2_MODELS"] = str(empty)
    try:
        build = _load("build")
        session = build.build_session(log=lambda *a: None)
        yield build, session.project
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def test_the_score_is_in_e_minor_and_leaves_the_engine_to_the_machine():
    from synth2.score import parse_score

    p = parse_score((SONG / "vocals.s2s").read_text(encoding="utf-8"))
    assert p.key == "E minor" and p.engine == "" and p.bpm == 168
    assert [t.name for t in p.tracks] == ["AI", "AI Answer", "AI Adlib", "Aliens", "Alien Adlib", "Alien Chant",
                                          "Alien Chant 2", "Crowd", "Crowd Shout"]
    assert [t.role for t in p.tracks[:3]] == ["lead", "answer", "adlib"]
    lead = p.tracks[0]
    assert min(n.pitch for n in lead.notes) >= 59 and max(n.pitch for n in lead.notes) == 78  # B3 .. F#5
    # the first sung line of the lead: "booting up" on E4 E4 C5, five semitones under the A minor original
    assert [(n.lyric, n.pitch) for n in lead.notes[:3]] == [("booting", 64), ("+", 64), ("up", 72)]


def test_one_kind_of_sound_per_track_with_roles_cast_and_pinned_seeds(song):
    build, p = song
    vocals = sys.modules["vocals"]
    names = [t.name for t in p.tracks]
    assert len(names) == 65 and len(set(names)) == 65 and sum(len(t.notes) for t in p.tracks) == 1544 + 272
    assert all(t.notes for t in p.tracks) and not any(n.startswith(("AI Adlib", "Alien Adlib")) for n in names)
    by = {t.name: t for t in p.tracks}
    # roles: where the neural engine has a chain for the job
    assert {n for n, t in by.items() if t.role == "lead"} == set(build.LEAD_TRACKS) == {
        "AI", "AI Verses", "AI Pre", "AI Bridge", "AI Breakdown", "AI Final"}
    assert by["AI Double"].role == by["AI Double R"].role == "double"
    assert by["Harmony"].role == by["Harmony Final"].role == "harmony" and by["AI Answer"].role == "answer"
    assert by["Harmony L"].role == "double:Harmony" and by["Harmony Final L"].role == "double:Harmony Final"
    assert {by[n].role for n in ("AI Click", "AI Hai", "AI Stutter")} == {"adlib"}
    assert all(not t.role for n, t in by.items() if n.startswith(("Alien", "Crowd")))
    # a double has the notes of what it doubles (the lead's take is copied over them), each copy its own detune
    assert [(n.onset, n.pitch) for n in by["AI Double R"].notes] == [(n.onset, n.pitch) for n in by["AI Double"].notes]
    assert by["AI Double"].notes[0].attrs["detune"] != by["AI Double R"].notes[0].attrs["detune"]
    # the cast: ad-lib mode for the hits, the plain setting for the aliens' and the crowd's lines
    from synth2.neural.presets import track_preset

    for n, t in by.items():
        pre = track_preset(t)
        hits = n.split()[1] in ("Click", "Hai", "Stutter", "Ouch", "Laugh", "Nande", "Chant", "Shout") if " " in n else False
        assert (pre.adlib is not None) == hits, n
        if n.startswith(("Aliens", "Alien Answer", "Alien Unison")):
            assert pre.gender == pytest.approx(-0.6, abs=1e-4), n  # whatever formant_scale 1.12 would add
            assert (pre.tone, pre.breaths, pre.cons_cap, pre.detune) == (12.0, 0.0, (), 0.0), n
    assert track_preset(by["AI"]).dial == "crisp" and track_preset(by["AI"]).breaths == 1.0
    assert dict(track_preset(by["Alien Click"]).adlib)["salt.click"] == 2.0
    assert track_preset(by["Alien Chant 2"]).tone == 0.0 and track_preset(by["Alien Chant 2 B"]).tone == 12.0
    # second takes: the same notes a moment later, another seed, 1.5 dB under the first take
    seconds = [n for n in names if vocals.second_take_of(n)]
    assert len(seconds) == 18 and all(n[:-2] in by for n in seconds)
    a, b = by["Crowd Click"], by["Crowd Click B"]
    assert b.seed == a.seed + 101 and len(a.notes) == len(b.notes) and b.gain_db == pytest.approx(a.gain_db - 1.5)
    assert 0.05 < b.notes[0].onset - a.notes[0].onset < 0.07 and names.index("Crowd Click B") == names.index("Crowd Click") + 1
    # seeds do not depend on the order of the tracks
    assert {n: by[n].seed for n in ("AI", "AI Verses", "AI Final", "Harmony", "Aliens Hook", "AI Stutter")} == {
        "AI": 17, "AI Verses": 63369, "AI Final": 95045, "Harmony": 55450, "Aliens Hook": 118802, "AI Stutter": 79221}
    # what moved where: the click run of bar 52, the whisper of bar 112, the lead's verse
    assert [n.lyric for n in by["AI Click"].notes] == ["click"] * 4 and by["AI Click"].notes[0].onset == 204.0
    assert [n.lyric for n in by["Crowd Whisper"].notes] == ["hurry"] and "Crowd 3 Whisper" not in by
    assert all(32.0 <= n.onset < 96.0 or 208.0 <= n.onset < 272.0 for n in by["AI Verses"].notes)


def test_every_track_and_stem_has_its_place_in_the_mix(song):
    build, p = song
    mo = p.master.mix_overrides
    for t in p.tracks:
        mix = build.vocal_mix(t.name)
        assert mix is not None and (t.gain_db, t.pan, t.reverb_send) == (mix["gain_db"], mix["pan"], mix["reverb_send"])
        assert mo[t.name]["delay"] == mix["delay"] and ("fx" in mo[t.name]) == bool(mix["fx"])
        assert "chain" not in mo[t.name] and "eq_add" not in mo[t.name] or not t.role  # a role's chain shapes the tone
    centre = [t.name for t in p.tracks if abs(t.pan) < 0.1]
    assert set(centre) == set(build.LEAD_TRACKS) | {"AI Stutter"}  # nothing else sits where the lead does
    assert p.track("AI Double").pan == -p.track("AI Double R").pan and p.track("Harmony").pan == -p.track("Harmony L").pan
    assert mo["AI Click"]["fx"] == {"slap": None} and mo["Crowd Hook B"]["fx"]["duck_db"] == 3.0
    assert p.master.loudness_target == -11.5 and p.master.ceiling_db == -1.0 and p.master.reverb_room == 0.1
    # nothing is installed here: every stem gets the settings of the synthesised sound
    assert mo["drums snare"]["eq_add"][-1][0] == "highshelf" and mo["gtrL"] == build.BAND_MIX["gtrL"]
    sampled = build.master_settings({"drums", "gtrL"}, p.tracks)["mix_overrides"]
    assert sampled["drums snare"]["gain_db"] == build.SAMPLED_MIX["drums snare"]["gain_db"]
    assert sampled["gtrL"]["gain_db"] == build.SAMPLED_MIX["gtrL"]["gain_db"] and sampled["gtrR"] == build.BAND_MIX["gtrR"]
    assert sampled["bass"] == build.BAND_MIX["bass"] and sampled["drums hat"]["duck"] == 1.5


def test_the_band_is_in_the_songs_key_and_falls_back_to_its_synthesised_presets(song):
    from synth2 import arranger
    from synth2.arranger.render import compiled_for

    build, p = song
    arr = json.loads((SONG / "backing.json").read_text(encoding="utf-8"))
    assert arr["key"] == "E minor" and arr["sections"][3]["chords"][:4] == ["C", "D", "Bm", "Em"]
    assert arr["sections"][10]["key"] == "F# minor" and arr["patterns"]["riff_a"]["score"].startswith("E5 e. B5 e.")
    inst = arr["instruments"]
    sounds = {k for k, v in inst.items() if "sound" in v}
    assert sounds == {"drums", "bass", "slap", "gtrL", "gtrR", "mute", "pluck", "bell", "mbox", "pad"}
    assert all("preset" in inst[k] or inst[k].get("type") == "drums" for k in sounds) and inst["drums"]["split"] == "each"
    assert {"tags", "tagslo", "glitch", "sfx"} <= set(inst)  # one kind of sound per stem
    comp = compiled_for(p)
    assert comp.end_beat == 536.0 and not comp.warnings
    assert sum(len(v) for v in comp.note_events.values()) == 4326 and sum(len(v) for v in comp.drum_events.values()) == 2791
    rep = arranger.sound_report(p)
    assert {k for k, v in rep["instruments"].items() if v["sound"]} == sounds
    assert all(v["plays"] == "synth" for v in rep["instruments"].values())
    assert len(rep["warnings"]) == len(sounds) and all("plays instead" in w for w in rep["warnings"]) and not rep["credits"]


def test_two_bars_of_the_chorus_render_with_the_formant_engine_and_the_synthesised_band(song):
    from synth2.engine import render_project

    build, p = song
    res = render_project(p, sample_rate=22050, start_beat=128.0, end_beat=136.0)  # bars 33-34
    rep = res.report
    assert rep["engines"] == ["formant"] and rep["mix"]["mode"] == "song" and "credits" not in rep
    from synth2 import arranger

    fallbacks = [w for w in rep["warnings"] if "plays instead" in w]  # the sounds of the stems that play here, once
    assert len(fallbacks) == len(set(fallbacks)) >= 5 and set(fallbacks) <= set(arranger.sound_report(p)["warnings"])
    assert 8 * 60 / 168 <= len(res.audio) / 22050 < 8 * 60 / 168 + 2.5 and np.isfinite(res.audio).all()  # with the tail
    assert rep["true_peak_db"] <= -0.9 and -40 < rep["loudness_lufs"] < -6
    names = {t["id"]: t["name"] for t in rep["tracks"]}
    sung = {names[k] for k in res.stems if k in names and res.stems.peak(k) > 1e-3}
    assert {"AI", "AI Double", "AI Double R", "Harmony", "Harmony L"} <= sung and "AI Verses" not in sung
    band = {k for k in res.stems if k not in names}
    assert {"drums kick", "drums snare", "bass", "gtrL", "gtrR", "saws"} <= band and "drums cymbals" not in band
    # the two doubles are not the same signal twice
    ids = {v: k for k, v in names.items()}
    left, right = res.stems[ids["AI Double"]].sum(axis=1), res.stems[ids["AI Double R"]].sum(axis=1)
    assert np.corrcoef(left, right)[0, 1] < 0.9
