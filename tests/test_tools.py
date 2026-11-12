"""Tests for the AI layer: composer helpers, tool registry, MCP server and director."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import surface_fakes as fakes
from synth2.ai import composer
from synth2.ai.tools import MUTATING_TOOLS, TOOLS, Session, ToolError, call_tool, tool_names
from synth2.project import Note, Project
from synth2.timing import midi_to_note, note_to_midi

ROOT = Path(__file__).resolve().parents[1]

PLAN_TOOLS = [
    "get_project", "new_project", "update_project", "set_tempo", "list_voices", "list_styles", "list_phonemes",
    "phonemize", "add_track", "update_track", "remove_track", "add_notes", "write_melody", "edit_notes",
    "delete_notes", "set_lyrics", "transpose", "quantize", "set_curve", "clear_curve", "apply_expression",
    "add_harmony", "render", "analyze_render", "save_project", "load_project", "export", "undo", "redo",
]


def names(notes):
    return [midi_to_note(n.pitch) for n in notes]


@pytest.fixture
def s():
    sess = Session()
    call_tool(sess, "new_project", {"title": "Test", "bpm": 120, "key": "C major"})
    return sess


# ================================================================== composer

def test_parse_key_variants():
    assert composer.parse_key("A minor")[:3] == (9, "minor", (9, 11, 0, 2, 4, 5, 7))
    assert composer.parse_key("Am").mode == "minor"
    assert composer.parse_key("Bb").tonic == 10
    assert composer.parse_key("F# dorian").scale == (6, 8, 9, 11, 1, 3, 4)
    assert composer.parse_key("c major pentatonic").scale == (0, 2, 4, 7, 9)
    tonic, mode, scale = composer.parse_key("E major")
    assert (tonic, mode, len(scale)) == (4, "major", 7)
    with pytest.raises(ValueError, match="valid modes"):
        composer.parse_key("C wobbly")
    with pytest.raises(ValueError):
        composer.parse_key("H major")


def test_scale_degree_utilities():
    assert composer.scale_degrees("F major") == ["F", "G", "A", "Bb", "C", "D", "E"]
    assert composer.scale_degrees("F# minor") == ["F#", "G#", "A", "B", "C#", "D", "E"]
    assert composer.scale_degree("E4", "C major") == 3
    assert composer.scale_degree("F#4", "C major") is None
    assert composer.in_scale(71, "G major") and not composer.in_scale(65, "G major")
    assert composer.snap_to_scale("C#4", "C major") == 60
    assert composer.snap_to_scale("C#4", "C major", "up") == 62
    step, alt = composer.step_index(64, "C major")
    assert composer.step_to_pitch(step, "C major") == 64 and alt == 0


def test_harmonize_diatonic_thirds():
    src = [Note(i, 1, note_to_midi(p), "la") for i, p in enumerate(["C4", "D4", "E4", "F4", "G4", "A4", "B4"])]
    up = composer.harmonize(src, 2, "C major")
    assert names(up) == ["E4", "F4", "G4", "A4", "B4", "C5", "D5"]
    down = composer.harmonize(src, -2, "C major")
    assert names(down) == ["A3", "B3", "C4", "D4", "E4", "F4", "G4"]
    fifth = composer.harmonize(src, 4, "C major")
    assert names(fifth)[:3] == ["G4", "A4", "B4"]
    assert all(a.id != b.id for a, b in zip(src, up))
    assert [n.lyric for n in up] == ["la"] * 7


def test_harmonize_chromatic_and_minor():
    src = [Note(0, 1, note_to_midi("F#4"), "a"), Note(1, 1, note_to_midi("Bb4"), "b"),
           Note(2, 1, note_to_midi("Eb4"), "c")]
    stats = {}
    out = composer.harmonize(src, 2, "C major", stats=stats)
    assert names(out) == ["A4", "D5", "G4"]  # consonant thirds above chromatic notes
    assert stats["chromatic_notes"] == 3
    # parallel keeps the chromatic alteration of the neighbouring scale tone
    assert names(composer.harmonize(src, 2, "C major", chromatic="parallel")) == ["A#4", "C#5", "F#4"]
    assert names(composer.harmonize(src, 2, "C major", chromatic="snap")) == ["A4", "C5", "F4"]
    a_minor = [Note(0, 1, note_to_midi("G#4"), "x")]  # leading tone
    assert names(composer.harmonize(a_minor, 2, "A minor")) == ["B4"]


def test_harmonize_range_shifts_whole_phrases():
    src = [Note(i, 1, note_to_midi(p), "la") for i, p in enumerate(["C5", "E5", "G5"])]
    stats = {}
    out = composer.harmonize(src, 2, "C major", voice_range=(45, 72), stats=stats)
    assert names(out) == ["E4", "G4", "B4"]  # whole phrase moved down an octave together
    assert stats["range_shifted"] == 3


def test_transpose_and_errors():
    src = [Note(0, 1, 60, "a"), Note(1, 1, 64, "b")]
    assert names(composer.transpose(src, 12)) == ["C5", "E5"]
    assert names(composer.transpose(src, steps=1, key="C major")) == ["D4", "F4"]
    assert composer.transpose(src, 2)[0].id == src[0].id
    with pytest.raises(ValueError, match="MIDI range"):
        composer.transpose(src, 80)


def test_quantize_never_zero_length_and_resolves_overlaps():
    src = [Note(0.1, 0.05, 60, "a"), Note(0.2, 0.9, 62, "b"), Note(1.9, 0.6, 64, "c")]
    q = composer.quantize(src, 0.5)
    assert all(n.duration > 0 for n in q)
    for a, b in zip(q, q[1:]):
        assert b.onset >= a.end - 1e-9  # no overlaps
    assert q[0].onset == 0.0 and q[2].onset == 2.0
    half = composer.quantize([Note(0.2, 1, 60)], "e", strength=0.5)
    assert abs(half[0].onset - 0.1) < 1e-9
    trip = composer.quantize([Note(0.3, 0.3, 60)], "e3")
    assert abs(trip[0].onset - 1 / 3) < 1e-9


def test_fit_lyrics():
    notes = [Note(i, 1, 60) for i in range(5)]
    out = composer.fit_lyrics(notes, "twinkle twinkle star")
    assert [n.lyric for n in out] == ["twinkle", "+", "twinkle", "+", "star"]
    out = composer.fit_lyrics(notes[:3], "twinkle twinkle little star")
    assert [n.lyric for n in out] == ["twinkle", "+", "twinkle little star"]
    out = composer.fit_lyrics(notes, "oh my")
    assert [n.lyric for n in out] == ["oh", "my", "-", "-", "-"]
    br = [Note(0, 1, 60, "x"), Note(1, 0.5, 60, "br"), Note(1.5, 1, 60, "y")]
    assert [n.lyric for n in composer.fit_lyrics(br, "a b")] == ["a", "br", "b"]
    out = composer.fit_lyrics(notes[:2], "[s t aa r] -")
    assert [n.lyric for n in out] == ["[s t aa r]", "-"]
    ja = composer.fit_lyrics([Note(i, 1, 60) for i in range(3)], "さくら", "ja")
    assert [n.lyric for n in ja] == ["sa", "ku", "ra"]


def test_double():
    d = composer.double([Note(0, 1, 60, "a", attrs={"detune": 2.0})], detune_cents=5, delay_beats=0.02)
    assert d[0].attrs["detune"] == 7.0 and d[0].onset == 0.02


# ================================================================== registry

def test_tool_registry_shape():
    assert len(set(tool_names())) == len(TOOLS)
    for name in PLAN_TOOLS:
        assert name in tool_names(), name
    for t in TOOLS:
        assert set(t) == {"name", "description", "input_schema"}
        assert t["input_schema"]["type"] == "object"
        assert len(t["description"]) > 40
        for req in t["input_schema"].get("required", []):
            assert req in t["input_schema"]["properties"]
    json.dumps(TOOLS)
    assert "write_melody" in MUTATING_TOOLS and "get_project" not in MUTATING_TOOLS


def test_unknown_tool_and_bad_arguments(s):
    with pytest.raises(ToolError, match="Did you mean write_melody"):
        call_tool(s, "write_melodie", {})
    with pytest.raises(ToolError, match="unknown argument 'scor'.*did you mean 'score'"):
        call_tool(s, "write_melody", {"scor": "C4 q la"})
    with pytest.raises(ToolError, match="missing required argument 'score'"):
        call_tool(s, "write_melody", {})
    with pytest.raises(ToolError, match="must be number"):
        call_tool(s, "set_tempo", {"bpm": "fast"})
    with pytest.raises(ToolError, match="must be one of"):
        call_tool(s, "get_project", {"detail": "everything"})
    with pytest.raises(ToolError, match="no track 'Nope'"):
        call_tool(s, "get_notes", {"track": "Nope"})
    with pytest.raises(ToolError, match="out of range"):
        call_tool(s, "get_notes", {"track": 5})


def test_results_are_json_serialisable(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q la D4 q la E4 h la"})
    for name, args in [("get_project", {}), ("get_project", {"detail": "full"}), ("get_project", {"detail": "score"}),
                       ("get_notes", {}), ("list_voices", {}), ("list_styles", {"include_parameters": True}),
                       ("list_phonemes", {"language": "en"}), ("list_parameters", {}),
                       ("phonemize", {"text": "hello beautiful world"}), ("check_project", {})]:
        json.dumps(call_tool(s, name, args))


# ================================================================== project / tracks

def test_new_project_from_score_and_summary():
    sess = Session()
    r = call_tool(sess, "new_project", {"score": "title: S\ntempo: 90\nkey: A minor\ntrack: Lead voice=seraph\n"
                                                 "A4 q la B4 q la C5 h la"})
    assert r["title"] == "S" and r["bpm"] == 90 and r["tracks"][0]["voice"] == "seraph"
    with pytest.raises(ToolError, match="line 2"):
        call_tool(sess, "new_project", {"score": "track: A\nZZ q la"})
    assert sess.project.title == "S"  # failed call left the project untouched
    summ = call_tool(sess, "get_project", {})
    assert summ["scale"] == ["A", "B", "C", "D", "E", "F", "G"] and summ["history"]["undo"] == 1


def test_update_project_mix_settings(s):
    r = call_tool(s, "update_project", {"master": {"loudness_target": -10.5, "vocal_delay": 0.2, "band_width": 1.2,
                                                   "delay_duck_db": 9.0,
                                                   "mix_overrides": {"Lead": {"gain_db": 1.0}, "bass": {"gain_db": -2}}}})
    m = r["master"]
    assert m["loudness_target"] == -10.5 and m["vocal_delay"] == 0.2 and m["band_width"] == 1.2
    assert m["delay_duck_db"] == 9.0 and s.project.master.delay_duck_db == 9.0
    # mix_overrides merge per key; a null entry removes it; null for the whole dict clears it
    r = call_tool(s, "update_project", {"master": {"loudness_target": None,
                                                   "mix_overrides": {"bass": None, "drums snare": {"crest_db": 9}}}})
    assert r["master"]["loudness_target"] is None
    assert r["master"]["mix_overrides"] == {"Lead": {"gain_db": 1.0}, "drums snare": {"crest_db": 9}}
    assert call_tool(s, "update_project", {"master": {"mix_overrides": None}})["master"]["mix_overrides"] == {}
    with pytest.raises(ToolError, match="must be an object"):
        call_tool(s, "update_project", {"master": {"mix_overrides": {"bass": -2}}})
    with pytest.raises(ToolError, match="unknown argument"):
        call_tool(s, "update_project", {"master": {"loudness": -10}})


def test_update_project_and_tempo(s):
    r = call_tool(s, "update_project", {"title": "New", "key": "Am", "time_signature": "3/4",
                                        "master": {"gain_db": -3, "normalize": False}})
    assert r["key"] == "A minor" and r["time_signature"] == "3/4" and r["master"]["gain_db"] == -3
    with pytest.raises(ToolError, match="valid modes"):
        call_tool(s, "update_project", {"key": "C sideways"})
    r = call_tool(s, "set_tempo", {"bpm": 90, "beat": 16})
    assert r["tempo"] == [{"beat": 0.0, "bpm": 120.0}, {"beat": 16.0, "bpm": 90.0}]
    r = call_tool(s, "set_tempo", {"remove": True, "beat": 16})
    assert len(r["tempo"]) == 1
    with pytest.raises(ToolError, match="no tempo change"):
        call_tool(s, "set_tempo", {"remove": True, "beat": 3})
    call_tool(s, "set_tempo", {"bpm": 70, "beat": 8})
    r = call_tool(s, "set_tempo", {"bpm": 100, "replace_all": True})
    assert r["tempo"] == [{"beat": 0.0, "bpm": 100.0}]


def test_track_management(s):
    r = call_tool(s, "add_track", {"name": "Bass", "voice": "basso", "pan": -0.5})
    assert r["track"]["style"] == "classical" and r["track"]["index"] == 1
    with pytest.raises(ToolError, match="unknown voice 'bob'"):
        call_tool(s, "add_track", {"voice": "bob"})
    with pytest.raises(ToolError, match="valid styles"):
        call_tool(s, "update_track", {"track": "Bass", "style": "polka"})
    r = call_tool(s, "update_track", {"track": "bass", "gain_db": -6, "mute": True,
                                      "voice_params": {"breathiness": 0.3}})
    assert r["track"]["gain_db"] == -6 and r["track"]["mute"] and r["track"]["voice_params"] == {"breathiness": 0.3}
    with pytest.raises(ToolError, match="unknown voice parameter"):
        call_tool(s, "update_track", {"track": 1, "voice_params": {"breathyness": 0.3}})
    tid = s.project.tracks[1].id
    r = call_tool(s, "remove_track", {"track": tid})
    assert r["removed"] == "Bass" and len(s.project.tracks) == 1


# ================================================================== notes

def test_write_melody_append_replace_and_warnings(s):
    r = call_tool(s, "write_melody", {"score": "@ 4 C4 q hel  E4 q +  G4 h lo | ", "track": "Lead"})
    assert r["added"] == 3 and r["start_beat"] == 4 and r["end_beat"] == 8 and r["warnings"] == []
    r = call_tool(s, "write_melody", {"score": "A4 q la B4 q. la | C5 h la"})  # appends at beat 8
    assert r["start_beat"] == 8
    assert any("bar line at beat 10.5" in w for w in r["warnings"])
    assert "hint" in r
    r = call_tool(s, "write_melody", {"score": "D5 w oh", "start_beat": 8, "replace": True})
    assert r["removed"] == 3
    notes = s.project.tracks[0].notes
    assert [(n.onset, midi_to_note(n.pitch)) for n in notes] == [(4, "C4"), (5, "E4"), (6, "G4"), (8, "D5")]
    with pytest.raises(ToolError, match="score error: line 1"):
        call_tool(s, "write_melody", {"score": "C4 q la QQ"})
    r = call_tool(s, "write_melody", {"score": "C4 q la", "start_beat": 8.5})
    assert any("overlap" in w for w in r["warnings"])
    r = call_tool(s, "write_melody", {"score": "C2 q la", "start_beat": 20})
    assert any("outside" in w for w in r["warnings"])
    r = call_tool(s, "write_melody", {"score": "curve: loudness 0:-6 8:0 (linear)\ntempo: 100"})
    assert s.project.tracks[0].curves["loudness"].interpolation == "linear"
    assert r["tempo"][-1]["bpm"] == 100


def test_add_edit_delete_notes(s):
    r = call_tool(s, "add_notes", {"notes": [
        {"pitch": "E4", "onset": 1, "duration": 1, "lyric": "hel"},
        {"pitch": 67, "onset": 2, "duration": "h", "lyric": "+", "attrs": {"vibrato_depth": 0.5}},
        {"pitch": "C5", "onset": 4, "duration": 1, "phonemes": "s t aa r"},
    ]})
    ids = [n["id"] for n in r["added"]]
    assert r["added"][1]["duration"] == 2 and r["added"][2]["phonemes"] == "s t aa r"
    with pytest.raises(ToolError, match="invalid notes\\[0\\].pitch"):
        call_tool(s, "add_notes", {"notes": [{"pitch": "H4", "onset": 0, "duration": 1}]})
    with pytest.raises(ToolError, match="unknown note attribute 'vibrato'"):
        call_tool(s, "add_notes", {"notes": [{"pitch": 60, "onset": 0, "duration": 1, "attrs": {"vibrato": 1}}]})
    with pytest.raises(ToolError, match="out of range"):
        call_tool(s, "add_notes", {"notes": [{"pitch": 60, "onset": 0, "duration": 1, "attrs": {"velocity": 5}}]})
    with pytest.raises(ToolError, match="unknown phoneme"):
        call_tool(s, "add_notes", {"notes": [{"pitch": 60, "onset": 0, "duration": 1, "lyric": "[zz]"}]})
    r = call_tool(s, "add_notes", {"notes": [{"pitch": 72, "onset": 1.5, "duration": 1}], "overwrite": True})
    assert r["removed"] == 2
    s.undo()
    r = call_tool(s, "edit_notes", {"edits": [{"id": ids[0], "pitch": "F4", "lyric": "hi", "attrs": {"velocity": 1.4}},
                                              {"id": ids[2], "phonemes": None, "move_by": 1}]})
    assert [e["note"] for e in r["edited"]] == ["F4", "C5"]
    t = s.project.tracks[0]
    assert t.find_note(ids[2]).onset == 5 and t.find_note(ids[2]).phonemes is None
    r = call_tool(s, "edit_notes", {"start_beat": 0, "end_beat": 3, "changes": {"transpose": 2,
                                                                                "attrs": {"velocity": None}}})
    assert len(r["edited"]) == 2 and "velocity" not in t.find_note(ids[0]).attrs
    with pytest.raises(ToolError, match="needs a selection"):
        call_tool(s, "edit_notes", {"changes": {"transpose": 1}})
    with pytest.raises(ToolError, match="not found"):
        call_tool(s, "edit_notes", {"edits": [{"id": "nope", "pitch": 60}]})
    with pytest.raises(ToolError, match="select notes"):
        call_tool(s, "delete_notes", {})
    r = call_tool(s, "delete_notes", {"ids": [ids[1]]})
    assert r["deleted"] == 1 and r["remaining"] == 2
    r = call_tool(s, "delete_notes", {"all": True})
    assert r["remaining"] == 0


def test_set_lyrics(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q C4 q G4 q G4 q | A4 q A4 q G4 h"})
    r = call_tool(s, "set_lyrics", {"text": "twinkle twinkle little star"})
    assert [a["lyric"] for a in r["assigned"]] == ["twinkle", "+", "twinkle", "+", "little", "+", "star"]
    assert "warning" not in r and "info" not in r
    r = call_tool(s, "set_lyrics", {"text": "oh", "start_beat": 8})
    assert [a["lyric"] for a in r["assigned"]] == ["oh", "-", "-"] and "info" in r
    r = call_tool(s, "set_lyrics", {"text": "one two three four five", "start_beat": 8})
    assert r["assigned"][-1]["lyric"] == "three four five" and "warning" in r
    with pytest.raises(ToolError, match="no notes selected"):
        call_tool(s, "set_lyrics", {"text": "x", "start_beat": 100})


def test_transpose_and_quantize_tools(s):
    call_tool(s, "write_melody", {"score": "@ 4.1 C4 0.9 a  E4 1.05 b  G4 1 c"})
    r = call_tool(s, "transpose", {"semitones": 12})
    assert r["range"] == "C5-G5"
    r = call_tool(s, "transpose", {"steps": -1})
    assert r["range"] == "B4-F5"
    with pytest.raises(ToolError, match="give semitones"):
        call_tool(s, "transpose", {})
    with pytest.raises(ToolError, match="MIDI range"):
        call_tool(s, "transpose", {"semitones": 60})
    r = call_tool(s, "quantize", {"grid": "e"})
    assert r["changed"] >= 2
    t = s.project.tracks[0]
    assert [n.onset for n in t.notes] == [4.0, 5.0, 6.0]
    assert not t.overlaps()


def test_curves(s):
    r = call_tool(s, "set_curve", {"curve": "loudness", "points": [[0, -6], {"beat": 8, "value": 0}],
                                   "interpolation": "linear"})
    assert r["points"] == 2 and r["interpolation"] == "linear"
    with pytest.raises(ToolError, match="outside -48..12 dB"):
        call_tool(s, "set_curve", {"curve": "loudness", "points": [[0, 50]]})
    with pytest.raises(ToolError, match="valid curves"):
        call_tool(s, "set_curve", {"curve": "volume", "points": [[0, 1]]})
    call_tool(s, "set_curve", {"curve": "loudness", "points": [[2, -3], [4, -2]], "mode": "replace_range"})
    assert s.project.tracks[0].curves["loudness"].points == [(0.0, -6.0), (2.0, -3.0), (4.0, -2.0), (8.0, 0.0)]
    call_tool(s, "set_curve", {"curve": "tension", "points": [[0, 0.5]]})
    r = call_tool(s, "clear_curve", {"curve": "loudness", "start_beat": 1, "end_beat": 5})
    assert r["removed_points"] == 2
    r = call_tool(s, "clear_curve", {})
    assert r["curves"] == []


def test_apply_expression(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q la D4 q la E4 h la"})
    r = call_tool(s, "apply_expression", {"style": "ballad", "style_overrides": {"vibrato_depth": 1.4, "scoop": 0},
                                          "voice_params": {"breathiness": 0.2},
                                          "note_attrs": {"velocity": 1.2}, "start_beat": 6})
    assert r["style"] == "ballad" and r["effective_style"]["vibrato_depth"] == 1.4
    assert r["effective_style"]["scoop"] == 0 and r["notes_updated"] == 1
    t = s.project.tracks[0]
    assert t.voice_params == {"style.vibrato_depth": 1.4, "style.scoop": 0.0, "breathiness": 0.2}
    assert t.notes[2].attrs == {"velocity": 1.2}
    with pytest.raises(ToolError, match="unknown style parameter 'vibrato'"):
        call_tool(s, "apply_expression", {"style_overrides": {"vibrato": 1}})
    r = call_tool(s, "apply_expression", {"reset_overrides": True})
    assert r["style_overrides"] == {}
    # the score format round-trips style overrides
    call_tool(s, "apply_expression", {"style_overrides": {"fall": 0.2}})
    text = call_tool(s, "get_project", {"detail": "score"})["score"]
    assert "style.fall=0.2" in text


def test_add_harmony_and_duplicate(s):
    call_tool(s, "write_melody", {"score": "@ 4 C5 q hel  E5 q +  G5 h lo  | A5 h oh"})
    r = call_tool(s, "add_harmony", {"steps": -2, "voice": "atlas"})
    t = s.project.tracks[1]
    assert r["track"]["voice"] == "atlas" and r["notes"] == 4
    assert [n.lyric for n in t.notes] == ["hel", "+", "lo", "oh"]
    assert all(45 <= n.pitch <= 72 for n in t.notes)  # fitted into atlas's range
    assert t.gain_db == -4 and t.pan < 0
    r = call_tool(s, "add_harmony", {"steps": 2, "start_beat": 8, "fit_range": False, "name": "High"})
    assert names(s.project.tracks[2].notes) == ["C6"]
    with pytest.raises(ToolError, match="duplicate_track"):
        call_tool(s, "add_harmony", {"steps": 0})
    r = call_tool(s, "duplicate_track", {"track": 0, "detune_cents": 6, "transpose": -12, "name": "Double"})
    d = s.project.tracks[3]
    assert d.name == "Double" and names(d.notes)[0] == "C4" and d.notes[0].attrs["detune"] == 6
    assert d.notes[0].id != s.project.tracks[0].notes[0].id
    r = call_tool(s, "add_harmony", {"steps": 2, "into_track": "High", "start_beat": 4, "end_beat": 6})
    assert len(s.project.tracks[2].notes) == 3


def test_check_project(s):
    call_tool(s, "write_melody", {"score": "C4 q twin  C4 q +  D4 q twinkle  E3 q -"})
    call_tool(s, "add_notes", {"notes": [{"pitch": "C2", "onset": 3.5, "duration": 1, "lyric": "low"}]})
    r = call_tool(s, "check_project", {})
    msgs = "\n".join(i["message"] for i in r["issues"])
    assert not r["ok"] and r["errors"] >= 1
    assert "overlap" in msgs
    assert "first note starts at beat 0" in msgs
    assert "'twin' has 1 syllable" in msgs
    assert "'twinkle' (2 syllables) is sung on a single note" in msgs
    assert "below Aurora's range" in msgs


def test_phonemize_and_lists(s):
    r = call_tool(s, "phonemize", {"text": "twinkle little star zyxq"})
    assert r["lyric_tokens"] == ["twinkle", "+", "little", "+", "star", "zyxq"]
    words = {w["word"]: w for w in r["words"]}
    assert words["twinkle"]["in_dictionary"] and not words["zyxq"]["in_dictionary"]
    assert len(words["twinkle"]["syllables"]) == 2
    r = call_tool(s, "phonemize", {"text": "さくら", "language": "ja"})
    assert r["words"][0]["morae"] == ["sa", "ku", "ra"]
    ph = call_tool(s, "list_phonemes", {"language": "ja"})["phonemes"]
    assert "N" in {p["name"] for p in ph} and "th" not in {p["name"] for p in ph}
    assert {v["id"] for v in call_tool(s, "list_voices", {})["voices"]} >= {"aurora", "atlas"}
    st = call_tool(s, "list_styles", {})
    assert "vibrato_depth" in st["override_parameters"]
    params = call_tool(s, "list_parameters", {})
    assert params["curves"]["loudness"]["unit"] == "dB" and "velocity" in params["note_attributes"]


def test_get_project_details(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q la D4 q la | E4 w la"})
    full = call_tool(s, "get_project", {"detail": "full", "track": 0, "start_beat": 5})
    assert [n["note"] for n in full["tracks"][0]["notes"]] == ["D4", "E4"]
    sc = call_tool(s, "get_project", {"detail": "score", "start_beat": 5, "end_beat": 6})
    assert sc["score"].splitlines()[0] == "@ 5" and "D4 q la" in sc["score"]
    notes = call_tool(s, "get_notes", {"format": "score"})
    assert "C4 q la" in notes["score"]
    summ = call_tool(s, "get_project", {"track": "Lead"})
    assert summ["track"]["notes"] == 3


# ================================================================== history

def test_undo_redo_and_atomicity(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q la"})
    call_tool(s, "write_melody", {"score": "D4 q la"})
    before = s.snapshot()
    n_undo = len(s.undo_stack)
    with pytest.raises(ToolError):
        call_tool(s, "edit_notes", {"edits": [{"id": s.project.tracks[0].notes[0].id, "pitch": 62},
                                              {"id": "missing", "pitch": 60}]})
    assert s.snapshot() == before and len(s.undo_stack) == n_undo  # failed call changed nothing
    call_tool(s, "get_project", {})
    assert len(s.undo_stack) == n_undo  # reads push nothing
    r = call_tool(s, "undo", {})
    assert r["undone"][0].startswith("write_melody") and len(s.project.tracks[0].notes) == 1
    r = call_tool(s, "redo", {})
    assert len(s.project.tracks[0].notes) == 2
    call_tool(s, "undo", {"steps": 5})
    with pytest.raises(ToolError, match="nothing to undo"):
        call_tool(s, "undo", {})
    call_tool(s, "redo", {"steps": 2})
    call_tool(s, "write_melody", {"score": "E4 q la"})
    with pytest.raises(ToolError, match="nothing to redo"):
        call_tool(s, "redo", {})


def test_history_is_bounded():
    sess = Session()
    for i in range(110):
        call_tool(sess, "set_tempo", {"bpm": 60 + i})
    assert len(sess.undo_stack) == 100


# ================================================================== files

def test_save_load_export(s, tmp_path):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q hello  D4 q +  E4 h world"})
    r = call_tool(s, "save_project", {"path": str(tmp_path / "song")})
    assert r["path"].endswith("song.s2p.json") and s.path == r["path"]
    r = call_tool(s, "save_project", {"path": str(tmp_path / "song.s2s")})
    assert r["format"] == "score"
    r = call_tool(s, "export", {"path": str(tmp_path / "copy"), "format": "score"})
    assert r["path"].endswith("copy.s2s")
    r = call_tool(s, "export", {"path": str(tmp_path / "x.json")})
    assert r["format"] == "json"
    with pytest.raises(ToolError, match="cannot infer"):
        call_tool(s, "export", {"path": str(tmp_path / "x.zzz")})
    fresh = Session()
    r = call_tool(fresh, "load_project", {"path": str(tmp_path / "song.s2s")})
    assert r["summary"]["tracks"][0]["notes"] == 3
    assert [n.lyric for n in fresh.project.tracks[0].notes] == ["hello", "+", "world"]
    call_tool(fresh, "load_project", {"path": str(tmp_path / "song.s2p.json")})
    assert fresh.path.endswith("song.s2p.json")
    call_tool(fresh, "undo", {})  # load is undoable
    with pytest.raises(ToolError, match="file not found"):
        call_tool(fresh, "load_project", {"path": str(tmp_path / "missing.json")})
    opened = Session.open(tmp_path / "song.s2p.json")
    assert opened.project.title == "Test"


def test_export_foreign_formats(s, tmp_path):
    pytest.importorskip("synth2.io.midi")
    io = pytest.importorskip("synth2.io")
    if not hasattr(io, "export_any"):
        pytest.skip("synth2.io.export_any not available yet")
    call_tool(s, "write_melody", {"score": "@ 4 C4 q hello  D4 q +  E4 h world"})
    r = call_tool(s, "export", {"path": str(tmp_path / "song"), "format": "midi"})
    assert Path(r["path"]).exists() and r["path"].endswith(".mid")


# ================================================================== render

def test_render_and_analyze(s, tmp_path):
    pytest.importorskip("synth2.engine.renderer")
    pytest.importorskip("synth2.analysis")
    with pytest.raises(ToolError, match="nothing to render"):
        call_tool(s, "render", {})
    with pytest.raises(ToolError, match="render first"):
        call_tool(s, "analyze_render", {})
    call_tool(s, "write_melody", {"score": "@ 1 A4 q la  C5 q la"})
    out = tmp_path / "take.wav"
    r = call_tool(s, "render", {"path": str(out), "stems": True})
    assert Path(r["path"]).exists() and r["report"]["duration_s"] > 0.5
    assert "phonemes" not in r["report"]["tracks"][0]
    assert len(r["stems"]) == 1 and Path(r["stems"][0]).exists()
    json.dumps(r)
    a = call_tool(s, "analyze_render", {})
    assert {"audio", "pitch", "issues", "ok"} <= set(a)
    assert a["pitch"]["tracks"][0]["notes_checked"] >= 1
    json.dumps(a)
    call_tool(s, "transpose", {"semitones": 1})
    a = call_tool(s, "analyze_render", {"include_pitch": False})
    assert any("changed since this render" in i for i in a["issues"])


def test_render_default_path(s, tmp_path, monkeypatch):
    pytest.importorskip("synth2.engine.renderer")
    monkeypatch.chdir(tmp_path)
    call_tool(s, "write_melody", {"score": "@ 1 A4 e la"})
    r = call_tool(s, "render", {"end_beat": 2})
    assert Path(r["path"]) == Path("out/Test.wav") and (tmp_path / "out" / "Test.wav").exists()
    s.path = str(tmp_path / "proj" / "song.s2p.json")
    r = call_tool(s, "render", {"end_beat": 2})
    assert r["path"] == str(tmp_path / "proj" / "song.wav")


# ================================================================== MCP server

def test_mcp_server_in_process():
    from synth2.ai.mcp_server import MCPServer

    srv = MCPServer(Session())
    init = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                       "params": {"protocolVersion": "2025-03-26", "capabilities": {}}})
    assert init["result"]["protocolVersion"] == "2025-03-26"
    assert init["result"]["serverInfo"]["name"] == "synth2"
    assert srv.handle({"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {}})["result"][
        "protocolVersion"] == "2025-06-18"
    assert srv.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert srv.handle({"jsonrpc": "2.0", "id": 2, "method": "ping"}) == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert srv.handle({"jsonrpc": "2.0", "id": 3, "method": "nope"})["error"]["code"] == -32601
    assert srv.handle({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                       "params": {"name": "zzz"}})["error"]["code"] == -32602
    batch = srv.handle([{"jsonrpc": "2.0", "id": 5, "method": "ping"}, {"jsonrpc": "2.0", "method": "x/y"}])
    assert batch == [{"jsonrpc": "2.0", "id": 5, "result": {}}]


def _rpc(proc, msg):
    proc.stdin.write((json.dumps(msg) + "\n").encode())
    proc.stdin.flush()
    if "id" not in msg:
        return None
    line = proc.stdout.readline()
    assert line, proc.stderr.read().decode()
    return json.loads(line)


def test_mcp_server_subprocess(tmp_path):
    proj = tmp_path / "mcp.s2p.json"
    env = dict(os.environ, PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.Popen([sys.executable, "-m", "synth2.ai.mcp_server", "--project", str(proj), "--autosave"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=str(tmp_path),
                            env=env)
    try:
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                        "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                                   "clientInfo": {"name": "pytest", "version": "1"}}})
        assert r["id"] == 1 and r["result"]["capabilities"] == {"tools": {}}
        assert r["result"]["protocolVersion"] == "2025-06-18"
        _rpc(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = r["result"]["tools"]
        assert {t["name"] for t in tools} >= set(PLAN_TOOLS)
        assert all("inputSchema" in t and "description" in t for t in tools)
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "write_melody", "arguments": {"score": "@ 4 C4 q hel E4 q + G4 h lo"}}})
        assert r["result"]["isError"] is False
        payload = json.loads(r["result"]["content"][0]["text"])
        assert payload["added"] == 3
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                        "params": {"name": "add_notes", "arguments": {"notes": [{"pitch": "X1", "onset": 0,
                                                                                 "duration": 1}]}}})
        assert r["result"]["isError"] is True and "invalid" in r["result"]["content"][0]["text"]
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 5, "method": "resources/list"})
        assert r["error"]["code"] == -32601
        r = _rpc(proc, {"jsonrpc": "2.0", "id": 6, "method": "ping"})
        assert r["result"] == {}
        proc.stdin.write(b"{not json\n")
        proc.stdin.flush()
        assert json.loads(proc.stdout.readline())["error"]["code"] == -32700
    finally:
        proc.stdin.close()
        proc.wait(timeout=20)
        proc.stdout.close()
        proc.stderr.close()
    assert proc.returncode == 0
    saved = Project.load(proj)  # --autosave wrote the project after the mutating call
    assert len(saved.tracks[0].notes) == 3


# ================================================================== director

class _FakeStream:
    def __init__(self, msg):
        self.msg = msg

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self.msg


class FakeClient:
    """Mimics ``anthropic.Anthropic().beta.messages.stream`` with scripted responses."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self._stream))

    def _stream(self, **params):
        self.calls.append(json.loads(json.dumps(params, default=lambda o: o.__dict__)))
        return _FakeStream(self.responses.pop(0))


def _msg(stop, *blocks):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


def _text(t):
    return SimpleNamespace(type="text", text=t)


def _tool(i, name, args):
    return SimpleNamespace(type="tool_use", id=i, name=name, input=args)


def test_director_tool_loop():
    from synth2.ai.director import DEFAULT_MODEL, Director

    sess = Session()
    client = FakeClient([
        _msg("tool_use", _text("Writing the melody."),
             _tool("t1", "write_melody", {"score": "@ 4 C4 q la D4 q la E4 h la"}),
             _tool("t2", "set_lyrics", {"text": "hello there"})),
        _msg("tool_use", SimpleNamespace(type="thinking", thinking="", signature="sig"),
             _tool("t3", "add_notes", {"notes": [{"pitch": "Q9", "onset": 0, "duration": 1}]})),
        _msg("end_turn", _text("Done: wrote 3 notes in bar 2.")),
    ])
    d = Director(sess, client=client)
    assert d.model == DEFAULT_MODEL
    out = d.chat("write a tiny melody")
    assert out["reply"] == "Done: wrote 3 notes in bar 2."
    assert [a["tool"] for a in out["actions"]] == ["write_melody", "set_lyrics", "add_notes"]
    assert [a["ok"] for a in out["actions"]] == [True, True, False]
    assert [n.lyric for n in sess.project.tracks[0].notes] == ["hello", "+", "there"]
    # history: user, assistant, user(tool results), assistant, user(tool results), assistant
    roles = [m["role"] for m in d.history]
    assert roles == ["user", "assistant", "user", "assistant", "user", "assistant"]
    results = d.history[2]["content"]
    assert [r["tool_use_id"] for r in results] == ["t1", "t2"] and not results[0].get("is_error")
    err = d.history[4]["content"][0]
    assert err["is_error"] is True and "invalid" in err["content"]
    assert d.history[3]["content"][0].type == "thinking"  # thinking blocks echoed back unchanged
    first = client.calls[0]
    assert first["model"] == DEFAULT_MODEL
    assert first["fallbacks"] == "default" and first["betas"] == ["server-side-fallback-2026-07-01"]
    assert first["cache_control"] == {"type": "ephemeral"}
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert first["output_config"]["effort"]
    assert "thinking" not in first and "tool_choice" not in first
    assert len(first["tools"]) == len(TOOLS) and all(t["eager_input_streaming"] for t in first["tools"])
    assert "<studio_state>" in first["messages"][0]["content"][0]["text"]
    # system prompt and tools are identical on every request (cache-stable, append-only history)
    assert all(c["system"] == first["system"] and c["tools"] == first["tools"] for c in client.calls)
    # conversation continues across chat() calls
    client.responses.append(_msg("end_turn", _text("Sure.")))
    out2 = d.chat("thanks")
    assert out2["reply"] == "Sure." and out2["actions"] == []
    assert len(client.calls[-1]["messages"]) == 7


def test_director_edge_cases(monkeypatch):
    from synth2.ai.director import Director

    sess = Session()
    # refusal: the exchange is dropped from the history
    client = FakeClient([_msg("refusal")])
    d = Director(sess, client=client)
    out = d.chat("something")
    assert "declined" in out["reply"] and d.history == []
    # truncated tool call at max_tokens is answered with an error and not executed
    client = FakeClient([_msg("max_tokens", _tool("a", "write_melody", {"score": "C4 q"})),
                         _msg("end_turn", _text("ok"))])
    d = Director(sess, client=client)
    out = d.chat("go")
    assert out["actions"][0]["ok"] is False and not sess.project.tracks[0].notes
    assert d.history[2]["content"][0]["is_error"] is True
    # max_turns limit
    loop = [_msg("tool_use", _tool(f"id{i}", "get_project", {})) for i in range(3)]
    d = Director(sess, client=FakeClient(loop), max_turns=3)
    out = d.chat("loop")
    assert out["turns"] == 3 and "Stopped after 3" in out["reply"]
    # model override from the environment; fallbacks only on supporting models
    monkeypatch.setenv("SYNTH2_MODEL", "claude-haiku-5-5")
    client = FakeClient([_msg("end_turn", _text("hi"))])
    d = Director(sess, client=client)
    d.chat("hi")
    assert client.calls[0]["model"] == "claude-haiku-5-5" and "fallbacks" not in client.calls[0]
    d = Director(sess, model="claude-opus-5", client=FakeClient([]))
    assert d.model == "claude-opus-5"


def test_director_echoes_fallback_turns():
    from synth2.ai.director import _echo_content

    blocks = [SimpleNamespace(type="thinking"), SimpleNamespace(type="tool_use"), SimpleNamespace(type="text"),
              SimpleNamespace(type="fallback"), SimpleNamespace(type="thinking"), SimpleNamespace(type="text")]
    assert [b.type for b in _echo_content(blocks)] == ["text", "fallback", "thinking", "text"]
    assert _echo_content(blocks[:3]) == blocks[:3]


def test_director_unavailable_without_credentials(monkeypatch, tmp_path):
    from synth2.ai import director as dmod

    for k in dmod._CREDENTIAL_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    ok, reason = dmod.director_available()
    if "not installed" in reason:
        pytest.skip("anthropic not installed")
    assert not ok and "ANTHROPIC_API_KEY" in reason
    with pytest.raises(dmod.DirectorUnavailable, match="ANTHROPIC_API_KEY"):
        dmod.Director(Session())


def test_director_with_real_sdk_against_local_stub():
    """The real anthropic SDK accepts the director's request shape (no network: a local SSE stub)."""
    anthropic = pytest.importorskip("anthropic")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from synth2.ai.director import Director

    def stream_body(blocks, stop):
        ev = [{"type": "message_start", "message": {"id": "msg_1", "type": "message", "role": "assistant",
                                                    "model": "claude-opus-5-5", "content": [], "stop_reason": None,
                                                    "stop_sequence": None,
                                                    "usage": {"input_tokens": 10, "output_tokens": 1}}}]
        for i, b in enumerate(blocks):
            if b["type"] == "tool_use":
                ev.append({"type": "content_block_start", "index": i, "content_block": {**b, "input": {}}})
                ev.append({"type": "content_block_delta", "index": i,
                           "delta": {"type": "input_json_delta", "partial_json": json.dumps(b["input"])}})
            else:
                ev.append({"type": "content_block_start", "index": i, "content_block": {"type": "text", "text": ""}})
                ev.append({"type": "content_block_delta", "index": i, "delta": {"type": "text_delta",
                                                                                "text": b["text"]}})
            ev.append({"type": "content_block_stop", "index": i})
        ev.append({"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                   "usage": {"output_tokens": 5}})
        ev.append({"type": "message_stop"})
        return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev).encode()

    responses = [
        stream_body([{"type": "text", "text": "Writing."},
                     {"type": "tool_use", "id": "toolu_1", "name": "write_melody",
                      "input": {"score": "@ 4 C4 q la D4 q la"}}], "tool_use"),
        stream_body([{"type": "text", "text": "Done."}], "end_turn"),
    ]
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["content-length"])))
            seen.append((dict(self.headers), body))
            data = responses.pop(0)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{srv.server_port}",
                                     http_client=anthropic.DefaultHttpxClient(trust_env=False), max_retries=0)
        sess = Session()
        out = Director(sess, client=client).chat("write two notes")
    finally:
        srv.shutdown()
    assert out["reply"] == "Done." and out["actions"][0]["ok"]
    assert len(sess.project.tracks[0].notes) == 2
    headers, body = seen[1]
    assert headers.get("anthropic-beta") == "server-side-fallback-2026-07-01"
    assert body["model"] == "claude-opus-5-5" and body["fallbacks"] == "default"
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    assert body["messages"][2]["content"][0]["tool_use_id"] == "toolu_1"


# ================================================================== regressions (MCP server, director)

def _serve_lines(srv, *lines):
    import io

    from synth2.ai.mcp_server import serve

    out = io.BytesIO()
    serve(srv, io.BytesIO(b"".join(ln + b"\n" for ln in lines)), out)
    return [json.loads(x) for x in out.getvalue().decode("ascii").splitlines()]


def test_mcp_lone_surrogate_and_internal_errors_do_not_end_the_session(monkeypatch):
    from synth2.ai.mcp_server import INTERNAL_ERROR, MCPServer

    srv = MCPServer(Session())
    ping = b'{"jsonrpc":"2.0","id":9,"method":"ping"}'
    out = _serve_lines(
        srv,
        b'{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"update_project",'
        b'"arguments":{"title":"Love \\ud83d"}}}',
        b'{"jsonrpc":"2.0","id":"\\ud83d","method":"ping"}',  # surrogate in the id
        b'{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"x\\ud83d"}}',  # echoed in the error
        ping)
    assert [r["id"] for r in out] == [1, "\ud83d", 3, 9]
    assert out[0]["result"]["isError"] is False and srv.session.project.title == "Love \ud83d"
    assert out[2]["error"]["code"] == -32602 and out[3]["result"] == {}
    # any unexpected failure while handling a line is answered and the loop goes on
    real = srv.handle
    monkeypatch.setattr(srv, "handle", lambda msg: 1 / 0 if msg.get("id") == 1 else real(msg))
    out = _serve_lines(srv, b'{"jsonrpc":"2.0","id":1,"method":"ping"}', ping)
    assert out[0] == {"jsonrpc": "2.0", "id": 1, "error": {"code": INTERNAL_ERROR,
                                                             "message": "internal error: ZeroDivisionError"}}
    assert out[1]["result"] == {}


def test_mcp_batches_do_not_nest_and_deep_nesting_is_rejected():
    from synth2.ai.mcp_server import INVALID_REQUEST, PARSE_ERROR, MCPServer

    srv = MCPServer(Session())
    ping = {"jsonrpc": "2.0", "id": 6, "method": "ping"}
    assert srv.handle([[ping]]) == [{"jsonrpc": "2.0", "id": None,
                                     "error": {"code": INVALID_REQUEST, "message": "invalid JSON-RPC 2.0 message"}}]
    assert srv.handle([ping, [ping]])[0] == {"jsonrpc": "2.0", "id": 6, "result": {}}
    deep = ("[" * 600 + json.dumps(ping) + "]" * 600).encode()
    out = _serve_lines(srv, deep, b"[" * 100000, json.dumps(ping).encode())
    assert out[0][0]["error"]["code"] == INVALID_REQUEST
    assert out[1]["error"]["code"] == PARSE_ERROR and out[1]["id"] is None
    assert out[2] == {"jsonrpc": "2.0", "id": 6, "result": {}}


def test_mcp_initialize_answers_only_supported_protocol_versions():
    from synth2.ai.mcp_server import DEFAULT_PROTOCOL_VERSION, SUPPORTED_PROTOCOL_VERSIONS, MCPServer

    srv = MCPServer(Session())

    def negotiated(v):
        return srv.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize",
                           "params": {"protocolVersion": v}})["result"]["protocolVersion"]

    assert DEFAULT_PROTOCOL_VERSION in SUPPORTED_PROTOCOL_VERSIONS
    for v in SUPPORTED_PROTOCOL_VERSIONS:
        assert negotiated(v) == v
    for v in ("1999-01-01", "garbage", "2099-01-01", 123, ["2025-06-18"], None):
        assert negotiated(v) == DEFAULT_PROTOCOL_VERSION


def test_mcp_autosave_follows_undo_and_redo(tmp_path, monkeypatch):
    from synth2.ai import mcp_server

    # the server must not rely on the tool registry listing undo/redo as mutating
    monkeypatch.setattr(mcp_server, "MUTATING_TOOLS", set(MUTATING_TOOLS) - {"undo", "redo"})
    path = tmp_path / "song.s2p.json"
    srv = mcp_server.MCPServer(Session(path=path), autosave=True)

    def tool(name, args=None):
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args or {}}})
        assert r["result"]["isError"] is False, r

    def saved_notes():
        return len(Project.load(path).tracks[0].notes)

    tool("write_melody", {"score": "@ 4 C4 q a D4 q b E4 q c"})
    assert saved_notes() == 3
    tool("undo")
    assert not srv.session.project.tracks[0].notes and saved_notes() == 0
    tool("redo")
    assert saved_notes() == 3


class _ScriptedStream(_FakeStream):
    def get_final_message(self):
        if isinstance(self.msg, BaseException):
            raise self.msg
        return self.msg


class ScriptedClient(FakeClient):
    """FakeClient whose scripted responses may also be exceptions (raised by get_final_message)."""

    def _stream(self, **params):
        self.calls.append(json.loads(json.dumps(params, default=lambda o: o.__dict__)))
        return _ScriptedStream(self.responses.pop(0))


def _user_texts(call):
    return [b["text"] for m in call["messages"] if m["role"] == "user" for b in m["content"] if b.get("text")]


def test_director_reissues_request_when_streamed_tool_json_is_unparseable():
    from synth2.ai.director import MAX_STREAM_RETRIES, Director

    bad = ValueError("Unable to parse tool parameter JSON from model. Please retry your request")
    client = ScriptedClient([bad, _msg("end_turn", _text("Done."))])
    d = Director(Session(), client=client)
    assert d.chat("go")["reply"] == "Done." and len(client.calls) == 2
    assert client.calls[0]["messages"] == client.calls[1]["messages"]
    # retries are bounded; when they run out the error surfaces and the history is left clean
    client = ScriptedClient([bad] * (MAX_STREAM_RETRIES + 2))
    d = Director(Session(), client=client)
    with pytest.raises(ValueError, match="Unable to parse"):
        d.chat("go")
    assert len(client.calls) == MAX_STREAM_RETRIES + 1 and d.history == []
    # other errors (the SDK's typed API errors, connection errors...) are never retried here
    client = ScriptedClient([RuntimeError("overloaded"), _msg("end_turn", _text("x"))])
    with pytest.raises(RuntimeError):
        Director(Session(), client=client).chat("go")
    assert len(client.calls) == 1
    anthropic = pytest.importorskip("anthropic")

    class TypedValueError(anthropic.AnthropicError, ValueError):
        pass

    client = ScriptedClient([TypedValueError("api"), _msg("end_turn", _text("x"))])
    with pytest.raises(TypedValueError):
        Director(Session(), client=client).chat("go")
    assert len(client.calls) == 1


def test_director_failed_chat_is_not_resent_with_the_next_message(monkeypatch):
    from synth2.ai import director as dmod

    # the first request fails before anything happened: the request is forgotten
    client = ScriptedClient([RuntimeError("529 overloaded"), _msg("end_turn", _text("Lowered."))])
    d = dmod.Director(Session(), client=client)
    with pytest.raises(RuntimeError):
        d.chat("delete every track except the lead")
    assert d.history == []
    d.chat("never mind, just lower the harmony gain by 3 dB")
    texts = _user_texts(client.calls[-1])
    assert len(texts) == 1 and "never mind" in texts[0] and "delete every track" not in texts[0]
    # it fails after some tool calls completed: they stay in the history and the turn is closed
    sess = Session()
    client = ScriptedClient([_msg("tool_use", _tool("t1", "write_melody", {"score": "@ 4 C4 q la D4 q la"})),
                             RuntimeError("connection reset"), _msg("end_turn", _text("ok"))])
    d = dmod.Director(sess, client=client)
    with pytest.raises(RuntimeError):
        d.chat("add a harmony track")
    assert len(sess.project.tracks[0].notes) == 2
    assert [m["role"] for m in d.history] == ["user", "assistant", "user", "assistant"]
    assert "RuntimeError" in d.history[-1]["content"][0]["text"]
    d.chat("try again")
    roles = [m["role"] for m in client.calls[-1]["messages"]]
    assert roles == ["user", "assistant", "user", "assistant", "user"]  # never two user turns in a row
    # Ctrl-C while a tool runs: the tool_use without a tool_result is dropped so later requests stay valid
    client = ScriptedClient([_msg("tool_use", _tool("t1", "write_melody", {"score": "C4 q la"}))])
    d = dmod.Director(Session(), client=client, include_state=False)

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(dmod, "call_tool", interrupted)
    with pytest.raises(KeyboardInterrupt):
        d.chat("write")
    assert d.history == []


def _stub_anthropic(replies):
    """A real ``anthropic`` client talking to a local HTTP stub that answers with scripted (status, body) pairs."""
    anthropic = pytest.importorskip("anthropic")
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["content-length"]))))
            status, body = replies.pop(0)
            self.send_response(status)
            self.send_header("content-type", "text/event-stream" if status == 200 else "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = anthropic.Anthropic(api_key="test", base_url=f"http://127.0.0.1:{srv.server_port}",
                                 http_client=anthropic.DefaultHttpxClient(trust_env=False), max_retries=0)
    return anthropic, client, seen, srv


def _sse(*events):
    start = {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [],
        "stop_reason": None, "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 1}}}
    ev = [start, *events, {"type": "message_stop"}]
    return 200, "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in ev).encode()


def _sse_text(text, stop="end_turn"):
    return _sse({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": stop, "stop_sequence": None},
                 "usage": {"output_tokens": 5}})


def test_director_with_real_sdk_retries_bad_tool_json_and_recovers_from_overload():
    from synth2.ai.director import Director

    bad_tool = _sse({"type": "content_block_start", "index": 0, "content_block": {
                        "type": "tool_use", "id": "toolu_1", "name": "write_melody", "input": {}}},
                    {"type": "content_block_delta", "index": 0,
                     "delta": {"type": "input_json_delta", "partial_json": "not json at all"}},
                    {"type": "content_block_stop", "index": 0},
                    {"type": "message_delta", "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                     "usage": {"output_tokens": 5}})
    overloaded = (529, json.dumps({"type": "error", "error": {"type": "overloaded_error",
                                                              "message": "Overloaded"}}).encode())
    anthropic, client, seen, srv = _stub_anthropic([bad_tool, _sse_text("Fixed it."), overloaded,
                                                    _sse_text("Lowered.")])
    try:
        d = Director(Session(), client=client)
        assert d.chat("write a melody")["reply"] == "Fixed it."  # the unparseable stream was re-issued
        assert len(seen) == 2 and seen[0]["messages"] == seen[1]["messages"]
        with pytest.raises(anthropic.OverloadedError):
            d.chat("delete every track except the lead")
        d.chat("never mind, just lower the harmony gain by 3 dB")
    finally:
        srv.shutdown()
    body = seen[-1]
    assert [m["role"] for m in body["messages"]] == ["user", "assistant", "user"]
    last = body["messages"][-1]["content"][0]["text"]
    assert "never mind" in last and "delete every track" not in json.dumps(body["messages"])


# ================================================================== regressions (tools review)

def _saved_song(tmp_path, name, score):
    sess = Session()
    call_tool(sess, "new_project", {"title": name})
    call_tool(sess, "write_melody", {"score": score})
    path = tmp_path / f"{name}.s2p.json"
    call_tool(sess, "save_project", {"path": str(path)})
    return sess, path


def test_undo_of_load_and_new_project_restores_the_file_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sa, path_a = _saved_song(tmp_path, "SongA", "@ 4 C4 q a")
    _, path_b = _saved_song(tmp_path, "SongB", "@ 4 G4 h bee")
    call_tool(sa, "load_project", {"path": str(path_b)})
    assert sa.path == str(path_b)
    r = call_tool(sa, "undo", {})
    assert sa.path == str(path_a) and r["path"] == str(path_a) and sa.project.title == "SongA"
    call_tool(sa, "save_project", {})  # must write A's own file, never over B
    assert Project.load(path_b).title == "SongB" and Project.load(path_a).title == "SongA"
    call_tool(sa, "redo", {})
    assert sa.path == str(path_b) and sa.project.title == "SongB"
    call_tool(sa, "undo", {})
    call_tool(sa, "new_project", {"title": "Fresh"})
    assert sa.path is None
    call_tool(sa, "undo", {})
    assert sa.path == str(path_a) and sa.project.title == "SongA"
    # an ordinary edit does not touch the path, even after a "save as"
    call_tool(sa, "set_tempo", {"bpm": 100})
    call_tool(sa, "save_project", {"path": str(tmp_path / "copy.s2p.json")})
    call_tool(sa, "undo", {})
    assert sa.path == str(tmp_path / "copy.s2p.json")
    # a failed call restores the path along with the project
    sa.path = str(path_a)
    with pytest.raises(ToolError):
        call_tool(sa, "load_project", {"path": str(tmp_path / "missing.s2p.json")})
    assert sa.path == str(path_a)


def test_mcp_autosave_after_undo_writes_the_restored_file(tmp_path):
    from synth2.ai.mcp_server import MCPServer

    assert {"undo", "redo"} <= MUTATING_TOOLS
    sa, path_a = _saved_song(tmp_path, "SongA", "@ 4 C4 q a")
    _, path_b = _saved_song(tmp_path, "SongB", "@ 4 G4 h bee")
    srv = MCPServer(sa, autosave=True)

    def tool(name, args, i=[0]):
        i[0] += 1
        r = srv.handle({"jsonrpc": "2.0", "id": i[0], "method": "tools/call",
                        "params": {"name": name, "arguments": args}})
        assert not r["result"]["isError"], r
        return json.loads(r["result"]["content"][0]["text"])

    tool("load_project", {"path": str(path_b)})
    tool("undo", {})
    tool("set_tempo", {"bpm": 90})
    a, b = Project.load(path_a), Project.load(path_b)
    assert b.title == "SongB" and [n.lyric for n in b.tracks[0].notes] == ["bee"]
    assert a.title == "SongA" and a.bpm == 90


def test_history_and_rollback_keep_exact_triplet_timing(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 e3 la D4 e3 la E4 e3 la F4 q la"})
    query = {"start_beat": 4 + 1 / 3, "end_beat": 5}
    assert call_tool(s, "get_notes", query)["count"] == 2
    with pytest.raises(ToolError):
        call_tool(s, "add_notes", {"notes": [{"pitch": "C4", "onset": 10, "duration": "zz"}]})
    assert call_tool(s, "get_notes", query)["count"] == 2  # rollback did not round onsets to 1e-6
    call_tool(s, "set_tempo", {"bpm": 90})
    call_tool(s, "undo", {})
    call_tool(s, "redo", {})
    call_tool(s, "undo", {})
    assert call_tool(s, "get_notes", query)["count"] == 2
    assert s.project.tracks[0].notes[1].onset == 4 + 1 / 3


def test_lyric_language_also_sets_pronunciation(s):
    from synth2.phonemes import resolve_notes

    t = s.project.tracks[0]
    assert t.language == "en"
    call_tool(s, "write_melody", {"score": "@ 4 C4 q la D4 q la E4 q la"})
    call_tool(s, "set_lyrics", {"text": "sakura", "language": "ja"})
    assert [n.attrs.get("language") for n in t.notes] == ["ja"] * 3
    assert [x.phones for x in resolve_notes(t.notes, t.language)] == [["s", "a"], ["k", "u"], ["dx", "a"]]
    call_tool(s, "set_lyrics", {"text": "hello world +"})  # back to the track's language: attr removed
    assert all("language" not in n.attrs for n in t.notes)
    r = call_tool(s, "write_melody", {"language": "ja", "score": "@ 8 C4 q sa D4 q ku E4 q ra"})
    new = [n for n in t.notes if n.id in r["note_ids"]]
    assert [x.phones for x in resolve_notes(new, t.language)] == [["s", "a"], ["k", "u"], ["dx", "a"]]
    call_tool(s, "write_melody", {"score": "@ 12 C4 q hi"})  # same language as the track: no attr
    assert "language" not in t.notes[-1].attrs


def test_voice_param_ranges(s, monkeypatch):
    for vp in ({"formant_scale": 0}, {"formant_scale": -1.2}, {"loudness_db": 1e6}, {"bandwidth_scale": 0},
               {"jitter": 0.5}, {"source_corner": 0}, {"tension": 2}):
        with pytest.raises(ToolError, match="out of range"):
            call_tool(s, "apply_expression", {"voice_params": vp})
        with pytest.raises(ToolError, match="out of range"):
            call_tool(s, "add_track", {"voice_params": vp})
    with pytest.raises(ToolError, match="range_high"):
        call_tool(s, "update_track", {"track": 0, "voice_params": {"range_low": 80, "range_high": 70}})
    assert s.project.tracks[0].voice_params == {} and len(s.project.tracks) == 1
    call_tool(s, "update_track", {"track": 0, "voice_params": {"formant_scale": 1.3, "loudness_db": -6}})
    params = call_tool(s, "list_parameters", {})["voice_params"]
    assert params["formant_scale"]["min"] > 0
    # a loaded project with a bad value is refused at render time instead of writing a silent/NaN file
    call_tool(s, "write_melody", {"score": "@ 1 A4 q la"})
    s.project.tracks[0].voice_params["formant_scale"] = 0.0
    with pytest.raises(ToolError, match="formant_scale"):
        call_tool(s, "render", {})
    # and non-finite audio from the engine is reported, not saved
    import numpy as np
    import synth2.engine as engine

    s.project.tracks[0].voice_params["formant_scale"] = 1.2
    bad = SimpleNamespace(audio=np.full((100, 2), np.nan, dtype=np.float32), sample_rate=44100, report={},
                          save=lambda *a, **k: pytest.fail("NaN render must not be saved"))
    monkeypatch.setattr(engine, "render_project", lambda *a, **k: bad)
    with pytest.raises(ToolError, match="NaN"):
        call_tool(s, "render", {})
    assert s.last_render is None


def test_diatonic_transpose_keeps_chromatic_lines(s):
    src = [Note(i, 1, note_to_midi(p)) for i, p in enumerate(["E4", "Eb4", "D4", "F4", "F#4", "G4"])]
    assert names(composer.transpose(src, steps=2, key="C major")) == ["G4", "F#4", "F4", "A4", "A#4", "B4"]
    assert names(composer.transpose(src, steps=1, key="C major"))[3:] == ["G4", "G#4", "A4"]
    call_tool(s, "write_melody", {"score": "@ 4 E4 q a Eb4 q b D4 q c F4 q d F#4 q e G4 q f"})
    call_tool(s, "transpose", {"steps": 2})
    assert names(s.project.tracks[0].notes) == ["G4", "F#4", "F4", "A4", "A#4", "B4"]


def test_add_harmony_into_track_fits_the_target_voice(s):
    call_tool(s, "write_melody", {"score": "@ 4 C5 q a D5 q b E5 q c C5 q d"})
    call_tool(s, "add_track", {"name": "Bass", "voice": "basso"})
    r = call_tool(s, "add_harmony", {"steps": -2, "into_track": "Bass"})
    bass = s.project.tracks[1]
    assert bass.voice == "basso" and r["warnings"] == [] and r["octave_shifted_notes"] == 4
    assert names(bass.notes) == ["A3", "B3", "C4", "A3"]


def test_add_harmony_octave_is_not_undone_by_range_fitting(s):
    call_tool(s, "write_melody", {"score": "@ 4 C5 q a D5 q b E5 q c C5 q d"})
    call_tool(s, "add_harmony", {"steps": 0, "octave": 1})
    assert names(s.project.tracks[-1].notes) == ["C6", "D6", "E6", "C6"]
    call_tool(s, "add_harmony", {"steps": 2, "octave": -1})
    assert names(s.project.tracks[-1].notes) == ["E4", "F4", "G4", "E4"]
    src = [Note(i, 1, note_to_midi(p)) for i, p in enumerate(["C5", "E5"])]
    assert names(composer.harmonize(src, 2, "C major", voice_range=(55, 86), octave=1)) == ["E6", "G6"]
    # a harmony the range fitting folds back onto the melody is refused, not silently doubled
    call_tool(s, "write_melody", {"track": 0, "score": "@ 12 E5 q a G5 q b A5 q c"})
    n_tracks = len(s.project.tracks)
    with pytest.raises(ToolError, match="identical to the source"):
        call_tool(s, "add_harmony", {"steps": 7, "start_beat": 12})
    assert len(s.project.tracks) == n_tracks


def test_set_lyrics_breaths_and_empty_text(s):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q a D4 q b E4 q c F4 q d"})
    r = call_tool(s, "set_lyrics", {"text": "hello br world"})
    assert [a["lyric"] for a in r["assigned"]] == ["hello", "+", "br", "world"]
    assert r["notes"] == 4 and "warning" not in r and "info" not in r
    for text in ("", "!!!", "  "):
        with pytest.raises(ToolError, match="no singable syllables"):
            call_tool(s, "set_lyrics", {"text": text})
    assert [n.lyric for n in s.project.tracks[0].notes] == ["hello", "+", "br", "world"]
    with pytest.raises(ToolError, match="breaths"):
        call_tool(s, "set_lyrics", {"text": "oh", "start_beat": 6, "end_beat": 7})
    r = call_tool(s, "set_lyrics", {"text": "oh", "start_beat": 6, "end_beat": 7, "keep_breaths": False})
    assert r["assigned"][0]["lyric"] == "oh"


def test_duplicate_track_rejects_negative_onsets_and_edit_checks_only_changed_onsets(s):
    call_tool(s, "write_melody", {"score": "@ 0.25 C4 q a"})
    with pytest.raises(ToolError, match="before beat 0"):
        call_tool(s, "duplicate_track", {"track": 0, "delay_beats": -0.5})
    assert len(s.project.tracks) == 1
    r = call_tool(s, "duplicate_track", {"track": 0, "delay_beats": -0.25})
    assert r["track"]["start_beat"] == 0.0
    # a note that already sits before beat 0 (e.g. from an imported file) can still be edited
    s.project.tracks[1].notes[0].onset = -0.25
    nid = s.project.tracks[1].notes[0].id
    call_tool(s, "edit_notes", {"track": 1, "edits": [{"id": nid, "lyric": "oh"}]})
    assert s.project.tracks[1].notes[0].lyric == "oh"
    with pytest.raises(ToolError, match="negative"):
        call_tool(s, "edit_notes", {"track": 1, "edits": [{"id": nid, "move_by": -0.5}]})


def test_quantize_selection_and_half_grid_rounding(s):
    call_tool(s, "add_notes", {"notes": [{"pitch": "C4", "onset": 0, "duration": 2},
                                         {"pitch": "D4", "onset": 1, "duration": 1},
                                         {"pitch": "E4", "onset": 8.1, "duration": 1}]})
    call_tool(s, "quantize", {"grid": 0.5, "start_beat": 8, "end_beat": 12})
    assert [(n.onset, n.duration) for n in s.project.tracks[0].notes] == [(0.0, 2.0), (1.0, 1.0), (8.0, 1.0)]
    q = composer.quantize([Note(0.25 + 0.5 * i, 0.5, 60) for i in range(6)], 0.5)
    assert [(n.onset, n.duration) for n in q] == [(0.5 + 0.5 * i, 0.5) for i in range(6)]


def test_lyric_tokens_hyphenation_and_overflow():
    from synth2.phonemes import resolve_notes

    assert composer.lyric_tokens("beau-ti-ful well-known") == ["beautiful", "+", "+", "well", "known"]
    notes = composer.fit_lyrics([Note(i, 1, 60) for i in range(3)], "beau-ti-ful")
    assert [x.phones for x in resolve_notes(notes)] == [["b", "y", "uw"], ["t", "ax"], ["f", "ax", "l"]]
    two = [Note(0, 1, 60), Note(1, 1, 60)]
    out = composer.fit_lyrics(two, "oh my [s t aa r] shine")
    assert out[1].lyric.startswith("[") and "s t aa r" in out[1].lyric and "sh" in out[1].lyric
    assert [n.lyric for n in composer.fit_lyrics(two, "oh my br shine")] == ["oh", "my shine"]
    assert [n.lyric for n in composer.fit_lyrics(two, "oh br br")] == ["oh", "br"]


def test_hyphenated_compound_words_keep_each_parts_syllables():
    from synth2.ai.composer import lyric_tokens

    assert lyric_tokens("never-ending") == ["never", "+", "ending", "+"]
    assert lyric_tokens("beau-ti-ful") == ["beautiful", "+", "+"]


# ================================================================== engines, roles, credits

@pytest.fixture
def auto_engine(monkeypatch):
    """No engine or quality pinned by the environment (the suite pins the formant engine)."""
    monkeypatch.delenv("SYNTH2_ENGINE", raising=False)
    monkeypatch.delenv("SYNTH2_QUALITY", raising=False)


def _twinkle(s, **track):
    call_tool(s, "write_melody", {"score": "@ 4 C4 q twinkle C4 q + G4 q twinkle G4 q + A4 h star"})
    if track:
        call_tool(s, "update_track", {"track": 0, **track})


def test_list_engines_without_models(auto_engine, s, monkeypatch, tmp_path):
    base = fakes.empty_models(monkeypatch, tmp_path)
    out = call_tool(s, "list_engines")
    formant, neural = out["engines"]
    assert (formant["name"], formant["available"]) == ("formant", True)
    assert (neural["name"], neural["available"]) == ("neural", False)
    assert str(base) in neural["reason"] and "synth2 models install" in neural["install_hint"]
    assert out["default_engine"] == "formant" and out["install_hint"] == neural["install_hint"]
    assert out["qualities"] == ["draft", "normal", "high"] and set(out["quality_notes"]) == set(out["qualities"])
    assert out["neural"]["banks"] == [] and out["credits"] == [] and out["licence_notes"] == []
    assert set(out["track_roles"]) == {"lead", "double", "harmony", "answer", "adlib", "crowd"}
    assert out["project"] == {"engine": "auto", "tracks": [{"track": "Lead", "asks_for": "auto", "engine": "formant"}]}
    json.dumps(out)


def test_list_engines_with_a_voicebank(auto_engine, s, monkeypatch, tmp_path):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    out = call_tool(s, "list_engines")
    neural = out["engines"][1]
    assert neural["available"] is True and "install_hint" not in neural and out["install_hint"] == ""
    assert neural["voicebanks"] == ["fakebank"] and neural["default_voicebank"] == "fakebank"
    assert out["default_engine"] == "neural"
    assert out["credits"] == [fakes.BANK_CREDIT, fakes.VOCODER_CREDIT]
    assert out["licence_notes"] == ["Fakevoc (vocoder): CC BY-NC 4.0 - commercial use is not allowed"]
    assert out["neural"]["commercial_use"] is False
    bank = out["neural"]["banks"][0]
    assert bank["terms"] == ["Credit the voicebank."] and bank["languages"] == ["en", "ja"]
    monkeypatch.setenv("SYNTH2_ENGINE", "neural:missing")          # a default that cannot run is said, not raised
    out = call_tool(s, "list_engines")
    assert out["default_engine"] is None and "voicebank 'missing' is not installed" in out["default_engine_problem"]


def test_list_voices_says_what_each_voice_becomes_on_the_neural_engine(auto_engine, s, monkeypatch, tmp_path):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    voices = {v["id"]: v for v in call_tool(s, "list_voices")["voices"]}
    assert voices["aurora"]["neural"]["voicebank"] == "fakebank" and voices["aurora"]["neural"]["available"]
    assert voices["basso"]["neural"]["tone"] == 12 and voices["basso"]["neural"]["gender"] < 0
    assert "voicebank fakebank" in voices["pixel"]["neural"]["summary"]
    assert voices["aurora"]["range"] and voices["aurora"]["default_style"] == "pop"     # the old fields stay


def test_project_and_track_engine_fields(auto_engine, s, monkeypatch, tmp_path):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    out = call_tool(s, "new_project", {"title": "E", "engine": " Neural "})
    assert out["engine"] == "neural" and s.project.engine == "neural"
    assert out["tracks"][0]["sung_by"] == "neural" and out["tracks"][0]["role"] == ""
    assert call_tool(s, "update_project", {"engine": "auto"})["engine"] == "auto" and s.project.engine == ""
    assert call_tool(s, "update_project", {"engine": "neural:fakebank"})["engine"] == "neural:fakebank"
    tr = call_tool(s, "add_track", {"name": "Robot", "voice": "nova", "engine": "formant", "role": "answer"})["track"]
    assert (tr["engine"], tr["role"]) == ("formant", "answer")
    tr = call_tool(s, "update_track", {"track": "Robot", "engine": "auto", "role": ""})["track"]
    assert (tr["engine"], tr["role"]) == ("auto", "") and s.project.tracks[1].engine == ""
    summary = call_tool(s, "get_project")
    assert summary["engine"] == "neural:fakebank"
    assert [t["sung_by"] for t in summary["tracks"]] == ["neural:fakebank", "neural:fakebank"]
    call_tool(s, "undo")
    assert s.project.tracks[1].engine == "formant" and s.project.tracks[1].role == "answer"
    # the score route: header line, track options and dotted voice parameters
    out = call_tool(s, "new_project", {"score": "engine: formant\ntrack: Lead voice=aurora role=lead\nr q C4 q la\n"
                                                "track: Ad voice=aurora role=adlib engine=neural "
                                                "voice.neural.dial=1\nr q C4 q hey"})
    assert out["engine"] == "formant" and [t["role"] for t in out["tracks"]] == ["lead", "adlib"]
    assert [t["sung_by"] for t in out["tracks"]] == ["formant", "neural"]
    assert s.project.tracks[1].voice_params == {"neural.dial": 1.0}
    for bad in ("vocaloid", "neural:", "formant:x"):
        with pytest.raises(ToolError, match="unknown engine"):
            call_tool(s, "update_project", {"engine": bad})
        with pytest.raises(ToolError, match="unknown engine"):
            call_tool(s, "update_track", {"track": 0, "engine": bad})
    assert s.project.engine == "formant"                         # a rejected call changes nothing


def test_an_engine_that_cannot_run_is_a_warning_when_set_and_an_error_when_rendered(auto_engine, s, monkeypatch,
                                                                                    tmp_path):
    fakes.empty_models(monkeypatch, tmp_path)
    _twinkle(s)
    out = call_tool(s, "update_project", {"engine": "neural"})
    assert len(out["warnings"]) == 1 and "synth2 models install" in out["warnings"][0]
    assert call_tool(s, "get_project")["tracks"][0]["sung_by"] is None
    check = call_tool(s, "check_project")
    assert check["ok"] is False and check["engines"] == [{"track": "Lead", "engine": None}]
    assert any(i["severity"] == "error" and "not available" in i["message"] for i in check["issues"])
    with pytest.raises(ToolError, match="synth2 models install") as exc:
        call_tool(s, "render", {"path": str(tmp_path / "x.wav")})
    assert "list_engines" in str(exc.value) and not (tmp_path / "x.wav").exists()
    with pytest.raises(ToolError, match="not available"):
        call_tool(s, "render", {"path": str(tmp_path / "x.wav"), "engine": "neural:nobank"})
    with pytest.raises(ToolError, match="unknown engine 'nueral'"):
        call_tool(s, "render", {"path": str(tmp_path / "x.wav"), "engine": "nueral"})
    with pytest.raises(ToolError, match="quality"):
        call_tool(s, "render", {"path": str(tmp_path / "x.wav"), "quality": "ultra"})
    out = call_tool(s, "render", {"path": str(tmp_path / "x.wav"), "engine": "formant"})   # formant still works
    assert out["engines"] == ["formant"] and out["credits"] == [] and (tmp_path / "x.wav").exists()
    out = call_tool(s, "update_track", {"track": 0, "engine": "formant"})                   # and per track
    assert "warnings" not in out and call_tool(s, "check_project")["engines"] == [{"track": "Lead",
                                                                                   "engine": "formant"}]


def test_render_returns_engines_quality_credits_and_licence_notes(auto_engine, s, monkeypatch, tmp_path):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    _twinkle(s)
    out = call_tool(s, "render", {"path": str(tmp_path / "n.wav"), "quality": "draft"})
    assert out["engines"] == ["neural"] and out["voicebanks"] == ["fakebank"] and out["quality"] == "draft"
    assert out["credits"] == [fakes.BANK_CREDIT, fakes.VOCODER_CREDIT]
    assert out["licence_notes"] == ["Fakevoc (vocoder): CC BY-NC 4.0 - commercial use is not allowed"]
    assert fakes.CALLS == [("Lead", "draft")]
    track = out["report"]["tracks"][0]
    assert track["engine"] == "neural" and "alignment" not in track["neural"] and "phonemes" not in track
    out = call_tool(s, "render", {"path": str(tmp_path / "n.wav"), "include_phonemes": True})
    assert out["quality"] == "normal" and "alignment" in out["report"]["tracks"][0]["neural"]
    out = call_tool(s, "render", {"path": str(tmp_path / "f.wav"), "engine": "formant"})
    assert out["engines"] == ["formant"] and out["voicebanks"] == [] and out["credits"] == []
    out = call_tool(s, "render", {"path": str(tmp_path / "n.wav"), "engine": "neural:fakebank", "quality": "high"})
    assert out["report"]["tracks"][0]["bank"] == "fakebank" and fakes.CALLS[-1] == ("Lead", "high")
    json.dumps(out)


def test_the_state_of_the_last_render_includes_engine_and_quality(auto_engine, s, monkeypatch, tmp_path):
    from synth2.ai.tools import _fingerprint

    fakes.install_fake_neural(monkeypatch, tmp_path)
    _twinkle(s)
    p = s.project
    keys = {_fingerprint(p), _fingerprint(p, "formant"), _fingerprint(p, None, "draft"),
            _fingerprint(p, "neural:fakebank", "high")}
    assert len(keys) == 4
    assert _fingerprint(p, "neural") == _fingerprint(p)             # auto resolves to the neural engine here
    call_tool(s, "render", {"path": str(tmp_path / "f.wav"), "engine": "formant", "quality": "draft"})
    assert s.last_render_options == ("formant", "draft")
    stale = "changed since this render"
    assert not [i for i in call_tool(s, "analyze_render", {"include_pitch": False})["issues"] if stale in i]
    call_tool(s, "update_project", {"engine": "formant"})           # the project changed
    assert [i for i in call_tool(s, "analyze_render", {"include_pitch": False})["issues"] if stale in i]
    call_tool(s, "undo")
    call_tool(s, "render", {"path": str(tmp_path / "n.wav")})
    assert not [i for i in call_tool(s, "analyze_render", {"include_pitch": False})["issues"] if stale in i]
    fakes.empty_models(monkeypatch, tmp_path)                       # the voicebank went away: auto is formant now
    monkeypatch.setattr("synth2.neural.available", lambda bank=None: (False, "no voicebank is installed"))
    assert [i for i in call_tool(s, "analyze_render", {"include_pitch": False})["issues"] if stale in i]


def test_render_reports_sampled_sound_credits_and_fallbacks(s, monkeypatch, tmp_path):
    fakes.install_fake_sound(monkeypatch, tmp_path, share_alike=True, commercial_use_statement="Keep it open.")
    _twinkle(s)
    call_tool(s, "set_backing", {"arrangement": fakes.backing()})
    out = call_tool(s, "render", {"path": str(tmp_path / "b.wav")})
    assert out["credits"] == [fakes.SOUND_CREDIT]
    assert out["licence_notes"] == ['Tiny Bass (samples): CC BY 4.0 - share-alike (the publisher says: "Keep it open.")']
    assert not [w for w in out["report"]["warnings"] if "plays instead" in w]
    assert call_tool(s, "check_project")["warnings"] == 0
    call_tool(s, "set_backing", {"arrangement": fakes.backing("no-such-set")})
    out = call_tool(s, "render", {"path": str(tmp_path / "b.wav")})
    fallback = [w for w in out["report"]["warnings"] if "plays instead" in w]
    assert len(fallback) == 1 and "synth2 sounds install no-such-set" in fallback[0] and out["credits"] == []
    check = call_tool(s, "check_project")
    assert [i for i in check["issues"] if i["severity"] == "warning" and "backing band: bass" in i["message"]]
    # vocals alone (a track subset) play no band: no sound credits, no fallback warning
    call_tool(s, "set_backing", {"arrangement": fakes.backing()})
    out = call_tool(s, "render", {"path": str(tmp_path / "v.wav"), "tracks": [0]})
    assert out["credits"] == [] and out["licence_notes"] == []


def test_roles_and_neural_voice_parameters(auto_engine, s, monkeypatch, tmp_path):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    _twinkle(s, role="Lead")
    assert s.project.tracks[0].role == "lead"
    tr = call_tool(s, "add_track", {"name": "Shouts", "role": "ad-lib",
                                    "voice_params": {"neural.dial": 1, "neural.breaths": 0, "neural.fx.robot": 0.5,
                                                     "neural.adlib.bend.hey": -2, "breathiness": 0.2}})["track"]
    assert tr["role"] == "adlib"                                    # stored in the engine's spelling
    assert tr["voice_params"] == {"neural.dial": 1.0, "neural.breaths": 0.0, "neural.fx.robot": 0.5,
                                  "neural.adlib.bend.hey": -2.0, "breathiness": 0.2}
    tr = call_tool(s, "update_track", {"track": "Shouts", "voice_params": {"neural.dial": None}})["track"]
    assert "neural.dial" not in tr["voice_params"]
    out = call_tool(s, "apply_expression", {"track": "Shouts", "voice_params": {"neural.gender": 0.3}})
    assert out["voice_params"]["neural.gender"] == 0.3
    for params, text in (({"neural.dial": 5}, "out of range 0..2"), ({"neural.gender": "high"}, "must be a number"),
                         ({"neural.fx.alien": 2}, "out of range 0..1"), ({"neural.": 1}, "not a neural parameter"),
                         ({"neural.a b": 1}, "not a neural parameter"), ({"nueral.dial": 1}, "unknown voice parameter")):
        with pytest.raises(ToolError, match=text):
            call_tool(s, "update_track", {"track": "Shouts", "voice_params": params})
    for bad in ("singer", "lead:Other", "double:"):
        if bad == "double:":
            assert call_tool(s, "update_track", {"track": "Shouts", "role": bad})["track"]["role"] == "double"
            continue
        with pytest.raises(ToolError, match="role"):
            call_tool(s, "update_track", {"track": "Shouts", "role": bad})
    # doubles: made from a lead of the same voice, or from the track they name
    out = call_tool(s, "duplicate_track", {"track": "Lead", "name": "Dbl", "role": "double"})
    assert out["track"]["role"] == "double" and out["warnings"] == []
    out = call_tool(s, "update_track", {"track": "Dbl", "role": "double:Nobody"})
    assert "no other track named 'Nobody'" in out["warnings"][0]
    out = call_tool(s, "update_track", {"track": "Dbl", "role": "double:lead"})
    assert "warnings" not in out and s.project.tracks[-1].role == "double:lead"
    call_tool(s, "update_track", {"track": "Lead", "role": ""})
    out = call_tool(s, "update_track", {"track": "Dbl", "role": "double"})
    assert "no track with role 'lead'" in out["warnings"][0]
    check = call_tool(s, "check_project", {"track": "Dbl"})
    assert [i for i in check["issues"] if "no track with role 'lead'" in i["message"]]
    # harmonies get their role and the source's engine
    call_tool(s, "update_track", {"track": "Lead", "engine": "formant"})
    h = call_tool(s, "add_harmony", {"source_track": "Lead", "steps": 2})["track"]
    assert (h["role"], h["engine"]) == ("harmony", "formant")
    h = call_tool(s, "add_harmony", {"source_track": "Lead", "steps": -2, "role": ""})["track"]
    assert h["role"] == ""
    # the roles survive a save and a load in both native formats
    for name in ("r.s2p.json", "r.s2s"):
        call_tool(s, "save_project", {"path": str(tmp_path / name)})
        loaded = Session.open(tmp_path / name).project
        assert [t.role for t in loaded.tracks] == [t.role for t in s.project.tracks], name
        assert loaded.tracks[1].voice_params["neural.fx.robot"] == 0.5


def test_check_project_flags_lines_too_fast_for_the_neural_engine(auto_engine, s, monkeypatch, tmp_path):
    fakes.install_fake_neural(monkeypatch, tmp_path)
    call_tool(s, "set_tempo", {"bpm": 120})
    fast = " ".join("C4 s la" for _ in range(16))                   # sixteenths at 120 bpm: 8 syllables a second
    slow = " ".join("C4 q la" for _ in range(8))                    # quarters: 2 a second
    call_tool(s, "write_melody", {"score": f"@ 4 {slow} r h {fast}"})
    check = call_tool(s, "check_project")
    hits = [i for i in check["issues"] if "syllables per second" in i["message"]]
    assert len(hits) == 1 and hits[0]["severity"] == "warning" and "up to 8.0/s" in hits[0]["message"]
    assert "starts at beat 14" in hits[0]["message"]
    call_tool(s, "update_track", {"track": 0, "engine": "formant"})  # the formant engine is not judged by it
    assert not [i for i in call_tool(s, "check_project")["issues"] if "syllables per second" in i["message"]]
    call_tool(s, "update_track", {"track": 0, "engine": "auto"})
    call_tool(s, "set_tempo", {"bpm": 50})                           # the same notes, slowly: 3.3 a second
    assert not [i for i in call_tool(s, "check_project")["issues"] if "syllables per second" in i["message"]]


def test_tool_text_teaches_engines_roles_and_sounds():
    """What a model has to learn from the tool definitions alone."""
    from synth2.ai import mcp_server
    from synth2.ai.director import SYSTEM_PROMPT

    by_name = {t["name"]: t for t in TOOLS}
    assert "list_engines" in by_name and "list_sounds" in by_name
    add = by_name["add_track"]
    for text in ("own track with role 'adlib'", "one track = one singer doing one kind of thing",
                 "3.5 syllables per second", "neural.dial", "neural.breaths"):
        assert text in add["description"], text
    for name in ("add_track", "update_track"):
        props = by_name[name]["input_schema"]["properties"]
        assert {"engine", "role", "voice_params"} <= set(props)
        assert "adlib" in props["role"]["description"] and "neural:<voicebank id>" in props["engine"]["description"]
    for name in ("new_project", "update_project", "render"):
        assert "engine" in by_name[name]["input_schema"]["properties"], name
    render = by_name["render"]
    assert render["input_schema"]["properties"]["quality"]["enum"] == ["draft", "normal", "high"]
    for text in ("credits", "licence_notes", "fell back to the synth"):
        assert text in render["description"], text
    assert "How to pick" in by_name["list_sounds"]["description"] and "gm" in by_name["list_sounds"]["description"]
    assert "sound" in by_name["set_backing"]["description"]
    params = call_tool(Session(), "list_parameters")
    assert "neural.dial" in params["neural_voice_params"] and "adlib" in params["track_roles"]
    for text in (SYSTEM_PROMPT, mcp_server.INSTRUCTIONS):
        for word in ("list_engines", "neural", "formant", "adlib", "3.5 syllables", "credits", "licence_notes",
                     "list_sounds", "quality"):
            assert word in text, word


def test_mcp_and_director_pass_engines_and_credits_through(auto_engine, monkeypatch, tmp_path):
    from synth2.ai.director import _state_block
    from synth2.ai.mcp_server import MCPServer

    fakes.install_fake_neural(monkeypatch, tmp_path)
    server = MCPServer()
    init = server.handle({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    assert "list_engines" in init["result"]["instructions"]
    listed = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})["result"]["tools"]
    assert "list_engines" in [t["name"] for t in listed]

    def call(name, args):
        res = server.handle({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                             "params": {"name": name, "arguments": args}})["result"]
        return res.get("isError", False), json.loads(res["content"][0]["text"]) if not res.get("isError") \
            else res["content"][0]["text"]

    assert call("list_engines", {})[1]["default_engine"] == "neural"
    call("write_melody", {"score": "@ 4 C4 q la D4 q la"})
    err, out = call("render", {"path": str(tmp_path / "m.wav"), "quality": "draft"})
    assert not err and out["engines"] == ["neural"] and out["credits"][0] == fakes.BANK_CREDIT
    assert out["licence_notes"] and fakes.CALLS[-1] == ("Lead", "draft")
    err, text = call("render", {"path": str(tmp_path / "m.wav"), "engine": "neural:nope"})
    assert err and "voicebank 'nope' is not installed" in text
    state = _state_block(server.session)
    assert '"engine": "auto"' in state and '"sung_by": "neural"' in state and '"role": ""' in state
