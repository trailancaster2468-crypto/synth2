"""Tests for the AI layer: composer helpers, tool registry, MCP server and director."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

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
