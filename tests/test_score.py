"""Tests for the compact text score notation (synth2.score)."""
from __future__ import annotations

import pytest

from synth2.project import Curve, MasterSettings, Note, Project
from synth2.score import (
    ScoreError, duration_token, notes_to_score, parse_notes, parse_score, parse_score_ex, parse_snippet,
    project_to_score,
)
from synth2.timing import parse_duration

PLAN_EXAMPLE = """\
# comments start with '#'
title: Twinkle
tempo: 112            # bpm (also: "tempo: 120 @ 16" = change at beat 16)
time: 4/4
key: C major

track: Lead voice=aurora style=pop lang=en gain=0 pan=0
C4 q twinkle  C4 q +  G4 q twinkle  G4 q +  |  A4 q little  A4 q +  G4 h star
F4 q how   F4 q I  E4 q won  E4 q der | D4 q what D4 q you C4 h are
r q            # rest
E4 e. la  F4 s -   # dotted eighth, then a melisma sixteenth
G4 h+q oh{vibrato_depth=0.5,tension=0.3}   # tied duration, per-note attrs
A4 q [s t aa r]    # explicit phonemes

track: Harmony voice=atlas style=pop
E4 q la
"""


def _strip_ids(d: dict) -> dict:
    for t in d["tracks"]:
        t.pop("id", None)
        for n in t["notes"]:
            n.pop("id", None)
    return d


def test_parse_plan_example():
    p, warnings = parse_score_ex(PLAN_EXAMPLE)
    assert warnings == []
    assert p.title == "Twinkle"
    assert p.bpm == 112
    assert p.time_signature == (4, 4)
    assert p.key == "C major"
    assert [t.name for t in p.tracks] == ["Lead", "Harmony"]
    lead = p.tracks[0]
    assert lead.voice == "aurora" and lead.style == "pop" and lead.language == "en"
    notes = lead.notes
    assert len(notes) == 18
    assert notes[0].note_name == "C4" and notes[0].lyric == "twinkle" and notes[0].onset == 0
    assert notes[1].lyric == "+"
    assert notes[6].lyric == "star" and notes[6].duration == 2.0 and notes[6].onset == 6.0
    # rest of one beat after "are" (ends at beat 16)
    la = notes[14]
    assert la.lyric == "la" and la.onset == 17.0 and la.duration == 0.75
    assert notes[15].lyric == "-" and notes[15].duration == 0.25
    oh = notes[16]
    assert oh.duration == 3.0 and oh.attrs == {"vibrato_depth": 0.5, "tension": 0.3}
    assert notes[17].lyric == "[s t aa r]" and notes[17].onset == 21.0
    harm = p.tracks[1]
    assert harm.voice == "atlas" and harm.style == "pop" and harm.notes[0].onset == 0


def test_duration_tokens():
    assert duration_token(1) == "q"
    assert duration_token(1.5) == "q."
    assert duration_token(1 / 3) == "e3"
    assert duration_token(3) == "h."
    assert duration_token(5) == "w+q"
    assert duration_token(2.5) == "h+e"
    assert duration_token(0.3) == "0.3"
    for x in [0.125, 0.25, 0.375, 0.5, 0.75, 1, 1.25, 1.75, 2, 2.25, 3.5, 4, 4.5, 5, 6, 7, 8, 1 / 3, 2 / 3, 4 / 3,
              0.3, 0.1, 13.7, 1 / 6, 1 / 7]:
        assert abs(parse_duration(duration_token(x)) - x) < 1e-9


def _complex_project() -> Project:
    p = Project(title='Song: "#1" = hit', key="F# minor", sample_rate=48000)
    p.tempo.set(0, 96)
    p.tempo.set(16, 120.5)
    p.time_signature = (3, 4)
    p.master = MasterSettings(gain_db=-2.0, reverb_room=0.3, reverb_wet=0.8, normalize=False, limiter=True)
    a = p.add_track("Lead #1", voice="seraph", language="en")
    a.style = "ballad"
    a.gain_db, a.pan, a.reverb_send, a.mute = -3.5, -0.25, 0.4, True
    a.voice_params = {"breathiness": 0.3, "style.vibrato_depth": 1.4}
    lyrics = ["hello", "+", "-", "br", "[s t aa r]", "", "hello world", "60", "r", "C4", "{weird}", "a#b", "|",
              "x=y", "さ", "@home", "end"]
    pos = 1.0
    for i, lyr in enumerate(lyrics):
        dur = [1, 0.5, 1 / 3, 0.3, 2.75, 1.5][i % 6]
        n = a.add_note(pos, dur, 60 + i, lyr)
        if i == 0:
            n.phonemes = "hh ax l ow"
            n.attrs = {"vibrato_depth": 0.4, "language": "en", "velocity": 1.2}
        if i == 3:
            n.attrs = {"detune": -12.5, "flag": True, "none_attr": None, "text": "two words"}
        pos += dur + [0, 0.25, 0, 7.0, 0, 0.1][i % 6]
    a.add_note(3.0, 1.0, 50, "overlap")  # overlapping note -> needs '@'
    a.curves["loudness"] = Curve([(b * 0.5, -b % 7 - 3) for b in range(40)], "linear")
    a.curves["tension"] = Curve([(0.0, 0.1), (4.0, -0.2)], "step")
    b = p.add_track("Harmony", voice="atlas", language="ja")
    b.style = "rock"
    b.solo = True
    b.seed = 1234
    b.add_note(0.0, 1.0, 55, "か")
    b.add_note(1.0, 1.0, 57, "ら")
    p.add_track("Empty", voice="ember")
    return p


def test_round_trip_complex_project():
    p = _complex_project()
    text = project_to_score(p)
    p2 = parse_score(text)
    assert _strip_ids(p2.to_dict()) == _strip_ids(p.to_dict())
    # serialisation is stable
    assert project_to_score(p2) == text


def test_round_trip_plan_example():
    p = parse_score(PLAN_EXAMPLE)
    p2 = parse_score(project_to_score(p))
    assert _strip_ids(p2.to_dict()) == _strip_ids(p.to_dict())


def test_serialised_score_is_readable():
    p = parse_score(PLAN_EXAMPLE)
    text = project_to_score(p)
    assert "track: Lead voice=aurora style=pop lang=en gain=0 pan=0" in text
    assert "G4 h. oh{vibrato_depth=0.5,tension=0.3}" in text
    assert "E4 e. la  F4 s -" in text
    assert "A4 q [s t aa r]" in text
    assert "r q" in text


def test_gaps_use_rests_or_at():
    notes = [Note(0, 1, 60, "a"), Note(1.5, 1, 62, "b"), Note(40, 1, 64, "c")]
    lines = notes_to_score(notes, 0.0, 4.0)
    joined = "\n".join(lines)
    assert "r e" in joined
    assert "@ 40" in joined
    back = parse_notes(joined)
    assert [(n.onset, n.duration, n.pitch, n.lyric) for n in back] == [(0, 1, 60, "a"), (1.5, 1, 62, "b"),
                                                                        (40, 1, 64, "c")]


def test_parse_notes_positions_and_omitted_lyrics():
    notes = parse_notes("C4 q  D4 q  E4 h la | @ 16 F#4 e x  G4 e", start_beat=8)
    assert [(n.onset, n.note_name, n.lyric) for n in notes] == [
        (8, "C4", "la"), (9, "D4", "la"), (10, "E4", "la"), (16, "F#4", "x"), (16.5, "G4", "la")]
    notes = parse_notes("60 1 61 0.5 do 62 3/2")
    assert [(n.pitch, n.duration, n.lyric) for n in notes] == [(60, 1, "la"), (61, 0.5, "do"), (62, 1.5, "la")]


def test_bar_and_attr_forms():
    notes = parse_notes("@ bar 3  C4 q la {velocity=1.5}  D4 q \"two words\"{tension=-0.2}  E4 q star[s t aa r]",
                        language="en")
    assert notes[0].onset == 8 and notes[0].attrs == {"velocity": 1.5}
    assert notes[1].lyric == "two words" and notes[1].attrs == {"tension": -0.2}
    assert notes[2].lyric == "star" and notes[2].phonemes == "s t aa r"


def test_comments_and_sharps():
    notes = parse_notes("F#4 q la # this is a comment\nC#5 q \"#tag\"  # another")
    assert [n.note_name for n in notes] == ["F#4", "C#5"]
    assert notes[1].lyric == "#tag"


def test_snippet_curves_and_tempo():
    sn = parse_snippet("C4 q la D4 q la\ntempo: 90\ncurve: loudness 0:-3 2:0 (linear)\nE4 h la", start_beat=4)
    assert [n.onset for n in sn.notes] == [4, 5, 6]
    assert sn.tempos == [(6.0, 90.0)]
    assert sn.curves["loudness"].points == [(0.0, -3.0), (2.0, 0.0)]
    assert sn.curves["loudness"].interpolation == "linear"
    assert sn.end_beat == 8


def test_tempo_inside_track_and_header_tempo_change():
    p = parse_score("tempo: 100\ntempo: 80 @ 32\ntrack: A\nC4 w la\ntempo: 120\nD4 w la\n")
    assert p.tempo.to_list() == [{"beat": 0.0, "bpm": 100.0}, {"beat": 4.0, "bpm": 120.0},
                                 {"beat": 32.0, "bpm": 80.0}]


def test_notes_without_track_line_create_lead():
    p = parse_score("title: x\nC4 q la D4 q la")
    assert len(p.tracks) == 1 and p.tracks[0].name == "Lead" and len(p.tracks[0].notes) == 2


def test_track_options():
    p = parse_score('track: "My Lead" voice=atlas lang=ja gain=-6 pan=0.5 reverb=0 mute=true solo=1 seed=5 '
                    'voice.breathiness=0.2 style.scoop=0.1\nC4 q か')
    t = p.tracks[0]
    assert t.name == "My Lead" and t.voice == "atlas" and t.style == "rock"  # atlas's default style
    assert (t.language, t.gain_db, t.pan, t.reverb_send, t.mute, t.solo, t.seed) == ("ja", -6, 0.5, 0, True, True, 5)
    assert t.voice_params == {"breathiness": 0.2, "style.scoop": 0.1}


@pytest.mark.parametrize("text, line, token, fragment", [
    ("track: A\nC4 q la\nX9 q la", 3, "X9", "invalid pitch"),
    ("C4 qq la", 1, "qq", "invalid duration"),
    ("C4 q la D4", 1, "D4", "missing its duration"),
    ("C4 q la\nfoo: bar", 2, "foo:", "unknown directive"),
    ("track: A\ncurve: loud 0:1", 2, "loud", "unknown curve"),
    ("track: A colour=red", 1, "colour=red", "unknown track option"),
    ("C4 q [s t xx r]", 1, "[s t xx r]", "unknown phoneme"),
    ('C4 q "open', 1, '"open', "unterminated"),
    ("C4 q [s t", 1, "[s t", "missing closing"),
    ("{velocity=1} C4 q la", 1, "{velocity=1}", "must follow a note"),
    ("r q la", 1, "la", "rests take no lyric"),
    ("130 q la", 1, "130", "MIDI range"),
    ("C4 q la @", 1, "@", "beat number"),
    ("C4 0 la", 1, "0", "invalid duration"),
    ("time: 4/5", 1, "4/5", "power of two"),
    ("tempo: fast", 1, "fast", "number"),
    ("C4 q la{velocity}", 1, None, "key=value"),
])
def test_errors_have_line_and_token(text, line, token, fragment):
    with pytest.raises(ScoreError) as ei:
        parse_score(text)
    err = ei.value
    assert err.line == line
    if token is not None:
        assert err.token == token
    assert fragment in str(err)
    assert f"line {line}" in str(err)


def test_snippet_rejects_headers():
    with pytest.raises(ScoreError, match="not allowed in a note snippet"):
        parse_notes("track: A\nC4 q la")
    with pytest.raises(ScoreError, match="score header"):
        parse_notes("key: C major")


def test_warnings():
    _, w = parse_score_ex("track: A voice=nobody style=weird\nC4 q la D4 q. la | E4 q la{vibrato=1}\n"
                          "curve: loudness 0:50")
    text = "\n".join(w)
    assert "unknown voice 'nobody'" in text
    assert "unknown style 'weird'" in text
    assert "bar line at beat 2.5" in text
    assert "unknown note attribute 'vibrato'" in text
    assert "outside" in text
    _, w = parse_score_ex("track: J lang=ja\nC4 q さくら")
    assert any("morae" in x for x in w)


def test_project_load_s2s(tmp_path):
    path = tmp_path / "song.s2s"
    path.write_text(PLAN_EXAMPLE, encoding="utf-8")
    p = Project.load(path)
    assert p.title == "Twinkle" and len(p.tracks[0].notes) == 18


# --------------------------------------------------------------- regression tests

@pytest.mark.parametrize("text", [
    "C4 q {tension=0.3}D4",                      # D4 used to be swallowed into the attribute value
    "C4 q la {velocity=1.5}{tension=0.2}",
    "track: A\nC4 q la {velocity=1.2}|  D4 q la",
])
def test_standalone_attrs_with_trailing_text_is_an_error(text):
    with pytest.raises(ScoreError, match="unexpected text after {attributes}") as ei:
        parse_score(text)
    assert ei.value.token.startswith("{")


def test_standalone_attrs_still_work():
    notes = parse_notes("C4 q la {velocity=1.5,tension=0.2}  D4 q la")
    assert notes[0].attrs == {"velocity": 1.5, "tension": 0.2} and notes[1].note_name == "D4"


@pytest.mark.parametrize("attrs", ["velocity=loud", "tension=true", "detune=nan", "vibrato_rate=inf",
                                   'velocity="1"', "language=5"])
def test_known_note_attributes_are_type_checked(attrs):
    with pytest.raises(ScoreError, match="attribute") as ei:
        parse_notes(f"C4 q la  D4 q la{{{attrs}}}")
    assert ei.value.line == 1 and ei.value.token == f"{{{attrs}}}"
    with pytest.raises(ScoreError):
        parse_notes(f"C4 q la {{{attrs}}}")


def test_valid_and_unknown_note_attributes_are_kept():
    sn = parse_snippet("C4 q la{velocity=1,language=ja,tension=null,foo=bar}  D4 q la{language=fr}")
    assert sn.notes[0].attrs == {"velocity": 1.0, "language": "ja", "tension": None, "foo": "bar"}
    assert sn.notes[1].attrs == {"language": "fr"}
    text = "\n".join(sn.warnings)
    assert "unknown note attribute 'foo'" in text and "language 'fr' is not supported" in text


def test_invalid_note_fields_are_dropped_on_export_so_the_score_loads():
    p = Project.from_dict({"title": "x", "tracks": [{"name": "A", "notes": [
        {"onset": 0, "duration": 1, "pitch": 60, "lyric": "star", "phonemes": "s t ar",
         "attrs": {"velocity": "loud", "tension": 0.2}},
        {"onset": 1, "duration": 1, "pitch": 62, "lyric": "x", "phonemes": 's]"\u2028{'},
        {"onset": 2, "duration": 1, "pitch": 64, "lyric": "star", "phonemes": "s t aa r"}]}]})
    text = project_to_score(p)
    assert "star[s t ar]" not in text
    assert '# dropped invalid phoneme override "s t ar", attribute velocity="loud" of C4 at beat 0' in text
    notes = parse_score(text).tracks[0].notes
    assert [(n.lyric, n.phonemes, n.attrs) for n in notes] == [
        ("star", None, {"tension": 0.2}), ("x", None, {}), ("star", "s t aa r", {})]


def test_only_real_newlines_split_lines():
    # \f, \v, \x1c-\x1e, \x85, U+2028/9 are whitespace inside a line, not line breaks
    for sep in ["\x0c", "\x0b", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]:
        p = parse_score(f"track: A\nC4 q la   # old:{sep}D4 q la\nE4 q la")
        assert [n.note_name for n in p.tracks[0].notes] == ["C4", "E4"], repr(sep)
        assert [n.onset for n in p.tracks[0].notes] == [0, 1]
    with pytest.raises(ScoreError) as ei:
        parse_score("track: A\nC4 q la # see\u2028note\nX9 q la")
    assert ei.value.line == 3 and ei.value.token == "X9"
    p = parse_score("track: A\r\nC4 q la\rD4 q la\n")
    assert [n.note_name for n in p.tracks[0].notes] == ["C4", "D4"]


def test_line_separators_in_strings_round_trip():
    p = Project(title="x\u2028y")
    t = p.add_track("A\x85B")
    t.add_note(0, 1, 60, "line\u2029sep")
    t.add_note(1, 1, 62, "la").attrs = {"text": "a\u2028b"}
    text = project_to_score(p)
    assert not any(ch in text for ch in "\x85\u2028\u2029")
    p2 = parse_score(text)
    assert _strip_ids(p2.to_dict()) == _strip_ids(p.to_dict())


def test_long_curves_parse_fast_and_merge_duplicates():
    import time

    p = Project()
    t = p.add_track("A")
    t.curves["pitchDelta"] = Curve([(i * 0.125, float(i % 50 - 25)) for i in range(20000)], "linear")
    text = project_to_score(p)
    t0 = time.perf_counter()
    p2 = parse_score(text)
    assert time.perf_counter() - t0 < 2.0  # was ~15 s (quadratic)
    assert p2.tracks[0].curves["pitchDelta"].points == t.curves["pitchDelta"].points
    sn = parse_snippet("curve: loudness 4:1 0:2 4:3\ncurve: loudness 0:5 2:0")
    assert sn.curves["loudness"].points == [(0.0, 5.0), (2.0, 0.0), (4.0, 3.0)]


def test_bad_json_in_track_style_option_is_a_score_error():
    with pytest.raises(ScoreError) as ei:
        parse_score('title: x\ntempo: 100\ntrack: A style.vibrato_depth="\\x"\nC4 q la')
    assert ei.value.line == 3 and ei.value.token == 'style.vibrato_depth="\\x"'
    assert "must be a number" in str(ei.value)


def test_track_subset_keeps_seed():
    p = Project()
    p.add_track("A")
    b = p.add_track("B")  # default seed for index 1
    c = p.add_track("C")
    c.seed = 17           # equals the default for index 0
    assert parse_score(project_to_score(p, [b])).tracks[0].seed == b.seed
    assert parse_score(project_to_score(p, [c])).tracks[0].seed == 17
    assert [t.seed for t in parse_score(project_to_score(p, [b, c])).tracks] == [b.seed, 17]
    assert [t.seed for t in parse_score(project_to_score(p)).tracks] == [t.seed for t in p.tracks]


# ---- engines ------------------------------------------------------------------------------------

ENGINE_SCORE = """\
title: Engines
tempo: 120
engine: Neural: qixuan
track: Lead voice=aurora engine=formant voice.neural.gender=0.2 voice.neural.fx.alien=0.6 voice.breathiness=0.3
C4 q la  D4 q la
track: Bank engine="neural:My Bank"
E4 q la
track: Auto engine=AUTO voice.neural.level_db=-3
G4 q la
"""


def test_engine_header_and_track_attribute():
    p = parse_score(ENGINE_SCORE)
    assert p.engine == "neural:qixuan"                    # stored in its canonical spelling
    lead, bank, auto = p.tracks
    assert lead.engine == "formant" and bank.engine == "neural:My Bank"
    assert auto.engine == ""                               # auto is "not set"
    assert lead.voice_params == {"neural.gender": 0.2, "neural.fx.alien": 0.6, "breathiness": 0.3}
    assert auto.voice_params == {"neural.level_db": -3.0}
    plain = parse_score("title: x\ntempo: 100\ntrack: A\nC4 q la")
    assert plain.engine == "" and plain.tracks[0].engine == ""
    assert parse_score("engine: auto\ntrack: A\nC4 q la").engine == ""
    assert parse_score('engine: "neural:My Bank"\ntrack: A\nC4 q la').engine == "neural:My Bank"
    assert parse_score("engine: neural:My Bank  # the header takes the rest of the line\ntrack: A\nC4 q la"
                       ).engine == "neural:My Bank"


def test_engine_round_trip():
    p = parse_score(ENGINE_SCORE)
    text = project_to_score(p)
    lines = text.splitlines()
    assert "engine: neural:qixuan" in lines
    lead = next(ln for ln in lines if ln.startswith("track: Lead"))
    assert " engine=formant" in lead
    assert " voice.neural.gender=0.2" in lead and " voice.neural.fx.alien=0.6" in lead
    assert ' engine="neural:My Bank"' in next(ln for ln in lines if ln.startswith("track: Bank"))
    assert "engine" not in next(ln for ln in lines if ln.startswith("track: Auto"))
    p2 = parse_score(text)
    assert _strip_ids(p2.to_dict()) == _strip_ids(p.to_dict())
    assert project_to_score(p2) == text
    # an unset engine writes nothing, so older scores are unchanged
    assert "engine" not in project_to_score(parse_score("title: x\ntempo: 100\ntrack: A\nC4 q la"))
    # a header bank id that needs quotes
    p.engine = "neural:My Bank"
    text = project_to_score(p)
    assert 'engine: "neural:My Bank"' in text.splitlines()
    assert parse_score(text).engine == "neural:My Bank"


def test_engine_survives_project_json_and_s2s_files(tmp_path):
    p = parse_score(ENGINE_SCORE)
    path = tmp_path / "song.s2s"
    path.write_text(project_to_score(p), encoding="utf-8")
    q = Project.load(path)
    assert q.engine == "neural:qixuan" and [t.engine for t in q.tracks] == ["formant", "neural:My Bank", ""]
    assert q.tracks[0].voice_params["neural.fx.alien"] == 0.6
    r = Project.from_json(q.to_json())
    assert r.tracks[0].voice_params == p.tracks[0].voice_params and r.tracks[1].engine == "neural:My Bank"


@pytest.mark.parametrize("text, line, token, fragment", [
    ("title: x\nengine: vocaloid\ntrack: A\nC4 q la", 2, "vocaloid", "unknown engine"),
    ("engine: formant:x\ntrack: A\nC4 q la", 1, "formant:x", "unknown engine"),
    ("title: x\ntrack: A engine=neural:\nC4 q la", 2, "engine=neural:", "neural:<voicebank id>"),
    ("track: A engine=diffsinger\nC4 q la", 1, "engine=diffsinger", "unknown engine"),
    ('engine: "neural:x\ntrack: A\nC4 q la', 1, '"neural:x', "quoted string"),
    ("track: A voice.neural.gender=abc\nC4 q la", 1, "abc", "expected a number"),
    ("track: A voice.neural.gender=nan\nC4 q la", 1, "nan", "finite"),
    ("track: A voice.neural=1\nC4 q la", 1, "voice.neural=1", "unknown voice parameter"),
    ("track: A voice.neural..x=1\nC4 q la", 1, "voice.neural..x=1", "unknown voice parameter"),
    ("track: A voice.neuralx.gender=1\nC4 q la", 1, "voice.neuralx.gender=1", "neural.<name>"),
])
def test_engine_errors_have_line_and_token(text, line, token, fragment):
    with pytest.raises(ScoreError) as ei:
        parse_score(text)
    assert ei.value.line == line and ei.value.token == token
    assert fragment in str(ei.value)


def test_snippet_rejects_engine_header():
    with pytest.raises(ScoreError) as ei:
        parse_snippet("engine: neural\nC4 q la")
    assert ei.value.token == "engine:" and "update_project" in str(ei.value)


def test_dotted_voice_params_keep_any_depth_and_sign():
    p = parse_score("track: A voice.neural.fx.alien=1 voice.neural.fx.double=0.25 voice.neural.tone=-0.5 "
                    "voice.neural.breath_db=-12.5\nC4 q la")
    vp = p.tracks[0].voice_params
    assert vp == {"neural.fx.alien": 1.0, "neural.fx.double": 0.25, "neural.tone": -0.5, "neural.breath_db": -12.5}
    again = parse_score(project_to_score(p)).tracks[0].voice_params
    assert again == vp and list(again) == list(vp)
