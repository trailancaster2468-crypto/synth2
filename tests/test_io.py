"""Tests for synth2.io: MIDI, SVP, MusicXML, UST readers/writers and load_any/export_any."""
from __future__ import annotations

import json
import struct
import zipfile

import pytest

from synth2.io import detect_format, export_any, load_any
from synth2.io.midi import dumps_midi, fifths_to_key, key_to_fifths, loads_midi, read_midi, write_midi
from synth2.io.musicxml import loads_musicxml, read_musicxml, write_musicxml
from synth2.io.svp import BLICKS_PER_BEAT, loads_svp, project_to_svp_dict, read_svp, write_svp
from synth2.io.ust import loads_ust, read_ust, write_ust
from synth2.project import Curve, Project
from synth2.timing import TempoMap, TempoPoint


# ---------------------------------------------------------------------------
# fixtures / helpers
# ---------------------------------------------------------------------------
def make_project() -> Project:
    p = Project.new("Round Trip", bpm=96)
    p.tempo.set(9.0, 132.0)
    p.time_signature = (3, 4)
    p.key = "Bb major"
    lead = p.tracks[0]
    for onset, dur, pitch, lyric in [
        (0, 1, "C4", "twinkle"), (1, 1, "C4", "+"), (2, 1, "G4", "star"),
        (3, 0.5, "A4", "-"), (3.5, 0.5, "B4", "-"), (4, 1, "G4", "how"),
        (5.5, 2.5, "F#4", "oh"), (8, 0.5, "E4", "beautiful"), (8.5, 0.5, "D4", "+"), (9, 1, "C4", "+"),
    ]:
        lead.add_note(onset, dur, pitch, lyric)
    lead.notes[5].attrs["velocity"] = 1.25
    lead.notes[6].attrs["velocity"] = 0.8
    harm = p.add_track("Harmony", language="ja", gain_db=-3.0, pan=0.25)
    for onset, dur, pitch, lyric in [(0, 2, 64, "さ"), (2, 1, 65, "く"), (4, 4, 67, "ら"), (8, 1, 62, "-")]:
        harm.add_note(onset, dur, pitch, lyric)
    return p


def notes_of(track, attrs: tuple[str, ...] = ()):
    out = []
    for n in sorted(track.notes, key=lambda n: n.onset):
        row = (round(n.onset, 4), round(n.duration, 4), n.pitch, n.lyric)
        if attrs:
            row += tuple(None if n.attrs.get(a) is None else round(float(n.attrs[a]), 4) if not isinstance(n.attrs[a], str) else n.attrs[a] for a in attrs)
        out.append(row)
    return out


def tempo_of(p: Project):
    return [(round(t["beat"], 4), round(t["bpm"], 3)) for t in p.tempo.to_list()]


# ---------------------------------------------------------------------------
# JSON + dispatch
# ---------------------------------------------------------------------------
def test_json_round_trip_via_export_any(tmp_path):
    p = make_project()
    path = export_any(p, tmp_path / "song", fmt="json")
    assert path.name == "song.s2p.json"
    assert export_any(p, tmp_path / "packed", fmt="mxl").name == "packed.mxl"
    assert zipfile.is_zipfile(tmp_path / "packed.mxl")
    assert export_any(p, tmp_path / "song", fmt="mid").name == "song.mid"
    q = load_any(path)
    assert q.to_dict() == p.to_dict()


def test_detect_format_and_unknown_extension(tmp_path):
    p = make_project()
    assert detect_format("a.mid") == "midi"
    assert detect_format("a.MXL") == "musicxml"
    assert detect_format("a.ust") == "ust"
    sniff = tmp_path / "noext.bin"
    sniff.write_bytes(dumps_midi(p))
    assert detect_format(sniff) == "midi"
    assert [t.name for t in load_any(sniff).tracks] == ["Lead", "Harmony"]
    with pytest.raises(ValueError):
        export_any(p, tmp_path / "x.unknown")
    with pytest.raises(ValueError):
        export_any(p, tmp_path / "x", fmt="nope")


@pytest.mark.parametrize("fmt,ext", [("midi", ".mid"), ("svp", ".svp"), ("musicxml", ".musicxml"), ("mxl", ".mxl"), ("ust", ".ust")])
def test_export_any_load_any_all_formats(tmp_path, fmt, ext):
    p = make_project()
    path = export_any(p, tmp_path / f"song{ext}", fmt=fmt)
    assert path.suffix == ext and path.exists()
    q = load_any(path)
    assert q.tracks and q.tracks[0].notes


def test_score_format_dispatch(tmp_path):
    score = pytest.importorskip("synth2.score")
    p = make_project()
    path = export_any(p, tmp_path / "song.s2s")
    assert path.read_text(encoding="utf-8").strip()
    q = load_any(path)
    assert notes_of(q.tracks[0]) == notes_of(p.tracks[0])
    assert hasattr(score, "parse_score")


# ---------------------------------------------------------------------------
# MIDI
# ---------------------------------------------------------------------------
def test_midi_round_trip(tmp_path):
    p = make_project()
    path = write_midi(p, tmp_path / "song.mid")
    q = read_midi(path)
    assert q.title == "Round Trip"
    assert tempo_of(q) == tempo_of(p)
    assert q.time_signature == (3, 4)
    assert q.key == "Bb major"
    assert [t.name for t in q.tracks] == ["Lead", "Harmony"]
    assert notes_of(q.tracks[0], ("velocity",)) == notes_of(p.tracks[0], ("velocity",))
    assert notes_of(q.tracks[1]) == notes_of(p.tracks[1])
    assert q.tracks[1].language == "ja"


def _vlq(v: int) -> bytes:
    out = [v & 0x7F]
    v >>= 7
    while v:
        out.append(0x80 | (v & 0x7F))
        v >>= 7
    return bytes(reversed(out))


def _smf(fmt: int, ppq: int, *tracks: bytes) -> bytes:
    data = b"MThd" + struct.pack(">IHHH", 6, fmt, len(tracks), ppq)
    for body in tracks:
        body += b"\x00\xff\x2f\x00"
        data += b"MTrk" + struct.pack(">I", len(body)) + body
    return data


def _meta(delta: int, mtype: int, payload: bytes) -> bytes:
    return _vlq(delta) + bytes([0xFF, mtype]) + _vlq(len(payload)) + payload


def test_midi_parse_minimal_format0():
    ppq = 96
    body = (
        _meta(0, 0x03, b"Melody")
        + _meta(0, 0x51, (500000).to_bytes(3, "big"))           # 120 bpm
        + _meta(0, 0x58, bytes([6, 3, 24, 8]))                   # 6/8
        + _meta(0, 0x59, bytes([0xFE, 1]))                       # 2 flats, minor -> G minor
        + _meta(0, 0x05, b"hel-")
        + b"\x00\x90\x3c\x50"                                    # C4 on
        + _vlq(ppq) + b"\x3c\x00"                                # running status, vel 0 = off
        + _meta(0, 0x05, b"lo")
        + b"\x00\x3e\x50"                                        # D4 on (running status)
        + b"\x00\x40\x40"                                        # E4 on together (chord: keep highest)
        + _vlq(ppq // 2) + b"\x43\x50"                           # G4 starts while D4/E4 sound
        + _meta(0, 0x51, (400000).to_bytes(3, "big"))            # 150 bpm at beat 1.5
        + _vlq(ppq // 2) + b"\x80\x3e\x00"                       # explicit note-offs
        + b"\x00\x40\x00" + b"\x00\x43\x00"
        + _meta(0, 0x05, b" ")                                   # empty lyric -> "la"
        + b"\x00\x99\x24\x64" + _vlq(ppq) + b"\x89\x24\x00"      # drum note on channel 10: ignored
        + b"\x00\x91\x48\x64" + _vlq(ppq) + b"\x81\x48\x00"      # channel 2 note -> second track
    )
    p = loads_midi(_smf(0, ppq, body), title="t")
    assert p.title == "Melody"
    assert p.time_signature == (6, 8)
    assert p.key == "G minor"
    assert tempo_of(p) == [(0.0, 120.0), (1.5, 150.0)]
    assert [t.name for t in p.tracks] == ["Melody ch1", "Melody ch2"]
    assert notes_of(p.tracks[0]) == [(0.0, 1.0, 60, "hello"), (1.0, 0.5, 64, "+"), (1.5, 0.5, 67, "la")]
    assert notes_of(p.tracks[1]) == [(3.0, 1.0, 72, "la")]


def test_midi_text_fallback_and_lyric_track():
    ppq = 480
    conductor = _meta(0, 0x03, b"Song") + _meta(0, 0x51, (600000).to_bytes(3, "big"))
    words = _meta(0, 0x03, b"Words") + _meta(0, 0x01, b"@KMIDI KARAOKE FILE") + _meta(0, 0x01, "\\さ".encode("utf-8")) + _meta(ppq, 0x01, "/く".encode("cp932"))
    melody = _meta(0, 0x03, b"Vocal") + b"\x00\x90\x3c\x64" + _vlq(ppq) + b"\x80\x3c\x00" + b"\x00\x90\x3e\x64" + _vlq(ppq) + b"\x80\x3e\x00"
    p = loads_midi(_smf(1, ppq, conductor, words, melody))
    assert p.title == "Song"
    assert tempo_of(p) == [(0.0, 100.0)]
    assert len(p.tracks) == 1 and p.tracks[0].name == "Vocal"
    assert [n.lyric for n in p.tracks[0].notes] == ["さ", "く"]
    assert p.tracks[0].language == "ja"


def test_midi_key_helpers():
    assert fifths_to_key(0) == "C major" and fifths_to_key(3, True) == "F# minor"
    assert key_to_fifths("Eb major") == (-3, False)
    assert key_to_fifths("a minor") == (0, True)
    assert key_to_fifths("Am") == (0, True)
    assert key_to_fifths("D# major") == (-3, False)  # enharmonic -> Eb
    assert key_to_fifths("nonsense") is None


# ---------------------------------------------------------------------------
# SVP
# ---------------------------------------------------------------------------
def make_svp_project() -> Project:
    p = make_project()
    lead = p.tracks[0]
    for n in lead.notes:
        n.attrs.pop("velocity", None)
    lead.notes[0].phonemes = "t w ih ng"
    lead.notes[2].attrs.update(vibrato_rate=6.0, vibrato_start=0.2, transition=0.12, detune=15.0)
    lead.notes[4].attrs["vibrato_depth"] = 0.0
    lead.notes[6].attrs["language"] = "ja"
    lead.curves["pitchDelta"] = Curve([(0.0, 0.0), (1.5, 50.0), (3.0, -25.0)], "cubic")
    lead.curves["loudness"] = Curve([(0.0, -2.0), (4.0, 3.0)], "linear")
    lead.curves["voicing"] = Curve([(0.0, 1.0), (2.0, 0.5)], "cubic")
    lead.curves["tension"] = Curve([(1.0, 0.4)], "cubic")
    lead.gain_db, lead.pan, lead.mute = -1.5, -0.5, True
    p.tracks[1].solo = True
    return p


def test_svp_round_trip(tmp_path):
    p = make_svp_project()
    path = write_svp(p, tmp_path / "song.svp")
    raw = path.read_bytes()
    assert raw.endswith(b"\x00")
    doc = json.loads(raw.rstrip(b"\x00"))
    assert doc["tracks"][0]["mainRef"]["groupID"] == doc["tracks"][0]["mainGroup"]["uuid"]
    assert doc["tracks"][1]["mainRef"]["database"]["language"] == "japanese"
    assert doc["tracks"][0]["mainGroup"]["notes"][0]["onset"] == 0
    assert doc["tracks"][0]["mainGroup"]["notes"][1]["onset"] == BLICKS_PER_BEAT
    q = read_svp(path)
    assert tempo_of(q) == tempo_of(p)
    assert q.time_signature == (3, 4)
    attrs = ("vibrato_rate", "vibrato_start", "transition", "detune", "vibrato_depth", "language")
    for a, b in zip(q.tracks, p.tracks):
        assert (a.name, a.language, a.gain_db, a.pan, a.mute, a.solo) == (b.name, b.language, b.gain_db, b.pan, b.mute, b.solo)
        assert notes_of(a, attrs) == notes_of(b, attrs)
        assert {k: c.to_dict() for k, c in a.curves.items()} == {k: c.to_dict() for k, c in b.curves.items()}
    assert q.tracks[0].notes[0].phonemes == "t w ih ng"


def test_svp_preserves_synthv_specific_data_through_json(tmp_path):
    """SynthV -> Synth2 (saved as JSON) -> SynthV keeps raw attributes, singer and phonemes."""
    sv = {
        "version": 153,
        "time": {"meter": [{"index": 0, "numerator": 4, "denominator": 4}], "tempo": [{"position": 0, "bpm": 120.0}]},
        "library": [],
        "tracks": [{
            "name": "Solo", "dispColor": "ff123456", "dispOrder": 0,
            "mixer": {"gainDecibel": 0.0, "pan": 0.0, "mute": False, "solo": False, "display": True},
            "mainGroup": {"name": "main", "uuid": "g0", "parameters": {}, "notes": [
                {"onset": 0, "duration": BLICKS_PER_BEAT, "lyrics": "la", "phonemes": "l a:", "pitch": 60,
                 "attributes": {"tF0Left": 0.05, "tF0Right": 0.09, "dF0Vbr": 1.5, "tF0Offset": -0.01, "evenSyllableDuration": True}},
            ]},
            "mainRef": {"groupID": "g0", "blickOffset": 0, "pitchOffset": 0, "isInstrumental": False,
                        "database": {"name": "Eleanor Forte", "language": "english", "phoneset": "arpabet"},
                        "voice": {"vocalModeInherited": True, "paramLoudness": 2}},
            "groups": [],
        }],
        "renderConfig": {"filename": "solo", "sampleRate": 48000},
    }
    p = loads_svp(json.dumps(sv))
    assert p.sample_rate == 48000
    note = p.tracks[0].notes[0]
    assert note.attrs["transition"] == pytest.approx(0.14)
    assert note.phonemes is None  # "a:" is not in our inventory -> kept in metadata
    assert p.metadata["svp"]["tracks"][p.tracks[0].id]["singer"] == "Eleanor Forte"
    p2 = Project.from_json(p.to_json())  # metadata survives the Synth2 JSON format
    out = project_to_svp_dict(p2)
    t = out["tracks"][0]
    assert out["version"] == 153
    assert t["dispColor"] == "ff123456"
    assert t["mainRef"]["database"]["name"] == "Eleanor Forte"
    assert t["mainRef"]["voice"]["paramLoudness"] == 2
    n = t["mainGroup"]["notes"][0]
    assert n["phonemes"] == "l a:"
    assert n["attributes"] == {"tF0Left": 0.05, "tF0Right": 0.09, "dF0Vbr": 1.5, "tF0Offset": -0.01, "evenSyllableDuration": True}
    # editing the transition in Synth2 rewrites the SynthV values
    p2.tracks[0].notes[0].attrs["transition"] = 0.2
    n = project_to_svp_dict(p2)["tracks"][0]["mainGroup"]["notes"][0]
    assert n["attributes"]["tF0Left"] == pytest.approx(0.1) and n["attributes"]["tF0Right"] == pytest.approx(0.1)


def test_svp_parse_minimal_with_library_group(tmp_path):
    B = BLICKS_PER_BEAT
    doc = {
        "version": 100,
        "unknownTopLevel": {"x": 1},
        "time": {"meter": [{"index": 0, "numerator": 3, "denominator": 4}],
                 "tempo": [{"position": 0, "bpm": 90}, {"position": 4 * B, "bpm": 150}]},
        "library": [{
            "name": "chorus", "uuid": "lib-1",
            "parameters": {"loudness": {"mode": "linear", "points": [0, -100.0, B, 3.0]}},
            "notes": [
                {"onset": 0, "duration": B, "lyrics": "", "pitch": 60},
                {"onset": B, "duration": B // 2, "lyrics": "-", "pitch": 62, "detune": -10},
            ],
        }],
        "tracks": [
            {"name": "Vox", "dispOrder": 1, "mixer": {"gainDecibel": -6, "pan": 0.3, "mute": False, "solo": False},
             "mainGroup": {"name": "main", "uuid": "m1",
                           "parameters": {"pitchDelta": {"mode": "cubic", "points": [0, 0, B, 2000]},
                                          "voicing": {"mode": "cubic", "points": [[0, 0.5]]}},
                           "notes": [{"onset": 0, "duration": B, "lyrics": "あ", "pitch": 67, "phonemes": "a",
                                      "attributes": {"dF0Vbr": 0, "fF0Vbr": 6.2, "tF0VbrStart": 0.3}}]},
             "mainRef": {"groupID": "m1", "blickOffset": 0, "pitchOffset": 0, "isInstrumental": False,
                         "database": {"name": "Saki", "language": "japanese"}},
             "groups": [{"groupID": "lib-1", "blickOffset": 2 * B, "pitchOffset": 12, "isInstrumental": False},
                        {"groupID": "missing", "blickOffset": 0, "pitchOffset": 0}]},
            {"name": "Backing", "dispOrder": 0, "mainGroup": {"notes": []},
             "mainRef": {"isInstrumental": True}, "groups": []},
        ],
    }
    path = tmp_path / "mini.svp"
    path.write_bytes(json.dumps(doc).encode("utf-8") + b"\x00\x00")
    p = load_any(path)
    assert p.title == "mini"
    assert p.time_signature == (3, 4)
    assert tempo_of(p) == [(0.0, 90.0), (4.0, 150.0)]
    assert len(p.tracks) == 1
    t = p.tracks[0]
    assert (t.name, t.language, t.gain_db, t.pan) == ("Vox", "ja", -6.0, 0.3)
    assert notes_of(t, ("detune", "vibrato_depth", "vibrato_rate", "vibrato_start")) == [
        (0.0, 1.0, 67, "あ", None, 0.0, 6.2, 0.3),
        (2.0, 1.0, 72, "la", None, None, None, None),
        (3.0, 0.5, 74, "-", -10.0, None, None, None),
    ]
    assert t.notes[0].phonemes == "a"
    assert t.curves["pitchDelta"].points == [(0.0, 0.0), (1.0, 1200.0)]  # clamped to +-1200 cents
    assert t.curves["loudness"].interpolation == "linear"
    assert t.curves["loudness"].points == [(2.0, -48.0), (3.0, 3.0)]  # shifted by blickOffset, clamped
    assert t.curves["voicing"].points == [(0.0, 0.5)]
    meta = p.metadata["svp"]
    assert meta["tracks"][t.id]["singer"] == "Saki"
    assert meta["missing_groups"] == ["missing"]
    assert meta["skipped_instrumental_tracks"] == ["Backing"]


# ---------------------------------------------------------------------------
# MusicXML
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("ext", [".musicxml", ".mxl"])
def test_musicxml_round_trip(tmp_path, ext):
    p = make_project()
    path = write_musicxml(p, tmp_path / f"song{ext}")
    if ext == ".mxl":
        assert zipfile.is_zipfile(path)
    q = read_musicxml(path)
    assert q.title == "Round Trip"
    assert tempo_of(q) == tempo_of(p)
    assert q.time_signature == (3, 4)
    assert q.key == "Bb major"
    assert [t.name for t in q.tracks] == ["Lead", "Harmony"]
    for a, b in zip(q.tracks, p.tracks):
        assert notes_of(a) == notes_of(b)
    assert q.tracks[1].language == "ja"


def test_musicxml_round_trip_odd_durations(tmp_path):
    p = Project.new("Odd", bpm=120)
    t = p.tracks[0]
    t.add_note(0.5, 1 / 3, 60, "a")
    t.add_note(0.5 + 1 / 3, 2 / 3, 61, "b")
    t.add_note(1.5, 4.75, 63, "c")  # crosses a bar line, needs ties
    t.add_note(6.25, 0.3, 66, "d")  # not notatable: written with a bare <duration>
    q = loads_musicxml(write_musicxml(p, tmp_path / "odd.xml").read_bytes())
    assert notes_of(q.tracks[0]) == notes_of(t)


MINIMAL_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.1 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">
<score-partwise version="3.1">
  <movement-title>Little Song</movement-title>
  <part-list>
    <score-part id="P1"><part-name>Soprano</part-name></score-part>
    <score-part id="P2"><part-name>Piano</part-name></score-part>
  </part-list>
  <part id="P1">
    <measure number="1">
      <attributes><divisions>2</divisions><key><fifths>-1</fifths><mode>minor</mode></key>
        <time><beats>3</beats><beat-type>4</beat-type></time></attributes>
      <direction><direction-type><metronome><beat-unit>quarter</beat-unit><per-minute>90</per-minute></metronome></direction-type>
        <sound tempo="88"/></direction>
      <note><pitch><step>D</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <lyric number="1"><syllabic>begin</syllabic><text>Hap</text></lyric>
        <lyric number="2"><syllabic>single</syllabic><text>verse2</text></lyric></note>
      <note><pitch><step>E</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <lyric><syllabic>end</syllabic><text>py</text></lyric></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <tie type="start"/><notations><tied type="start"/></notations>
        <lyric><syllabic>single</syllabic><text>day</text></lyric></note>
      <backup><duration>6</duration></backup>
      <note><pitch><step>A</step><octave>3</octave></pitch><duration>6</duration><voice>2</voice></note>
    </measure>
    <measure number="2">
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <tie type="stop"/><notations><tied type="stop"/></notations></note>
      <note><rest/><duration>1</duration><voice>1</voice></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice>
        <notations><slur type="start" number="1"/></notations>
        <lyric><syllabic>single</syllabic><text>oh</text></lyric></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>1</duration><voice>1</voice></note>
      <note><chord/><pitch><step>C</step><octave>5</octave></pitch><duration>1</duration><voice>1</voice></note>
      <note><pitch><step>B</step><alter>-1</alter><octave>4</octave></pitch><duration>1</duration><voice>1</voice>
        <notations><slur type="stop" number="1"/></notations></note>
    </measure>
    <measure number="3">
      <direction><direction-type><metronome><beat-unit>half</beat-unit><per-minute>60</per-minute></metronome></direction-type></direction>
      <note><grace/><pitch><step>C</step><octave>5</octave></pitch><voice>1</voice></note>
      <note><pitch><step>A</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <lyric><syllabic>single</syllabic><text>la</text><extend/></lyric></note>
      <note><pitch><step>G</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice></note>
      <note><pitch><step>F</step><octave>4</octave></pitch><duration>2</duration><voice>1</voice>
        <lyric><syllabic>single</syllabic><text>end</text></lyric></note>
    </measure>
  </part>
  <part id="P2">
    <measure number="1">
      <attributes><divisions>1</divisions><staves>2</staves></attributes>
      <note><pitch><step>C</step><octave>4</octave></pitch><duration>3</duration><voice>1</voice><staff>1</staff></note>
      <backup><duration>3</duration></backup>
      <note><pitch><step>C</step><octave>3</octave></pitch><duration>3</duration><voice>5</voice><staff>2</staff></note>
    </measure>
  </part>
</score-partwise>
"""


def test_musicxml_parse_minimal():
    p = loads_musicxml(MINIMAL_XML.encode("utf-8"))
    assert p.title == "Little Song"
    assert p.key == "D minor"
    assert p.time_signature == (3, 4)
    assert tempo_of(p) == [(0.0, 88.0), (6.0, 120.0)]  # sound tempo wins; half = 60 -> 120 q/min
    assert [t.name for t in p.tracks] == ["Soprano", "Piano"]
    assert notes_of(p.tracks[0]) == [
        (0.0, 1.0, 62, "Happy"), (1.0, 1.0, 64, "+"),
        (2.0, 2.0, 65, "day"),              # tie across the bar line merged
        (4.5, 0.5, 67, "oh"), (5.0, 0.5, 72, "-"), (5.5, 0.5, 70, "-"),  # chord keeps top note; slur melisma
        (6.0, 1.0, 69, "la"), (7.0, 1.0, 67, "-"), (8.0, 1.0, 65, "end"),  # <extend/> melisma
    ]
    assert notes_of(p.tracks[1]) == [(0.0, 3.0, 60, "la")]


def test_musicxml_mxl_container_and_timewise(tmp_path):
    path = tmp_path / "packed.mxl"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container><rootfiles><rootfile full-path="scores/main.xml"/></rootfiles></container>')
        z.writestr("scores/main.xml", MINIMAL_XML)
    p = load_any(path)
    assert p.title == "Little Song" and len(p.tracks) == 2
    timewise = """<score-timewise><part-list><score-part id="A"><part-name>V</part-name></score-part></part-list>
      <measure number="1"><part id="A"><attributes><divisions>1</divisions></attributes>
        <note><pitch><step>C</step><octave>4</octave></pitch><duration>4</duration><lyric><text>a</text></lyric></note></part></measure>
      <measure number="2"><part id="A">
        <note><pitch><step>D</step><alter>0.5</alter><octave>4</octave></pitch><duration>2</duration><lyric><text>b</text></lyric></note></part></measure>
    </score-timewise>"""
    q = loads_musicxml(timewise)
    assert notes_of(q.tracks[0], ("detune",)) == [(0.0, 4.0, 60, "a", None), (4.0, 2.0, 63, "b", -50.0)]


# ---------------------------------------------------------------------------
# UST
# ---------------------------------------------------------------------------
def test_ust_round_trip(tmp_path):
    p = make_project()
    harm = p.tracks[1]
    harm.notes[0].attrs["velocity"] = 1.3
    harm.notes[2].attrs.update(vibrato_depth=0.3, vibrato_rate=5.5, vibrato_start=0.4)
    path = write_ust(p, tmp_path / "song.ust", track="Harmony")
    text = path.read_bytes().decode("utf-8")
    assert text.startswith("[#VERSION]\r\nUST Version1.2\r\nCharset=UTF-8")
    q = read_ust(path)
    assert q.title == "Round Trip"
    assert tempo_of(q) == tempo_of(p)
    t = q.tracks[0]
    assert t.language == "ja"
    assert notes_of(t, ("velocity",)) == notes_of(harm, ("velocity",))
    vib = t.notes[2].attrs
    assert vib["vibrato_depth"] == pytest.approx(0.3)
    assert vib["vibrato_rate"] == pytest.approx(5.5, abs=1e-3)
    assert vib["vibrato_start"] == pytest.approx(0.4, abs=1e-3)


def test_ust_tempo_changes_on_rests_and_inside_notes(tmp_path):
    p = make_project()
    p.tempo = TempoMap([TempoPoint(0.0, 96.0), TempoPoint(5.25, 132.0)])  # inside the 5..5.5 rest
    q = loads_ust(write_ust(p, tmp_path / "lead.ust").read_bytes())
    assert tempo_of(q) == tempo_of(p)
    assert notes_of(q.tracks[0]) == notes_of(p.tracks[0])
    p.tempo = TempoMap([TempoPoint(0.0, 96.0), TempoPoint(6.0, 132.0)])  # inside the 5.5..8 note
    q = loads_ust(write_ust(p, tmp_path / "lead2.ust").read_bytes())
    assert tempo_of(q) == [(0.0, 96.0), (5.5, 132.0)]  # moved to the nearest note boundary


MINIMAL_UST = """[#SETTING]
Tempo=150,00
Tracks=1
ProjectName=テスト
VoiceDir=%VOICE%uta
[#0000]
Length=480
Lyric=- さ
NoteNum=60
Intensity=80
[#0001]
Length=240
Lyric=R
NoteNum=60
[#0002]
Length=720
Lyric=a く
NoteNum=62
Tempo=100
VBR=50,200,40,20,20,0,0,0
[#0003]
Length=480
Lyric=ら
NoteNum=64
[#TRACKEND]
"""


def test_ust_parse_minimal_shift_jis(tmp_path):
    path = tmp_path / "mini.ust"
    path.write_bytes(MINIMAL_UST.replace("\n", "\r\n").encode("cp932"))
    p = load_any(path)
    assert p.title == "テスト"
    assert p.tracks[0].name == "テスト"
    assert p.tracks[0].language == "ja"
    assert tempo_of(p) == [(0.0, 150.0), (1.5, 100.0)]
    assert notes_of(p.tracks[0], ("velocity",)) == [(0.0, 1.0, 60, "さ", 0.8), (1.5, 1.5, 62, "く", None), (3.0, 1.0, 64, "ら", None)]
    vib = p.tracks[0].notes[1].attrs
    assert vib["vibrato_depth"] == pytest.approx(0.4)
    assert vib["vibrato_rate"] == pytest.approx(5.0)
    assert vib["vibrato_start"] == pytest.approx(0.45)  # 1.5 beats at 100 bpm = 0.9 s, vibrato on the last 50%
    assert p.metadata["ust"]["voice_dir"] == "%VOICE%uta"
