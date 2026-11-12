"""Regression tests for the core model: project, timing, phonemes and analysis."""
import math
import tracemalloc
import uuid

import numpy as np
import pytest

from synth2 import analysis
from synth2.analysis import estimate_f0
from synth2.phonemes import english, japanese, lyric_phonemes, resolve_notes, split_lyrics
from synth2.project import Curve, Note, Project, Track, _new_id
from synth2.timing import TempoMap


def _notes(lyrics, language=None):
    out = [Note(i, 1, 60, lyr) for i, lyr in enumerate(lyrics)]
    if language:
        for n in out:
            n.attrs["language"] = language
    return out


def _resolve(lyrics, language="en"):
    return [(s.phones, s.melisma) for s in resolve_notes(_notes(lyrics), language)]


# ------------------------------------------------------------------ phonemes: Japanese
def test_split_lyrics_keeps_kana_on_non_japanese_track():
    toks = split_lyrics("さくら ほん", "en")
    assert toks == ["さ", "く", "ら", "ほ", "ん"]
    # each note is still recognised as Japanese (ん is the moraic nasal, not the letter 'n')
    assert [s.phones for s in resolve_notes(_notes(toks), "en")] == [
        ["s", "a"], ["k", "u"], ["dx", "a"], ["hh", "o"], ["N"]]
    assert split_lyrics("きって ラーメン", "en") == ["き", "っ", "て", "ラ", "-", "メ", "ン"]
    assert [s.phones for s in resolve_notes(_notes(["っ"]), "en")] == [["q"]]
    # Japanese tracks keep romaji morae
    assert split_lyrics("さくら", "ja") == ["sa", "ku", "ra"]


def test_long_vowel_mark_in_multi_note_kana_word_is_melisma():
    assert _resolve(["らー", "+"], "ja") == [(["dx", "a"], False), ([], True)]
    assert _resolve(["ラーメン", "+", "+", "+"], "ja") == [
        (["dx", "a"], False), ([], True), (["m", "e"], False), (["N"], False)]
    assert _resolve(["らー", "+"], "en") == [(["dx", "a"], False), ([], True)]  # kana auto-detected
    # surplus morae merged onto the last note: the long-vowel mark adds no phonemes
    assert _resolve(["ラーメン", "+"], "ja") == [(["dx", "a"], False), (["m", "e", "N"], False)]
    assert _resolve(["らーー", "+"], "ja") == [(["dx", "a"], False), ([], True)]


@pytest.mark.parametrize("kana,romaji,phones", [
    ("ティ", "thi", ["t", "i"]), ("トゥ", "twu", ["t", "u"]), ("ディ", "di", ["d", "i"]),
    ("ドゥ", "du", ["d", "u"]), ("デュ", "dyu", ["d", "y", "u"]), ("テュ", "tyu", ["t", "y", "u"]),
    ("フュ", "fyu", ["ph", "y", "u"]), ("ウィ", "wi", ["w", "i"]), ("ウェ", "we", ["w", "e"]),
    ("ウォ", "wo", ["w", "o"]), ("イェ", "ye", ["y", "e"]), ("スィ", "swi", ["s", "i"]),
    ("ファ", "fa", ["ph", "a"]), ("シェ", "she", ["sh", "e"]), ("ちゃ", "cha", ["ch", "a"]),
    ("きゃ", "kya", ["k", "y", "a"]), ("ち", "chi", ["ch", "i"]), ("つ", "tsu", ["ts", "u"]),
])
def test_loanword_small_kana(kana, romaji, phones):
    assert japanese.kana_to_romaji(kana) == [romaji]
    assert japanese.kana_morae(kana) == [kana]
    assert japanese.romaji_to_phonemes(romaji) == phones
    assert japanese.g2p(kana) == phones
    # the romaji code survives being split again (a ja-track note followed by '+')
    assert japanese.morae(romaji) == [romaji]


def test_loanword_words_and_note_mapping():
    assert japanese.g2p("パーティ") == ["p", "a", "t", "i"]
    assert _resolve(["ウィンター", "+", "+", "+"], "ja") == [
        (["w", "i"], False), (["N"], False), (["t", "a"], False), ([], True)]
    toks = split_lyrics("デュエット", "ja")
    assert toks == ["dyu", "e", "cl", "to"]
    assert [s.phones for s in resolve_notes(_notes(toks), "ja")] == [["d", "y", "u"], ["e"], ["q"], ["t", "o"]]
    # typed Kunrei romaji keep their reading
    assert japanese.romaji_to_phonemes("ti") == ["ch", "i"] and japanese.romaji_to_phonemes("tu") == ["ts", "u"]


def test_romaji_geminate_consonants_are_kept():
    assert split_lyrics("kitte matte", "ja") == ["ki", "cl", "te", "ma", "cl", "te"]
    assert split_lyrics("matcha", "ja") == ["ma", "cl", "cha"]
    assert japanese.romaji_to_phonemes("matcha") == ["m", "a", "q", "ch", "a"]
    assert japanese.morae("konnichiwa") == ["ko", "n", "ni", "chi", "wa"]
    expect = [(["k", "i"], False), (["q", "t", "e"], False)]
    assert _resolve(["kitte", "+"], "ja") == expect
    assert _resolve(["きって", "+"], "ja") == expect


# ------------------------------------------------------------------ phonemes: English
def test_typographic_apostrophes_keep_contractions():
    assert split_lyrics("I’m sure we’re fine, I’ll be there") == split_lyrics("I'm sure we're fine, I'll be there")
    assert split_lyrics("I’m sure we’re fine") == ["I'm", "sure", "we're", "fine"]
    for curly, plain in [("I’m", "I'm"), ("don’t", "don't"), ("we’re", "we're"), ("donʼt", "don't"),
                         ("it‘s", "it's")]:
        assert lyric_phonemes(curly) == lyric_phonemes(plain)
    assert lyric_phonemes("I’m") == ["ay", "m"]
    assert english.count_syllables("don’t") == 1


def test_yeah_is_one_syllable():
    assert english.count_syllables("yeah") == 1
    assert split_lyrics("oh yeah baby") == ["oh", "yeah", "baby", "+"]


def test_hyphenated_words_count_syllables_per_part():
    assert split_lyrics("never-ending self-esteem") == split_lyrics("never ending self esteem") == [
        "never", "+", "ending", "+", "self", "esteem", "+"]
    assert split_lyrics("twin-kle") == ["twin", "kle"]  # syllable hyphenation: one note per part
    assert split_lyrics("well-known, -") == ["well", "known", "-"]


# ------------------------------------------------------------------ project model
def test_curve_list_form_is_sorted():
    c = Track.from_dict({"curves": {"loudness": [[8, 0], [0, -6], [4, 6]]}}).curves["loudness"]
    assert c.points == [(0.0, -6.0), (4.0, 6.0), (8.0, 0.0)]
    assert Curve.from_dict([[8, 0], [0, -6], [4, 6]]).points == Curve.from_dict(
        {"points": [[8, 0], [0, -6], [4, 6]]}).points


def test_track_lookup_prefers_exact_id_over_index():
    p = Project.new()
    t = p.add_track("Harmony")
    t.id = "123456789012"  # an all-digit hex id
    assert p.track("123456789012") is t
    assert p.track("1") is t and p.track(1) is t and p.track("0") is p.tracks[0]
    assert p.track("harmony") is t
    with pytest.raises(LookupError):
        p.track("5")
    with pytest.raises(LookupError):
        p.track("nope")


def test_new_ids_are_never_all_digits(monkeypatch):
    real = uuid.uuid4
    seq = iter([uuid.UUID("12345678-9012-4345-8678-901234567890")])
    monkeypatch.setattr(uuid, "uuid4", lambda: next(seq, None) or real())
    assert not _new_id().isdigit()


def test_triplets_do_not_overlap_after_round_trip():
    p = Project.new()
    t = p.tracks[0]
    beat = 1.0
    for i in range(24):  # accumulated triplet onsets, as write_melody produces them
        t.add_note(beat, 2 / 3, 60 + i % 3, "la")
        beat += 2 / 3
    for q in (p.copy(), Project.from_json(p.to_json())):
        notes = q.tracks[0].notes
        assert not q.tracks[0].overlaps()
        assert all(b.onset - a.end == pytest.approx(0, abs=1e-9) for a, b in zip(notes, notes[1:]))
        assert all(n.duration == pytest.approx(2 / 3, abs=2e-6) for n in notes)


# ------------------------------------------------------------------ timing
@pytest.mark.parametrize("bpm", [0, -60, math.nan, math.inf])
def test_tempo_map_rejects_invalid_bpm(bpm):
    with pytest.raises(ValueError):
        TempoMap.from_list([{"beat": 0, "bpm": 120}, {"beat": 4, "bpm": bpm}])
    with pytest.raises(ValueError):
        Project.from_dict({"tempo": [{"beat": 0, "bpm": 120}, {"beat": 4, "bpm": bpm}],
                           "tracks": [{"notes": [{"onset": 0, "duration": 8, "pitch": 60}]}]})
    tm = TempoMap()
    with pytest.raises(ValueError):
        tm.set(4.0, bpm)
    assert tm.to_list() == [{"beat": 0.0, "bpm": 120.0}]  # unchanged


# ------------------------------------------------------------------ analysis
def test_estimate_f0_memory_is_bounded_and_blocking_is_exact(monkeypatch):
    sr = 22050
    x = np.sin(2 * np.pi * 220 * np.arange(10 * sr) / sr)
    tracemalloc.start()
    try:
        times, f0, conf = estimate_f0(x, sr, return_confidence=True)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert peak < 80e6  # one frame matrix for the whole input would need ~180 MB here
    assert len(f0) > 3 * analysis._F0_BLOCK
    assert np.median(f0) == pytest.approx(220, rel=0.01) and (f0 > 0).mean() > 0.95
    # block boundaries do not change the result
    monkeypatch.setattr(analysis, "_F0_BLOCK", 10 ** 6)
    ref = estimate_f0(x[: 3 * sr], sr, return_confidence=True)
    monkeypatch.setattr(analysis, "_F0_BLOCK", 97)
    got = estimate_f0(x[: 3 * sr], sr, return_confidence=True)
    assert all(np.array_equal(a, b) for a, b in zip(ref, got))


def test_pitch_accuracy_folds_octave_errors_per_frame():
    """A two-cluster f0 track (1x and 2x frames) must not median out to about +700 cents."""
    import numpy as np

    from synth2 import Project
    from synth2.analysis import pitch_accuracy
    from synth2.engine.renderer import RenderResult
    import synth2.analysis as analysis

    p = Project.new("t", bpm=60)
    tr = p.tracks[0]
    tr.add_note(1, 2, 59, "oh")
    sr = 8000
    res = RenderResult(np.zeros((sr * 4, 2), dtype=np.float32), sr, {tr.id: np.zeros((sr * 4, 2), dtype=np.float32)}, {})
    hz = 440 * 2 ** ((59 - 69) / 12)

    def fake_f0(mono, sr_, return_confidence=False, **kw):
        t = np.arange(0, 4, 0.005)
        f0 = np.where(np.arange(len(t)) % 2 == 0, hz, 2 * hz)
        return (t, f0, np.ones_like(t)) if return_confidence else (t, f0)

    orig = analysis.estimate_f0
    analysis.estimate_f0 = fake_f0
    try:
        out = pitch_accuracy(p, res)["tracks"][0]
    finally:
        analysis.estimate_f0 = orig
    assert out["max_abs_cents"] < 5


def test_a_project_remembers_its_file(tmp_path):
    from synth2.io import load_any

    p = Project.new("x", bpm=100)
    assert p.path is None and "path" not in p.to_dict()
    f = tmp_path / "song.s2p.json"
    p.save(f)
    assert p.path == str(f.resolve()) and "path" not in f.read_text()
    for q in (Project.load(f), load_any(f), load_any(str(f))):
        assert q.path == str(f.resolve()) and q.to_dict() == p.to_dict()
    assert p.copy().path == p.path and Project.from_dict(p.to_dict()).path is None
    other = Project.new("x", bpm=100)
    other.tracks[0].id = p.tracks[0].id
    assert other.to_dict() == p.to_dict() and other == p  # where a project lives is not part of what it is
    score = tmp_path / "tune.s2s"
    score.write_text("tempo: 120\ntrack: A\nC4 q la\n", encoding="utf-8")
    assert Project.load(score).path == str(score.resolve())


def test_relative_sound_paths_resolve_next_to_the_project_file(tmp_path, monkeypatch):
    from synth2.arranger import soundbank
    from synth2.io.wav import write_wav

    (tmp_path / "song" / "samples").mkdir(parents=True)
    t = np.arange(4000) / 22050
    write_wav(tmp_path / "song" / "samples" / "ping_C4.wav", 0.5 * np.sin(2 * np.pi * 262 * t) * np.exp(-t * 20), 22050)
    p = Project.new("x", bpm=120)
    p.backing = {"instruments": {"keys": {"preset": "piano", "sound": "samples/ping_C4.wav"}},
                 "patterns": {"hold": {"kind": "chords", "rhythm": "x---------------"}},
                 "sections": [{"name": "A", "bars": 1, "chords": ["C"], "parts": {"keys": "hold"}}]}
    f = tmp_path / "song" / "song.s2p.json"
    p.save(f)
    monkeypatch.chdir(tmp_path)  # not the project's folder
    loaded = Project.load(f)
    dirs = soundbank.project_dirs(loaded, loaded.backing)
    assert dirs[0] == str((tmp_path / "song").resolve())
    assert soundbank.find_path("samples/ping_C4.wav", dirs) is not None
    from synth2 import arranger

    assert arranger.sound_report(loaded)["instruments"]["keys"]["plays"] == "sampled"
    assert arranger.sound_report(Project.from_dict(loaded.to_dict()))["instruments"]["keys"]["plays"] == "synth"
    fresh = Project.from_dict(loaded.to_dict())  # no file: only the working directory is searched
    assert soundbank.find_path("samples/ping_C4.wav", soundbank.project_dirs(fresh, fresh.backing)) is None


def test_the_summary_names_a_tracks_role_when_it_has_one():
    p = Project.new("x", bpm=100)
    p.add_track("Double", role="double:Lead")
    tracks = p.summary()["tracks"]
    assert "role" not in tracks[0] and tracks[1]["role"] == "double:Lead"
