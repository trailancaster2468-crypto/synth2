"""Lyrics -> phonemes for whole tracks.

Lyric conventions (compatible with common vocal-synth practice):

* ``"star"``      a word / syllable (English G2P or Japanese kana/romaji)
* ``"-"``         melisma: keep singing the previous vowel on this note
* ``"+"``         next syllable of the previous multi-syllable word
                  (``"hello"`` ``"+"``  ->  "hel" "lo")
* ``"br"``        breath intake
* ``"[s t aa r]"`` or ``".s t aa r"``  explicit phonemes
* ``note.phonemes`` explicit phoneme override (space separated)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import english, japanese
from .inventory import PHONEMES, VOWELS, get, is_valid

MELISMA = {"-", "~", "ー"}
CONTINUATION = {"+", "+~"}


@dataclass
class Syllable:
    phones: list[str] = field(default_factory=list)
    melisma: bool = False  # sustain previous vowel
    breath: bool = False
    rest: bool = False  # nothing to sing (e.g. empty lyric)

    @property
    def vowel_positions(self) -> list[int]:
        return [i for i, p in enumerate(self.phones) if get(p).is_vowel or (get(p).kind == "nasal" and p == "N")]


def parse_phoneme_string(s: str) -> list[str]:
    toks = [t for t in re.split(r"[\s,]+", s.strip()) if t]
    out = []
    for t in toks:
        if not is_valid(t):
            raise ValueError(f"unknown phoneme {t!r} in {s!r}")
        out.append(get(t).name)
    return out


def word_to_phones(text: str, language: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if language == "ja" or japanese.is_kana(text):
        return japanese.g2p(text)
    return english.g2p(text)


def explicit_phones(lyric: str) -> list[str] | None:
    s = lyric.strip()
    if s.startswith("[") and s.endswith("]"):
        return parse_phoneme_string(s[1:-1])
    if s.startswith(".") and len(s) > 1:
        return parse_phoneme_string(s[1:])
    return None


def resolve_notes(notes, default_language: str = "en") -> list[Syllable]:
    """Resolve each note's lyric into a :class:`Syllable` (same length as ``notes``)."""
    out: list[Syllable] = [Syllable() for _ in notes]
    i = 0
    n = len(notes)
    while i < n:
        note = notes[i]
        lang = note.attrs.get("language") or default_language
        lyric = (note.lyric or "").strip()
        if note.phonemes:
            out[i] = Syllable(parse_phoneme_string(note.phonemes))
            i += 1
            continue
        if lyric in MELISMA or (lang == "ja" and lyric in ("ー",)):
            out[i] = Syllable(melisma=True)
            i += 1
            continue
        if lyric in CONTINUATION:
            out[i] = Syllable(melisma=True)  # orphan continuation -> melisma
            i += 1
            continue
        if lyric.lower() in ("br", "breath"):
            out[i] = Syllable(["br"], breath=True)
            i += 1
            continue
        if lyric == "" or lyric.lower() in ("r", "rest", "sil"):
            out[i] = Syllable(rest=True)
            i += 1
            continue
        exp = explicit_phones(lyric)
        if exp is not None:
            out[i] = Syllable(exp)
            i += 1
            continue
        # count following "+" notes
        j = i + 1
        while j < n and (notes[j].lyric or "").strip() in CONTINUATION and not notes[j].phonemes:
            j += 1
        group = j - i
        phones = word_to_phones(lyric, lang)
        if group == 1:
            out[i] = Syllable(phones)
        else:
            if lang == "ja" or japanese.is_kana(lyric):
                sylls = [japanese.romaji_to_phonemes(m) for m in japanese.morae(lyric)] or [phones]
            else:
                sylls = english.syllabify(phones)
            if not sylls:
                sylls = [phones]
            if len(sylls) > group:
                sylls = sylls[: group - 1] + [sum(sylls[group - 1:], [])]
            for k in range(group):
                if k < len(sylls):
                    out[i + k] = Syllable(sylls[k])
                else:
                    out[i + k] = Syllable(melisma=True)
        i = j
    return out


def lyric_phonemes(lyric: str, language: str = "en") -> list[str]:
    """Phonemes for a single lyric string (no context)."""
    exp = explicit_phones(lyric)
    if exp is not None:
        return exp
    return word_to_phones(lyric, language)


def split_lyrics(text: str, language: str = "en") -> list[str]:
    """Split free text into per-note lyric tokens (syllables).

    English words with several syllables are emitted as ``word`` followed by
    ``+`` tokens, so that the full word is pronounced correctly across notes.
    Japanese kana are split into morae.  Existing ``-``/``+`` tokens are kept.
    """
    tokens: list[str] = []
    for raw in text.split():
        if raw in MELISMA or raw in CONTINUATION:
            tokens.append(raw)
            continue
        if language == "ja" or japanese.is_kana(raw):
            tokens += [m if m != "-" else "-" for m in japanese.morae(raw)]
            continue
        # allow explicit hyphenation "twin-kle"
        if "-" in raw.strip("-"):
            tokens += [p for p in raw.split("-") if p]
            continue
        word = re.sub(r"[^A-Za-z']", "", raw)
        if not word:
            continue
        k = english.count_syllables(word)
        tokens.append(word)
        tokens += ["+"] * (k - 1)
    return tokens


__all__ = [
    "PHONEMES", "VOWELS", "Syllable", "get", "resolve_notes", "lyric_phonemes", "split_lyrics",
    "parse_phoneme_string", "english", "japanese",
]
