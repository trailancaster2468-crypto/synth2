"""Japanese kana / romaji to phoneme conversion (one mora per note is typical)."""
from __future__ import annotations

import re

_H_ROWS = {
    "": "あいうえお", "k": "かきくけこ", "s": "さしすせそ", "t": "たちつてと", "n": "なにぬねの",
    "h": "はひふへほ", "m": "まみむめも", "r": "らりるれろ", "g": "がぎぐげご", "z": "ざじずぜぞ",
    "d": "だぢづでど", "b": "ばびぶべぼ", "p": "ぱぴぷぺぽ",
}
_IRREG = {"し": "shi", "ち": "chi", "つ": "tsu", "ふ": "fu", "じ": "ji", "ぢ": "ji", "づ": "zu"}

KANA: dict[str, str] = {}
for cons, row in _H_ROWS.items():
    for kana, v in zip(row, "aiueo"):
        KANA[kana] = _IRREG.get(kana, cons + v)
KANA.update({
    "や": "ya", "ゆ": "yu", "よ": "yo", "わ": "wa", "を": "o", "ん": "N", "ゐ": "i", "ゑ": "e",
    "ゔ": "vu", "っ": "cl", "ー": "-", "ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o",
    "ゃ": "ya", "ゅ": "yu", "ょ": "yo",  # stray small y-kana: sung as their full-size mora
})
_SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
_SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}
# consonant stems that combine with a following small vowel into one (mostly loanword) mora
_SMALL_V_STEMS = ("ts", "f", "v", "t", "d", "sh", "ch", "j", "w")
# loanword morae whose Hepburn spelling romaji_to_phonemes reads the Kunrei way (ti = chi, tu = tsu,
# si = shi, zi = ji) get IME-style codes instead, so ティ is sung t-i and not chi
_LOAN_CODES = {"ti": "thi", "tu": "twu", "si": "swi", "zi": "zwi"}


def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def _kana_units(text: str) -> list[tuple[str, str]]:
    """Split kana into morae: ``(kana spelling, romaji code)`` pairs (non-kana characters are skipped)."""
    hira = kata_to_hira(text)
    out: list[tuple[str, str]] = []
    i = 0
    while i < len(hira):
        c = hira[i]
        nxt = hira[i + 1] if i + 1 < len(hira) else ""
        base = KANA.get(c)
        if base is None:
            i += 1
            continue
        code = None
        if nxt in _SMALL_Y and base.endswith("i") and base != "i":
            stem = base[:-1]
            code = stem + _SMALL_Y[nxt] if stem in ("sh", "ch", "j") else stem + "y" + _SMALL_Y[nxt]
        elif nxt in _SMALL_Y and base in ("te", "de", "fu", "vu"):  # テュ デュ フュ ヴュ
            code = base[0] + "y" + _SMALL_Y[nxt]
        elif nxt in _SMALL_V and base == "u" and nxt != "ぅ":  # ウィ ウェ ウォ (ウァ)
            code = "w" + _SMALL_V[nxt]
        elif nxt == "ぇ" and base == "i":  # イェ
            code = "ye"
        elif nxt in _SMALL_V and base not in ("a", "i", "u", "e", "o"):
            stem = base[:-1] if base[-1] in "aiueo" else base
            if stem in _SMALL_V_STEMS or base in ("su", "zu"):  # ティ トゥ ファ シェ スィ ...
                code = stem + _SMALL_V[nxt]
                code = _LOAN_CODES.get(code, code)
        if code is not None:
            out.append((text[i:i + 2], code))
            i += 2
            continue
        out.append((text[i], base))
        i += 1
    return out


def kana_to_romaji(text: str) -> list[str]:
    """Convert kana to a list of romaji morae."""
    return [r for _, r in _kana_units(text)]


def kana_morae(text: str) -> list[str]:
    """Split kana into morae, keeping the kana spelling (``"きって"`` -> ``["き", "っ", "て"]``)."""
    return [k for k, _ in _kana_units(text)]


_CONS = {
    "": [], "k": ["k"], "g": ["g"], "s": ["s"], "z": ["z"], "t": ["t"], "d": ["d"], "n": ["n"],
    "h": ["hh"], "b": ["b"], "p": ["p"], "m": ["m"], "y": ["y"], "r": ["dx"], "w": ["w"],
    "sh": ["sh"], "ch": ["ch"], "ts": ["ts"], "j": ["jh"], "f": ["ph"], "v": ["v"],
    "ky": ["k", "y"], "gy": ["g", "y"], "ny": ["n", "y"], "hy": ["c"], "by": ["b", "y"],
    "py": ["p", "y"], "my": ["m", "y"], "ry": ["dx", "y"], "dy": ["d", "y"], "ty": ["t", "y"],
    "fy": ["ph", "y"], "vy": ["v", "y"],
    # loanword codes (see _LOAN_CODES): plain consonant before i/u
    "th": ["t"], "tw": ["t"], "sw": ["s"], "zw": ["z"],
}
_CONS_ALT = "|".join(sorted((c for c in _CONS if c), key=len, reverse=True))
_MORA_RE = re.compile(rf"({_CONS_ALT}|)([aiueo])")
# one romaji mora; a doubled consonant ("kitte", "matcha") is the closure mora "cl" (っ)
_ROMAJI_MORA_RE = re.compile(
    rf"(?:{_CONS_ALT})?[aiueo]|n(?![aiueoy])|(?P<gem>(?P<c>[kgsztdhbpfjcv])(?=(?P=c))|t(?=ch))")


def romaji_to_phonemes(romaji: str) -> list[str]:
    s = romaji.strip()
    if s == "N" or s.lower() in ("n", "nn"):
        return ["N"]
    if s == "cl":
        return ["q"]
    s = s.lower()
    out: list[str] = []
    pos = 0
    while pos < len(s):
        if s[pos] == "n" and (pos + 1 == len(s) or s[pos + 1] not in "aiueoy"):
            out.append("N")
            pos += 1
            continue
        m = _MORA_RE.match(s, pos)
        if not m:
            # geminate consonant (e.g. "tte", "tch") -> glottal closure
            if pos + 1 < len(s) and s[pos] == s[pos + 1] or s.startswith("tch", pos):
                out.append("q")
            pos += 1
            continue
        cons, vowel = m.groups()
        if cons == "h" and vowel == "i":
            out.append("c")
        elif cons == "h" and vowel == "u":
            out.append("ph")
        elif cons == "s" and vowel == "i":
            out.append("sh")
        elif cons == "t" and vowel == "i":
            out.append("ch")
        elif cons == "t" and vowel == "u":
            out.append("ts")
        elif cons == "z" and vowel == "i":
            out.append("jh")
        else:
            out += _CONS[cons]
        out.append(vowel)
        pos = m.end()
    return out


def is_kana(text: str) -> bool:
    return any("぀" <= c <= "ヿ" for c in text)


def g2p(text: str) -> list[str]:
    morae = kana_to_romaji(text) if is_kana(text) else [text]
    out: list[str] = []
    for m in morae:
        if m == "-":
            continue
        out += romaji_to_phonemes(m)
    return out


def morae(text: str) -> list[str]:
    """Split a kana string into morae (each typically sung on its own note)."""
    if is_kana(text):
        return kana_to_romaji(text)
    return ["cl" if m.group("gem") else m.group(0) for m in _ROMAJI_MORA_RE.finditer(text.lower())]
