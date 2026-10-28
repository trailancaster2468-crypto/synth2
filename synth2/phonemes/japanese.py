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
})
_SMALL_Y = {"ゃ": "a", "ゅ": "u", "ょ": "o"}
_SMALL_V = {"ぁ": "a", "ぃ": "i", "ぅ": "u", "ぇ": "e", "ぉ": "o"}


def kata_to_hira(text: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in text)


def kana_to_romaji(text: str) -> list[str]:
    """Convert kana to a list of romaji morae."""
    text = kata_to_hira(text)
    out: list[str] = []
    i = 0
    while i < len(text):
        c = text[i]
        nxt = text[i + 1] if i + 1 < len(text) else ""
        base = KANA.get(c)
        if base is None:
            i += 1
            continue
        if nxt in _SMALL_Y and base.endswith("i") and base not in ("i",):
            stem = base[:-1]
            if stem in ("sh", "ch", "j"):
                out.append(stem + _SMALL_Y[nxt])
            else:
                out.append(stem + "y" + _SMALL_Y[nxt])
            i += 2
            continue
        if nxt in _SMALL_V and base not in ("a", "i", "u", "e", "o"):
            stem = base[:-1] if base[-1] in "aiueo" else base
            if stem == "ts" or stem == "f" or stem == "v" or stem in ("t", "d", "sh", "ch", "j", "w"):
                out.append(stem + _SMALL_V[nxt])
                i += 2
                continue
        out.append(base)
        i += 1
    return out


_CONS = {
    "": [], "k": ["k"], "g": ["g"], "s": ["s"], "z": ["z"], "t": ["t"], "d": ["d"], "n": ["n"],
    "h": ["hh"], "b": ["b"], "p": ["p"], "m": ["m"], "y": ["y"], "r": ["dx"], "w": ["w"],
    "sh": ["sh"], "ch": ["ch"], "ts": ["ts"], "j": ["jh"], "f": ["ph"], "v": ["v"],
    "ky": ["k", "y"], "gy": ["g", "y"], "ny": ["n", "y"], "hy": ["c"], "by": ["b", "y"],
    "py": ["p", "y"], "my": ["m", "y"], "ry": ["dx", "y"], "dy": ["d", "y"], "ty": ["t", "y"],
}
_MORA_RE = re.compile(r"(ky|gy|sh|ch|ts|ny|hy|by|py|my|ry|dy|ty|[kgsztdnhbpmyrwjfv]?)([aiueo])")


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
            # geminate consonant (e.g. "tte") -> glottal closure
            if pos + 1 < len(s) and s[pos] == s[pos + 1]:
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
    return [m.group(0) for m in re.finditer(r"(?:ky|gy|sh|ch|ts|ny|hy|by|py|my|ry|[kgsztdnhbpmyrwjfv])?[aiueo]|n(?![aiueoy])", text.lower())]
