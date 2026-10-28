"""English grapheme-to-phoneme conversion.

Uses the CMU Pronouncing Dictionary (``cmudict`` package) when available, with
a built-in letter-to-sound rule set as fallback for unknown words and word
fragments (e.g. lyrics split across notes like ``twin`` / ``kle``).
"""
from __future__ import annotations

import functools
import re

ARPA_MAP = {
    "AA": "aa", "AE": "ae", "AH": "ah", "AO": "ao", "AW": "aw", "AY": "ay", "B": "b", "CH": "ch",
    "D": "d", "DH": "dh", "EH": "eh", "ER": "er", "EY": "ey", "F": "f", "G": "g", "HH": "hh",
    "IH": "ih", "IY": "iy", "JH": "jh", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ng",
    "OW": "ow", "OY": "oy", "P": "p", "R": "r", "S": "s", "SH": "sh", "T": "t", "TH": "th",
    "UH": "uh", "UW": "uw", "V": "v", "W": "w", "Y": "y", "Z": "z", "ZH": "zh",
}
VOWELS = {"aa", "ae", "ah", "ao", "aw", "ay", "eh", "er", "ey", "ih", "iy", "ow", "oy", "uh", "uw", "ax"}

# Words whose dictionary pronunciation is not the best sung pronunciation.
SUNG_OVERRIDES = {
    "the": ["dh", "ax"],
    "a": ["ax"],
    "i": ["ay"],
    "my": ["m", "ay"],
    "to": ["t", "uw"],
    "you": ["y", "uw"],
    "your": ["y", "ao", "r"],
    "oh": ["ow"],
    "ooh": ["uw"],
    "ah": ["aa"],
    "la": ["l", "aa"],
    "na": ["n", "aa"],
    "da": ["d", "aa"],
    "ba": ["b", "aa"],
    "mm": ["m"],
    "hmm": ["hh", "m"],
    "yeah": ["y", "eh", "ah"],
    "love": ["l", "ah", "v"],
    "of": ["ah", "v"],
    "and": ["ae", "n", "d"],
}

LEGAL_ONSETS = {
    tuple(o.split()) for o in [
        "p r", "p l", "b r", "b l", "t r", "d r", "k r", "k l", "g r", "g l", "f r", "f l", "th r",
        "sh r", "s p", "s t", "s k", "s m", "s n", "s l", "s w", "s f", "t w", "d w", "k w", "g w",
        "th w", "p y", "b y", "f y", "v y", "m y", "k y", "hh y", "hh w", "s p r", "s p l", "s t r",
        "s k r", "s k w", "s k l", "s p y", "s k y", "v r", "sh w",
    ]
}


@functools.lru_cache(maxsize=1)
def _cmu():
    try:
        import cmudict  # type: ignore

        return cmudict.dict()
    except Exception:  # pragma: no cover - optional dependency
        return {}


def _convert_arpa(pron: list[str]) -> list[str]:
    out = []
    for ph in pron:
        base = ph.rstrip("012")
        stress = ph[len(base):]
        p = ARPA_MAP[base]
        if p == "ah" and stress == "0":
            p = "ax"
        out.append(p)
    return out


def lookup(word: str) -> list[str] | None:
    w = word.lower().strip("'")
    if w in SUNG_OVERRIDES:
        return list(SUNG_OVERRIDES[w])
    prons = _cmu().get(w)
    if prons:
        return _convert_arpa(prons[0])
    return None


# ---------------------------------------------------------------- fallback rules
# Ordered (pattern, phonemes) rules applied left to right; longest match first.
_RULES: list[tuple[str, list[str]]] = [
    ("tch", ["ch"]), ("dge", ["jh"]), ("igh", ["ay"]), ("ough", ["ao"]), ("augh", ["ao"]),
    ("tion", ["sh", "ax", "n"]), ("sion", ["zh", "ax", "n"]), ("ture", ["ch", "er"]),
    ("eau", ["ow"]), ("ear", ["iy", "r"]), ("air", ["eh", "r"]), ("are", ["aa", "r"]),
    ("our", ["aw", "er"]), ("ire", ["ay", "er"]), ("ore", ["ao", "r"]), ("ure", ["y", "uh", "r"]),
    ("sch", ["s", "k"]), ("chr", ["k", "r"]), ("wh", ["w"]), ("wr", ["r"]), ("kn", ["n"]),
    ("gn", ["n"]), ("ph", ["f"]), ("th", ["th"]), ("sh", ["sh"]), ("ch", ["ch"]), ("ck", ["k"]),
    ("ng", ["ng"]), ("qu", ["k", "w"]), ("gh", []), ("ee", ["iy"]), ("ea", ["iy"]), ("ie", ["iy"]),
    ("ei", ["ey"]), ("ey", ["ey"]), ("ai", ["ey"]), ("ay", ["ey"]), ("oa", ["ow"]), ("oo", ["uw"]),
    ("ou", ["aw"]), ("ow", ["ow"]), ("oi", ["oy"]), ("oy", ["oy"]), ("au", ["ao"]), ("aw", ["ao"]),
    ("ew", ["uw"]), ("ue", ["uw"]), ("ui", ["uw"]), ("ar", ["aa", "r"]), ("or", ["ao", "r"]),
    ("er", ["er"]), ("ir", ["er"]), ("ur", ["er"]), ("ll", ["l"]), ("ss", ["s"]), ("ff", ["f"]),
    ("tt", ["t"]), ("pp", ["p"]), ("mm", ["m"]), ("nn", ["n"]), ("rr", ["r"]), ("dd", ["d"]),
    ("bb", ["b"]), ("gg", ["g"]), ("zz", ["z"]), ("cc", ["k"]), ("le", ["ax", "l"]),
]
_SINGLE = {
    "a": ["ae"], "b": ["b"], "c": ["k"], "d": ["d"], "e": ["eh"], "f": ["f"], "g": ["g"], "h": ["hh"],
    "i": ["ih"], "j": ["jh"], "k": ["k"], "l": ["l"], "m": ["m"], "n": ["n"], "o": ["aa"], "p": ["p"],
    "q": ["k"], "r": ["r"], "s": ["s"], "t": ["t"], "u": ["ah"], "v": ["v"], "w": ["w"], "x": ["k", "s"],
    "y": ["y"], "z": ["z"],
}
_LONG = {"a": ["ey"], "e": ["iy"], "i": ["ay"], "o": ["ow"], "u": ["uw"]}
_V = set("aeiou")


def rules_g2p(word: str) -> list[str]:
    """Approximate letter-to-sound conversion for words missing from the dictionary."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return []
    out: list[str] = []
    i = 0
    n = len(w)
    # magic-e: vowel + single consonant + final e  => long vowel, silent e
    magic = n >= 3 and w[-1] == "e" and w[-2] not in _V and w[-3] in _V
    while i < n:
        if magic and i == n - 1:
            break
        ch = w[i]
        if magic and i == n - 3:
            out += _LONG[ch]
            i += 1
            continue
        for pat, ph in _RULES:
            if w.startswith(pat, i):
                if pat == "le" and i != n - 2:
                    continue
                out += ph
                i += len(pat)
                break
        else:
            if ch == "c" and i + 1 < n and w[i + 1] in "eiy":
                out.append("s")
            elif ch == "g" and i + 1 < n and w[i + 1] in "eiy" and i > 0:
                out.append("jh")
            elif ch == "y":
                if i == 0:
                    out.append("y")
                elif i == n - 1:
                    out.append("iy" if n > 2 else "ay")
                else:
                    out.append("ih")
            elif ch == "e" and i == n - 1 and n > 2:
                pass  # silent final e
            elif ch in _V and i == n - 1 and n <= 3:
                out += {"a": ["aa"], "e": ["iy"], "i": ["iy"], "o": ["ow"], "u": ["uw"]}[ch]
            else:
                out += _SINGLE[ch]
            i += 1
    if not any(p in VOWELS for p in out):
        out.append("ax")
    return out


def g2p_word(word: str) -> list[str]:
    return lookup(word) or rules_g2p(word)


def g2p(text: str) -> list[str]:
    phones: list[str] = []
    for word in re.findall(r"[A-Za-z']+", text):
        phones += g2p_word(word)
    return phones


def syllabify(phones: list[str]) -> list[list[str]]:
    """Split a phoneme sequence into syllables using the maximal-onset principle."""
    nuclei = [i for i, p in enumerate(phones) if p in VOWELS]
    if not nuclei:
        return [phones] if phones else []
    bounds = [0]
    for a, b in zip(nuclei, nuclei[1:]):
        cluster = phones[a + 1:b]
        split = len(cluster)  # all consonants go to the next syllable by default ...
        for k in range(len(cluster) + 1):
            onset = tuple(cluster[k:])
            if len(onset) <= 1 or onset in LEGAL_ONSETS:
                split = k
                break
        bounds.append(a + 1 + split)
    bounds.append(len(phones))
    return [phones[s:e] for s, e in zip(bounds, bounds[1:]) if e > s]


def word_syllables(word: str) -> list[list[str]]:
    return syllabify(g2p_word(word))


def count_syllables(word: str) -> int:
    return max(1, len(word_syllables(word)))


def split_word_text(word: str, n: int) -> list[str]:
    """Split the spelling of a word into ``n`` roughly syllable-shaped chunks (for display)."""
    w = word
    if n <= 1 or len(w) <= 1:
        return [w]
    # break after vowel groups
    parts = re.findall(r"[^aeiouy]*[aeiouy]+(?:[^aeiouy]*$)?", w.lower())
    if len(parts) < n:
        step = max(1, len(w) // n)
        parts = [w[i:i + step] for i in range(0, len(w), step)]
    while len(parts) > n:
        parts[-2] += parts[-1]
        parts.pop()
    # restore original casing lengths
    out, pos = [], 0
    for p in parts:
        out.append(w[pos:pos + len(p)])
        pos += len(p)
    if pos < len(w):
        out[-1] += w[pos:]
    return out
