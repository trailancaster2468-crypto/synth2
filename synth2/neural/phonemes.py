"""Synth2 phonemes as a voicebank's symbols.

DiffSinger banks name their phonemes ``<language>/<phoneme>`` (``en/aa``, ``ja/a``) next to
the global ``SP`` (silence) and ``AP`` (breath); older banks use bare names. English Synth2
phonemes are ARPABET names and exist under the same name in the banks seen so far. Japanese
ones differ (:data:`JA_MAP`). A phoneme the bank lacks is looked up under the other
language, then bare, then through :data:`FALLBACKS`; the last resort is ``SP`` (the phoneme
is not sung). Every substitution is counted so the render report can say what happened.
"""
from __future__ import annotations

SILENCE = "SP"
BREATH = "AP"

# Synth2's Japanese phonemes that are spelled differently in the banks' Japanese sets
# (checked against Qixuan's dsdict-ja.yaml: ひ = h i, ふ = f u, じ = j i, ら = r a, っ = cl).
JA_MAP = {"c": "h", "ph": "f", "dz": "z", "hh": "h", "jh": "j", "q": "cl", "l": "r", "dx": "r", "v": "v"}
# ひゃ ひゅ ひょ are "hy a" in the banks; Synth2 writes the palatal fricative "c" for both
JA_PALATAL = {"c": "hy"}

# nearest phonemes to try, in order, when a bank lacks one (names without the language prefix)
FALLBACKS = {
    "en": {"ax": ["ah"], "dx": ["d", "t"], "zh": ["jh", "sh"], "er": ["_r", "r"], "_r": ["er", "r"],
           "ao": ["aa", "ow"], "uh": ["uw"], "ih": ["iy"], "eh": ["ae"], "ng": ["n"], "th": ["s", "f"],
           "dh": ["d", "z"], "hh": ["h"], "ts": ["t"], "dz": ["z"], "c": ["hh", "h"], "ph": ["f"],
           "a": ["aa"], "i": ["iy"], "u": ["uw"], "e": ["eh"], "o": ["ow"], "N": ["n"]},
    "ja": {"cl": ["q"], "f": ["h"], "j": ["z"], "v": ["b"], "ts": ["t"], "ch": ["t"], "sh": ["s"],
           "N": ["n"], "w": ["u"], "y": ["i"], "hy": ["h"],
           "aa": ["a"], "ae": ["a"], "ah": ["a"], "ao": ["o"], "aw": ["a"], "ax": ["a"], "ay": ["a"],
           "eh": ["e"], "er": ["a"], "ey": ["e"], "ih": ["i"], "iy": ["i"], "ow": ["o"], "oy": ["o"],
           "uh": ["u"], "uw": ["u"], "th": ["s"], "dh": ["z"], "zh": ["j"], "ng": ["n"]},
}


class PhonemeMap:
    """Converts Synth2 phonemes for one bank and remembers what it had to substitute.

    ``has`` tells whether the bank knows a symbol (every model that reads phonemes must).
    """

    def __init__(self, has):
        self.has = has
        self.substituted: dict[str, str] = {}  # "ja/c" -> symbol sung instead
        self.dropped: set[str] = set()  # phonemes replaced by silence

    def _lookup(self, name: str, lang: str, other: str) -> str | None:
        for cand in (f"{lang}/{name}", f"{other}/{name}", name):
            if self.has(cand):
                return cand
        return None

    def convert(self, ph: str, ja: bool = False, nxt: str | None = None) -> str:
        """The bank's symbol for Synth2 phoneme ``ph`` of a Japanese (``ja``) or English
        syllable; ``nxt`` is the phoneme sung after it, when known."""
        if ph == "br":
            return BREATH
        if ph == "sil" or (ph == "q" and not ja):
            return SILENCE
        lang, other = ("ja", "en") if ja else ("en", "ja")
        name = ph
        if ja:
            name = JA_MAP.get(ph, ph)
            pal = JA_PALATAL.get(ph)
            if pal and nxt not in (None, "i") and self.has(f"ja/{pal}"):
                name = pal
        hit = f"{lang}/{name}"
        if self.has(hit):
            return hit
        alt = self._lookup(ph, lang, other)
        if alt is None:
            for cand in FALLBACKS[lang].get(name, []) + FALLBACKS[lang].get(ph, []):
                alt = self._lookup(cand, lang, other)
                if alt is not None:
                    break
        key = f"{lang}/{ph}"
        if alt is None:
            self.dropped.add(key)
            return SILENCE
        self.substituted[key] = alt
        return alt

    def warnings(self) -> list[str]:
        out = []
        if self.substituted:
            pairs = ", ".join(f"{k} as {v}" for k, v in sorted(self.substituted.items()))
            out.append(f"the voicebank lacks some phonemes, sung with the nearest it has: {pairs}")
        if self.dropped:
            out.append("the voicebank has no phoneme for " + ", ".join(sorted(self.dropped)) + "; left silent")
        return out
