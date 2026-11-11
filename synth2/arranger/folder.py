"""Instruments from plain folders of samples and from single sample files.

A drum kit: the file (or its folder) names the drum (``kick_1.wav``, ``Snare/hard.wav``,
``HiHat Open.wav``); every drum goes to its General MIDI key, so a folder kit is played
like any other kit.  A multisample: the file names carry the note (``Piano_C#4.wav``,
``bass_A1_v2.wav``); each sample covers the keys around its note.  Velocity layers
(``_v1`` ``_v2``, ``_p`` ``_mf`` ``_ff``, ``vel90``) and round robins (``_rr1`` ``_rr2``)
are read from the names.
"""
from __future__ import annotations

import os
import re

from . import gm
from .sampler import Envelope, Region, SampleInstrument
from .samples import AUDIO_EXTS, Sample, SampleError, layer_from_name, note_from_name, open_sample

MAX_FILES = 4000
# (words in a file or folder name, voice); the first match wins, so the specific names come first
_DRUM_WORDS = (
    (("ridebell", "ride bell", "bell ride", "ride_bell"), "ridebell"),
    (("open hat", "open hh", "open hihat", "open hi hat", "hat open", "hh open", "hihat open", "hi hat open",
      "openhat", "ohh", "ohat", "hho", "hat_o", "oh"), "ohat"),
    (("pedal hat", "pedal hh", "pedal hihat", "hat pedal", "hh pedal", "hihat pedal", "foot hat", "hat foot",
      "pedalhat", "phh", "phat"), "phat"),
    (("closed hat", "closed hh", "closed hihat", "hat closed", "hh closed", "hihat closed", "closedhat", "hihat",
      "hi hat", "chh", "hhc", "hat", "hh"), "hat"),
    (("kick", "bass drum", "bassdrum", "bd", "kik", "kck"), "kick"),
    (("side stick", "sidestick", "rimshot", "rim shot", "cross stick", "xstick", "rim", "stick"), "rim"),
    (("snare", "snr", "sd", "sn"), "snare"),
    (("handclap", "hand clap", "clap", "cp"), "clap"),
    (("finger snap", "fingersnap", "snap"), "snap"),
    (("splash",), "splash"), (("china",), "china"), (("crash", "cymbal", "cym"), "crash"), (("ride",), "ride"),
    (("floor tom", "floortom", "low tom", "lowtom", "tom low", "tom lo", "tom floor", "tom3", "tom 3", "ltom", "lt",
      "ft"), "tom3"),
    (("mid tom", "midtom", "tom mid", "tom2", "tom 2", "mtom", "mt"), "tom2"),
    (("high tom", "hi tom", "hightom", "hitom", "tom high", "tom hi", "rack tom", "tom1", "tom 1", "htom", "ht",
      "tom"), "tom1"),
    (("tambourine", "tamb", "tmb"), "tamb"), (("cowbell", "cow bell", "cowb", "cb"), "cowbell"),
    (("shaker", "shk"), "shaker"), (("maracas", "maraca"), "maracas"), (("cabasa",), "cabasa"),
    (("bongo",), "bongo_hi"), (("conga",), "conga_hi"), (("timbale",), "timbale_hi"), (("agogo",), "agogo_hi"),
    (("claves", "clave"), "claves"), (("woodblock", "wood block", "block"), "woodblock_hi"),
    (("triangle",), "triangle"), (("guiro",), "guiro"), (("cuica",), "cuica"), (("whistle",), "whistle"),
    (("vibraslap",), "vibraslap"),
)
_SIZES = (("ll", "lll", "lowest"), ("l", "lo", "low", "floor"), ("m", "mid", "middle"), ("h", "hi", "high"),
          ("hh", "hhh", "highest"))
_TOM_SIZES = ("tom3b", "tom3", "tom2", "tom1", "tom1b")


def _size(words: str) -> int | None:
    """0..4 (lowest .. highest) when a name carries a size word or letter, else None."""
    for i, names in enumerate(_SIZES):
        if any(f" {n} " in words for n in names):
            return i
    return None


def _words(name: str) -> str:
    """A name as lower-case words separated by single spaces (camel case and digits split off)."""
    s = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    s = re.sub(r"[^A-Za-z0-9]+", " ", s).lower()
    s = re.sub(r"([a-z])(\d)", r"\1 \2", s)
    return " " + re.sub(r"\s+", " ", s).strip() + " "


def drum_voice_from_name(path: str, depth: int = 2) -> str | None:
    """The kit voice a sample file is, from its name or the names of its folders."""
    parts = os.path.normpath(path).split(os.sep)
    stem = os.path.splitext(parts[-1])[0]
    for text in [stem] + parts[-1 - depth:-1][::-1]:
        w = _words(text)
        joined = w.replace(" ", "")
        for words, voice in _DRUM_WORDS:
            for word in words:
                if (f" {word} " in w) or (len(word) > 3 and word.replace(" ", "") in joined):
                    size = _size(w)
                    if word == "tom" and size is not None:
                        return _TOM_SIZES[size]
                    if voice.endswith("_hi") and size is not None and size <= 1:
                        return voice[:-3] + "_lo"
                    return voice
    return None


def _audio_files(folder: str) -> list[str]:
    out = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in AUDIO_EXTS and not f.startswith("."):
                out.append(os.path.join(root, f))
                if len(out) >= MAX_FILES:
                    return out
    return out


def _open_all(files: list[str], warnings: list[str]) -> list[tuple[str, Sample]]:
    out, bad = [], []
    for f in files:
        try:
            out.append((f, open_sample(f)))
        except SampleError as exc:
            bad.append(str(exc))
    if bad:
        warnings.append(f"{len(bad)} files could not be read ({bad[0]})")
    return out


def _velocity_ranges(values: list[int]) -> dict[int, tuple[int, int]]:
    """Velocity range of each layer number or velocity found in file names."""
    vals = sorted(set(values))
    if len(vals) <= 1:
        return {v: (1, 127) for v in vals}
    out = {}
    if vals[-1] <= 16:  # layer numbers: equal shares
        for i, v in enumerate(vals):
            out[v] = (1 + (127 * i) // len(vals), (127 * (i + 1)) // len(vals))
    else:  # velocities: each layer up to half way to the next
        lo = 1
        for a, b in zip(vals, vals[1:] + [None]):
            hi = 127 if b is None else (a + b) // 2
            out[a] = (lo, hi)
            lo = hi + 1
    return out


def _layered(items: list[tuple[str, Sample]], make) -> tuple[list[Region], int]:
    """Regions for the samples of one drum or one note: velocity layers and round robins from
    the file names. Files that carry neither are alternatives; the first is played and the
    rest are counted."""
    hinted = [(f, s, *layer_from_name(f)) for f, s in items]
    if all(v is None and rr is None for _f, _s, v, rr in hinted):
        f, s, _v, _rr = hinted[0]
        return [make(s, 1, 127, 1, 1)], len(hinted) - 1
    if any(v is not None for _f, _s, v, _rr in hinted):
        # a file without a layer mark next to marked ones is the normal layer; unmarked extras are alternatives
        plain = [h for h in hinted if h[2] is None and h[3] is None]
        mark = 3 if max(v for _f, _s, v, _rr in hinted if v is not None) <= 16 else 80
        hinted = [h for h in hinted if h[2] is not None or h[3] is not None] + [(f, s, mark, None) for f, s, _v, _r
                                                                               in plain[:1]]
        dropped = max(0, len(plain) - 1)
    else:
        dropped = 0
    ranges = _velocity_ranges([v for _f, _s, v, _rr in hinted if v is not None])
    by_layer: dict = {}
    for f, s, v, rr in hinted:
        by_layer.setdefault(v, []).append((rr if rr is not None else 0, f, s))
    regions = []
    for v, takes in by_layer.items():
        lo, hi = ranges.get(v, (1, 127))
        takes.sort(key=lambda x: (x[0], x[1]))
        for i, (_rr, _f, s) in enumerate(takes):
            regions.append(make(s, lo, hi, len(takes), i + 1))
    return regions, dropped


def kit_from_files(files: list[str], name: str, source: str = "") -> SampleInstrument:
    """A drum kit from sample files named after their drums (see the module text)."""
    warnings: list[str] = []
    by_voice: dict[str, list] = {}
    unknown = []
    for f, smp in _open_all(files, warnings):
        voice = drum_voice_from_name(f)
        (by_voice.setdefault(voice, []) if voice else unknown).append((f, smp))
    regions: list[Region] = []
    mapping = {}
    unused = 0
    for voice, items in by_voice.items():
        key = gm.drum_keys(voice)[0] if gm.drum_keys(voice) else 28
        group = 1 if voice in ("hat", "phat", "ohat") else 0

        def make(s, lo, hi, n, pos, key=key, group=group, voice=voice):
            return Region(sample=s, lokey=key, hikey=key, lovel=lo, hivel=hi, root=key, loop_mode="one_shot",
                          seq_length=n, seq_position=pos, group=group, off_by=1 if voice == "ohat" else 0,
                          off_time=0.012)

        regs, extra = _layered(items, make)
        regions += regs
        unused += extra
        mapping[voice] = key
    next_key = 84
    for f, smp in unknown[:40]:
        regions.append(Region(sample=smp, lokey=next_key, hikey=next_key, root=next_key, loop_mode="one_shot"))
        mapping[f"n{next_key}"] = os.path.basename(f)
        next_key += 1
    if unknown:
        listed = ", ".join(f"n{84 + i}: {os.path.basename(f)}" for i, (f, _s) in enumerate(unknown[:6]))
        warnings.append(f"{len(unknown)} files name no drum; they are on the keys from 84 ({listed}"
                        + (" ..." if len(unknown) > 6 else "") + ")")
    if unused:
        warnings.append(f"{unused} alternative samples not used (name one with voices.<drum>.sample, or mark "
                        "takes _rr1, _rr2 and layers _v1, _v2 in the file names)")
    if not regions:
        warnings.append("no sample files")
    return SampleInstrument(name=name, regions=regions, drums=True, warnings=warnings,
                            info={"format": "folder", "file": source, "kind": "kit", "voices": mapping})


def multisample_from_files(files: list[str], name: str, source: str = "") -> SampleInstrument:
    """A pitched instrument from sample files named after their notes."""
    warnings: list[str] = []
    by_note: dict[int, list] = {}
    unnamed = []
    for f, smp in _open_all(files, warnings):
        note = note_from_name(f)
        if note is None and smp.root is not None and smp.loop is not None:
            note = smp.root
        (by_note.setdefault(note, []) if note is not None else unnamed).append((f, smp))
    if not by_note and len(unnamed) == 1:
        return single_sample(unnamed[0][0], False)
    if unnamed:
        warnings.append(f"{len(unnamed)} files name no note and are not used (first: {os.path.basename(unnamed[0][0])})")
    notes = sorted(by_note)
    regions: list[Region] = []
    unused = 0
    for i, note in enumerate(notes):
        lokey = 0 if i == 0 else (notes[i - 1] + note) // 2 + 1
        hikey = 127 if i == len(notes) - 1 else (note + notes[i + 1]) // 2

        def make(s, lo, hi, n, pos, note=note, lokey=lokey, hikey=hikey):
            looped = s.loop is not None
            return Region(sample=s, lokey=lokey, hikey=hikey, lovel=lo, hivel=hi, root=note, tune=s.tune,
                          loop_mode="loop_continuous" if looped else "no_loop",
                          loop_start=s.loop[0] if looped else 0, loop_end=s.loop[1] if looped else 0,
                          seq_length=n, seq_position=pos, ampeg=Envelope(release=0.3))

        regs, extra = _layered(by_note[note], make)
        regions += regs
        unused += extra
    if unused:
        warnings.append(f"{unused} files share a note with another and are not used (mark layers _v1, _v2 or "
                        "takes _rr1, _rr2 in the file names)")
    if not regions:
        warnings.append("no sample files with a note in their name")
    return SampleInstrument(name=name, regions=regions, warnings=warnings,
                            info={"format": "folder", "file": source, "kind": "multisample", "notes": notes})


def single_sample(path: str, drums: bool) -> SampleInstrument:
    """One sample file as an instrument: a drum on the key of the drum it names (60 when it
    names none), or a pitched sound across the keyboard around its own note."""
    smp = open_sample(path)
    name = os.path.splitext(os.path.basename(path))[0]
    if drums:
        voice = drum_voice_from_name(path, depth=0)
        key = gm.drum_keys(voice)[0] if voice and gm.drum_keys(voice) else 60
        reg = Region(sample=smp, lokey=key, hikey=key, root=key, loop_mode="one_shot")
    else:
        root = note_from_name(path)
        if root is None:
            root = smp.root if smp.root is not None else 60
        looped = smp.loop is not None
        reg = Region(sample=smp, root=root, tune=smp.tune, loop_mode="loop_continuous" if looped else "no_loop",
                     loop_start=smp.loop[0] if looped else 0, loop_end=smp.loop[1] if looped else 0,
                     ampeg=Envelope(release=0.3))
    return SampleInstrument(name=name, regions=[reg], drums=drums, info={"format": "wav", "file": path})


def folder_instrument(folder: str, drums: bool | None = None) -> SampleInstrument:
    """An instrument from a folder of samples: a kit when ``drums`` (or, when it is None,
    when more files name drums than notes), else a multisample."""
    files = _audio_files(folder)
    name = os.path.basename(os.path.normpath(folder))
    if not files:
        raise SampleError(f"{folder} has no sample files ({', '.join(AUDIO_EXTS)})")
    if drums is None:
        named = sum(1 for f in files if drum_voice_from_name(f) is not None)
        noted = sum(1 for f in files if note_from_name(f) is not None)
        drums = named > noted
    inst = kit_from_files(files, name, folder) if drums else multisample_from_files(files, name, folder)
    if len(files) >= MAX_FILES:
        inst.warnings.append(f"only the first {MAX_FILES} files of the folder were read")
    return inst
