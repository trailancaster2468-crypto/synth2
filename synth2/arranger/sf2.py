"""SoundFont 2 (``.sf2``) reader for the sample player.

Read: presets, instruments, zones (global zones included), key and velocity ranges, sample
addresses and loop points with their offsets, sample modes (no loop, loop, loop until
release), root key and tuning (coarse, fine, scale tuning, the sample's pitch correction),
attenuation, pan, the volume envelope (with key-to-hold and key-to-decay), the low-pass
filter with its resonance, the modulation envelope to filter and pitch, the vibrato LFO,
the modulation LFO to pitch and volume, exclusive class, fixed key and velocity, and every
modulator whose sources are the note's key and velocity (velocity to attenuation, filter
cutoff, envelope times, filter envelope depth, sample start; key to filter cutoff, pan
...), with the linear, concave, convex and switch curves, at instrument and preset level.

Ignored (see :data:`IGNORED`): reverb and chorus sends (the mixer has its own), the
modulation LFO to the filter, 24-bit sample extensions (the 16-bit data is played), ROM
samples, and modulators driven by controllers, pitch wheel or pressure (an arrangement has
none; their destinations keep the value of the generators).  As in FluidSynth, initial
attenuation counts 0.4 dB per 10 units and the specification's default velocity-to-filter
modulator is off unless the file sets its own.

Compressed SoundFonts (``.sf3``) are refused with a message.
"""
from __future__ import annotations

import math
import os
import struct
import threading
from dataclasses import dataclass

import numpy as np

from .sampler import Envelope, Region, SampleInstrument
from .samples import Sample, SampleError

IGNORED = ("reverb and chorus sends", "modulation LFO to filter", "24-bit sample extension (sm24)", "ROM samples",
           "modulators driven by controllers, pitch wheel or pressure")

# generator numbers
(G_START, G_END, G_LOOP_START, G_LOOP_END, G_START_COARSE, G_MODLFO_PITCH, G_VIBLFO_PITCH, G_MODENV_PITCH, G_FC, G_Q,
 G_MODLFO_FC, G_MODENV_FC, G_END_COARSE, G_MODLFO_VOL) = range(14)
G_PAN = 17
G_DELAY_MODLFO, G_FREQ_MODLFO, G_DELAY_VIBLFO, G_FREQ_VIBLFO = 21, 22, 23, 24
G_DELAY_MODENV, G_ATTACK_MODENV, G_HOLD_MODENV, G_DECAY_MODENV, G_SUSTAIN_MODENV, G_RELEASE_MODENV = range(25, 31)
G_DELAY_VOLENV, G_ATTACK_VOLENV, G_HOLD_VOLENV, G_DECAY_VOLENV, G_SUSTAIN_VOLENV, G_RELEASE_VOLENV = range(33, 39)
G_KEY_HOLD_VOLENV, G_KEY_DECAY_VOLENV = 39, 40
G_INSTRUMENT, G_KEYRANGE, G_VELRANGE, G_LOOP_START_COARSE, G_KEYNUM, G_VELOCITY, G_ATTENUATION = 41, 43, 44, 45, 46, 47, 48
G_LOOP_END_COARSE, G_COARSE_TUNE, G_FINE_TUNE, G_SAMPLE_ID, G_SAMPLE_MODES = 50, 51, 52, 53, 54
G_SCALE_TUNING, G_EXCLUSIVE_CLASS, G_ROOT_KEY = 56, 57, 58

_DEFAULTS = {G_FC: 13500, G_DELAY_MODLFO: -12000, G_DELAY_VIBLFO: -12000, G_DELAY_MODENV: -12000,
             G_ATTACK_MODENV: -12000, G_HOLD_MODENV: -12000, G_DECAY_MODENV: -12000, G_RELEASE_MODENV: -12000,
             G_DELAY_VOLENV: -12000, G_ATTACK_VOLENV: -12000, G_HOLD_VOLENV: -12000, G_DECAY_VOLENV: -12000,
             G_RELEASE_VOLENV: -12000, G_KEYNUM: -1, G_VELOCITY: -1, G_SCALE_TUNING: 100, G_ROOT_KEY: -1}
# generators a preset zone may not set (they are not additive)
_INSTRUMENT_ONLY = {G_START, G_END, G_LOOP_START, G_LOOP_END, G_START_COARSE, G_END_COARSE, G_LOOP_START_COARSE,
                    G_LOOP_END_COARSE, G_KEYNUM, G_VELOCITY, G_SAMPLE_MODES, G_EXCLUSIVE_CLASS, G_ROOT_KEY,
                    G_SAMPLE_ID, G_INSTRUMENT, G_KEYRANGE, G_VELRANGE}

_PHDR = np.dtype([("name", "S20"), ("preset", "<u2"), ("bank", "<u2"), ("bag", "<u2"), ("x", "V12")])
_BAG = np.dtype([("gen", "<u2"), ("mod", "<u2")])
_MOD = np.dtype([("src", "<u2"), ("dest", "<u2"), ("amount", "<i2"), ("amt_src", "<u2"), ("trans", "<u2")])
_GEN = np.dtype([("oper", "<u2"), ("amount", "<i2")])
_INST = np.dtype([("name", "S20"), ("bag", "<u2")])
_SHDR = np.dtype([("name", "S20"), ("start", "<u4"), ("end", "<u4"), ("loop_start", "<u4"), ("loop_end", "<u4"),
                  ("rate", "<u4"), ("pitch", "u1"), ("correction", "i1"), ("link", "<u2"), ("type", "<u2")])


def _name(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("latin-1", "replace").strip()


def _seconds(timecents: float) -> float:
    return 0.0 if timecents <= -11950 else float(2.0 ** (min(timecents, 8000) / 1200.0))


def _hz(cents: float) -> float:
    return float(8.176 * 2.0 ** (cents / 1200.0))


def _curve(x: float, kind: int) -> float:
    """A modulator curve on 0..1: 0 linear, 1 concave, 2 convex, 3 switch."""
    x = min(max(x, 0.0), 1.0)
    if kind == 1:
        return 1.0 if x >= 0.999 else min(1.0, -(40.0 / 96.0) * math.log10(1.0 - x))
    if kind == 2:
        return 0.0 if x <= 0.001 else max(0.0, 1.0 + (40.0 / 96.0) * math.log10(x))
    if kind == 3:
        return 1.0 if x >= 0.5 else 0.0
    return x


def _usable_source(oper: int) -> bool:
    """True for a modulator source the note itself supplies: none, note-on velocity or key."""
    return oper == 0 or (not oper & 0x80 and (oper & 0x7F) in (2, 3))


def _source(oper: int, key: int, vel: int) -> float:
    """Value of a modulator source (see :func:`_usable_source`) for a note."""
    if oper == 0:
        return 1.0
    x = (vel if (oper & 0x7F) == 2 else key) / 127.0
    if oper & 0x100:
        x = 1.0 - x
    kind = (oper >> 10) & 0x3F
    if oper & 0x200:  # bipolar
        v = 2.0 * x - 1.0
        return math.copysign(_curve(abs(v), kind), v)
    return _curve(x, kind)


_VEL_ATTEN = 0x0502  # note-on velocity, negative, unipolar, concave: the default source of attenuation


@dataclass(frozen=True)
class Preset:
    name: str
    bank: int
    program: int
    index: int

    def label(self) -> str:
        return f"{self.bank}:{self.program} {self.name}"


class SoundFont:
    """A parsed ``.sf2`` file. ``presets`` lists what it holds; :meth:`instrument` builds the
    playable regions of one preset (sample audio is read from the file when notes need it)."""

    def __init__(self, path: str):
        self.path = os.fspath(path)
        self.info: dict = {}
        self.ignored_modulators = 0
        self._lock = threading.Lock()
        self._built: dict[int, SampleInstrument] = {}
        self._samples: dict[int, Sample] = {}
        self._parse()

    # ---------------------------------------------------------------------------- parsing
    def _parse(self):
        path = self.path
        if os.path.splitext(path)[1].lower() == ".sf3":
            raise SampleError(f"{path}: compressed SoundFonts (.sf3, Ogg Vorbis samples) are not supported; "
                              "use the .sf2 version of the set")
        try:
            size = os.path.getsize(path)
            with open(path, "rb") as f:
                head = f.read(12)
                if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"sfbk":
                    raise SampleError(f"{path}: not a SoundFont 2 file")
                chunks = {}
                pos = 12
                while pos + 8 <= size:
                    f.seek(pos)
                    h = f.read(12)
                    if len(h) < 8:
                        break
                    cid, n = h[:4], struct.unpack("<I", h[4:8])[0]
                    if cid == b"LIST" and len(h) == 12:
                        sub, stop = pos + 12, min(size, pos + 8 + n)
                        while sub + 8 <= stop:
                            f.seek(sub)
                            sh = f.read(8)
                            sid, sn = sh[:4], struct.unpack("<I", sh[4:8])[0]
                            chunks[(h[8:12], sid)] = (sub + 8, min(sn, max(0, stop - sub - 8)))
                            sub += 8 + sn + (sn & 1)
                    pos += 8 + n + (n & 1)

                def blob(kind, cid, limit=None):
                    if (kind, cid) not in chunks:
                        return b""
                    off, n = chunks[(kind, cid)]
                    f.seek(off)
                    return f.read(n if limit is None else min(n, limit))

                ifil = blob(b"INFO", b"ifil")
                major = struct.unpack("<H", ifil[:2])[0] if len(ifil) >= 2 else 2
                self.info = {"name": _name(blob(b"INFO", b"INAM", 256)), "version": major,
                             "copyright": _name(blob(b"INFO", b"ICOP", 512)),
                             "engineer": _name(blob(b"INFO", b"IENG", 256))}
                if major >= 3:
                    raise SampleError(f"{path}: SoundFont version {major} (compressed samples) is not supported")
                if (b"sdta", b"smpl") not in chunks:
                    raise SampleError(f"{path}: the SoundFont has no sample data")
                self._smpl_off, self._smpl_bytes = chunks[(b"sdta", b"smpl")]
                self.has_sm24 = (b"sdta", b"sm24") in chunks

                def table(cid, dtype):
                    raw = blob(b"pdta", cid)
                    return np.frombuffer(raw, dtype=dtype, count=len(raw) // dtype.itemsize)

                self._phdr, self._pbag = table(b"phdr", _PHDR), table(b"pbag", _BAG)
                self._pmod, self._pgen = table(b"pmod", _MOD), table(b"pgen", _GEN)
                self._inst, self._ibag = table(b"inst", _INST), table(b"ibag", _BAG)
                self._imod, self._igen = table(b"imod", _MOD), table(b"igen", _GEN)
                self._shdr = table(b"shdr", _SHDR)
        except OSError as exc:
            raise SampleError(f"cannot read {path}: {exc}") from None
        if len(self._phdr) < 2 or len(self._inst) < 1 or len(self._shdr) < 1:
            raise SampleError(f"{path}: the SoundFont has no presets")
        if np.any(self._shdr["type"][:-1] & 0x10):
            raise SampleError(f"{path}: the SoundFont holds compressed samples (SF3), which are not supported")
        st = os.stat(path)
        self._key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
        self.ignored_modulators = sum(
            1 for src, _d, _a, amt, _t in self._imod[:-1].tolist() + self._pmod[:-1].tolist()
            if not (_usable_source(src) and _usable_source(amt) and src != 0))
        presets = [Preset(_name(bytes(r["name"])), int(r["bank"]), int(r["preset"]), i)
                   for i, r in enumerate(self._phdr[:-1])]
        self.presets = sorted(presets, key=lambda p: (p.bank, p.program, p.index))

    # ----------------------------------------------------------------------------- lookup
    def find(self, preset) -> Preset | None:
        """A preset by ``(bank, program)``, ``"bank:program"``, a program number (bank 0), or
        name (exact, then the start of a name, then a part of one; case does not matter)."""
        if isinstance(preset, Preset):
            return preset
        bank = prog = None
        if isinstance(preset, (tuple, list)) and len(preset) == 2:
            bank, prog = int(preset[0]), int(preset[1])
        elif isinstance(preset, (int, np.integer)) and not isinstance(preset, bool):
            bank, prog = 0, int(preset)
        elif isinstance(preset, str):
            s = preset.strip()
            parts = s.split(":")
            if len(parts) == 2 and all(p.strip().lstrip("-").isdigit() for p in parts):
                bank, prog = int(parts[0]), int(parts[1])
            elif s.isdigit():
                bank, prog = 0, int(s)
            else:
                low = s.lower()
                for test in (lambda n: n == low, lambda n: n.startswith(low), lambda n: low in n):
                    for p in self.presets:
                        if test(p.name.lower()):
                            return p
                return None
        if bank is None:
            return None
        for p in self.presets:
            if p.bank == bank and p.program == prog:
                return p
        return None

    # --------------------------------------------------------------------------- building
    def _zones(self, bags, gens, mods, first: int, last: int, terminal: int):
        """(global generators, global modulators, [(generators, modulators)]) of the bags
        [first, last); a zone's generators are {operator: amount or (lo, hi)}, its modulators
        {(source, destination, amount source, transform): amount}."""
        glob, glob_mods, zones = {}, {}, []
        for b in range(first, min(last, len(bags) - 1)):
            g0, g1 = int(bags["gen"][b]), int(bags["gen"][b + 1])
            m0, m1 = int(bags["mod"][b]), int(bags["mod"][b + 1])
            d = {}
            for oper, amount in zip(gens["oper"][g0:g1].tolist(), gens["amount"][g0:g1].tolist()):
                if oper in (G_KEYRANGE, G_VELRANGE):
                    d[oper] = (amount & 0xFF, (amount >> 8) & 0xFF)
                else:
                    d[oper] = amount
            zmods = {(m[0], m[1], m[3], m[4]): m[2] for m in mods[m0:m1].tolist()} if m1 > m0 else {}
            if terminal in d:
                zones.append((d, zmods))
            elif b == first:
                glob, glob_mods = d, zmods
        return glob, glob_mods, zones

    def _sample(self, sid: int) -> Sample | None:
        smp = self._samples.get(sid)
        if smp is not None:
            return smp
        h = self._shdr[sid]
        start, end = int(h["start"]), int(h["end"])
        total = self._smpl_bytes // 2
        end = min(end, total)
        if int(h["type"]) & 0x8000 or end - start < 4 or int(h["rate"]) < 400:
            return None
        path, off, frames = self.path, self._smpl_off + 2 * start, end - start

        def read(a: int, b: int) -> np.ndarray:
            a, b = max(0, int(a)), max(0, int(b))
            out = np.zeros((max(0, b - a), 1), np.float32)
            m = max(0, min(b, frames) - a)
            if m:
                with open(path, "rb") as fh:
                    fh.seek(off + 2 * a)
                    raw = fh.read(2 * m)
                x = np.frombuffer(raw, dtype="<i2", count=len(raw) // 2)
                out[: len(x), 0] = x.astype(np.float32) / 32768.0
            return out

        pitch = int(h["pitch"])
        smp = Sample(key=f"{self._key}|{sid}", rate=int(h["rate"]), frames=frames, channels=1, read=read,
                     name=_name(bytes(h["name"])), root=pitch if pitch <= 127 else 60, tune=float(h["correction"]))
        self._samples[sid] = smp
        return smp

    def instrument(self, preset) -> SampleInstrument:
        """The playable instrument of a preset (see :meth:`find` for how to name one)."""
        p = self.find(preset)
        if p is None:
            raise SampleError(f"{self.path}: no preset {preset!r}")
        with self._lock:
            inst = self._built.get(p.index)
            if inst is None:
                inst = self._build(p)
                self._built[p.index] = inst
            return inst

    def _build(self, p: Preset) -> SampleInstrument:
        regions: list[Region] = []
        warnings: list[str] = []
        skipped_rom = 0
        b0, b1 = int(self._phdr["bag"][p.index]), int(self._phdr["bag"][p.index + 1])
        pglob, pglob_mods, pzones = self._zones(self._pbag, self._pgen, self._pmod, b0, b1, G_INSTRUMENT)
        for pz, pmods in pzones:
            ii = pz[G_INSTRUMENT]
            if not 0 <= ii < len(self._inst) - 1:
                continue
            i0, i1 = int(self._inst["bag"][ii]), int(self._inst["bag"][ii + 1])
            iglob, iglob_mods, izones = self._zones(self._ibag, self._igen, self._imod, i0, i1, G_SAMPLE_ID)
            pk = pz.get(G_KEYRANGE, pglob.get(G_KEYRANGE, (0, 127)))
            pv = pz.get(G_VELRANGE, pglob.get(G_VELRANGE, (0, 127)))
            preset_mods = dict(pglob_mods)
            preset_mods.update(pmods)
            for iz, imods in izones:
                ik = iz.get(G_KEYRANGE, iglob.get(G_KEYRANGE, (0, 127)))
                iv = iz.get(G_VELRANGE, iglob.get(G_VELRANGE, (0, 127)))
                lokey, hikey = max(pk[0], ik[0]), min(pk[1], ik[1])
                lovel, hivel = max(pv[0], iv[0]), min(pv[1], iv[1])
                if lokey > hikey or lovel > hivel:
                    continue
                sid = iz[G_SAMPLE_ID]
                if not 0 <= sid < len(self._shdr) - 1:
                    continue
                smp = self._sample(sid)
                if smp is None:
                    skipped_rom += 1
                    continue
                g = dict(_DEFAULTS)
                g.update(iglob)
                g.update(iz)
                added = dict(pglob)
                added.update(pz)
                for oper, amount in added.items():
                    if oper not in _INSTRUMENT_ONLY:
                        g[oper] = g.get(oper, 0) + amount
                inst_mods = dict(iglob_mods)
                inst_mods.update(imods)
                # velocity to attenuation: the default (960) unless the instrument sets its own; a preset adds
                atten = float(inst_mods.pop((_VEL_ATTEN, G_ATTENUATION, 0, 0), 960))
                atten += float(preset_mods.get((_VEL_ATTEN, G_ATTENUATION, 0, 0), 0))
                dynamic = []
                for mods in (inst_mods, preset_mods):
                    for (src, dest, amt_src, _trans), amount in mods.items():
                        if (src, dest, amt_src) == (_VEL_ATTEN, G_ATTENUATION, 0) and mods is preset_mods:
                            continue
                        # the specification's velocity-to-filter default, restated with its switch: off
                        if dest == G_FC and (src & 0xFF) == 2 and amt_src and amount == -2400:
                            continue
                        if amount and src and _usable_source(src) and _usable_source(amt_src):
                            dynamic.append((src, dest, amt_src, amount))
                reg = self._region(smp, sid, g, lokey, hikey, lovel, hivel, atten)
                if reg is None:
                    continue
                if dynamic:
                    reg.resolve = self._resolver(smp, sid, g, lokey, hikey, lovel, hivel, atten, dynamic)
                regions.append(reg)
        if skipped_rom:
            warnings.append(f"{skipped_rom} zones use ROM or empty samples and are silent")
        if not regions:
            warnings.append("the preset has no playable zones")
        return SampleInstrument(name=p.name, regions=regions, drums=(p.bank == 128), warnings=warnings,
                                info={"format": "sf2", "file": self.path, "preset": p.name, "bank": p.bank,
                                      "program": p.program})

    def _resolver(self, smp, sid, g, lokey, hikey, lovel, hivel, atten, dynamic):
        """A function (key, velocity) -> Region with the zone's key and velocity modulators
        added to its generators."""
        cache: dict = {}

        def resolve(key: int, vel: int) -> Region:
            k = g[G_KEYNUM] if 0 <= g[G_KEYNUM] <= 127 else key
            v = g[G_VELOCITY] if 1 <= g[G_VELOCITY] <= 127 else vel
            reg = cache.get((k, v))
            if reg is None:
                g2 = dict(g)
                for src, dest, amt_src, amount in dynamic:
                    g2[dest] = g2.get(dest, 0) + amount * _source(src, k, v) * _source(amt_src, k, v)
                reg = self._region(smp, sid, g2, lokey, hikey, lovel, hivel, atten)
                if len(cache) < 4096:
                    cache[(k, v)] = reg
            return reg

        return resolve

    def _region(self, smp: Sample, sid: int, g: dict, lokey, hikey, lovel, hivel, atten: float) -> Region | None:
        h = self._shdr[sid]
        s0 = int(h["start"])
        start = g.get(G_START, 0) + 32768 * g.get(G_START_COARSE, 0)
        end = smp.frames + g.get(G_END, 0) + 32768 * g.get(G_END_COARSE, 0)
        ls = int(h["loop_start"]) - s0 + g.get(G_LOOP_START, 0) + 32768 * g.get(G_LOOP_START_COARSE, 0)
        le = int(h["loop_end"]) - s0 + g.get(G_LOOP_END, 0) + 32768 * g.get(G_LOOP_END_COARSE, 0)
        start = max(0, min(int(start), smp.frames - 1))
        end = max(start + 1, min(int(end), smp.frames))
        ls, le = int(ls), int(le)
        mode = int(g.get(G_SAMPLE_MODES, 0)) & 3
        loop_mode = {1: "loop_continuous", 3: "loop_sustain"}.get(mode, "no_loop")
        if loop_mode != "no_loop" and not (start <= ls < le - 1 and le <= end):
            loop_mode = "no_loop"
        root = int(g[G_ROOT_KEY]) if 0 <= g[G_ROOT_KEY] <= 127 else (smp.root if smp.root is not None else 60)
        sustain_cb = max(0, g.get(G_SUSTAIN_VOLENV, 0))
        ampeg = Envelope(delay=_seconds(g[G_DELAY_VOLENV]), attack=_seconds(g[G_ATTACK_VOLENV]),
                         hold=_seconds(g[G_HOLD_VOLENV]), decay=_seconds(g[G_DECAY_VOLENV]),
                         sustain=0.0 if sustain_cb >= 960 else 10.0 ** (-sustain_cb / 200.0),
                         release=_seconds(g[G_RELEASE_VOLENV]), shape="db")
        reg = Region(sample=smp, lokey=lokey, hikey=hikey, lovel=lovel, hivel=hivel, root=int(root),
                     tune=100.0 * g.get(G_COARSE_TUNE, 0) + g.get(G_FINE_TUNE, 0) + smp.tune,
                     keytrack=float(g[G_SCALE_TUNING]), start=start, end=end, loop_mode=loop_mode,
                     loop_start=ls, loop_end=le, volume=-0.04 * max(0, g.get(G_ATTENUATION, 0)),
                     pan=max(-1.0, min(1.0, g.get(G_PAN, 0) / 500.0)), ampeg=ampeg,
                     group=int(g.get(G_EXCLUSIVE_CLASS, 0)), off_by=int(g.get(G_EXCLUSIVE_CLASS, 0)))
        if reg.group:
            reg.off_time = 0.01
        reg.ampeg_key2hold = float(g.get(G_KEY_HOLD_VOLENV, 0))
        reg.ampeg_key2decay = float(g.get(G_KEY_DECAY_VOLENV, 0))
        if atten != 960:  # velocity to attenuation with another amount than the default
            reg.vel_curve = 2.0 * max(atten, 0.0) / 960.0
            reg.amp_veltrack = 1.0 if atten > 0 else 0.0
        # filter (a cutoff at the top of the range with no downward envelope means no filter)
        fc_cents = min(max(g[G_FC], 1500), 13500)
        q_db = max(0.0, g.get(G_Q, 0) / 10.0)
        env_fc = g.get(G_MODENV_FC, 0)
        if fc_cents < 13500 or env_fc < 0:
            reg.cutoff = _hz(fc_cents)
            reg.resonance = q_db - 3.01  # 0 units = the flat (Butterworth) response
            reg.fil_type = "lpf_2p"
        need_modenv = env_fc or g.get(G_MODENV_PITCH, 0)
        if need_modenv:
            modeg = Envelope(delay=_seconds(g[G_DELAY_MODENV]), attack=_seconds(g[G_ATTACK_MODENV]),
                             hold=_seconds(g[G_HOLD_MODENV]), decay=_seconds(g[G_DECAY_MODENV]),
                             sustain=max(0.0, 1.0 - max(0, g.get(G_SUSTAIN_MODENV, 0)) / 1000.0),
                             release=_seconds(g[G_RELEASE_MODENV]), shape="lin")
            if env_fc:
                reg.fileg, reg.fileg_depth = modeg, float(env_fc)
            if g.get(G_MODENV_PITCH, 0):
                reg.pitcheg, reg.pitcheg_depth = modeg, float(g[G_MODENV_PITCH])
        vib = g.get(G_VIBLFO_PITCH, 0)
        modp = g.get(G_MODLFO_PITCH, 0)
        if vib:
            reg.vib_depth, reg.vib_freq = float(vib), _hz(g.get(G_FREQ_VIBLFO, 0))
            reg.vib_delay = _seconds(g[G_DELAY_VIBLFO])
        elif modp:
            reg.vib_depth, reg.vib_freq = float(modp), _hz(g.get(G_FREQ_MODLFO, 0))
            reg.vib_delay = _seconds(g[G_DELAY_MODLFO])
        if g.get(G_MODLFO_VOL, 0):
            reg.trem_depth, reg.trem_freq = g[G_MODLFO_VOL] / 10.0, _hz(g.get(G_FREQ_MODLFO, 0))
            reg.trem_delay = _seconds(g[G_DELAY_MODLFO])
        if 0 <= g[G_KEYNUM] <= 127:
            reg.fixed_key = int(g[G_KEYNUM])
        if 1 <= g[G_VELOCITY] <= 127:
            reg.fixed_vel = int(g[G_VELOCITY])
        return reg


_FONTS: dict[str, SoundFont] = {}
_FONTS_LOCK = threading.Lock()


def load_sf2(path: str) -> SoundFont:
    """Parse a SoundFont (cached per file and modification time). Raises
    :class:`~synth2.arranger.samples.SampleError` when it cannot be read."""
    path = os.path.abspath(os.fspath(path))
    try:
        st = os.stat(path)
    except OSError as exc:
        raise SampleError(f"cannot read {path}: {exc}") from None
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    with _FONTS_LOCK:
        font = _FONTS.get(key)
        if font is None:
            font = SoundFont(path)
            if len(_FONTS) > 8:
                _FONTS.pop(next(iter(_FONTS)))
            _FONTS[key] = font
        return font
