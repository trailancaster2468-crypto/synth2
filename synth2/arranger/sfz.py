"""SFZ (``.sfz``) reader for the sample player.

Read: the ``<control>``, ``<global>``, ``<master>``, ``<group>`` and ``<region>`` headers
with inheritance, ``#include`` and ``#define``, and the opcodes in :data:`SUPPORTED`:
sample mapping (key, velocity, round robin, random, keyswitch and controller ranges at
their default values), pitch, loops, offsets, level, pan, velocity tracking and velocity /
key crossfades, the amplitude envelope, the filter with its envelope and tracking, the
pitch envelope and LFO, tremolo, choke groups, release triggers.  Controller modulation of
envelope times, filter, tuning and level (``ampeg_release_oncc72`` ...) is applied once, at
the controller's default value (``set_ccN``, ``set_hdccN``).

Anything else is ignored, never an error: the instrument's ``warnings`` name the opcodes
that were ignored (controller modulation such as ``*_oncc`` is counted, not listed).
Samples may be WAV, or FLAC / Ogg when the ``soundfile`` package is importable; a region
whose sample is missing or unreadable is left out and counted.
"""
from __future__ import annotations

import math
import os
import re
import threading

from .sampler import FILTER_TYPES, Envelope, Region, SampleInstrument
from .samples import Sample, SampleError, open_sample

SUPPORTED = (
    "sample", "default_path", "note_offset", "octave_offset", "set_ccN", "key", "lokey", "hikey", "lovel", "hivel",
    "pitch_keycenter", "pitch_keytrack", "tune", "transpose", "volume", "pan", "amp_veltrack", "amp_keytrack",
    "amp_keycenter", "ampeg_delay", "ampeg_attack", "ampeg_hold", "ampeg_decay", "ampeg_sustain", "ampeg_release",
    "ampeg_vel2attack", "loop_mode", "loop_start", "loop_end", "offset", "end", "delay", "group", "off_by", "off_mode",
    "off_time", "polyphony", "note_polyphony", "seq_length", "seq_position", "lorand", "hirand", "trigger", "rt_decay",
    "cutoff", "resonance", "fil_type", "fil_veltrack", "fil_keytrack", "fil_keycenter", "fileg_depth", "fileg_delay",
    "fileg_attack", "fileg_hold", "fileg_decay", "fileg_sustain", "fileg_release", "pitcheg_depth", "pitcheg_delay",
    "pitcheg_attack", "pitcheg_hold", "pitcheg_decay", "pitcheg_sustain", "pitcheg_release", "pitchlfo_freq",
    "pitchlfo_depth", "pitchlfo_delay", "pitchlfo_fade", "amplfo_freq", "amplfo_depth", "amplfo_delay",
    "xfin_lovel", "xfin_hivel", "xfout_lovel", "xfout_hivel", "xfin_lokey", "xfin_hikey", "xfout_lokey", "xfout_hikey",
    "sw_lokey", "sw_hikey", "sw_default", "sw_last", "loccN", "hiccN", "set_hdccN", "amp_velcurve_N", "amplitude",
    "xfin_loccN", "xfin_hiccN", "xfout_loccN", "xfout_hiccN",
)
_SILENT = re.compile(r"^(region_label|group_label|master_label|global_label|sw_label|label_cc\d+|label_key\d+|"
                     r"hint_\w+|sample_quality|bend_up|bend_down|bend_step|sustain_cc|sostenuto_cc|sustain_sw|"
                     r"sostenuto_sw|note_selfmask|output|script|oscillator\w*|md5|global_volume|master_volume|"
                     r"group_volume|count|loop_count|sw_lokey|sw_hikey|lochan|hichan|lobend|hibend|lobpm|hibpm|"
                     r"lochanaft|hichanaft|lopolyaft|hipolyaft|lotimer|hitimer|ampeg_dynamic|bend_smooth|"
                     r"amp_random|pitch_random|fil_random|offset_random|delay_random)$")
_CC = re.compile(r"(_oncc|_cc|_curvecc|_smoothcc|_stepcc|_chanaft|_polyaft)\d*$|^(cc|curve)\w*$|cc\d+$")
# version 2 opcode families that are reported as a family, not opcode by opcode
_FAMILIES = ((re.compile(r"^lfo\d+_"), "lfoN_* (version 2 LFOs)"), (re.compile(r"^eg\d+_"), "egN_* (version 2 envelopes)"),
             (re.compile(r"^eq\d+_"), "eqN_* (equaliser)"))
_DIRECT = re.compile(r"(lo|hi|set_|set_hd)cc\d+|amp_velcurve_\d+|xf(in|out)_(lo|hi)cc\d+")
_TOKEN = re.compile(r"<(\w+)>|([A-Za-z_][\w.#$]*)=")
_NOTE = re.compile(r"^([a-gA-G])([#b]?)(-?\d+)$")
_SEMIS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
_CC_DEFAULTS = {7: 100, 10: 64, 11: 127}
_MAX_INCLUDE_DEPTH = 8


def _strip_comments(text: str) -> str:
    return "\n".join(line.split("//", 1)[0] for line in text.replace("\r", "\n").split("\n"))


def _read_text(path: str) -> str:
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", "replace")


def _expand(path: str, root: str, defines: dict, depth: int, problems: list[str]) -> str:
    """The text of an SFZ file with comments removed and ``#include`` / ``#define`` applied."""
    try:
        text = _strip_comments(_read_text(path))
    except OSError as exc:
        problems.append(f"cannot read {os.path.basename(path)}: {exc}")
        return ""
    out = []
    for line in text.split("\n"):
        s = line.strip()
        m = re.match(r'#define\s+(\$\w+)\s+(\S+)', s)
        if m:
            defines[m.group(1)] = m.group(2)
            continue
        if "$" in line and defines:
            for name in sorted(defines, key=len, reverse=True):
                line = line.replace(name, defines[name])
        if "#include" in line:  # anywhere on a line; the included text takes its place
            pieces = re.split(r'#include\s+"([^"]+)"', line)
            for j in range(1, len(pieces), 2):
                if depth >= _MAX_INCLUDE_DEPTH:
                    problems.append("#include nested too deep")
                    pieces[j] = ""
                    continue
                inc = _find_file(root, pieces[j]) or _find_file(os.path.dirname(path), pieces[j])
                if inc is None:
                    problems.append(f"#include {pieces[j]!r} not found")
                    pieces[j] = ""
                else:
                    pieces[j] = "\n" + _expand(inc, root, defines, depth + 1, problems) + "\n"
            line = "".join(pieces)
        out.append(line)
    return "\n".join(out)


_DIR_CACHE: dict[str, dict[str, str]] = {}


def _find_file(base: str, rel: str) -> str | None:
    """A file named relative to ``base`` with either kind of slash; when the exact name does
    not exist the name is matched without regard to case."""
    rel = rel.strip().replace("\\", "/")
    full = os.path.normpath(os.path.join(base, rel))
    if os.path.isfile(full):
        return full
    cur = base
    for part in [p for p in rel.split("/") if p not in ("", ".")]:
        if part == "..":
            cur = os.path.dirname(cur)
            continue
        names = _DIR_CACHE.get(cur)
        if names is None:
            try:
                names = {n.lower(): n for n in os.listdir(cur)}
            except OSError:
                return None
            if len(_DIR_CACHE) > 512:
                _DIR_CACHE.clear()
            _DIR_CACHE[cur] = names
        hit = names.get(part.lower())
        if hit is None:
            return None
        cur = os.path.join(cur, hit)
    return cur if os.path.isfile(cur) else None


def parse(text: str) -> tuple[dict, list[dict]]:
    """(control opcodes, regions): each region is the dict of its opcodes with the opcodes
    of its ``<group>``, ``<master>`` and ``<global>`` underneath."""
    control: dict = {}
    levels = {"global": {}, "master": {}, "group": {}}
    regions: list[dict] = []
    current: dict | None = None
    header = None
    for line in text.split("\n"):
        tokens = list(_TOKEN.finditer(line))
        for i, m in enumerate(tokens):
            if m.group(1):
                header = m.group(1).lower()
                if header == "region":
                    # the <control> values in force here: a file may change default_path between groups
                    current = {k: control[k] for k in ("default_path", "note_offset", "octave_offset") if k in control}
                    current.update(levels["global"])
                    current.update(levels["master"])
                    current.update(levels["group"])
                    regions.append(current)
                elif header == "global":
                    levels.update(**{"global": {}, "master": {}, "group": {}})
                    current = levels["global"]
                elif header == "master":
                    levels.update(master={}, group={})
                    current = levels["master"]
                elif header == "group":
                    levels["group"] = {}
                    current = levels["group"]
                elif header == "control":
                    current = control
                else:
                    current = None  # <curve>, <effect>, <midi>, <sample>: not read
                continue
            if current is None:
                continue
            stop = tokens[i + 1].start() if i + 1 < len(tokens) else len(line)
            name = m.group(2).lower()
            current.pop(name, None)  # a later opcode also comes later in the dict: order decides key vs lokey
            current[name] = line[m.end():stop].strip()
    return control, regions


def _note(v, offset: int = 0) -> int | None:
    s = str(v).strip()
    m = _NOTE.match(s)
    if m:
        semi = _SEMIS[m.group(1).lower()] + (1 if m.group(2) == "#" else -1 if m.group(2) == "b" else 0)
        return 12 * (int(m.group(3)) + 1) + semi + offset
    try:
        return int(float(s)) + offset
    except ValueError:
        return None


def _f(d: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(d[key])
    except (KeyError, ValueError):
        return default


# opcodes whose controller modulation (<opcode>_onccN, <opcode>ccN) is applied at the controller's default value
_CC_TARGETS = ("ampeg_delay", "ampeg_attack", "ampeg_hold", "ampeg_decay", "ampeg_sustain", "ampeg_release",
               "fileg_depth", "fileg_attack", "fileg_decay", "fileg_sustain", "fileg_release", "pitcheg_depth",
               "amp_veltrack", "cutoff", "resonance", "tune", "pitch", "volume", "pan", "offset", "delay",
               "pitchlfo_depth", "amplfo_depth", "fil_veltrack")
_CC_MOD = re.compile(r"^(" + "|".join(_CC_TARGETS) + r")(?:_oncc|_cc|cc)(\d+)$")


def _cc_curve(x: float, index: int) -> float:
    """The built-in controller curves 0..6 on 0..1."""
    x = min(1.0, max(0.0, x))
    return (x, 2.0 * x - 1.0, 1.0 - x, 1.0 - 2.0 * x, x * x, x ** 0.5, (1.0 - x) ** 0.5)[index] if 0 <= index <= 6 else x


def _fold_controllers(d: dict, cc: dict) -> set[str]:
    """Add each controller modulation of a region, at the controller's default value, to the
    opcode it modulates (in place). Returns the modulation opcodes that were applied."""
    used = set()
    for k in list(d):
        m = _CC_MOD.match(k)
        if not m:
            continue
        base, number = m.group(1), int(m.group(2))
        try:
            depth = float(d[k])
        except ValueError:
            continue
        used.add(k)
        value = cc.get(number, 0.0) / 127.0
        curve = int(_f(d, f"{base}_curvecc{number}", 0))
        amount = depth * _cc_curve(value, curve)
        if not amount:
            continue
        if base == "cutoff":  # modulated in cents
            if "cutoff" in d:
                d["cutoff"] = repr(_f(d, "cutoff", 20000.0) * 2.0 ** (amount / 1200.0))
        elif base == "pitch":
            d["tune"] = repr(_f(d, "tune") + amount)
        else:
            default = {"ampeg_sustain": 100.0, "fileg_sustain": 100.0, "amp_veltrack": 100.0,
                       "ampeg_release": 0.001}.get(base, 0.0)
            d[base] = repr(_f(d, base, default) + amount)
    return used


def _envelope(d: dict, prefix: str, release_default: float) -> Envelope:
    return Envelope(delay=max(0.0, _f(d, prefix + "_delay")), attack=max(0.0, _f(d, prefix + "_attack")),
                    hold=max(0.0, _f(d, prefix + "_hold")), decay=max(0.0, _f(d, prefix + "_decay")),
                    sustain=min(1.0, max(0.0, _f(d, prefix + "_sustain", 100.0) / 100.0)),
                    release=max(0.0, _f(d, prefix + "_release", release_default)), shape="exp")


def load_sfz_text(text: str, base: str, name: str = "sfz", source: str = "", cc: dict | None = None,
                  keyswitch=None) -> SampleInstrument:
    """Build an instrument from SFZ text whose sample paths are relative to ``base``.

    ``cc`` sets controller values ({number: 0..127}) over the file's defaults (they select
    regions and scale the controller modulation that is applied); ``keyswitch`` (a note)
    selects the articulation of a keyswitched instrument instead of its default."""
    control, raw = parse(text)
    warnings: list[str] = []
    ignored: set[str] = set()
    cc_ignored: set[str] = set()
    missing: list[str] = []
    unreadable: list[str] = []
    skipped = {"legato": 0, "keyswitch": 0, "controller": 0, "generator": 0}
    offset = int(_f(control, "note_offset")) + 12 * int(_f(control, "octave_offset"))
    given_cc = cc or {}
    cc = dict(_CC_DEFAULTS)
    for k, v in control.items():
        m = re.fullmatch(r"set_(hd)?cc(\d+)", k)
        if m:
            cc[int(m.group(2))] = _f(control, k) * (127.0 if m.group(1) else 1.0)
    for k, v in given_cc.items():
        try:
            cc[int(k)] = float(v)
        except (TypeError, ValueError):
            pass
    sw_default = _note(keyswitch, 0) if keyswitch is not None else None
    for d in raw:
        if sw_default is None and "sw_default" in d:
            sw_default = _note(d["sw_default"], offset)
            break
    if sw_default is None:
        lasts = [_note(d["sw_last"], offset) for d in raw if "sw_last" in d]
        lasts = [x for x in lasts if x is not None]
        sw_default = min(lasts) if lasts else None
    samples: dict[str, Sample | None] = {}
    regions: list[Region] = []
    known = set(SUPPORTED) | {"note_offset", "octave_offset", "default_path"}
    for d in raw:
        # controllers at their defaults; the key and velocity "controllers" (133, 131) at the region's middle
        region_cc = dict(cc)
        lo_k, hi_k = _note(d.get("lokey", d.get("key", 0)), offset), _note(d.get("hikey", d.get("key", 127)), offset)
        region_cc[133] = ((lo_k if lo_k is not None else 0) + (hi_k if hi_k is not None else 127)) / 2.0
        region_cc[131] = (_f(d, "lovel", 0) + _f(d, "hivel", 127)) / 2.0
        folded = _fold_controllers(d, region_cc)
        for k in d:
            if k in known or k in folded or _DIRECT.fullmatch(k) or _SILENT.match(k):
                continue
            family = next((label for rx, label in _FAMILIES if rx.match(k)), None)
            if family:
                ignored.add(family)
            elif _CC.search(k):
                cc_ignored.add(k)
            else:
                ignored.add(k)
        spec = d.get("sample")
        if not spec:
            continue
        if spec.startswith("*"):
            if spec.lower() != "*silence":
                skipped["generator"] += 1
            continue
        # regions that need a controller away from its default, another keyswitch or a legato transition
        ok = True
        xf_gain = 1.0
        xf_cc: dict = {}
        curve = []
        for k, v in d.items():
            m = re.fullmatch(r"(lo|hi)cc(\d+)", k)
            if m:
                cur, lim = cc.get(int(m.group(2)), 0), _f(d, k)
                if (m.group(1) == "lo" and cur < lim) or (m.group(1) == "hi" and cur > lim):
                    ok = False
                continue
            m = re.fullmatch(r"xf(in|out)_(lo|hi)cc(\d+)", k)
            if m:
                xf_cc.setdefault((m.group(1), int(m.group(3))), {})[m.group(2)] = _f(d, k)
                continue
            m = re.fullmatch(r"amp_velcurve_(\d+)", k)
            if m and 0 <= int(m.group(1)) <= 127:
                curve.append((int(m.group(1)), min(1.0, max(0.0, _f(d, k)))))
        for (direction, number), lim in xf_cc.items():  # controller crossfades at the controller's default value
            cur, lo, hi = cc.get(number, 0), lim.get("lo", 0.0), lim.get("hi", 127.0 if direction == "out" else 0.0)
            x = 1.0 if cur >= hi else 0.0 if cur <= lo or hi <= lo else (cur - lo) / (hi - lo)
            xf_gain *= x ** 0.5 if direction == "in" else (1.0 - x) ** 0.5
        if xf_gain <= 1e-4:
            ok = False
        if not ok:
            skipped["controller"] += 1
            continue
        if "sw_last" in d and _note(d["sw_last"], offset) != sw_default:
            skipped["keyswitch"] += 1
            continue
        if any(k in d for k in ("sw_down", "sw_up", "sw_previous")):
            skipped["keyswitch"] += 1
            continue
        trig = d.get("trigger", "attack").lower()
        if trig == "legato":
            skipped["legato"] += 1
            continue
        offset = int(_f(d, "note_offset")) + 12 * int(_f(d, "octave_offset"))
        rel = d.get("default_path", "").replace("\\", "/") + spec.replace("\\", "/")
        if rel not in samples:
            path = _find_file(base, rel)
            if path is None:
                samples[rel] = None
                missing.append(spec)
            else:
                try:
                    samples[rel] = open_sample(path)
                except SampleError as exc:
                    samples[rel] = None
                    unreadable.append(str(exc))
        smp = samples[rel]
        if smp is None:
            continue
        lokey, hikey, root = 0, 127, None
        for k, v in d.items():  # in the order written: the later of key / lokey / hikey / pitch_keycenter wins
            if k == "key":
                n = _note(v, offset)
                if n is not None:
                    lokey = hikey = root = n
            elif k in ("lokey", "hikey"):
                n = _note(v, offset)
                if n is not None:
                    lokey, hikey = (n, hikey) if k == "lokey" else (lokey, n)
            elif k == "pitch_keycenter":
                root = None if v.lower() == "sample" else _note(v, offset)
        if root is None:
            root = smp.root if smp.root is not None else 60
        frames = smp.frames
        end = frames
        if "end" in d:
            e = int(_f(d, "end", frames - 1))
            if e < 0:
                continue
            end = min(frames, e + 1)
        mode = d.get("loop_mode", "").lower()
        ls, le = smp.loop if smp.loop is not None else (0, end)
        if "loop_start" in d:
            ls = int(_f(d, "loop_start"))
        if "loop_end" in d:
            le = int(_f(d, "loop_end")) + 1  # loop_end is the last frame of the loop
        if mode not in ("no_loop", "one_shot", "loop_continuous", "loop_sustain"):
            mode = "loop_continuous" if smp.loop is not None or "loop_start" in d else "no_loop"
        if mode.startswith("loop") and not (0 <= ls < le - 1 and le <= end):
            mode = "no_loop"
        release = _f(d, "ampeg_release", 0.001)
        reg = Region(
            sample=smp, lokey=max(0, lokey), hikey=min(127, hikey), lovel=int(_f(d, "lovel", 0)),
            hivel=int(_f(d, "hivel", 127)), root=int(root),
            tune=_f(d, "tune") + 100.0 * _f(d, "transpose") + (smp.tune if "pitch_keycenter" not in d else 0.0),
            keytrack=_f(d, "pitch_keytrack", 100.0), start=max(0, int(_f(d, "offset"))), end=end, loop_mode=mode,
            loop_start=ls, loop_end=le, pan=max(-1.0, min(1.0, _f(d, "pan") / 100.0)),
            volume=_f(d, "volume") + 20.0 * math.log10(max(1e-4, xf_gain * _f(d, "amplitude", 100.0) / 100.0)),
            amp_veltrack=_f(d, "amp_veltrack", 100.0) / 100.0, amp_keytrack=_f(d, "amp_keytrack"),
            amp_keycenter=_note(d.get("amp_keycenter", 60), 0) or 60, ampeg=_envelope(d, "ampeg", 0.001),
            ampeg_vel2attack=_f(d, "ampeg_vel2attack"), group=int(_f(d, "group")), off_by=int(_f(d, "off_by")),
            seq_length=max(1, int(_f(d, "seq_length", 1))), seq_position=max(1, int(_f(d, "seq_position", 1))),
            lorand=_f(d, "lorand", 0.0), hirand=_f(d, "hirand", 1.0),
            trigger="release" if trig in ("release", "release_key") else "attack", rt_decay=_f(d, "rt_decay"),
            delay=max(0.0, _f(d, "delay")))
        if curve:
            reg.vel_points = tuple(sorted(dict([(0, 0.0), (127, 1.0)] + curve).items()))
        off_mode = d.get("off_mode", "fast").lower()
        if off_mode == "normal":
            reg.off_time = max(0.006, release)
        elif off_mode == "time":
            reg.off_time = max(0.002, _f(d, "off_time", 0.006))
        if reg.group and not reg.off_by and (_f(d, "polyphony", 0) == 1 or _f(d, "note_polyphony", 0) == 1):
            reg.off_by = reg.group  # one voice at a time in the group
        if "cutoff" in d:
            ftype = d.get("fil_type", "lpf_2p").lower()
            if ftype in FILTER_TYPES:
                reg.cutoff = max(16.0, _f(d, "cutoff", 20000.0))
                reg.resonance, reg.fil_type = _f(d, "resonance"), ftype
                reg.fil_veltrack, reg.fil_keytrack = _f(d, "fil_veltrack"), _f(d, "fil_keytrack")
                reg.fil_keycenter = _note(d.get("fil_keycenter", 60), 0) or 60
                if _f(d, "fileg_depth"):
                    reg.fileg, reg.fileg_depth = _envelope(d, "fileg", 0.0), _f(d, "fileg_depth")
            else:
                ignored.add(f"fil_type={ftype}")
        if _f(d, "pitcheg_depth"):
            reg.pitcheg, reg.pitcheg_depth = _envelope(d, "pitcheg", 0.0), _f(d, "pitcheg_depth")
        if _f(d, "pitchlfo_depth"):
            reg.vib_depth, reg.vib_freq = _f(d, "pitchlfo_depth"), _f(d, "pitchlfo_freq", 5.0)
            reg.vib_delay, reg.vib_fade = _f(d, "pitchlfo_delay"), _f(d, "pitchlfo_fade")
        if _f(d, "amplfo_depth"):
            reg.trem_depth, reg.trem_freq = _f(d, "amplfo_depth"), _f(d, "amplfo_freq", 5.0)
            reg.trem_delay = _f(d, "amplfo_delay")
        if any(k in d for k in ("xfin_lovel", "xfin_hivel", "xfout_lovel", "xfout_hivel")):
            reg.xf_vel = (_f(d, "xfin_lovel", 0), _f(d, "xfin_hivel", 0), _f(d, "xfout_lovel", 127),
                          _f(d, "xfout_hivel", 127))
        if any(k in d for k in ("xfin_lokey", "xfin_hikey", "xfout_lokey", "xfout_hikey")):
            g = lambda k, dflt: float(_note(d[k], offset)) if k in d and _note(d[k], offset) is not None else dflt  # noqa: E731
            reg.xf_key = (g("xfin_lokey", 0.0), g("xfin_hikey", 0.0), g("xfout_lokey", 127.0), g("xfout_hikey", 127.0))
        if reg.lokey <= reg.hikey and reg.lovel <= reg.hivel:
            regions.append(reg)
    if missing:
        warnings.append(f"{len(missing)} samples not found (first: {missing[0]})")
    if unreadable:
        warnings.append(f"{len(unreadable)} samples could not be read ({unreadable[0]})")
    if ignored:
        names = sorted(ignored)
        warnings.append(f"{len(names)} opcodes not supported and ignored: {', '.join(names[:12])}"
                        + (" ..." if len(names) > 12 else ""))
    if cc_ignored:
        warnings.append(f"{len(cc_ignored)} controller modulations ignored (an arrangement has no controllers; "
                        "envelope, filter, pitch and level modulations are applied at the controllers' defaults)")
    for what, n in skipped.items():
        if n:
            warnings.append({"legato": f"{n} legato-transition regions left out",
                             "keyswitch": f"{n} regions of other keyswitch articulations left out",
                             "controller": f"{n} regions that need a controller away from its default left out",
                             "generator": f"{n} oscillator regions (sample=*sine ...) left out"}[what])
    if not regions:
        warnings.append("no playable regions")
    keys = {k for r in regions for k in (r.lokey, r.hikey)}
    drums = bool(regions) and all(r.loop_mode == "one_shot" or r.lokey == r.hikey for r in regions) and \
        any(r.loop_mode == "one_shot" for r in regions) and min(keys) >= 24
    return SampleInstrument(name=name, regions=regions, drums=drums, warnings=warnings,
                            info={"format": "sfz", "file": source, "regions": len(regions)})


_BUILT: dict[str, SampleInstrument] = {}
_LOCK = threading.Lock()


def bank_defines(path: str) -> dict:
    """``#define`` values an ARIA bank file (``*.bank.xml`` next to or above the instrument)
    gives its programs, e.g. ``{"$sample_dir": "../Samples"}``."""
    out = {}
    folder = os.path.dirname(os.path.abspath(path))
    for d in (folder, os.path.dirname(folder)):
        try:
            names = [n for n in sorted(os.listdir(d)) if n.lower().endswith(".bank.xml")]
        except OSError:
            continue
        for n in names[:2]:
            try:
                text = _read_text(os.path.join(d, n))
            except OSError:
                continue
            for m in re.finditer(r'<Define\s+name="(\$\w+)"\s+value="([^"]*)"', text):
                out.setdefault(m.group(1), m.group(2))
    return out


def load_sfz(path: str, defines: dict | None = None, cc: dict | None = None, keyswitch=None) -> SampleInstrument:
    """Load an ``.sfz`` file (cached per file, modification time and options). ``defines``
    presets ``#define`` values (``{"$sample_dir": "../Samples"}``; an ARIA bank file next to
    the instrument is read for them too), ``cc`` and ``keyswitch`` as in
    :func:`load_sfz_text`. Raises :class:`~synth2.arranger.samples.SampleError` only when the
    file itself cannot be read."""
    path = os.path.abspath(os.fspath(path))
    try:
        st = os.stat(path)
    except OSError as exc:
        raise SampleError(f"cannot read {path}: {exc}") from None
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}|{sorted((defines or {}).items())}|{sorted((cc or {}).items())}|{keyswitch}"
    with _LOCK:
        inst = _BUILT.get(key)
        if inst is None:
            problems: list[str] = []
            base = os.path.dirname(path)
            start = bank_defines(path)
            start.update(defines or {})
            text = _expand(path, base, start, 0, problems)
            inst = load_sfz_text(text, base, os.path.splitext(os.path.basename(path))[0], path, cc, keyswitch)
            inst.warnings[:0] = problems
            if len(_BUILT) > 24:
                _BUILT.pop(next(iter(_BUILT)))
            _BUILT[key] = inst
        return inst
