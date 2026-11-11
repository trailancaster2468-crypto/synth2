"""From the ``sound`` an arrangement names to a playable sample instrument.

A sound is an installed set id (:mod:`synth2.sounds`), or a path to a ``.sf2`` file, a
``.sfz`` file, a WAV file or a folder of samples (absolute, or relative to the project).
:func:`resolve` never raises for a sound that is missing or unreadable: it returns a
:class:`Resolved` without an instrument and says why, and the arranger plays the
synthesised instrument instead.
"""
from __future__ import annotations

import json
import os
import struct
import threading
from collections import OrderedDict
from dataclasses import dataclass, field, replace

import numpy as np

from . import gm
from .sampler import Region, SampleInstrument, loudness, midi_velocity, render_note
from .samples import AUDIO_EXTS, SampleError

# keys of a sound layer in an arrangement (see FORMAT.md)
SOUND_KEYS = {
    "set", "path", "file", "preset", "tune", "transpose", "gain_db", "pan", "normalize", "attack", "hold", "decay",
    "sustain", "release", "cutoff", "resonance", "keytrack", "vel_to_cutoff", "velocity_curve", "vel_sens",
    "vibrato", "vibrato_rate", "vibrato_delay", "mono", "glide", "key_range", "vel_range", "round_robin", "drift",
    "gate", "cc", "keyswitch", "comment",
}
SOUND_OPTION_DOCS = {
    "set": "id of an installed sound set (synth2 sounds list)",
    "path": "a .sf2 or .sfz file, a WAV file or a folder of samples; absolute, or relative to the project",
    "file": "which file of a set with several (.sf2 or .sfz name, or part of it)",
    "preset": "preset by name, 'bank:program', program number, or a General MIDI name such as 'strings' "
              "(default: the program that fits the instrument's synth preset; kits: bank 128)",
    "tune": "cents added to every note", "transpose": "semitones: which key of the sound a note plays",
    "gain_db": "level trim in dB", "pan": "-1..1 added to the sound's own panning",
    "normalize": "true (default) = level-matched to the synthesised instrument (a kit: every drum to the "
                 "synthesised drum it replaces); false = the samples' own level and the kit's own balance",
    "attack": "amp attack, s", "hold": "amp hold, s", "decay": "amp decay, s", "sustain": "amp sustain 0..1",
    "release": "amp release, s (time to fade out after the key is released)",
    "cutoff": "low-pass cutoff at full velocity, Hz", "resonance": "dB of filter peak",
    "keytrack": "0..1 how much the cutoff follows the key", "vel_to_cutoff": "octaves the cutoff falls at velocity 0",
    "velocity_curve": "loudness = velocity ** this (2 = the formats' default, 1 = linear)",
    "vel_sens": "0..1 how much velocity changes loudness",
    "vibrato": "vibrato depth, semitones", "vibrato_rate": "Hz", "vibrato_delay": "s before the vibrato starts",
    "mono": "true = each note cuts the previous one (default: as the synth preset)",
    "glide": "s of portamento between legato notes (mono)",
    "key_range": "[low, high] notes this layer plays (for splits and layers)",
    "vel_range": "[low, high] MIDI velocities 1..127 this layer plays",
    "round_robin": "false = always the first take of each drum or note",
    "drift": "cents of random detuning per note (seeded)",
    "gate": "kits: seconds a SoundFont drum key is held (one-shot samples ignore it)",
    "cc": "SFZ: controller values {number: 0..127} over the instrument's defaults (blend, tone, articulation ...)",
    "keyswitch": "SFZ: the note of the articulation to play in a keyswitched instrument (default: the file's own)",
}
DRUM_GATE = 0.3  # s a kit key is held when the kit's samples are not one-shots (SoundFont kits)


@dataclass
class Resolved:
    """What a sound layer resolved to. ``instrument`` is None when it could not be loaded;
    ``error`` then says why in one sentence. ``label`` names the sound for reports."""

    instrument: SampleInstrument | None
    label: str = ""
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    credits: list[str] = field(default_factory=list)
    options: dict = field(default_factory=dict)
    voices: dict = field(default_factory=dict, repr=False)  # kits: per-voice instruments, built on demand


_LOCK = threading.RLock()
_CACHE: "OrderedDict[str, Resolved]" = OrderedDict()


def clear_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def project_dirs(project=None, arrangement: dict | None = None) -> list[str]:
    """Folders a relative sound path is looked up in: the arrangement's ``sample_dir``, the
    project file's folder (when the project knows its path), the working directory."""
    out = []
    sd = (arrangement or {}).get("sample_dir") if isinstance(arrangement, dict) else None
    if isinstance(sd, str) and sd:
        out.append(os.path.abspath(os.path.expanduser(sd)))
    for attr in ("path", "source_path", "file_path", "base_dir"):
        v = getattr(project, attr, None) if project is not None else None
        if isinstance(v, (str, os.PathLike)) and str(v):
            v = os.path.abspath(os.fspath(v))
            out.append(v if os.path.isdir(v) else os.path.dirname(v))
    out.append(os.getcwd())
    seen, uniq = set(), []
    for d in out:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def find_path(path: str, dirs: list[str]) -> str | None:
    p = os.path.expanduser(str(path))
    if os.path.isabs(p):
        return p if os.path.exists(p) else None
    for d in dirs:
        q = os.path.join(d, p)
        if os.path.exists(q):
            return os.path.abspath(q)
    return None


def _installed(sound_id: str):
    try:
        from .. import sounds
    except ImportError:
        return None
    try:
        return sounds.find(sound_id)
    except Exception:
        return None


def looks_like_path(s: str) -> bool:
    return any(c in s for c in "/\\") or os.path.splitext(s)[1].lower() in (".sf2", ".sf3", ".sfz") + AUDIO_EXTS


def _stat_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}:{st.st_size}"
    except OSError:
        return "-"


# ------------------------------------------------------------------------------------ loading

def _pick_file(files: list[str], want: str | None, exts: tuple) -> list[str]:
    """Files with one of the extensions, those matching ``want`` (a name or a part of one) first."""
    cands = [f for f in files if os.path.splitext(f)[1].lower() in exts]
    if want:
        w = str(want).lower()
        exact = [f for f in cands if os.path.basename(f).lower() in (w, w + os.path.splitext(f)[1].lower())]
        part = [f for f in cands if w in os.path.basename(f).lower() and f not in exact]
        return exact + part
    return cands


def _walk(folder: str, exts: tuple, limit: int = 20000) -> list[str]:
    out = []
    for root, dirs, files in os.walk(folder):
        dirs.sort()
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() in exts:
                out.append(os.path.join(root, f))
                if len(out) >= limit:
                    return out
    return out


def _from_sf2(path: str, preset, drums: bool, synth_preset: str | None, kit: str | None):
    from .sf2 import load_sf2

    font = load_sf2(path)
    tried = []
    if preset is not None:
        p = font.find(preset)
        if p is None:
            bp = gm.program_of(preset)
            if bp is not None:
                p = font.find((128, bp[1]) if drums and bp[0] == 0 else bp)
        if p is None:
            raise SampleError(f"no preset {preset!r} in {os.path.basename(path)} "
                              f"({len(font.presets)} presets; see list_sounds)")
        return font.instrument(p), p.label()
    if drums:
        for prog in gm.KIT_PROGRAMS.get(kit or "pop", (0,)):
            p = font.find((128, prog))
            if p is not None:
                return font.instrument(p), p.label()
        kits = [p for p in font.presets if p.bank == 128] or font.presets[:1]  # a kit-only file may use bank 0
        return font.instrument(kits[0]), kits[0].label()
    for prog in gm.PRESET_PROGRAMS.get(synth_preset or "", ()):
        p = font.find((0, prog))
        if p is not None:
            return font.instrument(p), p.label()
        tried.append(prog)
    melodic = [p for p in font.presets if p.bank != 128]
    if not melodic:
        raise SampleError(f"{os.path.basename(path)} has only drum kits")
    return font.instrument(melodic[0]), melodic[0].label()


def _load_target(target: str, layer: dict, drums: bool, synth_preset: str | None, kit: str | None,
                 listed: list[str] | None = None, set_presets: list[dict] | None = None):
    """(instrument, label) for a file or folder."""
    preset, want = layer.get("preset"), layer.get("file")
    ext = os.path.splitext(target)[1].lower()
    if os.path.isfile(target):
        if ext == ".sf2":
            return _from_sf2(target, preset, drums, synth_preset, kit)
        if ext == ".sf3":
            raise SampleError(f"{os.path.basename(target)}: compressed SoundFonts (.sf3) are not supported; "
                              "use the .sf2 version of the set")
        if ext == ".sfz":
            from .sfz import load_sfz

            inst = load_sfz(target, cc=layer.get("cc"), keyswitch=layer.get("keyswitch"))
            return inst, os.path.basename(target)
        if ext in AUDIO_EXTS:
            from .folder import single_sample

            return single_sample(target, drums), os.path.basename(target)
        raise SampleError(f"{os.path.basename(target)}: not a .sf2, .sfz or sample file")
    if not os.path.isdir(target):
        raise SampleError(f"{target} does not exist")
    files = listed if listed is not None else _walk(target, (".sf2", ".sfz"))
    sf2s = _pick_file(files, want, (".sf2",))
    sfzs = _pick_file(files, want, (".sfz",))
    if sf2s:
        last = None
        for f in sf2s:  # the first SoundFont that has the preset
            try:
                return _from_sf2(f, preset, drums, synth_preset, kit)
            except SampleError as exc:
                last = exc
        if not sfzs:
            raise last
    if sfzs:
        from .sfz import load_sfz

        pick, meta = _pick_sfz(sfzs, preset, drums, set_presets or [], target)
        cc = dict(meta.get("cc") or {})
        cc.update(layer.get("cc") or {})
        inst = load_sfz(pick, defines=meta.get("defines"), cc=cc or None, keyswitch=layer.get("keyswitch"))
        return inst, str(meta.get("name") or os.path.basename(pick))
    from .folder import folder_instrument

    sub = target
    if preset is not None and isinstance(preset, str) and os.path.isdir(os.path.join(target, preset)):
        sub = os.path.join(target, preset)
    return folder_instrument(sub, drums), os.path.basename(os.path.normpath(sub))


def _match_meta(metas: list[dict], want) -> dict | None:
    """The manifest preset a name means: exact, then the start of a name, then a part of one."""
    if not isinstance(want, str) or not want.strip():
        return None
    w = gm._norm(want)
    named = [m for m in metas if m.get("name") and m.get("file")]
    for test in (lambda n: n == w, lambda n: n.startswith(w), lambda n: w in n):
        for m in named:
            if test(gm._norm(str(m["name"]))):
                return m
    return None


def _pick_sfz(sfzs: list[str], preset, drums: bool, metas: list[dict], target: str) -> tuple[str, dict]:
    """(file, manifest entry) of the SFZ instrument a preset names among a set's files. The
    manifest's presets are matched by name first (an entry may carry ``defines`` and ``cc``
    for its file), then the file names."""
    def path_of(meta):
        rel = str(meta.get("file", "")).replace("\\", "/")
        return next((f for f in sfzs if rel and f.replace("\\", "/").endswith(rel)), None)

    metas = [m for m in metas if path_of(m)]
    stems = [(gm._norm(os.path.splitext(os.path.basename(f))[0]), f) for f in sfzs]

    def meta_of(f):
        return next((m for m in metas if path_of(m) == f), {})

    if preset is not None:
        w = gm._norm(str(preset))
        for test in (lambda n: n == w, lambda n: n.startswith(w), lambda n: w in n):
            for m in metas:
                if test(gm._norm(str(m.get("name", "")))):
                    return path_of(m), m
            for stem, f in stems:
                if test(stem):
                    return f, meta_of(f)
        raise SampleError(f"no instrument {preset!r} among the {len(sfzs)} .sfz files of {target}")
    if drums:
        f = next((f for s_, f in stems if "kit" in s_ or "drum" in s_), None)
        if f is not None:
            return f, meta_of(f)
    if metas:
        return path_of(metas[0]), metas[0]
    return sfzs[0], {}


def _label(path: str, label: str) -> str:
    base = os.path.basename(os.path.normpath(path))
    return label if label == base else f"{base}: {label}"


def _load(layer: dict, drums: bool, synth_preset: str | None, kit: str | None, dirs: list[str]) -> Resolved:
    sid, path = layer.get("set"), layer.get("path")
    credits: list[str] = []
    try:
        if sid:
            snd = _installed(str(sid))
            if snd is None:
                hit = find_path(str(sid), dirs)  # a bare name may also be a folder next to the project
                if hit is None:
                    if looks_like_path(str(sid)):
                        return Resolved(None, str(sid), f"sound file or folder {sid!r} was not found "
                                                        f"(looked in: {', '.join(dirs)})")
                    return Resolved(None, str(sid), f"sound set {sid!r} is not installed "
                                                    f"(install it with: synth2 sounds install {sid})")
                inst, label = _load_target(hit, layer, drums, synth_preset, kit)
                return Resolved(inst, _label(hit, label))
            credits = list(snd.credits)
            # a preset the set's manifest names: its file or folder, with the entry's own settings
            meta = _match_meta(snd.presets, layer.get("preset")) or _match_meta(snd.presets, layer.get("file"))
            if meta is not None and os.path.exists(str(snd.file(str(meta["file"])))):
                f = str(snd.file(str(meta["file"])))
                if os.path.isdir(f):
                    from .folder import folder_instrument

                    inst = folder_instrument(f, drums)
                elif f.lower().endswith(".sfz"):
                    from .sfz import load_sfz

                    cc = dict(meta.get("cc") or {})
                    cc.update(layer.get("cc") or {})
                    inst = load_sfz(f, defines=meta.get("defines"), cc=cc or None, keyswitch=layer.get("keyswitch"))
                else:
                    inst, _label2 = _load_target(f, dict(layer, preset=None, file=None), drums, synth_preset, kit)
                return Resolved(inst, f"{snd.id}: {meta['name']}", credits=credits)
            files = [str(f) for f in snd.files]
            root = str(snd.root)
            expanded = []
            for f in files:
                expanded += _walk(f, (".sf2", ".sfz")) if os.path.isdir(f) else [f]
            if not any(os.path.splitext(f)[1].lower() in (".sf2", ".sfz") for f in expanded):
                # a set of sample folders: the folder named by file/preset, else the first one, else the root
                folders = [f for f in files if os.path.isdir(f)] or [root]
                want = layer.get("file") or (layer.get("preset") if isinstance(layer.get("preset"), str) else None)
                pick = next((f for f in folders if want and str(want).lower() in os.path.basename(f).lower()),
                            folders[0])
                inst, label = _load_target(pick, dict(layer, preset=None), drums, synth_preset, kit)
                label = next((str(m["name"]) for m in snd.presets if m.get("name") and m.get("file")
                              and os.path.normpath(str(snd.file(str(m["file"])))) == os.path.normpath(pick)), label)
            elif len(expanded) == 1 and expanded[0].lower().endswith(".sf2"):
                inst, label = _load_target(expanded[0], layer, drums, synth_preset, kit)
            else:
                inst, label = _load_target(root, layer, drums, synth_preset, kit, listed=expanded,
                                           set_presets=snd.presets)
            return Resolved(inst, f"{snd.id}: {label}", credits=credits)
        hit = find_path(str(path), dirs)
        if hit is None:
            return Resolved(None, str(path), f"sound file or folder {path!r} was not found "
                                             f"(looked in: {', '.join(dirs)})")
        inst, label = _load_target(hit, layer, drums, synth_preset, kit)
        return Resolved(inst, _label(hit, label))
    except SampleError as exc:
        return Resolved(None, str(sid or path), str(exc), credits=credits)
    except (OSError, ValueError, KeyError, IndexError, struct.error) as exc:  # a damaged file
        return Resolved(None, str(sid or path), f"cannot read {sid or path}: {exc}", credits=credits)



# ------------------------------------------------------------------------------------ options

def _fade_seconds(seconds: float, shape: str) -> float:
    """An envelope time for a fade the user states in seconds (to about -60 dB)."""
    return seconds * (100.0 / 60.0) if shape == "db" else seconds * 1.3 if shape == "exp" else seconds


def tweak_region(r: Region, o: dict) -> Region:
    """A region with a layer's options applied."""
    ch: dict = {}
    if o.get("tune"):
        ch["tune"] = r.tune + float(o["tune"])
    env = r.ampeg
    ech = {}
    for k in ("attack", "hold"):
        if o.get(k) is not None:
            ech[k] = max(0.0, float(o[k]))
    for k in ("decay", "release"):
        if o.get(k) is not None:
            ech[k] = _fade_seconds(max(0.0, float(o[k])), env.shape)
    if o.get("sustain") is not None:
        ech["sustain"] = min(1.0, max(0.0, float(o["sustain"])))
    if ech:
        ch["ampeg"] = replace(env, **ech)
        if o.get("release") is not None and r.loop_mode == "one_shot":
            ch["loop_mode"] = "no_loop"
    if o.get("cutoff") is not None:
        v2c = float(o.get("vel_to_cutoff") or 0.0)
        ch.update(cutoff=float(o["cutoff"]) * 2.0 ** (-v2c), fil_veltrack=1200.0 * v2c, fileg=None, fileg_depth=0.0)
        ch["fil_keytrack"] = 100.0 * float(o.get("keytrack") or 0.0)
        ch["resonance"] = float(o.get("resonance") or 0.0)
    else:
        if o.get("resonance") is not None and r.cutoff is not None:
            ch["resonance"] = float(o["resonance"])
        if o.get("keytrack") is not None and r.cutoff is not None:
            ch["fil_keytrack"] = 100.0 * float(o["keytrack"])
    if o.get("velocity_curve") is not None:
        ch.update(vel_curve=float(o["velocity_curve"]), vel_points=None)
    if o.get("vel_sens") is not None:
        ch["amp_veltrack"] = min(1.0, max(0.0, float(o["vel_sens"])))
    if o.get("vibrato") is not None:
        ch.update(vib_depth=100.0 * float(o["vibrato"]), vib_freq=float(o.get("vibrato_rate") or 5.5),
                  vib_delay=float(o.get("vibrato_delay") or 0.0), vib_fade=0.15)
    if o.get("pan"):
        ch["pan"] = min(1.0, max(-1.0, r.pan + float(o["pan"])))
    return replace(r, **ch) if ch else r


_TWEAKS = ("tune", "attack", "hold", "decay", "sustain", "release", "cutoff", "resonance", "keytrack",
           "vel_to_cutoff", "velocity_curve", "vel_sens", "vibrato", "pan")


def apply_options(inst: SampleInstrument, o: dict) -> SampleInstrument:
    """The instrument with a sound layer's options applied to its regions."""
    regions = inst.regions
    kr, vr = o.get("key_range"), o.get("vel_range")
    if kr or vr:
        out = []
        for r in regions:
            lo, hi = (max(r.lokey, int(kr[0])), min(r.hikey, int(kr[1]))) if kr else (r.lokey, r.hikey)
            vl, vh = (max(r.lovel, int(vr[0])), min(r.hivel, int(vr[1]))) if vr else (r.lovel, r.hivel)
            if lo <= hi and vl <= vh:
                out.append(replace(r, lokey=lo, hikey=hi, lovel=vl, hivel=vh))
        regions = out
    if o.get("round_robin") is False:
        regions = [replace(r, seq_length=1, seq_position=1, lorand=0.0, hirand=1.0) for r in regions
                   if r.seq_position == 1 and r.lorand <= 0.0]
    if any(o.get(k) is not None for k in _TWEAKS):
        out = []
        for r in regions:
            t = tweak_region(r, o)
            if r.resolve is not None:
                t = replace(t, resolve=_tweaked_resolver(r.resolve, o))
            out.append(t)
        regions = out
    if regions is inst.regions:
        return inst
    return inst.with_regions(regions)


def _tweaked_resolver(resolve, o: dict):
    cache: dict = {}

    def resolved(key: int, vel: int) -> Region:
        r = cache.get((key, vel))
        if r is None:
            r = tweak_region(resolve(key, vel), o)
            if len(cache) < 4096:
                cache[(key, vel)] = r
        return r

    return resolved


# ------------------------------------------------------------------------------ level matching

_REF_SR = 22050
_REF_VELOCITY = 0.8


def _geomean(ratios: list[float]) -> float:
    ratios = [r for r in ratios if r > 0 and np.isfinite(r)]
    if not ratios:
        return 1.0
    return float(min(30.0, max(1.0 / 30.0, np.exp(np.mean(np.log(ratios))))))


def match_synth_level(inst: SampleInstrument, synth_preset: str, low: int, transpose: int = 0,
                      chords: bool = True) -> float:
    """The gain that makes the sampled instrument as loud as the synth preset it replaces:
    the loudest 400 ms of the same notes at a normal velocity, single notes for a monophonic
    part, chords and single notes for a part that plays chords (a driven synth preset
    compresses a chord; samples add up)."""
    from .synth import Hit, preset_params, render_hit

    p = preset_params(synth_preset)
    keys = set(inst.keys())
    if chords and not p.get("mono"):
        phrases = [((low + 7, low + 14, low + 19), 0.4), ((low + 12, low + 19, low + 24), 0.25), ((low + 14,), 0.4)]
    else:
        phrases = [((low + 7,), 0.4), ((low + 14,), 0.4), ((low + 21,), 0.25)]
    ratios = []
    for notes, gate in phrases:
        if any(n + transpose not in keys for n in notes):
            continue
        ref = loudness(render_hit(p, Hit(0.0, gate, notes, _REF_VELOCITY), _REF_SR), _REF_SR)
        parts = [render_note(inst, n + transpose, midi_velocity(_REF_VELOCITY), gate, _REF_SR) for n in notes]
        mix = np.zeros((max(len(y) for y in parts), 2), np.float32)
        for y in parts:
            mix[: len(y)] += y
        got = loudness(mix, _REF_SR)
        if ref > 0 and got > 0:
            ratios.append(ref / got)
    return _geomean(ratios)


def match_voice_level(inst: SampleInstrument, key: int, voice: str, params: dict, gate: float = DRUM_GATE) -> float:
    """The gain that makes one drum of a sampled kit as loud as the synthesised voice it replaces."""
    from .drums import one_shot, velocity_gain

    ref = loudness(one_shot(voice, params, _REF_VELOCITY, 0, _REF_SR), _REF_SR) * velocity_gain(_REF_VELOCITY) * float(
        params.get("level", 1.0))
    got = loudness(render_note(inst, key, midi_velocity(_REF_VELOCITY), gate, _REF_SR), _REF_SR)
    return _geomean([ref / got]) if ref > 0 and got > 0 else 1.0


def match_kit_level(inst: SampleInstrument, kit_params: dict, gate: float = DRUM_GATE) -> float:
    """The gain that makes a sampled kit as loud as the synthesised kit: kick, snare and
    closed hat are compared, the kit's own balance between its drums is kept."""
    from .drums import drum_params, one_shot, velocity_gain

    keys = set(inst.keys())
    ratios = []
    for voice in ("kick", "snare", "hat"):
        key = next((k for k in gm.drum_keys(voice) if k in keys), None)
        if key is None:
            continue
        params = drum_params(voice, (kit_params or {}).get(voice))
        shot = one_shot(voice, params, _REF_VELOCITY, 0, _REF_SR)
        ref = loudness(shot, _REF_SR) * velocity_gain(_REF_VELOCITY) * float(params.get("level", 1.0))
        got = loudness(render_note(inst, key, midi_velocity(_REF_VELOCITY), gate, _REF_SR), _REF_SR)
        if ref > 0 and got > 0:
            ratios.append(ref / got)
    return _geomean(ratios)


# ------------------------------------------------------------------------------------ resolve

def resolve(layer: dict, *, drums: bool = False, synth_preset: str | None = None, kit: str | None = None,
            kit_params: dict | None = None, low: int = 48, dirs: list[str] | None = None,
            chords: bool = True) -> Resolved:
    """Load the sound a layer names, apply the layer's options and match its level.

    ``chords`` says the part plays chords (its level is then matched on chords as well as
    single notes). Results are cached (keyed by the layer, the context and the files'
    modification times).
    """
    dirs = list(dirs) if dirs else [os.getcwd()]
    target = str(layer.get("set") or layer.get("path") or "")
    snd = _installed(target) if layer.get("set") else None
    hit = find_path(target, dirs) if snd is None else None
    sig = _stat_sig(hit) if hit else (_stat_sig(str(snd.path)) if snd is not None else "-")
    key = json.dumps([layer, drums, synth_preset, kit, kit_params if drums else None, low, chords, dirs, sig,
                      os.environ.get("SYNTH2_SOUNDS", "")], sort_keys=True, default=str)
    with _LOCK:
        res = _CACHE.get(key)
        if res is not None:
            _CACHE.move_to_end(key)
            return res
        res = _load(layer, drums, synth_preset, kit, dirs)
        if res.instrument is not None:
            inst = apply_options(res.instrument, layer)
            res.warnings = list(inst.warnings)
            if not inst.regions:
                why = "; ".join(w for w in inst.warnings[:3] if w != "no playable regions")
                res = Resolved(None, res.label, f"{res.label} has no playable samples" + (f" ({why})" if why else ""),
                               credits=res.credits)
            else:
                gain = 1.0
                if layer.get("normalize", True) and not drums and synth_preset:
                    # (a kit is matched drum by drum when it is played: see render._kit_voice)
                    gain = match_synth_level(inst, synth_preset, low, int(layer.get("transpose") or 0), chords)
                gain *= 10.0 ** (float(layer.get("gain_db") or 0.0) / 20.0)
                res.instrument = inst.with_regions(inst.regions, gain=inst.gain * gain)
        res.options = dict(layer)
        _CACHE[key] = res
        while len(_CACHE) > 48:
            _CACHE.popitem(last=False)
        return res


def describe(layer: dict) -> str:
    """Short text naming a layer's sound for messages."""
    s = str(layer.get("set") or layer.get("path") or "?")
    return f"{s} ({layer['preset']})" if layer.get("preset") is not None else s


def catalogue(dirs: list[str] | None = None, max_presets: int = 400) -> list[dict]:
    """Installed sound sets and what each can play: for a SoundFont its presets (bank,
    program, name), for SFZ sets their instruments, for sample folders their names."""
    try:
        from .. import sounds
    except ImportError:
        return []
    out = []
    for snd in sounds.installed():
        entry = {"id": snd.id, "kind": snd.kind, "name": snd.name, "tags": snd.tags, "license": snd.license,
                 "credits": snd.credits, "presets": [], "kits": []}
        if snd.kind == "impulse":
            out.append(entry)
            continue
        files = []
        for f in snd.files:
            f = str(f)
            files += _walk(f, (".sf2", ".sfz"), 2000) if os.path.isdir(f) else [f]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            rel = os.path.relpath(f, str(snd.root))
            if ext == ".sf2":
                try:
                    from .sf2 import load_sf2

                    font = load_sf2(f)
                except SampleError as exc:
                    entry.setdefault("problems", []).append(str(exc))
                    continue
                kit_file = "drums" in snd.tags and not any(p.bank == 128 for p in font.presets)
                for p in font.presets:
                    item = {"preset": f"{p.bank}:{p.program}", "name": p.name, "file": rel}
                    (entry["kits"] if p.bank == 128 or kit_file else entry["presets"]).append(item)
            elif ext == ".sfz":
                named = [p for p in snd.presets if str(p.get("file", "")).replace("\\", "/") == rel.replace("\\", "/")]
                low = rel.lower()
                for meta in named or [{"name": os.path.splitext(os.path.basename(f))[0]}]:
                    item = {"preset": str(meta.get("name")), "name": str(meta.get("name")), "file": rel}
                    (entry["kits"] if ("drum" in low or "kit" in low or "drums" in snd.tags)
                     else entry["presets"]).append(item)
            elif ext == ".sf3":
                entry.setdefault("problems", []).append(f"{rel}: compressed SoundFonts (.sf3) are not supported")
        if not files:  # folders of samples: the manifest's names, else the folders
            metas = [m for m in snd.presets if m.get("name") and m.get("file")] or [
                {"name": os.path.basename(str(f)), "file": os.path.relpath(str(f), str(snd.root))}
                for f in snd.files if os.path.isdir(str(f))]
            for m in metas:
                item = {"preset": str(m["name"]), "name": str(m["name"]), "file": str(m["file"])}
                (entry["kits"] if snd.kind == "drumkit" or m.get("category") == "drums" else entry["presets"]).append(item)
        for k in ("presets", "kits"):
            if len(entry[k]) > max_presets:
                entry[f"{k}_total"] = len(entry[k])
                entry[k] = entry[k][:max_presets]
        out.append(entry)
    return out
