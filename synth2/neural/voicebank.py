"""A DiffSinger voicebank in OpenUtau layout: configs, phoneme tables, speakers and ONNX sessions.

Nothing here is specific to one bank. A bank folder holds a root ``dsconfig.yaml`` (the
acoustic model) and the sub-folders ``dsdur/`` (linguistic encoder and duration model),
``dsvariance/`` (breathiness, voicing, tension, energy), ``dspitch/`` (pitch predictor) and
``dsvocoder/`` (a bundled vocoder), each with its own ``dsconfig.yaml``, phoneme table,
language table and ``speakers:`` list. Banks differ in which of these they have and in the
inputs each model takes, so every model feed is built from the session's own input list
(:meth:`Voicebank.io`, :func:`feed`).

``onnxruntime`` and ``yaml`` are imported on first use. Sessions are created lazily under a
lock and shared. The diffusion models draw their noise inside the graph: a session made by
:meth:`Voicebank.fresh_session` right after ``onnxruntime.set_seed`` gives the same output
on its first run every time, which is how the engine makes a phrase reproducible.
"""
from __future__ import annotations

import json
import os
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REST_SYMBOLS = ("SP", "AP")
THREADS_ENV = "SYNTH2_NEURAL_THREADS"
SUBFOLDERS = {"dur": "dsdur", "variance": "dsvariance", "pitch": "dspitch"}
# models that run a diffusion sampler; a vocoder is checked by its operators
DIFFUSION = ("acoustic", "variance", "pitch")
_RANDOM_OPS = (b"RandomNormal", b"RandomUniform", b"Multinomial", b"Bernoulli")
_SYMBOL = re.compile(r"\bsymbol:\s*([^,}\s]+)")
_TYPE = re.compile(r"\btype:\s*([A-Za-z_]+)")
# one lock for every (set_seed, InferenceSession) pair in the process: set_seed is global
_SEED_LOCK = threading.Lock()


class VoicebankError(RuntimeError):
    """A voicebank or vocoder folder cannot be used; the message names the file and the fault."""


def thread_count() -> int:
    """onnxruntime threads per session: ``SYNTH2_NEURAL_THREADS``, else ``min(4, cpu_count)``."""
    try:
        n = int(os.environ.get(THREADS_ENV) or 0)
    except ValueError:
        n = 0
    return n if n > 0 else max(1, min(4, os.cpu_count() or 1))


def read_yaml(path: Path) -> dict:
    """A YAML mapping (read as ``utf-8-sig``: OpenUtau files carry a byte-order mark)."""
    import yaml

    try:
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    except OSError as exc:
        raise VoicebankError(f"{path}: cannot be read ({exc.strerror or exc})") from None
    except yaml.YAMLError as exc:
        raise VoicebankError(f"{path}: not valid YAML ({exc})") from None
    if not isinstance(data, dict):
        raise VoicebankError(f"{path}: expected a mapping of settings")
    return data


def read_table(path: Path) -> dict[str, int]:
    """A phoneme or language table: a JSON object ``{name: id}`` or a text file with one name
    per line (the line number is the id)."""
    try:
        text = Path(path).read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise VoicebankError(f"{path}: cannot be read ({exc.strerror or exc})") from None
    if Path(path).suffix.lower() != ".json":
        return {p: i for i, p in enumerate(text.splitlines())}
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise VoicebankError(f"{path}: not valid JSON ({exc})") from None
    if not isinstance(data, dict):
        raise VoicebankError(f"{path}: expected an object of names and ids")
    return {str(k): int(v) for k, v in data.items()}


def scan_symbols(path: Path) -> dict[str, str]:
    """``{symbol: type}`` from the ``symbols:`` list of a ``dsdict*.yaml``.

    The file is scanned line by line: a dictionary can run to a million lines and a YAML
    parser then takes minutes. Both layouts are read, ``- {symbol: a, type: vowel}`` and
    ``- symbol: a`` followed by ``type: vowel``.
    """
    out: dict[str, str] = {}
    sym = None
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                if "symbol:" in line:
                    m = _SYMBOL.search(line)
                    if m:
                        sym = m.group(1).strip("'\"")
                if sym is not None and "type:" in line:
                    m = _TYPE.search(line)
                    if m:
                        out[sym] = m.group(1)
                        sym = None
    except OSError:
        return {}
    return out


def draws_random(path: Path) -> bool:
    """True when the ONNX file names an operator that draws random numbers."""
    tail = b""
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(1 << 20)
                if not chunk:
                    return False
                data = tail + chunk
                if any(op in data for op in _RANDOM_OPS):
                    return True
                tail = chunk[-32:]
    except OSError:
        return False


def speedup_for(steps: int) -> int:
    """The ``speedup`` an older export takes instead of ``steps``: the largest divisor of 1000
    not above ``1000 / steps`` (OpenUtau's rule)."""
    limit = max(1, 1000 // max(1, int(steps)))
    return max(d for d in range(1, limit + 1) if 1000 % d == 0)


def match_speaker(speakers: list[str], want: str) -> str | None:
    """The ``speakers`` entry for a suffix: an exact match, then equal trailing path segments."""
    parts = [p for p in want.replace("\\", "/").split("/") if p]
    for sp in speakers:
        if sp == want:
            return sp
    for sp in speakers:
        if parts and sp.replace("\\", "/").split("/")[-len(parts):] == parts:
            return sp
    return None


@dataclass
class ModelIO:
    """The inputs and outputs of one ONNX model, read from a session."""

    inputs: list[str]
    shapes: dict[str, list]
    outputs: list[str]

    def takes(self, name: str) -> bool:
        return name in self.shapes


@dataclass
class SubModel:
    """One ``dsconfig.yaml`` and what it points at."""

    name: str
    folder: Path
    config: dict
    phonemes: dict[str, int]
    languages: dict[str, int]
    speakers: list[str]
    models: dict[str, Path] = field(default_factory=dict)

    @property
    def config_path(self) -> Path:
        return self.folder / "dsconfig.yaml"

    def has(self, phoneme: str) -> bool:
        return phoneme in self.phonemes

    def tokens(self, phones: list[str]) -> np.ndarray:
        """``int64 [1, P]`` phoneme ids."""
        try:
            return np.array([[self.phonemes[p] for p in phones]], np.int64)
        except KeyError as exc:
            raise VoicebankError(f"{self.config_path}: the {self.name} model has no phoneme {exc.args[0]!r}") from None

    def language_ids(self, phones: list[str]) -> np.ndarray:
        """``int64 [1, P]`` language ids from each phoneme's ``lang/`` prefix (0 without one)."""
        lg = self.languages
        return np.array([[lg.get(p.split("/")[0], 0) if "/" in p else 0 for p in phones]], np.int64)


def _load_sub(name: str, folder: Path, roles: tuple[str, ...], required: tuple[str, ...]) -> SubModel:
    cfg_path = folder / "dsconfig.yaml"
    cfg = read_yaml(cfg_path)
    if not cfg.get("phonemes"):
        raise VoicebankError(f"{cfg_path}: no 'phonemes' entry")
    phonemes = read_table(folder / str(cfg["phonemes"]))
    languages = read_table(folder / str(cfg["languages"])) if cfg.get("languages") else {}
    models = {}
    for role in roles:
        rel = cfg.get(role)
        if not rel:
            if role in required:
                raise VoicebankError(f"{cfg_path}: no '{role}' model entry")
            continue
        path = folder / str(rel)
        if not path.is_file():
            raise VoicebankError(f"{cfg_path}: the '{role}' model {rel!r} does not exist")
        models[role] = path
    speakers = [str(s) for s in cfg.get("speakers") or []]
    return SubModel(name, folder, cfg, phonemes, languages, speakers, models)


class Voicebank:
    """One loaded voicebank and its vocoder.

    ``root`` is the folder with the root ``dsconfig.yaml``. ``vocoder`` is a folder with a
    ``vocoder.yaml``; None uses the bank's own ``dsvocoder/``. Session keys are ``acoustic``,
    ``dur.linguistic``, ``dur``, ``variance.linguistic``, ``variance``, ``pitch.linguistic``,
    ``pitch`` and ``vocoder``.
    """

    def __init__(self, root, vocoder=None, *, bank_id: str | None = None, vocoder_id: str | None = None,
                 credits: list[str] | None = None, threads: int | None = None):
        self.root = Path(root)
        self.id = bank_id or self.root.name
        self.credits = list(credits or [])
        self.threads = int(threads) if threads else thread_count()
        self._lock = threading.Lock()
        self._shared: dict[str, object] = {}
        self._io: dict[str, ModelIO] = {}
        self._noisy: dict[str, bool] = {}
        self._embeds: dict[tuple, np.ndarray] = {}
        self._vowels: set[str] | None = None
        self.comfort_range: tuple[float, float] | None = None  # MIDI notes the bank sings well, when known

        if not (self.root / "dsconfig.yaml").is_file():
            raise VoicebankError(f"{self.root}: no dsconfig.yaml, not a DiffSinger voicebank folder")
        self.acoustic = _load_sub("acoustic", self.root, ("acoustic",), ("acoustic",))
        cfg = self.acoustic.config
        self.sample_rate = int(cfg.get("sample_rate", 44100))
        self.hop = int(cfg.get("hop_size", 512))
        self.mel_bins = int(cfg.get("num_mel_bins", 128))
        self.frame = self.hop / self.sample_rate
        self.max_depth = float(cfg.get("max_depth", 1.0))
        shift = ((cfg.get("augmentation_args") or {}).get("random_pitch_shifting") or {}).get("range")
        self.key_shift_range = (float(shift[0]), float(shift[1])) if shift else (0.0, 0.0)

        if not (self.root / "dsdur" / "dsconfig.yaml").is_file():
            raise VoicebankError(f"{self.root / 'dsdur' / 'dsconfig.yaml'}: missing; the bank has no duration model")
        self.dur = _load_sub("dur", self.root / "dsdur", ("linguistic", "dur"), ("linguistic", "dur"))
        self.variance: SubModel | None = None
        if (self.root / "dsvariance" / "dsconfig.yaml").is_file():
            self.variance = _load_sub("variance", self.root / "dsvariance", ("linguistic", "variance"),
                                      ("linguistic", "variance"))
        self.pitch: SubModel | None = None
        self.pitch_note = ""  # why a dspitch/ folder is not used
        if (self.root / "dspitch" / "dsconfig.yaml").is_file():
            self._load_pitch()
        for sub in self.subs():
            for key in ("sample_rate", "hop_size"):
                want = self.sample_rate if key == "sample_rate" else self.hop
                if sub.config.get(key) is not None and int(sub.config[key]) != want:
                    raise VoicebankError(f"{sub.config_path}: {key} {sub.config[key]} differs from the acoustic "
                                         f"model's {want}")

        self.modes, self.default_mode = self._read_modes()
        self.vocoder_id = vocoder_id
        self._load_vocoder(Path(vocoder) if vocoder is not None else None)
        self._paths = {"acoustic": self.acoustic.models["acoustic"], "dur.linguistic": self.dur.models["linguistic"],
                       "dur": self.dur.models["dur"], "vocoder": self.vocoder_path}
        if self.variance is not None:
            self._paths.update({"variance.linguistic": self.variance.models["linguistic"],
                                "variance": self.variance.models["variance"]})
        if self.pitch is not None:
            self._paths.update({"pitch.linguistic": self.pitch.models["linguistic"],
                                "pitch": self.pitch.models["pitch"]})

    # ---- configuration
    def subs(self) -> list[SubModel]:
        return [s for s in (self.acoustic, self.dur, self.variance, self.pitch) if s is not None]

    def _load_pitch(self) -> None:
        folder = self.root / "dspitch"
        try:
            sub = _load_sub("pitch", folder, ("linguistic", "pitch"), ("linguistic", "pitch"))
        except VoicebankError as exc:
            self.pitch_note = str(exc)
            return
        if sub.config.get("predict_dur"):
            self.pitch_note = f"{sub.config_path}: pitch models with predict_dur are not supported"
        elif not sub.config.get("use_continuous_acceleration", False):
            self.pitch_note = f"{sub.config_path}: pitch models that take 'speedup' are not supported"
        else:
            self.pitch = sub

    def _read_modes(self) -> tuple[dict[str, str], str | None]:
        """Voice modes: ``{lower-case name: speaker suffix}`` and the default mode's suffix.

        Names are the ``color`` of each ``subbanks:`` entry of ``character.yaml`` and the last
        path segment of every speaker; the entry with an empty colour is the default mode.
        """
        modes: dict[str, str] = {}
        default = None
        path = self.root / "character.yaml"
        if path.is_file():
            try:
                subbanks = read_yaml(path).get("subbanks") or []
            except VoicebankError:
                subbanks = []
            for sb in subbanks:
                if not isinstance(sb, dict) or not sb.get("suffix"):
                    continue
                color, suffix = str(sb.get("color") or ""), str(sb["suffix"])
                if color:
                    modes.setdefault(color.lower(), suffix)
                elif default is None:
                    default = suffix
        for sub in self.subs():
            for sp in sub.speakers:
                modes.setdefault(sp.replace("\\", "/").split("/")[-1].lower(), sp)
        return modes, default

    def _load_vocoder(self, folder: Path | None) -> None:
        folder = folder if folder is not None else self.root / "dsvocoder"
        path = folder / "vocoder.yaml"
        if not path.is_file():
            raise VoicebankError(f"{path}: missing; voicebank {self.id!r} has no vocoder")
        cfg = read_yaml(path)
        if not cfg.get("model"):
            raise VoicebankError(f"{path}: no 'model' entry")
        model = folder / str(cfg["model"])
        if not model.is_file():
            raise VoicebankError(f"{path}: the model {cfg['model']!r} does not exist")
        for key, want in (("sample_rate", self.sample_rate), ("hop_size", self.hop), ("num_mel_bins", self.mel_bins)):
            if cfg.get(key) is not None and int(cfg[key]) != int(want):
                raise VoicebankError(f"{path}: {key} is {cfg[key]}, voicebank {self.id!r} needs {want}")
        self.vocoder_config = cfg
        self.vocoder_path = model
        self.vocoder_name = str(cfg.get("name") or model.stem)
        if self.vocoder_id is None:
            self.vocoder_id = self.vocoder_name

    def has(self, phoneme: str) -> bool:
        """True when every model that reads phonemes knows this one."""
        return all(s.has(phoneme) for s in (self.acoustic, self.dur, self.variance) if s is not None)

    @property
    def vowels(self) -> set[str]:
        """Symbols typed ``vowel`` in the bank's dictionaries, plus ``SP`` and ``AP``."""
        if self._vowels is None:
            out = set(REST_SYMBOLS)
            files = []
            if self.pitch is not None:
                files.append(self.pitch.folder / "dsdict.yaml")
            if not any(f.is_file() for f in files):
                files = sorted(self.dur.folder.glob("dsdict*.yaml"))
            for f in files:
                out |= {s for s, t in scan_symbols(f).items() if t == "vowel"}
            self._vowels = out
        return self._vowels

    def is_vowel(self, phoneme: str) -> bool:
        return phoneme in self.vowels

    def identity(self) -> list:
        """What the audio depends on in the model files: name, size and modification time of
        every model, configuration and embedding file. For cache keys."""
        files = set(self._paths.values())
        for sub in self.subs():
            files.add(sub.config_path)
            files.update(sub.folder / (sp + ".emb") for sp in sub.speakers)
        out = []
        for p in sorted(files):
            try:
                st = p.stat()
                out.append([p.name, st.st_size, st.st_mtime_ns])
            except OSError:
                out.append([p.name, -1, -1])
        return [self.id, self.vocoder_id, out]

    # ---- speakers
    def blend(self, weights: dict[str, float] | None) -> tuple[tuple[str, float], ...] | None:
        """A normalised speaker blend ``((suffix, weight), ...)`` from ``{mode name: weight}``.
        None (the default mode) for an empty blend. Unknown names raise :class:`VoicebankError`."""
        if not weights:
            return None
        out: dict[str, float] = {}
        for name, w in weights.items():
            suffix = self.modes.get(str(name).lower())
            if suffix is None:
                raise VoicebankError(f"voicebank {self.id!r} has no voice mode {name!r}; "
                                     f"modes: {', '.join(sorted(self.modes)) or 'none'}")
            if float(w) > 0:
                out[suffix] = out.get(suffix, 0.0) + float(w)
        total = sum(out.values())
        if total <= 0:
            return None
        return tuple(sorted((s, w / total) for s, w in out.items()))

    def embed(self, sub: SubModel, dim: int, blend=None) -> np.ndarray:
        """The speaker embedding of one model: ``float32 [dim]``, the weighted sum of the
        blend's ``<speaker>.emb`` files (raw little-endian float32)."""
        key = (sub.name, int(dim), blend)
        hit = self._embeds.get(key)
        if hit is not None:
            return hit
        if not sub.speakers:
            raise VoicebankError(f"{sub.config_path}: the {sub.name} model takes spk_embed and lists no speakers")
        if blend is None:
            first = match_speaker(sub.speakers, self.default_mode) if self.default_mode else None
            picks = [(first or sub.speakers[0], 1.0)]
        else:
            picks = []
            for suffix, w in blend:
                sp = match_speaker(sub.speakers, suffix)
                if sp is None:
                    raise VoicebankError(f"{sub.config_path}: no speaker {suffix!r} for the {sub.name} model")
                picks.append((sp, w))
        vecs = []
        for sp, w in picks:
            path = sub.folder / (sp + ".emb")
            try:
                e = np.fromfile(str(path), "<f4")
            except OSError as exc:
                raise VoicebankError(f"{path}: cannot be read ({exc.strerror or exc})") from None
            if e.size != dim:
                raise VoicebankError(f"{path}: {e.size} values, the {sub.name} model takes {dim}")
            vecs.append((e, w))
        out = vecs[0][0] if len(vecs) == 1 else sum(e * np.float32(w) for e, w in vecs).astype(np.float32)
        self._embeds[key] = out
        return out

    # ---- sessions
    def _create(self, key: str):
        import onnxruntime as ort

        path = self._paths.get(key)
        if path is None:
            raise VoicebankError(f"voicebank {self.id!r} has no {key} model")
        so = ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        so.log_severity_level = 3
        try:
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise VoicebankError(f"{path}: onnxruntime cannot load the model ({exc})") from None
        if key not in self._io:
            ins = sess.get_inputs()
            self._io[key] = ModelIO([i.name for i in ins], {i.name: list(i.shape) for i in ins},
                                    [o.name for o in sess.get_outputs()])
        return sess

    def session(self, key: str):
        """The shared session of a model, created on first use."""
        sess = self._shared.get(key)
        if sess is None:
            with self._lock:
                sess = self._shared.get(key)
                if sess is None:
                    sess = self._shared[key] = self._create(key)
        return sess

    def fresh_session(self, key: str, seed: int):
        """A new session whose first run draws the noise that ``seed`` selects."""
        import onnxruntime as ort

        with _SEED_LOCK:
            ort.set_seed(int(seed) & 0x7FFFFFFF)
            return self._create(key)

    def io(self, key: str) -> ModelIO:
        """Inputs and outputs of a model (a session of it must have been created)."""
        if key not in self._io:
            self.session(key)
        return self._io[key]

    def draws_noise(self, key: str) -> bool:
        """True for the diffusion models and for any other model whose graph draws random
        numbers (some bundled vocoders do)."""
        hit = self._noisy.get(key)
        if hit is None:
            hit = self._noisy[key] = key in DIFFUSION or (key == "vocoder" and draws_random(self._paths[key]))
        return hit

    def release(self) -> None:
        """Drop the shared sessions (they are created again on the next use)."""
        with self._lock:
            self._shared.clear()

    def summary(self) -> dict:
        return {"id": self.id, "root": str(self.root), "sample_rate": self.sample_rate, "hop": self.hop,
                "mel_bins": self.mel_bins, "max_depth": self.max_depth, "key_shift_range": list(self.key_shift_range),
                "variance": self.variance is not None, "pitch": self.pitch is not None,
                "modes": sorted(self.modes), "vocoder": self.vocoder_id,
                "languages": sorted(self.acoustic.languages)}


def feed(io: ModelIO, values: dict, what: str = "model") -> dict:
    """The feed of a session: one value per input, in the model's own order."""
    missing = [n for n in io.inputs if n not in values]
    if missing:
        raise VoicebankError(f"the {what} takes inputs this engine does not provide: {', '.join(missing)}")
    return {n: values[n] for n in io.inputs}


def resolve_vocoder(model):
    """``(folder with vocoder.yaml or None, vocoder id, credits)`` for an installed voicebank.

    In this order: the bank's own ``dsvocoder/`` (None: :class:`Voicebank` looks there), the
    vocoder its manifest names, an installed vocoder whose ``vocoder.yaml`` ``name`` equals the
    ``vocoder`` field of the bank's ``dsconfig.yaml``, the default vocoder.
    """
    from . import models

    if (model.root / "dsvocoder" / "vocoder.yaml").is_file():
        return None, None, []
    hit = models.find("vocoder", model.vocoder) if model.vocoder else None
    if hit is None:
        try:
            want = read_yaml(model.root / "dsconfig.yaml").get("vocoder")
        except VoicebankError:
            want = None
        for voc in models.installed("vocoder") if want else []:
            try:
                if read_yaml(voc.root / "vocoder.yaml").get("name") == want:
                    hit = voc
                    break
            except VoicebankError:
                continue
    if hit is None:
        hit = models.find("vocoder")
    if hit is None:
        raise VoicebankError(f"no vocoder is installed for voicebank {model.id!r}")
    return hit.root, hit.id, list(hit.credits)


def load_installed(model) -> Voicebank:
    """The :class:`Voicebank` of an installed voicebank model (``models.InstalledModel``)."""
    folder, voc_id, voc_credits = resolve_vocoder(model)
    credits = list(model.credits) + [c for c in voc_credits if c not in model.credits]
    return Voicebank(model.root, folder, bank_id=model.id, vocoder_id=voc_id, credits=credits)
