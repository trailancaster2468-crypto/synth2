# The neural engine

Synth2 has two singing engines behind one interface (`synth2/engine/backend.py`):

| Engine | What it is | Needs |
|---|---|---|
| `formant` | The built-in parametric singer. Always available. | numpy |
| `neural` | A DiffSinger voicebank (acoustic, duration, pitch and variance models) and a neural vocoder, run with `onnxruntime`. | `pip install "synth2[neural]"` and `synth2 models install` |

The neural engine is what makes lyrics intelligible and the voice sound like a modern
vocal synthesizer. The formant engine stays as the dependency-free fallback and for
deliberately synthetic voices.

This document is the contract between the parts of the neural engine. Change it in the
same commit as the code.

## Choosing the engine

An *engine spec* is one of:

```
auto                  (also "" or nothing) neural when it is installed, formant otherwise
formant               the built-in engine
neural                the neural engine with the track's or the default voicebank
neural:<bank-id>      the neural engine with that installed voicebank
```

`resolve_engine(project, track, override)` takes the first spec that is not `auto` from:

1. the render argument (`render_project(..., engine=)`, the `render` tool's `engine`,
   `synth2 render --engine`),
2. `track.engine`,
3. `project.engine`,
4. the `SYNTH2_ENGINE` environment variable.

With none set (auto), a voice id that a plug-in backend registered keeps that backend.
Any other voice is sung by the neural engine when `synth2.neural.available()` is true and
by the formant engine when it is not.

Asking for `neural` by name when it cannot run raises `EngineUnavailable` with the
install hint. Auto never raises: it falls back to `formant`.

Quality presets `draft`, `normal` and `high` trade render time for detail in the neural
engine (the formant engine ignores them). They come from the render argument `quality`,
then `SYNTH2_QUALITY`, then `normal`. Backends read the value with
`synth2.engine.backend.current_quality()`.

`Track.engine` and `Project.engine` are stored in project JSON only when set. Score
notation writes them as the `engine=` track attribute and the `engine:` header.

## API in `synth2/engine/backend.py`

```python
ENGINE_CHOICES = ("auto", "formant", "neural")
QUALITY_CHOICES = ("draft", "normal", "high")

class EngineUnavailable(RuntimeError)
parse_engine(spec) -> (name, bank | None)          # ValueError on a malformed spec
normalize_engine(spec) -> str                      # canonical text, "" for auto
parse_quality(q) -> "draft" | "normal" | "high" | None
resolve_engine(project=None, track=None, override=None) -> (name, bank | None)
backend_for_track(project, track, override=None) -> VoiceBackend
engine_signature(project, override=None) -> list[str]   # for cache keys, never raises
render_options(**opts)                             # context manager, sets per-render options
current_render_options() -> dict
current_quality(default="normal") -> str
```

A backend implements:

```python
class VoiceBackend(Protocol):
    name: str
    def voices(self) -> list[str]: ...
    def render_track(self, project, track, sample_rate, t0, t1, note_range=None)
        -> tuple[np.ndarray, dict]: ...
```

`render_track` returns mono float32 audio covering `[t0, t1)` seconds at `sample_rate`
(sample `i` is at `t0 + i / sample_rate`) and a track report. For a partial render
(`note_range` set, in beats) the audio must equal what the full render has over the same
window, and the report covers only the notes that overlap `note_range`.

Build the report with `synth2.engine.renderer.track_report(project, track, plan, voice,
style, audio, note_range, engine="neural")`. It returns `id`, `name`, `voice`, `style`,
`engine`, `notes`, `warnings`, `phonemes` and `peak_db`. The neural backend adds:

- `credits`: the attribution lines of the voicebank and vocoder that sang the track.
  `render_project` collects them into `report["credits"]`.
- `bank`: the voicebank id.

`render_project`'s report has `engines` (sorted names of the engines used) and, when any
model asks for attribution, `credits`.

## Package layout: `synth2/neural/`

| Module | Role |
|---|---|
| `__init__.py` | `available(bank=None) -> (ok, reason)`, `backend(bank=None)`, `status()`, `credits(bank=None)`, `refresh()`. Cheap to import, never raises. |
| `models.py` | Where models live and which are installed (`models_dir`, `InstalledModel`, `read_manifest`, `installed`, `find`, `vocoder_for`), plus the installer (`REGISTRY`, `install`, `add_local`, `remove`). |
| `cli.py` | `synth2 models list`, `install`, `add`, `remove` and `path`. Exposes `add_parser(subparsers)` and `run(args)`. |
| `voicebank.py` | Loads a DiffSinger voicebank folder: `dsconfig.yaml` files, phoneme and language tables, ONNX sessions (created lazily, shared, thread count from `SYNTH2_NEURAL_THREADS`). |
| `phonemes.py` | Maps Synth2 phonemes to a voicebank's phoneme set per language, with fallbacks for phonemes the bank lacks. |
| `align.py` | Turns the planner's timed phonemes into phrases, word groups and frame durations (vowel on the note onset). |
| `engine.py` | Runs the models for one phrase: durations, pitch, variance, acoustic, vocoder. |
| `presets.py` | `VoicePreset` for each Synth2 voice id: how the built-in voices (`aurora`, `atlas`, ...) map onto a voicebank (formant shift, tone shift, breathiness, level, optional effect, optional bank). |
| `fx.py` | Character effects for non-human voices (ring modulation, comb, bit crush). numpy only. |
| `resample.py` | Windowed-sinc polyphase resampler (numpy only) for project sample rates other than the model's. |
| `cache.py` | Phrase cache on disk, keyed by everything that affects a phrase's audio. |
| `backend.py` | `NeuralBackend` (the `VoiceBackend`), `get_backend(bank=None)`, `reset()`. |

Rules:

- `import synth2` and `import synth2.neural` must not import `onnxruntime`, `yaml` or any
  model. Heavy imports happen inside functions, on the first neural render.
- A broken or half-installed neural engine must never break the formant engine. Checks
  return `(False, reason)`; only an explicit `neural` request raises.
- numpy is the only array dependency. No scipy, no torch, no librosa at runtime.
- Model output must be deterministic for a given project, engine version and settings:
  seed the diffusion noise from a stable hash of the phrase content.

### Rendering pipeline

For one track:

1. `plan_track` gives timed phoneme segments, exactly as for the formant engine, so
   lyrics, phoneme overrides, `-`, `+`, `br` and note attributes behave the same.
2. `align.py` cuts the plan into phrases at rests, converts phonemes to the bank's
   symbols and builds word groups whose vowels start on the note onsets.
3. `engine.py` runs, per phrase: the duration model (then rescales its prediction to the
   note grid), the pitch source (Synth2's expression engine `build_pitch`, so styles,
   vibrato, `pitchDelta` and note attributes keep working; the bank's own pitch predictor
   is an option), the variance model (breathiness, voicing and whatever else the bank
   predicts), the acoustic model (mel spectrogram) and the vocoder (waveform).
4. `backend.py` places phrases on a global frame grid, applies the voice preset, the
   parameter curves, gain and effects, resamples to the project rate and crops `[t0, t1)`.

Phrases are independent, so a phrase whose inputs did not change is served from the
cache, and a partial render only synthesizes the phrases that overlap the window.

### Parameter curves

| Synth2 parameter | Neural engine |
|---|---|
| `pitchDelta`, `vibratoEnv`, note vibrato and transition attributes | through `build_pitch` into f0 |
| `loudness` | gain ride in dB |
| `gender` | the acoustic model's formant shift (`gender` input) |
| `toneShift` | pitch-shifted timbre: the acoustic model sees shifted f0, the vocoder the real f0 |
| `breathiness` | offset in dB on the predicted breathiness |
| `voicing` | offset on the predicted voicing |
| `tension` | the bank's tension input when it has one, else gain and a presence tilt |
| dynamics (`build_dynamics`, note velocity) | gain, and the bank's energy input when it has one |

### Environment variables

| Variable | Meaning |
|---|---|
| `SYNTH2_ENGINE` | default engine spec |
| `SYNTH2_QUALITY` | default quality preset |
| `SYNTH2_MODELS` | models directory |
| `SYNTH2_NEURAL_BANK` | default voicebank id |
| `SYNTH2_NEURAL_STEPS` | diffusion steps (overrides the quality preset) |
| `SYNTH2_NEURAL_THREADS` | onnxruntime threads per session |
| `SYNTH2_NEURAL_CACHE` | `0` turns the phrase cache off |

## Models

Model weights are never part of the Synth2 package or repository. `synth2 models install`
downloads them from their official release pages into the per-user models directory:

```
<models_dir>/<id>/synth2-model.json     manifest
<models_dir>/<id>/files/...             the unmodified contents of the downloaded archive
```

`models_dir` is `$SYNTH2_MODELS`, else `%LOCALAPPDATA%\synth2\models` on Windows, else
`${XDG_DATA_HOME:-~/.local/share}/synth2/models`.

### Manifest (`synth2-model.json`)

```json
{
  "format": "synth2-model",
  "version": 1,
  "id": "qixuan",
  "kind": "voicebank",
  "name": "绮萱 (Qixuan)",
  "model_version": "2.7.0",
  "engine": "diffsinger",
  "root": "files/Qixuan_v2.7.0_DiffSinger_OpenUtau",
  "vocoder": "pc-nsf-hifigan",
  "languages": ["zh", "ja", "en"],
  "license": "Qixuan DiffSinger voicebank Terms of Use",
  "license_file": "files/Qixuan_v2.7.0_DiffSinger_OpenUtau/Terms_of_Use.zh-CN.pdf",
  "license_url": "https://github.com/yqzhishen/qixuan-diffsinger/tree/main/terms_of_use",
  "commercial_use": true,
  "terms": ["..."],
  "credits": ["..."],
  "homepage": "https://github.com/yqzhishen/qixuan-diffsinger",
  "source": "https://github.com/.../Qixuan_v2.7.0_DiffSinger_OpenUtau.zip",
  "sha256": "...",
  "default": true,
  "external": false,
  "installed_at": "2026-01-01T00:00:00Z"
}
```

- `kind` is `voicebank` or `vocoder`. `root` is the folder with `dsconfig.yaml`
  (voicebank) or `vocoder.yaml` (vocoder), relative to the model folder. It may be an
  absolute path for a model registered in place with `synth2 models add` (`external:
  true`); removing such a model deletes only the manifest folder.
- `vocoder` names the vocoder model a voicebank uses.
- `commercial_use` is `true`, `false` or absent (unknown). `terms` is a short plain
  summary of what the licence asks. `credits` are the lines a published work should carry.

### Which vocoder a voicebank uses

1. `dsvocoder/vocoder.yaml` inside the voicebank folder, when the bank ships its own.
2. The manifest's `vocoder` id.
3. An installed vocoder whose `vocoder.yaml` `name` equals the `vocoder` field of the
   bank's `dsconfig.yaml`.
4. The default vocoder.

### Licences

Each model keeps its own licence, and Synth2 only links to the official downloads.

- The installer shows the model's licence summary and requires `--accept-license` (or an
  interactive yes) before it downloads anything.
- Archives are verified against a pinned SHA-256 and extracted without following links
  or leaving the target folder. The files are stored unmodified: Synth2 does not convert,
  re-export or patch model weights.
- `synth2 models list`, the `list_engines` tool, the studio and every render report show
  the credits a model asks for and whether commercial use is allowed.

The default models:

| Model | Licence | Notes |
|---|---|---|
| 绮萱 (Qixuan) v2.7.0 voicebank, DiffSinger Community | Qixuan Terms of Use | Outputs may be used commercially and non-commercially. State that the vocals are AI-synthesised and credit "绮萱 (Qixuan)" and "DiffSinger". No offensive, violent, pornographic, political-propaganda or religious content. Outputs must not be used to train voice synthesis or conversion models. Weights must not be modified or converted, and may only be redistributed complete, unmodified and free with the terms. |
| PC-NSF-HiFiGAN 44.1 kHz 2025.02 vocoder, OpenVPI | CC BY-NC-SA 4.0 | Non-commercial. Credit the DiffSinger Community Vocoder Project. For commercial work install a vocoder with a licence that allows it. |

Other DiffSinger voicebanks in OpenUtau format can be installed with `synth2 models
install --from-file <zip>` or registered in place with `synth2 models add <folder>`.

## Tests

- `tests/conftest.py` pins `SYNTH2_ENGINE=formant`, so the suite never depends on
  installed models. Neural tests ask for `engine="neural"` explicitly.
- `tests/test_engine_select.py`: engine specs, resolution order, project fields, model
  manifests, availability.
- `tests/test_neural_unit.py`: alignment, phoneme mapping, resampler, cache keys, presets
  and effects. No models needed.
- `tests/test_neural_models.py`: the installer against local fixture archives (checksum
  mismatch, path traversal, links, licence gate, atomic install, removal).
- `tests/test_neural_integration.py`: real renders. Skipped unless `onnxruntime` and the
  models are installed (`SYNTH2_MODELS` points at them).
