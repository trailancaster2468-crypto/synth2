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
- `role`: the track's role in the mix ("" for none), and `neural`: what was sung and how
  (see "Rendering pipeline").

`render_project`'s report has `engines` (sorted names of the engines used) and, when any
model asks for attribution, `credits`.

## Package layout: `synth2/neural/`

| Module | Role |
|---|---|
| `__init__.py` | `available(bank=None) -> (ok, reason)`, `backend(bank=None)`, `status()`, `credits(bank=None)`, `refresh()`. Cheap to import, never raises. |
| `models.py` | Where models live and which are installed (`models_dir`, `InstalledModel`, `read_manifest`, `installed`, `find`, `vocoder_for`), plus the installer. |
| `cli.py` | The `synth2 models` commands. |
| `voicebank.py` | Loads a DiffSinger voicebank folder: `dsconfig.yaml` files, phoneme and language tables, speakers and voice modes, ONNX sessions (lazy and shared, or fresh and seeded; thread count from `SYNTH2_NEURAL_THREADS`). |
| `phonemes.py` | Maps Synth2 phonemes to a voicebank's phoneme set per language, with fallbacks for phonemes the bank lacks. |
| `align.py` | Pure functions from the planner's timed units to what the models are fed: phrases, word groups (vowel on the note onset), breaths, consonant classes and caps, frame durations, slow-motion grids, the burst squeeze, the voicing floors, the breath curve, the phrase seed. |
| `pitch.py` | The bank's pitch predictor laid under Synth2's pitch controls (`BankPitch`): note inputs, transitions and vibrato owned by the user, repair of outlier frames, fallback to Synth2's curve. |
| `engine.py` | `Runner` (the models of one bank for one phrase: durations, pitch, variance, acoustic, vocoder), `Options` and `synth_phrase`, the phrase pipeline. `ENGINE_VERSION` is part of every cache key. |
| `adlib.py` | Ad-lib mode: a track sung as detached hits with a designed pitch, one stamped take per distinct sound. |
| `presets.py` | `VoicePreset` per Synth2 voice id, the `neural.*` voice parameters, the dial (`DIALS`), what a role changes in how a track is sung (`track_preset`). |
| `fx.py` | Character effects for non-human voices (`alien`, `robot`, `radio`, `crush`, `double`). numpy only. |
| `resample.py` | Windowed-sinc polyphase resampler (numpy only) for project sample rates other than the model's. |
| `cache.py` | Phrase cache on disk and `stable_hash`, the key of everything a phrase's audio depends on. |
| `renderer.py` | `NeuralBackend` (the `VoiceBackend`), `get_backend(bank=None)`, `reset()`, `load_bank`: one track from plan to audio on the project's sample grid; a double as a copy of another track's take. |

The vocal chain is not in this package: `synth2/engine/vocal_chain.py` (numpy only) has the
chain, the roles and the copies made from a take, and the mixer (`synth2/engine/mixing.py`)
applies them. See "Roles and the vocal chain".

Rules:

- `import synth2` and `import synth2.neural` must not import `onnxruntime`, `yaml` or any
  model. Heavy imports happen inside functions, on the first neural render.
- A broken or half-installed neural engine must never break the formant engine. Checks
  return `(False, reason)`; only an explicit `neural` request raises.
- numpy is the only array dependency. No scipy, no torch, no librosa at runtime.
- A render is reproducible: see "Determinism and the cache".

### Rendering pipeline

For one track (`renderer.py`):

1. `plan_track` gives timed units, exactly as for the formant engine, so lyrics, phoneme
   overrides, `-`, `+`, `br` and note attributes behave the same.
2. `align.build_phrases` cuts the plan into phrases at silences of 0.3 s or more (a phrase
   longer than 20 s is cut at its longest gap). Only the phrases that can reach the render
   window go on.
3. Per phrase, `align.phrase_groups` converts the phonemes to the bank's symbols and builds
   word groups whose vowels start on the note onsets. A gap of 0.12 s or more inside a
   phrase and the silence before a phrase get a breath (`AP`): the whole gap up to 0.45 s,
   the last 0.3 s of a longer one. No breath is put where an ad-lib track of the same voice
   sounds.
4. The duration model predicts phoneme lengths; `align.align` fits them to the note grid.
   The consonant caps of the dial shorten the consonants that share a note with a vowel
   (never under half the predicted length, and only where the consonants take more than
   their share of the note); consonants after a rest keep their predicted length. A breath
   left shorter than 0.1 s becomes a rest. The phrase is laid on the track's frame grid as
   `[8 frames SP][phonemes][10 frames SP]`.
5. Pitch: the bank's pitch predictor (`pitch.BankPitch`) at an expression that follows
   the style (robotic 0.5, soft 0.75, the others 1.0), with Synth2's curve kept around
   notes that carry `transition` / `overshoot` attributes, the user's vibrato on notes with
   vibrato attributes (and everywhere in the robotic style), the predicted vibrato scaled
   by the style and `vibratoEnv`, outlier frames bridged. A bank without a usable
   `dspitch/`, `neural.expr=0`, `SYNTH2_NEURAL_PITCH=expression` or a phrase the predictor
   cannot sing use Synth2's `build_pitch` curve. `pitchDelta` and the detune are added.
6. `engine.synth_phrase`: notes shorter than 0.25 s are sung in slow motion (up to 2 times
   slower) by the variance and acoustic models; with the burst option stops and affricates
   are sung at least 90 / 120 ms long (up to 4 times slower). The variance model predicts
   breathiness, voicing and what else the bank has. Then: a floor under the voicing of
   vowels (6 dB under the phrase's running median), floors under the voicing of voiced
   consonants (per class, under the vowels around them), the breath curve on `AP` frames
   (breathiness peaking at -44 dB, voicing off), the offset on the breathiness of vowel
   frames. The acoustic model gives the mel; slow-motion frames are squeezed back onto the
   real grid, the stops of the burst option are cut out around their release instead. The
   vocoder gets the real f0, so timing and pitch are as written. An optional gain per
   consonant class follows.
7. The phrases are overlap-added (10 ms fades) at the bank's sample rate; level, the
   `loudness` curve and note velocities are applied as gain over absolute time, then the
   character effects, then the resampler puts the window on the project's sample grid.

A phrase's audio does not depend on the render window, so a partial render equals the same
span of the full render, and a phrase whose inputs did not change is served from the cache.

The report of a track carries `bank`, `credits`, `role` and `neural`: `pitch` (`bank`,
`expression`, `adlib` or `copy`), `dial`, `detune`, `expr`, `steps`, `seeded`, the resolved
`preset`, counts (`phrases`, `cached`, `pitch_fallbacks`, `pitch_repaired`, `failed`),
`alignment` (per phrase: first frame, phonemes, their lengths in frames, which start a
note; `frame_s` is the frame length), `hits` in ad-lib mode and the seconds spent per model.

### The dial

One named setting decides how the consonants are sung. The user's choice by ear is
`crisp`, the default.

| | `smooth` (0) | `balanced` (1) | `crisp` (2, default) |
|---|---|---|---|
| consonant caps, ms: stop / vstop / aff / fric / vfric / nasal / glide | 35 / 35 / 60 / 50 / 50 / 60 / 50 | 45 / 40 / 70 / 65 / 50 / 60 / 50 | 50 / 45 / 80 / 80 / 60 / 70 / 60 |
| the caps bind where a note's consonants take more than | always | 35 % of the note | 40 % of the note |
| voicing floor under the vowels, dB: vstop / vfric / nasal, glide | 11 / 8 / 8 | 14 / 8 / 8 | 16 / 11 / 8 |
| stops and affricates sung long and cut out around their release | no | no | 90 / 120 ms |
| the lead's chain (`role=lead`) | full rider, 2:1 leveller | lighter rider, 1.25:1, sibilant lift +2 dB | lightest rider, no leveller, transient emphasis, lift +2.5 dB |
| the lead's reverb send (`role=lead`) | x 2 | x 1 | x 0.6 |

A track picks it with `neural.dial` (0, 1, 2; the Python API also takes the names),
`SYNTH2_NEURAL_DIAL` (a name or a number) sets it for the tracks that do not. Every part
can be overridden by its own parameter. Consonant classes: `stop` (p t k), `vstop` (b d g
jh), `aff` (ch ts), `fric` (s sh f th h), `vfric` (z zh v dh), `nasal` (m n ng), `glide`
(l r w y).

### Voice parameters

`Track.voice_params` keys (numbers; in a score `voice.neural.<name>=v`). The first column
of defaults is what a track gets when it sets nothing.

| Parameter | Default | Meaning |
|---|---|---|
| `neural.gender` | per voice | the acoustic model's formant shift: +1 is the top of the bank's key-shift range (brighter), -1 the bottom; kept within 1.2 |
| `neural.tone` | per voice | semitones the timbre is shifted while the pitch stays as written (12: the acoustic model sings an octave above the rendered pitch) |
| `neural.level_db` | per voice | fixed gain of the voice |
| `neural.breath_db` | 0 | dB added to the predicted breathiness |
| `neural.vowel_breath_db` | -3 | dB added to the breathiness of vowel frames only (half on a vowel's first and last frame) |
| `neural.voicing_db` | 0 | dB added to the predicted voicing |
| `neural.tension` | 0 | offset on the predicted tension (banks that have one) |
| `neural.velocity` | 1 | articulation speed of consonants (0.5..2) |
| `neural.detune` | 7 where the bank's pitch predictor sings, else 0 | cents added to the sung pitch (the default voicebank's predictor sings about 7 cents flat) |
| `neural.expr` | by style | how much of the bank's learned pitch expression is used (0.5..1; 0 = Synth2's own pitch curve) |
| `neural.slow` | 0.25 | notes shorter than this many seconds are sung in slow motion (0 = off) |
| `neural.voicing_floor` | 6 | dB a vowel's voicing may fall under the phrase's running median (0 = off) |
| `neural.dial` | 2 | the dial: 0 smooth, 1 balanced, 2 crisp |
| `neural.cons_cap`, `neural.cons_cap.<class>` | the dial's | seconds: the longest a consonant (of a class) sharing a note with a vowel may be (0 = no cap) |
| `neural.cons_cap_share` | the dial's | the caps bind only where a note's consonants take more than this share of the note (0 = everywhere) |
| `neural.cons_cap_keep` | 0.5 | no cap takes a consonant under this share of its predicted length |
| `neural.onset_keep` | 1 | consonants after a rest or a breath keep their predicted length |
| `neural.cons_floor`, `neural.cons_floor.<class>` | the dial's | dB under the vowels the voicing of a voiced consonant may fall (0 = off) |
| `neural.burst`, `neural.burst.<class>` | the dial's | seconds a stop (`stop`) or affricate (`aff`) is sung in slow motion before it is cut out around its release (0 = off, `neural.burst=1` = 0.09 / 0.12) |
| `neural.cons_gain.<class>`, `neural.cons_gain.hf` | none | dB on a consonant class after the vocoder; with `hf` (Hz) only above that frequency |
| `neural.breaths` | 1 (0 for the roles double, harmony, answer) | breaths between lines: 0 = none, 1 = as designed, other values scale their level (0.5 = 6 dB quieter) |
| `neural.adlib`, `neural.adlib.<option>` | 0 (1 for the role adlib) | ad-lib mode and its options, see below |
| `neural.mode.<name>` | none | weight of a voice mode of the bank (several = a blend) |
| `neural.fx.<preset>` | none | amount (0..1) of a character effect (several = a chain) |

The voice parameters of the formant engine keep a meaning: `formant_scale` moves the
formant shift, `breathiness` the breathiness offset, and `robotic` lowers the pitch
expression.

Voice presets (`presets.PRESETS`): formant shift, timbre shift and level per built-in
voice. The low voices sing with the acoustic model an octave above the rendered pitch,
which is what lets a soprano bank keep its words below about A3. The levels bring a line
in the middle of each voice's range to the level the formant engine sings it at.

| voice | `gender` | `tone` | `level_db` |
|---|---|---|---|
| `nova` | 0 | 0 | 1.5 |
| `aurora` | 0 | 0 | 2.0 |
| `seraph` | 0.1 | 0 | 0.5 |
| `pixel` | 0.35 | 0 | 0 |
| `atlas` | -0.3 | 12 | 4.5 |
| `ember` | -0.7 | 12 | 6.0 |
| `basso` | -0.6 | 12 | 8.5 |

### Parameter curves

| Synth2 parameter | Neural engine |
|---|---|
| `pitchDelta` | added to the sung pitch (acoustic model and vocoder) |
| `vibratoEnv`, note vibrato and transition attributes | through `pitch.BankPitch` (or `build_pitch`) into the pitch |
| `loudness` | gain ride in dB |
| note `velocity` | gain from the note's onset consonants to the next note's |
| `gender` | the acoustic model's formant shift (`gender` input) |
| `toneShift` | pitch-shifted timbre: the acoustic model sees the shifted f0, the vocoder the real f0 |
| `breathiness` | 12 dB per unit on the predicted breathiness |
| `voicing` | up to 40 dB off the predicted voicing |
| `tension` | the bank's tension input when it has one, else 1.5 dB of gain per unit |

In ad-lib mode the timbre curves are read once per hit, at its start.

### Ad-lib mode (`adlib.py`)

Laughs, shouts, sound words and chant syllables sung as lines come out as a slur: the
words glide into each other, a breath is drawn before them and the pitch predictor sings a
melody over them. With `neural.adlib=1` (or `role=adlib`) every hit is a take of its own
between rests, with a designed pitch: the written pitch held flat from before the first
consonant to after the last. A take is as long as its notes are written; the planner's
humanised times place it. With `stamp` the take of each distinct sound is made once, with
a seed derived from the sound and not from its place, and every repetition is a copy of
the same samples. There are no breaths, and vowel breathiness and consonant floors are
the bank's own unless the track sets them.

| `neural.adlib.<option>` | Default | Meaning |
|---|---|---|
| `split` | 0 | what a hit is: 0 a word (a note and its `+` notes), 1 every note, 2 a run of touching notes |
| `bend`, `bend.<lyric>` | 0 | semitones the pitch falls (negative) or rises (positive) from `bend_from` of the hit to its end; per lyric with `bend.<lyric>` |
| `bend_from` | 0.3 | where the bend starts, as a share of the hit |
| `glide` | 0.04 | seconds of pitch transition between the notes of one hit |
| `vibrato` | 0 | 1: the voice's designed vibrato (rate and depth of the voice or the note) |
| `stamp` | 1 | one take per distinct sound; 0: every hit a fresh take |
| `salt`, `salt.<lyric>` | 0 | another take of every sound, or of one lyric: a stamped hit repeats one take, so the take is worth choosing |
| `tight` | 1 | a hit that touches the next one ends where that hit's onset consonants start |
| `tight_onset` | 0.4 | the most of its first note the onset of a hit in a touching run may take |
| `onset_max` | 0 | seconds: the most the onset of a free-standing hit may take (0 = as predicted) |
| `onset_cap.<class>` | 25 to 70 ms | the most one onset consonant of a class takes in a touching run |
| `onset_floor.<class>` | none | the least it gets there |
| `cons_cap.<class>` | none | consonant caps inside a hit |
| `gap`, `hold` | 0.02, 0.07 | seconds of silence a tight hit that ends in a vowel leaves before the next onset; seconds it is sung past its end before the cut |
| `run_gap`, `accent.<n>` | 0.25, none | hits closer than `run_gap` s form a run; `accent.0`, `accent.1`, ... are the dB of the hits of a run (cycled) |
| `pre`, `release` | 0.012, 0.05 | seconds kept before the first phoneme and after a final consonant |
| `chop_hz`, `chop_db` | 0 | a loudness chop locked to the start of every take |
| `glottal` | 0 | 1: a hit that starts with a vowel gets a closure before it |

### Roles and the vocal chain (`synth2/engine/vocal_chain.py`)

A track may say which role it has in the mix: `Track.role`, `role=` on a score's track
line, stored in project JSON only when set. A track without a role is sung and mixed as
before. For a track the neural engine sings, the role does two things.

How it is sung (`presets.track_preset`, `renderer.py`):

| role | sung |
|---|---|
| `lead` | as any track |
| `harmony`, `answer` | without breaths of their own |
| `adlib` | in ad-lib mode |
| `crowd` | as any track |
| `double` | not sung: its audio is the take of the track it doubles (the leads of its voice, or the track named in `role="double:NAME"`), from 60 ms before to 120 ms after each of the double's notes, with the double's `loudness` curve. Two voices from one take have one timing, so the consonants do not blur. A double with nothing to double sings its own notes without breaths. |

How it is mixed (`vocal_chain.apply_role`, called by `mixing.mix_sources` in place of the
vocal channel strip):

| role | chain | then |
|---|---|---|
| `lead` | `lead_chain(dial)` | its reverb send is scaled by the dial (x 2, x 1, x 0.6) |
| `answer` | `lead_chain(dial)` | a top shelf of -1.5 dB, a small room |
| `harmony` | `HARMONY`: the smooth chain with the consonants pushed back and a darker top | -3 dB at 2.8 kHz, a small room |
| `double` | `HARMONY` | `adt`: 18 ms late (27 ms on the right), a slow wobble, 6 cents sharp (5 flat on the right), darker; a double of a harmony 22 ms late and 7 cents flat |
| `adlib` | the lead's static EQ only | band-limited to 330-5200 Hz, 2 dB down while the lead sings, one repeat 95 ms later |
| `crowd` | none | darker, 3 dB down while the lead sings, a room |

The chain (`vocal_chain.process`) reads the sung level only: `eq`, `deess`, `rider` (slow
gain ride toward the take's median), `comp` (noise frames get the gain of the sung frames
next to them), `cons` (consonant leveller or limiter), `lift` (sibilants up, above a
frequency, up to a ceiling), `trans` (transient emphasis), `breaths` (quiet noise in the
gaps is not lifted), `crest`; also `bursts` and `expand`. Its output has the input's
active level. `lift` finds its frames with the spectral-tilt detector; `cons` and `bursts`
use the periodicity detector unless a stage sets `"detector": "tilt"`.

The backend tells the mixer about a role with `vocal_chain.note_role(track id, audio,
...)`; the mixer uses the note only for the audio it was left for, so a track sung by the
formant engine, or with `mix_overrides` `chain: false`, or in a project with
`master.vocal_chain` off, goes the way it always did.

### Determinism and the cache

The diffusion models draw their noise inside the graph. In reproducible mode (the
default) every phrase runs them in sessions created right after `onnxruntime.set_seed`
with a 31-bit seed made from the track seed and the phrase's first frame and length
(`align.phrase_seed`), not from its notes or lyrics. The same project gives the same
samples in any order of rendering, whole or in parts, and editing a note leaves the rest
of the phrase's take in place as long as the phrase keeps its place and length. Another
track seed is another take. A stamped ad-lib take is seeded by its sound (phonemes, frame
lengths, pitches, pitch shape) and its salt instead of its place.
`SYNTH2_NEURAL_REPRODUCIBLE=0` uses the bank's shared sessions: faster (no session per
phrase), every render another take.

The phrase cache (`cache.py`) stores what the vocoder returned (after the consonant gain,
before level, gain rides and effects) under a key of everything the waveform depends on:
`ENGINE_VERSION`, the bank's identity, the voice-mode blend, every option of
`engine.Options`, the phonemes, frame lengths, vowel flags, slow-motion factors, the pitch
inputs, `pitchDelta` and detune, timbre shift, formant shift, variance offsets and the
seed. Raise `ENGINE_VERSION` when the same inputs give other audio.

### Environment variables

| Variable | Meaning |
|---|---|
| `SYNTH2_ENGINE` | default engine spec |
| `SYNTH2_QUALITY` | default quality preset (`draft` 5, `normal` 10, `high` 20 acoustic diffusion steps) |
| `SYNTH2_MODELS` | models directory |
| `SYNTH2_NEURAL_BANK` | default voicebank id |
| `SYNTH2_NEURAL_DIAL` | the dial of tracks that do not set `neural.dial`: `smooth`, `balanced`, `crisp` or 0, 1, 2 |
| `SYNTH2_NEURAL_STEPS` | acoustic diffusion steps (overrides the quality preset) |
| `SYNTH2_NEURAL_PITCH` | `bank` (default) or `expression`: Synth2's own pitch curve for every track |
| `SYNTH2_NEURAL_REPRODUCIBLE` | `0` / `false` / `off` / `no`: shared sessions, unseeded noise |
| `SYNTH2_NEURAL_THREADS` | onnxruntime threads per session (default: up to 4) |
| `SYNTH2_NEURAL_CACHE` | `0` turns the phrase cache off |
| `SYNTH2_NEURAL_CACHE_MB` | size limit of the phrase cache folder (default 2048) |
| `SYNTH2_CACHE` | cache folder (phrases go into `phrases/` inside it) |

### What is not implemented

- The chain reads the levels of the span it is given. A region render through a role's
  chain is levelled against the region, so it can differ a little from the same bars of a
  full render (the dry track does not).
- The consonant limiter of the lead's chain uses the periodicity detector, which misses
  short bursts and sibilants between vowels; stop bursts in slow passages at `balanced`
  and `crisp` measure 9 to 10 dB over the bank's sample songs. `neural.cons_gain.stop`
  with `neural.cons_gain.hf` takes them down at the source.
- No duck of the doubles and the harmony keyed by the lead's own stop closures.
- The crowd's second takes (each voice doubled by a detuned, late take of its own) are
  not made by the role; write a second track.
- In ad-lib mode the timbre curves are read at each hit's start, not along it, and the
  pitch predictor is not used.
- A double copies the neural take of its source track, whatever engine that track is set
  to in the project.
- The voice presets (formant shift, level, the 7 cents of detune) are measured on the
  default voicebank; another bank gets the same numbers.
- The settings were chosen with speech recognisers and level measures and one round of
  listening (the dial). The research notes in `docs/research/neural-engine/` have the
  evidence and what is still wrong.

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
- `tests/test_neural_unit.py`: alignment, breaths, consonant options, the dial, presets,
  phoneme mapping, the loader and the backend (windows, seeds, cache keys, ad-lib mode,
  roles) against a fake `onnxruntime`. No models needed.
- `tests/test_neural_util.py`: resampler, phrase cache, effects.
- `tests/test_vocal_chain.py`: the chain's stages, the detectors, role presets, the copy
  of a take, the mixer's use of a role, the `role` field in projects and scores.
- `tests/test_neural_models.py`: the installer against local fixture archives (checksum
  mismatch, path traversal, links, licence gate, atomic install, removal).
- `tests/test_neural_integration.py`: real renders. Skipped unless `onnxruntime` and the
  models are installed (`SYNTH2_MODELS` points at them).
