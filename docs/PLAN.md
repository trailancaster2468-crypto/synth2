# Synth2 Vocal Studio: build plan and interface contracts

Synth2 is an AI-native singing voice synthesizer studio. It does the job a
desktop vocal synth like Synthesizer V Studio does: write notes and lyrics,
shape the performance (pitch, vibrato, dynamics, tension, breathiness, gender),
mix several vocal tracks, and render finished vocal audio. The difference is
that every capability is also exposed as structured tools, so an AI model can
compose, edit, render and check vocals at the same level a human uses the GUI.

## Goals

1. **Studio parity (the feature set):** piano-roll note editing, lyrics with
   automatic phonemes, phoneme overrides, per-note expression, parameter
   automation lanes, multiple tracks and voices, mixer (gain, pan, mute,
   solo, reverb), tempo map, and WAV export with stems.
2. **AI parity:** whatever the GUI can do, an AI can do through the tool layer
   (`synth2.ai.tools`). The same tool layer backs the MCP server, the
   Claude-powered "director" chat, the REST API and the CLI.
3. **Quality:** natural-sounding automatic expression (portamento, vibrato,
   scoops, falls, drift, phrase dynamics) and a clean, artifact-free vocoder.
4. **Interop:** import and export of Synthesizer V `.svp`, MIDI, MusicXML,
   UTAU `.ust` and a compact text score format.
5. **Zero-friction install:** runtime dependencies are `numpy` and `cmudict`
   only; `anthropic` is optional (director only). Pure Python, no compiled code.

## Honest quality note

Commercial products ship neural voice models trained on licensed recordings of
real singers. We cannot download or train those here. The built-in engine is a
parametric source-filter / pulse-synchronous vocoder (WORLD-style overlap-add
with formant-synthesised spectral envelopes). It is clear, controllable and
fast, but it sounds synthetic. The engine is built behind a **backend
interface** (`synth2.engine.backend`) so that a neural acoustic model and
vocoder (for example a DiffSinger-style ONNX voicebank via `onnxruntime`) can be
plugged in later without changing the project model, tools or GUI.

## Architecture

```
            +----------------------- interfaces -------------------------+
            |  Web studio (piano roll)   CLI   MCP server   AI director   |
            +------------+--------------+------+-------------+-----------+
                         |   REST       |      |  stdio      | Claude API
                         v              v      v             v
                     synth2.ai.tools  (Session + JSON tool registry)
                         |
         +---------------+----------------+-------------------+
         v               v                v                   v
   synth2.project   synth2.score     synth2.io.*         synth2.engine
   (model, JSON)    (text notation)  (svp/midi/xml/ust)  render_project()
                                                              |
              phonemes (G2P) -> planner -> expression -> params -> vocoder -> effects/mix
```

### Conventions (everyone must follow these)

* **Time** is in **beats** (quarter notes, float) everywhere in the project
  model and tools. Seconds only appear inside the engine and analysis code.
  Use `project.tempo.beat_to_sec()` and `project.tempo.sec_to_beat()`.
* **Pitch** is a MIDI note number (int, C4 = 60). Tools accept note names
  ("C4", "F#3", "Bb5") through `synth2.timing.note_to_midi`.
* **Curves**: `Track.curves[name] = Curve(points=[(beat, value)], interpolation)`.
  Names and ranges are in `synth2.project.CURVE_SPECS`.
* **Note attributes**: `Note.attrs` keys are in `synth2.project.NOTE_ATTRIBUTES`.
* **Lyrics**: `"-"` is a melisma (hold the previous vowel), `"+"` is the next
  syllable of the previous word, `"br"` is a breath, `"[s t aa r]"` gives
  explicit phonemes. See `synth2/phonemes/__init__.py`.
* **Audio** is a float32 numpy array with shape `(n, 2)` in [-1, 1]. Mono
  helpers take shape `(n,)`.
* Python 3.10+, type hints, module docstrings, and no new runtime
  dependencies beyond `numpy` and `cmudict`.
* Tests go in `tests/test_<area>.py` and use pytest. Keep them fast: render
  only short clips, under 3 seconds of audio.

## Module ownership and contracts

### Already written (main thread)
* `synth2/timing.py`: `TempoMap`, `note_to_midi`, `midi_to_note`,
  `midi_to_hz`, `hz_to_midi`, `parse_duration`.
* `synth2/project.py`: `Project`, `Track`, `Note`, `Curve`,
  `MasterSettings`, `CURVE_SPECS`, `NOTE_ATTRIBUTES`, `iter_active_tracks`.
  `Project.load(path)` dispatches on extension to `io.svp.read_svp`,
  `io.midi.read_midi`, `score.parse_score` or JSON.
* `synth2/phonemes/`: `inventory.py` (acoustic phoneme table),
  `english.py` (CMUdict + rules, syllabify), `japanese.py` (kana/romaji),
  `__init__.py` (`resolve_notes`, `split_lyrics`, `lyric_phonemes`).
* `synth2/voices/__init__.py`: `Voice`, `VOICES`, `get_voice`,
  `list_voices`, `load_voice_file`.
* `synth2/engine/expression.py`: `Style`, `STYLES`, `get_style`,
  `list_styles`, `build_pitch`, `build_dynamics`, `evaluate_curve`.
* `synth2/io/wav.py`: `write_wav(path, audio, sr, bits=16)`,
  `read_wav(path) -> (audio (n, ch) float64, sr)`.

### Engine (main thread): `synth2/engine/`
```python
from synth2.engine import render_project, RenderResult
res: RenderResult = render_project(project, *, sample_rate=None, track_ids=None,
                                   start_beat=None, end_beat=None, progress=None)
res.audio        # float32 (n, 2)
res.sample_rate  # int
res.stems        # dict[track_id, float32 (n, 2)]: post-fader, pre-reverb
res.report       # dict (JSON-serialisable): duration_s, peak_db, rms_db, render_time_s,
                 #   realtime_factor, tracks: [{id, name, warnings: [str], notes, phonemes}]
res.save(path, bits=16)            # write master WAV
res.save_stems(directory, bits=16) # write <track name>.wav per stem
render_track(project, track) -> (mono float32 array, report dict)  # dry, unmixed
```
Internals: `planner.py` (notes to timed phoneme segments), `params.py`
(segments to frame parameter tracks), `vocoder.py` (pulse-OLA harmonic plus
noise synthesis), `effects.py` (reverb, pan, limiter), `renderer.py`
(orchestration), `backend.py` (backend interface and registry).

### Analysis (main thread): `synth2/analysis.py`
```python
estimate_f0(mono, sr, hop_s=0.005, fmin=60, fmax=1200) -> (times_s, f0_hz)  # 0 = unvoiced
analyze_audio(audio, sr) -> dict       # peak/rms dB, duration, clipping, voiced ratio
pitch_accuracy(project, result) -> dict  # per track: mean/max abs cents error vs notes
```

### Team A: file formats (`synth2/io/`, except `wav.py`)
* `midi.py`: `read_midi(path) -> Project` and `write_midi(project, path)`.
  Handles SMF format 0 and 1, tempo map, time signature, lyric/text meta
  events (FF 05 / FF 01) mapped to note lyrics, one Track per MIDI track or
  channel with notes, and trimming of overlapping notes into a monophonic
  line. Writes lyrics as FF 05 events.
* `svp.py`: `read_svp(path) -> Project` and `write_svp(project, path)` for the
  Synthesizer V Studio JSON format (blicks: 705600000 per quarter note; `time`
  meter and tempo; tracks, `mainGroup` notes, onset, duration, lyrics,
  phonemes, pitch; parameter curves `pitchDelta`, `loudness`, `tension`,
  `breathiness`, `voicing`, `gender`, `toneShift`, `vibratoEnv` as flat point
  lists; groups and `groups` references flattened on import; mixer gain and
  pan). The `.svp` reader must tolerate a trailing NUL byte and unknown keys.
* `musicxml.py`: `read_musicxml(path) -> Project` (partwise `.musicxml` or
  `.xml`, plus `.mxl` zip). Notes, rests, ties (merged, or the tied note
  becomes a `-` melisma when it carries no lyric), lyrics with syllabic
  begin/middle/end mapped to `word` + `+`, tempo from `<sound tempo>` or
  metronome, divisions, time signature, key. Optional:
  `write_musicxml(project, path)`.
* `ust.py`: `read_ust(path) -> Project` and `write_ust(project, path, track=0)`
  for UTAU (480 ticks per quarter note, `R` rests, Shift-JIS or UTF-8).
* `__init__.py`: `load_any(path) -> Project` and
  `export_any(project, path, fmt=None)`, dispatching by extension
  (`.json` / `.s2p` / `.svp` / `.mid` / `.musicxml` / `.xml` / `.mxl` /
  `.ust` / `.s2s`; `.s2s` goes through `synth2.score`).
* Tests: `tests/test_io.py` round-trips each format through
  `Project.to_dict()` comparisons, using small fixtures generated in the test.

### Team B: AI layer (`synth2/score.py`, `synth2/ai/`)
* `score.py`: the compact text score notation that AI models write. Grammar
  below. `parse_score(text) -> Project`, `project_to_score(project) -> str`
  (round-trips), `parse_notes(text, start_beat=0.0, language="en") -> list[Note]`.
* `ai/tools.py`: `Session` (holds `project`, `path`, `last_render`, and an
  undo/redo history of project snapshots) plus the tool registry:
  `TOOLS: list[dict]` (each `{name, description, input_schema}` as JSON
  Schema) and `call_tool(session, name, args: dict) -> dict` (JSON-serialisable;
  raises `ToolError` with a helpful message on bad input). Every mutating
  tool pushes an undo snapshot first. Required tools:
  `get_project`, `new_project`, `update_project` (title, key, time
  signature, master settings), `set_tempo`, `list_voices`, `list_styles`,
  `list_phonemes`, `phonemize`, `add_track`, `update_track`,
  `remove_track`, `add_notes`, `write_melody` (score-notation snippet into a
  track at a beat), `edit_notes`, `delete_notes`, `set_lyrics` (free text
  distributed over notes in a range via `phonemes.split_lyrics`),
  `transpose`, `quantize`, `set_curve`, `clear_curve`,
  `apply_expression` (style plus overrides, optional per-note attrs),
  `add_harmony` (diatonic harmony track from a source track: interval in
  scale steps, key-aware, voice choice), `render` (WAV path, beat range, track
  subset; returns `report`), `analyze_render` (`analysis.pitch_accuracy` +
  `analyze_audio`), `save_project`, `load_project`, `export`
  (svp/midi/musicxml/ust/score/json), `undo`, `redo`.
* `ai/composer.py`: helpers behind the tools, all pure functions on the
  model: `harmonize(track, steps, key, ...)`, `quantize(notes, grid)`,
  `transpose(...)`, `fit_lyrics(notes, text, language)`, `parse_key(key)`
  giving tonic and scale, and `scale_degrees`.
* `ai/mcp_server.py`: a dependency-free MCP server over stdio (JSON-RPC 2.0,
  newline-delimited; `initialize`, `notifications/initialized`, `ping`,
  `tools/list`, `tools/call` with `content: [{type: "text", text: json}]` and
  `isError`). Entry point `main()`; run with `python -m synth2.ai.mcp_server`.
* `ai/director.py`: a natural-language production assistant using the
  Anthropic Python SDK (optional import) with tool use over `TOOLS`.
  `Director(session, model=None).chat(message) -> {"reply": str, "actions": [...]}`.
  It runs a manual agentic loop until `stop_reason != "tool_use"` and has a
  strong system prompt that teaches the model the studio conventions and score
  notation. Load the `claude-api` skill before writing this.
* Tests: `tests/test_score.py` and `tests/test_tools.py`.

### Team C: web studio (`synth2/server/`)
* `server/app.py`: a stdlib `http.server` (ThreadingHTTPServer) application
  serving `static/`, plus JSON REST endpoints:
  * `GET /api/project` returns the project dict.
  * `PUT /api/project` replaces the project (and pushes an undo snapshot).
  * `GET /api/tools` returns `TOOLS`.
  * `POST /api/tool {name, args}` calls `call_tool` and returns the result, or
    `{error}` with status 400.
  * `POST /api/render {start_beat?, end_beat?, track_ids?}` returns the WAV
    bytes (`audio/wav`) and puts the report in the `X-Render-Report` header
    (JSON); `GET /api/render/last` serves the last WAV.
  * `GET /api/meta` returns voices, styles, curve specs, note attribute specs,
    phoneme inventory and director availability.
  * `POST /api/ai {message}` calls `Director.chat`, or returns 501 when no
    `ANTHROPIC_API_KEY` is set or `anthropic` is missing.
  * `run(host="127.0.0.1", port=8765, project_path=None, open_browser=False)`.
* `static/index.html`, `static/app.js`, `static/style.css`: a vanilla JS
  single-page studio with no build step and no CDN. It has a track list with
  mixer (voice, style, gain, pan, mute, solo), a canvas piano roll (create,
  select, move, resize and delete notes; double-click to edit lyrics; snap
  grid; zoom and scroll; keyboard shortcuts), a parameter lane editor under
  the roll (draw and erase curve points for each curve), a transport (render
  and play with Web Audio, a playhead synced to playback, loop region,
  tempo), a properties panel for the selected note's attributes and phoneme
  override, an "AI Director" chat panel, undo/redo, and import/export
  buttons. The layout is dark and studio-style and still works at about 1280px.
* Tests: `tests/test_server.py` starts the server on an ephemeral port and
  exercises the endpoints.

### Integration (main thread, last)
`synth2/cli.py` (`synth2 render | new | info | studio | mcp | director |
convert | voices | styles | phonemize | score`), `README.md`, `examples/`, an
end-to-end test, a DSP quality review pass by a separate reviewer agent, and
a final code review.

## Score notation (`.s2s`): the language AI models write

```
# comments start with '#'
title: Twinkle
tempo: 112            # bpm (also: "tempo: 120 @ 16" = change at beat 16)
time: 4/4
key: C major

track: Lead voice=aurora style=pop lang=en gain=0 pan=0
C4 q twinkle  C4 q +  G4 q twinkle  G4 q +  |  A4 q little  A4 q +  G4 h star
F4 q how   F4 q I  E4 q wonder  E4 q + | D4 q what D4 q you C4 h are
r q            # rest
E4 e. la  F4 s -   # dotted eighth, then a melisma sixteenth
G4 h+q oh{vibrato_depth=0.5,tension=0.3}   # tied duration, per-note attrs
A4 q [s t aa r]    # explicit phonemes

track: Harmony voice=atlas style=pop
...
```
Rules:
* Multi-syllable words go on their FIRST note with `+` on each following note
  (`twinkle +`), never as fragments (`twin kle`), so the dictionary pronunciation is used.
* A note is `PITCH DURATION LYRIC` separated by whitespace. `PITCH` is a note
  name or a MIDI number, `r` is a rest (`r DURATION`, no lyric). `DURATION`
  is any `timing.parse_duration` token (`w h q e s t`, dots, `3` triplets,
  `+` ties, numbers). `LYRIC` is a single token (`-`, `+`, `br`, a word, kana,
  or `[...]` phonemes, which may contain spaces inside the brackets).
* An optional `{k=v,...}` directly after the lyric sets note attrs (values
  parsed as float when numeric).
* `|` bar lines are ignored for timing. Notes are sequential from the
  track's current position. `@ BEAT` on its own sets the position
  (for example `@ 16`).
* Header keys come before the first `track:` line. A `tempo:` line inside a
  track adds a tempo change at the current position.
* Curve lines inside a track: `curve: loudness 0:0 4:-3 8:0 (linear)`, as
  `beat:value` pairs.

## Execution order

1. Main thread: plan and contracts (this doc), then the engine.
2. In parallel: teams A, B and C, against the contracts above.
3. Main thread: analysis, CLI, integration, README and examples.
4. Reviewer agent: DSP quality critique (spectrogram and pitch-accuracy
   driven) and fixes to the engine.
5. Full test suite, end-to-end demo render, then commit and push.
