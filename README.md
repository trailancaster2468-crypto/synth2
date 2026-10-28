# Synth2 Vocal Studio

**An AI-native singing voice synthesizer and vocal production studio.**

Synth2 does the job of a desktop vocal synthesizer such as Synthesizer V
Studio: write notes and lyrics, shape the performance (pitch, vibrato,
dynamics, tension, breathiness, gender), arrange several vocal tracks, mix,
and render finished vocals to WAV. Every studio capability is also exposed
as **structured tools**, so an AI model can design, perform and produce full
vocal tracks at the same level a human does in the editor. The AI can also
check its own output with built-in audio analysis.

```
            Web studio (piano roll)    CLI    MCP server    AI Director (Claude)
                         \               |         |              /
                          +------ synth2.ai.tools (one tool layer) -----+
                                         |
       project model  ·  score notation  ·  SVP / MIDI / MusicXML / UST  ·  engine
                                                                             |
        lyrics → phonemes → timing planner → expression → vocoder → mixer → WAV
```

> **About quality.** Commercial products ship neural voice models trained on
> licensed recordings of real singers. Those can't be bundled here. The
> built-in engine is a fast parametric singer: a pulse-synchronous
> source-filter vocoder with formant-synthesized vocal tracts and a full
> automatic-expression model. It is clear, controllable and fully offline,
> but it sounds synthetic, somewhere between a classic vocaloid and a
> formant synth. The engine sits behind a backend interface
> (`synth2/engine/backend.py`), so a neural acoustic model and vocoder (for
> example an ONNX DiffSinger-style voicebank on `onnxruntime`) can be added
> later without changing projects, tools or the GUI.

## Features

| Area | What you get |
|---|---|
| **Editing** | Notes with pitch, timing and lyrics. Per-note phoneme override and expression attributes (vibrato depth, rate and start, transition time, overshoot, consonant length, tension, breathiness, detune, velocity, language). Multiple tracks. Tempo map. Time signature and key. Undo and redo. |
| **Lyrics** | English (CMU dictionary plus letter-to-sound fallback, automatic syllable splitting across notes) and Japanese (kana or romaji). `-` melisma, `+` word continuation, `br` breath, `[s t aa r]` explicit phonemes. |
| **Automatic expression** | "AI tuning" styles (`pop`, `ballad`, `rock`, `jazz`, `rnb`, `soft`, `classical`, `robotic`, `flat`). Portamento with preparation and overshoot, phrase scoops and falls, delayed natural vibrato, pitch drift, phrase dynamics, pitch-dependent tension, automatic breaths. |
| **Parameter lanes** | `pitchDelta` (cents), `loudness` (dB), `tension`, `breathiness`, `gender`, `voicing`, `toneShift`, `vibratoEnv`, with linear, cubic or step interpolation. |
| **Voices** | `aurora` (bright female pop), `seraph` (soft breathy female), `atlas` (male tenor), `ember` (warm baritone), `nova` (synthetic vocaloid-style idol), `pixel` (young, high), `basso` (bass). Each can be tweaked per track, and custom voices can be loaded from JSON. |
| **Mixer** | Gain, pan, mute and solo per track. Reverb send to a synthetic stereo hall. Master gain, normalization and limiter. Per-track stems. |
| **AI** | A tool layer with JSON schemas, an MCP server for Claude Desktop, Claude Code and other clients, a Claude-powered **Director** that turns natural language into productions, a compact score language for writing songs, harmony generation, and objective render checks (pitch accuracy in cents, levels, clipping, voicing). |
| **Interop** | Import and export of Synthesizer V `.svp`, MIDI (with lyric events), MusicXML and `.mxl`, UTAU `.ust`, Synth2 JSON and score text. |
| **Studio GUI** | A browser studio with a canvas piano roll, parameter lane drawing, inspector, mixer, transport with playback, and an AI chat panel. |

## Install

```bash
pip install -e .            # numpy + cmudict
pip install -e ".[ai]"      # + anthropic SDK for the AI Director
```

Python 3.10 or newer. No compiled code and no GPU are needed. Rendering is
several times faster than real time on a laptop CPU.

## Quick start

```bash
synth2 demo -o out/demo                     # render the bundled demo songs
synth2 render examples/neon_hearts.s2s -o out/neon.wav --check
synth2 studio --open                        # web studio on http://127.0.0.1:8765
synth2 phonemize "twinkle twinkle little star"
synth2 voices && synth2 styles
synth2 convert song.svp song.s2s            # Synthesizer V -> score notation
```

### Score notation (what an AI writes)

```text
title: Neon Hearts
tempo: 112
time: 4/4
key: C major

track: Lead voice=aurora style=pop lang=en
r w
E4 e city  E4 e +  G4 q lights  G4 e are  A4 e calling  G4 e +  E4 e out  D4 e my | E4 h. name  r e
C5 q neon{tension=0.3}  B4 e +  G4 q. hearts  r e  A4 e we | C5 q light  D5 e up  C5 e the  E5 h sky{vibrato_depth=0.45}
```

A note is `PITCH DURATION LYRIC[{attr=value,...}]`. Durations are `w h q e s t`,
with dots (`q.`), triplets (`e3`), ties (`h+q`) or plain beat numbers.
`r DUR` is a rest and `|` is a visual bar line. See `docs/PLAN.md` for the full
grammar and `examples/` for complete songs.

### Python

```python
from synth2 import Project, render_project
from synth2.phonemes import split_lyrics

p = Project.new("Demo", bpm=100, voice="atlas")
lead = p.tracks[0]
for i, (pitch, lyric) in enumerate(zip(["C4", "E4", "G4", "E4"], split_lyrics("hello my friend"))):
    lead.add_note(1 + i, 1, pitch, lyric)
lead.curve("tension").add(1, 0.0)
lead.curve("tension").add(5, 0.6)

result = render_project(p)
result.save("demo.wav")
print(result.report)
```

## AI integration

<!-- AI-SECTION -->

## Architecture

| Module | Role |
|---|---|
| `synth2/project.py` | Project, Track, Note, Curve model and JSON serialization (time in beats) |
| `synth2/phonemes/` | Phoneme inventory with acoustic targets, English and Japanese G2P, lyric resolution |
| `synth2/voices/` | Parametric singer presets and custom voice loading |
| `synth2/engine/planner.py` | Notes and lyrics to timed phoneme segments (vowel on the beat, consonants before it) |
| `synth2/engine/expression.py` | Automatic pitch and dynamics performance styles |
| `synth2/engine/params.py` | Segments to frame-level formant, voicing and noise tracks, with coarticulation |
| `synth2/engine/vocoder.py` | Pulse-synchronous minimum-phase harmonic synthesis, plus HNR-shaped breath, aspiration and frication noise |
| `synth2/engine/renderer.py` | Track rendering, mixing, reverb, limiter, stems and render reports |
| `synth2/engine/backend.py` | Voice backend interface (plug in neural voices here) |
| `synth2/analysis.py` | YIN pitch tracking, level analysis, pitch accuracy versus the score |
| `synth2/score.py` | Text score notation, parse and serialize |
| `synth2/io/` | WAV, SVP, MIDI, MusicXML and UST |
| `synth2/ai/` | Tool registry, composer helpers, MCP server, Claude Director |
| `synth2/server/` | Web studio, REST API and static frontend |

## Development

```bash
pip install -e ".[dev,ai]"
pytest -q
```

## License

MIT
