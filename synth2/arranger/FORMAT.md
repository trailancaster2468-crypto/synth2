# Backing arrangement format (`project.backing`)

The arranger turns a JSON arrangement into a synthesised band (drums, bass, "guitars",
synths, leads, pads, FX) and hands the mixer one stereo stem per instrument
(`synth2.arranger.render_backing`). It is stored in `Project.backing` and saved as
`"backing"` in the project JSON. AI models edit it with the `set_backing`, `get_backing`
and `clear_backing` tools (`get_backing(detail="reference")` lists every preset, parameter,
drum voice and FX sound).

The arrangement lives in the native project JSON (`.s2p.json`); the `.s2s` score text and the
interchange formats (SVP, MIDI, MusicXML, UST) carry the vocals only.

## Conventions

* **Tempo comes from the project.** The arrangement only speaks in bars and beats.
* **Bars are absolute and 1-based**: bar 1 starts at beat 0, bar `k` at beat `4*(k-1)` in 4/4.
  Every `bar` in sections, segments, fills, events and stops is a song bar number.
* **`beat` inside a bar is 1-based**: `beat: 1` = downbeat, `3` = third quarter, `2.5` = the "and" of 2.
* **Pattern steps are 16ths by default** (`grid: 16` steps per bar). `|` in step strings is ignored
  (use it to mark beats or bars). A step string with spaces is split into tokens, otherwise every
  character is one step.
* **Levels**: presets are loudness-calibrated, so a typical part at `gain_db: 0` already sits in a
  balanced pop mix (kit and bass are the anchor, rhythm guitars and leads a little lower, pads,
  arps and bells further back). Use `gain_db` for taste, usually between -6 and +3.

## Skeleton

```jsonc
{
  "format": "synth2-arrangement", "version": 1,
  "key": "A minor",              // default: the project key; used for scale degrees (2 4 6 and missing 7ths)
  "swing": 0.0,                  // 0..0.75: fraction of a step that off-beat 16ths/8ths are late (0.33 = shuffle)
  "humanize": {"timing_ms": 1.5, "velocity": 0.05},   // optional, per instrument too
  "energy_response": 1.0,        // 0..2: how much section energy changes velocity and filter brightness
  "length_bars": 134,            // optional: explicit song length (default: end of the last section)

  "instruments": {
    "<name>": {
      "preset": "bass_slap",              // pitched instrument (see Presets)  - or -
      "type": "drums", "kit": "pop",      // a drum kit (kits: pop, rock, electro, chip)  - or -
      "type": "fx",                       // an FX rack played by fx events
      "params": {"cutoff": 900, "drive": 0.4},     // synth parameter overrides (see Synth parameters)
      "voices": {"kick": {"pitch": 50}},           // drum kits: per-voice parameter overrides
      "gain_db": 0, "pan": 0,                      // mix defaults handed to the mixer with the stem
      "role": "bass",          // drums|bass|keys|lead|pad|fx|perc (default from the preset)
      "sidechain": true,       // the mixer ducks this stem under the kick (default from the preset)
      "width": 1.0,            // 0..2 stereo width
      "low": "E1",             // register: chord roots are placed at or above this note
      "center": "B3",          // where chord voicings sit (voice leading stays near it)
      "voicing": "power8",     // default chord voicing for chords patterns
      "transpose": 0, "octave": 0,
      "strum": 0.006,          // s between chord notes (guitar strum)
      "eq": {"low_cut": 120, "high_cut": 9000, "bands": [{"hz": 2500, "db": 2, "q": 1}]},
      "drive": 0.0,            // bus saturation 0..1
      "delay": {"time": "8d", "feedback": 0.3, "mix": 0.15},   // times: 1 2 2d 4 4d 4t 8 8d 8t 16 16d 16t 32 or beats
      "reverb": 0.0, "reverb_decay": 1.2,   // built-in room (the mixer also adds role-based reverb)
      "humanize": {"timing_ms": 2, "velocity": 0.05},
      "split": true,           // drum kits: separate kick / snare / hats / cymbals stems
      "mute": false
    }
  },

  "patterns": {
    "<name>": {"kind": "drums",  "lanes": {"kick": "x...x...x...x...", "snare": "....X.......X..."}},
    "<name>": {"kind": "line",   "steps": "R.R8.R5.R.Lx8.5."},
    "<name>": {"kind": "chords", "rhythm": "x--.x-x-.x--x-x.", "voicing": "power8"},
    "<name>": {"kind": "arp",    "mode": "updown", "octaves": 2, "rhythm": "x.xxx.xx", "gate": 0.5},
    "<name>": {"kind": "notes",  "bars": 2, "score": "A5 e C6 e E6 q ..."}
    // common keys: grid (steps per bar), bars, swing, gate (0..1.5 of a step), velocity ("9.5.7." or [..]),
    // transpose, octave
  },

  "sections": [
    {"name": "Verse 1", "start_bar": 9, "bars": 16,   // start_bar defaults to right after the previous section
     "chords": ["Am", "Am/G", "F", "E"],       // one entry per bar, cycles; "F G" splits a bar; "%" repeats
     "energy": 0.5,                            // 0..1
     "key": "A minor",                         // optional key change (e.g. "B minor" after a modulation)
     "transpose": 0,                           // semitones for absolute "notes" riffs in this section
     "crash": true,                            // crash cymbal on the first downbeat
     "like": "Verse 1",                        // copy parts/chords/energy of an earlier section
     "parts": {
       "drums": "half",                                    // pattern name: loops through the section
       "bass": {"pattern": "bass_verse", "octave": 0, "velocity": 1.0},
       "gtr": [{"pattern": "chug", "bars": [9, 16]}, {"pattern": "chug8", "bars": [17, 24]}],
       "pad": null                                         // silent (unlisted instruments are silent too)
     },
     "fills": [...], "events": [...], "stops": [...]      // same as the top-level lists
    }
  ],

  "fills":  [{"part": "drums", "bar": 24, "beat": 3, "pattern": "fill_toms"}],
  "events": [
    {"type": "fx", "sound": "laser", "bar": 27, "beat": 4, "pitch": "A6", "beats": 0.5, "gain_db": 0, "pan": 0.3},
    {"type": "hit", "part": "drums", "voice": "crash", "bar": 33},
    {"type": "note", "part": "bass", "pitch": "B1", "bar": 133, "beats": 4},
    {"type": "chord", "part": "gtr", "bar": 133, "beats": 4},
    {"type": "stutter", "bar": 48, "beat": 4, "beats": 1, "slice": 0.25, "parts": ["gtr", "lead"]},
    {"type": "tapestop", "bar": 108, "beats": 4, "parts": "all", "except": ["fx"]}
  ],
  "stops":  [{"bar": 32, "beat": 3, "beats": 2, "parts": "all", "except": ["fx"]}]
}
```

## Patterns

### drums
`lanes` maps kit voices to step strings. Step characters: `.` rest, `x` hit (0.8), `X` accent
(1.0), `o` ghost (0.5), `g` soft (0.4), `1`-`9` velocity (n/9), `r` roll (two 32nds), `R` triplet
roll, `f` flam. Each lane loops on its own length, so `"hat": "x.X."` repeats every beat inside a
one-bar pattern. Open hats are choked by the next closed hat.

Voices: `kick snare clap rim hat ohat crash ride tom1 tom2 tom3 shaker tamb cowbell snap`
(aliases: bd sd cp hh oh cr rd ht mt lt ...). Stems: `<kit> kick`, `<kit> snare` (snare, clap, rim,
snap, toms), `<kit> hats` (hats, shaker, tamb, cowbell), `<kit> cymbals` (crash, ride).

### line (bass lines and other chord-relative monophonic parts)
`steps` holds one degree token per step, relative to the chord playing at that step:

| token | meaning |
|---|---|
| `R` | the chord's bass note (the slash note for `Am/G`), placed at or above the instrument's `low` |
| `1` | chord root |
| `3` `5` `7` | the chord's own third/fifth/seventh; a 7th the chord lacks comes from the key (E in A minor -> D) |
| `2` `4` `6` `9` `11` `13` | diatonic notes of the key above the root |
| `8` | root an octave up; `10` `12` = 3rd/5th an octave up |
| `L` `H` `T` `O` | compact aliases: `R,` (octave down), `5'`, `3'`, `8'` |
| `x` | dead/ghost note (short, muted thump on the root) |
| `-` | tie (hold the previous note) |
| `.` | rest |

In the spaced form (`"R . b7 . 5 , 8 ."`) tokens may carry `b`/`#` (`b7`), octave marks
`'` (up) and `,` (down) (`5,` = fifth below), and `!` (accent) / `?` (ghost) suffixes.
Notes held across a chord change are re-struck on the new chord.

### chords (guitar strums, stabs, pads)
`rhythm` steps: `x` hit, `X` accent, `o` soft, `d`/`u` down/up strum, `m` palm-muted (short and dark),
`-` tie, `.` rest, digits = velocity. `voicing`: `triad` (the chord as written in close position,
smooth voice leading near `center`), `seventh`, `power` (root + 5th), `power8` (+ octave), `open`,
`full` (six-string spread), `shell`, `octave`, `root`, `cluster`, `spread`. Optional `center`,
`low`, `strum` (s).

### arp
Arpeggiates the current chord: `mode` = `up | down | updown | downup | pingpong | random | order`,
`octaves` (1-4), `rhythm` (which steps play; default every step), `order` (a degree list like
`"1 5 8 10 8 5"` for mode `order`), `low`, `gate`.

### notes (riffs and melodies in absolute pitch)
`score`: pitch/duration pairs, `"A4 e C5 e r q E5+A5 q. G4! s G4? s"` (`r` rest, `+` joins a chord,
`!` accent, `?` ghost; durations `w h q e s t`, dots, triplets `e3`, ties `q+e`, or beats).
Or `notes: [[beat, pitch, beats, velocity], ...]` with beats from the pattern start. `bars` sets the
loop length. `relative: true` lets pitches be degree tokens instead. A section's `transpose`
applies to these riffs (chord-relative patterns already follow the chords).

## Sections, parts, fills

* A part value is a pattern name, `null` (silent), a segment object `{pattern, bars: [from, to],
  transpose, octave, velocity, offset}` or a list of segments. Patterns restart at each segment
  start and loop. Instruments not listed in a section's `parts` are silent in it.
* `energy` (0..1, default 0.8) scales velocities a little and opens filters on high-energy sections.
* A fill `{part, bar, beat, pattern}` replaces the part's groove from that beat to the end of the
  fill pattern (a one-bar fill starting at `beat: 3` plays its own steps 9-16).

## Events and stops

* `fx` one-shots (`sound`, `beats` length, `pitch`, `gain_db`, `pan`, `velocity`, sound params):
  `laser zap ufo riser downlifter whoosh sweep impact boom radio bleeps glitch modem alarm
  powerdown coin error reverse_cymbal`. Risers/reverse cymbals should end on the target downbeat.
  Played by the first `type: fx` instrument (one is created automatically when none exists).
* `hit` (a drum voice), `note` (pitch or list of pitches) and `chord` (the current chord in the
  instrument's voicing) one-offs.
* Processing effects rewrite `beats` of the rendered stems of `parts` (default all, minus `except`):
  `stutter {slice, decay, pitch, gate}`, `tapestop {curve}`, `crush {bits, downsample, mix}`,
  `filter {mode lp|hp|bp, from_hz, to_hz, resonance}`, `gate {steps, grid}`, `reverse {slice}`,
  `telephone`.
* `stops` are sudden silences: notes starting inside are dropped, everything sounding is cut with a
  4 ms fade, and echoes/reverb tails do not come back afterwards. `beats` defaults to the end of the
  bar; `to_bar`/`to_beat` give an absolute end.

## Presets

| preset | role | sound |
|---|---|---|
| `bass_pick` | bass | picked electric-bass-like saw + square, 24 dB filter pluck, drive |
| `bass_slap` | bass | bouncy slap/pop bass: bright snap, pitch blip, mid "pop" |
| `bass_sub` | bass | clean sine/triangle sub |
| `bass_growl` | bass | detuned-saw rock bass with growl |
| `bass_chip` | bass | NES triangle bass |
| `power_saw` | keys | crunchy distorted power-chord "guitar" (saws into amp drive + cab EQ) |
| `gtr_mute` | keys | palm-muted distorted chug |
| `gtr_clean` | keys | bright clean strum |
| `supersaw` | keys | wide 7-voice supersaw chords |
| `brass` | keys | synth brass stab with filter swell |
| `chip_lead` | lead | 25 % pulse chiptune lead with delayed vibrato |
| `chip_square` | lead | hollow square lead |
| `chip_arp` | keys | 12.5 % pulse blips for fast arps |
| `saw_lead` | lead | detuned saw lead, mono glide, vibrato |
| `sync_lead` | lead | nasal resonant square/saw with per-note wah |
| `pluck` | keys | snappy synth pluck |
| `organ` | keys | drawbar organ |
| `pad`, `pad_glass` | pad | warm wide pad / airy glass pad |
| `bell`, `glock`, `music_box` | keys | FM bell / glockenspiel / music box |
| `epiano` | keys | FM electric piano |
| `piano` | keys | additive pop piano |
| `acoustic` | keys | strummed steel-string "acoustic guitar" (use voicing `full` and `d`/`u` strums) |

## Synth parameters (`params`)

`osc` (list of layers `{wave: saw|square|pulse25|pulse12|triangle|sine|organ|soft_saw|hollow,
level, octave, semi, cents, voices, detune (cents), spread (0..1)}`), `sub`, `noise`, `attack`,
`decay`, `sustain`, `release`, `attack_curve`, `filter` (lp|hp|bp|none), `slope` (12|24), `cutoff`
(Hz at middle C), `resonance` (0..1), `keytrack`, `env` (filter-envelope octaves), `f_attack`,
`f_decay`, `f_sustain`, `f_release`, `vel_to_cutoff`, `vel_sens`, `drive` (pre-filter amp
distortion), `post_drive`, `pitch_env` (semitones), `pitch_decay`, `vibrato` (semitones),
`vibrato_rate`, `vibrato_delay`, `glide`, `mono`, `chip_arp` ([0, 4, 7] NES arpeggio),
`chip_arp_rate`, `click`, `eq`, `level`, `gate`; FM presets: `ratio`, `index`, `index_decay`,
`index_sustain`, `ratio2`, `index2`, `index2_decay`, `chorus`; piano: `partials`,
`inharmonicity`, `hammer`, `brightness`.

Drum voice parameters: kick `pitch` (Hz, ~50-60), `sweep`, `sweep_time`, `decay`, `click`, `drive`;
snare `tone` (Hz), `tone_decay`, `snappy`, `noise_decay`, `crack`, `brightness`, `drive`; clap `tone`,
`decay`, `spread`, `bursts`; hats/cymbals `tone`, `decay`, `brightness` (ride `bell`); toms `pitch`,
`decay`; every voice `level`, `pan`.

## Complete small example

Four bars in A minor using every pattern kind, an FX one-shot, a stutter, a fill and a stop
(`synth2.arranger.example_arrangement()` returns it):

```json
{
 "format": "synth2-arrangement", "version": 1, "key": "A minor", "swing": 0.0,
 "instruments": {
  "drums": {"type": "drums", "kit": "pop", "gain_db": 0},
  "bass": {"preset": "bass_slap", "gain_db": 0},
  "gtr": {"preset": "power_saw", "gain_db": -1, "pan": -0.3},
  "lead": {"preset": "chip_lead", "gain_db": -1, "pan": 0.1, "delay": {"time": "8d", "feedback": 0.3, "mix": 0.18}},
  "arp": {"preset": "chip_arp", "gain_db": -2, "pan": 0.35},
  "fx": {"type": "fx", "gain_db": 0}
 },
 "patterns": {
  "beat": {"kind": "drums", "lanes": {"kick": "x.....x...x.....", "snare": "....X.......X...",
           "clap": "....x.......x...", "hat": "x.x.x.x.x.x.x.X.", "ohat": "..............x."}},
  "fill": {"kind": "drums", "lanes": {"snare": "....x.x.xxxxXXXX", "kick": "x.......x.......",
           "tom1": "........x.x....."}},
  "bounce": {"kind": "line", "steps": "R.R8.R5.R.Lx8.5."},
  "chug": {"kind": "chords", "rhythm": "x--.x-x-.x--x-x.", "voicing": "power8"},
  "riff": {"kind": "notes", "bars": 2,
           "score": "A5 e C6 e E6 e A5 e G5 q E5 e D5 e  C5 e D5 e E5 q. r e E5! e G5 e"},
  "sparkle": {"kind": "arp", "mode": "updown", "octaves": 2, "rhythm": "x.xxx.xxx.xxx.xx", "gate": 0.5}
 },
 "sections": [
  {"name": "Demo", "start_bar": 1, "bars": 4, "chords": ["Am", "F", "G", "E7"], "energy": 0.9, "crash": true,
   "parts": {"drums": "beat", "bass": "bounce", "gtr": "chug", "lead": "riff", "arp": "sparkle"},
   "fills": [{"part": "drums", "bar": 4, "beat": 3, "pattern": "fill"}]}
 ],
 "events": [
  {"type": "fx", "sound": "laser", "bar": 2, "beat": 4.5, "pitch": "A6"},
  {"type": "stutter", "bar": 3, "beat": 4, "beats": 1, "slice": 0.25, "parts": ["gtr", "lead"]}
 ],
 "stops": [{"bar": 4, "beat": 4.5, "beats": 0.5, "parts": "all", "except": ["drums"]}]
}
```

## API

```python
from synth2.arranger import render_backing, kick_times, validate_arrangement

stems = render_backing(project, 44100, t0, t1)   # list[BackingStem(name, role, audio (n, 2) float32, gain_db, pan, sidechain)]
kicks = kick_times(project)                       # np.ndarray, seconds of every kick hit (sidechain)
problems = validate_arrangement(arrangement)      # ["error: ...", "warning: ..."], [] when clean
```

`render_backing` covers `[t0, t1)` exactly (`round((t1 - t0) * sr)` samples), renders with a pre-roll
so a window is sample-identical to the same span of a full render, returns `[]` without a backing,
omits instruments that are silent in the window, and raises `ArrangementError` for an invalid
arrangement. `gain_db`/`pan` are not applied to `audio`; they are the mixer's defaults.
`song_length_seconds(project)` gives the end of the last section or event.
