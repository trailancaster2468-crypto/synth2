# Q6: why the voice sounds out of breath (work in progress, kept current)

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad

## How to continue from here

- Harness: `$S/q6/x6.py` (the spike's render loop split into stages with one option per suspect;
  every option defaults to the spike's behaviour). `x6.run(tag, ds, **options)` renders verse1, chorus1
  and plain, measures voiceq + WER tiny.en/base.en and appends to `$S/q6/runs/LOG.txt` (readable)
  and `$S/q6/runs/results.jsonl` (everything, with the transcripts). Clips: `$S/q6/runs/<tag>/<case>.wav`.
- `$S/q6/vq6.py`: `tools/voiceq.py`'s maths with the per-frame arrays exposed (checked equal to
  `voiceq.measure` on `$S/ds1/verse1.wav`), so numbers can be split by phoneme class with the known
  alignment (`v_cpp` = vowel frames, `core_cpp` = frames whose whole 64 ms window is inside one vowel).
- Experiment scripts are `$S/q6/x*.py`, run as
  `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/q6/xNN_name.py`.
- Noise: every run makes fresh acoustic + variance sessions after `onnxruntime.set_seed(seed)`
  (checked with `x_check.py`: same seed = identical audio in one process; seed 1 vs 2 = 7.2 dB
  waveform SNR), so runs that differ in one option share their diffusion noise where the tensor
  shapes are the same.
- Trap: `yaml.safe_load` on `dsdur/dsdict-en.yaml` takes about 215 s (pure-Python YAML, the whole
  pronouncing dictionary). `x6.load` reads the `symbols:` list line by line instead.

## Results

(rows are appended below as they are measured)

### F0. The `cpp` number of tools/voiceq.py depends on pitch (measured, `x_cal.py`)

`python3 -I $S/q6/x_cal.py`: one synthetic vowel (fixed spectral envelope, fixed harmonics-to-noise
ratio) at different pitches. voiceq `cpp` / `hnr` at a true HNR of 20 dB:

| midi | 60 | 64 | 67 | 69 | 72 | 74 | 76 | 77 |
|---|---|---|---|---|---|---|---|---|
| cpp | 30.7 | 28.9 | 32.1 | 29.8 | 26.1 | 28.2 | 23.6 | 24.6 |
| hnr | 21.9 | 21.6 | 22.9 | 21.6 | 22.1 | 23.3 | 21.7 | 23.2 |

`plain` and `twinkle` sing C4..A4 (60..69); verse1 and chorus1 sing A4..F5 (69..77, mostly 71..76).
The same voice quality therefore reads 3 to 5 dB lower `cpp` in the song sections. Two reasons: the
peak search of voiceq stops at 600 Hz (`lo = int(SR / 600)`), so for E5 and F5 (659, 698 Hz) the first
cepstral peak is outside the search range; and fewer harmonics fit under 8 kHz. `hnr` is flat with
pitch. So the "4 dB of cpp lost in the song sections" in the brief is mostly the ruler, not the voice:
compare `cpp` only between clips in the same pitch range (transposed controls below), and use `hnr`
across ranges. voiceq of Q1's existing verse1 controls (`$S/q1/runs/diag_*`, no new renders):
168 BPM 20.8-21.1 | 84 BPM 21.8 | 120 BPM 21.3 | transposed -7: 22.3 | -12: 21.1 | all notes E4: 22.8
| all notes B4: 21.2 | style pop: 21.2.

### Noise floor (3 noise seeds, spike settings; `x01_noise.py`, rows `base_s1..3` in runs/LOG.txt)

| case | cpp | hnr | lvl_sd | dip | WER tiny | WER base |
|---|---|---|---|---|---|---|
| verse1 | 20.81 21.04 20.98 | 18.26 18.01 18.32 | 6.72 6.42 6.47 | -10.87 -10.37 -9.71 | .568 .670 .693 | .534 .534 .489 |
| chorus1 | 21.06 21.08 21.14 | 21.53 21.51 21.50 | 5.55 5.92 5.91 | -7.09 -7.60 -7.76 | .800 .600 .771 | .429 .429 .286 |
| plain | 24.64 25.08 24.63 | 20.53 20.48 20.07 | 4.79 4.80 4.68 | -6.34 -5.29 -5.33 | .083 .083 .042 | .083 .042 .042 |

Real differences: cpp > 0.5 dB, hnr > 0.5 dB, lvl_sd > 0.4 dB, dip > 1.2 dB. WER of one take:
verse1 +-0.07 (tiny) / +-0.03 (base); chorus1 +-0.10 / +-0.07 and worse (the spike's own take in
`$S/ds1` scored 0.29 with tiny.en, these three 0.60 to 0.80: with 35 words and "u f o" in the
reference, chorus1 WER of a single take says almost nothing); plain +-0.03.
With the known alignment: over vowel frames only, `dip` is -5.8 (verse1), -3.7 (chorus1), -3.3
(plain): the -10 dB "dip inside voiced stretches" of verse1 is mostly voiced consonants (nasals,
liquids, voiced stops), which are 5 to 15 dB quieter than vowels and make up more of a fast line.

### 1. Level rides: innocent (in the demo sections)

The spike applies exactly one gain curve: `ds_backend.py` multiplies the stem by the track's
`loudness` lane, then by one constant that matches the formant render's active RMS. Synth2's style
dynamics (phrase arc, note swell, end taper) are not applied at all. For the lead (`AI`, `AI Verses`,
`AI Pre`) the lane is 0 dB everywhere except 154.1-159.8 s (bars 109-112, -4 dB); verse1 is
11.4-34.3 s and chorus1 45.7-68.6 s. So in everything the user heard the gain was a constant, and
all of `lvl_sd` / `dip` comes from the model. (Evaluated with `evaluate_curve` on
`build.build_session()`; no render needed.)

### 2a. What the variance model predicts (dump, no acoustic model; `x02_var.py all`, log `runs/x02.log`)

Medians over vowel frames, dB. "b-v" = breathiness minus voicing = the model's own noise-to-harmonics
ratio. Vowel length and the consonants' share of the time between vowel onsets come from the alignment.

| condition | pitch | breath | voicing | voicing sd | voicing p10 | b-v | vowel median | vowels < 60 ms | consonant share |
|---|---|---|---|---|---|---|---|---|---|
| plain (pop, 120 BPM) | 64 | -46.1 | -23.7 | 2.8 | -26.9 | -22.7 | 186 ms | 3 % | 0.41 |
| twinkle | 65 | -48.3 | -24.7 | 3.1 | -28.9 | -23.6 | 389 ms | 0 % | 0.37 |
| verse1 (robotic, 168) | 72 | -44.4 | -21.5 | 7.7 (seed 2: 10.1) | -27.7 | -22.6 | 81 ms | 26 % | 0.48 |
| chorus1 (robotic, 168) | 74 | -44.4 | -21.1 | 4.6 | -25.3 | -23.9 | 209 ms | 4 % | 0.36 |
| plain +7 semitones | 71 | -45.5 | -21.2 | 2.7 | -24.9 | -23.6 | | | |
| plain +12 | 76 | -46.9 | -19.9 | 3.3 | -24.1 | -26.5 | | | |
| verse1 -7 | 65 | -44.1 | -23.3 | 12.8 | -32.6 | -21.5 | 70 ms | 35 % | 0.52 |
| verse1 -12 | 60 | -43.4 | -23.9 | 9.1 | -31.9 | -19.5 | | 40 % | 0.55 |
| verse1 style pop | 72 | -43.6 | -21.2 | 4.9 | -25.9 | -22.6 | 81 ms | 26 % | 0.48 |
| verse1 glide 40 ms | 72 | -44.4 | -21.4 | 7.7 | -28.2 | -22.7 | | | |
| verse1 glide 80 ms | 72 | -44.3 | -21.3 | 5.7 | -27.6 | -22.8 | | | |
| plain style robotic | 64 | -46.4 | -24.1 | 3.8 | -28.6 | -23.2 | | | |
| verse1 84 BPM | 72 | -43.4 | -20.1 | 3.4 | -24.2 | -22.9 | 221 ms | 4 % | 0.37 |
| verse1 120 BPM | 72 | -43.3 | -20.6 | 6.9 | -25.2 | -22.7 | 128 ms | 8 % | 0.45 |
| verse1 140 BPM | 72 | -43.9 | -20.8 | 4.3 | -25.7 | -23.1 | 104 ms | 17 % | 0.47 |
| plain 168 BPM | 64 | -45.6 | -23.4 | 2.7 | -26.5 | -22.5 | 116 ms | 17 % | 0.47 |
| verse1 variance steps 5 / 20 | 72 | -44.2 / -44.4 | -21.4 / -21.5 | 7.6 / 7.6 | | | | | |

Reading:
- The variance model does NOT ask for more breath in the fast robotic sections: breathiness minus
  voicing on vowels is -22.6 (verse1) and -23.9 (chorus1) against -22.7 (plain). Higher pitch gets
  MORE voicing, not less. Transposing the verse DOWN makes it worse (more breath, more weak vowels).
- What does change is the spread: voicing sd on vowels is 7.7-10 dB in verse1 against 2.8 in plain:
  a minority of vowels is predicted far below the rest (weak, half-whispered syllables). Flat robotic
  pitch adds to that (pop style 4.9, 80 ms glides 5.7, 84 BPM 3.4); the number of variance steps does
  not (7.6 at 5, 10 and 20 steps).
- Short vowels: at 168 BPM the median vowel is 81 ms and a quarter are under 60 ms; consonants take
  48 % of every syllable (37 % at 84 BPM): consonant length does not shrink in proportion.

### 3a/4a. One factor at a time on `plain` (seed 1; batch `b1.json`, rows in runs/LOG.txt)

| render | cpp | hnr | lvl_sd | dip | dip on vowels | WER tiny / base |
|---|---|---|---|---|---|---|
| plain as written (C4-A4, pop, 120 BPM) | 24.64 | 20.53 | 4.79 | -6.34 | -3.24 | .083 / .083 |
| plain +7 semitones (G4-E5) | 22.05 | 21.56 | 5.07 | -6.42 | -3.59 | .042 / .083 |
| plain style robotic | 24.79 | 20.85 | 5.23 | -7.01 | -4.46 | .042 / .000 |
| plain at 168 BPM | 24.24 | 19.61 | 4.43 | -5.34 | -2.77 | .375 / .083 |
| plain +7, robotic, 168 BPM | 22.20 | 20.83 | 5.24 | -7.67 | -4.57 | .125 / .125 |
| chorus1 as written (A4-F5) | 21.06 | 21.53 | 5.55 | -7.09 | -3.76 | |
| chorus1 -7 semitones | 22.90 | 20.20 | 5.86 | -7.72 | -4.60 | .314 / .171 |
| verse1 as written | 20.81 | 18.26 | 6.72 | -10.87 | -5.83 | .568 / .534 |
| verse1 -7 semitones | 22.17 | 17.31 | 6.35 | -8.76 | -7.66 | .511 / .443 |

- Register alone moves `cpp` by 2.6 dB (plain) / 1.8 dB (chorus1) / 1.4 dB (verse1) while `hnr`
  moves the OTHER way (+1.0 / -1.3 / -0.9 when going down). The voice is not noisier in the song's
  register; the cpp number is lower there. Sung in the song's register, style and tempo, `plain`
  reads cpp 22.2: chorus1 (21.1) and verse1 (20.8) are within 1.1 and 1.4 dB of it.
- Robotic style: no cpp cost; about 1 dB more level dip on vowels (-4.5 against -3.2).
- 168 BPM on plain: cpp -0.4, hnr -0.9 (inside or at the edge of the noise floor).

### 8a. Rests are clean; soft vowel onsets start in the rest (traces: `x_trace.py`, `x_sp.py`)

In-phrase rests of the verse (7 to 11 of them, 93-186 ms) sit at -40 to -50 dB under the vowels in
their middle. The "loud" frames that `x_sp.py` finds in rests are the last 30 ms before a vowel
(the vowel starts early: -15, -5, 0 dB) and a nasal ringing out (-22, -33, -41). No inhale, gasp or
noise burst is generated in SP frames, head, tail or inside.
Bug found on the way: a phrase whose first vowel is at t = 0 loses its onset: `plain` starts with
"g" of "good" at -0.14 s and the render buffer starts at 0, so the clip begins on the vowel "uh".
Only the test cases that start on beat 0 are hit (plain, twinkle); the song sections start at beat 4.

### 2b. Weak syllables are whispered vowels the variance model asks for (CAUSE; `x_weak.py tr-7`, npz of the runs)

Per-vowel level of the rendered audio against the voicing the variance model predicted for it:

| clip | vowels | level sd | vowels weaker than -8 dB | corr(level, voicing asked) | dB of level per dB of voicing |
|---|---|---|---|---|---|
| verse1 (-7 st run) | 114 | 8.9 dB | 13 % (down to -31 dB) | 0.87 | 0.61 |
| chorus1 (-7 st run) | 53 | 2.8 dB | 2 % | 0.95 | 0.96 |
| plain +7, robotic, 168 BPM | 30 | 2.5 dB | 0 % | 0.97 | 0.90 |

In the verse about 10 % of the vowel frames get a voicing more than 10 dB under the median (down to
-50..-89 dB) while their breathiness stays at the normal -44.7 dB: breath minus voicing is +11.7 dB
there, i.e. the vowel is pure breath. The acoustic model obeys, the syllable comes out 20-30 dB down
(examples: "ay" 70 ms -31 dB, "aa" 81 ms -30 dB, "ey" 104 ms -23 dB, "er" 174 ms -24 dB). These are
the level dips, and a whispered syllable every second or two is what "out of breath" sounds like.
(The untransposed verse is measured below with the fix.)

### 9. The song's mix chain: the lead EQ makes the neural voice thin and airy (CONTRIBUTES; `x09_mix.py base_s1 --asr`)

`synth2.engine.mixing._processed_vocal` with the song's settings (build.py LEAD_STRIP + master EQ, band
mode) on the dry clips of `runs/base_s1`, level-matched. Stems written by the song build are post-strip.

| stage | verse1 cpp | hnr | hf | lvl_sd | dip | chorus1 cpp | hnr | hf |
|---|---|---|---|---|---|---|---|---|
| dry | 20.80 | 18.26 | -30.2 | 6.72 | -10.87 | 21.05 | 21.53 | -30.9 |
| whole strip | 19.81 | 15.72 | -23.3 | 6.17 | -9.00 | 20.02 | 18.48 | -24.3 |
| strip preset EQ only | 20.02 | 17.39 | -28.4 | 6.59 | -10.05 | 20.22 | 20.35 | -29.3 |
| preset + lead EQ + master EQ | 19.69 | 15.72 | -23.2 | 6.37 | -8.83 | 19.92 | 18.50 | -24.4 |
| compressor only | 20.63 | 18.20 | -30.0 | 6.49 | -10.87 | 20.86 | 21.56 | -30.8 |
| de-harsh + de-ess only | 20.81 | 18.29 | -30.3 | 6.71 | -10.90 | 21.05 | 21.53 | -30.9 |
| transient limiter only | 20.80 | 18.26 | -30.2 | 6.72 | -10.87 | 21.05 | 21.53 | -30.9 |

WER of the whole strip against dry: verse1 .568/.500 against .534/.557, chorus1 .657/.400 against
.829/.429 (inside the noise). Spectral balance of voiced frames (`x_tilt.py`, dB re 0.1-1 kHz): dry
neural 2-5 kHz -17..-19, 5-10 kHz -31; the demo's processed lead stem -12.3 and -23.4.
The EQ alone costs 2.5 dB (verse) and 3.0 dB (chorus) of `hnr` and adds 7 dB above 5 kHz: -4 dB at
600 Hz, -2 dB at 300 Hz and -1 dB at 630 Hz take the fundamental down (A4-E5 = 440-660 Hz) while
+3.5 dB at 4.8 kHz, +3 dB at 3.2 kHz and two high shelves (+2 dB at 9 kHz, +2.5 dB at 10 kHz) lift
the band where this voice is mostly aspiration noise. The curve was tuned for the formant engine,
which "fell off a cliff above 3.8 kHz" (comment in build.py). Compressor, de-esser, de-harsher and
transient limiter do nothing measurable: no pumping, no gate.

### 8b. Edges: innocent (`x08_edges.py`, log `runs/x08.log`; raw phrase waveforms before the fades)

Levels in dB relative to the phrase's median vowel level, spike settings (head 8 frames, tail 10).

| | verse1 (2 phrases) | chorus1 (6 phrases) |
|---|---|---|
| head SP, loudest 10 ms (median / worst) | -41 / -37 | -33 / -8.5 (vowel-initial "all", "every": the vowel starts early) |
| tail SP: first 10 ms / last 10 ms | -31 / -45 (worst -36, after "l") | -52 / -62 |
| time for the tail to fall under -20 dB | 5 ms (worst 10) | 0 ms |
| samples the 10 ms fades touch (peak, start / end) | -33 / -39 | -35 / -55 |
| in-phrase SP frames, median level | -37 | -50 |

Nothing is cut: releases have decayed 45 to 60 dB by the end of the 116 ms tail, the fades only touch
samples under -30 dB, and no burst, click or gasp is generated at phrase starts or ends. Head 8 / tail
8 is what OpenUtau uses (`DiffSingerUtils.headFrames/tailFrames`); the spike's tail of 10 is fine.

### 3b. Pitch source and breathiness: see Q5 (`$S/q5/FINDINGS.md`, "Audio A/B")

Q5 rendered verse1 / chorus1 with Synth2's robotic curve, Synth2's pop curve and the bank's predicted
pitch (expr 0, 0.5, 1): cpp 20.8 / 21.0 / 20.8 / 21.1 / 21.8 on verse1, 21.1-21.4 on chorus1; hnr
18.3 -> 17.1 (verse1, predictor at expr 1); lvl_sd 6.7 -> 6.3. The pitch the models see is not what
makes the voice breathy. It does change how often the variance model whispers a vowel (2a, 2c).

### 2c. Which conditions make the variance model whisper vowels (`x02b_dropouts.py`, 3 seeds, log `runs/x02b.log`)

"Weak vowel" = its median predicted voicing is more than 8 dB under the median of all vowel frames.

| condition (verse1, 114 vowels) | weak vowels (seeds 1, 2, 3) | weak vowel frames | voicing sd |
|---|---|---|---|
| robotic, as in the demo | 13, 12, 12 (11 %) | 8.0-8.8 % | 8.9 |
| style pop | 8, 5, 7 (6 %) | 3.9-4.6 % | 5.3 |
| robotic + 40 ms glides | 12, 11, 13 | 8.1-9.0 % | 8.6 |
| robotic + 80 ms glides | 11, 9, 12 | 6.4-7.8 % | 6.4 |
| robotic + 120 ms glides | 9, 9, 8 | 6.4-7.4 % | 6.0 |
| variance steps 5 / 20 / 50 | 12,11,12 / 13,12,11 / 12,12,11 | 7.6-8.7 % | 8.9 / 8.9 / 8.7 |
| chorus1, any of the above | 0 of 53 | 3.2-4.5 % (note ends) | 4.0-5.0 |

- The same vowels are hit at every step count and seed (seed 1: `ax` 5 frames -11 dB, `uw` 6 f
  -25, `er` 4 f -16, `ih` 5 f -11, `ax` 5 f -23 ...): it is systematic, short reduced vowels (35-80 ms)
  between consonants, not sampling noise. More variance steps do not help.
- Flat robotic pitch doubles the count against pop-style pitch (12 against 7 of 114); smoothing the
  steps helps only from about 80 ms up and never reaches pop. Pitch shape contributes, it is not the
  root: 6 % of the vowels are whispered with pop pitch too.
- The chorus (long notes) has none. Its 3-4 % "weak frames" are the last frames of long notes.

### 2d. The fix for the whispered vowels: a floor on vowel voicing (`b3.json` rows `v_floor6`; `x_weak.py v_floor6 verse1`)

`v_floor_rel=6`: every vowel frame's voicing is held at or above (running median of the phrase's vowel
voicing over 1.5 s of vowel time) minus 6 dB; consonants, rests and breathiness are untouched.
Seed 1, against `base_s1` (same noise):

| case | cpp | hnr | lvl_sd | dip | dip on vowels | per-vowel level sd | vowels under -8 dB |
|---|---|---|---|---|---|---|---|
| verse1 before | 20.81 | 18.26 | 6.72 | -10.87 | -5.83 | 7.4 | 11.4 % (13 of 114): -32, -31, -30, -25, -24, -23, -20, -16, -16, -15 dB ... |
| verse1 floor 6 dB | 20.89 | 18.03 | 6.46 | -9.51 | -5.07 | 2.9 | 2.6 %, weakest -9 dB |
| chorus1 before | 21.06 | 21.53 | 5.55 | -7.09 | -3.76 | 2.1 | 0 % |
| chorus1 floor 6 dB | 21.20 | 21.50 | 5.13 | -6.36 | -3.73 | 2.1 | 0 % |

(`x_weak.py base_s1:v_floor6 verse1 chorus1` for the "before" rows: alignment and raw prediction are those of the
v_floor6 run, which shares them.) In the demo's verse one syllable in nine is whispered, about one
every 1.7 s; with the floor none is more than 9 dB down. The chorus has no whispered vowel to begin with.

WER of that one take: verse1 tiny .568 -> .773, base .534 -> .500. The tiny.en jump is a Whisper
repetition loop ("Pop-pop-pop-..." 40 times at "pop up on a pop up"); the words before it are the same
or better. One take is not evidence either way: see the multi-seed confirmation at the end.
INTERIM clips (seed 1): `$S/q6/out/{verse1,chorus1}_{before,after}.{wav,mp3}` = runs/base_s1 and
runs/v_floor6 (replaced by the final run when it is done).

### 2e. Cross-check with Q1's slow-motion render (`x_weak.py q1_final_sN:v_floor6 verse1`, Q1 clips `$S/q1/runs/d_final`)

Q1's fix for word clarity (notes under 0.25 s are stretched for the models, the mel is squeezed back)
gives the variance model longer vowels. Per-vowel level on the same alignment, verse1:

| render | seed 1 | seed 2 | seed 3 |
|---|---|---|---|
| spike: level sd / vowels under -8 dB | 7.4 dB / 11.4 % | 7.5 / 10.5 % | 6.9 / 11.4 % |
| Q1 slow motion | 2.6 dB / 2.6 % (worst -14 dB) | 3.3 / 1.8 % (worst -26 dB) | 2.6 / 0.9 % (worst -16 dB) |
| spike + voicing floor 6 dB | 2.9 dB / 2.6 % (worst -9 dB) | | |

Both fixes remove the same defect from two sides (voiceq of Q1's clips: lvl_sd 6.5 -> 6.15, dip
-10.3 -> -8.4). Slow motion still leaves a -14..-26 dB syllable per take; the floor is the cheap
safety net and should stay on when slow motion is adopted. Part of Q1's WER gain is very likely
these syllables coming back.

### 2f. Where the floor acts, and the rule that went into `dsinfer.py`

Frames lifted by the plain 6 dB floor (`runs/v_floor6/*.npz`, `vo` against `vo0`): verse1 10.4 % of
the vowel frames, median lift 5.7 dB, 34 vowels of which 12 are low from the first frame to the last
(`ax` [-10, -9, -11, -15, -24], `uw` [-11, -15, -21, -30, -39, -46], `ax` [-12, -16, -23, -30, -34]
dB re the median): the whispered ones. chorus1 6.1 %, median lift 3.2 dB, 27 vowels, none low
throughout: single edge frames (harmless) and the last frames of notes that end before a rest, e.g.
`ow`, 62 frames, falling -12 .. -47 dB over its last 15 frames. Holding that at -6 dB would cut the
note off instead of letting it die away, so `dsinfer.floor_voicing` leaves the trailing low frames of
a vowel alone when the next phoneme is a rest (unless the whole vowel is low).
Floor depth: 3 dB levels more (verse1 dip -8.4, vowels -3.9; chorus1 dip -5.4) but natural vowel-to-vowel
level sd is 2.1-2.9 dB, so a 3 dB floor would also flatten ordinary unstressed syllables; 6 dB only
catches collapses. `VOICING_FLOOR = 6.0` is the recommended default, 4 dB is worth an A/B by ear.

| floor (plain rule, seed 1) | verse1 cpp / hnr / lvl_sd / dip / vowel dip / WER tiny, base | chorus1 | plain |
|---|---|---|---|
| none | 20.81 / 18.26 / 6.72 / -10.87 / -5.83 / .568 .534 | 21.06 / 21.53 / 5.55 / -7.09 / -3.76 / .800 .429 | 24.64 / 20.53 / 4.79 / -6.34 / -3.24 / .083 .083 |
| 6 dB | 20.89 / 18.03 / 6.46 / -9.51 / -5.07 / .773 (loop) .500 | 21.20 / 21.50 / 5.13 / -6.36 / -3.73 / .829 .457 | 24.66 / 20.53 / 4.72 / -6.29 / -3.23 / .083 .042 |
| 3 dB | 21.09 / 17.82 / 6.41 / -8.44 / -3.92 / .489 .500 | 21.32 / 21.58 / 5.09 / -5.39 / -3.11 / .686 .371 | (running) |

### 4a. Short vowels: tempo sweep of the same verse (Q1's clips `$S/q1/runs/diag_t84`, `diag_t120`, `base0`; voiceq, means of 2-3 seeds; alignment numbers from 2a)

| BPM | vowel median | vowels < 60 ms | consonant share | cpp | hnr | lvl_sd | dip | run ms | WER tiny / base (Q1) |
|---|---|---|---|---|---|---|---|---|---|
| 84 | 221 ms | 4 % | 0.37 | 21.8 | 21.0 | 6.0 | -8.1 | 310 | .426 / .216 |
| 120 | 128 ms | 8 % | 0.45 | 21.3 | 19.4 | 6.4 | -9.5 | 230 | .375 / .347 |
| 140 | 104 ms | 17 % | 0.47 | (row `bpm140` if it ran) | | | | | |
| 168 | 81 ms | 26 % | 0.48 | 20.9 | 18.2 | 6.5 | -10.3 | 185 | .644 / .519 |

No cliff: from 84 to 168 BPM `cpp` loses 0.9 dB, `hnr` 2.8 dB and `dip` 2.2 dB, evenly. The verse at
half tempo still reads cpp 21.8 because it is in the high register (F0). What does fall off a cliff
is the variance model's treatment of vowels under about 80 ms (2c): contributes through the whispered
vowels, not through the phonation of the vowels that are sung (vowel-core cpp 23.0 in verse1 against
21.7 in chorus1 at the same register).

### 2g. Offsets on the variance curves (`b3.json`; all frames, seed 1; WER "collapsed" = `x_rewer.py`, repetition loops removed)

| change | verse1 cpp | hnr | hf | lvl_sd | dip | run ms | WER tiny raw / collapsed | WER base | plain cpp / hnr / hf / WER |
|---|---|---|---|---|---|---|---|---|---|
| none (seeds 1, 2, 3) | 20.81 21.04 20.98 | 18.26 18.01 18.32 | -30.2 | 6.72 | -10.87 | 180 | .568 .670 .693 | .534 .534 .489 | 24.64 / 20.53 / -28.5 / .083 .083 |
| voicing +3 dB | 21.71 | 18.91 | -32.3 | 6.80 | -11.24 | 210 | .670 / .670 | .534 | 25.66 / 21.83 / -31.5 / .083 .083 |
| voicing +6 dB | 22.65 | 19.21 | -34.3 | 7.11 | -11.95 | 300 | .875 / .693 | .523 | |
| breathiness -6 dB | 22.75 | 19.01 | -34.2 | 7.03 | -12.00 | 240 | .909 / .716 | .580 | 26.95 / 22.77 / -34.1 / .083 .083 |
| voicing floor 6 dB (2d) | 20.89 | 18.03 | -30.2 | 6.46 | -9.51 | 170 | .773 / .636 | .500 | 24.66 / 20.53 / -28.5 / .083 .042 |
| voicing floor 3 dB | 21.09 | 17.82 | -30.5 | 6.41 | -8.44 | 170 | .489 / .489 | .500 | 24.77 / 20.42 / -28.6 / .292 .083 |

- The acoustic model follows both curves: +6 dB voicing or -6 dB breathiness buys about +1.9 dB cpp,
  +0.8..2.2 dB hnr and 4-6 dB less energy above 5 kHz. So the voice CAN be made cleaner than the
  bank's own prediction, but offsets do nothing for the weak syllables (dip gets worse, -12) and on
  all frames they thin the consonants (breathiness is the fricative and aspiration energy): WER of
  the verse is at the top of or above the baseline's range. Offsets are a timbre control, not the fix.
- tools note: tiny.en falls into repetition loops on "pop up on a pop up" (37-43 inserted words
  in three of these takes). `x_rewer.py` re-scores the stored transcripts with repeats collapsed.

### 9b. A lead EQ that does not thin the neural voice (`x09b_eq.py`, dry clips of `runs/base_s1`, whole strip with only `eq_add` changed)

| LEAD_EQ (`eq_add`) | verse1 hnr | hf | chorus1 hnr | hf |
|---|---|---|---|---|
| dry stem | 18.26 | -30.2 | 21.53 | -30.9 |
| as built: 600 Hz -4, 3150 -2.5, 4800 +3.5, shelf 9 kHz +2 | 15.72 | -23.3 | 18.48 | -24.3 |
| none (strip preset + master EQ only) | 17.28 | -28.0 | 20.23 | -28.8 |
| 600 Hz -2 only | 16.70 | -26.7 | 19.53 | -27.6 |
| 600 Hz -2, 3150 -2.5 | 17.19 | -26.7 | 20.24 | -27.7 |
| 600 Hz -2, 3150 -3 (Q 0.9), shelf 9 kHz -2.5 | 17.47 | -28.1 | 20.55 | -29.0 |

Dropping the 4.8 kHz bell and the 9 kHz shelf and halving the 600 Hz cut gives back 1.5-1.8 dB of the
2.5-3.0 dB of `hnr` the strip costs. The rest is the strip preset itself (+3 dB at 3.2 kHz, +2.5 dB
shelf at 10 kHz, -2 dB at 300 Hz) and the master EQ. Not checked: the vocal-to-band balance the song
build aims at (1-5 kHz, +5.2 dB); a duller lead may need 1-2 dB more fader instead of EQ.

### 2h. More rows of the variance sweep (`b3.json`, verse1, seed 1)

| change | cpp | hnr | hf | lvl_sd | dip | WER tiny raw / collapsed | base |
|---|---|---|---|---|---|---|---|
| breathiness -12 dB (all frames) | 24.35 | 19.36 | -36.6 | 7.06 | -12.31 | .693 / .693 | .568 |
| median of 3 variance samples | 20.95 | 18.01 | -30.5 | 6.81 | -10.74 | 1.386 (loop) | .523 |

- -12 dB of breathiness lifts verse1 `cpp` to 24.35, inside 1.5 dB of plain (24.6-25.1): the brief's
  cpp target can be "met" this way, but it is met by removing the noise the consonants are made of
  (hf -36.6, voiced share 0.97, base.en WER +0.05) to compensate for a register effect of the ruler
  (F0). Not recommended.
- Taking the median of three variance samples changes nothing (lvl_sd 6.81, dip -10.74): the
  whispered vowels are not sampling noise (2c).
