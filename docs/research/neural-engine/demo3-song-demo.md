# Demo 3: SELECT ALL INVADERS sung by Qixuan through the combined engine (kept current)

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad

## How to continue from here (read first if you take this over)

- STATE (18:06): all nine clips are in `out/` in their final form; steps 1-5 of the brief are done once. Sections
  below: status log, step 2 (lead), step 3 (cast), step 4 (mix), step 5 (checks), the clips, how to render again,
  the key change in build.py, what is still wrong.
- Engine: `$S/demo3/dsinfer.py` + `$S/demo3/dspitch.py`, copied from `$S/v67/` at 14:00 UTC. Three options added to
  `render_track_ds` in the copy: `vowel_breath_off`, `vowel_voicing_off` (dB on vowel frames only, function
  `vowel_weight`) and `detune` (cents on the sung pitch). With all three at 0 the code path is v67's.
- Backend: `ds_backend3.py` (`OPTIONS` = config.json, `LEAD_OPTS`, `CAST`; takes cached in `cache/`). Song:
  `build3.py` (copy of the song's build.py: `KEY_SHIFT`, `transpose_arrangement`, neural lead EQ). Entry points:
  `run_demo.py`, `final_mix.sh`, `make_clips.py`.
- Measurements: lead harness `h3.py` + `batch3.py SPEC` (`runs/results.jsonl`, table `python3 -I table3.py PREFIX`),
  cast `cast_lab.py` (`runs/cast.jsonl`), mixes `mix_check.py`, `asr_clip.py`, `mix_srt.py`, checks `run_checks.py`.
  While `demo3/PAUSE` exists a running batch waits between runs.
- Run everything as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I ...`.
- No new Nishiren renders of this song (terms, `$S/q4/FINDINGS.md` section 6). Clips 07/08 are byte copies of
  `$S/v67/out/{verse1,chorus1}_nishiren_best.*` (checked with cmp).
- Disk: `cache/` 1.8 GB (every take is stored from time 0 of the song), `mix/` 2.7 GB (24-bit stems of nine renders).
  `mix/a`, `mix/b*`, `mix/other` and `cache/` can be deleted; their numbers are in `runs/*.jsonl`. `mix/final` and
  `runs/` are what the clips are made from.
- Speed seen (2 threads): dry lead 22.7 s sung in 25-28 s at load 0.5-2 (30 s with the first run's model load),
  first chorus mix 144-157 s for 24.4 s (seven sung channels with doubles), a mix change on cached takes about 50 s.
- Not done: the rest of the song in the new key; a full-song render (section renders are mastered on their own);
  re-tuning the lead EQ's 600 Hz cut for the lower key; the chorus without the double and harmony.

## Status log (newest last)

- 14:00 Second run of this job. The first run left nothing on disk. Engine copied, harness copied.
- 17:05 Third run (the second was cut off at 14:05). Found on disk: `runs/results.jsonl` with the register rows
  `q_t-5` (verse1 and chorus1, seeds 1 and 2) and `q_t-3` (seed 1). Reference at the written pitch = the colleague's
  `q_all` rows in `$S/v67/runs/results.jsonl` (three seeds).
  Quick pass of step 2, WER tiny / base / small, loops collapsed (config.json options in every row):
  verse1 written pitch (3 seeds) 0.443 / 0.424 / 0.261; 5 lower (2 seeds) 0.365 / 0.300 / 0.250; 3 lower (1 seed)
  0.48 / 0.33 / 0.23. chorus1 written 0.353 / 0.200 / 0.114; 5 lower 0.270 / 0.140 / 0.085; 3 lower 0.17 / 0.03 / 0.03.
  cpp +1.2 to +1.5 dB at 5 lower (not comparable across registers), hnr -0.2 (verse) / -0.9 (chorus).
  => first set of clips is made 5 semitones lower (E minor), vocals and backing.
- 17:09 Written: `ds_backend3.py` (backend: config.json options, CAST per track-name prefix: gender, register shift,
  tone, doubles; level matched to the formant engine's take; no ring modulator), `build3.py` (copy of the song's
  build.py: `KEY_SHIFT` moves every vocal note and the backing arrangement, chord symbols, riffs and pitched fx;
  lead EQ without the high boosts and with half the 600 Hz cut), `run_demo.py OUT A B [--key N] [--cast F]`.
  Engine copy got a third option, `detune` (cents), for the crowd doubles.
- 17:17 FIRST COMPLETE SET OF CLIPS is in `out/` (wav + 160k mp3; `out/clips.json` has level and peak of each):
  `01_verse_lead_dry_5_lower`, `02_chorus_lead_dry_5_lower`, `02b_chorus_lead_dry_original_key`,
  `03_chorus_all_vocals_5_lower`, `04_chorus_full_mix_5_lower`, `05_verse_full_mix_5_lower`, `06_cast_5_lower`,
  `07_compare_verse_other_voicebank`, `08_compare_chorus_other_voicebank` (07/08 = copies of
  `$S/v67/out/{verse1,chorus1}_nishiren_best.*`, nothing re-rendered).
  Made with: `run_demo.py $S/demo3/mix/a 33 48` and `... 9 24` (144 s and 64 s at 2 threads, load 1), dry lead
  clips = harness renders `runs/q_t-5/{verse1,chorus1}_s1.wav` (config.json options, lead 5 lower) and
  `$S/v67/runs/q_all/chorus1_s1.wav` (written key), cast lines = `cast_first.py` (rows in `runs/cast.jsonl`),
  then `python3 -I make_clips.py`. Dry clips -16.8 LUFS (mp3), mixes -12.1 / -12.7 LUFS held by the -1 dBFS ceiling.
  First-pass cast (not yet measured against alternatives): AI gender 0; Aliens gender -0.6 sung an octave above
  the written notes; Aliens 2 gender -1.0 as written; Crowd x3 gender -0.5 / -0.8 / +0.2, low two an octave up, each
  with one detuned, delayed second take. Lines, WER tiny / base: AI "does the corner of the saucer count..."
  0.50 / 0.08; Aliens "we come in peace..." 0.00 / 0.20; Crowd "who is the invader" x2 1.38 / 0.00 (tiny loops).
  Mix report first pass: chorus lead over band (1-5 kHz) +4.7 dB, verse +5.8 dB; verse limiter max 6.0 dB (to look at).
  NEXT: step 2 batch `spec_vb` (vowel-only breathiness, running), then steps/gender, cast measurements, mix checks.
- 17:24-17:31 Cast sweeps (`cast_spec1.py`, `cast_spec2.py`): `tone=+12` lets the bank sing below MIDI 57.
- 17:36 Lead decided: 5 lower, vowel breathiness -3 dB, +7 cents (three seeds, `spec_vb2.py`).
- 17:36-17:49 Mix variants of the chorus on cached takes; final = neural lead EQ + 6.5 kHz shelf, hats and snare
  top down, vocals 8.0 dB over the band. No recogniser reads the full chorus mix, in any variant.
- 17:49-17:57 `spec_final` (the lead as used, three seeds, three recognisers), cast lines, Aliens 2 +10 cents.
- 17:58 Final clips written (`make_clips.py`), checks run. 18:03 steps 20 and gender +-0.15: no gain.
- 18:04 Register row for 3 lower completed (three seeds). `python3 -I t_same3.py`: the engine copy with its added
  options off reproduces the colleague's stored clip (`v67/runs/q_all/chorus1_s1.wav`) to less than one 16-bit step
  (max difference 2.5e-05; the stored wav is 16 bit) and repeats bit for bit in the same process.
- 18:06 `run_demo.py $S/demo3/mix/other 49 52` (post-chorus, default mix settings): the entry point renders other
  bars (7.3 s of audio in 71 s, nothing cached; -11.5 LUFS, true peak -1.02 dBTP, no silent note, no click on any
  channel). The aliens' processed channel there ("ouch, thank you, did we pass, click click click click", 10 words
  of one-word calls) reads 0.70 / 0.40 / 0.60 ("How should thank you? Did we pass click, click, click?"): their
  short calls are less clear than their full lines (0.00 / 0.00 dry). Their pitch there: -8.6 cents (9 notes).
- From 17:54 another engineer's jobs ran on the machine (load 3.5-4.4): render times after that are not comparable.

## Step 2: the lead on Qixuan (kept current; `python3 -I table3.py q_all q_t q5` prints the rows)

Lead = AI track, robotic, config.json options (slow motion 0.25, voicing floor 6 dB, bank pitch at expr 0.5, seeded).
WER tiny.en / base.en (/ small.en), loops collapsed. cpp is only comparable between rows in the same register.

### Less air on the vowels only (17:18-17:30, lead 5 lower, seed 1, `spec_vb.py`; engine options `vowel_breath_off`, `vowel_voicing_off`)

| vowel breathiness / voicing | verse1 WER t / b | chorus1 WER t / b | verse1 cpp / hnr / hf | chorus1 cpp / hnr / hf |
|---|---|---|---|---|
| none (q_t-5; seeds 1, 2) | 0.34 / 0.29, 0.39 / 0.31 | 0.31 / 0.14, 0.23 / 0.14 | 22.87 / 17.42 / -27.5 | 22.52 / 20.02 / -27.7 |
| -3 dB | 0.41 / 0.28 | 0.29 / 0.14 | 23.91 / 17.65 / -31.4 | 23.52 / 20.97 / -31.3 |
| -6 dB | 0.46 / 0.34 | 0.26 / 0.23 | 24.73 / 17.97 / -33.2 | 24.49 / 21.80 / -33.8 |
| -9 dB | 0.52 / 0.43 | 0.40 / 0.26 | 25.24 / 18.27 / -34.9 | 25.39 / 22.24 / -35.6 |
| -12 dB | 0.76 / 0.42 | 0.37 / 0.37 | 25.85 / 18.55 / -36.0 | 26.05 / 22.60 / -36.8 |
| -6 dB, voicing +1.5 | 0.55 / 0.36 | 0.29 / 0.26 | 25.09 / 17.69 / -33.8 | 25.06 / 21.69 / -34.7 |
| -6 dB, voicing +3 | 0.50 / 0.39 | 0.34 / 0.26 | 25.47 / 17.17 / -34.2 | 25.69 / 21.40 / -35.5 |
| -9 dB, voicing +3 | 0.46 / 0.36 | 0.29 / 0.29 | 26.10 / 17.40 / -35.4 | 26.54 / 21.63 / -36.7 |

- The tone gets cleaner in step with the offset (cpp +1.0 per 3 dB at first, hnr +0.2 to +1 per step, 4 dB less
  energy above 5 kHz at -3 dB), and the words get worse in step with it, also with the consonant frames left alone:
  verse tiny.en 0.34-0.39 -> 0.41 -> 0.46 -> 0.52 -> 0.76. -3 dB is the largest offset without a word cost on seed 1.
- A voicing lift on the vowels adds cpp and takes hnr and words: not used.
- Decision between none, -3 and -6 on three seeds: `spec_vb2.py` (below).

### The register and the vowel breathiness on three seeds (17:26-17:36, `spec_reg.py`, `spec_vb2.py`)

WER mean of seeds 1, 2, 3 (tiny.en / base.en / small.en; small.en only where run), loops collapsed.

| lead | verse1 WER | chorus1 WER | verse1 cpp / hnr / hf | chorus1 cpp / hnr / hf |
|---|---|---|---|---|
| stock engine, written key (the first demo; `q_stock`) | 0.637 / 0.568 / 0.417 | 0.638 / 0.190 / 0.181 | 20.93 / 18.13 / -30.2 | 21.07 / 21.44 / -30.9 |
| config.json fixes, written key (`q_all`) | 0.443 / 0.424 / 0.261 | 0.353 / 0.200 / 0.114 | 21.43 / 17.57 / -27.8 | 21.09 / 20.89 / -29.5 |
| fixes, 3 lower (`q_t-3`; seeds 2 and 3 added 18:04) | 0.451 / 0.375 / 0.292 | 0.276 / 0.134 / 0.057 | 22.32 / 17.23 / -27.4 | 21.50 / 20.55 / -28.3 |
| fixes, 5 lower (`q_t-5`) | 0.386 / 0.295 / 0.231 | 0.248 / 0.172 / 0.067 | 22.84 / 17.39 / -27.5 | 22.49 / 20.04 / -27.7 |
| fixes, 5 lower, vowel breathiness -3 dB (`q5_vb-3`) | 0.402 / 0.273 | 0.333 / 0.209 | 24.11 / 17.59 / -31.2 | 23.66 / 21.08 / -31.2 |
| fixes, 5 lower, vowel breathiness -6 dB (`q5_vb-6`) | 0.440 / 0.292 | 0.371 / 0.276 | 25.00 / 17.89 / -33.3 | 24.63 / 21.92 / -33.6 |

- 3 lower against 5 lower: the chorus is the same inside the noise (0.28 / 0.13 / 0.06 against 0.25 / 0.17 / 0.07),
  the verse only gains at 5 lower (3 lower: 0.45 / 0.38 / 0.29, no better than the written key on tiny.en and small.en).
- **5 semitones lower holds on three seeds**: verse -0.06 / -0.13 / -0.03, chorus -0.11 / -0.03 / -0.05 against the
  written key with the same fixes; against the first demo's engine and key the verse is 0.64 / 0.57 / 0.42 ->
  0.39 / 0.30 / 0.23 and the chorus 0.64 / 0.19 / 0.18 -> 0.25 / 0.17 / 0.07. The song is moved to E minor.
- **Vowel breathiness is a trade, not a free fix.** -3 dB: verse words equal (+0.02 / -0.02), chorus +0.09 / +0.04
  (inside the chorus' seed noise, but worse on two seeds of three and in line with the dose table above), cpp +1.2 dB,
  4 dB less energy above 5 kHz, hnr +0.2 (verse) and +1.0 (chorus). -6 dB: chorus +0.12 / +0.10, verse +0.05 / 0.00,
  cpp +2.1. DECISION for the demo: -3 dB on the vowels of the AI, its double and the harmony (the user's "always out
  of breath"), nothing more; -6 dB and deeper buy tone with words.
- The bank's pitch predictor (expr 0.5) sings flat: median -9.0 cents (verse) and -6.0 (chorus) in the curve the
  engine sings, -6.5 and -6.8 measured in the audio (`run_checks.py dry q_t-5 ...`). The lead, double and harmony get
  `detune=+7` cents (engine copy option) in `ds_backend3.LEAD_OPTS`.

### The lead as used: before and after (17:49-17:57, `spec_final.py`, tag `q5_final`; three seeds, three recognisers)

Final lead = config.json options + lead 5 semitones lower + vowel breathiness -3 dB + 7 cents up, steps 10, gender 0.

| | verse1 WER t / b / s | chorus1 WER t / b / s | plain WER | verse1 cpp / hnr | chorus1 cpp / hnr | weak vowels, verse |
|---|---|---|---|---|---|---|
| BEFORE: the first demo's engine and key (`q_stock`) | 0.637 / 0.568 / 0.417 | 0.638 / 0.190 / 0.181 | 0.111 / 0.042 / 0.042 | 20.93 / 18.13 | 21.07 / 21.44 | 12.3 % |
| combined fixes, written key (`q_all`) | 0.443 / 0.424 / 0.261 | 0.353 / 0.200 / 0.114 | 0.111 / 0.042 / 0.042 | 21.43 / 17.57 | 21.09 / 20.89 | 0 % |
| AFTER: the demo's lead (`q5_final`) | 0.424 / 0.265 / 0.235 | 0.305 / 0.190 / 0.067 | 0.083 / 0.042 / 0.000 (seed 1, written pitch) | 24.11 / 17.49 | 23.50 / 21.22 | 0.3 % |

Per seed (tiny / base): verse 0.42 / 0.27, 0.39 / 0.26, 0.47 / 0.26; chorus 0.23 / 0.14, 0.31 / 0.17, 0.37 / 0.26.
The same settings in the written key, chorus1 seed 1 (`q0_final`, clip 02b): 0.46 / 0.03 / 0.29, cpp 21.99.
cpp is not comparable across the key change (the measure rises as the pitch falls, Q6 F0); within the low key the
breathiness offset is worth +1.2 dB. So: about one verse word in four is still missed by base.en and small.en
(first demo: more than one in two), the chorus is where the first demo's lucky draw was on base.en and better on
the other two, `plain` is unchanged.

Words of the verse still missed (base.en, seed 1): "I sent it to the spam folder unread" ("campfooler and red"),
"they knocked politely" ("they not colloquy"), "cookie banner wall" ("panorole"), "accept all reject all sorry
that is still accept all" ("Tori checko, sorry that is a latador"), "pop up on a pop up" (loops), "ninety thousand
pages" ("no need, thousand cages"), "trained" ("aim"). These are the 16th-note lines (5.4 syllables a second).

### steps and gender (17:57-18:03, `spec_sg.py`, seed 1, on the -3 dB lead; tiny / base)

| | verse1 | chorus1 | |
|---|---|---|---|
| steps 10, gender 0 (`q5_vb-3` seed 1) | 0.41 / 0.28 | 0.29 / 0.14 | |
| steps 20 | 0.39 / 0.29 | 0.29 / 0.17 | no effect; render 26 s -> 40-45 s (load 4, noisy). Stays 10 |
| gender +0.15 | 0.47 / 0.32 | 0.29 / 0.17 | no gain (cpp +0.15 / +0.26) |
| gender -0.15 | 0.42 / 0.34 | 0.23 / 0.14 | inside the noise (cpp -0.25 / -0.21). Stays 0 |

## Step 3: the cast (17:24-17:31; dry lines, song 5 lower; rows in `runs/cast.jsonl`, wavs `runs/cast/TAG.wav`)

`cast_spec1.py` (where each voice is clearest) and `cast_spec2.py` (the cast as used). Lines: Aliens A = "we come in
peace please take us to your leader" (bars 25-26, at -5 MIDI 45-54), Aliens B = "we demand to speak to the manager
of the human race" (bars 53-54, MIDI 40-52), Crowd = "who is the invader" twice (bars 99-100, MIDI 52-59).
WER tiny.en / base.en; `hnr` from voiceq (falls with pitch: compare within a register only).

Where the bank is clearest (line Aliens A, one take):

| register (MIDI) | gender | tone | WER t / b | hnr | |
|---|---|---|---|---|---|
| as written, 45-54 | -0.6 | 0 | 0.70 / 0.40 | 10.1 | breaks: the bank has no voice down there |
| as written, 45-54 | -0.6 | +12 | 0.00 / 0.10 | 11.6 | acoustic model sings an octave up, vocoder the real pitch: clear |
| as written, 45-54 | 0.0 | +12 | 0.20 / 0.30 | 10.6 | |
| octave up, 57-66 | -1.0 | 0 | 0.00 / 0.10 | 15.8 | |
| octave up, 57-66 | -0.3 | 0 | 0.00 / 0.00 | 15.6 | |
| octave up, 57-66 | -0.6 | -5 | 0.20 / 0.00 | 15.6 | |
| octave up, 57-66 | +0.8 | 0 | 0.20 / 0.30 | 15.0 | small formants cost words |
| octave up, 57-66 | -1.0 | +5 | 0.30 / 0.10 | 15.6 | |

Crowd line, one take: octave up (64-71) gender -0.5: 0.75 (tiny repeats the line) / 0.00, hnr 19.5; as written (52-59):
0.25 / 0.00, hnr 14.2; as written with tone +12: 0.00 / 0.00, hnr 15.4.
=> `tone=+12` (the acoustic model sings an octave above the real pitch) is what lets this bank sing below about
MIDI 57 with the words intact. It is not an effect: one take, no modulation.

### Cast table (the settings in `ds_backend3.CAST`)

| character (track, Synth2 voice) | gender | tone | register sung (song 5 lower) | effect | WER dry, t / b | WER processed, t / b | hnr dry / processed |
|---|---|---|---|---|---|---|---|
| AI lead (`AI`, nova, robotic) | 0.0 | 0 | as written: E4-C5 (64-72) | none (vowel breathiness -3 dB, +7 cents) | line "does the corner of the saucer count prove you're not a robot" 0.33 / 0.08 | = dry | 19.9 |
| AI Double (nova) | +0.45 | 0 | as written | none (a second take) | | | |
| Harmony (pixel) | +0.35 | 0 | as written (59-69) | none | | | |
| Aliens (`Aliens`, basso, robotic) | -0.6 | +12 | as written: E2-G3 (40-55) | none | A 0.00 / 0.10, B 0.00 / 0.00 | = dry | 11.6, 10.2 |
| Aliens 2 (`Aliens 2`, atlas, robotic) | -1.0 | 0 | as written: E3-G4 (52-67) | a second take an octave below (tone +12, gender -0.6, -4 dB); both +10 cents | A 0.00 / 0.10, B 0.00 / 0.00 | A 0.00 / 0.20, B 0.18 / 0.00; B with the +10 cents 0.00 / 0.00 | 15.8, 15.4 / 11.9, 9.1 |
| Crowd (`Crowd`, atlas; `Crowd 2`, ember; `Crowd 3`, aurora; rock) | -0.3, -0.7, +0.2 | +12, +12, 0 | as written: 50-63, 50-63, 62-75 | each voice doubled by a second take (9-14 cents off, 17-26 ms late) | one voice 0.00 / 0.00 | one voice doubled 0.00 / 0.00; all three doubled 0.00 / 0.00 | 15.4 / 13.5; all 12.8 |

Aliens 2 only sings Japanese chants in the song, so it is measured on the Aliens lines sung an octave up (its
register). Also tried and not used for the Aliens: a quiet take an octave above the low voice (A 0.20 / 0.10, B 0.00 /
0.00: no better, less distinct from Aliens 2). The three crowd voices without their doubles scored 0.75 / 0.25 on
this line in one draw ("who is the inventor"), with the doubles 0.00 / 0.00: ensemble WER on 8 words is one draw.
The first demo's ring modulator and comb are gone; nothing here modulates the voice.

## Step 4: the mix (17:36-17:49; `mix_variants.sh`, `mix_check.py`, `asr_clip.py`, `mix_srt.py`; rows in `runs/mix_checks.jsonl`, `runs/asr_clips.jsonl`, `runs/mix_srt.jsonl`)

All variants sing the same cached takes (lead with vowel breathiness -3 dB and +7 cents, the cast above), song 5
lower, first chorus (bars 33-48) unless the row says verse. `cons` = the lead's consonant frames against the whole
backing in 4-10 kHz (median, dB); `masked` = share of those frames where the backing is louder; `hats` = the same
margin against the hats and cymbals stems only. Lead-stem numbers are the processed lead channel (voiceq, WER
tiny / base / small).

| mix | lead over band 1-5 kHz | cons | masked | hats | lead stem hnr / hf | lead stem WER | limiter max / mean |
|---|---|---|---|---|---|---|---|
| b0: the song's chain as built for the formant engine (lead EQ 600 Hz -4, 4.8 kHz +3.5, shelf 9 kHz +2) | +4.9 | -3.5 | 68 % | +5.6 | 18.61 / -25.3 | 0.29 / 0.31 / 0.11 | 1.73 / 0.23 |
| b: neural lead EQ (600 Hz -2, 3.15 kHz -2.5, no high boosts) | +4.8 | -5.7 | 79 % | +2.7 | 20.11 / -28.5 | 0.83 (tiny loops) / 0.31 / 0.11 | 2.01 / 0.24 |
| b1: b + lead high shelf +3 dB at 6.5 kHz | +4.8 | -4.1 | 73 % | +4.7 | | | 1.92 / 0.24 |
| b2: b + hats -1.5 dB and ducked 1.5 under the vocals, snare shelf -6.5 (was -4.5) | +4.8 | -4.5 | 72 % | +5.5 | | | 1.99 / 0.21 |
| b3: b + vocal balance 6.5 dB (was 5.2) | +6.1 | -4.4 | 73 % | +4.0 | | | 2.04 / 0.22 |
| b4: b1 + b2 + b3 | +6.1 | -1.4 | 58 % | +8.8 | | | 1.91 / 0.20 |
| **final: b1 + b2 + vocal balance 8.0 dB** | +7.5 | +0.1 | 50 % | +10.3 | 20.01 / -26.2 | 0.51 / 0.31 / 0.11 | 1.85 / 0.18 |
| verse, first pass (neural EQ, balance 5.2, mastered to -11.5 LUFS) | +6.2 | +18.6 | 20 % | +47 | | | 6.0 / 0.42 |
| verse, final (as the chorus; mastered to -13 LUFS) | +9.2 | +23.7 | 7 % | +54 | | | 3.33 / 0.30 |

- Q6's finding holds in the song: without the 4.8 kHz bell, the 9 kHz shelf and half of the 600 Hz cut the lead stem
  gains 1.5 dB of hnr and loses 3 dB above 5 kHz; the words of the stem do not move (base 0.31, small 0.11 both).
- What that EQ costs is the consonants: in the chorus the lead's s / t / k frames were already 3.5 dB under the
  band in 4-10 kHz with the old chain and fall to -5.7 dB. The masker is not the hats and cymbals (the lead's
  consonants are 3-6 dB over them) but the snare (the loudest 4-10 kHz stem of the chorus, hats 1.8 dB under it,
  the lead 8 dB under it on average). A +3 dB shelf at 6.5 kHz gives the consonants back at 0.1 dB of hnr
  (the vowels carry 3 dB less air at the source now), the hats and the snare top come down a little, and the
  vocals go up 2.8 dB against the band: consonants level with the band (+0.1 dB, half of the frames in the clear).
- No clipping (0 clipped samples), true peak -1.02 dBTP in every render, chorus -11.5 LUFS (first demo's mix clip:
  -11.7 LUFS as mp3 with a -0.4 dBFS peak; ours are held under -1 dBFS).
- **The full mix as a recogniser test FAILS, in every variant, the old chain included**: tiny.en, base.en and
  small.en all answer "[Music]" (WER 1.00) on the chorus mix. Graded version (`mix_srt.py`: vocal stems + backing
  stems turned down): small.en follows the chorus at 0.37 once the band is 6 dB lower (variants b and b4), at 0.29
  with the band 12 dB lower, and with the final balance already at 3 dB lower (0.37); tiny.en and base.en still say
  "*Music*" with the band 12 dB down. All vocals without the band: 0.63 / 0.60 / 0.31 (the crowd's "click" shouts
  count as insertions; base.en hears "select all invaders" there, which it misses on the lead alone).
  So the recognisers need the vocals about 3 dB higher than even the final balance to start reading the chorus in
  the mix. I did not go further: at +11 dB the band is a backing track. This is a limit of the test as much as of
  the mix (Whisper labels dense music instead of transcribing it), but it means the mix has no objective word score.
- Low end after the key change: the bass and the slap bass keep their register (the arranger re-voices the
  chords: energy by band 36-48 / 48-62 / 62-90 Hz is -7.9 / -3.7 / -6.6 dB against -5.1 / -7.5 / -6.9 in A minor).

## Step 5: checks on the delivered clips (17:53-17:58; `checks.py`, `run_checks.py`; rows in `runs/checks.jsonl`)

`silent` = notes more than 20 dB under the clip's median note (the worst note's level in brackets); `join` = largest
sample step within 15 ms of a phrase edge, as a share of the clip's peak; `clicks` = discontinuities that stand
alone; pitch = median error / median absolute error against the score in cents / share of notes more than 50 cents
off (notes measured); onset = where the vowel's low band rises, against the note start, ms: median / interquartile
range (notes measured).

| clip (source) | notes | silent | DC, dB re RMS | join | clicks | pitch | onset |
|---|---|---|---|---|---|---|---|
| 01 verse lead dry (`q5_final/verse1_s1`) | 114 | 0 (-8.1) | -76 | 0.0006 | 0 | +0.8 / 6.7 / 1 % (91) | -1.4 / 10.5 (77) |
| 02 chorus lead dry (`q5_final/chorus1_s1`) | 53 | 0 (-7.1) | -77 | 0.0008 | 0 | -0.2 / 6.1 / 2 % (52) | +1.0 / 12.3 (36) |
| 02b chorus dry, written key (`q0_final/chorus1_s1`) | 53 | 0 (-4.1) | -77 | 0.0008 | 0 | -2.0 / 5.4 / 0 % (51) | -0.1 / 12.5 (38) |
| 03 / 04 chorus, lead channel | 53 | 0 (-3.6) | -109 | | 0 | +0.2 / 4.7 / 0 % (51) | -4.1 / 9.8 (35) |
| 03 / 04 chorus, AI Double | 53 | 0 (-2.8) | -112 | | 0 | +8.6 / 9.6 / 0 % (51) | -2.2 / 8.3 (36) |
| 03 / 04 chorus, Harmony | 30 | 0 (-3.7) | -105 | | 0 | +0.6 / 4.1 / 3 % (30) | -3.7 / 7.5 (19) |
| 03 / 04 chorus, Crowd / Crowd 2 / Crowd 3 (5 shouts each) | 5 | 0 | -84 / -84 / -102 | | 0 | +0.2, -23.2, -7.8 (5 each) | 1-2 notes each: -39, +2, -47 |
| 05 verse, lead channel | 114 | 0 (-5.0) | -96 | | 0 | +1.1 / 7.0 / 1 % (91) | -3.1 / 11.5 (74) |
| 05 verse, Aliens ("cookies, delicious", 5 notes) | 5 | 0 (-1.0) | -68 | | 0 | one note measurable: -54 | -7.2 / 9.5 (3) |
| 06 cast: AI line | 15 | 0 (-4.2) | -78 | | 0 | +0.8 / 3.3 / 0 % (14) | -2.1 / 12.0 (12) |
| 06 cast: Aliens line | 15 | 0 (-3.0) | -72 | | 0 | +0.3 / 7.7 / 0 % (8) | -4.5 / 13.8 (10) |
| 06 cast: Aliens 2 chant | 11 | 0 (-8.5) | -73 | | 0 | -3.1 / 8.8 / 0 % (9) | -2.1 / 8.1 (6) |
| 06 cast: Crowd line (3 voices, doubled) | 12 | 0 (-6.1) | -70 | | 0 | +0.3 / 6.2 / 0 % (12) | +14.9 / 5.6 (8) |

- No silent note, no click, no DC in any clip. Phrase joins are clean (10 ms fades on silence).
- Pitch: before the +7 cents the lead sat 6.5 cents flat in the verse and 6.8 in the chorus (`q_t-5`, the bank's
  predictor); now within 1 cent in the median. The AI Double is 8.6 cents sharp by design (vocals.py detunes it 6 cents,
  plus the 7). Aliens 2 measured 12.9 cents flat before its +10 cents, 3.1 after. The Crowd's shouts are detuned by the
  score's own pitchDelta lanes (Crowd 2: -23 cents). The one measurable Aliens note in the verse reads -54 cents:
  its style has an 11 Hz vibrato of 0.4 semitones on every note, on a 0.2 s note that is not a pitch error I can
  separate; its longer cast line is in tune (+0.3).
- Timing: the lead's vowels start 1-4 ms before the note start (IQR 10-12 ms). The crowd reads 15 ms late: its
  doubles are 17-26 ms behind the main takes on purpose.
- Levels (`out/clips.json`): dry clips -16.8 LUFS as mp3 (demo2: -16.6 to -17.2), cast -17.2 (held by the peak
  ceiling), all vocals -15.2 (first demo: -15.2), chorus mix -12.1 and verse mix -13.6 (first demo's mix clip: -11.7
  with a -0.4 dBFS peak). True peak of every wav and decoded mp3 is at or under -1.0 dBFS. One gain per clip, no
  loudnorm (it rides the level).
- Engine: no pitch-predictor fallback and no repaired phrase on any track of either mix (`cast_bars_*.json`).

## The clips (`out/`, wav + 160k mp3; rebuilt 17:58 with `python3 -I make_clips.py`)

| clip | what it is | what to judge (nobody has listened to these: every statement in this file is a measurement) |
|---|---|---|
| `01_verse_lead_dry_5_lower` | AI track, bars 9-24, Qixuan, dry, E minor | the fast lines ("spam folder unread", "accept all, reject all, sorry that is still accept all"): the recognisers still lose them. No vowel is more than 8 dB under the median now (first demo: 12 % of them, down to -36 dB) |
| `02_chorus_lead_dry_5_lower` | AI track, bars 33-48, dry, E minor | the hook and "prove you're not a robot"; whether the voice still sounds out of breath on the held notes (4 dB less noise above 5 kHz than the same key without the offset) |
| `02b_chorus_lead_dry_original_key` | the same settings in the written key (A minor) | A/B for the key change only. This take: 0.46 / 0.03 / 0.29 against 0.23 / 0.14 / 0.06 in E minor (one seed each; three seeds without the air offset: 0.35 / 0.20 / 0.11 against 0.25 / 0.17 / 0.07) |
| `03_chorus_all_vocals_5_lower` | all vocal channels of the chorus mix, no band | the lead against its double and harmony (4.6 dB under it together); the crowd's "click" / "hey" shouts left and right |
| `04_chorus_full_mix_5_lower` | first chorus, band moved to E minor with the vocals | whether the words survive the band: vocals 2.8 dB higher than the song's own balance, consonants level with the band in 4-10 kHz. No recogniser reads this clip |
| `05_verse_full_mix_5_lower` | first verse, full mix | the patter over the half-time band (small.en reads half of it in the mix: 0.49); the aliens' "cookies, delicious" at bar 20 |
| `06_cast_5_lower` | one line each: AI, Aliens (low), Aliens 2 (Japanese chant with its octave shadow), Crowd (three voices, doubled) | whether the four are clearly different characters. Words measured: Aliens 0.00 / 0.00, Crowd 0.00 / 0.00, AI line 0.33 / 0.08 |
| `07_compare_verse_other_voicebank` | Nishiren v2.1, the colleague's existing clip, written key, dry; copied, not re-rendered | the same verse on the clearer bank, whose terms do not let us use it until the author answers (0.44 / 0.38 / 0.17 over three seeds) |
| `08_compare_chorus_other_voicebank` | the same for the chorus | (0.24 / 0.08 / 0.01 over three seeds) |

## How to render again

- Any bars: `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 python3 -I $S/demo3/run_demo.py OUT A B [--key N]
  [--cast FILE.json] [--lead-eq JSON|formant] [--set JSON] [--formant]`. The demo's two sections with the final mix
  settings: `$S/demo3/final_mix.sh [OUT]`. Sung takes are cached in `$S/demo3/cache/` (key: notes, character, options,
  engine file hashes; `DEMO3_NO_CACHE=1` turns it off), so a mix change re-renders in about a minute.
- Dry lead: `python3 -I $S/demo3/batch3.py spec_final` (harness `h3.py`, results `runs/results.jsonl`, wavs
  `runs/q5_final/`). Cast lines: `cast_final.py`, `cast_al2.py`. Clips: `make_clips.py` (its SOURCES name the wavs).
- Checks: `run_checks.py dry TAG CASE SEED TRANSPOSE | stems MIXDIR A B | cast TAG...`, `mix_check.py MIXDIR A B`,
  `asr_clip.py CASE FILE...`, `mix_srt.py CASE A B MIXDIR ATT,...`.

## Moving the song to the lower key in `examples/select_all_invaders/build.py`

What `build3.py` does (45 lines, no change to `vocals.s2s`, `vocals.py` or `backing.json`):
1. `transpose_arrangement(arrangement, n)`: every section's chord symbols are renamed (Am -> Em, F#dim -> C#dim,
   Am/G -> Em/D), `key` strings moved, each section's `transpose` (which moves the riffs, the `notes` patterns)
   changed by n, pitched events (lasers, the outro's notes) renamed. The arranger voices chords from their symbols, so
   bass and guitars stay in their registers (checked on the bass stems above).
2. After `apply_vocal_production`, every vocal note is moved by n (the harmony and the doubles are derived before,
   so their intervals hold).
3. The lead EQ's 600 Hz cut was aimed at the A4-E5 fundamentals; in E minor they are at 330-520 Hz. The demo keeps
   the cut at 600 Hz, halved; not re-tuned.
Not done: the title/key line of the score, SPEC.md, the final chorus' upward key change stays +2 (F# minor), and no
section other than bars 9-24 and 33-48 has been rendered or measured in the new key.

## What is still wrong, from the user's chair (written 18:03; nobody has listened, these are the measurements read plainly)

1. **The fast verse is still not fully understandable.** base.en and small.en miss about one word in four of the dry
   verse (first demo: more than one in two), tiny.en four in ten. The misses are the 16th-note lines at 168 BPM
   (5.4 syllables a second; the bank's own sample songs sing 2.3-2.7). Nothing left in the engine moves them: the
   remaining levers are the song (fewer syllables on those lines, or a slower tempo: Q7 measured base.en 0.24 at
   half tempo on the stock engine) or a clearer bank.
2. **Qixuan stays an airy voice.** cpp 24.1 / 23.5 (verse / chorus) in the low key with -3 dB of vowel air;
   the standard voices of other DiffSinger banks measure 27-29, Nishiren 26.9 on the same verse in the higher key.
   Taking more air out costs words on every recogniser (the dose table), so "out of breath" is reduced, not gone.
3. **The other bank is still clearer where it counts.** Nishiren's existing clips (07, 08; written key) score
   0.44 / 0.38 / 0.17 and 0.24 / 0.08 / 0.01; Qixuan's final lead 0.42 / 0.27 / 0.24 and 0.31 / 0.19 / 0.07, and that is
   with Qixuan given the lower key. Given the same 5 semitones Nishiren measured 0.36 / 0.24 / 0.14 on the verse
   (V1), an octave lower 0.25 / 0.16 / 0.10. Its terms forbid this song until the author answers.
4. **In the full mix no recogniser reads the chorus** (all three answer "[Music]"), although the vocals are 2.8 dB
   higher than the song's balance and the consonants now level with the band in 4-10 kHz. small.en starts reading
   with the band another 3 dB lower. The snare is the masker in 4-10 kHz, not the hats. The verse mix is read by
   small.en at 0.49 (dry 0.24).
5. **The key change is audible as a different song**: E minor, a fourth lower; the chiptune riff and every chord
   voicing move with it. It was done because it is the largest single gain for this bank; only bars 9-24 and 33-48
   exist in the new key. The final chorus (written up to B5, G5 in the hook) would sit at F#5 / D5, still above the
   register measured here.
6. **The cast is measured on one or two lines each** (and the aliens' one-word calls of bars 49-52 read 0.70 / 0.40 / 0.60 on their processed channel). Aliens and Crowd lines read 0.00 / 0.00 dry, but the low
   alien is made by letting the acoustic model sing an octave above the pitch the vocoder renders (hnr 10-12 dB
   there): understandable by the recognisers, its naturalness unjudged. Aliens 2 only chants Japanese, which the
   English recognisers cannot score; its settings were scored on English lines. The crowd is 15 ms late.
7. **The lead's double is a second unison take** 8.6 cents sharp and 6.5 dB down (as the song defines it); with the
   harmony it sits 4.6 dB under the lead. I did not test the chorus without them.
8. **Engine provenance**: the demo sings through the scratch copy `demo3/dsinfer.py` (v67 + three options:
   `vowel_breath_off`, `vowel_voicing_off`, `detune`), the backend `ds_backend3.py` and the song copy `build3.py`.
   None of it is in the repo.
