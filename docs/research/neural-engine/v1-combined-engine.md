# V1 verdict: one engine from four studies, on Nishiren (kept current)

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad

## Status log (newest last)

- 11:14 FIRST LISTEN is on disk: `$S/v67/out/first_{verse1,chorus1,plain}.mp3` (160k, loudnorm -16 LUFS; wav next
  to them). Nishiren v2.1 mode Standard, its own vocoder, steps 10, seed 1, all four options on
  (slow_note 0.25, voicing_floor 6 dB, bank pitch predictor, per-phrase seeded noise). The track's own style
  decides expr: the lead (verse1, chorus1) is robotic -> expr 0.5; `plain` is a pop track -> expr 1.0.
  Made with `cd /home/user/synth2 && python3 -I $S/v67/first.py`.
  voiceq (not yet compared with stock on this bank): verse1 cpp 26.88 hnr 18.67 lvl_sd 5.42 dip -7.47, no vowel
  more than 9 dB under the median vowel; chorus1 cpp 25.85 hnr 25.80; plain cpp 30.65 hnr 21.16.
  For scale, the demo the user heard (Qixuan, `$S/ds1`): verse1 cpp 20.78 hnr 18.20, chorus1 20.99 / 21.40,
  plain 24.96 / 20.20. Render 28 s for 22.6 s sung at 2 threads, load 2-3.
  No pitch-predictor fallbacks on any phrase. WER not measured yet.

## Files

- `$S/v67/dsinfer.py` the engine (options `slow_note`, `voicing_floor`, `pitch_source`, `seed`), `$S/v67/dspitch.py`
  the pitch-predictor port it imports (Q5's, generalised to `spk_embed` banks, line-scanned dictionary).
- `$S/v67/h.py` harness; every measured run is one JSON line in `$S/v67/runs/results.jsonl`, wavs in
  `$S/v67/runs/<tag>/<case>_s<seed>.wav`.
- 11:17 First-listen clips scored (WER tiny.en / base.en / small.en, one take, seed 1): verse1 0.43 / 0.34 / 0.11,
  chorus1 0.14 / 0.06 / 0.03, plain 0.00 / 0.04 / 0.00. The Qixuan demo the user heard scored verse1
  0.56 / 0.56 / 0.38, chorus1 0.29 / 0.26 / 0.06, plain 0.08 / 0.04 / 0.04 (team brief; chorus1 was a lucky draw,
  seeded Qixuan chorus1 is 0.72 / 0.38 / 0.12 over three seeds in Q1). Transcripts: tiny.en and base.en still loop on
  "pop up on a pop up" ("pop pop pop pop pop"); small.en reads the verse almost word for word.
- 11:22 Engine checks pass.
  `python3 -I $S/v67/t_equiv.py --real`: with every option off `v67/dsinfer.py` hands every model the same feeds as
  `q4/dsinfer4.py` (recorder in place of onnxruntime sessions; twinkle, plain, verse1, chorus1, verse2 on Nishiren
  Standard and on Qixuan, verse1 on Nishiren Power: all IDENTICAL), and the real audio of `plain` is bit-identical
  on both banks. `python3 -I $S/v67/t_repro.py`: with `seed=N` a render repeats bit for bit after another render
  in the same process on both banks; a different seed is a different take; the generalised pitch port returns
  exactly what Q5's `dspitch.py` returns on Qixuan for the same seed.
  One deliberate difference from dsinfer4: `phrase_groups` looks notes up by `NoteTiming.index` (Q5's fix for
  scores with `br` notes). On the five test cases the feeds are unchanged (the recorder run above).
- 11:42 Step 3, seed 1 only (INTERIM, three seeds follow; `python3 -I $S/v67/table.py n_` prints the current
  means). Nishiren Standard, per-phrase seeded noise in every row. WER tiny / base / small with loops collapsed.
  verse1: stock 0.56 / 0.43 / 0.41, slow motion 0.49 / 0.29 / 0.15, floor 0.51 / 0.43 / 0.42, bank pitch
  0.49 / 0.40 / 0.36, all three 0.46 / 0.29 / 0.14. verse2 (held out): stock 0.14 / 0.12 / 0.14, all 0.12 / 0.10 / 0.05.
  chorus1: stock 0.26 / 0.23 / 0.06, all 0.17 / 0.06 / 0.03. plain: stock 0.04 / 0.08 / 0.04, all 0.00 / 0.04 / 0.00.
  Weak vowels (more than 8 dB under the clip's median vowel): verse1 stock 5.3 % (worst -31 dB), floor 4.4 %
  (worst -32), slow motion 0.9 % (worst -8), all 0.9 %. Only 0.9 % of Nishiren's verse1 vowels get a predicted
  voicing more than 10 dB under the median (Qixuan: about 10 %): this bank's weak vowels are not the variance
  model's doing, the floor has almost nothing to lift, slow motion is what removes them.
  ASR jitter: the same audio scored from the 16-bit wav instead of the float samples moved verse1 from
  0.455 / 0.295 to 0.432 / 0.341, so single-take WER differences under about 0.05 mean nothing.

## Step 3: what holds on Nishiren (three seeds; `python3 -I $S/v67/batch.py spec3`, table `python3 -I $S/v67/table.py --md n_`)

Nishiren v2.1 Standard, steps 10, its own vocoder, per-phrase seeded noise in every row (seeds 1, 2, 3), 2 threads,
load 3-4. WER = tiny.en / base.en / small.en with repetition loops collapsed, mean of three seeds, then the median
in brackets (the mean is pulled by recogniser accidents: one small.en transcript is just "*sings*", two transcripts
end in a loop that swallows the rest of the clip). `weak` = share of vowels more than 8 dB under the clip's median
vowel (worst one). Full table: `$S/v67/runs/table_step3.md`.

| case | variant | WER mean | WER median | cpp | hnr | lvl_sd | dip | weak |
|---|---|---|---|---|---|---|---|---|
| verse1 (88 words) | stock | 0.507 / 0.387 / 0.296 | 0.55 / 0.40 / 0.27 | 25.66 | 18.79 | 5.65 | -8.18 | 3.8 % (-38 dB) |
| | slow motion | 0.455 / 0.344 / 0.178 | 0.47 / 0.35 / 0.19 | 26.57 | 19.04 | 5.59 | -7.88 | 1.5 % (-28) |
| | voicing floor 6 dB | 0.541 / 0.401 / 0.356 | 0.51 / 0.39 / 0.35 | 25.68 | 18.72 | 5.63 | -8.16 | 2.9 % (-32) |
| | bank pitch (expr 0.5) | 0.500 / 0.436 / 0.549* | 0.49 / 0.42 / 0.36 | 25.99 | 18.54 | 5.75 | -8.28 | 3.2 % (-37) |
| | all three | 0.436 / 0.379 / 0.167 | 0.44 / 0.31 / 0.18 | 26.87 | 18.73 | 5.43 | -7.64 | 1.5 % (-25) |
| verse2, held out (42 words) | stock | 0.135 / 0.095 / 0.119 | 0.14 / 0.10 / 0.12 | 26.73 | 20.95 | 5.56 | -8.20 | 1.2 % (-10) |
| | slow motion | 0.103 / 0.079 / 0.071 | 0.12 / 0.07 / 0.07 | 27.35 | 20.48 | 5.38 | -8.18 | 0 % |
| | voicing floor | 0.151 / 0.095 / 0.119 | 0.14 / 0.10 / 0.12 | 26.74 | 20.96 | 5.51 | -8.04 | 0.6 % |
| | bank pitch | 0.143 / 0.095 / 0.103 | 0.14 / 0.10 / 0.10 | 27.14 | 20.95 | 5.21 | -7.72 | 0.6 % |
| | all three | 0.119 / 0.095 / 0.087 | 0.12 / 0.10 / 0.10 | 27.53 | 20.51 | 5.00 | -7.61 | 0 % |
| chorus1 (35 words) | stock | 0.200 / 0.181 / 0.086 | 0.23 / 0.17 / 0.06 | 25.59 | 25.70 | 4.97 | -6.87 | 1.3 % (-10) |
| | slow motion | 0.286 / 0.048 / 0.019 | 0.29 / 0.06 / 0.03 | 25.63 | 25.65 | 5.01 | -6.64 | 0 % |
| | voicing floor | 0.162 / 0.172 / 0.086 | 0.14 / 0.14 / 0.11 | 25.61 | 25.70 | 4.89 | -6.67 | 0 % |
| | bank pitch | 0.172 / 0.124 / 0.048 | 0.20 / 0.14 / 0.06 | 25.90 | 25.73 | 5.05 | -6.81 | 0.6 % |
| | all three | 0.238** / 0.076 / 0.010 | 0.17 / 0.09 / 0.00 | 26.09 | 25.72 | 4.99 | -6.42 | 0 % |
| plain (24 words) | stock = slow motion (no fast note: bit-identical) | 0.028 / 0.028 / 0.014 | 0.04 / 0.00 / 0.00 | 29.89 | 24.12 | 4.05 | -5.03 | 0 % |
| | voicing floor | 0.028 / 0.028 / 0.014 | 0.04 / 0.00 / 0.00 | 29.89 | 24.11 | 4.05 | -5.01 | 0 % |
| | bank pitch (pop: expr 1.0) | 0.014 / 0.028 / 0.014 | 0.00 / 0.04 / 0.00 | 30.51 | 21.72 | 3.99 | -5.07 | 0 % |
| | all three | 0.000 / 0.028 / 0.014 | 0.00 / 0.04 / 0.00 | 30.50 | 21.71 | 3.97 | -5.14 | 0 % |

\* one of the three small.en transcripts is "*sings*" (WER 1.00); the other two are 0.36 and 0.28.
\** one tiny.en transcript loops ("notra-r-r-r-...") and loses the rest of the clip (0.43 after collapsing);
the other two are 0.17 and 0.11.

What this says:

- Stock Nishiren is already far ahead of stock Qixuan on the same notes (Qixuan, Q1's seeded runs: verse1
  0.644 / 0.519 / 0.432, verse2 0.516 / 0.397 / 0.246, chorus1 0.724 / 0.381 / 0.124).
- **Slow motion still gains, less than on Qixuan.** verse1 -0.05 / -0.04 / -0.12 by the mean, -0.08 / -0.05 / -0.08
  by the median (Qixuan: -0.14 / -0.17 / -0.19); small.en is better on every seed (0.41, 0.20, 0.27 -> 0.15, 0.19,
  0.19), tiny.en on every seed, base.en on one of three. chorus1 base.en 0.18 -> 0.05 and small.en 0.09 -> 0.02
  (every seed better); its tiny.en is noise (0.11-0.46). cpp +0.9 dB on verse1 (same notes), hnr equal. Keep ON.
- **The voicing floor does nothing on this bank** (verse1 0.447 -> 0.471 on tiny+base, inside the noise; chorus1,
  verse2 and plain unchanged). Nishiren's variance model hardly ever predicts a collapsed vowel: 1.5 % of verse1's
  vowels more than 10 dB under the median voicing, 0 % in the other three cases (Qixuan verse1: about 10 %). The
  floor is a Qixuan fix. It costs nothing here, so it can stay on as a safety net, but it is not what makes
  Nishiren better.
- **Nishiren does drop syllables in the fast verse, for another reason.** 3.8 % of verse1's vowels are more than
  8 dB down (worst -24 to -38 dB per take; Qixuan: 11 %). `python3 -I $S/v67/x_weak.py d_stock verse1 1`: they are
  the vowels the duration model squeezed to 1-2 frames (12-23 ms) between consonant clusters on 16th notes:
  "ac-c[e]pt" (k s [eh] p t, 2 frames, -31 dB), "that [i]s still" (t [ih] z s, 1 frame, -17 dB), "I w[a]s trained"
  (w [aa] z t, 2 frames, -13 dB). 3 of the 4 vowels of 1-2 frames are weak, against 3 % of the longer ones. The
  floor lifts their voicing (-38.6 -> -4.6 dB) and the syllable stays at -32 dB: two frames are not enough for a
  vowel. Slow motion gives them twice the frames inside the models and brings them back (weak 1.5 %, in seed 1
  0.9 % with the worst at -8 dB).
- **The bank's pitch predictor neither helps nor hurts the words here.** verse1 0.447 -> 0.468 (tiny+base mean),
  verse2 0.115 -> 0.119, chorus1 0.191 -> 0.148 (base.en 0.18 -> 0.12, small.en 0.09 -> 0.05), plain equal. On
  top of slow motion (all three against slow motion alone): verse1 0.399 -> 0.407, chorus1 0.167 -> 0.157,
  verse2 0.091 -> 0.107: nothing outside the noise. cpp +0.3 to +0.6 dB in every case. On `plain` (pop, expr 1.0)
  hnr falls from 24.1 to 21.7 dB; whether that is vibrato in the 64 ms analysis window or rougher phonation is
  checked with an expr sweep (`spec7`, below).
- All three together against stock: verse1 0.507 / 0.387 / 0.296 -> 0.436 / 0.379 / 0.167 (median 0.55 / 0.40 /
  0.27 -> 0.44 / 0.31 / 0.18), chorus1 base.en 0.18 -> 0.08 and small.en 0.09 -> 0.01, verse2 and plain unchanged
  (already 0.10 and 0.03), cpp +1.2 dB on verse1, level dips 0.5 dB shallower.
- 12:00 Words still missed on verse1 with all three fixes (`python3 -I $S/v67/words.py verse1 n_stock n_slow n_all
  --min 0.6`; miss rate over 9 transcripts = 3 seeds x 3 recognisers; overall 0.361 stock -> 0.290 all):
  "parked across" (heard "park to cross"), "valued users" ("value juicers"), "spam" ("pam" / "bam"), "unread"
  ("and read", a fair hearing), "knocked" ("not"), "accept all" twice, "still", "pop up" (the "up"s), "ninety"
  ("nineteen"). One word got worse with slow motion: "eight" billion (0.22 -> 0.89, heard "a billion": the
  unreleased t before b). chorus1 misses only "a" robot ("the robot").
- 12:20 Step 4, seed 1 only (INTERIM; controls one at a time on top of all three fixes, `spec4.py`; WER tiny /
  base / small collapsed; reference n_all verse1 0.46 / 0.29 / 0.14, chorus1 0.17 / 0.06 / 0.03):
  velocity 1.11: 0.46 / 0.28 / 0.12 and 0.17 / 0.06 / 0.03; velocity 1.23: 0.47 / 0.33 / 0.12 and 0.14 / 0.06 / 0.06;
  tension +2: 0.43 / 0.31 / 0.12 and 0.23 / 0.17 / 0.03 (cpp +1.4, hnr -1.7); tension +4: 0.69 / 0.41 / 0.15 and
  1.00 / 0.17 / 0.03; tension -2: 0.36 / 0.29 / 0.15 and 0.29 / 0.09 / 0.06 (cpp -3.0, hnr +1.8);
  Power: 0.44 / 0.35 / 0.27 and 0.40 / 0.14 / 0.03; steps 20: 0.43 / 0.35 / 0.12 and 0.17 / 0.06 / 0.03 (+25 % time);
  5 semitones lower: 0.32 / 0.23 / 0.10 and 0.14 / 0.14 / 0.00; 12 semitones lower: 0.23 / 0.10 / 0.07 and
  0.09 / 0.00 / 0.00. The register is the only control that moves the words on seed 1.

## Step 4: Nishiren's own controls (three seeds; `python3 -I $S/v67/batch.py spec4`, table `$S/v67/runs/table_step4.md`)

One factor at a time on top of all three fixes (reference row). WER tiny / base / small, loops collapsed, mean of
three seeds (median). Differences under about 0.06 on verse1 and 0.10 on chorus1 are noise.

| control | verse1 WER | chorus1 WER | verse1 cpp / hnr | chorus1 cpp / hnr | verdict |
|---|---|---|---|---|---|
| reference: Standard, steps 10, velocity 1, tension +0, written pitch | 0.436 / 0.379 / 0.167 (0.44 / 0.31 / 0.18) | 0.238 / 0.076 / 0.010 (0.17 / 0.09 / 0.00) | 26.87 / 18.73 | 26.09 / 25.72 | |
| velocity 1.11 (VELC 115) | 0.485 / 0.326 / 0.151 | 0.143 / 0.029 / 0.010 | 26.97 / 18.87 | 26.12 / 25.74 | no effect |
| velocity 1.23 (VELC 130) | 0.481 / 0.299 / 0.170 | 0.114 / 0.029 / 0.019 | 27.06 / 19.05 | 26.18 / 25.80 | no effect outside the noise; leave at 1.0 |
| tension +2 (TENC +40) | 0.455 / 0.311 / 0.148 | 0.162 / 0.086 / 0.010 | 28.27 / 17.02 | 26.99 / 24.14 | words equal; pressed tone: cpp +1.4, hnr -1.7 |
| tension +4 (TENC +80) | 0.519 / 0.367 / 0.201 | 0.390 / 0.067 / 0.029 | 28.37 / 16.36 | 26.98 / 23.31 | worse: hnr -2.4, more recogniser loops |
| tension -2 (TENC -40) | 0.504 / 0.307 / 0.174 | 0.238 / 0.048 / 0.019 | 23.99 / 20.44 | 23.71 / 26.64 | words equal; breathier: cpp -2.9 (the wrong way for "out of breath") |
| mode Power | 0.462 / 0.371 / 0.254 | 0.248 / 0.114 / 0.010 | 27.39 / 17.56 | 26.44 / 24.27 | not better: small.en +0.09 on the verse, hnr -1.2, level less even (lvl_sd +0.5) |
| steps 20 | 0.424 / 0.356 / 0.163 | 0.143 / 0.038 / 0.010 | 26.90 / 18.68 | 26.05 / 25.57 | no effect; render 27 s -> 34 s |
| 5 semitones lower | 0.356 / 0.235 / 0.136 (0.35 / 0.23 / 0.12) | 0.114 / 0.086 / 0.000 | 28.54 / 18.35 | 28.67 / 24.95 | GAIN on the verse, every recogniser, every seed (-0.08 / -0.14 / -0.03) |
| 12 semitones lower | 0.246 / 0.159 / 0.095 (0.23 / 0.18 / 0.10) | 0.133 / 0.010 / 0.000 | 28.69 / 16.46 | 30.52 / 23.22 | BIG GAIN on the verse (-0.19 / -0.22 / -0.07, every seed); hnr -2.3 to -2.5 dB (cpp is not comparable across registers) |

- The register is the one control that moves the words. The verse sits at A4-E5 (MIDI 69-76); an octave lower
  (A3-E4) halves its WER and keeps the song's key. 5 semitones lower gets half of that gain and keeps the tone
  (hnr unchanged), but changes the key against the band.
- Velocity (the README's advice for fast songs), tension offsets, Power and 20 steps do not buy intelligibility.
  Recommended: Standard, steps 10, velocity 1.0, tension offset 0.

## Step 5: regression on Qixuan (three seeds; `python3 -I $S/v67/batch.py spec5`, table `$S/v67/runs/table_step5.md`)

Qixuan v2.7.0 + pc-nsf-hifigan from `$S/models/installed`, steps 10, depth 0.6, per-phrase seeded noise. WER tiny /
base / small, loops collapsed, mean of three seeds (median).

| case | variant | WER mean | WER median | cpp | hnr | lvl_sd | dip | weak vowels (worst) | vowels with collapsed voicing |
|---|---|---|---|---|---|---|---|---|---|
| verse1 | stock | 0.637 / 0.568 / 0.417 | 0.65 / 0.56 / 0.43 | 20.93 | 18.13 | 6.32 | -9.95 | 12.3 % (-36 dB) | 9.6 % |
| | slow motion | 0.655* / 0.394 / 0.299 | 0.49 / 0.39 / 0.32 | 21.54 | 17.93 | 6.00 | -8.21 | 1.5 % (-28) | 1.2 % |
| | voicing floor 6 dB | 0.640 / 0.621 / 0.405 | 0.64 / 0.57 / 0.41 | 20.94 | 17.89 | 6.03 | -8.80 | 2.6 % (-14) | 9.6 % predicted, 0 % fed |
| | all three | 0.443 / 0.424 / 0.261 | 0.44 / 0.43 / 0.25 | 21.43 | 17.57 | 5.84 | -7.73 | 0 % (-6) | 0.3 % |
| chorus1 | stock | 0.638 / 0.190 / 0.181 | 0.66 / 0.14 / 0.17 | 21.07 | 21.44 | 5.74 | -7.62 | 0 % | 0 % |
| | all three | 0.353 / 0.200 / 0.114 | 0.40 / 0.26 / 0.11 | 21.09 | 20.89 | 5.33 | -6.75 | 0 % | 0 % |
| plain | stock | 0.111 / 0.042 / 0.042 | 0.12 / 0.04 / 0.04 | 24.84 | 20.29 | 4.55 | -5.43 | 0 % | 0 % |
| | all three | 0.111 / 0.042 / 0.042 | 0.12 / 0.04 / 0.04 | 25.22 | 19.69 | 4.84 | -5.83 | 0 % | 0 % |
| twinkle (10 words) | stock | 0.333 / 0.233 / 0.133 | 0.30 / 0.20 / 0.10 | 24.73 | 20.98 | 4.82 | -4.92 | 0 % | 0 % |
| | all three | 0.500 / 0.400 / 0.133 | 0.50 / 0.40 / 0.10 | 24.94 | 21.04 | 4.53 | -5.37 | 0 % | 0 % |

\* one tiny.en transcript of the slow-motion verse scores 1.00; the other two are 0.48 and 0.49.

- Q1's gain is reproduced: stock 0.637 / 0.568 / 0.417 (Q1 measured 0.644 / 0.519 / 0.432), slow motion median
  0.49 / 0.39 / 0.32 (Q1: 0.504 / 0.349 / 0.246 mean). Gain by the medians -0.16 / -0.17 / -0.11 (Q1: -0.14 / -0.17 /
  -0.19).
- Q6's gain is reproduced: vowels more than 8 dB down 12.3 % -> 2.6 %, the worst from -36 dB to -14 dB (Q6: 11.4 %
  -> 2.6 %, worst -9 dB on its seed); as in Q6, the floor alone does not move WER.
- Combined on Qixuan: verse1 0.637 / 0.568 / 0.417 -> 0.443 / 0.424 / 0.261, no vowel more than 6 dB down in any
  seed, level dips 2.2 dB shallower; chorus1 tiny.en 0.64 -> 0.35 (Q5's predictor gain), base.en equal.
- plain is unchanged on Qixuan. **twinkle on Qixuan looks worse with everything on** (tiny.en 0.33 -> 0.50, base.en
  0.23 -> 0.40; one word of ten is 0.10). Both versions mishear "twinkle" ("wing-cote", "wink-old"); stock gets
  it right in two base.en transcripts of three, the combined engine in none. Slow motion is inactive on twinkle
  (no fast note), so it is the floor or the pitch predictor at expr 1.0: `spec9` isolates it with six seeds.
  On Nishiren twinkle goes the other way (stock 0.267 / 0.100 / 0.000, all 0.167 / 0.100 / 0.000).

## expr of the pitch predictor on Nishiren (three seeds; `spec7`, table `$S/v67/runs/table_expr.md`)

All three fixes on; WER tiny / base / small collapsed, mean of three seeds.

| case (track style) | expr | WER | cpp | hnr |
|---|---|---|---|---|
| verse1 (robotic) | Synth2 flat pitch (slow motion only) | 0.455 / 0.344 / 0.178 | 26.57 | 19.04 |
| | 0.5 (default for robotic) | 0.436 / 0.379 / 0.167 | 26.87 | 18.73 |
| | 0.75 | 0.439 / 0.322 / 0.185 | 27.17 | 18.68 |
| | 1.0 | 0.443 / 0.341 / 0.182 | 26.90 | 18.51 |
| chorus1 (robotic) | Synth2 flat pitch | 0.286 / 0.048 / 0.019 | 25.63 | 25.65 |
| | 0.5 | 0.238 / 0.076 / 0.010 | 26.09 | 25.72 |
| | 0.75 | 0.114 / 0.114 / 0.000 | 26.80 | 24.40 |
| | 1.0 | 0.124 / 0.105 / 0.000 | 27.21 | 23.63 |
| plain (pop) | Synth2 pop pitch | 0.028 / 0.028 / 0.014 | 29.89 | 24.12 |
| | 0.5 | 0.000 / 0.056 / 0.014 | 29.68 | 23.96 |
| | 0.75 | 0.000 / 0.028 / 0.000 | 30.32 | 22.77 |
| | 1.0 (default for pop) | 0.000 / 0.028 / 0.014 | 30.50 | 21.71 |

- expr does not move the words on this bank (all inside the noise). It is a style control.
- hnr falls steadily as expr rises (plain 24.0 -> 22.8 -> 21.7, chorus1 25.7 -> 24.4 -> 23.6) while cpp rises
  (+0.8 on plain, +1.1 on chorus1): the predictor's vibrato and drift move the pitch inside voiceq's 64 ms
  analysis window. It is not rougher phonation; do not read the hnr drop of a predicted-pitch clip as a loss.
- Keep Q5's table: robotic 0.5, soft 0.75, every other style 1.0.

## Follow-ups on Nishiren (three seeds; `spec8`, table `$S/v67/runs/table_step4b.md`)

| question | rows (WER tiny / base / small, mean of 3 seeds) | answer |
|---|---|---|
| Are the fixes still needed an octave lower? | verse1 -12: stock 0.295 / 0.295 / 0.208, all three 0.246 / 0.159 / 0.095. chorus1 -12: stock 0.219 / 0.133 / 0.038, all three 0.133 / 0.010 / 0.000 | Yes. The register and the fixes add up: verse1 stock at pitch 0.507 / 0.387 / 0.296 -> fixes 0.436 / 0.379 / 0.167 -> fixes an octave lower 0.246 / 0.159 / 0.095 |
| Held-out verse an octave lower | verse2 all three: 0.119 / 0.095 / 0.087 at pitch, 0.079 / 0.072 / 0.040 at -12 | holds (small numbers, 42 words) |
| Is `plain` better lower? | all three: 0.000 / 0.028 / 0.014 at pitch, 0.000 / 0.014 / 0.000 at -5, 0.070 / 0.000 / 0.000 at -12 (C#3-A3: hnr 21.0, one weak vowel) | No: plain (C#4-A4) is already in the bank's range. No `plain_nishiren_best_low` clip |
| Is velocity a cheaper stand-in for slow motion? | verse1 floor + pitch + velocity 1.23, no slow motion: 0.534 / 0.413 / 0.307 (stock 0.507 / 0.387 / 0.296) | No |

## Qixuan twinkle: the pitch predictor costs about one word in ten (six seeds; `spec9`)

| variant (Qixuan, twinkle, 10 words) | WER tiny / base / small, mean of 6 seeds | base.en per seed |
|---|---|---|
| stock | 0.367 / 0.217 / 0.117 | 0.2 0.1 0.4 0.1 0.4 0.1 |
| voicing floor | 0.367 / 0.217 / 0.117 | same transcripts as stock |
| bank pitch, expr 1.0 (pop default) | 0.450 / 0.350 / 0.133 | 0.4 0.4 0.4 0.4 0.4 0.1 |
| bank pitch, expr 0.5 | 0.417 / 0.300 / 0.100 | 0.4 0.4 0.4 0.1 0.4 0.1 |
| all three (= bank pitch: no fast note, floor inactive) | 0.450 / 0.350 / 0.133 | 0.4 0.4 0.4 0.4 0.4 0.1 |

The loss is the word "twinkle" (twice): stock is heard "Winkle, twinkle little star" by base.en in four seeds of
six, with the predicted pitch it is "Wing cold, wing cold" in five of six. small.en is unchanged. So step 5's
"twinkle does not get worse" FAILS on Qixuan by one to two words of ten on the two small recognisers, and the
cause is the pitch predictor, not slow motion (inactive) or the floor (no change). On Nishiren twinkle does not
get worse (stock 0.267 / 0.100 / 0.000, all 0.167 / 0.100 / 0.000, three seeds); `plain` does not get worse on
either bank. Against that, the predictor's gain on Qixuan's chorus1 (tiny.en 0.64 -> 0.35 here with everything on,
0.72 -> 0.51 in Q5 for the predictor alone, 6 of 6 pairs) is larger and on the song material.

## A fifth option tried and NOT adopted: minimum vowel length (three seeds; `spec10`, table `$S/v67/runs/table_minv.md`)

`min_vowel=0.04`: in the alignment a vowel left shorter than 0.04 s takes the difference from the consonants of its
note (at most half the note). It does what it was built for and the recognisers do not care:

| case | all three fixes | + min_vowel 0.04 |
|---|---|---|
| Nishiren verse1 | 0.436 / 0.379 / 0.167, weak 1.5 % (worst -25 dB) | 0.508 / 0.413 / 0.216, weak 0.3 % (worst -10 dB) |
| Nishiren verse1, octave lower | 0.246 / 0.159 / 0.095, weak 0.9 % (-14) | 0.265 / 0.159 / 0.118, weak 0 % (-8) |
| Qixuan verse1 | 0.443 / 0.424 / 0.261 | 0.485 / 0.436 / 0.341 |
| Nishiren chorus1, verse2, plain | | bit-identical (no vowel under 0.04 s) |

The dropped syllables come back in level, WER does not improve (it is 0.02-0.08 worse on every recogniser at the
written pitch, inside the noise but never better): the time the vowel gains is taken from consonant clusters that
were already short. The option stays in `dsinfer.py`, off, and is not in `config.json`.

## Other banks, and a vocoder that draws noise (13:45)

`python3 -I $S/v67/t_fallback.py` (verse2, every option on, seeded): Aurelian v108 (no `dsvariance/`, no `dspitch/`)
sings with Synth2's pitch (`info["pitch_source"] == "synth2"`), the floor has nothing to act on; Alemon v3.5 sings
with its own `dspitch` (speaker embedding, own phoneme table), no phrase falls back. Both repeat bit for bit.
`python3 -I $S/v67/x_det.py`: Aurelian's bundled vocoder (`tgm_hifigan_v107.onnx`) draws noise inside the graph
(the same session run twice on the same mel gives different audio); Nishiren's and the OpenVPI vocoder do not. So
`DS.reseed` now also makes a fresh vocoder session per phrase (0.07 s). `python3 -I $S/v67/t_same.py`: after that
change the engine still reproduces four stored measurement clips sample for sample, so every number above is the
final engine's.

## The predictor's octave drops on consonants (13:55; `python3 -I $S/v67/x_range.py nishiren`, pitch models only)

Q5's sanity rule sends a whole phrase back to Synth2's pitch when one sung frame lies more than 6 semitones outside
the notes within 0.25 s. Qixuan never came near it (max 2.9). Nishiren's predictor does, for a few frames at a
time, on consonants, and the value is exactly 12 semitones under the note (the model's own clamp):

| case | expr 0.5: phrases with a frame over 6 st (worst) | expr 1.0 |
|---|---|---|
| verse1 (2 phrases of up to 16.8 s, 3 seeds) | 0 of 6 (5.1 st on a z) | 3 of 6 (10.0 st: 4 frames on a t, predicted 64.0 at note 76) |
| chorus1 | 0 of 18 (4.8 st on a b) | 1 of 18 (9.0 st, 3 frames on a b) |
| verse2 | 0 of 12 (2.6) | 0 of 12 (5.2 on a z) |
| plain | 0 of 9 (0.7) | 0 of 9 (2.8) |
| twinkle | 1 of 3 (10.0 st: 7 frames on the hh of "how") | 1 of 3 |

In the measurements above this rule fired in 13 runs: `n_expr0.75` and `n_expr1.0` (verse1 in 5 of 6 runs, chorus1
in 2 of 6), `n_power` (3 of 6), `n_t-5` (2 of 6) and `n_all` twinkle seed 2; in each, one phrase was sung with
Synth2's pitch. No run of `n_all` (verse1, chorus1, verse2, plain), `n_t-12` or any Qixuan row was affected.
So the expr 0.75 / 1.0 rows of verse1 are half Synth2 pitch, which does not change their reading (no effect on the
words) but means expr 1.0 on a long Nishiren phrase was not really tested there.

Change made in `dsinfer.py` (`BankPitch.phrase`, `bridge`): outlier frames (over 6 st out, at most 10 % of the
phrase's sung frames) are replaced by a straight line between their neighbours, two frames of margin; only a phrase
with more outliers than that, or a non-finite prediction, goes back to Synth2's pitch. A phrase without outliers is
untouched, so every clip without a fallback is reproduced bit for bit (`t_same.py` passes after the change). The
affected runs are measured again as `b_*` (`spec11`).
