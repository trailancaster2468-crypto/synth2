# Q1 findings: intelligibility of fast verses (working file, appended as results arrive)

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad

## 0. Summary (kept current; details and commands below)

- Cause: the bank cannot articulate English at this rate. 168 BPM eighths are 5.6 syllables/s,
  15 frames per syllable. The same notes at 120 BPM score 0.375 / 0.347, at 84 BPM 0.426 / 0.216.
  Alignment, phoneme conventions, phrase length, velocity and register are not the cause
  (each tested, none gives a gain above the noise).
- Fix that works: SLOW-MOTION RENDERING. Notes shorter than 0.25 s are sung up to 2x slower by
  the variance and acoustic models and the mel is squeezed back onto the real frame grid before
  the vocoder (which gets the real f0). Timing and pitch are unchanged.
  verse1: 0.644 / 0.519 / 0.432 -> 0.504 / 0.349 / 0.246 (3 seeds, every seed better).
  verse2 (held out): 0.516 / 0.397 / 0.246 -> 0.349 / 0.278 / 0.182.
  twinkle, plain, neon, grace: bit-identical to the spike (no fast notes).
  chorus1: see section 3 (recogniser repetition loops; more seeds).
  Tone (voiceq): cpp +0.5 dB, hnr equal, level steadier. No tone penalty.
- Target (verse1 < 0.30 on tiny and base) is NOT reached: 0.50 / 0.35. The bank itself, singing
  this text at half tempo, does not reach it on tiny.en (0.43).
- Deliverables: `$S/q1/dsinfer.py` (= `dsfinal.py`), `$S/q1/recommended.diff`, clips in `$S/q1/out/`.
- Noise floor: one seed moves one recogniser by up to 0.18 on verse1 and more on the 35-42 word
  cases; only 3-seed means that differ by more than ~0.06 on tiny+base are treated as findings.

## How to reproduce anything here

- Harness: `$S/q1/h.py` (cases, references, seeded fresh sessions, scoring, word alignment).
- One run: `cd /home/user/synth2 && ASR_THREADS=2 python3 -I $S/q1/run.py TAG --mod MODULE
  --cases verse1 --seeds 1,2,3 [--kw key=value ...] [--case-kw tempo=120 ...]`
  writes `$S/q1/runs/TAG/result.json` (wavs next to it). MODULE is a file in `$S/q1/`
  (`dsx` = spike + logging of predicted durations).
- Batches: `python3 -I $S/q1/batch.py SPEC.py` (list of run dicts; appends to `runs/LOG.txt`).
- Per-word miss table: `python3 -I $S/q1/words.py TAG`; durations: `python3 -I $S/q1/durs.py TAG`.
- Every run uses a fresh session per seed (`onnxruntime.set_seed(seed)`), 2 threads.
- WER triples are tiny.en / base.en / small.en; `[min-max]` over seeds.

## 1. Noise floor and diagnostics (first run of this job, re-read from `runs/`)

Spike, verse1, 168 BPM, seeds 1,2,3 (`runs/base0`):
tiny 0.644 [0.57-0.69] sd 0.067, base 0.519 [0.49-0.53] sd 0.026, small 0.432 [0.36-0.55] sd 0.099.
One seed moves a single-recogniser score by up to 0.1-0.18; a 3-seed mean of tiny+base has a
standard error of about 0.03. Differences under ~0.06 on the tiny+base mean are not findings.

Same lyrics, changed musical conditions (2 seeds each, spike unchanged):

| condition | tiny | base | small |
|---|---|---|---|
| 168 BPM as written (3 seeds) | 0.644 | 0.519 | 0.432 |
| 120 BPM | 0.375 | 0.347 | 0.238 |
| 84 BPM | 0.426 | 0.216 | 0.199 |
| transpose -7 | 0.585 | 0.483 | 0.461 |
| transpose -12 | 0.545 | 0.522 | 0.375 |
| monotone 64 | 0.517 | 0.420 | 0.409 |
| monotone 71 | 0.637 | 0.534 | 0.364 |

Reading: tempo is the big lever (base halves at half tempo); register and melody are not.
Even at 84 BPM tiny stays at 0.43, so part of the residue is the lyric itself / recogniser.

Words missed in 9/9 (3 seeds x 3 recognisers) at 168 BPM: parked, lawn, valued, users, beaming,
(to) the spam folder unread, knocked, accept all reject (all), that is still accept, up on (a pop) up,
trained. The 16th-note run "sorry that is still accept all" has 7-8 frame syllables with
1-3 frame consonants.

Second run: `runs/a_base` (parametrised module `dsq.py`, all options off) reproduces `runs/base0`
exactly (0.644 / 0.519 on seeds 1-3), so seeded fresh sessions are repeatable across processes
and `dsq.py` with defaults is the spike.

### Where the misses are (`python3 -I $S/q1/loc.py base0 diag_t84 diag_t120`, tiny+base miss rate per reference word)

| group of reference words | 168 BPM | 120 BPM | 84 BPM |
|---|---|---|---|
| all 88 words | 0.56 | 0.33 | 0.29 |
| on 16th notes (n=14) | 0.71 | 0.39 | 0.32 |
| on 8th notes (n=68) | 0.50 | 0.26 | 0.25 |
| on a quarter or longer (n=6: parked, eight, spam, all, all, unread) | 0.86 | 0.88 | 0.71 |
| pitch E5 or higher (n=16) | 0.73 | 0.50 | 0.38 |
| below E5 (n=72) | 0.52 | 0.29 | 0.27 |
| reached by a jump of 4+ semitones (n=26) | 0.67 | 0.48 | 0.35 |
| function words (n=33) | 0.38 | 0.09 | 0.14 |
| content words (n=55) | 0.66 | 0.47 | 0.38 |

- The verse is HIGH, not low: notes are G#4..A5, mostly B4-C5, joke words on F5 ("spam", "all")
  and A5 ("up" of "pop up", an octave jump). Those words fail at every tempo.
- Phrase position does not matter (after a rest 0.48, before a rest 0.57, inside 0.56).
- The whole verse is two phrases for the spike: 16.4 s (218 phonemes) and 5.5 s, because the
  rests are one eighth (0.18 s) and `PHRASE_GAP` is 0.30 s.
- Predicted and final durations are almost equal (the duration model already sees the word
  length), so the `max(pred, 0.5)` scaling is not what squeezes consonants. The model itself
  gives 1-3 frames to `dh`, `t`, `d` before another consonant and 2-4 frames to everything in
  the 16th-note run (`python3 -I $S/q1/durs.py base0`).
- Many "misses" are legitimate connected-speech readings of an odd text, not engine faults:
  "parked across" -> "park to cross", "valued users" -> "value juices" (d+y), "accept all" ->
  "(ac)cep tall / at the door", "sent it to the spam folder unread" -> "send it to some folder
  and read". small.en recovers part of them from context.

### Phoneme conventions (CMUdict as Synth2 emits it versus the bank's `dsdict-en.yaml`)

Over 9282 words found in both, 88 % are identical. The systematic differences are:
`t r` -> `ch _r` and `d r` -> `jh _r` (bank: 8140 + 2572 entries, only 2 plain `t r`), and
`t`/`d` between a vowel and an unstressed vowel -> `dx` (flap; after `r` never). On the test
texts this touches 7 of 159 words: trained, tried, invaders, deleting, little, city, and (ae -> ax).
`ax`/`ah`, `er` and syllabic consonants already agree.

OpenUtau's `ProcessWord` has one rule the spike lacks: in a consonant-glide-vowel syllable
(`en/_r l r w y` are type `liquid`) the glide starts the note, not the vowel ("three" =
`th | r iy`, "from" = `f | r ah m`, "valued" = `v ae l | y uw d`). The duration model was
trained with that word division.

## 2. Hypotheses, one at a time (verse1, 168 BPM, seeds 1-3, tiny / base; baseline 0.644 / 0.519)

Screening runs score tiny.en and base.en only (small.en is added for the candidates).
Specs: `spec_a.py`, `spec_b.py` (run with `batch.py`); module `dsq.py`, options in `opt={...}`.
The machine was at load 6-10 during these runs (render 45 s for 22.7 s sung), timings are not
representative.

| run | change | tiny | base | verdict |
|---|---|---|---|---|
| a_base | none | 0.644 [0.57-0.69] | 0.519 [0.49-0.53] | baseline |
| a_split8 | phrases cut at rests to <= 8 s (`max_phrase=8`) | 0.708 [0.62-0.77] | 0.583 [0.50-0.69] | no gain (worse, inside noise) |
| a_vel14 | acoustic `velocity` 1.4 | 0.591 [0.51-0.65] | 0.583 [0.44-0.80] | no gain |
| b_slow14 | slow-motion rendering x1.4: variance + acoustic model run with every phoneme 1.4x longer and the pitch curve stretched to match, the mel is then squeezed back to the real frame grid (linear interpolation, phoneme edges aligned) and vocoded with the real f0 (`slowmo=1.4`) | 0.523 [0.48-0.57] | 0.405 [0.40-0.41] | GAIN -0.12 / -0.11, every seed |
| b_slow20 | slow-motion x2.0 | 0.557 [0.52-0.62] | 0.356 [0.34-0.36] | GAIN -0.09 / -0.16; not better than x1.4 on the tiny+base mean (0.457 vs 0.464) and costs 2x the acoustic time |

Tone check for the slow-motion clip (`python3 -I $S/tools/voiceq.py`, verse1 seed 1): spike cpp 20.80,
hnr 18.3, lvl_sd 6.72, dip -10.9; slow-motion x1.4 cpp 21.35, hnr 17.8, lvl_sd 6.17, dip -8.3; the
same verse really sung at 120 BPM cpp 21.34, hnr 19.4, lvl_sd 6.30, dip -9.5. No tone penalty.
| c_slow_v2 | slow-motion on vowels only (x2), consonants unstretched | 0.553 [0.52-0.61] | 0.436 [0.41-0.46] | about two thirds of the uniform gain: the clearer vowels are most of it, stretching consonants too adds the rest |
| d_final | per-note slow motion (module `dsfinal.py`): a note shorter than 0.25 s is stretched to 0.25 s, at most x2 (8ths at 168 BPM x1.4, 16ths x2), silence unstretched | 0.504 [0.41-0.60] | 0.349 [0.33-0.36] | GAIN -0.14 / -0.17 (small.en 0.432 -> 0.246), every seed better than its baseline seed on all three recognisers |

### What slow-motion rendering is

At 168 BPM the bank gets 15 frames per syllable and 3-8 per phoneme; it then undershoots
vowel targets and smears consonants. The same lyrics at 120 BPM score 0.375 / 0.347. So the
phrase is sung slower than written and the result is sped up again, in the mel domain where
that is free of artefacts:

1. alignment as before gives the phoneme lengths `durs` on the real 512-hop grid;
2. every phoneme of a note shorter than `SLOW_NOTE` (0.25 s) gets the factor
   `k = clip(SLOW_NOTE / note_length, 1, SLOW_MAX)`; SP/AP get 1; onset consonants take the
   factor of the note they lead into;
3. `slow_grid` builds the stretched lengths and a piecewise-linear frame map whose knots are the
   phoneme edges; the pitch curve is resampled through it;
4. the variance model and the acoustic model run on the stretched grid;
5. the mel is resampled back to the real grid (linear interpolation of log-mel rows) and the
   vocoder runs with the real f0, so timing and pitch are exactly as written.

Phrases without fast notes take the old path bit for bit (same session, same inputs).
Cost: acoustic + variance time grows with the stretched frame count (verse1: x1.5).

Checks on the slow-motion clip (verse1 seed 1, `out/verse1_before_spike.wav` vs
`out/verse1_after_slowmotion.wav`): same length, RMS-envelope correlation 0.958 with best lag
0 ms (timing unchanged), log-spectral flux 0.58 -> 0.61 (faster spectral movement), energy above
5 kHz -23.2 -> -22.0 dB.

Where the gain is (`python3 -I $S/q1/loc.py base0 d_final`, tiny+base miss rate):

| words | spike | slow motion |
|---|---|---|
| all | 0.56 | 0.40 |
| on 16th notes (n=14) | 0.71 | 0.39 |
| on 8th notes (n=68) | 0.50 | 0.36 |
| quarter or longer (n=6) | 0.86 | 0.83 |
| E5 or higher (n=16) | 0.73 | 0.53 |
| function words | 0.38 | 0.17 |
| content words | 0.66 | 0.53 |

Still missed by 8-9 of 9 (3 seeds x 3 recognisers) after the change: valued, sent ("send it"),
spam (F5), unread ("and read"), knocked ("not politely"), accept all x2 ("exit door": the
released `p t` runs into the vowel of "all" = "tall/tor"), still, trained ("trying to"),
up (A5, twice). These are high notes, coda clusters released into a vowel-initial word, and
readings a listener without the lyric sheet would also make.

### No-regression by construction on slow material

`python3 -I $S/q1/mock_eq.py CASE dsinfer_spike dsfinal` replaces the models with a recorder and
compares every input handed to the variance model, the acoustic model and the vocoder:
twinkle, plain, neon and grace are IDENTICAL (no note under 0.25 s, so the old path runs);
verse1, verse2 and chorus1 differ. With the same seed the first four therefore render bit for
bit as before. The same tool shows `dsfinal.py` = `dsq.py` with `opt={"slow_note": 0.25}`, so
the `d_*` screening runs below compare against `d_final` seed for seed.

### Tone numbers next to the words (`python3 -I $S/tools/voiceq.py $S/q1/runs/base0` and `.../d_final`, verse1, mean of seeds 1-3)

| variant | cpp | cpp_v | hnr | voiced | hf | lvl_sd | dip | run ms |
|---|---|---|---|---|---|---|---|---|
| spike | 20.95 | 21.51 | 18.20 | 0.87 | -30.06 | 6.54 | -10.32 | 183 |
| slow motion (`dsfinal.py`) | 21.45 | 21.96 | 18.00 | 0.88 | -28.98 | 6.15 | -8.41 | 180 |

Clearer phonation by 0.5 dB cpp, same hnr, 1 dB more energy above 5 kHz, steadier level
(lvl_sd -0.4 dB, dips 1.9 dB shallower). The word gain is not paid for in tone.

## 3. Held-out and regression checks (seeds 1-3, tiny / base / small; `spec_e.py`)

| case | spike (`dsinfer_spike`) | slow motion (`dsfinal`) |
|---|---|---|
| verse1 (168 BPM, 88 words) | 0.644 / 0.519 / 0.432 | 0.504 / 0.349 / 0.246 |
| verse2, held out (AI track bars 53-68, 42 words) | 0.516 [0.38-0.69] / 0.397 [0.31-0.48] / 0.246 [0.14-0.31] | 0.349 [0.29-0.38] / 0.278 [0.19-0.36] / 0.182 [0.14-0.21] |
| chorus1 (35 words, mixed quarters and 8ths) | 0.724 [0.60-0.80] / 0.381 [0.29-0.43] / 0.124 [0.09-0.14] | 0.838 [0.49-1.46] / 0.571 [0.34-1.00] / 0.143 [0.09-0.17] |

chorus1 needs care. Two of the nine slow-motion transcripts are Whisper repetition loops
("Hohohoho...", "All the hoo-hoo-hoo..."), which score 1.00 and 1.46. Without those two the
slow-motion numbers are tiny 0.49 / 0.57 and base 0.37 / 0.34, i.e. better than the spike on
tiny and equal on base. The chorus opens with a sung "Ah, all in-" that every recogniser
hears as "ha / ho(ney)"; whether the loops are chance or caused by the change is tested with
five more seeds per variant (`f_base_ch`, `f_final_ch`). Note also that the seeded baseline for
chorus1 (0.72 / 0.38 / 0.12) is far from the single spike run quoted in the team brief
(0.29 / 0.26 / 0.06): that run was a lucky draw, tiny.en ranges 0.60-0.80 over three seeds.

## 4. Other hypotheses on top of slow motion (verse1, tiny / base; reference `d_final` 0.504 / 0.349)

| run | change | tiny | base | verdict |
|---|---|---|---|---|
| d_conv (3 seeds) | bank conventions: `t r` -> `ch _r`, `d r` -> `jh _r`, flap `dx`, and OpenUtau's rule that a glide after a consonant starts the note | 0.492 [0.42-0.59] | 0.428 [0.40-0.48] | no gain; base.en worse on every seed (+0.08). Not adopted |
