# F2 notes: the song and the vocal production (kept current)

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad, this folder = $S/prod

## How to continue from here

- Files: `chain4.py` (the chain, the double, the small room), `ds_backend4.py` (backend: engine choice, chain on the
  lead / harmony, double made from the lead take, takes cached in `cache/`), `build4.py` (the song build:
  `PROD_VERSE=relaxed`, chain channels, reverb, sends), `run4.py OUT A B [--verse relaxed] [--engine demo3|flow]
  [--chain off] [--double take]`, `final_mix.sh ENGINE` (the three mixes of the clips), `dry4.py NAME A B` (the lead
  alone, dry and through the chain, measured), `make_clips.py [--engine=flow]` (prod/out), `relaxed_verse.s2s`.
- Measuring: `m.py` (flow + voiceq + WER of any wav; every row of every tool is appended to `runs.jsonl`), `lab.py`
  (chain variants on dry takes, `variants.json`), `fxlab.py` (reverb / delay settings on a processed lead), `vsum.py`
  (flow numbers of a sum of stems).
- Run everything as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/prod/...`.
- Engines: `demo3` = the second demo's engine and options; `flow` = the colleague's `$S/flow/dsinfer.py` with the
  lead options of `$S/flow/config.json` (exists since 23:49). Mix folders `mix/ENGINE`, `mix/ENGINE_relaxed`; dry
  takes `dry/ENGINE/`. Clips of the flow engine carry the suffix `_flow`.

## Status log (newest last)

- 23:24 start. 23:33 baseline measured. 23:37 chain, backend, build, relaxed verse written. 23:41 first chorus mix.
- 23:47 FOUND: a plain compressor chain lifts the breaths of the relaxed verse from -27..-34 dB under the sung level
  to -6..-8 dB (gasps). Fixed in chain4 (periodicity detector, noise frames follow the sung level around them).
- 00:02 FIRST CLIPS in out/ (demo-3 engine, new chain): chorus mix, both verse mixes, dry and processed leads.
- 00:05-00:11 chain settled on three takes (WER, stage ablation, EQ bands); reverb settled with fxlab.py.
- 00:16 CLIPS RE-EXPORTED with the final chain and the short room (demo-3 engine): the ten clips of section "Clips".
- 00:17 the colleague's config.json was rewritten at 00:12 (final candidate): `final_mix.sh flow` + dry4 running.

## 1. The relaxed verse (`relaxed_verse.s2s`, used with `--verse relaxed` / `PROD_VERSE=relaxed`)

Written key A minor, chords (Am Am/G F E) x4 as in SPEC.md; the build moves it 5 semitones down with the song.

    Three in the morning, saucer on the lawn
    Eight billion users: "what is going on?"           (HUMANS: Help!)
    Mothership: a message from the sky
    Filed under spam, unread. Goodbye.
    They knocked. I made a cookie wall:
    "Reject all" is still accept all                   (ALIENS: Cookies? Delicious.)
    Ninety thousand pop-ups in their face
    Meant for emails, not to save your race

- 71 syllables in 19.3 s of sung time = 3.7 syllables a second (written verse: 114 notes in 20.7 s = 5.5, 16 of
  them sixteenths, 10 a quarter or longer). Nothing shorter than an eighth note (179 ms); 29 of the 71 notes are a
  quarter or longer. The line ends are held on open vowels
  (lawn, on, sky, bye, wall, all, face, race), the joke words keep their jumps ("spam" and "e-mails" on the top note).
- Seven written breaths (`br`, a quarter note = 357 ms) before the lines; lines 5 and 6 run on in one breath so that
  the fine print lands before the aliens' "Cookies? Delicious." (bar 20, beat 3). The crowd's "Help!" and the
  aliens' answer keep their places.
- Three wordings were sung. 1: "Three a.m., a saucer on the lawn / Valued users ... / They knocked. I built a ... /
  Trained for emails": the recognisers lost "a.m." (two bare vowels), "built a", "trained for". 2: "... in the
  morning", "made", "trained on": "valued users" and "trained on" still lost. 3 (kept): "Eight billion users",
  "Meant for emails". Still missed by base.en in wording 3: "knocked. I" ("not come"), "wall" ("war"), "meant"
  ("vans": the m after the breath), "unread" ("and red").
- 3.7 is over the 3 to 3.5 asked for; the next cut would be line 1 or 2 down to eight syllables (not done: both
  carry the scene).

Same render path for both (`dry4.py`, demo-3 engine, seed of the track, lead alone, dry):

| verse lead, dry | holes/s | hole % | voiced run ms | breaks/s | lvl_sd | hnr | WER tiny / base / small |
|---|---|---|---|---|---|---|---|
| written (5.5 syll/s) | 3.27 | 28.5 | 160 | 3.22 | 8.58 | 17.27 | 0.50 / 0.47 / 0.22 |
| relaxed (3.7 syll/s) | 1.77 | 16.3 | 305 | 1.90 | 7.82 | 19.92 | 0.36 / 0.23 / 0.18 |
| target (FLOW_COMMON) | <= 1.8 | <= 15 | >= 250 | | <= 7 | | <= 0.42 / 0.27 / 0.24 |

Through the chain (the processed lead): written 2.99 / 19.8 % / 150 ms / lvl_sd 6.93 / WER 0.50 / 0.32 / 0.24;
relaxed 1.54 / 9.8 % / 330 ms / lvl_sd 5.68 / WER 0.39 / 0.21 / 0.27. WER is against each version's own text (85
and 48 words: one missed word weighs 1.2 % and 2.1 %). The relaxed verse meets holes/s and the voiced run dry and
all four flow targets through the chain; the written verse meets none of them with either. Breaths in the relaxed
dry take: -27 to -34 dB under the sung level (median of each breath; the bank's sample songs: -21 to -28).

## 2. The vocal chain (`chain4.py`, run by the backend on the lead channels, the double and the harmony)

What the demo-3 chain did to the lead (the renderer's strip + lead EQ + 6.5 kHz shelf; dry take against its stem):
verse lvl_sd 8.50 -> 8.21, chorus 7.35 -> 6.82 (its compressor took up to 7.7 / 6.3 dB off the peaks and left the
spread of the level where it was), energy
above 5 kHz +5.5 dB (hf -31.4 -> -25.8), noise bursts 0.31 -> 0.77 /s in the verse, hnr -1.2 dB.

The new chain, in order (settings `chain4.LEAD`):

1. eq: 110 Hz low cut (24 dB/oct), +1.5 dB bell at 3.2 kHz (Q 0.8), -2 dB shelf from 10 kHz. No air: above 9 kHz
   the bank is aspiration noise.
2. deess: dynamic cut at 6.8 kHz, up to 6 dB (active 12-20 % of the time, mean 0.3-0.5 dB).
3. rider: the sung level rides toward the take's median, +-8 dB, 120 ms window. In a gap it holds the value of the
   sung stretch around it.
4. comp1, the leveller: 2:1 from 26 dB under the loud level, 4 ms attack, 20 ms release (mean 10-11 dB, max 15).
5. comp2: 3:1 on the top 6 dB, 2 ms / 80 ms (mean 2.5-3 dB, max 6).
6. cons, a consonant leveller: noise frames between -24 and 0 dB re the sung level move halfway toward -8 dB
   (+4 dB at most, -9 dB at most); breaths, which sit under that range, are not touched.
7. crest: transient limiter 11 dB over the loud level (0-1 dB), then one static make-up gain to the input level.

The detector that makes it safe: "sung" = periodic (autocorrelation of the signal under 1.5 kHz, 48 ms frames; on
a relaxed-verse take 95 % of the breath frames are under 0.47 and 94 % of the other active frames over it). Rider and
compressors read the sung level only; consonants, breaths and gaps get the gain of the sung frames around them.
Without this the same compressors lifted the breaths from -27..-34 dB to -6..-8 dB under the voice (first version,
23:47); with it they come out 2 to 5 dB up (-22..-30 dB).

Stage by stage (`lab.py +eq +deess +rider +comp1 +comp2 +cons full`; holes/s, hole %, lvl_sd, hnr):

| up to | written verse (harness take) | chorus (harness take) | relaxed verse |
|---|---|---|---|
| dry | 3.14, 27.3 %, 8.50, 17.57 | 1.42, 13.7 %, 7.35, 21.16 | 1.77, 16.3 %, 7.82, 19.92 |
| eq (earlier: with a 480 Hz cut and +2.5 dB bell) | 3.19, 27.2 %, 8.29, 16.87 | 1.42, 13.8 %, 7.18, 20.05 | 1.77, 15.8 %, 7.52, 18.53 |
| deess | 3.19, 27.3 %, 8.47, 16.84 | 1.42, 13.8 %, 7.26, 20.05 | 1.77, 16.1 %, 7.70, 18.56 |
| rider | 3.14, 25.9 %, 8.01, 16.82 | 1.38, 12.5 %, 6.55, 20.08 | 1.77, 13.9 %, 6.88, 18.55 |
| comp1 | 2.87, 21.1 %, 7.04, 16.69 | 1.37, 9.9 %, 5.67, 20.10 | 1.49, 10.3 %, 5.91, 18.49 |
| comp2 | 2.68, 19.9 %, 6.86, 16.62 | 1.27, 8.6 %, 5.58, 20.13 | 1.58, 9.2 %, 5.74, 18.50 |
| cons = full chain | 2.73, 19.4 %, 6.74, 16.64 | 1.22, 8.2 %, 5.39, 20.12 | 1.63, 10.5 %, 5.66, 18.51 |
| full chain, final EQ (dry4.py takes) | 2.99, 19.8 %, 6.93, 16.76 (dry 17.27) | 1.21, 9.2 %, 5.37, 20.91 (dry 21.33) | 1.54, 9.8 %, 5.68, 19.41 (dry 19.92) |

- lvl_sd: target 4 to 5. Reached: chorus 5.4, relaxed verse 5.7, written verse 6.9 (in the mix stems 5.25 / 5.65 /
  6.85). A 3:1 leveller from -30 dB (variant H2c) gives 5.04 / 5.31 / 6.36 with the same words on three takes; not
  used, it is 50 % more ratio for 0.35 dB. Under 5 needs the consonants and note edges lifted, which is the thing
  that turned the breaths into gasps.
- hnr: the dynamics cost nothing (low cut alone + all dynamics: 21.24 / 19.89 against dry 21.16 / 19.92). The bell
  costs 0.3 dB per dB (+1.5 dB: -0.4 to -0.5), the 480 Hz cut cost 0.45 and was dropped, the shelf is free. So
  the processed lead is 0.4-0.5 dB under dry in hnr: the price of the presence bell.
- WER, three takes (verse harness take, chorus harness take, relaxed verse), dry -> chain: tiny.en 0.40 -> 0.43,
  0.29 -> 0.34, 0.34 -> 0.39; base.en 0.33 -> 0.27, 0.14 -> 0.23, 0.23 -> 0.21; small.en 0.27 -> 0.24, 0.06 -> 0.00,
  0.18 -> 0.27. Means 0.343 / 0.233 / 0.170 -> 0.387 / 0.237 / 0.170. On the dry4 takes: written verse 0.50 / 0.47 /
  0.22 -> 0.50 / 0.32 / 0.24, chorus 0.40 / 0.31 / 0.11 -> 0.57 / 0.43 / 0.11. So: verses the same or better, the
  chorus loses 2 to 4 of its 35 words for tiny.en and base.en on both takes (the hook: "select all invaders" is
  "select calling invaders" dry and "select Tolin Vaders" processed; small.en reads it either way). No single
  stage owns that loss (nocons 0.40, no leveller 0.34, 45 ms release 0.40, no rider 0.37 against 0.37 full, base.en).
- Pumping (`gr_check.py`): over sung frames the gain spans -20 dB (loudest) to +1 dB (softest), median -15; it moves
  1.8-4.5 dB per 100 ms at the median and 14-15 dB at the 95th percentile, with the syllables. Frames where the
  input level is steady (under 2 dB per 100 ms) and the gain still moves more than 3 dB: 0.7 % (chorus), 1.3 %
  (relaxed), 2.6 % (written verse). The gain on quiet noise (breaths, tails) is 1.6 to 5.4 dB over the sung median.
- True peak of the three mixes: -1.02 dBTP (mp3s -1.3); limiter max 1.6 dB chorus, 3.4 / 2.8 dB verses.

Space (`fxlab.py`: processed lead + the mixer's own reverb and delay returns, measured as one clip):

| lead + returns | chorus holes/s, hole %, run, breaks/s | relaxed verse | WER chorus | WER relaxed |
|---|---|---|---|---|
| no effects | 1.21, 8.8 %, 480, 1.52 | 1.63, 10.5 %, 320, 1.81 | 0.63 / 0.34 / 0.11 | 0.39 / 0.21 / 0.27 |
| demo 3: hall 1.7 s, send 0.22 | 1.11, 6.8 %, 415, 1.64 | 1.40, 8.4 %, 370, 1.84 | 0.46 / 0.26 / 0.11 | 0.45 / 0.25 / 0.20 |
| plate 0.95 s, send 0.35 | 1.02, 6.5 %, 640, 1.11 | 1.36, 7.6 %, 410, 1.62 | 0.66 / 0.43 / 0.17 | 0.45 / 0.25 / 0.29 |
| plate 0.95 s, send 0.50 | 0.89, 4.8 %, 860, 0.89 | 0.92, 4.8 %, 520, 1.32 | 0.66 / 0.37 / 0.17 | 0.45 / 0.30 / 0.41 |
| room 0.6 s, send 0.35 | 1.03, 6.6 %, 360, 1.43 | 1.27, 7.7 %, 355, 1.89 | | |
| **room 0.6 s, send 0.50 (used)** | 0.85, 5.1 %, 620, 1.16 | 1.10, 5.7 %, 590, 1.32 | 0.66 / 0.34 / 0.17 | 0.46 / 0.23 / 0.30 |
| room 0.6 s, send 0.70 | 0.71, 3.3 %, 660, 1.07 | 0.57, 3.2 %, 700, 1.05 | 0.57 / 0.34 / 0.17 | 0.43 / 0.25 / 0.36 |
| plate 0.35 + a 180 ms mono room in the channel | 0.85, 4.6 %, 460, 1.34 | 0.88, 4.6 %, 390, 1.67 | 0.89 / 0.43 / 0.11 | 0.48 / 0.34 / 0.29 |

- Used: the song's reverb shortened to RT60 0.6 s (`reverb_room` 0.10) with the lead's send at 0.50 (double 0.40,
  harmony 0.50). With every setting the lines join into one phrase (median phrase 22 s against 2.6 s dry in the
  chorus). The short room at 0.50 fills as much as the plate at 0.50 and costs fewer words (relaxed small.en 0.30
  against 0.41); under 0.50 the fill is no better than demo 3's. Words with the room against no effects: base.en
  equal, small.en +0.03 to +0.06, tiny.en +0.03 to +0.07 (tiny.en swings 0.46-0.66 on this chorus take, so read
  base and small).
- The delay is the song's (dotted eighth, ping-pong, 30 % feedback, ducked 8 dB while the voice sings, so it only
  blooms in the gaps). A quarter-note version measured the same; not changed.
- The small mono room inside the channel (chain4.room) is not used: tiny.en 0.89 on the chorus.
- `reverb_room` is a project setting: the band's reverb sends get the shorter room too (not judged separately).

## 3. Doubles and harmony

- The double is no longer a second take. `ds_backend4` takes the lead's own take over the double's notes, runs the
  chain with the consonant leveller turned into a consonant duck (target -18 dB) and a darker top, then
  `chain4.adt`: 18 ms +- 3 ms delay (0.31 Hz), +6 cents, -5 dB shelf from 4.5 kHz. `--double take` restores demo 3.
- The harmony keeps its own take (it sings other notes) with the same consonant duck and a -4 dB shelf from 5 kHz.
- Summed vocal, chorus stems (`vsum.py`; holes/s, hole %, voiced run, breaks/s, lvl_sd):

| | lead | lead + double | lead + double + harmony |
|---|---|---|---|
| demo 3 (second take, old chain) | 1.47, 14.4 %, 480, 1.47, 6.82 | 1.42, 13.3 %, 375, 1.53, 6.56 | 1.47, 13.4 %, 380, 1.47, 6.35 |
| now (copy of the lead take, new chain) | 1.26, 8.7 %, 500, 1.42, 5.25 | 1.22, 8.0 %, 490, 1.27, 5.25 | 1.22, 7.7 %, 540, 1.32, 5.34 |

  With the second take the voiced run fell from 480 to 375 ms when the double came in (its consonants cut the
  lead's vowels); with the copy it stays (500 -> 490) and the voice stops less often (1.42 -> 1.27 /s).
