# clar: word clarity back into the lead without losing the line

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad, this folder = $S/clar.
Run everything as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/clar/...`, one render
or recognition process at a time. Nobody has listened to any of this: every statement below is a measurement.
No repo file and no other scratch folder was changed.

## How to continue from here

- Settings: `config.json` "dial" = A (the fourth demo), B (balanced), C (crisp). Each has `engine` (options of
  `dsinfer.py` render_track_ds merged over the fourth demo's lead options), `chain` (stages of `chain5.py` merged over
  `chain5.LEAD4` = the fourth demo's chain) and `mix` (the lead channels' reverb send).
- Render any bars with a setting: `run6.py OUT A B --dial B` (the fourth demo's arrangement and mix, only the lead's
  engine options, chain and send change; the harmony and the lead's answers sing with the same consonant options;
  the doubles are made from the lead's take as before). For bars 9-24 add `--set '{"loudness_target": -13.0}'` (the
  fourth demo's verse mix was rendered so). NOT the whole song in one go (13.5 GB in the mix).
- The lead alone in the song: `lead6.py NAME A B --dial B --wer CASE` (clar/lead/NAME_B_chain.wav = the channel as it
  goes to the mix bus, NAME_B_room.wav = with its own reverb and delay returns, NAME_B.dump.json = phoneme times).
- Harness takes (verse1 / chorus1 / verse2 cut out of the AI track): `hc.py TAG --seeds=1,2,3` (sets.py names the
  option sets), measured by `lab.py SET` (rows.jsonl) and printed by `table.py PREFIX ...`.
- Measures: `cm.py` (consonants by class against phoneme times; `cm.py ref` = the bank's sample songs), `stack.py DIAL`
  (lead stem, + doubles, + harmony, all vocals, full mix), `mask.py DIAL` (lead consonant noise against everything else
  above 4 kHz). Clips: `mkclips6.py`.
- Engine options added in clar/dsinfer.py (all off by default; with them off a take is the fourth demo's):
  `cons_cap_share`, `slow_cons_min` + `squeeze` ("burst" / "loud" / "peak"), `slow_cons`, `cons_gain`.
  Chain stages added in clar/chain5.py: `lift` (detector "tilt"), `trans`; `dial_chain`.
- Takes are cached in clar/cache (read from clar/cache, mix5/cache, sep/cache). The cache key still carries the hash
  of sep/'s engine files, so every other channel's take is reused.

## Result in short

1. Where the words went (three seeds, tiny.en / base.en WER, means over verse1, chorus1, verse2):
   second demo's lead dry 0.34 / 0.21 -> fourth demo's lead dry 0.48 / 0.19 -> through the chain 0.45 / 0.25 -> with
   its room (send 0.50) and the delay 0.67 / 0.39. In the song base.en answered the fourth demo's verse lead with its
   room once with "[Music]" (WER 1.00) and once with 0.61 on the same audio.
2. What the measures say, in the order of size (stop = p t k, closure = quietest 10 ms re the median vowel):
   - the consonant caps: stops 52 -> 35 ms in the verse, 116 -> 58 ms in the chorus (the reference: 116 ms),
     sibilants 81 -> 46 and 116 -> 70 ms with 1.5-4 dB less noise (5-7 with the caps alone), stop closures 8-10 dB shallower, release rise 15-21
     -> 4-8 dB. Dry they cost the recognisers little (0.39 / 0.27 against 0.38 / 0.21, seed 1); under the
     same chain and room the fourth demo's take reads 0.58 / 0.37 where the second demo's take, with its long
     consonants, reads 0.38 / 0.27 (seed 1): short consonants do not survive the room.
   - the room: send 0.50 fills every closure to -8 (verse) / -12 dB (chorus) re the vowels; an uncapped 116 ms
     closure that is -40 dB through the chain sits at -17 dB with it. It is the largest single cost for the recognisers (+0.22 / +0.14).
   - the chain: the 2:1 leveller and the rider take the stop closure from -22 to -12 dB (capped stops), lvl_sd from
     6.7 to 4.3 (reference 5.9-6.7), the modulation depth 2-20 Hz from -6.5 to -10.8 dB (reference -6.6); voiced
     stops end 4 dB under the vowels, voiced fricatives at the vowels' level.
   - the voiced-consonant floor: voiced stops -21 -> -8 dB re the vowels (closure -34 -> -11), but the recognisers
     read it the same or better (0.31 / 0.21 against 0.38 / 0.21) and without it 0.6 holes/s come back. Kept.
   - the breaths: nothing lost (0.37 / 0.18).
   - the stack (chorus only): doubles and harmony fill the lead's closures by 4 dB (A: -16 -> -12) to 12 dB
     (B: -35 -> -23, C: -36 -> -25); in the full mix the band fills every closure (-5 to -7 dB in every setting).
3. What bought the flow: the caps (verse voiced run 160 -> 310 ms, hole time 27 -> 20 %), the floor (holes/s 3.1 ->
   2.5, chorus 1.4 -> 0.9), the leveller (hole time 15 -> 11 %) and the room (holes/s 1.7 -> 0.55). Chain plus room
   overshoot the reference: the fourth demo's lead measures 0.4 holes/s and 2-4.5 % hole time where the reference
   has 1.05-1.30 and 8-11 %.
4. The dial (clips in out/): B and C give the consonants their length back where there is room for it (caps bind
   only where the consonants take over 35 / 40 % of a note), lengthen the caps (stops 45 / 50 ms, fricatives
   65 / 80 ms), level less (B 1.25:1, C no leveller), halve the lead's reverb send (0.25 / 0.15) and lift the
   sibilants' noise by 2 / 2.5 dB. C also cuts every stop around its release (deeper closures) and sharpens releases.
   Numbers against the reference in "The dial" below.
   Three seeds on the harness cases, tiny.en / base.en, means over verse1 / chorus1 / verse2, lead with its returns:
   A 0.67 / 0.39, B 0.42 / 0.16, C 0.35 / 0.20 (the second demo's dry lead, the target: 0.34 / 0.21). Through the chain
   only: A 0.45 / 0.25, B 0.37 / 0.15, C 0.30 / 0.17.
5. Not reached: tiny.en stays over the second demo's in the chorus (B 0.47, C 0.42 with returns against 0.31; base.en
   is under it: 0.06 / 0.23 against 0.19) and in B's verse 1 (0.47 against 0.42); the verse cannot be both at the
   reference's hole time and at its consonant lengths at 5.4 syllables a second (B 13-14 %, C 18-19 % with returns);
   chorus stop bursts in B / C measure 9-10 dB over the reference's in the song take. See "Still wrong".

## The measures (cm.py)

Levels are RMS in 10 ms windows every 5 ms, dB re the clip's median vowel level. Per consonant class: length; `med`
(median level of the consonant); `hf` (its loudest 10 ms above 4 kHz); `dip` (its quietest 10 ms: the closure); for
released stops (a vowel follows) `burst` (loudest 10 ms above 4 kHz up to 15 ms after the stop) and `rise` (largest
10 ms rise of the level above 2 kHz). Modulation: rms depth of the level envelope inside phrases, 2-20 Hz, dB.
The reference (three sample songs of the bank, Mandarin, 2.3-2.7 vowels/s, consonants 22-25 % of the sung time):

| class | n | length ms | med | hf | dip | burst | rise |
|---|---|---|---|---|---|---|---|
| aspirated stops p t k | 9 | 116 | -22.2 | -25.1 | -37.6 | -23.7 | 12.5 |
| unaspirated stops and affricates b d g z zh j | 63 | 81 | -11.7 | -19.3 | -20.1 | -16.3 | 10.6 |
| aspirated affricates c ch q | 17 | 128 | -12.4 | -11.3 | -23.0 | -11.1 | 6.7 |
| sibilants s sh x | 33 | 116 | -8.1 | -8.9 | -15.3 | | |
| f h | 19 | 81 | -14.7 | -22.7 | -18.7 | | |
| nasals | 21 | 81 | -3.6 | | -4.6 | | |
| glides | 48 | 81 | -2.8 | | -3.9 | | |

flow.py on the reference: 1.05-1.30 holes/s, 8-11 % hole time, voiced run 375-470 ms, lvl_sd 5.9-6.7; modulation
2-20 Hz -5.8 to -7.7 dB (pooled -6.6). Only nine aspirated stops: their numbers are a rough mark, not a target to
the dB. The verse sings 5.4 vowels/s, the chorus about 2.5.

## Step 1: where the words went, one switch at a time

Harness takes (the AI track's bars cut out, 5 semitones lower), dry unless a row says otherwise. "demo2" = the
second demo's lead settings, "flow" = the fourth demo's (demo2 + caps + voiced-consonant floor + breaths).
`+x` = added to demo2, `-x` = left out of flow, `flow +stage` = the fourth demo's take through the chain up to and
including that stage, `flow chain` = the whole chain, `flow chain+room` = with the mixer's reverb (RT60 0.6 s, send
0.50) and delay returns. Seed 1 except where the last column says 3; WER = tiny.en / base.en, loops collapsed.
Columns: flow.py (holes/s, hole %, voiced run, lvl_sd), modulation 2-20 Hz, unvoiced stops (length, closure, burst,
rise), sibilants (length, noise above 4 kHz), voiced stops (closure), voiced fricatives (level).

| row | case | holes/s | hole % | run ms | lvl_sd | mod 2-20 | stop ms | stop dip | burst | rise | sib ms | sib hf | vstop dip | vfric | WER tiny / base | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| demo2 dry | verse1 | 3.14 | 27.3 | 160 | 8.50 | -3.8 | 52 | -30.9 | -25.7 | 15.3 | 81 | -10.0 | -34.0 | -4.8 | 0.40 / 0.33 | 1 |
| demo2 dry | chorus1 | 1.42 | 13.7 | 450 | 7.35 | -7.8 | 116 | -40.9 | -18.4 | 20.8 | 116 | -10.9 | -23.5 | -20.9 | 0.29 / 0.14 | 1 |
| demo2 dry | verse2 | 3.01 | 24.0 | 170 | 9.38 | -4.7 | 46 | -24.7 | -21.7 | 20.4 | 70 | -11.1 | -22.2 | -14.0 | 0.45 / 0.14 | 1 |
| demo2 dry | MEAN | 2.52 | 21.7 | 260 | 8.41 | -5.4 | 72 | -32.2 | -21.9 | 18.8 | 89 | -10.7 | -26.6 | -13.2 | 0.38 / 0.21 |  |
| flow dry | verse1 | 2.29 | 15.2 | 330 | 7.38 | -4.7 | 35 | -22.4 | -29.2 | 8.4 | 46 | -13.5 | -11.6 | -6.2 | 0.40 / 0.29 | 1 |
| flow dry | chorus1 | 0.82 | 5.4 | 1040 | 6.03 | -8.8 | 58 | -30.8 | -27.4 | 4.1 | 70 | -15.0 | -14.3 | -7.4 | 0.63 / 0.11 | 1 |
| flow dry | verse2 | 1.65 | 9.6 | 290 | 6.56 | -6.2 | 35 | -17.4 | -21.5 | 5.7 | 46 | -12.6 | -5.4 | -7.6 | 0.26 / 0.12 | 1 |
| flow dry | MEAN | 1.59 | 10.1 | 553 | 6.66 | -6.5 | 43 | -23.5 | -26.0 | 6.1 | 54 | -13.7 | -10.4 | -7.1 | 0.43 / 0.18 |  |
| demo2 +caps | verse1 | 3.05 | 19.7 | 310 | 8.42 | -5.1 | 35 | -24.0 | -28.9 | 6.7 | 46 | -15.0 | -15.0 | -6.9 | 0.47 / 0.48 | 1 |
| demo2 +caps | chorus1 | 1.36 | 8.4 | 670 | 6.93 | -9.1 | 58 | -30.4 | -26.0 | 11.3 | 70 | -18.0 | -21.0 | -19.6 | 0.40 / 0.14 | 1 |
| demo2 +caps | verse2 | 2.72 | 16.6 | 310 | 8.36 | -6.1 | 35 | -19.1 | -19.9 | 4.6 | 46 | -13.9 | -5.3 | -15.5 | 0.31 / 0.19 | 1 |
| demo2 +caps | MEAN | 2.38 | 14.9 | 430 | 7.90 | -6.7 | 43 | -24.5 | -24.9 | 7.5 | 54 | -15.6 | -13.8 | -14.0 | 0.39 / 0.27 |  |
| demo2 +cons floor | verse1 | 2.51 | 20.0 | 260 | 7.86 | -4.4 | 52 | -30.3 | -25.4 | 15.1 | 81 | -9.7 | -10.6 | -4.5 | 0.39 / 0.33 | 1 |
| demo2 +cons floor | chorus1 | 0.89 | 9.6 | 665 | 6.47 | -8.3 | 116 | -41.0 | -18.5 | 21.1 | 116 | -11.0 | -16.9 | -6.6 | 0.31 / 0.17 | 1 |
| demo2 +cons floor | verse2 | 1.75 | 12.6 | 375 | 7.35 | -5.7 | 46 | -24.1 | -21.7 | 20.0 | 70 | -11.1 | -11.9 | -6.4 | 0.24 / 0.14 | 1 |
| demo2 +cons floor | MEAN | 1.72 | 14.0 | 433 | 7.23 | -6.1 | 72 | -31.8 | -21.9 | 18.7 | 89 | -10.6 | -13.1 | -5.8 | 0.31 / 0.21 |  |
| demo2 +breaths | verse1 | 3.11 | 26.7 | 160 | 8.43 | -3.3 | 46 | -28.7 | -23.7 | 15.8 | 81 | -8.9 | -27.3 | -4.7 | 0.50 / 0.28 | 1 |
| demo2 +breaths | chorus1 | 1.19 | 11.8 | 530 | 7.69 | -6.8 | 116 | -44.1 | -20.9 | 22.3 | 116 | -9.8 | -31.6 | -21.0 | 0.26 / 0.17 | 1 |
| demo2 +breaths | verse2 | 2.73 | 22.3 | 170 | 8.79 | -4.1 | 46 | -22.0 | -20.9 | 15.1 | 70 | -10.0 | -15.3 | -14.9 | 0.36 / 0.10 | 1 |
| demo2 +breaths | MEAN | 2.34 | 20.3 | 287 | 8.30 | -4.7 | 70 | -31.6 | -21.8 | 17.7 | 89 | -9.6 | -24.7 | -13.5 | 0.37 / 0.18 |  |
| flow -caps | verse1 | 2.34 | 19.6 | 175 | 7.63 | -3.8 | 46 | -27.8 | -23.0 | 16.5 | 81 | -8.6 | -10.7 | -4.0 | 0.34 / 0.26 | 1 |
| flow -caps | chorus1 | 0.82 | 8.1 | 675 | 6.49 | -7.8 | 116 | -44.4 | -20.8 | 22.0 | 116 | -9.9 | -15.7 | -7.5 | 0.31 / 0.14 | 1 |
| flow -caps | verse2 | 1.75 | 12.4 | 320 | 7.13 | -5.0 | 46 | -22.2 | -20.8 | 14.5 | 70 | -9.3 | -12.2 | -6.2 | 0.38 / 0.17 | 1 |
| flow -caps | MEAN | 1.64 | 13.4 | 390 | 7.08 | -5.5 | 70 | -31.5 | -21.5 | 17.7 | 89 | -9.3 | -12.9 | -5.9 | 0.35 / 0.19 |  |
| flow -cons floor | verse1 | 2.94 | 19.8 | 290 | 8.13 | -4.3 | 35 | -23.0 | -29.2 | 7.9 | 46 | -13.8 | -12.8 | -6.6 | 0.41 / 0.27 | 1 |
| flow -cons floor | chorus1 | 1.19 | 7.5 | 720 | 6.71 | -8.2 | 58 | -30.8 | -27.3 | 4.1 | 70 | -15.0 | -18.9 | -18.3 | 0.60 / 0.31 | 1 |
| flow -cons floor | verse2 | 2.48 | 15.3 | 265 | 7.77 | -5.5 | 35 | -17.4 | -21.5 | 5.7 | 46 | -12.6 | -5.2 | -12.7 | 0.41 / 0.17 | 1 |
| flow -cons floor | MEAN | 2.20 | 14.2 | 425 | 7.54 | -6.0 | 43 | -23.7 | -26.0 | 5.9 | 54 | -13.8 | -12.3 | -12.5 | 0.47 / 0.25 |  |
| flow +rider | verse1 | 2.24 | 14.5 | 330 | 6.66 | -6.0 | 35 | -18.9 | -28.0 | 7.8 | 46 | -11.9 | -8.7 | -1.6 | 0.39 / 0.32 | 1 |
| flow +rider | chorus1 | 0.68 | 4.2 | 750 | 5.32 | -9.3 | 58 | -26.1 | -23.9 | 4.1 | 70 | -14.1 | -10.7 | -2.3 | 0.66 / 0.23 | 1 |
| flow +rider | verse2 | 1.42 | 8.4 | 245 | 5.66 | -7.7 | 35 | -15.1 | -20.3 | 4.3 | 46 | -10.5 | -3.3 | -4.4 | 0.26 / 0.17 | 1 |
| flow +rider | MEAN | 1.45 | 9.1 | 442 | 5.88 | -7.7 | 43 | -20.0 | -24.1 | 5.4 | 54 | -12.2 | -7.6 | -2.8 | 0.44 / 0.24 |  |
| flow +comp1 | verse1 | 1.86 | 11.3 | 330 | 4.95 | -8.1 | 35 | -13.0 | -25.2 | 7.4 | 46 | -5.7 | -5.0 | -0.9 | 0.53 / 0.27 | 1 |
| flow +comp1 | chorus1 | 0.46 | 3.0 | 780 | 4.34 | -11.2 | 58 | -22.3 | -18.6 | 5.4 | 70 | -10.7 | -6.9 | -1.2 | 0.66 / 0.40 | 1 |
| flow +comp1 | verse2 | 0.97 | 6.4 | 270 | 4.29 | -10.2 | 35 | -10.9 | -20.2 | 4.0 | 46 | -5.0 | -2.2 | -2.7 | 0.26 / 0.10 | 1 |
| flow +comp1 | MEAN | 1.10 | 6.9 | 460 | 4.53 | -9.8 | 43 | -15.4 | -21.3 | 5.6 | 54 | -7.1 | -4.7 | -1.6 | 0.48 / 0.26 |  |
| flow +comp2 | verse1 | 1.64 | 10.6 | 330 | 4.62 | -9.0 | 35 | -12.2 | -25.3 | 7.6 | 46 | -4.7 | -4.2 | -0.5 | 1.07 / 0.29 | 1 |
| flow +comp2 | chorus1 | 0.41 | 2.6 | 755 | 4.23 | -11.9 | 58 | -20.9 | -17.1 | 5.6 | 70 | -9.1 | -6.6 | -0.5 | 0.71 / 0.34 | 1 |
| flow +comp2 | verse2 | 0.75 | 5.1 | 200 | 4.37 | -11.3 | 35 | -10.0 | -20.2 | 3.7 | 46 | -3.6 | -1.6 | -1.9 | 0.24 / 0.12 | 1 |
| flow +comp2 | MEAN | 0.93 | 6.1 | 428 | 4.41 | -10.8 | 43 | -14.4 | -20.9 | 5.6 | 54 | -5.8 | -4.1 | -1.0 | 0.67 / 0.25 |  |
| flow +cons | verse1 | 1.64 | 10.5 | 330 | 4.57 | -9.0 | 35 | -12.2 | -24.2 | 7.8 | 46 | -4.7 | -4.2 | -0.5 | 0.40 / 0.29 | 1 |
| flow +cons | chorus1 | 0.64 | 6.0 | 755 | 4.09 | -12.0 | 58 | -20.9 | -17.5 | 5.9 | 70 | -8.9 | -6.6 | -0.5 | 0.71 / 0.34 | 1 |
| flow +cons | verse2 | 0.82 | 6.3 | 195 | 4.29 | -11.4 | 35 | -10.0 | -20.2 | 3.1 | 46 | -3.6 | -1.6 | -1.9 | 0.29 / 0.12 | 1 |
| flow +cons | MEAN | 1.03 | 7.6 | 427 | 4.32 | -10.8 | 43 | -14.4 | -20.6 | 5.6 | 54 | -5.7 | -4.1 | -1.0 | 0.47 / 0.25 |  |
| flow chain | verse1 | 1.60 | 10.3 | 372 | 4.65 | -9.1 | 35 | -12.0 | -25.0 | 7.8 | 46 | -5.4 | -4.8 | -0.5 | 0.47 / 0.31 | 3 |
| flow chain | chorus1 | 0.44 | 2.4 | 748 | 4.00 | -11.9 | 58 | -19.4 | -20.8 | 8.7 | 70 | -8.8 | -7.0 | -0.5 | 0.50 / 0.20 | 3 |
| flow chain | verse2 | 0.85 | 5.2 | 232 | 4.35 | -11.3 | 35 | -8.3 | -20.1 | 4.8 | 46 | -3.6 | -2.4 | -1.8 | 0.36 / 0.25 | 3 |
| flow chain | MEAN | 0.96 | 6.0 | 451 | 4.34 | -10.8 | 43 | -13.2 | -22.0 | 7.1 | 54 | -5.9 | -4.7 | -0.9 | 0.45 / 0.25 |  |
| flow chain+room | verse1 | 0.57 | 4.5 | 1027 | 6.00 | -10.5 | 35 | -7.7 | -25.0 | 5.6 | 46 | -5.4 | -3.7 | -0.1 | 0.85 / 0.58 | 3 |
| flow chain+room | chorus1 | 0.36 | 1.5 | 808 | 7.74 | -12.1 | 58 | -12.1 | -20.9 | 6.8 | 70 | -8.9 | -6.4 | -0.9 | 0.75 / 0.32 | 3 |
| flow chain+room | verse2 | 0.25 | 1.9 | 455 | 9.49 | -12.0 | 35 | -7.2 | -20.3 | 4.6 | 46 | -3.7 | -3.0 | -1.0 | 0.41 / 0.26 | 3 |
| flow chain+room | MEAN | 0.40 | 2.7 | 763 | 7.75 | -11.6 | 43 | -9.0 | -22.1 | 5.7 | 54 | -6.0 | -4.4 | -0.7 | 0.67 / 0.39 |  |

Three seeds, WER tiny.en / base.en / small.en (dry rows from the flow study's and the second demo's own logs):

| lead | verse1 | chorus1 | verse2 | mean |
|---|---|---|---|---|
| second demo, dry | 0.42 / 0.27 / 0.24 | 0.31 / 0.19 / 0.07 | 0.30 / 0.18 / 0.10 | 0.34 / 0.21 / 0.14 |
| fourth demo, dry | 0.61 / 0.31 / 0.22 (tiny: 0.39, 0.52, 0.93) | 0.52 / 0.10 / 0.08 | 0.31 / 0.18 / 0.10 | 0.48 / 0.19 / 0.13 |
| fourth demo, through the chain | 0.47 / 0.31 | 0.50 / 0.20 | 0.36 / 0.25 | 0.45 / 0.25 |
| fourth demo, chain + room (send 0.50) + delay | 0.85 / 0.58 (tiny 0.52, 0.64, 1.39; base 0.55, 0.82, 0.39) | 0.75 / 0.32 | 0.41 / 0.26 | 0.67 / 0.39 |
| second demo's take, chain + room (seed 1 only) | 0.52 / 0.43 | 0.34 / 0.23 | 0.26 / 0.14 | 0.38 / 0.27 |

The backing stack, from the stems of the section renders (post-fader, before the returns; phoneme times of the
lead's own take; no recognisers on these rows). The verse has no doubles or harmony. mix5's own figures for the
recognisers on the fourth demo's stems (sep.json): chorus lead alone 0.37 / 0.54 / 0.14, all vocals 1.14 / 0.71 /
0.17; verse 1 0.44 / 0.33 / 0.28 and 0.53 / 0.39 / 0.33; verse 2 0.31 / 0.14 / 0.10 and 0.36 / 0.21 / 0.15.

| row | case | holes/s | hole % | run ms | lvl_sd | mod 2-20 | stop ms | stop dip | burst | rise | sib ms | sib hf | vstop dip | vfric | WER tiny / base | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A lead stem | chorus1 | 0.46 | 2.6 | 920 | 3.68 | -11.9 | 58 | -16.2 | -21.0 | 8.7 | 70 | -6.9 | -6.6 | -0.6 |  | 1 |
| A lead stem | verse1 | 1.41 | 9.5 | 350 | 4.30 | -9.6 | 35 | -9.7 | -26.0 | 12.0 | 58 | -6.9 | -3.5 | -0.2 |  | 1 |
| A lead+doubles | chorus1 | 0.37 | 2.4 | 1050 | 4.04 | -11.3 | 58 | -15.3 | -21.2 | 8.3 | 70 | -7.0 | -5.5 | -0.5 |  | 1 |
| A lead+doubles+harmony | chorus1 | 0.37 | 2.3 | 930 | 4.75 | -11.5 | 58 | -12.0 | -19.4 | 8.2 | 70 | -7.3 | -5.0 | -0.8 |  | 1 |
| A all vocals | chorus1 | 0.58 | 4.7 | 680 | 4.78 | -11.8 | 58 | -12.0 | -19.4 | 8.2 | 70 | -7.3 | -5.0 | -0.8 |  | 1 |
| A all vocals | verse1 | 1.53 | 10.2 | 300 | 4.15 | -9.9 | 35 | -9.7 | -25.3 | 12.0 | 58 | -6.9 | -3.5 | -0.1 |  | 1 |
| A full mix | chorus1 | 0.00 | 0.0 | 140 | 2.66 | -16.5 | 58 | -5.4 | -17.4 | 2.4 | 70 | -10.0 | -3.3 | -0.1 |  | 1 |
| A full mix | verse1 | 0.17 | 0.6 | 130 | 3.68 | -12.2 | 35 | -3.4 | -23.7 | 7.7 | 58 | -11.6 | -3.4 | -0.6 |  | 1 |
| B lead stem | chorus1 | 0.79 | 6.2 | 690 | 4.82 | -10.2 | 93 | -35.0 | -14.4 | 20.6 | 116 | -9.4 | -12.9 | -1.8 |  | 1 |
| B lead stem | verse1 | 2.43 | 17.3 | 320 | 6.38 | -6.9 | 46 | -20.9 | -26.9 | 15.1 | 70 | -6.6 | -9.9 | -2.2 |  | 1 |
| B lead+doubles | chorus1 | 0.69 | 5.5 | 760 | 4.88 | -10.0 | 93 | -27.5 | -14.4 | 20.7 | 116 | -9.5 | -9.4 | -1.9 |  | 1 |
| B lead+doubles+harmony | chorus1 | 0.69 | 5.3 | 760 | 5.58 | -10.2 | 93 | -23.1 | -14.7 | 20.1 | 116 | -9.8 | -8.9 | -2.1 |  | 1 |
| B all vocals | chorus1 | 0.98 | 7.8 | 540 | 5.45 | -10.4 | 93 | -23.1 | -14.7 | 20.1 | 116 | -9.8 | -9.0 | -2.1 |  | 1 |
| B all vocals | verse1 | 2.53 | 17.9 | 285 | 6.24 | -7.3 | 46 | -20.8 | -26.0 | 15.1 | 70 | -6.6 | -10.0 | -2.2 |  | 1 |
| B full mix | chorus1 | 0.00 | 0.0 | 120 | 2.71 | -16.1 | 93 | -7.0 | -15.6 | 4.1 | 116 | -11.5 | -4.4 | -0.9 |  | 1 |
| B full mix | verse1 | 0.51 | 2.3 | 125 | 4.01 | -11.2 | 46 | -4.2 | -24.3 | 9.3 | 70 | -11.3 | -5.2 | -0.5 |  | 1 |
| C lead stem | chorus1 | 1.06 | 8.7 | 720 | 5.79 | -9.0 | 104 | -36.2 | -13.1 | 16.1 | 116 | -7.9 | -20.4 | -6.4 |  | 1 |
| C lead stem | verse1 | 3.09 | 20.6 | 230 | 7.35 | -5.3 | 46 | -31.2 | -24.8 | 13.5 | 81 | -5.9 | -14.8 | -5.2 |  | 1 |
| C lead+doubles | chorus1 | 0.93 | 7.3 | 690 | 5.42 | -9.1 | 104 | -28.0 | -13.1 | 16.3 | 116 | -7.8 | -14.8 | -6.6 |  | 1 |
| C lead+doubles+harmony | chorus1 | 0.83 | 6.7 | 690 | 5.99 | -9.3 | 104 | -25.0 | -13.5 | 16.8 | 116 | -8.1 | -12.8 | -5.9 |  | 1 |
| C all vocals | chorus1 | 1.07 | 9.3 | 450 | 5.72 | -9.6 | 104 | -25.0 | -13.5 | 16.7 | 116 | -8.1 | -12.8 | -5.9 |  | 1 |
| C all vocals | verse1 | 3.18 | 21.3 | 165 | 7.25 | -5.8 | 46 | -30.9 | -22.1 | 13.4 | 81 | -5.9 | -14.8 | -4.4 |  | 1 |
| C full mix | chorus1 | 0.00 | 0.0 | 115 | 2.76 | -15.7 | 104 | -7.3 | -14.0 | 4.3 | 116 | -9.9 | -6.2 | -1.6 |  | 1 |
| C full mix | verse1 | 0.73 | 3.6 | 120 | 4.23 | -10.8 | 46 | -4.4 | -21.3 | 10.1 | 81 | -10.3 | -5.4 | -1.1 |  | 1 |

## Step 2: the levers, one row each

Engine levers, each alone on the fourth demo's lead settings, dry, seed 1, verse1 and chorus1 (MEAN = both).
The recognisers do not separate them (all within 0.1 of `flow dry`, which is the noise of one seed); the consonant
and flow columns do.
- `burst` = stops and affricates sung at least 90 / 120 ms in slow motion and cut out around their release
  (`slow_cons_min` + `squeeze: burst`): closure 8 dB deeper, chorus release rise 4 -> 17 dB, burst level unchanged;
  verse voiced run 330 -> 220 ms. In C only.
- `burstf` = the same plus the fricatives' loudest stretch: no gain over `burst`. Not used.
- `peak` = peak-keeping squeeze: +2 dB on bursts, nothing else. Not used.
- `slow1` = consonants not slowed at all: sibilants 3 dB weaker. Rejected.
- `cap50` = longer caps (stops 50 ms, fricatives 70): verse stops 35 -> 46 ms, closure -22 -> -25, sibilants +1.7 dB;
  the chorus does not move because the half-length rule binds there.
- `share35` = caps bind only where the consonants take over 35 % of a note: the chorus gets the reference's
  consonants (stops 93 ms, closure -35, burst -22.9, rise 21.5, sibilants 116 ms at -11.4) at the reference's flow
  numbers (1.01 holes/s, 7.7 %, run 680, lvl_sd 6.4). In B and C.
- `nofloor` / `floorlow` = no / a lower voicing floor on voiced consonants: voiced fricatives -7 -> -18 / -15 dB in
  the chorus, 0.4-0.7 more holes/s, base.en worse without it (0.29 against 0.20). B keeps the floor (voiced stops
  14 dB instead of 11: the flow study measured no word cost at 14), C lowers it (16 / 11).
- `gain` = +6 dB on stops and +3 dB on fricatives above 2 kHz after the vocoder (`cons_gain`): bursts +1.4 / +3.4 dB,
  sibilants +3. Not used (the chain's lift does the sibilants; the option stays for taming or raising bursts).

| row | case | holes/s | hole % | run ms | lvl_sd | mod 2-20 | stop ms | stop dip | burst | rise | sib ms | sib hf | vstop dip | vfric | WER tiny / base | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow dry | verse1 | 2.29 | 15.2 | 330 | 7.38 | -4.7 | 35 | -22.4 | -29.2 | 8.4 | 46 | -13.5 | -11.6 | -6.2 | 0.40 / 0.29 | 1 |
| flow dry | chorus1 | 0.82 | 5.4 | 1040 | 6.03 | -8.8 | 58 | -30.8 | -27.4 | 4.1 | 70 | -15.0 | -14.3 | -7.4 | 0.63 / 0.11 | 1 |
| flow dry | MEAN | 1.55 | 10.3 | 685 | 6.71 | -6.7 | 46 | -26.6 | -28.3 | 6.2 | 58 | -14.2 | -12.9 | -6.8 | 0.51 / 0.20 |  |
| flow +burst | verse1 | 2.59 | 17.1 | 220 | 7.37 | -4.4 | 35 | -31.5 | -29.7 | 8.8 | 46 | -15.3 | -11.6 | -6.5 | 0.40 / 0.27 | 1 |
| flow +burst | chorus1 | 0.82 | 5.7 | 770 | 5.97 | -8.7 | 58 | -37.8 | -27.4 | 17.5 | 70 | -15.8 | -14.9 | -7.1 | 0.57 / 0.14 | 1 |
| flow +burst | MEAN | 1.70 | 11.4 | 495 | 6.67 | -6.5 | 46 | -34.6 | -28.5 | 13.2 | 58 | -15.6 | -13.2 | -6.8 | 0.48 / 0.21 |  |
| flow +burstf | verse1 | 2.38 | 16.9 | 200 | 7.13 | -4.4 | 35 | -31.9 | -30.1 | 8.6 | 46 | -12.7 | -12.3 | -5.4 | 0.40 / 0.28 | 1 |
| flow +burstf | chorus1 | 0.73 | 5.6 | 680 | 6.06 | -8.5 | 58 | -33.6 | -27.3 | 18.8 | 70 | -14.9 | -14.9 | -7.1 | 0.63 / 0.06 | 1 |
| flow +burstf | MEAN | 1.55 | 11.3 | 440 | 6.59 | -6.5 | 46 | -32.8 | -28.7 | 13.7 | 58 | -13.8 | -13.6 | -6.2 | 0.51 / 0.17 |  |
| flow +peak | verse1 | 2.25 | 14.4 | 335 | 6.94 | -4.9 | 35 | -20.8 | -27.1 | 8.8 | 46 | -12.2 | -11.6 | -5.1 | 0.41 / 0.27 | 1 |
| flow +peak | chorus1 | 0.82 | 5.2 | 770 | 6.03 | -8.9 | 58 | -30.8 | -26.3 | 7.2 | 70 | -14.2 | -14.3 | -7.4 | 0.69 / 0.11 | 1 |
| flow +peak | MEAN | 1.53 | 9.8 | 552 | 6.49 | -6.9 | 46 | -25.8 | -26.7 | 8.0 | 58 | -13.2 | -12.9 | -6.2 | 0.55 / 0.19 |  |
| flow +slow1 | verse1 | 2.46 | 14.4 | 530 | 6.90 | -4.9 | 35 | -16.6 | -30.0 | 6.8 | 46 | -17.1 | -12.4 | -6.3 | 0.94 / 0.36 | 1 |
| flow +slow1 | chorus1 | 0.68 | 5.0 | 770 | 5.91 | -8.7 | 58 | -31.1 | -26.5 | 8.0 | 70 | -17.4 | -12.4 | -7.4 | 0.20 / 0.09 | 1 |
| flow +slow1 | MEAN | 1.57 | 9.7 | 650 | 6.41 | -6.8 | 46 | -23.9 | -28.2 | 7.4 | 58 | -17.2 | -12.4 | -6.8 | 0.57 / 0.22 |  |
| flow +cap50 | verse1 | 2.29 | 16.5 | 290 | 7.63 | -4.2 | 46 | -25.0 | -27.5 | 10.2 | 70 | -11.8 | -12.6 | -5.2 | 0.38 / 0.25 | 1 |
| flow +cap50 | chorus1 | 0.87 | 5.6 | 900 | 6.12 | -8.7 | 58 | -33.3 | -27.0 | 4.0 | 70 | -14.1 | -15.0 | -7.3 | 0.66 / 0.23 | 1 |
| flow +cap50 | MEAN | 1.58 | 11.1 | 595 | 6.88 | -6.4 | 52 | -29.1 | -27.2 | 7.1 | 70 | -12.9 | -13.8 | -6.2 | 0.52 / 0.24 |  |
| flow +share35 | verse1 | 2.29 | 16.6 | 290 | 7.55 | -4.4 | 35 | -25.0 | -28.7 | 11.3 | 58 | -11.9 | -11.9 | -6.3 | 0.59 / 0.27 | 1 |
| flow +share35 | chorus1 | 1.01 | 7.7 | 680 | 6.43 | -8.0 | 93 | -34.7 | -22.9 | 21.5 | 116 | -11.4 | -14.2 | -7.3 | 0.63 / 0.03 | 1 |
| flow +share35 | MEAN | 1.65 | 12.2 | 485 | 6.99 | -6.2 | 64 | -29.9 | -25.8 | 16.4 | 87 | -11.7 | -13.1 | -6.8 | 0.61 / 0.15 |  |
| flow +nofloor | verse1 | 2.94 | 19.8 | 290 | 8.13 | -4.3 | 35 | -23.0 | -29.2 | 7.9 | 46 | -13.8 | -12.8 | -6.6 | 0.41 / 0.27 | 1 |
| flow +nofloor | chorus1 | 1.19 | 7.5 | 720 | 6.71 | -8.2 | 58 | -30.8 | -27.3 | 4.1 | 70 | -15.0 | -18.9 | -18.3 | 0.60 / 0.31 | 1 |
| flow +nofloor | MEAN | 2.06 | 13.6 | 505 | 7.42 | -6.3 | 46 | -26.9 | -28.2 | 6.0 | 58 | -14.4 | -15.8 | -12.4 | 0.50 / 0.29 |  |
| flow +floorlow | verse1 | 2.98 | 18.6 | 335 | 7.88 | -4.5 | 35 | -22.4 | -29.2 | 7.5 | 46 | -13.8 | -13.2 | -6.4 | 0.35 / 0.26 | 1 |
| flow +floorlow | chorus1 | 1.19 | 7.4 | 1040 | 6.45 | -8.4 | 58 | -30.8 | -27.4 | 4.1 | 70 | -15.0 | -18.9 | -14.8 | 0.71 / 0.09 | 1 |
| flow +floorlow | MEAN | 2.08 | 13.0 | 688 | 7.17 | -6.4 | 46 | -26.6 | -28.3 | 5.8 | 58 | -14.4 | -16.0 | -10.6 | 0.53 / 0.17 |  |
| flow +gain | verse1 | 2.29 | 15.0 | 335 | 7.03 | -4.8 | 35 | -22.0 | -27.9 | 10.6 | 46 | -10.6 | -11.6 | -6.2 | 0.38 / 0.28 | 1 |
| flow +gain | chorus1 | 0.82 | 5.2 | 905 | 5.73 | -8.9 | 58 | -30.0 | -24.0 | 4.0 | 70 | -12.1 | -14.3 | -7.4 | 0.69 / 0.11 | 1 |
| flow +gain | MEAN | 1.55 | 10.1 | 620 | 6.38 | -6.8 | 46 | -26.0 | -25.9 | 7.3 | 58 | -11.3 | -12.9 | -6.8 | 0.53 / 0.20 |  |

Chain levers on the fourth demo's take (`flow | X`), seed 1, verse1 + chorus1: A = the fourth demo's chain;
`nocons` = without the consonant leveller (nothing changes: see "Still wrong", the detector); `c1_off` = without the
2:1 leveller (lvl_sd 4.25 -> 5.54, stop closure -16.5 -> -22.9, sibilants 4.6 dB quieter); B1 = rider 5 dB x 0.7 +
leveller 1.4:1, no consonant leveller; C1 = rider 4 dB x 0.5, no leveller, comp2 2:1; C2 = C1 + trans + lift.
Room levers on a B take through B's chain (`dial_B_v2 | Bc X`): no returns, reverb at send 0.25 alone, the delay
alone, both, and send 0.50 with the delay. The reverb sets the closure depth (-18.6 at 0.25, -13.1 at 0.50, -29.2
without); the delay alone leaves -22. A mono room inside the channel, ducked 10 dB while the lead sings, filled
more than it saved (verse closure -15.1 against -18.6 for send 0.15 alone) and was dropped. base.en 0.62 on
"delay only" is one recogniser accident (the pair of cases reads 0.13-0.31 in the rows around it).

| row | case | holes/s | hole % | run ms | lvl_sd | mod 2-20 | stop ms | stop dip | burst | rise | sib ms | sib hf | vstop dip | vfric | WER tiny / base | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| flow | A | verse1 | 1.68 | 10.6 | 330 | 4.53 | -9.0 | 35 | -12.2 | -24.2 | 7.9 | 46 | -4.7 | -4.2 | -0.5 | 0.44 / 0.29 | 1 |
| flow | A | chorus1 | 0.41 | 2.6 | 825 | 3.97 | -11.9 | 58 | -20.9 | -17.5 | 6.1 | 70 | -8.9 | -6.6 | -0.5 | 0.74 / 0.37 | 1 |
| flow | A | MEAN | 1.04 | 6.6 | 578 | 4.25 | -10.5 | 46 | -16.5 | -20.9 | 7.0 | 58 | -6.8 | -5.4 | -0.5 | 0.59 / 0.33 |  |
| flow | nocons | verse1 | 1.64 | 10.6 | 330 | 4.58 | -9.0 | 35 | -12.2 | -25.3 | 8.1 | 46 | -4.7 | -4.2 | -0.5 | 0.46 / 0.29 | 1 |
| flow | nocons | chorus1 | 0.41 | 2.6 | 825 | 4.07 | -11.9 | 58 | -20.9 | -17.1 | 5.7 | 70 | -9.1 | -6.6 | -0.5 | 0.71 / 0.37 | 1 |
| flow | nocons | MEAN | 1.02 | 6.6 | 578 | 4.33 | -10.4 | 46 | -16.5 | -21.2 | 6.9 | 58 | -6.9 | -5.4 | -0.5 | 0.58 / 0.33 |  |
| flow | c1_off | verse1 | 2.33 | 14.5 | 335 | 6.31 | -6.9 | 35 | -18.9 | -27.7 | 7.3 | 46 | -11.4 | -8.1 | -1.5 | 0.49 / 0.28 | 1 |
| flow | c1_off | chorus1 | 0.68 | 4.1 | 820 | 4.77 | -10.3 | 58 | -26.8 | -23.6 | 5.0 | 70 | -11.4 | -10.7 | -1.4 | 0.69 / 0.37 | 1 |
| flow | c1_off | MEAN | 1.51 | 9.3 | 578 | 5.54 | -8.6 | 46 | -22.9 | -25.6 | 6.2 | 58 | -11.4 | -9.4 | -1.4 | 0.59 / 0.33 |  |
| flow | B1 | verse1 | 2.16 | 13.2 | 330 | 5.55 | -7.8 | 35 | -15.8 | -26.1 | 8.5 | 46 | -8.1 | -6.5 | -1.2 | 0.40 / 0.27 | 1 |
| flow | B1 | chorus1 | 0.64 | 3.7 | 820 | 4.45 | -11.1 | 58 | -26.8 | -21.4 | 4.7 | 70 | -12.3 | -8.9 | -1.3 | 0.51 / 0.34 | 1 |
| flow | B1 | MEAN | 1.40 | 8.5 | 575 | 5.00 | -9.4 | 46 | -21.3 | -23.8 | 6.6 | 58 | -10.2 | -7.7 | -1.2 | 0.46 / 0.31 |  |
| flow | C1 | verse1 | 2.29 | 14.8 | 330 | 6.81 | -6.2 | 35 | -20.7 | -28.2 | 8.2 | 46 | -13.0 | -9.3 | -2.8 | 0.38 / 0.28 | 1 |
| flow | C1 | chorus1 | 0.73 | 4.7 | 860 | 5.24 | -9.9 | 58 | -27.2 | -25.4 | 4.2 | 70 | -14.4 | -12.1 | -3.8 | 0.69 / 0.14 | 1 |
| flow | C1 | MEAN | 1.51 | 9.8 | 595 | 6.03 | -8.0 | 46 | -23.9 | -26.8 | 6.2 | 58 | -13.7 | -10.7 | -3.3 | 0.53 / 0.21 |  |
| flow | C2 | verse1 | 2.29 | 14.6 | 330 | 6.44 | -6.5 | 35 | -20.3 | -26.6 | 9.7 | 46 | -10.7 | -9.2 | -2.7 | 0.44 / 0.26 | 1 |
| flow | C2 | chorus1 | 0.73 | 4.5 | 945 | 4.93 | -10.1 | 58 | -26.7 | -22.4 | 3.2 | 70 | -11.6 | -12.1 | -3.8 | 0.69 / 0.11 | 1 |
| flow | C2 | MEAN | 1.51 | 9.6 | 638 | 5.69 | -8.3 | 46 | -23.5 | -24.5 | 6.4 | 58 | -11.1 | -10.6 | -3.2 | 0.56 / 0.19 |  |
| dial_B_v2 | Bc none | verse1 | 2.29 | 16.9 | 320 | 6.46 | -6.6 | 46 | -20.7 | -25.4 | 14.6 | 58 | -7.9 | -7.7 | -1.5 | 0.41 / 0.26 | 1 |
| dial_B_v2 | Bc none | chorus1 | 0.64 | 5.9 | 660 | 5.07 | -10.2 | 93 | -37.8 | -19.7 | 20.7 | 116 | -7.9 | -8.4 | -1.9 | 0.66 / 0.06 | 1 |
| dial_B_v2 | Bc none | MEAN | 1.47 | 11.4 | 490 | 5.77 | -8.4 | 70 | -29.2 | -22.5 | 17.6 | 87 | -7.9 | -8.1 | -1.7 | 0.53 / 0.16 |  |
| dial_B_v2 | Bc rev.25 only | verse1 | 2.15 | 14.6 | 530 | 7.27 | -7.3 | 46 | -16.6 | -25.4 | 11.8 | 58 | -7.9 | -7.5 | -1.3 | 0.35 / 0.24 | 1 |
| dial_B_v2 | Bc rev.25 only | chorus1 | 0.63 | 5.3 | 670 | 6.65 | -10.6 | 93 | -20.5 | -19.6 | 16.7 | 116 | -7.9 | -8.3 | -2.3 | 0.63 / 0.03 | 1 |
| dial_B_v2 | Bc rev.25 only | MEAN | 1.39 | 9.9 | 600 | 6.96 | -8.9 | 70 | -18.6 | -22.5 | 14.2 | 87 | -7.9 | -7.9 | -1.8 | 0.49 / 0.13 |  |
| dial_B_v2 | Bc delay only | verse1 | 2.22 | 16.1 | 290 | 8.13 | -6.9 | 46 | -19.6 | -25.4 | 13.5 | 58 | -7.9 | -7.9 | -1.6 | 0.46 / 1.12 | 1 |
| dial_B_v2 | Bc delay only | chorus1 | 0.62 | 5.7 | 370 | 9.15 | -10.2 | 93 | -24.4 | -19.7 | 19.2 | 116 | -7.8 | -8.5 | -1.7 | 0.60 / 0.11 | 1 |
| dial_B_v2 | Bc delay only | MEAN | 1.42 | 10.9 | 330 | 8.64 | -8.5 | 70 | -22.0 | -22.5 | 16.4 | 87 | -7.8 | -8.2 | -1.6 | 0.53 / 0.62 |  |
| dial_B_v2 | Bc rev.25+delay | verse1 | 2.06 | 13.7 | 875 | 7.43 | -7.4 | 46 | -16.4 | -25.3 | 10.3 | 58 | -7.9 | -7.4 | -1.2 | 0.40 / 0.29 | 1 |
| dial_B_v2 | Bc rev.25+delay | chorus1 | 0.62 | 5.0 | 430 | 8.64 | -10.5 | 93 | -20.4 | -19.6 | 16.5 | 116 | -7.9 | -8.4 | -2.4 | 0.63 / 0.14 | 1 |
| dial_B_v2 | Bc rev.25+delay | MEAN | 1.34 | 9.3 | 652 | 8.04 | -8.9 | 70 | -18.4 | -22.5 | 13.4 | 87 | -7.9 | -7.9 | -1.8 | 0.51 / 0.22 |  |
| dial_B_v2 | Bc rev.50+delay | verse1 | 1.01 | 6.5 | 1150 | 6.39 | -8.4 | 46 | -11.7 | -25.2 | 8.0 | 58 | -7.8 | -6.1 | -1.0 | 0.50 / 0.31 | 1 |
| dial_B_v2 | Bc rev.50+delay | chorus1 | 0.62 | 3.1 | 445 | 8.01 | -10.8 | 93 | -14.5 | -19.7 | 13.8 | 116 | -7.9 | -8.2 | -2.5 | 0.63 / 0.31 | 1 |
| dial_B_v2 | Bc rev.50+delay | MEAN | 0.81 | 4.8 | 798 | 7.20 | -9.6 | 70 | -13.1 | -22.4 | 10.9 | 87 | -7.8 | -7.1 | -1.8 | 0.56 / 0.31 |  |

## The dial: A (the fourth demo), B (balanced), C (crisp)

Settings (config.json "dial"; everything not named is the fourth demo's):

| | A | B | C |
|---|---|---|---|
| consonant caps, ms: stops / voiced stops / affricates / fricatives / voiced fricatives / nasals / glides | 35 / 35 / 60 / 50 / 50 / 60 / 50 | 45 / 40 / 70 / 65 / 50 / 60 / 50 | 50 / 45 / 80 / 80 / 60 / 70 / 60 |
| caps bind only where a note's consonants take more than | (always) | 35 % of the note | 40 % of the note |
| never under half the duration model's length | yes | yes | yes |
| voicing floor under the vowels, dB: voiced stops / voiced fricatives / nasals, glides | 11 / 8 / 8 | 14 / 8 / 8 | 16 / 11 / 8 |
| stops and affricates cut out around their release (burst squeeze) | no | no | yes (sung 90 / 120 ms in slow motion) |
| breaths | as the flow study | same | same |
| chain: rider | +-8 dB, full | +-5 dB x 0.6 | +-4 dB x 0.5 |
| chain: leveller (comp1) | 2:1 | 1.25:1 | off |
| chain: comp2 on the top 6 dB | 3:1 | 3:1 | 2:1 |
| chain: consonant leveller toward -8 dB | +4 / -9 dB | limiter only (over -10 dB, down to 6 dB) | limiter only, after the lift |
| chain: transient emphasis above 2 kHz | no | no | 0.4 dB per dB, up to 5 dB |
| chain: sibilant lift above 2.5 kHz (tilt detector, no further than -5 dB re the sung level) | no | +2 dB | +2.5 dB |
| lead's send to the room (RT60 0.6 s) | 0.50 | 0.25 | 0.15 |

The lead in the song, one take per section (bars 33-48 and 9-24; `lead6.py`, `dialtable.py`). "chain only" = the
channel as it goes to the mix bus; "with returns" = plus its own reverb and delay returns (what the lead clips are).
dB values are re the median vowel; the reference column is the bank's sample songs, dry.

| measure | reference | chorus A | chorus B | chorus C | verse A | verse B | verse C |
|---|---|---|---|---|---|---|---|
| holes/s, with returns | 1.05-1.30 | 0.40 | 0.66 | 1.06 | 0.44 | 2.15 | 2.81 |
| hole time %, with returns | 8-11 | 2.0 | 5.1 | 8.2 | 4.5 | 14.1 | 18.9 |
| holes/s, chain only |  | 0.46 | 0.74 | 1.06 | 1.45 | 2.43 | 3.04 |
| hole time %, chain only |  | 2.7 | 6.5 | 8.8 | 9.8 | 17.4 | 21.0 |
| voiced run ms, chain only | 375-470 | 780 | 665 | 720 | 540 | 320 | 230 |
| lvl_sd, chain only | 5.9-6.7 | 3.81 | 4.94 | 5.87 | 4.48 | 6.47 | 7.35 |
| modulation 2-20 Hz dB, with returns | -6.6 | -12.6 | -10.6 | -9.1 | -10.8 | -7.5 | -5.7 |
| consonants, % of sung time | 22-25 | 23 | 30 | 32 | 36 | 41 | 45 |
| stops p t k: length ms | 116 | 58 | 93 | 104 | 35 | 46 | 46 |
| stop closure dB, chain / with returns | -38 | -17 / -12 | -36 / -22 | -37 / -24 | -10 / -7 | -21 / -15 | -32 / -20 |
| stop burst above 4 kHz dB | -23.7 | -21.6 | -15.0 | -13.6 | -26.2 | -27.2 | -24.7 |
| stop release rise dB | 12.5 | 8.5 | 20.8 | 16.0 | 10.8 | 15.0 | 13.4 |
| sibilants: length ms | 116 | 70 | 116 | 116 | 58 | 70 | 81 |
| sibilant noise above 4 kHz dB | -8.9 | -7.5 | -9.9 | -8.4 | -7.3 | -6.8 | -5.7 |
| f th hh noise above 4 kHz dB | -22.7 (f h) | -11.3 | -13.6 | -11.6 | -20.9 | -16.4 | -9.0 |
| voiced stops: closure dB | -20 (unaspirated) | -7.2 | -14.1 | -21.4 | -4.3 | -10.6 | -14.6 |
| voiced fricatives: level dB |  | -0.7 | -2.4 | -6.8 | -1.0 | -3.3 | -6.3 |
| hnr, chain only |  | 20.8 | 20.4 | 20.2 | 17.3 | 16.9 | 16.8 |
| WER tiny / base / small, chain only |  | 0.54 / 0.49 / 0.20 | 0.69 / 0.17 / 0.09 | 0.51 / 0.23 / 0.11 | 0.46 / 0.41 / 0.32 | 0.44 / 0.31 / 0.25 | 0.44 / 0.26 / 0.23 |
| WER tiny / base / small, with returns |  | 0.83 / 0.23 / 0.23 | 0.51 / 0.20 / 0.17 | 0.46 / 0.26 / 0.14 | 0.58 / 0.61 / 0.27 | 0.49 / 0.50 / 0.28 | 0.52 / 0.36 / 0.25 |

Read against the targets:
- holes/s 1.0-1.5 and hole time 8-12 %: chorus C is there (1.06, 8.2 %), chorus B is between A and the reference
  (0.66, 5.1 %); the verse is over in B (2.15, 14.1 %) and C (2.81, 18.9 %) and far under in A (0.44, 4.5 %). The flow
  study's sum for this verse's lyric at the reference's own share of holes is 1.7-2.0 holes/s.
- voiced run 350 ms or more: chorus yes in all; verse 320 (B) and 230 (C) through the chain.
- lvl_sd 5.5-6.5: B 4.94 chorus / 6.47 verse, C 5.87 / 7.35, A 3.81 / 4.48. One static chain cannot put both sections
  in the window: dry they are 1.2 apart.
- stop bursts within 3 dB of the reference: verse yes in B (-27.2, 3.5 under) and C (-24.7); chorus NO in B (-15.0)
  and C (-13.6): 9-10 dB over. They are the bank's own bursts at their natural length (dry -22.9, the reference's
  level) plus what the chain adds; A's capped stops read -21.6.
- sibilants within 3 dB of the reference (-8.9): B -9.9 / -6.8, C -8.4 / -5.7 (the verse in C is 3.2 over), A -7.5 / -7.3.
- consonant length: chorus stops 93 / 104 ms and sibilants 116 ms against the reference's 116 / 116; by share of the
  sung time the chorus is now past the reference (30-32 % against 22-25 %) because English has more consonants per
  syllable. Verse stops 46 ms (A 35, second demo 52).
- modulation 2-20 Hz (reference -6.6 dB): chorus -12.6 (A) -> -10.6 (B) -> -9.1 (C); verse -10.8 -> -7.5 -> -5.7.
- recognisers, one take per section (noise about 0.1; the same clip read twice by base.en gave 1.00 and 0.61):
  mean of the two sections with returns: base.en A 0.42, B 0.35, C 0.31; small.en 0.25, 0.23, 0.20; tiny.en 0.71,
  0.50, 0.49. Three seeds on the harness cases are in the next table.

In the chorus mix (mask.py: the lead's loudest 10 ms above 4 kHz against the sum of all other stems above 4 kHz at
that moment): sibilants A +8.6, B +7.7, C +5.4 dB; f / th / hh +8.9, +7.2, +7.5; released stops -2.6, -0.6, +1.1;
vowels in 1-5 kHz +6.1, +5.9, +5.6. So in the full chorus the stop bursts sit at the level of the band's top in
every setting, and C's sibilants stand 3 dB less over it than A's (C levels less). The stack: the doubles (copies
of the lead 18 / 27 ms late) fill the lead's chorus closures from -35 to -27.5 dB (B) and -36 to -28 (C), the harmony
to -23 / -25; the lead's own returns alone fill them to -22 / -24; the band to -7.

Three seeds on the harness cases (`hc.py dial_B_v3 dial_C_v2 --seeds=1,2,3`, `lab.py dial:B dial:C --seeds=1,2,3`;
A = the rows `flow chain` and `flow chain+room`). `X v4 chain` = through the setting's chain, `room` = with the
returns at the setting's send. WER tiny.en / base.en; the first two rows are seed 1 only (their three-seed WER is in
the step-1 table: second demo dry 0.34 / 0.21, fourth demo dry 0.48 / 0.19).

| WER tiny / base, three seeds | verse1 | chorus1 | verse2 | mean |
|---|---|---|---|---|
| target: second demo's lead, dry | 0.42 / 0.27 | 0.31 / 0.19 | 0.30 / 0.18 | 0.34 / 0.21 |
| A through its chain | 0.47 / 0.31 | 0.50 / 0.20 | 0.36 / 0.25 | 0.45 / 0.25 |
| A with its returns | 0.85 / 0.58 | 0.75 / 0.32 | 0.41 / 0.26 | 0.67 / 0.39 |
| B through its chain | 0.42 / 0.26 | 0.42 / 0.06 | 0.28 / 0.14 | 0.37 / 0.15 |
| B with its returns | 0.47 / 0.28 | 0.47 / 0.06 | 0.32 / 0.13 | 0.42 / 0.16 |
| C through its chain | 0.34 / 0.22 | 0.36 / 0.14 | 0.19 / 0.16 | 0.30 / 0.17 |
| C with its returns | 0.38 / 0.23 | 0.42 / 0.23 | 0.25 / 0.14 | 0.35 / 0.20 |

So against the target: base.en is at or under it in every case for B and C (B's verse 1 with returns 0.28 against
0.27); tiny.en is over it in the chorus (0.42-0.47 against 0.31) and in B's verse 1 with returns (0.47 against 0.42),
under it elsewhere. Flow on the same takes with returns: B 2.00 holes/s / 13.3 % (verse1), 0.63 / 4.9 % (chorus),
1.13 / 6.4 % (verse2); C 2.73 / 17.8 %, 0.99 / 7.6 %, 1.51 / 9.8 %. lvl_sd through the chain: B 6.41 / 5.02 / 5.40
(mean 5.6), C 7.28 / 5.75 / 6.40 (mean 6.5). Stop bursts through the chain, mean of the three cases: B -22.0, C -21.3
(reference -23.7); sibilants B -6.5, C -7.4 (reference -8.9).

Measures of the same rows:


| row | case | holes/s | hole % | run ms | lvl_sd | mod 2-20 | stop ms | stop dip | burst | rise | sib ms | sib hf | vstop dip | vfric | WER tiny / base | seeds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| demo2 dry | MEAN | 2.52 | 21.7 | 260 | 8.41 | -5.4 | 72 | -32.2 | -21.9 | 18.8 | 89 | -10.7 | -26.6 | -13.2 | 0.38 / 0.21 |  |
| flow dry | MEAN | 1.59 | 10.1 | 553 | 6.66 | -6.5 | 43 | -23.5 | -26.0 | 6.1 | 54 | -13.7 | -10.4 | -7.1 | 0.43 / 0.18 |  |
| flow chain | verse1 | 1.60 | 10.3 | 372 | 4.65 | -9.1 | 35 | -12.0 | -25.0 | 7.8 | 46 | -5.4 | -4.8 | -0.5 | 0.47 / 0.31 | 3 |
| flow chain | chorus1 | 0.44 | 2.4 | 748 | 4.00 | -11.9 | 58 | -19.4 | -20.8 | 8.7 | 70 | -8.8 | -7.0 | -0.5 | 0.50 / 0.20 | 3 |
| flow chain | verse2 | 0.85 | 5.2 | 232 | 4.35 | -11.3 | 35 | -8.3 | -20.1 | 4.8 | 46 | -3.6 | -2.4 | -1.8 | 0.36 / 0.25 | 3 |
| flow chain | MEAN | 0.96 | 6.0 | 451 | 4.34 | -10.8 | 43 | -13.2 | -22.0 | 7.1 | 54 | -5.9 | -4.7 | -0.9 | 0.45 / 0.25 |  |
| flow chain+room | verse1 | 0.57 | 4.5 | 1027 | 6.00 | -10.5 | 35 | -7.7 | -25.0 | 5.6 | 46 | -5.4 | -3.7 | -0.1 | 0.85 / 0.58 | 3 |
| flow chain+room | chorus1 | 0.36 | 1.5 | 808 | 7.74 | -12.1 | 58 | -12.1 | -20.9 | 6.8 | 70 | -8.9 | -6.4 | -0.9 | 0.75 / 0.32 | 3 |
| flow chain+room | verse2 | 0.25 | 1.9 | 455 | 9.49 | -12.0 | 35 | -7.2 | -20.3 | 4.6 | 46 | -3.7 | -3.0 | -1.0 | 0.41 / 0.26 | 3 |
| flow chain+room | MEAN | 0.40 | 2.7 | 763 | 7.75 | -11.6 | 43 | -9.0 | -22.1 | 5.7 | 54 | -6.0 | -4.4 | -0.7 | 0.67 / 0.39 |  |
| B v4 chain | verse1 | 2.30 | 16.7 | 323 | 6.41 | -6.7 | 46 | -20.6 | -26.3 | 12.7 | 58 | -6.7 | -11.0 | -2.3 | 0.42 / 0.26 | 3 |
| B v4 chain | chorus1 | 0.70 | 5.9 | 672 | 5.02 | -10.1 | 93 | -35.5 | -19.8 | 20.6 | 116 | -6.8 | -12.9 | -1.6 | 0.42 / 0.06 | 3 |
| B v4 chain | verse2 | 1.51 | 9.9 | 370 | 5.40 | -8.3 | 46 | -15.9 | -20.0 | 12.0 | 58 | -6.1 | -4.9 | -3.6 | 0.28 / 0.14 | 3 |
| B v4 chain | MEAN | 1.51 | 10.9 | 455 | 5.61 | -8.4 | 62 | -24.0 | -22.0 | 15.1 | 77 | -6.5 | -9.6 | -2.5 | 0.37 / 0.15 |  |
| B v4 room | verse1 | 2.00 | 13.3 | 770 | 7.43 | -7.5 | 46 | -15.7 | -26.3 | 9.7 | 58 | -6.6 | -9.2 | -1.7 | 0.47 / 0.28 | 3 |
| B v4 room | chorus1 | 0.63 | 4.9 | 650 | 8.52 | -10.3 | 93 | -20.5 | -19.8 | 15.7 | 116 | -6.8 | -12.5 | -1.9 | 0.47 / 0.06 | 3 |
| B v4 room | verse2 | 1.13 | 6.4 | 313 | 9.92 | -8.7 | 46 | -13.6 | -20.0 | 10.0 | 58 | -6.2 | -4.8 | -3.3 | 0.32 / 0.13 | 3 |
| B v4 room | MEAN | 1.25 | 8.2 | 578 | 8.62 | -8.8 | 62 | -16.6 | -22.0 | 11.8 | 77 | -6.5 | -8.8 | -2.3 | 0.42 / 0.16 |  |
| C v4 chain | verse1 | 3.03 | 20.2 | 257 | 7.28 | -5.0 | 46 | -31.8 | -27.5 | 11.3 | 81 | -6.4 | -14.9 | -4.5 | 0.34 / 0.22 | 3 |
| C v4 chain | chorus1 | 0.99 | 8.1 | 637 | 5.75 | -8.8 | 104 | -37.9 | -18.0 | 27.2 | 116 | -8.9 | -21.5 | -6.1 | 0.36 / 0.14 | 3 |
| C v4 chain | verse2 | 2.09 | 13.4 | 322 | 6.40 | -6.1 | 46 | -27.6 | -18.4 | 23.2 | 70 | -6.8 | -7.9 | -7.6 | 0.19 / 0.16 | 3 |
| C v4 chain | MEAN | 2.04 | 13.9 | 405 | 6.48 | -6.7 | 66 | -32.4 | -21.3 | 20.6 | 89 | -7.4 | -14.8 | -6.1 | 0.30 / 0.17 |  |
| C v4 room | verse1 | 2.73 | 17.8 | 295 | 8.34 | -5.6 | 46 | -19.5 | -27.3 | 9.0 | 81 | -6.3 | -13.7 | -4.6 | 0.38 / 0.23 | 3 |
| C v4 room | chorus1 | 0.99 | 7.6 | 352 | 8.73 | -9.0 | 104 | -23.1 | -17.9 | 20.7 | 116 | -8.9 | -18.9 | -6.3 | 0.42 / 0.23 | 3 |
| C v4 room | verse2 | 1.51 | 9.8 | 365 | 10.19 | -6.3 | 46 | -19.2 | -18.4 | 18.2 | 70 | -6.8 | -7.7 | -7.3 | 0.25 / 0.14 | 3 |
| C v4 room | MEAN | 1.74 | 11.7 | 337 | 9.09 | -7.0 | 66 | -20.6 | -21.2 | 16.0 | 89 | -7.3 | -13.4 | -6.1 | 0.35 / 0.20 |  |

## Clips (out/, wav 24 bit + 160k mp3; out/clips.json has levels and peaks)

Lead clips: the AI lead channels alone at their faders, through the setting's chain, WITH the lead's own reverb and
delay returns as the mixer makes them (the room is one of the things the dial changes); all six at -17.0 LUFS (mp3),
true peaks -6.3 to -8.1 dBFS, one gain each. Mix clips: every track, at the fourth demo's mix loudness (chorus
-11.6 LUFS wav / -12.1 mp3 against 03's -11.5 / -12.0; verse -13.1 / -13.6 against 07's -13.0 / -13.5), true peak
-1.0 dBFS. A of the mixes is mix5/out/03_chorus_full_mix and 07_verse_full_mix.

- chorus_lead_A / B / C: bars 33-48. Listen to the k and t of "corner", "saucer count", "can", "apocalypse" and the
  s of "select", "saucer", "nice": 58 ms stops in A, 93 / 104 ms with a real gap before them in B / C. Check that the
  k / t in B and C are not spat (they measure 9-10 dB over the reference's bursts) and whether C's lines still
  connect (1.06 holes/s against 0.40 in A).
- verse_lead_A / B / C: bars 9-24. The fast lines ("there's a saucer parked across the lawn", "pop up on a pop up").
  B should keep most of A's line with consonants one could write down; C is the stop-and-start end (2.8 holes/s,
  19 % of the time in a hole: close to the second demo's 3.2 / 28 %, but with breaths and a voiced floor).
- chorus_mix_B / C against mix5/out/03: do the words come through the band; are the lead's s sounds as present as
  in 03 (measured B -1 dB, C -3 dB against 03); is the drier lead (send 0.25 / 0.15) still glued to the track.
- verse_mix_B / C against mix5/out/07: the same questions; the verse has no doubles or harmony, so the lead clips
  tell most of it.

## Still wrong / not verified

- Nobody has listened. The measures were built in this session and have no listening test behind them; the
  recognisers move by 0.1 between takes and by more between two readings of one clip.
- The recognisers' target (the second demo's dry lead: verse about 0.42 / 0.27, chorus 0.31 / 0.19): base.en meets
  it in B and C; tiny.en does not in the chorus (0.42-0.47) nor in B's verse 1 with returns (0.47). The flow study
  traced tiny.en's chorus loss to the breaths at the line starts (0.30 without them); the breaths were not touched
  here. In the song (one take per section) the numbers are worse than on the harness for every setting.
- The verse at 5.4 syllables a second cannot have the reference's consonant lengths and the reference's hole time
  together: B gives it 46 ms stops and 14 % hole time (with returns), C 19 %. The relaxed verse of the production
  study (3.7 syllables a second) was not tried with the dial.
- Chorus stop bursts in B / C are 9-10 dB over the reference's level (see above). The chain's burst limiter does not
  catch them: its detector (periodicity under 1.5 kHz in 48 ms frames) reads a short burst or a sibilant between two
  vowels as "sung". The same flaw means the fourth demo's consonant leveller acted on few consonants (switching it
  off changes nothing in the table), and that every "noise frames follow the sung level" rule of chain4 covers only
  long or isolated consonants. The new lift uses a spectral-tilt detector; the limiter does not yet.
  `cons_gain: {"stop": -4, "hf": 2000}` in the engine would take 4 dB off every stop burst (verse included).
- The sibilant lift reached the chorus sibilants of B only partly in the song take (-10.3 -> -9.9 dB for a 2 dB lift;
  on the harness take the same stage gives +2.9 of 3): not investigated.
- Voiced stops stay shallow in B (closure -14 chorus / -11 verse against -20 for the reference's unaspirated stops);
  C reaches -21 / -15. Lowering the floor further costs holes and, on one seed, base.en words.
- Doubles and harmony fill 12 dB of the lead's chorus closures in B / C. A duck of the copies keyed by the lead's
  own level (so a copy never sounds in the lead's closure) was not built.
- lvl_sd is under the window in B's chorus (4.94) and over it in C's verse (7.35).
- Only the lead's settings changed. The harmony and the lead's answer lines were re-sung with the same consonant
  options but not measured on their own; aliens and crowd are untouched. Only bars 9-24 and 33-48 exist with B / C;
  the whole song was not rendered (do it in sections of 16-20 bars).
- The first word of the verse ("three") is missing from every transcript in every setting; not looked into.
- The harness evaluation uses the B engine with version tag v3 and the C engine tag v2: the engine options of the
  final settings (B v4 / C v4 differ from them only in the chain).

## Status log (newest last)

- start: read briefs and docs; folder made; engine (mix5/dsinfer.py, dspitch.py), chain (prod/chain4.py -> chain5.py),
  backend (mix5/ds_backend5.py -> ds_backend6.py), run5.py -> run6.py copied. Nothing changed yet.
- NEXT: cm.py (consonant measures against the reference songs), step-1 table from existing takes, engine options.
- 16:02 tools: cm.py (consonants by class against phoneme times: length, level, hf noise, closure dip, release rise,
  modulation spectrum; reference in ref.json), lab.py (rows.jsonl: flow + voiceq + cm + WER tiny/base of a take through
  a chain / the returns), hc.py + sets.py (harness takes in runs/TAG), lead6.py (the lead in the song with a dial
  setting), ds_backend6.py + run6.py (dial in the song mix), chain5.py (stages lift, trans added), dsinfer.py options
  cons_cap_share, slow_cons, slow_cons_min, squeeze (burst / loud / peak), cons_gain.
- FIRST FINDINGS (seed 1, no recognisers yet; stop = p t k; dip = quietest 10 ms re median vowel; rise = largest
  15 ms rise above 2 kHz; reference: stop 116 ms, dip -38, rise 13; unaspirated stops dip -20):
  verse1: demo2 dry stop 52 ms dip -31 rise 9.1, sib hf -10 | +caps: 35 ms, rise 1.7, sib hf -15 | +cons floor:
  voiced stops med -21 -> -8, dip -34 -> -11 | +breaths: nothing lost.
  fourth demo's take through the chain: stop dip -22 (dry) -> -19 rider -> -13 comp1 -> -12 full chain -> -8 with
  the room; voiced stop dip -11.6 -> -4; modulation 4-8 Hz -7 -> -12.4 chain -> -14.5 room; lvl_sd 7.4 -> 4.5.
  chorus: caps cut stops 116 -> 58 ms, rise 18 -> 3-4; without caps (floor + breaths) the DRY chorus already sits at
  the reference's flow numbers (0.82 holes/s, 8.1 %, run 675, lvl_sd 6.5).
- 16:12 cm.py: burst / rise now only on released stops (a vowel follows); rows.jsonl restarted. Lever takes (seed 1,
  verse1 + chorus1, no recognisers yet), each alone on the fourth demo's lead settings ("flow"):
  share35 (caps bind only where consonants take > 35 % of the note): chorus 1.01 holes/s, 7.7 %, run 680, lvl_sd 6.43,
    stops 93 ms dip -34.7 burst -22.9 rise 21.5, sib 116 ms hf -11.4 = the reference's numbers; verse run 330 -> 290.
  cap50 (stop 50 ms, fric 70): verse stop dip -25, sib hf -11.8, run 290; chorus unchanged (keep rule binds).
  burst squeeze (stops sung 90 ms in slow motion, cut around the release): closure dip -22 -> -31 (verse), -31 -> -38
    (chorus), chorus rise 4 -> 17; burst level unchanged; verse voiced run 330 -> 220.
  peak squeeze: +2 dB on bursts, nothing else. slow1 (consonants not slowed): worse (sib hf -17, burst -30). REJECT.
  no cons floor / lower floor: with caps the voiced stops' median level does not move (-6); holes/s 2.29 -> 2.94.
  cons_gain (stop +6, fric +3 above 2 kHz after the vocoder): burst -29 -> -28 / -27 -> -24, sib hf +3.
  ROOM (on the uncapped take through a lighter chain): send 0.50 fills a 116 ms stop closure to -17 dB re the vowel
  (dry -41), send 0.25 -21, send 0.15 -23; an in-channel ducked room fills more than it saves (REJECT). The lines stay
  joined (phrase 23 s) at every send.
- 16:14 dial v0 in config.json (B: longer caps + share 0.35 + lower floors, chain rider 5 dB x 0.7, leveller 1.4:1,
  no consonant leveller, send 0.25; C: unvoiced consonants uncapped, burst squeeze, floors 14 / 20, no leveller,
  trans + lift, send 0.15). Lead clips v0 rendering (tmp/v0.sh, v0a.sh); then tmp/s1wer.sh = recogniser rows of step 1.
- 16:18 FIRST CLIPS (dial v0) in out/: chorus_lead_{A,B,C}, verse_lead_{A,B,C} (wav + mp3, -17.0 LUFS each; the
  lead with its own reverb / delay returns). v0 numbers in rows.jsonl (set "lead"): verse C was back at the second
  demo's stutter (3.44 holes/s, 24.5 %, run 170) -> C v1 gets caps again (50 ms stops); chorus bursts in B / C were
  7-12 dB over the reference (-14 / -12 against -23.7) -> the consonant leveller comes back as a limiter only.
- 16:24 step-1 recogniser rows done (tmp/s1wer.log; `table.py "demo2 dry" "demo2 +" "flow dry" "flow -" "flow +"
  "flow chain" --mean`). Seed-1 means over verse1 / chorus1 / verse2, WER tiny / base:
  demo2 dry .38 / .21 | +caps .39 / .27 | +cons floor .31 / .21 | +breaths .37 / .18 | flow dry (all three) .43 / .18
  | flow -caps .35 / .19 | flow -cons floor .47 / .25 | flow +rider .44 / .24 | +comp1 .48 / .26 | full chain
  .44 / .24 | chain + room (send 0.50) .58 / .37. base.en answers the fourth demo's verse lead with its room in the
  song with "[Music]" (WER 1.00).
  The backing stack on the fourth demo's stems (cm.py, no recognisers): chorus stop closure -16.2 dB lead alone,
  -15.3 with the doubles, -12.0 with the harmony, -5.4 in the full mix; the verse has no stack.
- dial v1 in config.json (ver "v1"); harness takes runs/dial_B_v1, dial_C_v1 rendering. NEXT: 3 seeds of the final
  B and C (hc.py dial_X_VER --seeds=1,2,3; lab.py dial:X --seeds=1,2,3), final lead clips (lead6.py), mixes
  (run6.py clar/mix_B 33 48 --dial B ...), mkclips6.py, table in this file.
- 16:32 dial B v3 / C v2 set (config.json), final lead clips + four mixes rendered (tmp/final.sh, 16:42): all ten
  clips in out/. B: caps 45 / 65 ms + share 0.35, voiced-stop floor 14, chain rider 5 dB x 0.6 + leveller 1.25:1 +
  consonant limiter, send 0.25. C: caps 50 / 80 + share 0.40, floors 16 / 11, burst squeeze, no leveller, comp2 2:1,
  trans, limiter, send 0.15.
- 16:44 FOUND: the fourth demo's verse mix (mix5/out/07) was rendered at a -13 LUFS target (mix5/batch2.sh); my verse
  mixes were at -11.5. Re-rendering both at -13 (tmp/final2.sh). Also C gets a consonant lift (+3 dB above 2.5 kHz
  on noise frames; its sibilants were 2.5 dB under the reference in the chorus) = C v3 (engine as v2).
  mask.py: in the chorus the lead's stop bursts sit AT the level of everything else above 4 kHz (-2.6 dB A, -0.6 B),
  sibilants +8.6 (A) / +5.9 (B) over it: in the full mix consonant level matters, not only length.
  stack.py (B): the doubles fill the lead's chorus stop closures from -35.0 to -27.5 dB, the harmony to -23.1.
- queue now: tmp/levers.sh (recogniser rows of the step-2 levers) -> tmp/final2.sh -> tmp/eval.sh (three seeds of B
  and C on the harness cases). NEXT: tables into this file, final report.
- 16:54 FOUND: chain4's noise detector (periodicity under 1.5 kHz, 48 ms frames) reads most sibilants between two
  vowels as sung (noise share 0.0-0.15 on four of seven chorus sibilants), so the periodicity-gated lift did nothing
  for them. `lift` got a spectral-tilt detector (level above 4 kHz over the level under 1.5 kHz, held 30 ms) with a
  ceiling. Dial B v4 (+2 dB) / C v4 (+2.5 dB); tmp/final3.sh re-rendered the B and C lead clips and all four mixes
  (verse mixes at the -13 LUFS target). Done 17:00. In the chorus mix B's sibilants now stand +7.7 dB over the rest
  above 4 kHz (A +8.6; before the lift +5.9), C's +5.4.
- 17:01 tmp/eval.sh: three seeds of B and C on the harness cases (engine tags dial_B_v3, dial_C_v2 = the engines of
  B v4 / C v4), rows `B v4 chain`, `B v4 room`, `C v4 chain`, `C v4 room`. NOTES.md is assembled by tmp/mknotes.sh
  from tmp/notes_*.md and the tables printed from rows.jsonl.
- note: mix5's build (build5.install) rewrites mix5/song/backing.json and vocals.s2s on every run; run6.py / lead6.py
  call it, so those two files carry today's time stamp with the content the build always gives them. Nothing else
  outside clar/ was written.
- crude click check on the lead channels (largest sample step, isolated 1 ms frames 12 dB over both neighbours):
  nothing in the chorus (0 frames in A, B, C); verse A 2, B 3, C 8 such frames (stop releases or small clicks: not
  told apart).
- 17:10 tmp/eval.sh done: three seeds, lead with returns, WER tiny / base means A 0.67 / 0.39, B 0.42 / 0.16,
  C 0.35 / 0.20 (target, the second demo's dry lead: 0.34 / 0.21). NOTES.md assembled. DONE: config.json (dial B v4,
  C v4), ten clips in out/, tables above. Superseded harness takes (runs/dial_B_v1, dial_C_v1) deleted.
