# Q7 findings: the ceiling and the ladder

(Working file, appended as results arrive. Second run of this job; the first run left only
`dl/DiffSinger` and `inspect_ds.py`.)

## Log
- 01:03 First ceiling clips: `out/00_ceiling_sajiao_human.{wav,mp3}`, `out/01_ceiling_sajiao_predicted.{wav,mp3}`
  (OpenVPI sample `03_撒娇八连.ds`, 15 s, transpose 0, steps 10, depth 0.6, seed 1; `ceiling.py --song 03 --name sajiao`).
  voiceq: human cpp 22.40 hnr 17.47 lvl_sd 5.67 dip -7.52; predicted cpp 22.40 hnr 18.79 lvl_sd 5.68 dip -6.78.
  Demo the user heard (`$S/ds1`): verse1 cpp 20.78 hnr 18.20 lvl_sd 6.46 dip -10.34; chorus1 20.99 / 21.40 / 5.77 / -7.35;
  plain 24.96 / 20.20 / 4.60 / -6.06.
- Outside yardstick, published DiffSinger demos of other banks (`$S/ref/lunaisite/singers/*/*.mp3`, voiceq):
  "standard" modes cpp 27.0-29.3 (hnr 15-22); deliberately airy modes (soft, flos, delicate, lullaby) cpp 20.0-24.9;
  strong/powerful cpp 26.2-30.6. So cpp ~21-22 sits in the airy range of finished DiffSinger work.
- Pitfall: `yaml.safe_load` of the bank's `dsdict-*.yaml` (1.1-2.2 million lines each) takes minutes. q7lib scans lines
  and caches (`cache_dicts.json`, 1.8 s first time).
- Pitfall: one 15 s phrase through the acoustic model took > 4 min and 2.5 GB under load; .ds segments are cut at rests
  into pieces of at most ~7 s (`q7lib.split_segment`), as OpenUtau does.
- 01:08 Second ceiling song `05_恋人心.ds` (48 s, transpose 0): `out/00_ceiling_lianrenxin_human`, `out/01_ceiling_lianrenxin_predicted`.
  voiceq (score_q7.py -> scores.json): human cpp 24.04 hnr 19.42 lvl_sd 4.95 dip -6.23; predicted cpp 23.21 hnr 20.27
  lvl_sd 5.16 dip -5.76. Sajiao again: human 22.40 / 17.47 / 5.67 / -7.52, predicted 22.40 / 18.79 / 5.68 / -6.78.
  => ceiling cpp 22.4-24.0 so far; the demo's fast sections were at 20.8-21.0, its slow phrase at 25.0.
- Render cost seen (2 threads, load average 9, so noisy): 48.3 s of song in 88 s; the vocoder takes as long as the
  acoustic model (38.6 s vs 39.8 s), variance 9 s, pitch 15 s per 48 s.
- analyze.py (analysis.json) on the ceiling, model inputs and audio: consonant median 81-116 ms, vowel median
  244-313 ms, consonant share 0.22-0.26; vowel onset in the audio -1..-10 ms from the note start (IQR 6-11 ms);
  predicted pitch transitions 54-60 ms (20-80 %), centred 33-46 ms BEFORE the note start; phrase-final notes fade
  over 46-81 ms and reach -20 dB 19-32 ms before the notated note end.
- 01:12 Third ceiling song `00_我多想说再见啊.ds` segments 0-4 (42 s, transpose 0, pitch median 60): human cpp 23.63 hnr 20.03
  lvl_sd 5.84 dip -9.08; predicted cpp 22.83 hnr 20.40 lvl_sd 5.95 dip -10.31.
- Pitch scan on the sajiao ceiling clip, human-driven (`scan/`, `ceiling.py --song 03 --name sajiao --modes human
  --transpose T`): T=-5 (median MIDI 60.8) cpp 22.50 hnr 16.65; T=0 (65.8) cpp 22.40 hnr 17.47; T=+5 (70.8, p95 77.1)
  cpp 21.44 hnr 18.11. The demo's verse sits at median MIDI 72 (p5/p95 69/76), its chorus at 74 (69/77): about 8
  semitones above the median of the sample songs (60-66). At that height the ceiling itself measures cpp ~21.4, the
  demo measured 20.8-21.0. (The formant engine also drops from cpp 28.7-29.7 on plain/twinkle to 27.0-27.4 on
  verse1/chorus1, `$S/base`, so part of the cpp fall with pitch is the measure, not the voice.)
- Test material facts (planner dry run): verse1 114 notes, median note 0.179 s, MIDI 69-76; chorus1 53 notes, median
  0.357 s, MIDI 69-77; plain 29 notes, median 0.5 s, MIDI 61-69. The AI track is `voice=nova style=robotic` and carries
  loudness, breathiness and tension curves; the spike ignores all three in the dry clips.
- The dry clips the user heard are `$S/ds1/{chorus1,verse1,plain}.wav` (lengths match `$S/demo/02..04*.mp3`), i.e. the
  unchanged spike, steps 10, depth 0.6, the track's own style, no gain.
- 01:20 Ladder batch 1 (verse1, native and demo rung, seeds 1-3; `ladder.py --rungs 10,50 --cases verse1 [--seed N]`).
  `50_demo_verse1` reproduces the clip the user heard (`$S/ds1/verse1.wav`): same length, waveform correlation 0.88,
  cpp 20.57 vs 20.78.
  Native verse1: cpp 21.51 / 21.48 / 21.13 (seeds 1/2/3), hnr 17.10 / 17.26 / 17.15, WER tiny 0.86 / 0.49 / 0.53,
  base 0.81 / 0.56 / 0.50. Seed 1's WER is a recogniser loop on the "pop up on a pop up" line (tiny.en prints
  "p-p-p-p-p-..."), not a worse render: read the transcripts in scores.json before trusting one verse1 WER.
  Demo verse1 seed 1: cpp 20.57 hnr 18.26 WER 0.56 / 0.50.
  => on the verse, driving the bank natively changes cpp by about +0.8 and WER by nothing outside the noise.
- phrasing.py (model inputs): ceiling songs sing 2.3-2.7 syllables per second, with 5-25 breath phonemes (AP) per minute
  and rests for 7-17 % of the time. verse1 sings 5.4 syllables per second, no AP at all, rests 8 %.
  analyze.py on native verse1: consonant median 58 ms, vowel median 81 ms, consonants take 49 % of the sung time
  (ceiling 22-26 %), 26 % of the vowels are shorter than 60 ms (ceiling 0-1 %).
- 01:24 Noise floor, verse1, three fresh processes with seeds 1/2/3 (the default seed repeats bit for bit, so
  different seeds are the honest floor): native cpp 21.51/21.48/21.13, hnr 17.10/17.26/17.15, lvl_sd 6.34/6.30/6.27,
  dip -9.57/-10.06/-9.14, WER tiny 0.86/0.49/0.53, base 0.81/0.56/0.50. Demo cpp 20.57/20.68/20.63, hnr
  18.26/18.01/18.32, lvl_sd 6.72/6.42/6.47, dip -10.87/-10.37/-9.71, WER tiny 0.56/1.24/0.69, base 0.50/0.59/0.56.
  => floor: cpp +-0.2, hnr +-0.15, lvl_sd +-0.15, dip +-0.5, verse1 WER base +-0.05 with rare jumps of +0.3 and
  tiny.en WER unusable on verse1 (loops push it over 1.0). Median base WER: native 0.56, demo 0.56.
- Batch 2 (seed 1): native chorus1 cpp 21.13 hnr 20.31 WER 0.49/0.26; native plain cpp 24.91 hnr 20.13 WER 0.17/0.00.

## Provisional conclusions (01:24, before rungs 20-40, half tempo and transposition are in)

1. The ladder is flat. Native (bank durations, bank pitch, bank variance, no gain) and the demo differ by cpp +0.8
   (just outside the floor) and by nothing in WER. How Synth2 drives the bank is not where the quality goes.
2. The ceiling is an airy voice: cpp 22.4-24.0 on three sample songs, against 27-29 for the standard voices of
   other published DiffSinger banks, and 21.4 when the ceiling song is moved up to the demo's pitch range.
3. What separates the demo material from the ceiling material: twice the syllable rate (5.4 vs 2.3-2.7 per second),
   consonants taking half of the sung time (49 % vs 22-26 %), vowels of 81 ms median, a pitch range 8 semitones
   above the sample songs, English instead of Chinese, and no breath phonemes.
Running now: `run_ladder.sh` (log `run_ladder.log`), then `run_extra.sh` (log `run_extra.log`). Scores land in
`scores.json` (make_table.py prints them).
- 01:28 `30_robotic_verse1.wav` is bit for bit `50_demo_verse1.wav` (max abs difference 0.0): q7lib.render_track with
  Synth2 pitch + robotic style is the spike, so for verse1 and chorus1 rung 30 and rung 50 are one rung, and the dry
  demo had no gain (rung 40 goes beyond what the user heard). For plain the track's own style is pop, so rung 50 = rung 20.
- Batch 2, demo rung: chorus1 cpp 20.69 hnr 21.53 WER 0.83/0.43 (the earlier spike run of the same thing scored
  0.29/0.26: chorus1 has 35 words and its WER moves that much between runs); plain cpp 24.53 hnr 20.53 WER 0.08/0.08.
- analyze.py, verse1: pitch transitions 36 ms centred -14 ms (native), 43 ms centred -35 ms (Synth2 pop), 14 ms centred
  -14 ms (robotic); pitch deviation inside notes 15 / 2 / 0 cents. Predicted breathiness and voicing on vowels are the
  same on every rung (-44 dB, -21 dB) and the same as on the ceiling (-44..-48, -22..-25): the variance model does not
  make the demo breathier.
- chorus1 native: 2.8 syllables/s, consonant share 0.32, vowel median 209 ms. plain native: 2.7 syllables/s, share 0.39,
  vowel median 186 ms. Same rate as the ceiling songs; chorus1 sits at MIDI 69-77, plain at 61-69.
- 01:34 Batch 3 (seed 1), cpp / hnr / lvl_sd / dip / WER tiny, base:
  20 synth2pitch: verse1 20.63 / 18.47 / 6.68 / -9.97 / 0.52, 0.50; chorus1 20.59 / 21.16 / 5.63 / -7.37 / 0.51, 0.31;
  plain 24.53 / 20.53 / 4.79 / -6.34 / 0.08, 0.08.
  30 robotic: verse1 20.57 / 18.26 / 6.72 / -10.87 / 0.56, 0.50; chorus1 20.69 / 21.53 / 5.55 / -7.09 / 0.83, 0.43;
  plain 24.50 / 20.85 / 5.23 / -7.01 / 0.04, 0.00.
  40 gain: chorus1 20.72 / 21.57 / 5.48 / -7.03 / 0.80, 0.43; plain 24.58 / 20.86 / 5.24 / -6.94 / 0.04, 0.00.
  => the only step that moves cpp is native -> Synth2 pitch: -0.88 (verse1), -0.54 (chorus1), -0.38 (plain). Pop vs
  robotic pitch and the gain ride change nothing outside the floor (robotic style dynamics are within 1 dB anyway).
- analysis.json table (make_table.py prints it): vowel onset in the audio sits -1..-11 ms from the note start on
  every clip, ceiling and ladder alike. Transitions: human 31-115 ms, bank 36-60 ms, Synth2 pop 42-43 ms, robotic 14 ms.
  Pitch deviation inside notes: human 11-50 cents, bank 9-15, pop 1-7, robotic 0. Vibrato on notes >= 0.45 s: human
  38-76 cents on 67-100 % of them in two of three songs, bank 11-24 cents on 13-40 %, pop 14-16 cents on 7-18 %.
  Phrase-final fade 35-96 ms everywhere (the acoustic model does it). Note level follows note pitch inside phrases on
  every clip (r 0.2-0.6): the acoustic model does that too, without any Synth2 gain.
- 01:41 40 gain verse1: cpp 20.56 hnr 18.26 lvl_sd 6.74 dip -10.82 WER 0.55/0.50 (= rung 30 within the floor).
- Batch 4, half tempo (84 BPM; `ladder.py --rungs 60,61 --cases verse1,chorus1`), cpp / hnr / lvl_sd / dip / WER tiny, base:
  60 native verse1 21.90 / 19.82 / 5.60 / -7.53 / 0.32, 0.24; chorus1 21.73 / 22.31 / 4.88 / -6.07 / 0.49, 0.17.
  61 demo   verse1 21.41 / 20.89 / 5.91 / -8.10 / 0.34, 0.24; chorus1 21.47 / 23.06 / 4.88 / -5.37 / 0.49, 0.29.
  => verse1: half tempo takes base WER from 0.56 (median of 3 seeds, both rungs) to 0.24 on both rungs, hnr +2.6,
  dip +2.5 dB, lvl_sd -0.8, cpp +0.4 (native) / +0.8 (demo). The rung makes no difference at either tempo: the verse is
  too fast for this bank, it is not driven badly. chorus1 (already 2.8 syllables/s at full tempo) does not move outside
  its noise. Half-tempo verse1 native: 2.7 syllables/s, vowel median 221 ms, consonant share 0.36.
- 01:47 Batch 5, same notes 5 semitones lower (MIDI 64-72; `ladder.py --rungs 62,63 --cases verse1,chorus1`):
  62 native verse1 cpp 22.54 hnr 16.52 lvl_sd 6.20 dip -8.66 WER 0.50/0.41; chorus1 22.41 / 19.49 / 5.81 / -8.02, WER 0.43/0.09.
  63 demo chorus1 cpp 21.95 hnr 20.36 lvl_sd 5.86 dip -7.97 WER 0.51/0.17.
  => 5 semitones lower is worth cpp +1.0 (verse1 native) to +1.3 (chorus1, both rungs) and base WER 0.56 -> 0.41 (verse1
  native), 0.26 -> 0.09 (chorus1 native). The song's key costs more clarity than anything on the ladder.
