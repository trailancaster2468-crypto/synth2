# G1: ad-libs as their own clean sounds (final report)

The engineer's final report, followed by the running notes. Paths point into a scratch folder.

## Summary

Ad-libs are now sung as separate, identical hits on their own tracks, and all five before/after clip pairs are made. One spot is not solved: in bar 52 (three voices, "click" x4 on eighth notes) no recogniser hears "click" in the new version of any voice, and the aliens' clicks there were heard before. Nobody has listened; every decision rests on envelopes, spectra, pitch tracks and three recognisers.

**What the score held.** 149 of the 860 notes are ad-libs (sound words, the laugh, interjections, stutters, shouts, chants). In 15 places two different jobs were glued into one phrase on one track. The bar-71 "laugh" on the lead is ハイハイ ("yeah, yeah"), not a laugh; the only written laugh is the aliens' "ha ha ha" in bar 66. "blocked" (bar 8) was glued to verse 1's first word.

**What the third demo's engine did to them.** It sang each as one legato phrase:
- Lead stutter (bar 93): the level between "re-re-re" fell only 1.2-1.4 dB, so there was no second or third onset, after a 279 ms breath.
- Alien laugh: the three "ha" had vowels of 70 / 81 / 186 ms and levels 3.5 dB apart.
- Crowd "Click! Click!": the first of each pair had a 267 ms onset, the second 58 ms.
- Pitch was always on the way to the neighbouring note (up to 1.9 semitones at note starts).

**Engine.** A copy of the flow engine has a new `adlib` option. A hit is one word, sung alone between rests with a designed pitch (flat, fall or rise), no breath, and nothing of its neighbours. Each distinct sound is sung once and stamped at every repetition. Stamped repeats are the same samples; inside the line they differed by 1.9-4.2 dB in spectrum and up to 4.6 dB in level.

**Arrangement.** Nine written tracks with one job each: AI, AI Answer, AI Adlib, Aliens, Alien Adlib, Alien Chant, Alien Chant 2 (was Aliens 2), Crowd, Crowd Shout. Crowd Shout 2 and 3 are derived. Every note of the repo's score is still there with the same timing and words. The seeds of all existing derived and split channels are pinned, so their takes do not change.

**Found along the way.**
- With the stutter on its own track, the lead's "reboot" phrase began with a breath lying 10.9 dB under the stutter. The lead now takes no breath where its own ad-lib sounds (-73 dB there).
- The build's "Aliens Hook" split ended one syllable early (beat 500 instead of 501); fixed in the build wrapper and the patch.

No repo file was changed; the patch is for the lead to apply.

## Measurements run

All commands run as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I ...`; S5 = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/sep. Load average while rendering was 1.5 to 3.8, so no timing is reported.

- **Repo suite:** `python3 -I -m pytest -q -x -p no:cacheprovider` gave 509 passed, 18 skipped in 40 s. I changed no repo file; `git status --short` is empty.
- **Score check:** `S5/check_score.py` finds all 860 notes of the repo's score in `S5/vocals.s2s` with the same onset, length, pitch, lyric and attributes, each on the track `S5/arrange.py` MOVES names; 0 differences.
- **Patch:** `git apply --check S5/repo.patch` passes (4 files under examples/select_all_invaders: vocals.s2s, vocals.py, build.py, SPEC.md). The patched example's `build_session()` gives 30 tracks and 1276 notes, the same (onset, length, pitch) multiset as the unpatched 20 tracks, and no seed of an existing channel changes. A render through the patched repo build with the formant engine was not run.
- **Hit measurements:** `S5/runs.sh TAG A B before|after` renders the vocals of bars A..B with stems and the engine's alignment; `S5/summ.py --json S5/summ.json` gives the table for 10 spots in three ways (inside the line, detached, detached and stamped). Bars covered: 5-8, 49-52, 65-68, 69-72, 93-96.
- **Recognisers** (tiny.en, base.en, small.en; the run as a whole and each hit cut out alone): `S5/asr_all.sh` for the line / detached / first stamped version, `S5/asr_final.json` for the final version, `S5/asr_click.json` for the onset experiment on "click".
- **Whole song, ad-lib tracks only** (`runs.sh full_adlib 1 135 after --tracks ...`): 204 hits from 128 takes, no failure. This was made before the last two engine changes and not repeated.
- **With the band:** `S5/run5.py S5/mix5 49 53` builds and renders the post-chorus with all 30 vocal channels, no error.
- **Not repeated on the final engine:** the detached-without-stamping renders (`runs/*_det`).

## Numbers

Reproduce with `python3 -I S5/summ.py` (hit table) and the recogniser JSON files; S5 = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/sep. Frames are 11.6 ms.

**Inventory** (`S5/inventory.py`): 860 notes = lead line 481, other voices' lines 126, unison 13, backing hook 48, answers 43, ad-libs 149. Fifteen places where two jobs were one phrase.

**Inside the line (third demo's engine)**

| Spot | How it was sung | Between the hits | Repeats |
|---|---|---|---|
| Lead ハイハイ, bar 71 | `AP:26 h:6 a:16 i:11 h:4 a:15 i:16 AP:11`, breaths of 302 and 128 ms joined on | level falls 1.6-5.0 dB | spectrum 3.1 dB, match 0.87 |
| Alien "ha ha ha" | vowels 70 / 81 / 186 ms; "type" bends +1.07 semitones into it | valleys -13 / -18 dB | level 3.5 dB apart, spectrum 3.3 dB, match 0.72 |
| Lead "re-re-re" | after a 279 ms breath | falls 1.3 dB, rises 0.9 dB in 20 ms | spectrum 3.3 dB |
| Crowd click pairs | onsets 267 ms and 58 ms | — | level up to 4.6 dB apart, spectrum 4.2 dB, match 0.45 |
| Chant, bars 5-6 | one 42-phoneme phrase; ダ starts +1.86 semitones high | — | spectrum 1.9 dB, match 0.81 |

**Final version (detached and stamped)**

| Spot | Waveform match of repeats | Spectrum difference | Valley between hits |
|---|---|---|---|
| Lead / alien / crowd click x4 | 1.0 | 0.2-0.5 dB | -39 to -53 dB |
| Crowd click pairs | 0.997 | 1.5 dB | -29 dB |
| Alien "ha" | 0.999 | 0.2 dB | -17 dB (the h fills it) |
| Lead "re" | 1.0 | 0.6 dB | -33.5 dB (rise 30 dB in 20 ms) |
| Chant | 0.997-1.0 | 0.5-0.8 dB | -25 to -26 dB |

- Each "ha" falls from +0.15 to -3.15 semitones.
- Detached but not stamped (first onset setting): spectrum 1.5-2.9 dB, level 0.5-2.6 dB apart, match 0.79-0.96.
- Chant evenness (level sd of the eighth-note syllables), before the glottal start was added: line 2.85 dB, per-syllable stamped 1.66 dB, whole chant as one detached take 2.87 dB. Time under median-25 dB: 1.3 % line, 0.1 % stamped.
- Glottal start on bare ウ: 2.4-5.9 dB louder; ン 2.5 dB quieter, so it is on for vowels only.

**Recognisers: "click" found in the run by tiny.en / base.en / small.en**

| Spot (hits) | In the line | Final |
|---|---|---|
| Crowd pairs, bars 49-51 (6) | 5 / 3 / 6 | 6 / 6 / 0 (small.en writes "Clike") |
| Alien x4, bar 52 | 0 / 3 / 4 | 0 / 0 / 0 ("tech tech") |
| Lead x4, bar 52 | 0 / 0 / 0 ("Let's let's") | 0 / 0 / 0 ("Sit sit") |
| Crowd x4, bar 52 | 0 / 0 / 0 ("Clack clack") | 0 / 0 / 0 ("Check check") |

- Crowd pairs with onsets held to 85 ms (first setting): 0 / 0 / 0 ("tic tic tic").
- Crowd pairs cut out alone: 1 / 6 / 3 in the line, 1 / 2 / 1 final.
- Fresh takes of the first setting gave "Click click click click" (small.en) for the lead and crowd x4; the stamped take of the same setting gave "Check".
- ハイハイ: 2 / 2 / 2 in the line, 2 / 1 / 2 final.
- "ha ha ha": small.en "(laughs)" in the line, "hahahahaha" final; base.en "*laughs*" final.

**Lead under its own stutter:** -31.0 dB before the breath fix (stutter at -20.1), -73.4 dB after.

**With the band, bars 49-53:** -11.5 LUFS, true peak -1.02 dBTP, limiter max 4.24 dB / mean 0.49 dB. There is no third-demo figure for these bars.

**Clips:** five pairs at -18.0 LUFS (clicks -18.7), pair difference at most 0.1 LU, true peaks -1.3 to -3.2 dBFS.

## Recommendations

- Listen to the five pairs in /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/sep/out before adopting anything: nothing here was judged by ear. The pair named laugh_* is the lead's bar-71 ハイハイ ("yeah, yeah"); alien_laugh_* is the only real laugh.
- Adopt the arrangement by applying /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/sep/repo.patch; for the neural build use sep/build5.py (install() on prod/build4), sep/ds_backend5.py and sep/run5.py OUT A B.
- Per kind: sing sound words, the laugh, interjections, stutters and shouts as stamped hits (one word = one hit); sing the kana chant per syllable with the fan chop locked to each syllable and a glottal start on bare vowels; keep short answers in line mode without breaths. All options are in sep/config.json under "adlib".
- Keep the stutter in the voice of the word it belongs to (AI Adlib Bridge: lead's gender, centre, bridge lead's level) and set the other lead ad-libs apart (AI Adlib: gender +0.3, left, -3 dB, drier).
- Keep free-standing hits on the duration model's own onset lengths. Holding the onset to 85 ms turned the crowd's "click" into "tick" / "kick" for all three recognisers.
- Before the next demo, audition takes for the stamped sounds: render a few values of the engine's `salt` option per distinct sound and keep the one the recognisers score best. A stamped take repeats its luck; this decides most for "click".
- For bar 52, try a floor of about 46 ms on an onset l in a touching run (the line gave the aliens k:2 l:4 ih:5 k:5; the stamped take has l:2). If that does not help, consider singing that one bar's click run as a single detached take per voice (option split: "run").
- Keep the seed pinning (sep/vocals.py SEEDS, the "seed" field of SPLITS in the patched build.py). Without it, adding a written track re-rolls every derived and split channel's take, the lead's verse and chorus included.

## Open issues

- Bar 52 (click x4 on eighth notes, three voices): no recogniser hears "click" in the final version. The aliens' run was heard inside the line (base.en 3, small.en 4 of 4) and is not heard now; the lead's and the crowd's were not heard before either.
- Free-standing clicks cut out alone score lower than before (1 / 2 / 1 of 6 against 1 / 6 / 3), although the run as a whole is heard as "click" by tiny.en and base.en.
- Nobody has listened. Voice settings and faders of the new tracks (gender, pan, level, send) are first values chosen without listening.
- The recogniser test is weak for single syllables of 0.2 s, and English recognisers cannot score the stutter or the kana chant; those rest on acoustic measures only.
- The detached-without-stamping renders and the whole-song check of the ad-lib tracks (204 hits, 128 takes) predate the last two engine changes (onsets of free-standing hits, glottal start of the chant) and were not repeated.
- The limiter reached 4.24 dB in the post-chorus render with the band; there is no figure of the third demo for these bars, so it is unknown whether the stamped clicks of several voices landing together raised it.
- After the split, the onset of the lead's "try" (AI Answer) overlaps the last click of AI Adlib in bar 52; the two are different tracks. Not judged.
- "wow" (bar 59), "cool" (31), "hold on" (67) and "funny" (43) are interjections at the head of the lead's own lines and stayed on the lead.
- The `adlib` mode exists only in the scratch engine copy (sep/dsinfer.py). With the repo's formant engine the patch only moves notes to their own tracks, and no render through the patched repo build was run.
- A removal of the old clips in sep/out was refused by the shell's safety check and not done; the exports overwrote the files of the same names, so out/ holds the ten current clips as wav and mp3 plus clips.json.

---

# Running notes

# G1 notes: ad-libs as their own clean sounds

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad, this folder = $S/sep.
Everything runs as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/sep/...`.
The working log of the session (times, what was tried in which order) is NOTES_log.md.

## 0. Result in short

- The score has 860 notes; 149 of them are ad-libs (sound words, the laugh, interjections, stutters, shouts, chants)
  and 15 times an ad-lib and a line (or two ad-lib kinds) were glued into one phrase on one track. The engine sang
  each of those as one legato phrase: no onset of their own for the second and later repeats, pitch on the way to
  the neighbour, breaths joined on, every repeat a different sound (section 2).
- New engine option `adlib` (sep/dsinfer.py `render_adlib`): a track of ad-libs is sung hit by hit. A hit is one
  word, sung alone between rests with a designed pitch, no breath, nothing of its neighbours; every distinct sound
  is sung once and stamped at each repetition (section 3).
- New arrangement (sep/vocals.s2s, sep/vocals.py, sep/build5.py; sep/repo.patch for the repo's files): nine written
  tracks with one job each; the lead track holds only the lead's lines (section 4).
- Not solved: in the one bar where clicks follow each other on eighth notes (bar 52, "everybody clicks") no
  recogniser hears "click" reliably, before or after. A stamped take repeats whatever its one take is, so a weak
  take is weak every time: takes should be auditioned (option `salt`), which was not done (section 5).

## 1. Inventory (inventory.py -> inventory.md; kinds per lyrics.md: `[ROLE]` = that voice's line, `(ROLE: ...)` = a call)

Found while reading: the lead's bar 71 "ハ イ ハ イ" is not a laugh. It is ハイハイ ("yeah, yeah", the snooze button of
pre-chorus 2). The only written laugh is the aliens' "ha ha ha" (bar 66). The clip pair `laugh_*` is that bar 71.
Words the first filter caught that are not ad-libs: "pop" (pop up), "no" (no squares found), "error", "oh" (four oh
four), "ow" (the O of U-F-O); "wow" (bar 59) is the first word of the lead's line and stays on the lead (open
question in section 5).

| bars | written on | text | notes | kind | goes to |
|---|---|---|---|---|---|
| 3.1-5.1 | AI | booting up that's spam good morning valued users | 12 | lead line | stays |
| 8.4-9.1 | AI | blocked | 1 | answer (the lead calling in another voice's section) | AI Answer |
| 9.1-20.3 | AI | three in the morning there's a saucer parked across the law... | 84 | lead line | stays |
| 21.1-24.4.5 | AI | pop up on a pop up ninety thousand pages in their face I wa... | 30 | lead line | stays |
| 27.1-28.2.5 | AI | peace not found error four oh four | 8 | lead line | stays |
| 29.1-30.3 | AI | security question mother's maiden name | 11 | lead line | stays |
| 31.1-34.2.5 | AI | cool to continue please verify you're human select all inva... | 18 | lead line | stays |
| 34.4-36.3 | AI | [eh v] [r iy] square with a [y uw] [eh f] [ow] | 8 | lead line | stays |
| 36.4.5-38.2.5 | AI | select all invaders | 6 | lead line | stays |
| 38.4-40.4 | AI | does the corner of the saucer count | 9 | lead line | stays |
| 41.1-42.2.5 | AI | prove you're not a robot | 6 | lead line | stays |
| 43.1-43.3 | AI | funny | 2 | lead line | stays |
| 43.4-46.2.5 | AI | neither can I select all invaders | 10 | lead line | stays |
| 46.4-48.3 | AI | have a nice apocalypse | 7 | lead line | stays |
| 49.3-50.1 | AI | accept all | 3 | answer (the lead calling in another voice's section) | AI Answer |
| 52.1-52.3 | AI | click click click click | 4 | ad-lib: sound word | AI Adlib |
| 52.3-53.1 | AI | try again | 3 | answer (the lead calling in another voice's section) | AI Answer |
| 54.4-57.1 | AI | that's me I also run your toasters please hold we'll save y... | 15 | lead line | stays |
| 59.1-61.1 | AI | wow nobody's ever read them not even me that's nice | 14 | lead line | stays |
| 63.1-65.1 | AI | have you tried deleting all the photos of your face | 13 | lead line | stays |
| 67.1-69.1 | AI | hold on that's exactly how my valued users are | 13 | lead line | stays |
| 71.1-71.3 | AI | ハ イ ハ イ | 4 | ad-lib: interjection (ハイハイ, "yeah, yeah") | AI Adlib |
| 71.3.5-72.3 | AI | remind me tomorrow | 6 | lead line | stays |
| 75.1-78.2.5 | AI | human sure then verify you're human select all invaders | 16 | lead line | stays |
| 78.4-80.1.5 | AI | [eh v] [r iy] square in pajamas | 7 | lead line | stays |
| 80.4.5-82.2.5 | AI | select all invaders | 6 | lead line | stays |
| 82.4-84.4 | AI | does the guy who never blinks count | 8 | lead line | stays |
| 85.1-86.2.5 | AI | prove you're not a robot | 6 | lead line | stays |
| 87.1-87.3 | AI | funny | 2 | lead line | stays |
| 87.4-90.2.5 | AI | neither can they select all invaders | 10 | lead line | stays |
| 90.4-92.3 | AI | have a nice apocalypse | 7 | lead line | stays |
| 93.1-93.2.5 | AI | [r iy] [r iy] [r iy] | 3 | ad-lib: stutter (phoneme-literal) | AI Adlib |
| 93.2.5-93.4 | AI | reboot | 2 | lead line | stays |
| 95.3.5-96.4 | AI | your spaceship off and on again | 8 | lead line (cuts into the aliens' line, then in unison) | stays |
| 97.1-99.1 | AI | patch notes earth is now in beta known issue everyone | 14 | lead line | stays |
| 101.1-101.4 | AI | you're the invader | 5 | lead line | stays |
| 103.1-106.1 | AI | who is who and who is you and who is human who select all h... | 18 | lead line | stays |
| 107.1-112.3.5 | AI | no squares found no squares found select all me loading hum... | 26 | lead line | stays |
| 112.4.5-114.2.5 | AI | select all invaders | 6 | lead line | stays |
| 114.4-116.1 | AI | I selected [eh v] [r iy] square | 7 | lead line | stays |
| 116.4.5-118.2.5 | AI | select all invaders | 6 | lead line | stays |
| 118.4-120.4 | AI | even the one that's just a mirror | 9 | lead line | stays |
| 121.1-122.2.5 | AI | prove you're not a robot | 6 | lead line | stays |
| 123.1-123.3 | AI | funny | 2 | lead line | stays |
| 123.4-126.2.5 | AI | neither can you select all invaders | 10 | lead line | stays |
| 126.4-130.3 | AI | so I ticked the box for you thank you for choosing earth pl... | 20 | lead line | stays |
| 133.1-134.1 | AI | sorry your war auto renews | 8 | lead line | stays |
| 1.2-3.1 | Aliens | congratulations earth you have been selected | 12 | line (the aliens' own) | stays |
| 5.1-8.4 | Aliens | ワ レ ワ レ ワ ウ チュ ウ ジ ン ダ ワ ワ ワ レ ワ レ ワ ウ チュ ウ ジ ン ダ | 24 | ad-lib: chant | Alien Chant |
| 20.3-21.1 | Aliens | cookies delicious | 5 | answer | stays |
| 25.1-26.4.5 | Aliens | we come in peace please take us to your leader | 11 | line (the aliens' own) | stays |
| 30.3-31.1 | Aliens | the mothership | 4 | answer | stays |
| 50.3-50.4 | Aliens | ouch | 1 | ad-lib: interjection | Alien Adlib |
| 50.4-51.1 | Aliens | thank you | 2 | answer | stays |
| 51.3-52.1 | Aliens | did we pass | 3 | answer | stays |
| 52.1-52.3 | Aliens | click click click click | 4 | ad-lib: sound word | Alien Adlib |
| 53.1-54.4 | Aliens | we demand to speak to the manager of the human race | 15 | line (the aliens' own) | stays |
| 57.1-59.1 | Aliens | we read your terms of service ninety thousand pages twice | 14 | line (the aliens' own) | stays |
| 61.1-63.1 | Aliens | your forced update ate our hard drive now we're out of space | 13 | line (the aliens' own) | stays |
| 65.1-66.3 | Aliens | so we'll blend in we wear pajamas and we type | 12 | line (the aliens' own) | stays |
| 66.3-66.4.5 | Aliens | ha ha ha | 3 | ad-lib: laugh | Alien Adlib |
| 69.1-70.3 | Aliens | your planet will restart in five minutes | 10 | line (the aliens' own) | stays |
| 72.3-73.1 | Aliens | four minutes | 3 | answer | stays |
| 73.1-73.4 | Aliens | ワ レ ワ レ ワ | 5 | ad-lib: chant | Alien Chant |
| 74.1-74.4 | Aliens | ニ ン ゲ ン ダ | 5 | ad-lib: chant | Alien Chant |
| 86.2.5-87.1 | Aliens | we are not a robot | 6 | answer | stays |
| 93.4-94.1.5 | Aliens | [r iy] [r iy] [r iy] | 3 | ad-lib: stutter (phoneme-literal) | Alien Adlib |
| 94.1.5-94.4 | Aliens | reply all | 3 | answer | stays |
| 95.1-96.4 | Aliens | have you tried turning your planet off and on again | 13 | line (the aliens' own) | stays |
| 99.4-100.1 | Aliens | ナ ン デ | 3 | ad-lib: interjection (ナンデ, "why?") | Alien Adlib |
| 100.4-101.1 | Aliens | ナ ン デ | 3 | ad-lib: interjection (ナンデ, "why?") | Alien Adlib |
| 101.4-102.1.5 | Aliens | no you are | 3 | answer | stays |
| 102.2-102.4 | Aliens | respectfully | 4 | answer | stays |
| 103.1-105.1 | Aliens | who is who and who is you and who is human who | 13 | line in unison with the lead | stays |
| 112.4.5-114.2.5 | Aliens | select all invaders | 6 | backing: the hook two octaves under the lead | stays |
| 116.4.5-118.2.5 | Aliens | select all invaders | 6 | backing: the hook two octaves under the lead | stays |
| 118.3.5-118.4 | Aliens | click | 1 | ad-lib: sound word | Alien Adlib |
| 121.1-122.2.5 | Aliens | prove you're not a robot | 6 | backing: the hook two octaves under the lead | stays |
| 124.4.5-126.2 | Aliens | select all invaders | 6 | backing: the hook two octaves under the lead | stays |
| 126.2-126.4 | Aliens | ワ レ ワ レ ワ | 5 | ad-lib: chant | Alien Chant |
| 131.1-132.4 | Aliens | one star would invade again unsubscribe | 10 | line (the aliens' own) | stays |
| 5.1-8.4 | Aliens 2 | ワ レ ワ レ ワ ウ チュ ウ ジ ン ダ ワ ワ ワ レ ワ レ ワ ウ チュ ウ ジ ン ダ | 24 | ad-lib: chant | Alien Chant 2 |
| 73.1-73.4 | Aliens 2 | ワ レ ワ レ ワ | 5 | ad-lib: chant | Alien Chant 2 |
| 74.1-74.4 | Aliens 2 | ニ ン ゲ ン ダ | 5 | ad-lib: chant | Alien Chant 2 |
| 126.2-126.4 | Aliens 2 | ワ レ ワ レ ワ | 5 | ad-lib: chant | Alien Chant 2 |
| 131.3-133.1 | Aliens 2 | ワ レ ワ レ ワ ウ チュ ウ ジ ン | 10 | ad-lib: chant | Alien Chant 2 |
| 12.4.5-13.1.5 | Crowd | help | 1 | ad-lib: shout | Crowd Shout |
| 28.3-29.1 | Crowd | four oh four | 3 | answer (echo) | stays |
| 34.3-34.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 36.3-36.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 38.3-38.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 42.3-42.3.75 | Crowd | hey | 1 | ad-lib: shout | Crowd Shout |
| 46.3-46.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 49.1-49.2.75 | Crowd | click click | 2 | ad-lib: sound word | Crowd Shout |
| 50.1-50.2.75 | Crowd | click click | 2 | ad-lib: sound word | Crowd Shout |
| 51.1-51.2.75 | Crowd | click click | 2 | ad-lib: sound word | Crowd Shout |
| 52.1-52.3 | Crowd | click click click click | 4 | ad-lib: sound word | Crowd Shout |
| 70.3.5-70.4.5 | Crowd | what | 1 | ad-lib: shout | Crowd Shout |
| 74.4-75.1 | Crowd | huh | 1 | ad-lib: shout | Crowd Shout |
| 78.3-78.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 80.3-80.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 82.3-82.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 90.3-90.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 99.1-99.4 | Crowd | who is the invader | 6 | line (the crowd's own) | stays |
| 100.1-100.4 | Crowd | who is the invader | 6 | line (the crowd's own) | stays |
| 106.1-107.1 | Crowd | here here over here | 5 | ad-lib: shout | Crowd Shout |
| 112.3.5-112.4.5 | Crowd | hurry | 2 | answer (whispered) | stays |
| 112.4.5-114.2.5 | Crowd | select all invaders | 6 | backing: the hook an octave under the lead | stays |
| 114.3-114.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 116.3-116.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 116.4.5-118.2.5 | Crowd | select all invaders | 6 | backing: the hook an octave under the lead | stays |
| 118.3-118.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 121.1-122.2.5 | Crowd | prove you're not a robot | 6 | backing: the hook an octave under the lead | stays |
| 122.3-122.3.75 | Crowd | hey | 1 | ad-lib: shout | Crowd Shout |
| 124.4.5-126.2.5 | Crowd | select all invaders | 6 | backing: the hook an octave under the lead | stays |
| 126.3-126.3.75 | Crowd | click | 1 | ad-lib: sound word | Crowd Shout |
| 130.3.5-131.1 | Crowd | five stars | 2 | answer | stays |

Two jobs in one phrase (notes of different kinds less than 0.30 s apart on one track, so the engine sang them as one phrase):

- AI, bar 9.1: "blocked" (answer (the lead calling in another voice's section)) -> "three in the morning there's" (lead line), gap 0 ms
- AI, bar 52.3: "click click click click" (ad-lib: sound word) -> "try again" (answer (the lead calling in another voice's section)), gap 0 ms
- AI, bar 71.3.5: "ハ イ ハ イ" (ad-lib: interjection (ハイハイ, "yeah, yeah")) -> "remind me tomorrow" (lead line), gap 179 ms
- AI, bar 93.2.5: "[r iy] [r iy] [r iy]" (ad-lib: stutter (phoneme-literal)) -> "reboot" (lead line), gap 0 ms
- Aliens, bar 50.4: "ouch" (ad-lib: interjection) -> "thank you" (answer), gap 0 ms
- Aliens, bar 52.1: "did we pass" (answer) -> "click click click click" (ad-lib: sound word), gap 0 ms
- Aliens, bar 66.3: " we wear pajamas and we type" (line (the aliens' own)) -> "ha ha ha" (ad-lib: laugh), gap 0 ms
- Aliens, bar 73.1: "four minutes" (answer) -> "ワ レ ワ レ ワ" (ad-lib: chant), gap 0 ms
- Aliens, bar 94.1.5: "[r iy] [r iy] [r iy]" (ad-lib: stutter (phoneme-literal)) -> "reply all" (answer), gap 0 ms
- Aliens, bar 126.2: "select all invaders" (backing: the hook two octaves under the lead) -> "ワ レ ワ レ ワ" (ad-lib: chant), gap 0 ms
- Crowd, bar 112.4.5: "hurry" (answer (whispered)) -> "select all invaders" (backing: the hook an octave under the lead), gap 0 ms
- Crowd, bar 114.3: "select all invaders" (backing: the hook an octave under the lead) -> "click" (ad-lib: sound word), gap 179 ms
- Crowd, bar 116.4.5: "click" (ad-lib: sound word) -> "select all invaders" (backing: the hook an octave under the lead), gap 268 ms
- Crowd, bar 118.3: "select all invaders" (backing: the hook an octave under the lead) -> "click" (ad-lib: sound word), gap 179 ms
- Crowd, bar 122.3: "prove you're not a robot" (backing: the hook an octave under the lead) -> "hey" (ad-lib: shout), gap 179 ms
- Crowd, bar 126.3: "select all invaders" (backing: the hook an octave under the lead) -> "click" (ad-lib: sound word), gap 179 ms

Notes by kind: ad-lib 149, answer 47, backing 48, lead line 481, line 122, line in unison with the lead 13

Also: the build's split "Aliens Hook" ends at beat 500, so the last syllable of the aliens' final "invaders" (beat
500.5) stayed on the main Aliens channel, glued to the chant. build5 / the patch end the split at 501.

## Step 2: what the third demo's engine did to the ad-libs (runs/*_before, meas.py; frames are 11.6 ms)

"fed" = the pitch the engine fed the vocoder against the written note (semitones) at the note's start / end;
"valley" = lowest level between two hits against the hits' peaks; "match" = best normalised cross-correlation of
two repeats' waveforms (1 = same samples); "spectrum" = RMS difference of the repeats' band spectra (13 bands).

| spot | how it was sung | glides | joins / breaths | repeats of one syllable |
|---|---|---|---|---|
| lead bar 71 ハイハイ (not a laugh: "yeah, yeah") + "remind me tomorrow" | one phrase of 22 phonemes: `AP:26 h:6 a:16 i:11 h:4 a:15 i:16 AP:11 r:4 iy:10 ...` | ハ ends -0.81, イ starts +0.72, second ハ starts -0.53 (always on the way to the next note) | a 302 ms breath in front, a 128 ms breath joined on behind; level between the four syllables falls only 1.6 to 5.0 dB | ハ/ハ: lengths differ 64 ms, spectrum 3.6 dB, match 0.53; only the first has an onset |
| aliens bar 66 "... and we type ha ha ha" | the laugh is the end of a 38-phoneme phrase: `ay:7 p:4 hh:5 aa:6 hh:9 aa:7 hh:8 aa:16` | "type" bends +1.07 up into the first "ha" | no rest before the laugh (the p of "type" runs into the h) | vowels 70 / 81 / 186 ms, levels -25.2 / -25.6 / -22.1 dB (3.5 dB apart), spectrum 3.3 dB (max 4.6), match 0.72; valleys -13 / -18 dB |
| lead bar 52 click x4 + "try again" | one phrase of 24 phonemes: `k:6 l:4 ih:7 k:3 k:3 l:3 ih:7 k:3 k:2 l:3 ih:7 k:3 k:2 l:4 ih:5 k:3 t:3 r:4 ay:15` | 0.2 to 0.6 | the last k of one click and the first k of the next are one double stop of 58-70 ms; the fourth click runs into "try" | level 1.2 dB apart, lengths 72 ms, spectrum 3.5 dB (max 4.5), match 0.40 |
| crowd bars 49-51 "click! click!" | first of each pair `k:12 l:11 ih:12 k:11` (onset 267 ms), second `k:2 l:3 ih:8 k:15` (onset 58 ms) | none | the 89 ms rest between the two is 3 frames of SP | level up to 4.6 dB apart, lengths 94-122 ms, spectrum 3.5 to 5.2 dB, match 0.19 to 0.40 |
| aliens bar 5 chant ワレワレワ ウチュウジンダ | one phrase of 42 phonemes over two bars: `w:4 a:28 r:3 e:10 w:5 a:11 r:5 e:10 w:5 a:46 u:6 ch:10 u:15 u:19 j:12 i:15 N:12 d:4 a:21 w:9 ...` | ダ starts +1.86 (coming down from E3 to A2), ン ends -1.28 | ダ runs into the next bar's ワ; the first ウ gets 70 ms of its 179 ms | レ/レ level 2.0 dB apart, spectrum 1.9 dB, match 0.81; vowels of equal eighths 116-128 ms |
| lead bar 93 "re-re-re-reboot" | `AP:24 r:7 iy:11 r:5 iy:11 r:4 iy:11 b:4 uw:27 t:4` | "re(boot)" bends +1.0 toward F5 | a 279 ms breath in front; between the stutters the level falls 1.2-1.4 dB and rises 1.0 dB in 20 ms: there is no second or third onset | spectrum 3.3 dB (max 4.2), match 0.83 |
| aliens bar 93 "re-re-re-reply all" | `r:7 iy:5 r:10 iy:7 r:8 iy:7 r:9 ih:5 p:6 l:4 ay:16 ...` | "reply" starts +3.14 high | vowels of 58-81 ms between r's of 93-116 ms | level 1.7 dB apart, lengths 115 ms, spectrum 3.1 dB, match 0.89 |

## 3. Singing an ad-lib as an ad-lib (sep/dsinfer.py, options in `ADLIB`, per track in config.json "adlib")

Three ways, same notes: (a) inside the line as the third demo sang it (runs/*_before), (b) detached: every hit its
own take with rests around it, designed pitch, no breath (runs/*_det, `stamp: false`), (c) as (b) with one take per
distinct sound stamped at every repetition (runs/*_after). `summ.py` prints the table, `asr_spot.py` the
recognisers. "own onset" = hits before which the level falls 20 dB under the hit's peak; "rise" = the largest
level rise within 20 ms at the later hits of a run; "valley" = lowest level between two hits against their peaks.

| spot | way | hits with an onset of their own | rise at the later hits, dB in 20 ms | valley dB | repeats: level dB / spectrum dB / waveform match | sung (frames of 11.6 ms) |
|---|---|---|---|---|---|---|
| lead ハイハイ b71 | (a) in the line | 1 of 4 | 4.0 | -1.7 | 0.5 / 3.08 / 0.87 | `AP:26 h:6 a:16 i:11 h:4 a:15` |
| lead ハイハイ b71 | (b) detached | 2 of 2 | 17.5 | -5.8 | 1.1 / 2.39 / 0.962 | `` |
| lead ハイハイ b71 | (c) detached + stamped | 2 of 2 | 20.0 | -7.7 | 0.3 / 0.83 / 1.0 | `` |
| alien ha ha ha b66 | (a) in the line | 1 of 3 | 11.1 | -15.8 | 3.5 / 3.29 / 0.718 | `aa:6 hh:9 aa:7 hh:8 aa:16 SP:10` |
| alien ha ha ha b66 | (b) detached | 1 of 3 | 10.4 | -20.2 | 1.4 / 1.52 / 0.85 | `SP:8 hh:5 aa:21 SP:10` |
| alien ha ha ha b66 | (c) detached + stamped | 1 of 3 | 9.4 | -17.4 | 0.2 / 0.21 / 0.999 | `SP:8 hh:6 aa:21 SP:10` |
| lead click x4 b52 | (a) in the line | 1 of 4 | 19.4 | -20.3 | 1.2 / 3.47 / 0.707 | `k:3 k:3 l:3 ih:7 k:3 k:2 l:3 ih:7 k:3` |
| lead click x4 b52 | (b) detached | 4 of 4 | 35.8 | -46.9 | 1.3 / 2.43 / 0.955 | `SP:8 k:3 l:2 ih:5 k:5 SP:10` |
| lead click x4 b52 | (c) detached + stamped | 1 of 4 | 32.2 | -48.1 | 0.0 / 0.38 / 1.0 | `SP:8 k:2 l:2 ih:6 k:5 SP:10` |
| alien click x4 b52 | (a) in the line | 4 of 4 | 30.3 | -39.5 | 2.7 / 3.11 / 0.785 | `k:5 k:2 l:4 ih:5 k:4 k:2 l:4 ih:5 k:5` |
| alien click x4 b52 | (b) detached | 4 of 4 | 24.0 | -45.5 | 0.9 / 2.08 / 0.927 | `SP:8 k:3 l:2 ih:5 k:5 SP:10` |
| alien click x4 b52 | (c) detached + stamped | 4 of 4 | 19.3 | -38.7 | 0.5 / 0.18 / 1.0 | `SP:8 k:2 l:2 ih:5 k:6 SP:10` |
| crowd click pairs b49-51 | (a) in the line | 6 of 6 | 24.3 | -37.4 | 4.6 / 4.22 / 0.453 | `k:2 l:3 ih:8 k:15 SP:10` |
| crowd click pairs b49-51 | (b) detached | 6 of 6 | 31.0 | -50.5 | 1.9 / 2.28 / 0.94 | `SP:8 k:3 l:2 ih:8 k:15 SP:10` |
| crowd click pairs b49-51 | (c) detached + stamped | 6 of 6 | 17.5 | -28.9 | 0.4 / 1.52 / 0.997 | `SP:8 k:15 l:11 ih:9 k:14 SP:10` |
| crowd click x4 b52 | (a) in the line | 4 of 4 | 16.7 | -28.0 | 2.8 / 2.83 / 0.559 | `k:2 l:4 ih:4 k:5 k:3 l:4 ih:5 k:5` |
| crowd click x4 b52 | (b) detached | 4 of 4 | 27.1 | -42.4 | 1.9 / 1.66 / 0.929 | `SP:8 k:3 l:2 ih:5 k:5 SP:10` |
| crowd click x4 b52 | (c) detached + stamped | 4 of 4 | 29.0 | -52.7 | 0.1 / 0.47 / 1.0 | `SP:8 k:2 l:2 ih:5 k:6 SP:10` |
| lead re-re-re b93 | (a) in the line | 1 of 3 | 0.9 | -1.3 | 0.1 / 3.28 / 0.828 | `r:7 iy:11 r:5 iy:11 r:4 iy:11` |
| lead re-re-re b93 | (b) detached | 1 of 3 | 35.0 | -29.9 | 2.1 / 2.92 / 0.964 | `SP:8 r:4 iy:21 SP:10` |
| lead re-re-re b93 | (c) detached + stamped | 1 of 3 | 30.2 | -33.5 | 0.2 / 0.61 / 1.0 | `SP:8 r:5 iy:21 SP:10` |
| alien re-re-re b93 | (a) in the line | 3 of 3 | 17.3 | -17.3 | 1.7 / 3.12 / 0.891 | `iy:5 r:10 iy:7 r:8 iy:7 r:9` |
| alien re-re-re b93 | (b) detached | 2 of 3 | 26.1 | -29.8 | 2.6 / 1.6 / 0.948 | `SP:8 r:5 iy:21 SP:10` |
| alien re-re-re b93 | (c) detached + stamped | 3 of 3 | 26.0 | -31.0 | 0.0 / 0.22 / 1.0 | `SP:8 r:5 iy:21 SP:10` |
| chant b5-6 | (a) in the line | 4 of 11 | 17.0 | -23.6 | 2.0 / 1.86 / 0.808 | `a:28 r:3 e:10 w:5 a:11 r:5` |
| chant b5-6 | (b) detached | 5 of 11 | 23.1 | -21.6 | 0.5 / 1.55 / 0.785 | `SP:8 r:4 e:21 SP:10` |
| chant b5-6 | (c) detached + stamped | 4 of 11 | 25.3 | -26.2 | 1.6 / 0.81 / 0.997 | `SP:8 r:5 e:21 SP:10` |
| chant b5-6 | the chant as one detached take | 6 of 11 | 18.9 | -26.1 | 0.9 / 2.83 / 0.786 | `` |
| chant b7-8 | (a) in the line | 5 of 13 | 12.3 | -20.4 | 0.5 / 2.57 / 0.839 | `w:9 a:10 w:6 a:10 w:5 a:11` |
| chant b7-8 | (b) detached | 7 of 13 | 27.5 | -25.1 | 0.2 / 2.05 / 0.786 | `SP:8 w:3 a:21 SP:10` |
| chant b7-8 | (c) detached + stamped | 3 of 13 | 24.4 | -25.2 | 2.8 / 0.45 / 1.0 | `SP:8 w:4 a:21 SP:10` |
| chant b7-8 | the chant as one detached take | 8 of 13 | 16.0 | -24.2 | 0.1 / 4.01 / 0.61 | `` |

Column (b) was rendered with the first onset setting (onsets held to 85 ms); (c) is the final engine. "level" of the
repeats is measured over the written note and includes the next hit's onset; the stamped copies are the same samples.

Recognisers (asr_spot.py; "click" etc. found in the run by tiny.en / base.en / small.en ; hits heard when cut out alone):

| spot (hits) | (a) in the line | (b) detached, first onset setting | (c) stamped, first onset setting | (c) final |
|---|---|---|---|---|
| lead click x4, bar 52 (eighth notes) (4) | 0 / 0 / 0 ; 0 / 0 / 0 | 0 / 0 / 4 ; 0 / 0 / 1 | 0 / 0 / 0 ; 0 / 0 / 0 | 0 / 0 / 0 ; 0 / 0 / 0 |
| alien click x4, bar 52 (4) | 0 / 3 / 4 ; 4 / 3 / 4 | 0 / 0 / 0 ; 2 / 1 / 0 | 0 / 0 / 0 ; 0 / 1 / 1 | 0 / 0 / 0 ; 0 / 0 / 0 |
| crowd click x4, bar 52 (4) | 0 / 0 / 0 ; 0 / 0 / 1 | 0 / 0 / 4 ; 1 / 1 / 1 | 0 / 0 / 0 ; 0 / 1 / 1 | 0 / 0 / 0 ; 0 / 0 / 0 |
| crowd "click! click!" pairs, bars 49-51 (6) | 5 / 3 / 6 ; 1 / 6 / 3 | 0 / 0 / 0 ; 0 / 1 / 0 | 0 / 0 / 0 ; 0 / 1 / 0 | 6 / 6 / 0 ; 1 / 2 / 1 |
| alien ha ha ha (3) | 0 / 0 / 0 ; 1 / 1 / 1 | 0 / 0 / 3 ; 0 / 0 / 1 | 0 / 0 / 3 ; 0 / 0 / 1 | 0 / 0 / 0 ; 0 / 0 / 1 |
| lead ハイ ハイ (2) | 2 / 2 / 2 ; 2 / 1 / 1 | 2 / 2 / 2 ; 1 / 2 / 2 | 2 / 2 / 2 ; 1 / 1 / 2 | 2 / 1 / 2 ; 1 / 2 / 2 |

What a take is (the points that needed measuring):

- Length and cut. The acoustic model lets the last 60 to 80 ms before a rest die away: the イ of the first ハイ came
  out 18 dB under the ハ. A hit that ends in a vowel is therefore sung 70 ms past its written end and cut with a
  30 ms fade; a hit that touches the next one ends where that hit's onset consonants start (first version: its own
  onset length, which left a 58 ms hole between チュ and ウ), 20 ms earlier after a vowel. The last hit of a run of
  equal hits is cut like the others, or it is a different sound (the fourth click was).
- Onsets. Free-standing hits keep the duration model's onset lengths. With the onset held to 85 ms (first version)
  the crowd's "Click! Click!" was `k:3 l:2 ih:8 k:15` and all three recognisers heard "tic tic tic" / "kick kick";
  with the model's own onset they hear "Click click, click click, click click" (run test: tiny.en 31, base.en 8,
  small.en 6 "click" found in 6 hits; inside the line before: 5 / 3 / 6). In a touching run (eighth notes) the
  onset must fit: at most 0.4 of the note, a stop at most 25 ms, so that "click" is `k:2 l:4 ih:5 k:4`, the
  lengths the line gave it.
- Pitch. Flat on the written note from before the first consonant to the end (measured f0 against the note at the
  start / end of the lead's stutter: +0.04 / +0.04; inside the line the fed pitch moved up to 1.0 semitone toward
  the next note). "fall": the aliens' "ha" and "ouch" drop 3 semitones from 30 % of the hit to where it stops
  sounding (measured +0.15 -> -3.15). "rise": the crowd's "what" and "huh". A word over several notes (ハイ,
  ナンデ, "over") steps with a 40 ms glide. The aliens' ad-libs and chants keep their 11 Hz warble, phase 0 at
  the note, so copies stay identical; the track's pitchDelta curve is added ("Click?", bar 80, is its own take).
- ハ is sung `h a` (ja/h, 46-58 ms of h before the note); the three recognisers hear "Hi, hi" / "Hi hi" before and
  after. "click" has both stops in every version (`k l ih k`).
- Chant syllables that start with a vowel: with a closure (ja/cl, 3 frames) before it the bare ウ is 2.4 to 5.9 dB
  louder (-23.0 / -22.9 / -21.6 / -20.8 dB against -26.0 / -28.8 / -23.0 / -25.4) and rises 17.5-21.7 dB in 20 ms
  against 10.9-14.4; the syllabic ン got 2.5 dB quieter with it. `glottal` is on for the chant, for vowels only.
- Fan chop. The 11 Hz / -9 dB chop of the chant is applied to every take, locked to its start (`chop`); the
  backend then applies only the upper envelope of the track's loudness lane (the outro fade survives, the chop is
  not doubled).
- Noise between the hits: before the first phoneme and after the release a take is digital silence.

Decision per kind:

| kind | where | decision | why (numbers above) |
|---|---|---|---|
| sound word ("click") | AI Adlib, Alien Adlib, Crowd Shout | (c) stamped, flat pitch | repeats 2.8-4.2 dB spectrum / up to 4.6 dB level apart inside the line, identical stamped; every click has its own onset |
| laugh ("ha ha ha") | Alien Adlib | (c) stamped, each "ha" falls 3 semitones | lengths 70 / 81 / 186 ms and 3.5 dB apart inside the line; final take: small.en "hahahahaha", base.en "*laughs*" (the word count of the table misses both); before: "(laughs)" / "It's a fafafafa..." |
| interjection (ハイハイ, "ouch", ナンデ) | AI Adlib, Alien Adlib | (c), one hit per word (ハイ + ハイ, not four syllables) | the line sang ハイハイ as one word between two breaths |
| stutter ("re-re-re-") | AI Adlib Bridge, Alien Adlib Bridge (the voice of the word that follows) | (c) stamped | inside the line the level between the stutters fell 1.3 dB: no second or third onset; stamped 20-30 dB |
| shout ("help", "hey", "what", "huh", "here") | Crowd Shout (+ 2, 3) | (c), flat ("what", "huh" rise); the three crowd voices each stamp their own take | recognised alone after: "Help", "Hey", "What?", "Huh?", "Here" (base.en) |
| chant (ワレワレハ ...) | Alien Chant, Alien Chant 2 | (c) per syllable, even, chop locked to the syllable | level of the eighth-note syllables sd 1.66 dB against 2.85 (line) and 2.87 (the whole chant as one detached take); no more time in holes than the line (0.1 % under median-25 dB against 1.3 %) |
| answer (a short line) | AI Answer; the aliens' and the crowd's stay on their tracks | line mode, the lead's options without breaths | it is a phrase, not a hit |

## 4. The new arrangement

Written tracks (sep/vocals.s2s; check_score.py: every one of the repo's 860 notes is there with the same onset,
length, pitch, lyric and attributes, on the track arrange.MOVES names; ハ イ became the word ハイ, ナ ン デ the
word ナンデ):

| track | notes | job | voice setting (ds_backend5.CAST) | fader / pan / send (build5.TRACK_MIX5) |
|---|---|---|---|---|
| AI | 481 | the lead's lines, nothing else | gender 0, lead options, chain | as before (0 dB, centre) |
| AI Answer | 7 | the lead's calls in other voices' sections: "blocked" (bar 8), "accept all" (49), "try again" (52) | gender +0.2, lead options without breaths, lead chain | -2 dB, +0.15, 0.50 |
| AI Adlib | 11 | click x4 (52), ハイ ハイ (71), re- re- re- (93) | gender +0.3, ad-lib mode, the lead chain's EQ only | -3 dB, -0.20, 0.20 |
| Aliens | 180 | the aliens' lines and answers | as before | as before |
| Alien Adlib | 18 | ouch (50), click x4 (52), ha ha ha (66), re- re- re- (93), ナンデ x2 (99, 100), click (118) | gender -0.75 (lines: -0.6), tone 12, ad-lib mode, warble | +1.5 dB, +0.50, 0.20 |
| Alien Chant | 39 | the kana chants of the first alien | gender -0.6, tone 12, ad-lib mode per syllable, chop | +1.5 dB, +0.30, 0.30 |
| Alien Chant 2 | 49 | the second alien (was "Aliens 2") | gender -1.0, its octave double, ad-lib mode per syllable, chop | +0.5 dB, -0.35, 0.40 |
| Crowd | 43 | the crowd's sung lines and answers | as before | as before |
| Crowd Shout | 32 | help, click x22, hey x2, what, huh, here here over here | as Crowd (gender -0.3, tone 12, its double), ad-lib mode | -2 dB, -0.70, 0.25 |

Derived by sep/vocals.py `apply_vocal_production`: AI Double, Harmony, Crowd 2, Crowd 3 (unchanged) and new Crowd
Shout 2, Crowd Shout 3. Split channels added by build5 (and the patch): AI Adlib Bridge (the stutter: the lead's
own voice, gender 0, centre, the bridge lead's level), Alien Adlib Verse and Alien Adlib Bridge (the aliens' voice
and level of those sections). Seeds: a new track's default seed comes from its position in the project, so five
more written tracks would have re-rolled every derived and split channel. vocals.py and build5 / the patched
build.py pin the seeds those channels had in the third demo (seeds_before.json); verified: no existing channel's
seed changes, the patched example builds 30 tracks with the same 1276 notes.

The lead no longer breathes under its own ad-libs: with the stutter on its own track the lead's phrase "reboot"
started with a 279 ms breath that lay under the stutter, 10.9 dB under it. `breath["avoid"]` (dsinfer.py) and the
backend (intervals of the AI Adlib notes) turn such a breath into a rest.

Files: vocals.s2s, vocals.py (production), arrange.py (the moves as a table, `rearrange(session)`), build5.py
(`install()` into prod/build4), ds_backend5.py, config.json, run5.py (the song with the band), repo.patch
(examples/select_all_invaders: vocals.s2s, vocals.py, build.py, SPEC.md; `git apply --check` passes).

## 5. Open

- Bar 52, "everybody clicks" (three voices, click x4 on eighth notes of 179 ms): the recognisers do not hear "click"
  in the final version of any voice (lead "Sit sit sit sit", aliens "tech tech tech tech", crowd
  "Check, check, check, check, check.", small.en). Inside the line they heard the aliens' ("Click, click, click, click.") and
  neither the lead's ("Let's let's let's let's try") nor the crowd's ("Clack, clack, clack, clack, clack."). The hits are
  now separate and identical, but the l is 23 ms (`k:2 l:2 ih:5 k:6`; the line gave the aliens `k:2 l:4 ih:5 k:5`).
  Not tried: a floor of about 46 ms for an onset l in a touching run; auditioning takes.
- Stamping repeats one take. Fresh takes of the same settings were heard as "Click click click click" by small.en
  for the lead and the crowd (column b), the stamped take of the same settings as "Check, check, ...". The engine
  has `salt` ({lyric: n} gives another take of that sound); a script that renders a few salts of every distinct
  sound and keeps the one the recognisers score best does not exist yet.
- Free-standing "click" (19 of the 31 written clicks: 18 of the crowd's 22 and the aliens' single one; the other
  12 are bar 52): final version heard as "click" in the run test by
  tiny.en and base.en (6 and 6 of 6; small.en writes "Clike"), before 5 / 3 / 6. Cut out alone the hits score
  lower than before (1 / 2 / 1 of 6 against 1 / 6 / 3): alone, base.en and small.en often write "like".
- The recogniser test is weak evidence for single syllables of 0.2 s: the same audio gives "[Music]", "(static)" or
  a word depending on the model. "re-re-re" and the kana chant cannot be scored by English recognisers at all.
- Nobody has listened. Every decision above rests on envelopes, spectra, pitch tracks and three recognisers.
- The detached-without-stamping renders (runs/*_det) and the whole-song check of the ad-lib tracks
  (runs/full_adlib: 204 hits from 128 takes, no failure) were made before the last two engine changes (onsets of
  free-standing hits, the glottal start of the chant); they were not repeated. The five clip pairs, the table's
  column (c) and the "final" recogniser column are from the final engine.
- "wow" (bar 59), "cool" (31), "hold on" (67), "funny" (43) are interjections at the head of the lead's own lines
  and stayed on the lead. Moving them is one line each in arrange.MOVES and the score.
- The voice settings and faders of the new tracks (gender, pan, level, send) are first values chosen without
  listening; the mix of main against backing is not this study's subject.
- The `adlib` mode exists only in the engine copy sep/dsinfer.py. With the repo's formant engine the patch only
  moves the notes to their own tracks.

## 6. Clips, checks, how to reproduce

Clips in $S/sep/out (wav 24 bit + mp3 160k; each pair at one loudness, -18.0 LUFS, clicks -18.7; all vocals of the
bars, dry backend output post-fader and panned, no channel strip, no reverb, no band; out/clips.json has the levels):

| pair | bars | what to listen for |
|---|---|---|
| laugh_before / laugh_after | 69-72 | the lead's ハイハイ (bar 71; "yeah, yeah", not a laugh) before "remind me tomorrow" |
| alien_laugh_before / _after | 65-68 | "... and we type ha ha ha" |
| clicks_before / _after | 49-52 | the crowd's "Click! Click!" pairs, then bar 52 where all three voices click on eighth notes |
| chant_before / _after | 5-8 | ワレワレワ ウチュウジンダ, twice (and "blocked", now on AI Answer) |
| line_with_adlib_before / _after | 93-96 | "re-re-re-reboot" / "re-re-re-reply all", then the IT-script lines |

- With the band: `run5.py $S/sep/mix5 49 53` -> $S/sep/mix5/preview/select_all_invaders_bars_49-53.wav (30 vocal
  channels, no error; -11.5 LUFS, true peak -1.02 dBTP, limiter max 4.24 dB / mean 0.49 dB; no figure of the third
  demo for these bars to compare the limiter with).
- Renders: `runs.sh TAG A B before|after` (vocals of bars A..B with stems and the engine's alignment in runs/TAG),
  `SEP_CAST='{"Track": {"opts": {"adlib": {...}}}}'` overrides a track's ad-lib options for one run.
- Numbers: `summ.py --json summ.json` (table of section 3), `meas.py STEMS TRACK` (every hit), `asr_all.sh`
  (asr_spots.json: a / b / first c), batch8.sh (asr_final.json: final c), `asr_click.log` (the onset experiment:
  va = stop 25 ms in a 0.4 room, vb = the duration model's onsets), `check_score.py`, `inventory.py`.
- Load average while rendering: 1.5 to 3.8 (other engineers on the machine); no timing is reported.
- I wanted to empty sep/out before the last export; the shell refused the removal (a safety check on a relative
  `rm`), so nothing was deleted. The exports overwrote the files of the same names; out/ holds the ten clips (wav and mp3)
  and clips.json.

