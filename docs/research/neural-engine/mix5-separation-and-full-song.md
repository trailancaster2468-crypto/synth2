# G2: main against backing, one kind of sound per track, the whole song (final report)

The engineer's final report, followed by the running notes. Paths point into a scratch folder.

## Summary

The fourth demo is made: eight clips, including the whole song. In chorus 1 the lead now stands 8.2 dB over all the other vocals in 1-5 kHz while it sings; the third demo had 5.7 dB. Nobody has listened: every decision rests on levels, spectra, stereo measures and three recognisers. One thing is not solved: the lead's "click" in bar 52 is heard as "click" by no recogniser in any of ten takes.

**Step 0: one kind of sound per track**
- G1's 30 vocal channels still mixed kinds. They are now 47 channels, each with one job.
  - The lead's ad-lib track became AI Click, AI Hai and AI Stutter.
  - The aliens' ad-lib track became Ouch, Click, Laugh, Stutter and Nande.
  - The aliens' line track gave up its answers and the unison with the lead.
  - The crowd's line, answers, whisper, hook, shouts and clicks are separate, each in three voices.
- Every moved note keeps the seed of the channel it came from, and no existing channel's seed changed.
- The band: I listed every stem and split four that carried two kinds each. The chip lead's chorus tags are apart from its riffs, the same for the low chip lead, the bridge's glitch percussion apart from the intro's tick, and the gag effects apart from the transitions. The split changes no loudness, band gain or limiter figure.
- The drum stems still carry several kinds each; that grouping is in repo code.

**The mix**
- The lead stays alone in the centre.
- The double is a pair close around it (±0.28), 9.7 dB under.
- The harmony is on both sides (±0.70), 8.6 dB under, darker and with a small pre-delayed room.
- The lead's answers sit a step right; single-word ad-libs are off-centre, band-limited, with a slap and a throw.
- Click runs and the laugh are off-centre and dry, because a slap would land between eighth-note hits.
- The crowd is wide (±0.85) and far; the chant voices are left and right; backing ducks 1.5 to 3 dB under the lead.
- The aliens' lines got no slap: it cost 1 to 3 of 43 words in the recognisers.

**Found on the way**
- The disk filled six minutes into rendering, because every take was stored as a 33 MB array. Takes are now stored trimmed in 16 bit and stems as FLAC.
- The first whole-song mix was killed for lack of memory. It now runs on one mixer thread and peaks at 13.5 GB of 15.
- After the split, the whispered "hurry" sat 19 to 22 dB under the lead, because a channel of its own is level-matched alone. Its fader is corrected; it is now 9 to 12 dB under.
- Auditioning takes of the stamped "click" paid off for the aliens (take 2 is heard as four clicks by all three recognisers; take 0 by none) and for two crowd voices.

The ad-lib clips cover bars 49-68 (20 bars, not 16), so they hold the click runs and the laugh in bar 66. No repo file was changed.

## Measurements run

All commands run as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I ...`; M = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/mix5. Load average while rendering was 1.0 to 2.0; no timing is reported as a result.

- **Repo suite:** `python3 -I -m pytest -q -x -p no:cacheprovider` gave 509 passed, 18 skipped in 55 s. `git status --short` is empty; I changed no repo file.
- **Build:** `M/tmp/list6.py` builds 47 vocal channels with 1544 notes. That is G1's 1276 plus the second double (159) and the harmony's left copy (62 + 47). No seed of a channel that existed in G1's build changed. `M/inventory6.py` lists what every channel sings.
- **Everything from cached takes:** `M/final.sh` renders the whole song and the three sections, then runs the measures, clips and checks. `M/final2.sh` redoes the whole song and what follows from it; it was run last, after the whisper's fader change.
- **Separation, before and after:** `M/meas_all.sh` writes `M/sep.json`. It measures chorus 33-48, verse 9-24, post-chorus 49-52 and verse 2 53-68, with tiny.en, base.en and small.en on the lead alone and on the sum of all vocals. "Before" is the third demo's stems: prod/mix/flow/preview for chorus and verse 1, and a fresh render of bars 49-68 with the third demo's build and backend (`run5.py --arr before`).
- **Every section of the whole song:** `M/sections.sh` writes `M/sep_full.json` and `M/sections.log`, without recognisers.
- **Nothing silent:** `M/silent_check.py M/full/stems 1 134` checked 1117 words of 47 channels; none is silent.
- **Clip checks:** `M/checks.py M/out/0*.wav M/out/0*.mp3` writes `M/checks.log`: clipping, true peak, loudness, loss in mono, correlation, for all 16 files.
- **Band split:** bars 33-48 rendered with and without it. Loudness, band gain, glue and limiter figures are the same; the band sum differs by -25 dB because the moved part is played with another humanisation.
- **Aliens' slap:** recognisers on the Aliens Verse stem of bars 53-68 with and without it (`MIX5_FX` override, `M/tmp/stem_wer.py`).
- **Click takes:** `M/audition.py 0 1 2 3` for all five click channels, salts 4-9 for the lead, and salts 0-2 with the new `onset_floor` option. Results are in `M/audition.json`.
- **Not done:** no whole-song render of the third demo's build, so sections outside the clips have no "before". Clips 01-07 were not re-exported after the whisper change, which only touches bar 112.

## Numbers

Reproduce with `M/meas_all.sh` (sep.json), `M/sections.sh` (sep_full.json), `M/checks.py`, `M/roles_md.py M/full/stems`, `M/audition.py`; M = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/mix5.

**Lead against the sum of all other vocals, 1-5 kHz, while the lead sings**

| Section | Third demo | Now | In mono (before / now) |
|---|---|---|---|
| Chorus 1 (33-48) | 5.7 dB | 8.2 dB | 6.0 / 8.8 |
| Verse 1 (9-24) | 15.0 | 15.3 | 15.3 / 16.2 |
| Post-chorus (49-52) | 2.7 (lead channel held answers and clicks) | 8.4 (main = AI Answer) | 3.2 / 9.4 |
| Verse 2 (53-68) | 14.8 | 15.7 | 15.0 / 16.4 |

Whole song, now only: chorus 2 7.5, final chorus 8.1, bridge 12.2, pre-chorus 1 8.4, pre-chorus 2 6.6, breakdown 11.9.

**Recognisers (WER tiny.en / base.en / small.en)**

| Section | Lead alone, before | Lead alone, now | Sum of vocals, before | Sum of vocals, now |
|---|---|---|---|---|
| Chorus 1 | 0.40 / 0.54 / 0.26 | 0.37 / 0.54 / 0.14 | 3.37 / 0.74 / 0.37 | 1.14 / 0.71 / 0.17 |
| Verse 1 | 0.49 / 0.32 / 0.25 | 0.44 / 0.33 / 0.28 | 0.65 / 0.38 / 0.52 | 0.53 / 0.39 / 0.33 |
| Verse 2 | 0.31 / 0.17 / 0.10 | 0.31 / 0.14 / 0.10 | 0.36 / 0.21 / 0.17 | 0.36 / 0.21 / 0.15 |

The lead's take is the same before and after, yet its WER alone moves by up to 0.12; differences under that are noise. tiny.en loops on the chorus sum.

**Chorus 1 by group (level vs lead / level in 1-5 kHz / pan / side vs mid / L-R correlation)**

| Group | Third demo | Now |
|---|---|---|
| Double | -6.7 / -7.2 / -0.52 / -11.0 dB / 1.00 | -9.7 / -10.3 / 0.00 / -13.0 dB / 0.90 |
| Harmony | -6.8 / -8.3 / +0.52 / -11.0 dB / 1.00 | -8.6 / -10.1 / +0.14 / -4.4 dB / 0.47 |
| Crowd hits | -0.7 / -5.9 / +0.12 / -7.5 dB / 0.70 | -3.3 / -7.7 / -0.01 / -3.8 dB / 0.42 |

- Lead over the next loudest element in 1-5 kHz (the build's own figure): 7.9 dB (AI Double) before, 12.9 dB (left guitar) now.
- Final chorus: harmony -13.0, aliens' hook -12.0, crowd hook -9.3 dB against a lead that rides 2.5 dB higher. The lead there has holes 0.61 /s (chorus 1: 0.46), voiced run 950 ms, hnr 21.1.
- Vocal stems of bars 49-68 summed: peak +0.9 dBFS in the third demo (bar 52), -2.3 dBFS now.

**Checks**

| Item | Value |
|---|---|
| Whole song | -11.5 LUFS, true peak -1.02 dBTP, 0 clipped, short-term max -8.86 LUFS |
| Limiter, whole song | max 3.42 / mean 0.32 dB |
| Limiter, chorus mix | 1.13 / 0.21 dB (third demo 1.48 / 0.27) |
| Limiter, verse mix | 2.72 / 0.32 dB (third demo 2.79 / 0.33) |
| Limiter, bars 49-68 at -11.5 LUFS | 3.23 / 0.35 dB at the second mix values (third demo 3.37 / 0.40) |
| Lost in mono, mixes | chorus 0.7 LU (third demo 0.7), whole song 0.6, verse 0.3 |
| Backing against lead in mono | loses 0.8 dB in the chorus (third demo 0.4); harmony pair alone 1.3 dB |
| Silent words | 0 of 1117 |

**Clips (wav / mp3 LUFS; true peak wav / mp3)**
- 01 and 02: -13.5 / -14.0 and -13.6 / -14.0; peaks -2.4 / -2.9 and -3.6 / -4.0.
- 03: -11.5 / -12.0; -1.0 / -1.3 (third demo's chorus mix: -11.6 / -12.1).
- 04 and 05: both -16.6 / -17.0; peaks -1.0 / -1.5 and -1.6 / -2.1.
- 06: -12.5 / -12.9; -1.0 / -1.1.
- 07: -13.0 / -13.5; -1.0 / -1.4 (third demo's verse mix: the same).
- 08: -11.6 / -12.1; -1.0 / -1.0.

**Experiments**
- Aliens' slap, verse-2 lines (43 words): without 0.30 / 0.26 / 0.12; with a 125 ms repeat at -14 dB 0.33 / 0.33 / 0.16; third demo 0.40 / 0.37 / 0.16.
- Click audition, "click" found in bar 52 by tiny / base / small:
  - Aliens: take 0 gives 0 / 0 / 0, take 2 gives 4 / 12 (loop) / 4.
  - Crowd voice 1: take 1 gives 0 / 0 / 4 in bar 52 and 9 / 6 / 6 in the pairs (take 0: 7 / 6 / 0).
  - Crowd voice 2: take 0 gives 0 / 4 / 4 in bar 52 and 0 / 0 / 0 in the pairs.
  - Lead: 0 / 0 / 0 in ten takes.
  - With a 46 ms floor on the l, lead and crowd are heard as "twit", "quack", "fight".
- "hurry": 19 to 22 dB under the breakdown's lead before the fader change, 9.3 and 12.1 dB after.
- Resources: whole-song mix peaks at 13.47 GB resident of 15; the take cache went from 2.8 GB to 21 MB in the trimmed 16-bit form.

## Recommendations

- Listen to the clips in /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/mix5/out in order before adopting anything: 01 against 02 for main against backing in the chorus, 04 against 05 for the ad-libs, then 03, 06, 07 and the whole song 08. Nothing here was judged by ear.
- Adopt the 47-channel arrangement and the roles: mix5/build5.py (MOVES6, ROLES, TRACK_MIX6, CROWD_KINDS, OVERRIDES6), mix5/ds_backend5.py (COPIES, ROLE_FX) and mix5/run5.py. The per-channel table is mix5/roles.md.
- Keep slaps off anything that repeats on eighth notes (click runs, the laugh) and off the aliens' lines. Use the slap only on single-word ad-libs (ハイ ハイ, ouch, ナンデ).
- Audition takes for every stamped sound before a demo, as audition.py does for "click". The take chosen for the aliens turned a run no recogniser heard into one all three hear.
- After moving a note to a channel of its own, check its level in roles.md. Such a channel is level-matched alone, which left the whisper "hurry" 19 to 22 dB under the lead until its fader was corrected.
- Render the whole song on one mixer thread (run5.py does this for bars 1-134) and keep takes trimmed and stems as FLAC. With four threads the mix was killed, and full-length float takes filled the disk in six minutes.
- If the last chorus should feel bigger, raise the crowd hook and the aliens' hook there first (Crowd Hook -8.5, Aliens Hook -5.0 in build5); they sit 9 to 12 dB under the lead.
- Keep the band split (build5.split_band). It gives the chip lead's tags, the low tags, the glitch percussion and the gag effects their own faders at no change in sound.

## Open issues

- Nobody has listened. Pans, levels, the slap, the rooms, the band limits and the ducking depths are values chosen from figures.
- The lead's "click" in bar 52 is heard as "click" by no recogniser in ten takes ("sit", "chit", "that's it", "t-t-t-t"). A 46 ms floor on the l did not help. The crowd's second voice is not heard in the pairs of bars 49-51. The aliens' "click?" in bar 118 uses the chosen take 2 but was not auditioned itself.
- The recogniser evidence is weak for this purpose: the same lead take scores up to 0.12 apart between two renders, and single hits of 0.2 s give "(barking)" or a word depending on the model.
- There is no whole-song render of the third demo's build, so the sections outside the clips (pre-choruses, chorus 2, bridge, breakdown, final chorus, outro) have figures only for the new mix.
- Harmony L and the second double are delayed, detuned copies of one take, not second performances. Summed to mono the harmony pair loses 1.3 dB.
- The song has one room that all voices feed in mono. "Further back" is a small mono room inside the channel, panned with the voice. A stereo room per group or a pre-delay per send needs the mixer.
- Pre-chorus 1 and 2 have the lowest lead-against-others figures (8.4 and 6.6 dB), set by the frames where answers and shouts border the lead's lines. Whether that is heard as crowding is not judged.
- In the final chorus the backing is 9 to 13 dB under a lead that rides 2.5 dB higher than in chorus 1; the last chorus may feel smaller than intended. The two whispering "hurry" voices together are about 7.5 dB under the lead, a first value.
- The drum stems still carry several kinds each: snare with clap, rim and toms (all shaped by the snare's EQ); hats with shaker, tambourine and cowbell; crash with ride. The bell keeps its chorus tags and its post-chorus riff on one stem.
- "funny", "cool", "wow" and "hold on" stay in the lead's lines, and the two chant texts stay on one channel per voice; both are judgement calls.
- The vocals-only clips are sums of stems: they carry the in-channel effects but not the song's reverb and delay returns.
- The whole-song mix needs 13.5 GB of the machine's 15 GB. Another memory-heavy job at the same time would get it killed again.
- The engine copy mix5/dsinfer.py has one new ad-lib option (onset_floor, off by default). The backend keys its takes by the hash of sep/'s engine files so the cache stays valid; an edit to sep/dsinfer.py would invalidate it.
- The ad-lib clips are 20 bars (49-68), not the 16 the brief asked for, so that the click runs and the laugh in bar 66 are both in.

---

# Running notes

# mix5 (G2): main against backing, the fourth demo

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad, this folder = $S/mix5.
Run everything as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/mix5/...`.
Nobody has listened to any of this: every decision below rests on levels, spectra, stereo measures and three
recognisers.

## Result in short

- 47 vocal channels, one kind of sound each (G1: 30, third demo: 20), each with a role; the band has four more
  stems (step 0 below).
- Chorus 1: the lead stands 8.2 dB over the sum of all other vocals in 1-5 kHz while it sings (third demo 5.7).
  The double went from one copy at the left (-6.7 dB) to a pair close around the lead (-9.7 dB together), the
  harmony from one voice at the right (-6.8 dB, L/R correlation 1.00) to both sides (-8.6 dB, correlation 0.47).
- Post-chorus: the main voice (the lead's answers) 8.4 dB over the others; in the third demo the lead's channel
  (answers and clicks) stood 2.7 dB over them.
- Verses: unchanged within 1 dB (15.0 -> 15.3 dB verse 1, 14.8 -> 15.7 dB verse 2); there the voices take turns.
- The whole song is rendered (08): -11.5 LUFS, no clipping, true peak -1.0 dBTP, nothing silent, 0.6 LU lost in mono.
- Open: the lead's "click" in bar 52 is heard as "click" by no recogniser in ten takes; see "Still wrong".

## Status log (newest last)

- 13:20 start. Read the briefs, G1's notes (sep/NOTES.md), prod/NOTES.md, build4, the backends.
- 13:27 build5.py (MOVES6: one kind per channel; ROLES; TRACK_MIX6), ds_backend5.py (ROLE_FX, COPIES), run5.py
  written. 47 vocal channels (G1: 30; third demo: 20), 1544 notes = G1's 1276 + the second double (159) and the
  harmony's left copy (62 + 47). No seed of a channel that existed in G1's build changed (tmp/list6.py).
- 13:30 test render bars 51-52 ok (1 m 50 s). batch1.sh started: after 33-48, after 49-68, after 9-24, before 49-68.

- 13:34 chorus v1 measured: lead / other vocals 1-5 kHz 6.1 dB (third demo 5.7). Too little: second values
  (doubles -12.5, harmony -11.5 / -13, presence dip on the harmony, crowd clicks -5).
- 13:36 DISK FULL (the volume had 12 KB left): the backend stored every take as a float32 array from time 0
  (33 MB each), 2.8 GB in six minutes; three of four batch renders died. Fixed: takes stored as their sounding
  span in 16 bit (2.8 GB -> 21 MB), stems as FLAC without the silent ones. Free now 3.2 GB (shared).
- 13:40 batch2.sh: after 49-68, 33-48, 9-24, before 49-68.
- Band (step 0): MIX5_BAND=split puts the chip lead's chorus hook tags, the low chip lead's tags, the bridge's
  glitch percussion and the gag effects on stems of their own (build5.split_band); off until compared.

- 13:49 before 49-68 rendered (third demo's build and backend). 13:56 FIRST SET of clips in out/ (01-07), 36 minutes
  after the start; measures with the recognisers in sep.json (meas_all.sh), parts of the ad-lib section in
  sep_parts.json.
- 13:57 band split compared on bars 33-48 (same loudness / band gain / limiter; band sum differs by -25 dB: the
  moved part is played with another humanisation) and switched on. 13:58 full.sh: the whole song is rendering.

- 14:05 v3 values: click runs and the laugh get no slap (a repeat 95-110 ms later lands between eighth-note hits);
  harmony +1.5 dB (it was 7.0 dB under the band in 1-5 kHz; third demo 4.2), crowd clicks / shouts +1.5 / +1 dB,
  AI Click 0 dB (9.1 dB under the band).
- 14:15 the whole-song render was KILLED in the mix for lack of memory (47 vocal channels + 21 band stems of 191 s,
  four mixer threads; all takes were sung and cached before). run5.py now mixes the whole song on one thread and
  drops the singer's models before the mix; a watcher writes the resident memory to full.rss. 14:17 started again.

- 14:23 WHOLE SONG rendered (v3): -11.5 LUFS, true peak -1.02 dBTP (renderer; ffmpeg reads -0.9), limiter max 3.41 /
  mean 0.32 dB, 0 clipped samples, 1117 words checked and none silent, peak memory 13.45 GB of 15.
- 14:30 the aliens' slap tested with the recognisers on their verse-2 lines (43 words): without 0.30 / 0.26 / 0.12,
  with (-14 dB, 125 ms) 0.33 / 0.33 / 0.16, third demo 0.40 / 0.37 / 0.16. Slap removed from the aliens' lines and
  answers (kept on the single-word ad-libs). Crowd shouts -1.5 dB (they were +1.3 dB over the lead in pre-chorus 2).
- 14:35-14:46 audition.py: takes of the stamped "click" (engine option salt). Aliens: take 2 is heard as
  "click click click click" by all three recognisers in bar 52 (take 0: "tack" / "tech"). Crowd voice 1: take 1
  (pairs 9 / 6 / 6 of 6, bar 52 small.en 4 of 4; take 0: 7 / 6 / 0 and none). Crowd voice 3: take 1. Crowd voice 2
  and the lead keep take 0. The lead's click is heard as "click" in none of ten takes ("sit", "chit", "that's it",
  "t-t-t-t"). A floor of 46 ms on the l (new ad-lib option onset_floor in mix5/dsinfer.py, off by default) gave
  "twit" / "quack" / "fight" for the lead and the crowd and no gain: not used.
- 14:47 final.sh started: whole song, the three sections, measures, clips, checks (all from cached takes).
- 14:51 whole song (v4) done; 15:00 measures with the recognisers and the eight clips; 15:04 sections, role table,
  checks. Repo suite 509 passed, 18 skipped; no repo file changed.
- 15:05 FOUND in the role table: "hurry" (Crowd Whisper) sat 19-22 dB under the breakdown's lead. On its own channel
  the whisper is level-matched to the formant engine's whisper, which is quiet; on the hook's channel it had been
  matched with the hook. Fader -7 -> +3; whole song rendered again (final2.sh).
- 15:15 DONE: whole song (final values) -11.5 LUFS, true peak -1.02 dBTP, limiter max 3.42 / mean 0.32 dB, 0 clipped,
  1117 words none silent, peak memory 13.47 GB; "hurry" now 9.3 / 12.1 dB under the lead; clip 08 exported again.
  Clips 01-07 are from 15:00 (bar 112 is in none of them).

## Step 0: one kind of sound per track (build5.MOVES6, inventory6.py -> inventory6.md)

G1's arrangement had 30 channels; these still carried more than one kind and were split (the takes keep their
seeds, so every ad-lib hit is the take G1 measured):

| was (G1) | kinds on it | now |
|---|---|---|
| AI Adlib (+ Bridge) | sound word (click x4, bar 52), interjection (ハイ ハイ, bar 71), stutter (re- re- re-, bar 93) | AI Click, AI Hai, AI Stutter |
| Alien Adlib (+ Verse, Bridge) | interjection "ouch" (50), sound word click (52, 118), laugh (66), stutter (93), ナンデ (99, 100) | Alien Ouch, Alien Click, Alien Laugh, Alien Stutter, Alien Nande |
| Aliens (+ Bridge) | their own lines; short answers to the lead (cookies? delicious / the mothership / thank you / did we pass / four minutes / we are not a robot / reply all / no, you are / respectfully); the unison with the lead (103-104) | Aliens (+ Verse, Bridge): lines only; Alien Answer (+ Bridge); Alien Unison |
| Crowd, Crowd 2, Crowd 3 | their line "who is the invader"; the echo "four oh four" and the answer "five stars" | Crowd x3: the line; Crowd Answer x3 |
| Crowd Hook x3 | the hook under the lead; the whisper "hurry" (112) | Crowd Hook x3; Crowd Whisper x2 (the third voice has no whisper) |
| Crowd Shout x3 | the sound word "click" (22 hits); shouted words (help, hey x2, what, huh, here here over here) | Crowd Click x3; Crowd Shout x3 |

Added as copies in the backend: AI Double R (second double), Harmony L, Harmony Final L (the harmony's other side).
Now 47 vocal channels, 21 jobs. Left as they are, with the reason:
- AI (lead) channels: only the lead's lines. "funny" (43, 87, 123), "cool" (31), "wow" (59), "hold on" (67) are the
  first words of the lead's own sentences and are sung in line with them (G1 left them too).
- Alien Chant / Alien Chant 2: two texts (ワレワレワ ウチュウジンダ; ワレワレワ ニンゲンダ at 73-74) on one channel per voice:
  the same chant figure, sung per syllable from the same stamped syllables. One kind.
- Crowd Shout: five different shouted words, each one or two hits, already separate takes that never overlap.
- AI Answer: three calls of the same kind (blocked / accept all / try again).

The band (backing.json: 18 instruments -> stems):

| stem | what it carries | verdict |
|---|---|---|
| drums kick | kick | one kind |
| drums snare | snare, clap (choruses, post-chorus, chaos), rim (verses, breakdown), tom fills (bars 8, 24, 52, 68 ...) | FOUR kinds on one stem; its EQ (-6.5 dB shelf from 5.5 kHz, 3 kHz dip, vocal carve) was set for the snare and also shapes the claps, rims and toms. Not split: the grouping is in synth2/arranger/drums.py DRUM_GROUPS (repo code); needs "clap", "rim" and "toms" groups there |
| drums hats | closed hat, open hat, shaker, tambourine (final chorus), cowbell (bridge) | several kinds; same reason, not split |
| drums cymbals | crash, ride (bridge) | two kinds; not split |
| chip (a second kit) | the intro's tick pattern; the bridge's glitch percussion | split: chip (tick), glitch |
| bass / slap / growl / sub | pick bass / verse slap bass / bridge growl / breakdown sub | one kind each |
| gtrL, gtrR | power chords (eighths, ringing chords in the pre-chorus, kime hits) | one instrument, one job |
| mute | palm mutes | one kind |
| saws, pad | held chords | one kind each |
| lead (chip lead) | the riffs (intro, post-chorus calls, outro) and the chorus hook tags (answers at the lead's phrase ends) | split: lead (riffs), tags (hook tags) |
| leadlo | the riffs an octave lower; the final chorus' hook tags | split: leadlo, tagslo |
| bell | chorus hook tags and the post-chorus call riff, both answers of two to four notes | left |
| arp, pluck, mbox | arpeggio / verse plucks / breakdown music box | one kind each |
| fx | 17 kinds of one-shots: transitions (riser, reverse cymbal, impact, downlifter, whoosh, sweep) and gags of the story (radio, coin, bleeps, ufo, zap, error, laser, alarm, glitch, modem, powerdown) | split: fx (transitions), sfx (gags) |

The four band splits are in build5.split_band (the arrangement is rewritten before set_backing: same notes, same
instrument settings, same mix overrides for the new stems). They change no sound on purpose; they give the
mixer a fader for each kind.

## The roles (build5.ROLES, TRACK_MIX6, CROWD_*; ds_backend5.ROLE_FX, COPIES; every channel: roles.md)

"level" = measured active level of the role's channels against the lead of the section they sound in (whole
song, roles.md); pan -1 = left, +1 = right; "send" = into the song's short room (0.6 s) / into the song's ducked
ping-pong delay (1 = the lead's). In-channel effects come after the level match, so a fader keeps its meaning.

| role | channels | pan | level vs lead | tone | space | ducks under the lead |
|---|---|---|---|---|---|---|
| lead | AI, AI Verses / Pre / Bridge / Breakdown / Final | centre | 0 | the lead chain (prod/chain4.LEAD), unchanged | room 0.50, delay 1 | - |
| double | AI Double, AI Double R | -0.28 / +0.28 | -13.2 each (pair -9.7 in chorus 1) | copies of the lead's take, 18 ms +6 cents / 27 ms -5 cents, -5 dB shelf from 4.5 kHz, consonants pushed back | room 0.40, delay 0.4 | - |
| harmony | Harmony, Harmony L (+ Final) | +0.70 (its take) / -0.70 (copy 22 ms, -7 cents) | -10.9 / -12.1 (pair -8.6 in chorus 1; final chorus -14.3 / -15.9) | harmony chain (-4 dB from 5 kHz, consonants pushed back), -3 dB at 2.8 kHz | room 0.65; in channel a 0.28 s room after 30 ms at -13 dB | - |
| answer (the lead's voice) | AI Answer | +0.18 | -1.8 | lead chain, -1.5 dB shelf from 6 kHz | room 0.55; in channel 0.22 s after 20 ms at -16 dB | - |
| ad-lib (single words) | AI Hai | -0.40 | -0.5 | lead EQ, band 330 Hz - 5.2 kHz | slap 95 ms at -9 dB, thrown into the delay (1.3), room 0.15 | 2 dB |
| click (runs on eighth notes) | AI Click | -0.40 | -1.9 | band 330 Hz - 5.2 kHz | dry: no slap, no delay, room 0.15 | 2 dB |
| stutter (start of the lead's word) | AI Stutter | centre | +2.6 | lead EQ | as the lead | - |
| alien lead | Aliens, Aliens Verse / Bridge | +0.25 to +0.30 | -1.1 to -2.5 | alien EQ (-3 dB 280 Hz, +2 dB 2.6 kHz) | room 0.30, delay 0.6; no slap (it cost words) | - |
| alien answer | Alien Answer (+ Bridge) | +0.50 | -1.5 to -2.5 | alien EQ | room 0.30 | 1.5 dB |
| alien backing | Aliens Hook, Alien Unison | +0.25 / -0.45 | -11.2 / -6.3 | -4 dB shelf from 3.8 kHz | room 0.40; in channel 0.30 s after 30 ms at -12 dB | 3 dB |
| alien ad-lib | Alien Ouch, Alien Nande | +0.70 | -3.3 / -4.9 | band 260 Hz - 4.6 kHz | slap 110 ms at -8 dB, delay 0.8 / 1.3, room 0.15 | 2 dB |
| alien click / laugh | Alien Click, Alien Laugh | +0.70 | -8.7 / -2.6 | band 260 Hz - 4.6 kHz / low cut 260 Hz | no slap; the laugh is thrown into the delay (1.0) | 2 dB |
| alien stutter | Alien Stutter | +0.50 | -1.4 | alien EQ | as the answers | - |
| alien chant | Alien Chant, Alien Chant 2 | +0.60 / -0.60 | -3.3 / -4.3 | -3 dB shelf from 4.2 kHz | room 0.45 / 0.50; in channel 0.30 s after 30 ms at -12 dB | 2.5 dB |
| crowd (line, answer, whisper, hook) | Crowd, Crowd 2, Crowd 3 x (line, Answer, Whisper, Hook) | -0.85 / +0.85 / -0.30 | line -7 to -9, answers -5 to -8, hook -12 to -15 | low cut 170 Hz, -4 dB shelf from 3.8 kHz | room 0.55; in channel 0.38 s after 38 ms at -9 dB; no delay | 3 dB |
| crowd shout | Crowd Shout x3, Crowd Click x3 | -0.85 / +0.85 / -0.30 | shouts -5 to -9, clicks -7 to -10 | low cut 220 Hz, -2 dB shelf from 5 kHz | room 0.40; in channel 0.38 s after 38 ms at -11 dB | 1.5 dB |

Decisions that came from a measurement:
- Doubles and harmony down, harmony to both sides: chorus 1 lead / others 5.7 -> 6.1 dB with the first values, 8.7
  with the second (harmony then 7.0 dB under the band in 1-5 kHz; third demo 4.2), 8.2 with the harmony 1.5 dB
  up again (5.5 dB under the band).
- No slap on runs: the clicks and the laugh are hits on eighth notes (179 ms); a repeat 95-110 ms later lands
  between them.
- No slap on the aliens' lines: recognisers on their verse-2 lines (43 words) 0.30 / 0.26 / 0.12 without, 0.33 /
  0.33 / 0.16 with a 125 ms repeat at -14 dB (third demo 0.40 / 0.37 / 0.16).
- Crowd shouts 1.5 dB down: "what" / "huh" were 1.3 dB over the lead in pre-chorus 2 (3.4 dB in 1-5 kHz).
- AI Click to 0 dB: at -4 it was 9.1 dB under the band in 1-5 kHz.
- Takes of "click" by audition (audition.py, audition.json): aliens take 2, crowd voices 1 and 3 take 1.
- The whisper's fader (see the log, 15:05). Any note moved to a channel of its own is level-matched alone from
  then on: the other moved channels were checked in roles.md (levels -0.5 to -8.7 dB against the lead).

## Separation, before (third demo's stems) and after (sep.json: meas_all.sh; sep_full.json: sections.sh)

Lead against the sum of all other vocals in 1-5 kHz over the frames where the lead sings; WER = tiny.en / base.en /
small.en (prod/m.py, loops collapsed); "sum" = all vocal stems summed to mono.

| section | lead / others before | after | in mono before / after | WER lead alone before / after | WER sum before / after |
|---|---|---|---|---|---|
| chorus 1 (33-48) | 5.7 dB | 8.2 dB | 6.0 / 8.8 | 0.40 0.54 0.26 / 0.37 0.54 0.14 | 3.37 0.74 0.37 / 1.14 0.71 0.17 |
| verse 1 (9-24) | 15.0 | 15.3 | 15.3 / 16.2 | 0.49 0.32 0.25 / 0.44 0.33 0.28 | 0.65 0.38 0.52 / 0.53 0.39 0.33 |
| post-chorus (49-52) | 2.7 (lead channel = answers + clicks) | 8.4 (main = AI Answer) | 3.2 / 9.4 | - | - |
| verse 2 (53-68) | 14.8 | 15.7 | 15.0 / 16.4 | 0.31 0.17 0.10 / 0.31 0.14 0.10 | 0.36 0.21 0.17 / 0.36 0.21 0.15 (against the dialogue of both voices) |

The lead's take is the same before and after (same seed; flow numbers equal to two digits), yet its WER alone
moves by up to 0.12 (small.en, chorus): differences under that are noise. tiny.en loops on the chorus sum.

Per group, chorus 1 (level / level in 1-5 kHz / pan of the energy / side against mid / L-R correlation / lost in mono):

| group | before | after |
|---|---|---|
| double | -6.7 / -7.2 / -0.52 / -11.0 dB / 1.00 / 0.3 dB | -9.7 / -10.3 / 0.00 / -13.0 dB / 0.90 / 0.2 dB |
| harmony | -6.8 / -8.3 / +0.52 / -11.0 dB / 1.00 / 0.3 dB | -8.6 / -10.1 / +0.14 / -4.4 dB / 0.47 / 1.3 dB |
| crowd (clicks, "hey") | -0.7 / -5.9 / +0.12 / -7.5 dB / 0.70 / 0.7 dB | -3.3 / -7.7 / -0.01 / -3.8 dB / 0.42 / 1.5 dB |

Per group, bars 49-68 (before: aliens -1.6 / -4.1 / +0.38 / -14.0 / 1.00; crowd +0.8 / -1.6 / +0.15 / -7.0 / 0.67):
after: lead answer -0.8 / +1.7 / +0.28; lead ad-lib (clicks) -1.6 / -7.0 / -0.59; aliens -1.7 / -5.1 / +0.43;
alien ad-lib -3.0 / -1.3 / +0.89; crowd shout 0.0 / -2.6 / -0.06, side -4.0 dB, correlation 0.43.

Against the band (vsband.py; 1-5 kHz level of the group over the whole band while the group sounds), chorus 1:
before lead +4.2, double -3.0, harmony -4.2, crowd -1.9; second values: lead +4.8, double -5.4, harmony -7.0,
crowd shout -4.4 (the final values lift harmony 1.5 and the clicks 1.5 dB from there).
The build's own figure, lead over the next loudest element in 1-5 kHz, chorus 1: 7.9 dB (AI Double) -> 12.9 dB (gtrL).

Limiter, bars 49-68 at -11.5 LUFS: third demo max 3.37 / mean 0.40 dB, after 3.23 / 0.35 dB. Sum of the vocal
stems of these bars peaks at +0.9 dBFS in the third demo (bar 52, everything at once) and -2.3 dBFS after.

## The whole song (full/, 08_full_song; sections.log, full.log)

| section | LUFS | lead / others 1-5 kHz | note |
|---|---|---|---|
| Intro 1-8 | -12.5 | (main = the aliens' line) 20.4 | chant: two voices L/R, side -4.9 dB, correlation 0.51 |
| Verse 1 9-24 | -13.5 | 15.5 | |
| Pre-chorus 1 25-32 | -12.2 | 8.4 | a dialogue: the figure is set by the hand-over frames (aliens +12 dB over the lead's breaths there) |
| Chorus 1 33-48 | -10.6 | 7.9 | |
| Post-chorus 49-52 | -11.8 | (main = AI Answer) 7.5 | |
| Verse 2 53-68 | -13.7 | 15.6 | |
| Pre-chorus 2 69-76 | -12.2 | 6.6 | the lowest: "what" / "huh", the chant and ハイハイ border the lead's lines |
| Chorus 2 77-92 | -10.5 | 7.5 | |
| Bridge 93-108 | -10.8 | 12.2 | the stutter "re-re-re" is 11.5 dB under the lead in 1-5 kHz at +1.8 dB full band (the vowel i) |
| Breakdown 109-112 | -16.1 | 11.9 | "hurry" (two whispering voices) now 9.3 and 12.1 dB under the lead, together about 7.5 dB under; before the fader change 19 to 22 dB under (15.0 dB then) |
| Final chorus 113-128 | -9.5 (short-term max -8.9) | 8.1 | the lead (two semitones up) as continuous as in chorus 1 (holes 0.61 /s against 0.46, voiced run 950 ms, hnr 21.1); harmony -13.0, aliens' hook -12.0, crowd hook -9.3 dB against a lead that rides 2.5 dB higher there |
| Outro 129-134 | -12.4 | 42.6 | the chant here is Alien Chant 2 alone, at the left (-0.81); the aliens' review at the right |

No third-demo render of the whole song exists, so there is no "before" for the sections outside the clips.

## Checks (checks.py -> checks.log; silent_check.py -> silent_full.log)

| clip | LUFS wav / mp3 | true peak wav / mp3 dBTP | clipped | lost in mono LU | L-R correlation |
|---|---|---|---|---|---|
| 01_chorus_vocals_before | -13.5 / -14.0 | -2.4 / -2.9 | 0 | 0.2 | 0.94 |
| 02_chorus_vocals_after | -13.6 / -14.0 | -3.6 / -4.0 | 0 | 0.2 | 0.92 |
| 03_chorus_full_mix | -11.5 / -12.0 | -1.0 / -1.3 | 0 | 0.7 | 0.76 |
| 04_adlib_section_vocals_before | -16.6 / -17.0 | -1.0 / -1.5 | 0 | 0.1 | 0.94 |
| 05_adlib_section_vocals_after | -16.6 / -17.0 | -1.6 / -2.1 | 0 | 0.3 | 0.88 |
| 06_adlib_section_full_mix | -12.5 / -12.9 | -1.0 / -1.1 | 0 | 0.4 | 0.86 |
| 07_verse_full_mix | -13.0 / -13.5 | -1.0 / -1.4 | 0 | 0.3 | 0.91 |
| 08_full_song | -11.6 / -12.1 | -1.0 / -1.0 | 0 | 0.6 | 0.78 |

- Third demo's mixes: chorus -11.6 / -12.1 LUFS (0.7 LU lost in mono, correlation 0.77), verse -13.0 / -13.5.
- Limiter of the section mixes (max / mean dB): chorus 1.13 / 0.21 (third demo 1.48 / 0.27), verse 2.72 / 0.32
  (third demo 2.79 / 0.33), ad-lib section at -12.5 LUFS 2.34 / 0.29, whole song 3.42 / 0.32.
- Vocal clips: sums of the vocal stems (post-fader, panned, with the in-channel effects; without the song's
  reverb and delay returns, as the third demo's all-vocals clip). The ad-lib pair is at -17 because the third
  demo's take of these bars peaks 15.6 dB over its loudness.
- Mono: summed to mono the other vocals lose 0.8 dB against the lead in the chorus (third demo 0.4), 1.0 dB in the
  final chorus; the harmony pair loses 1.3 dB (its left side is a delayed copy of its right).
- Nothing silent: 1117 words of 47 channels checked against their stems, none more than 30 dB under its
  channel's median word.
- Ad-lib section clips are bars 49-68 (20 bars, 30 s: the post-chorus with the click runs and the whole of
  verse 2 with the laugh in bar 66), mastered at -12.5 LUFS; the verse at -13.0 as in the third demo.

## Still wrong, from the user's chair

- Nobody has listened. Pans, levels, the slap, the rooms and the band limits are values chosen from figures.
- The lead's "click" (bar 52) is not heard as "click" by any recogniser in ten takes; the crowd's second voice
  is not heard in the pairs (bars 49-51; its bar 52 is). The aliens' "click?" in bar 118 uses the auditioned
  take 2 but was not auditioned itself.
- Harmony L and the second double are delayed, detuned copies, not second performances. In mono the harmony
  pair combs (1.3 dB lost).
- The song has one room, fed mono by all voices: "further back" is done with a small mono room inside the
  channel, which sits where the voice is panned. A stereo room per group, or a pre-delay per send, needs the
  mixer (synth2/engine/mixing.py).
- The dialogue sections (pre-chorus 1 and 2) have the lowest figures (8.4, 6.6 dB): answers and shouts border
  the lead's lines. Not judged whether that is heard as crowding.
- In the final chorus the backing is 9 to 13 dB under a lead that rides 2.5 dB higher than in chorus 1: the
  "everyone joins in" of the last chorus may be smaller than intended.
- The drums' stems still carry several kinds each (snare + clap + rim + toms; hats + shaker + tambourine +
  cowbell; crash + ride).
- "funny", "cool", "wow", "hold on" stay in the lead's lines.

## Files and how to re-render

- build5.py: `build_session("before"|"after")`; MOVES6 (one kind per channel), TWINS, ROLES / role_of, TRACK_MIX6,
  CROWD_PAN / CROWD_KINDS, OVERRIDES6, split_band (MIX5_BAND=off: the band's stems as they were). Needs
  prod/build4.py, prod/chain4.py and the copies vocals.s2s, vocals.py (from sep/).
- ds_backend5.py: cast of the new channels (with the auditioned click takes), COPIES, ROLE_FX (`MIX5_FX` JSON
  overrides a role, `MIX5_FX=off` switches the role effects off), takes cached in cache/ as their sounding span
  in 16 bit (sep/cache is read too). Engine: dsinfer.py, dspitch.py, config.json (sep/'s, plus the unused ad-lib
  option onset_floor).
- run5.py OUT A B [--arr before|after] [--set JSON]: mix, stems (FLAC), report, tracks_bars_A-B.json. `1 134` =
  the whole song (one mixer thread, 13.6 GB of memory at the peak; OUT.rss shows it).
- final.sh = everything from the cached takes (about 17 minutes): full.sh, batch2.sh "after 33 48" "after 49 68"
  "after 9 24", meas_all.sh (sep.json), mkclips.py (out/), sections.sh (sep_full.json), silent_check.py,
  roles_md.py (roles.md), checks.py. final2.sh = the whole song and what follows from it.
- Measuring: sepmeas.py (the separation figures of a stems folder), vsband.py (groups against the band),
  audition.py (takes of the stamped click), inventory6.py (inventory6.md: what every channel sings).
- Third demo's stems of bars 49-68: mix_before/preview (run5.py --arr before); of bars 33-48 and 9-24:
  prod/mix/flow/preview.
- Load average while rendering 1.0 to 2.0; no timing is reported as a result.

