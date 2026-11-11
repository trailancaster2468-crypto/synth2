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

## Files
