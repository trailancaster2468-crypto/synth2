# F1: a continuous vocal line from the engine (final report)

The engineer's final report, followed by the running log. Paths point into a scratch folder.

## Summary

F1 (continuous vocal line), done in $S/flow/ (S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad). No repo file was changed (git status clean). Nobody has listened: everything below is a measurement.

RESULT against the targets (verse lead, 3 seeds, config.json): voiced run 162 -> 367 ms (target 250: reached), hole time 28.0 -> 15.3 % (target 15: 0.3 over; 10.5 % if the breaths are not counted), holes/s 3.22 -> 2.36 (target 1.8: NOT reached; 2.06 without the breaths), lvl_sd 8.48 -> 7.37 (target 7: NOT reached on verse 1; 6.61 on verse 2, 6.07 on the chorus), phrase length 11 -> 23 s (lines joined by breaths). Chorus: holes/s 1.44 -> 0.88, hole time 14.1 -> 5.5 %, run 367 -> 855 ms, median phrase 2.6 -> 10.9 s. Words: small.en unchanged in all three cases; base.en +0.04 on verse 1, -0.10 on the chorus, equal on the held-out verse 2; tiny.en WORSE on verse 1 (0.39 / 0.52 / one loop against 0.42) and on the chorus (0.52 against 0.31), equal on verse 2. So "words no worse" holds on small.en and on the held-out verse, not on tiny.en.

WHY 1.8 holes/s is out of reach: in the reference clips the holes ARE the unvoiced consonants (holes/s 1.05-1.30 = 1.19-1.45 unvoiced clusters per second, 71-76 % of them read as a hole, plus a few breaths; 0.02-0.08 holes/s anywhere else). The verse's lyric carries 2.33-2.42 unvoiced clusters per second: at the reference's own share that is 1.72 holes/s before one breath, 1.98 with the six breaths. What the demo had on top and the reference does not: 0.76 holes/s on voiced consonants (z v dh b d g sung as whispers) and dead rests between lines. Those are gone now (62 voiced-consonant holes in three seeds -> 6).

WHAT WAS BUILT (engine copy flow/dsinfer.py, options off = the demo's take to under one 16-bit step): (1) cons_cap + cons_cap_keep + onset_keep: consonants that share a note with a vowel are capped by class (stops 35 ms, fricatives 50, nasals 60, glides 50) but never under half of what the duration model gave them; consonants after a rest are not capped; (2) cons_voicing_floor: voiced consonants held 8 dB (voiced stops 11 dB) under the vowel voicing around them; (3) breath: an AP in every gap of 0.12 s or more whatever the style, the whole gap up to 0.45 s, else the last 0.30 s, also before each phrase, with a breathiness curve shaped like the reference's breaths.

TWO FINDINGS THAT CHANGE THE BRIEF'S PREMISES: (a) Puffs: flow.py reads a puff's level 32 ms before the centre of the window that calls it unvoiced, so it often reads the vowel before. The demo verse's puffs (-2.0 / -3.6 dB) read -14.5 dB where the window sits; the reference's read -8.4 to -9.9. Above 4 kHz our fricatives peak like the reference's (-12.6 dB re the median vowel against -10 to -12) and our p/t/k bursts are 12-18 dB weaker. There is nothing to tame; nothing was changed. (b) Our consonants are quieter than the reference's, not louder, yet raising fricatives closes no hole (flow.py measures below 8 kHz) and cost words.

DEVIATIONS: FINDINGS.md was NOT written: the harness refuses report files from this agent; its content is this report, and the running log is $S/flow/NOTES.md. The repo test suite was not run (no repo file touched). Timeline: first verse before/after clips 15 minutes after start, final clips after 55 minutes.

## Measurements run

All run as `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I ...`, one render or recognition process at a time, load average 1-4 throughout (another engineer rendering), so render times are noisy. F = $S/flow.

1. Baseline reproduced: `t1.py base` (the demo's settings through the engine copy) gives the demo's numbers exactly (verse 1 seed 1: 3.14 holes/s, 27.3 %, run 160, WER 0.42 / 0.27). `t_same.py`: options off vs demo3/runs/q5_final/chorus1_s1.wav largest difference 2.57e-05 (one 16-bit step is 3.1e-05); config.json vs its stored take 2.55e-05. `render.py verse1 OUT.wav` vs runs/final/verse1_s1.wav: difference 0.0. PASS.
2. Hole table: `holes.py WAV DUMP ...` on three seeds of the demo's verse and of the final verse. 
3. Reference comparison: `inherent.py ref lianrenxin sajiao zaijian`, `inherent.py ours WAV DUMP`; `reflev.py ref NAME --ap` (levels by phoneme class, every breath); `puffs.py` (band above 4 kHz); `edges.py` (releases, noise before lines).
4. Lever rows, verse 1 seed 1, tiny.en + base.en: `t1.py NAME` for base, cap, cvf8, breath, scale06, minv, share5, vel15, legato, c1-c4, d1-d3, e1-e3, g1, g2, h1 (rows `f_NAME` in runs/results.jsonl). Chorus rows on 1-3 seeds for the cap rules (k1, k1nb, m1, m1nb, P, Q, n1, n1nb, r1, r1nb, r1p).
5. Final: `final.py final` = config.json on verse1, chorus1, verse2 x seeds 1-3 x tiny.en, base.en, small.en (9 runs); `final.py before --case=verse2` (3 runs; verse1 and chorus1 "before" are the demo's own rows and wavs); control `final.py nobreath --case=chorus1,verse1 --fast` (6 runs, tiny + base). Table: `table.py final flow --seeds`.
6. Engine health in the 9 final rows: 0 pitch fallbacks, 0 repaired frames, 0 weak vowels (worst vowel -6.7 dB), render 27-32 s for 23.2 s sung at load 2-4 (the demo's settings: 25-28 s), i.e. no slowdown.
7. Clips: `make_clips.py`; `tools/flow.py` on every clip wav; verse_before.mp3 decodes to the same samples as demo3/out/01_verse_lead_dry_5_lower.mp3; all seven clips -16.8 LUFS as mp3, true peaks -0.4 to -3.8 dBFS (wav), no dynamics processing.

NOT run: the repo test suite (no repo file changed); any listening; any render of the other cast tracks, the full song or a mix; small.en on the lever rows and on the control.

FAILED or abandoned along the way (kept in results.jsonl): the first recommended config (tag `flow`: absolute caps without cons_cap_keep) passed on both verses and lost chorus words (2 seeds: 0.73 / 0.43 / 0.09 against 0.31 / 0.19 / 0.07) and was replaced; caps growing with note length or local tempo (cons_cap_share, removed from the engine again) did not protect the chorus; 25 ms stops, louder stop bursts and louder fricatives all cost words.

## Numbers

WER = tiny.en / base.en / small.en, loops collapsed. Flow columns = tools/flow.py. Means of seeds 1-3 unless a seed is named.

A. BEFORE / AFTER (`python3 -I $S/flow/table.py final --seeds`)
case | set | holes/s | hole % | run ms | breaks/s | lvl_sd | phrase s | cpp | hnr | WER
verse1 | demo | 3.22 | 28.0 | 162 | 3.02 | 8.48 | 11.1 | 24.11 | 17.49 | 0.424 / 0.265 / 0.235
verse1 | final | 2.36 | 15.3 | 367 | 1.77 | 7.37 | 23.1 | 25.13 | 17.60 | tiny 0.39, 0.52, 0.93 (seed 3: tiny.en loops on "saucer" and loses the rest) / 0.307 / 0.220
chorus1 | demo | 1.44 | 14.1 | 367 | 1.47 | 7.55 | 2.6 | 23.50 | 21.22 | 0.305 / 0.190 / 0.067
chorus1 | final | 0.88 | 5.5 | 855 | 0.74 | 6.07 | 10.9 | 24.18 | 21.19 | 0.524 / 0.095 / 0.076 (tiny per seed 0.69, 0.60, 0.29: "invaders" heard as "in favor")
verse2 (held out) | demo | 2.99 | 24.1 | 172 | 2.88 | 9.24 | 2.9 | 24.33 | 18.21 | 0.302 / 0.175 / 0.103
verse2 | final | 1.69 | 9.6 | 302 | 1.11 | 6.61 | 3.2 | 25.27 | 18.50 | 0.310 / 0.175 / 0.095
Control without breaths (tiny / base): chorus 0.30 / 0.28, verse1 0.50 / 0.37. So tiny.en's chorus loss comes with the breaths, base.en's chorus is better with them, and in verse 1 the consonant options cost base.en +0.04 to +0.11. Per-seed recogniser noise is about +-0.1 on the chorus (35 words).
The delivered clips (seed 1): verse 3.19 -> 2.33 holes/s, 28.5 -> 15.5 %, run 160 -> 325, lvl_sd 8.65 -> 7.37, phrase 11.1 -> 23.1 s; chorus 1.37 -> 0.91, 13.9 -> 5.4 %, 390 -> 955 ms, 7.32 -> 6.08, 2.56 -> 10.95 s; verse2 2.93 -> 1.58, 23.1 -> 9.6 %, 210 -> 270, 9.37 -> 6.62.

B. WHERE THE HOLES ARE: demo verse, 3 seeds, 215 holes in 66.8 s (`holes.py`): class | holes | hole time s | share of the class's time
unvoiced stops p t k ch | 79 | 6.00 | 79 %
unvoiced fricatives | 55 | 4.07 | 73 %
rests between lines (written 8th rests, 0.18 s) | 19 | 2.73 | 60 %
voiced stops b d g jh | 29 | 2.61 | 76 %
voiced fricatives z v dh | 33 | 2.25 | 70 %
vowels 0, nasals 0, glides 0; no hole at a line edge other than the rest itself.
Cause of the voiced-consonant holes: the variance model predicts their voicing 16-27 dB under the vowels (vowels -21 dB, voiced fricatives -37 verse / -46 chorus, voiced stops -48).
Final verse, 3 seeds, 164 holes in 69.4 s: unvoiced stops 80 (3.59 s), breaths 21 (3.31 s), unvoiced fricatives 57 (2.51 s), voiced stops 4, voiced fricatives 2. Voice breaks 202 -> 123.

C. REFERENCE (`inherent.py`): clip | holes/s | unvoiced clusters/s | share read as holes | holes elsewhere/s
lianrenxin 1.20 | 1.39 | 71 % | 0.02; sajiao 1.05 | 1.19 | 76 % | 0.07; zaijian 1.30 | 1.45 | 74 % | 0.08
demo verse s1 3.14 | 2.42 | 89 % | 0.76; demo chorus s1 1.42 | 1.00 | 74 % | 0.63
final verse s1 2.29 | 2.33 | 87 % | 0.04; final chorus s1 0.82 | 0.96 | 67 % | 0.18; final verse2 s1 1.65 | 1.73 | 70 % | 0.30

D. LEVERS, verse 1 seed 1, each alone on the demo: lever | holes/s | hole % | run | breaks/s | lvl_sd | WER tiny / base
demo 3.14 | 27.3 | 160 | 3.05 | 8.50 | 0.42 / 0.27
caps (stops 45 ms, fricatives 70) 3.03 | 21.6 | 240 | 2.54 | 8.21 | 0.44 / 0.31
duration model's consonants x 0.6: 3.21 | 23.6 | 180 | 2.62 | 8.35 | 0.40 / 0.28
min vowel share 0.5: 3.14 | 24.4 | 180 | 2.96 | 8.67 | 0.38 / 0.35
min_vowel 0.04: 3.14 | 27.3 | 160 | 3.05 | 8.58 | 0.40 / 0.32 (no effect)
velocity 1.5: 3.19 | 26.4 | 160 | 3.01 | 8.22 | 0.49 / 0.33 (no effect)
voicing floor 8 dB on all voiced consonants: 2.38 | 19.6 | 260 | 2.42 | 7.73 | 0.42 / 0.38
breaths only: 3.11 | 26.8 | 160 | 3.11 | 8.71 | 0.48 / 0.32, phrase 11 -> 23 s
legato (sing through the 8th rests): 3.23 | 24.7 | 160 | 2.88 | 8.57 | 0.42 / 0.31
Combinations: caps + floor 8 + breaths 2.20 | 15.4 | 310 | 2.16 | 7.67 | 0.47 / 0.35; with caps 35 / 50 ms 2.21 | 14.1 | 340 | 1.64 | 7.09 | 0.42 / 0.33; voiced stops at 14 dB instead of 8: 2.64 | 17.1 | 320 | 2.03 | 7.51 | 0.42 / 0.28; at 11 dB with caps 35 / 50 (config `flow`, 3 seeds) 2.37 | 14.7 | 337 | 1.79 | 7.18 | 0.409 / 0.307 / 0.220.
Rejected: fricatives +up to 9 dB after the vocoder 2.20 | 15.4 | lvl_sd 6.99 | 0.46 / 0.49; fricative breathiness +6 dB 0.49 / 0.56; stop breathiness floor -36 dB base.en 0.39 and a tiny.en loop; stops capped at 25 ms 2.08 | 12.7 | lvl_sd 6.70 | 0.47 / 0.52.
Cost of capping: p/t/k bursts above 4 kHz (median re the median vowel) demo -28.9 dB, 46 ms caps -33.1, 35 ms caps -35.5 (reference's aspirated stops -10 to -17).
lvl_sd is consonants, not vowels: of the variance of the voiced-frame level in the demo's verse 35 % sits on stops, 16 % fricatives, 10 % voiced fricatives, 9 % rests, 25 % vowels (the vowels' own sd is 3.5 dB, per-vowel 2.4 before and 2.3 after).
Legato: the planner leaves no gap between notes in the robotic style; every gap in the three cases is a written rest (0.179 s in the verses, 0.36-0.89 s in the chorus, 2.9 s in verse 2).

E. BREATHS (`reflev.py ref NAME --ap`; 31 AP in the reference clips): between lines 160-325 ms (median 250), directly after the line's last vowel with no rest; 400-520 ms at phrase starts; level median over the breath -23 / -28 / -25 dB re the clip's median vowel per song, loudest 10 ms -17 to -23; quarters of the breath -20 -19 -24 -40 dB; fed breathiness -46 to -49 dB, voicing -88 to -96.
Ours in `final`: 17-23 breaths per minute (reference songs 5-25), 104-302 ms (verse median 174 = the 8th rests; 284-290 in chorus and verse 2), level median -28 to -31 dB, loudest 10 ms -23 to -24 dB, quarters -28 -25 -27 -39 to -44. The quiet end of the reference range.
Line edges (`edges.py`): shortest release from -6 to -30 dB 40-50 ms in the demo (reference 35-50), none cut off; loudest 5 ms in the 80 ms before a line -37 dB median in the demo, -24 dB with breaths.

F. PUFFS (`puffs.py`, loudest 10 ms above 4 kHz re median vowel, median / 90th percentile): unvoiced fricatives reference -12.4 / -7.6 and -10.3 / -4.9, demo verse -12.6 / -8.3, demo chorus -16.0 / -8.8; unvoiced stops reference -16.7 / -9.6 and -10.2 / -3.9, demo -28.9 / -16.5. flow.py's puff level: demo verse -3.6 dB as printed, -14.5 dB read where its voicing window sits; reference -5.0 to -5.8 printed, -8.4 to -9.9 aligned.

To reproduce a take: `cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 python3 -I $S/flow/render.py verse1 OUT.wav [--seed N] [--before]`.

## Recommendations

- Use config.json's "flow" block on the lead: every key is a keyword argument of flow/dsinfer.py render_track_ds, added on top of the demo's lead options (5 semitones lower). In demo3/ds_backend3.py that means importing the flow copy of dsinfer.py and merging the block into LEAD_OPTS.
- Give AI Double and Harmony the same cons_cap, cons_cap_keep, onset_keep and cons_voicing_floor as the lead so their consonants land together, and consider leaving `breath` off on them so three breaths do not stack. This is reasoning, not a measurement: neither was rendered.
- Judge the verse's hole count against its lyric, not against 1.8: this text has 2.3-2.4 unvoiced consonant clusters per second and the reference voice reads 71-76 % of its own as holes. Further gain on that number is in the song (fewer unvoiced consonants per second on the 16th-note lines, or tempo), not in the engine.
- Listen to chorus_after_two_lines_with_breath first to set the breath level. `breath.breathiness` -44 gives -28 to -31 dB re the vowels; about -41 would sit at the reference's median, lower values make it less present. Raising `breath.min_gap` to 0.2 sings through the verse's 8th rests and removes its six catch-breaths (17 s without a breath).
- Do not use: gain or breathiness lifts on fricatives, a breathiness floor on stops, stops under 35 ms, or class caps without cons_cap_keep (they pass on the verses and lose chorus words). No puff limiter is needed: the demo's unvoiced bursts are quieter than the reference's.
- Count breaths separately from holes when scoring flow: flow.py counts every breath shorter than 250 ms as a hole (21 of the final verse's 164), which penalises exactly what the user asked for.
- When this is ported into synth2/neural, carry four options (cons_cap with cons_cap_keep, onset_keep, cons_voicing_floor, breath) and drop the rest (cons_scale, min_vowel_share, sp_gap, stop_breath, fric_breath, level stay in the copy, off, as tested dead ends). The breath is inserted by the engine whatever the style, because the robotic style's planner inserts none.

## Open issues

- Target holes/s <= 1.8 on the verse lead is not reached: 2.36 (2.06 without the breaths). Target lvl_sd <= 7 is not reached on verse 1: 7.37 (the earlier config without cons_cap_keep measured 7.18, and lost chorus words). Hole time is 15.3 % against 15.
- Words: tiny.en is worse on verse 1 (0.39, 0.52 and one repetition loop against 0.42 / 0.39 / 0.47) and on the chorus (0.52 against 0.31); base.en is +0.04 on verse 1. small.en and the held-out verse 2 are unchanged. The demo's settings were chosen on verse 1 and the chorus with these recognisers, so part of this may be regression from a tuned optimum; that is an interpretation, not a measurement.
- Most lever decisions rest on one seed of verse 1 with two recognisers, where the noise is 0.05 to 0.1: the voiced-stop floor (8 / 11 / 14 dB), rejecting the stop breathiness floor, and the breath shape were not confirmed on three seeds. Only the final config, the earlier `flow` config and the no-breath control have three seeds.
- Nobody has listened. Whether the breaths read as breaths, whether 35 ms stops sound soft (their bursts are 6 dB weaker than in the demo, which were already 12-18 dB under the reference's), and whether voiced consonants held at -8 dB sound natural are all unjudged.
- FINDINGS.md was not written: the harness blocks report files from this agent. Its content is in this report; the running log is /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad/flow/NOTES.md.
- The repo test suite was not run: no repo file was changed (git status clean).
- Only the AI lead was rendered, 5 semitones lower, three sections, dry. Not tested: the other cast tracks, the written key, a full song or mix, Japanese lyrics (the consonant class table lists the bank's Japanese phonemes but none was sung), scores with gaps under 0.12 s (sp_gap), and a track whose first note starts less than 0.4 s in (the opening breath is then cut at time 0).
- Breath density is 17-23 per minute, the upper end of the reference songs' 5-25, because every gap of 0.12 s or more gets one. A breath that the next line's consonants leave under 0.10 s becomes a rest.
- runs/results.jsonl holds rows from engine revisions that no longer exist: tag `flow` is the earlier config (not config.json), and the t1.py variants k1, k2, m1, P, Q, n1 used cons_cap_share, which was removed; t1.py says so. Breath rows before `final` used earlier breath shapes.
- verse_after needed a -0.4 dBFS true peak (the demo's clips used a -1 dBFS ceiling) to sit at the same -16.8 LUFS as verse_before: one vowel peaks 6.8 dB over the median vowel.

---

# Running log

# F1 notes (kept current)

- 23:25 start. Folder made, engine copied from demo3 (dsinfer.py, dspitch.py unchanged so far).
- Plan: (1) holes.py: hole/break table by phoneme class from q5_final dumps; (2) harness hf.py (render + flow + WER t/b);
  (3) levers one by one on verse1 seed 1; (4) combination, 3 seeds, 3 cases; (5) clips.
- 23:33 holes.py table done (stops 35 holes / 2.87 s, unvoiced fricatives 18 / 1.31 s, rests 6 / 0.85 s, voiced fricatives
  11 / 0.73 s, vowels+nasals+glides 0). reflev.py: reference consonants are LOUDER than ours re the vowels (fric peak -6 dB
  vs ours -10/-12; stops peak -6 vs -9/-15), reference AP 160-325 ms between lines (median 250), 400-520 at phrase start,
  level median -21..-28 dB re median vowel, fed breathiness -46..-49, directly after the vowel, no SP.
- 23:36 engine options added in flow/dsinfer.py: cons_scale, cons_cap, min_vowel_share, onset_keep, sp_gap, breath,
  cons_voicing_floor, stop_breath, fric_breath, level (even_level). Harness hf.py (rows runs/results.jsonl), t1.py variants.
- 23:38 verse1 s1: base 3.14 holes/s 27.3 % run 160 lvl_sd 8.50 WER .42/.27 (= demo). c1 (caps + cons voicing floor 8 +
  breath): 2.20 / 15.4 % / run 310 / lvl_sd 7.67 / phr 23 s, WER .47/.35 (worse words, one seed). Clips out/verse_before,
  out/verse_after_first_pass. NEXT: levers one by one (cap, cvf8, breath) to find what costs words.
- 23:43 Seed-1 verse rows so far (holes/s, hole %, run ms, breaks/s, lvl_sd | WER tiny/base collapsed):
  base 3.14 27.3 160 3.05 8.50 | .42 .27;  cap(CONS_CAP)+onset_keep 3.03 21.6 240 2.54 8.21 | .44 .31;
  cons voicing floor 8 (all voiced cons) 2.38 19.6 260 2.42 7.73 | .42 .38;  breath only 3.11 26.8 160 3.11 8.71 phr 23 | .48 .32
  c1 = cap+cvf8+breath 2.20 15.4 310 2.16 7.67 | .47 .35;  c2 = c1 + level lift 8 (fricatives): 2.20 15.4 320 2.03 6.99 | .46 .49 (words worse, holes same)
  c3 = c1 + fric_breath +6: 2.21 15.3 320 1.99 7.23 | .49 .56 (worse);  c4 = c1 + stop_breath -40: 2.20 15.1 310 1.90 7.11 | .51 .34 (no gain)
  d1 = c1 with caps stop .035 fric .05: 2.21 14.1 340 1.64 7.09 | .42 .33;  d2 = c1 with vstop floor 14: 2.64 17.1 320 2.03 7.51 | .42 .28
  d3 = c1 + min_vowel_share .5: 2.20 14.8 330 1.82 7.32 | .42 .35
  e1 = CAP2 + floors {vfric,nasal,glide 8, vstop 14} + breath: 2.64 15.5 340 1.64 7.27 | .38 .32;  e2 = e1 with vstop 11: 2.34 14.5 340 1.64 7.18 | .39 .29
- KEY FINDINGS: (a) inherent.py: in the reference clips holes/s (1.05-1.30) = their unvoiced-consonant clusters (1.19-1.45/s, 71-76 % of
  them read as holes) + some rests; holes elsewhere 0.02-0.08/s. Demo verse: 2.42 unvoiced clusters/s (89 % holes) + rests + 0.76/s holes
  elsewhere (voiced consonants: engine fault, removed by the voicing floor). So 1.8 holes/s on the verse is under what the reference voice
  would give at this consonant rate (2.33/s x 0.74 = 1.72, + breaths 0.26 = 1.98).
  (b) puffs: flow.py reads the level 32 ms before the centre of its voicing window; the demo's "puffs at -2..-3.6 dB" read -14.5 dB where
  the window sits (reference -8.4..-9.9). puffs.py (band above 4 kHz): our fricatives peak like the reference's (-12.6 vs -10..-12 dB re
  vowel median), our p/t/k bursts are far weaker (-29 vs -10..-17). No loud unvoiced bursts in the demo: nothing to tame.
  (c) edges.py: no cut-off release (shortest -6 -> -30 dB 40-50 ms, reference 35-50), no noise before line starts (-37 dB median).
  (d) lifting fricatives (post gain or breathiness) costs words and closes no holes (flow measures below 8 kHz).
  (e) voiced stops held at -8 dB voicing cost base.en words (.27 -> .35/.38); at 14 no cost but they read as holes; 11 in between.
- Running: batch2 (e3, scale06, minv, share5, vel15, legato rows). NEXT: 3 seeds for e2 / e1, chorus1 + verse2, clips.
- 23:50 More seed-1 verse rows: e3 = e1 + share .5: 2.60 15.3 340 1.69 7.28 | .35 .33; alone on the demo: cons_scale 0.6: 3.21 23.6 180 2.62 8.35 | .40 .28;
  min_vowel .04: 3.14 27.3 160 3.05 8.58 | .40 .32; min_vowel_share .5: 3.14 24.4 180 2.96 8.67 | .38 .35; velocity 1.5: 3.19 26.4 160 3.01 8.22 | .49 .33;
  legato sp_gap 0.2: 3.23 24.7 160 2.88 8.57 | .42 .31.
  g1 = e2 + stop_breath -36: 2.29 14.1 325 1.86 6.77 | .94(tiny loops) .39; g2 = g1 + stop cap 25 ms: 2.03 12.3 340 1.60 6.48 | .69 .40;
  h1 = e2 + stop cap 25 ms: 2.08 12.7 350 1.38 6.70 | .47 .52  => 25 ms stops and louder bursts cost words. NOT used.
  Capping weakens p/t/k bursts (band > 4 kHz, median re vowel): base -28.9, cap 46 ms -33.1, cap 35 ms -35.5; stop_breath -36 gives -32.1.
- DECISION: recommended = e2 (config.json "flow"): caps stop/vstop 35 ms, aff 60, fric/vfric 50, nasal 60, glide 50 + onset_keep;
  voicing floor vfric/nasal/glide 8 dB, vstop 11 dB; breath min_gap .12 fill .45 length .30 breathiness peak -44 (shaped).
- 23:50 final.py launched: tag `flow` and `before`, 3 seeds, 3 recognisers. verse2 done: flow 1.65-1.74 holes/s 9.4-9.7 % run 290-320 lvl_sd 6.5-6.9,
  WER mean .36/.16/.10; before 2.97-3.01 / 23.6-24.6 % / 160-185 / 9.0-9.4, WER .30/.17/.11. verse1 + chorus1 running (final_b.log).
- Clips so far: out/verse_before (= demo), out/verse_after (e2 seed 1).
- 23:56 e2 (tag `flow`, config at that time) on 3 seeds: verse1 2.34-2.38 holes/s, 14.5-14.8 %, run 335-340, lvl_sd 7.14-7.21, WER .41/.31/.22 (before .42/.27/.24);
  verse2 .36/.16/.10 (before .30/.17/.11). BUT chorus1 s1 .83/.46/.06 (before .23/.14/.06): words lost on tiny/base. Run stopped.
  Causes looked at on chorus s1 (tiny/base): (1) breaths are heard by tiny/base as a syllable at the line start ("select" -> "to let", "have a nice" ->
  "camp on ice"); small.en is not affected. (2) absolute caps shorten the few long consonants of slow notes (apocalypse p 10 -> 3 frames).
- 00:00 added: cons_cap_share + cons_cap_note (caps grow with the note length above 0.2 s), breath shape with a quiet tail (50 ms) before the consonant.
  Rows chorus1 (tiny/base): k1 (cap = max(cap, .3 x note) + breath) .49/.20; k1nb (no breath) s1 .37/.03 s2 .51/.06; m1 (cap + .5 x (note - .2), breath)
  .74/.29; m1nb s1 .51/.29 s2 .54/.11 s3 .63/.17 (mean .56/.19; before .305/.19). verse1 s1: k1 2.46 17.8 310 2.03 7.84 | .43/.39; m1 2.38 15.8 325 1.95 7.40 | .40/.40.
  => tiny.en loses chorus words even without breaths: suspect the voicing floor on voiced consonants at slow tempo ("neither" -> "me look", "robot" -> "all but").
- 00:05 running batch4: chorus1 3 seeds, no breath: P = caps only (m rule), Q = P + floors vfric 12 / vstop 14.
- 00:07 chorus1, no breath, 3 seeds (tiny/base): P = caps only (rule "cap + .5 x (note - .2)"): .49/.31 .34/.11 .60/.26 = .48/.23 (before .305/.19).
  So tight caps alone cost chorus words: the duration model gives some consonants of the chorus 8-15 frames (p of apocalypse 116 ms, s of saucer,
  r of square) and a 35-50 ms cap cuts them to a third. A local-tempo rule did not help either (n1nb s1 .43/.29): those words sit in 8th-note runs.
- 00:10 NEW RULE cons_cap_keep 0.5: a cap never takes a consonant under half the length the duration model gave it. r1nb chorus s1: 0.99 holes/s,
  6.0 %, run 690, lvl_sd 6.22 | .40/.09; verse s1 (no breath) 2.47 / 15.3 / 350 / 1.75 / 7.81 | .50/.32. With breath (r1) chorus s1: 0.78 / 5.3 / 950 /
  6.14 / phr 10.9 | .31/.17 (before s1 .23/.14; seeds .23-.37 / .14-.26). Breath from the variance model's own curve + 4 dB (r1p): .49/.09.
  cons_cap_share / cons_cap_note (rules k, m, P, Q, n in t1.py) were REMOVED from the engine again; their rows stay in results.jsonl.
- 00:12 breath: softer end (out to -22 dB under the peak over the last 90 ms; the 50 ms dead-quiet tail made short breaths a blip), a breath
  left under 0.10 s by the consonants becomes a rest (min_len). config.json rewritten = caps + keep 0.5 + onset_keep + floors 8/8/8/11 + breath.
- 00:13 running: final.py final (tag `final`), 3 cases, seed 1 first, then seeds 2, 3; three recognisers. Log final_c.log.
- NOTE: the harness refuses FINDINGS.md (subagent rule: no report files); the findings go into the structured report. This log stays.
- 00:19 FINAL config (config.json, tag `final`), 3 seeds, 3 recognisers; `python3 -I table.py final flow --seeds` prints it. Means:
  verse1  before 3.22 holes/s 28.0 % run 162 brk 3.02 lvl_sd 8.48 phr 11.1 | cpp 24.11 hnr 17.49 | WER .424/.265/.235
          final  2.36 15.3 367 1.77 7.37 23.1 | 25.13 17.60 | tiny .39 .52 .93(loop at "saucer") / base .307 / small .220
  chorus1 before 1.44 14.1 367 1.47 7.55 2.6 | 23.50 21.22 | .305/.190/.067
          final  0.88 5.5 855 0.74 6.07 10.9 | 24.18 21.19 | .524/.095/.076   (tiny: "invaders" -> "in favor" x3 on two seeds)
  verse2  before 2.99 24.1 172 2.88 9.24 2.9 | 24.33 18.21 | .302/.175/.103
          final  1.69 9.6 302 1.11 6.61 3.2 | 25.27 18.50 | .310/.175/.095
  earlier config e2 (tag `flow`: absolute caps, no keep): verse1 2.37 14.7 337 1.79 7.18 | .409/.307/.220; chorus (2 seeds) .73/.43/.09 (words lost).
  No pitch fallback, no repaired frame, no weak vowel (worst -6.7 dB) in any final row.
- final verse hole table (3 seeds, holes.py): 164 holes (2.36/s): unvoiced stops 80 (3.59 s), breaths 21 (3.31 s), unvoiced fricatives 57 (2.51 s),
  voiced stops 4, voiced fricatives 2. Without the breaths: 143 holes = 2.06/s and 10.5 % of the time.
- Clips built (make_clips.py, seed 1): out/verse_before (mp3 decodes identical to the demo's clip 01), verse_after, chorus_before, chorus_after,
  chorus_after_two_lines_with_breath (9.29-14.83 s of the chorus, breath at 12.50 s, 174 ms), verse2_before, verse2_after. All -16.8 LUFS as mp3.
- 00:20 running: final.py nobreath (consonant options without breaths), chorus1 + verse1, 3 seeds, tiny/base: is tiny.en's loss the breath?
- 00:23 control `nobreath` (consonant options without breaths, tiny/base): chorus1 .40/.09 .14/.31 .37/.43 = .30/.28; verse1 .50/.32 .58/.48 .41/.32 = .50/.37.
  So tiny.en's chorus loss in `final` (.52 against .305) comes with the breaths (.30 without), base.en's chorus is better with them (.095 against .28);
  in verse1 the consonant options cost base.en +.04 (with breaths) to +.11 (without), small.en nothing. Recogniser noise is +-.1 per seed.
- 00:24 t_same.py: engine copy with the flow options off = the demo's stored chorus take to 2.6e-05 (under one 16-bit step); render.py verse1 = runs/final
  take, difference 0. Breaths in `final`: 17-23 per minute, 104-302 ms (verse median 174), level median -28..-31 dB re the vowel median, loudest 10 ms -23..-24.
- DONE: dsinfer.py (+ unchanged dspitch.py), config.json, render.py, final.py, make_clips.py, clips in out/. FINDINGS.md was not written (harness rule);
  the findings are in the structured report and in this log.

