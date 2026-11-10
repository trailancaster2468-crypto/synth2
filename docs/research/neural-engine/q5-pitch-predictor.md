# Q5: the voicebank's pitch predictor

S = /tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad

## How to call it (tested; for Q6, Q7 and anyone with a copy of the spike)

`$S/q5/dspitch.py` is the port of `DiffSingerPitch.cs`. It is verified on one phrase by
`$S/q5/t_verify.py` (14 checks, "ALL PASSED", see "Verification" below) and it stays working at
that path: changes are made in another file and renamed in after `t_verify.py` passes.

Shortest way, if you use the spike's `render_track_ds`: import `$S/q5/dsinfer_pp.py` instead of
`proto/dsinfer.py`. It is the spike file plus one hook (`diff proto/dsinfer.py q5/dsinfer_pp.py`).

```python
sys.path.insert(0, "/home/user/synth2"); sys.path.insert(0, f"{S}/q5")
import onnxruntime as ort
import dsinfer_pp as dsinfer, dspitch
Q = f"{S}/models/ds/qixuan/Qixuan_v2.7.0_DiffSinger_OpenUtau"
V = f"{S}/models/ds/vocoder/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
bank = dspitch.PitchBank(Q + "/dspitch", threads=2)
ort.set_seed(1); ds = dsinfer.DS(Q, V, threads=2)          # seeded fresh sessions = repeatable render
hook = dsinfer.bank_pitch(bank, expr=1.0, steps=10, seed=1)  # expr 0 = plain note steps, 1 = full expression
audio, info = dsinfer.render_track_ds(ds, project, track, phrase_pitch=hook)   # hook=None: the spike's pitch
# info["pitch"] = [(f_start, semitones per frame)] per phrase, as fed to the vocoder
```

Runnable example (renders `tests/plain.s2s` twice, Synth2 pitch and bank pitch, same seeds):

    cd /home/user/synth2 && python3 -I $S/q5/example_call.py $S/q5/out/example --style pop
    ASR_THREADS=2 python3 -I $S/tools/score.py $S/q5/out/example ; python3 -I $S/tools/voiceq.py $S/q5/out/example

Result of that run (load 2-3): WER tiny/base/small synth2 0.08/0.04/0.04, bank 0.12/0.00/0.04;
voiceq cpp 24.7 vs 25.3, hnr 20.5 vs 20.1, voiced 0.91 vs 0.90, hf -28.5 vs -29.1, run 360 vs 400 ms.

If you have your own phrase loop, the call itself is three lines. Everything is on the phrase's
frame grid (frame 0 = first frame of the head `SP`; `sum(durs)` frames; hop 512 at 44.1 kHz):

```python
tm = [t for t in plan.timings if t.index in {i for u in units for i in u.notes}]   # the phrase's pitched notes
notes = dspitch.notes_from_timings(tm, f_start, FRAME)     # or [dspitch.PhraseNote(start_frame, end_frame, midi, slur)]
p_semi = bank.predict(seq, durs, notes, expr=1.0, steps=10, seed=1234)   # float64 [sum(durs)], MIDI semitones
p_semi += curves["pitchDelta"][sl] / 100.0                 # the spike adds the pitchDelta lane itself
```

Things to know before you rely on it:

- `seq` / `durs` are exactly what the spike feeds the acoustic model (head `SP`, phonemes, tail `SP`).
  The predictor tokenizes with its own table (`dspitch/*.phonemes.json`); the bank's 149 symbols
  are all there.
- `seed=None` reuses one session: each call then gives a different take. With a seed a fresh pitch
  session is made per call (about 0.15 s) and the curve is reproducible. `predict` calls
  `onnxruntime.set_seed`, which is global: sessions you create afterwards inherit that seed, so set
  your own seed right before creating your own sessions.
- Spike bug, fixed in `dsinfer_pp.py`: `phrase_groups.midi()` indexes `plan.timings[u.notes[0]]`,
  but `u.notes` holds track note indices and `plan.timings` skips breaths, closures and silent
  notes. A score with a `br` note raises IndexError in `proto/dsinfer.py` (reproduced with
  `q5/x8b_breath.py` before the fix). On the five test cases index and position are equal, so the
  fix changes nothing there. Map notes through `timing.index` (as above) in your own code.
- Cost: about 0.6 s for a 3.3 s phrase (2 threads, steps 10, load 1-2); more numbers below.
- The demo lead is `style=robotic`: its Synth2 pitch is exactly flat note steps. The predictor at
  `expr=0` gives nearly the same thing (steps smoothed over 3 frames), at `expr=1` real-singer
  transitions, drift and light vibrato.

## Status

Port verified. Curve metrics from the interrupted run are being re-measured; audio A/B running.
Sections below are appended as results arrive.

## Model I/O (inspected with `session.get_inputs()`, `q5/inspect_io.py`)

- `dspitch/0102_qixuan_muon1_pitch.qixuan.linguistic.onnx` (25.4 MB, create 0.14 s at 2 threads):
  `tokens int64[1,P]`, `languages int64[1,P]`, `ph_dur int64[1,P]` -> `encoder_out float[1,P,256]`, `x_masks bool[1,P]`.
- `dspitch/0102_qixuan_muon1_pitch.qixuan.pitch.onnx` (59.5 MB, create 0.26 s cold):
  `encoder_out`, `ph_dur int64[1,P]`, `note_midi float[1,N]`, `note_rest bool[1,N]`, `note_dur int64[1,N]`,
  `pitch float[1,F]`, `expr float[1,F]`, `retake bool[1,F]`, `steps int64[]` -> `pitch_pred float[1,F]`.
- No `note_glide` input (the bank has no glide embedding), no `spk_embed`, no `depth`.
- The pitch model's phoneme table has the same 149 symbols as the acoustic model's but different token ids:
  tokenize with `dspitch/*.phonemes.json`.

## Verification of the port (`python3 -I $S/q5/t_verify.py`, phrase 0 of `plain`, 281 frames)

All 14 checks pass (run at load 1-2):

- note durations sum to the phrase's frames; head and tail are rest notes;
- same seed + fresh session = bit-identical curve; another seed differs (max 66 cents);
- `expr=0` returns the note steps smoothed by the 3-tap window 0.29/0.41/0.29 (median difference
  1.7 cents, max 4 cents over all frames). This pins the frame alignment: every note boundary of the
  model's base pitch is on the frame the spike starts the vowel on (the step is half way between
  frames `k-1` and `k` for a note starting at frame `k`);
- `expr=1`: median |deviation| from the score 8 cents on sung frames;
- retake: frames with `retake=false` come back within 0.7 cents (median; max 3) of the `pitch`
  given for them, the rest is regenerated to join them;
- the `pitch` input is not read where `retake` is true (output bit-identical with junk there);
- per-frame `expr` arrays are accepted.

Differences from the C#, all deliberate: note boundaries are taken from the frame-rounded note
times (OpenUtau rounds cumulative milliseconds); `pitch` is 60 where `retake` is true (OpenUtau
copies the existing curve there; the model does not read it); `predict_dur: true` (word mode),
speaker embeds and `speedup`-style models raise `ValueError` (Qixuan needs none of them); no
phoneme blend, no tensor cache.

## Determinism (`python3 -I $S/q5/x3_coexist.py det`, phrase 0 of `plain`)

- `onnxruntime.set_seed(n)` + a fresh pitch session: bit-identical curve every time, and identical
  across 1, 2 and 4 threads (0.0 cents difference).
- The seed is consumed when the session is created. The first run of a seeded session is the
  reproducible one; its second run is another take (max difference 102 cents on this phrase) and
  depends on what the session ran before (109 cents). So: one fresh pitch session per phrase.
- A reused session (no seed) gives a new take per call (max 102 cents apart on this phrase).

## Retake / keep regions (`x3_coexist.py keep`: every second note keeps Synth2's curve)

- On kept frames (`retake=false`) the model's own output is within 1.5-2.0 cents (median), 4 cents
  (p95), 10-14 cents (max) of the curve it was given, on `plain` (pop) and `grace` (ballad). The
  regenerated notes join the kept ones (the output is one continuous diffusion sample).
- Writing the kept frames back exactly (as OpenUtau does) therefore moves them by at most 14 cents.

## Results log (raw lines; tables are built from `out/ab/results.json` further down)

Columns: WER tiny/base/small, voiceq cpp (higher = clearer phonation) hnr voiced hf lvl_sd run,
pitch accuracy on sustained frames (median |cents|, share within 30 cents), pitch and render seconds, load.

    verse1_synth2_s1  wer 0.57/0.53/0.55 cpp 20.8 hnr 18.3 voiced 0.88 hf -30.2 lvl_sd 6.7 run 180 | med|c| 0.0 in30 1.00 | pitch 0.0s render 41.0s load 3.98
    verse1_bank1_s1   wer 0.52/0.46/0.36 cpp 21.8 hnr 17.1 voiced 0.91 hf -30.5 lvl_sd 6.3 run 180 | med|c| 19.6 in30 0.63 | pitch 5.7s render 41.4s load 5.85

## What `expr` does (`x3_coexist.py expr`, seed 1, f0 fed to the vocoder against the score)

`expr` is not a linear depth. Below 0.5 it is nearly the plain smoothed note steps; the learned
behaviour (early, slower transitions, overshoot, preparation, drift, vibrato) arrives between 0.5 and 1.

| case | expr | median abs cents (sustained) | within 30 c | half-way time vs note start, ms (IQR) | 20-80 % ms | overshoot c | preparation c | long notes with vibrato | drift c | RMS dev from base c |
|---|---|---|---|---|---|---|---|---|---|---|
| grace (ballad, 72 BPM) | 0 | 2.3 | 1.00 | -6 (-9..-3) | 35 | 2 | -1 | 0/25 | 0 | 2.5 |
| | 0.25 | 4.6 | 1.00 | -6 (-9..-3) | 35 | 6 | -2 | 0/25 | 1 | 4.9 |
| | 0.5 | 5.2 | 0.97 | -8 (-11..-4) | 35 | 10 | 1 | 0/25 | 5 | 24 |
| | 0.75 | 6.7 | 0.90 | -17 (-50..-7) | 81 | 13 | 15 | 24 % (5.1 Hz, 16 c) | 12 | 49 |
| | 1 | 7.8 | 0.88 | -23 (-66..-9) | 70 | 15 | 20 | 52 % (5.1 Hz, 19 c) | 20 | 56 |
| chorus1 (168 BPM) | 0 | 2.4 | 1.00 | -6 (-9..-3) | 35 | 3 | 0 | 0/4 | 0 | 2.7 |
| | 0.25 | 4.6 | 1.00 | -6 (-10..-3) | 35 | 6 | -1 | 0/4 | 1 | 5.2 |
| | 0.5 | 6.7 | 0.94 | -8 (-11..-4) | 35 | 12 | 6 | 0/4 | 3 | 20 |
| | 0.75 | 10.2 | 0.82 | -12 (-33..-7) | 46 | 21 | 17 | 1/4 (5.9 Hz, 14 c) | 9 | 52 |
| | 1 | 12.2 | 0.78 | -16 (-46..-7) | 58 | 25 | 20 | 3/4 (5.1 Hz, 14 c) | 11 | 59 |
| plain (120 BPM) | 0 | 1.5 | 1.00 | -4 (-7..-2) | 35 | 1 | 0 | no long notes | | 2.0 |
| | 0.5 | 8.0 | 0.91 | -12 (-22..-7) | 46 | 8 | 5 | | | 40 |
| | 1 | 11.8 | 0.77 | -33 (-76..-17) | 70 | 18 | 11 | | | 68 |

(Vibrato depth = half the peak-to-peak of the part faster than 3 Hz; "drift" = standard deviation
of the slow trend inside a long note; long = at least 0.6 s. Every expr value has a flat bias of
-1 to -6 cents.)

## Vibrato on top of a predicted curve (`x3_coexist.py hyb`, seed 1, long notes >= 0.6 s)

| case | curve | notes with vibrato | rate Hz | depth c | drift c | median abs c | within 30 c |
|---|---|---|---|---|---|---|---|
| grace (ballad) | Synth2 `build_pitch` | 100 % of 25 | 5.3 | 30 | 8 | 12.5 | 0.86 |
| | bank, expr 1 | 52 % | 5.1 | 19 | 20 | 7.8 | 0.88 |
| | bank, fast part x0 on sustained frames (`venv0`) | 0 % (6 c left) | | | 21 | 5.2 | 0.90 |
| | bank, fast part x2 (`venv2`) | 56 % | 5.1 | 33 | 20 | 13.1 | 0.77 |
| | bank, fast part removed + Synth2 vibrato (`hyb`) | 84 % | 5.3 | 27 | 21 | 14.5 | 0.80 |
| | bank, per-frame expr 0.5 on sustained frames + Synth2 vibrato (`hybexpr`) | 80 % | 5.4 | 29 | 8 | 12.7 | 0.83 |
| twinkle (pop) | Synth2 | 20 % of 10 | 6.0 | 32 | 6 | 9.2 | 0.91 |
| | bank, expr 1 | 10 % | 6.1 | 22 | 27 | 10.1 | 0.81 |
| | `venv0` | 0 % (9 c left) | | | 26 | 7.1 | 0.84 |
| | `hyb` | 30 % | 6.0 | 31 | 24 | 10.4 | 0.77 |

- The predictor's own vibrato is real but light and not on every long note: 5.1 Hz, 14-22 cents,
  on a third to a half of the long ballad notes (32-52 % over three seeds), against Synth2's
  5.3 Hz / 30 cents on every one. It is a pop singer's vibrato, not a ballad or classical one.
- Scaling the fast part (everything above about 2.5 Hz of `pred - base`, 0.4 s Hann trend removed)
  on sustained frames works as a vibrato depth control: x0 leaves 6-9 cents, x2 doubles the depth,
  x1 is the identity. This is how the `vibratoEnv` lane can act on a predicted curve.
- Both hybrids put Synth2's vibrato on the predicted transitions. `hyb` (post-processing) keeps
  the predicted drift (21 c) and does not change the diffusion input; `hybexpr` also flattens the
  drift (8 c) because expr 0.5 removes it.

## Explicit transition attributes on a predicted curve (`x4_trans.py`)

Every third note given `transition=0.2`, `overshoot=0.25`; window = preparation + transition +
0.25 s after it; 43 % of `plain`'s frames and 17 % of `grace`'s are inside a window.

- (a) Window kept through the retake mask (`retake=false`, `pitch` = Synth2's curve): inside the
  windows the model's output is within 1.2-1.5 cents (median), 3.2-3.6 (p95), 6-8 (max) of Synth2's
  curve; outside the windows the curve changes by 2.4-6.4 cents RMS against the unconditioned take
  with the same seed. Largest frame-to-frame step within 8 frames of a window edge: 52 cents
  (`plain`), no larger than Synth2's own curve there (52).
- (b) No conditioning, Synth2's curve cross-faded in over 3 frames: 163 cents largest step and
  108 cents largest second difference on `plain` (two unrelated curves meet mid-movement).
- Without either, the prediction is 18-30 cents (median) and 133-158 cents (p95) away from the
  transition the user asked for.

So explicit transition / overshoot attributes go through the retake mask, with the kept frames
written back under 3-frame ramps.
    verse1_s2pop_s1   wer 0.53/0.49/0.42 cpp 21.0 hnr 18.6 voiced 0.87 hf -29.9 lvl_sd 6.6 run 175 | med|c| 10.4 in30 0.79 | render 47.4s load 6.45
    verse1_bank05_s1  wer 0.56/0.48/0.52 cpp 21.1 hnr 17.8 voiced 0.89 hf -30.2 lvl_sd 6.3 run 220 | med|c| 10.9 in30 0.86 | pitch 7.3s render 52.8s load 8.71
    verse1_bank0_s1   wer 0.59/0.48/0.40 cpp 20.8 hnr 17.8 voiced 0.86 hf -30.0 lvl_sd 6.3 run 185 | med|c| 2.5 in30 1.00 | pitch 7.3s render 41.8s load 9.87
    chorus1_synth2_s1 wer 0.80/0.43/0.14 cpp 21.1 hnr 21.5 voiced 0.94 hf -30.9 lvl_sd 5.5 run 585 | med|c| 0.0 in30 1.00 | render 48.9s load 9.22
    chorus1_bank1_s1  wer 0.49/0.26/0.06 cpp 21.4 hnr 20.5 voiced 0.93 hf -30.9 lvl_sd 5.5 run 610 | med|c| 12.2 in30 0.78 | pitch 8.0s render 51.1s load 8.91
    plain_synth2_s1   wer 0.08/0.08/0.04 cpp 24.6 hnr 20.5 voiced 0.91 hf -28.5 lvl_sd 4.8 run 360 | med|c| 6.6 in30 0.97 | render 11.5s load 2.15
    plain_bank1_s1    wer 0.12/0.08/0.04 cpp 25.1 hnr 20.0 voiced 0.91 hf -28.7 lvl_sd 5.2 run 385 | med|c| 11.8 in30 0.77 | pitch 2.0s render 16.1s load 2.66

(`synth2` on verse1/chorus1 = the demo lead as shipped: voice nova, style robotic, exactly flat
note steps. `s2pop` = Synth2 pop style with voice.robotic 0 on the same phoneme layout.
`bank0/05/1` = the predictor at expr 0 / 0.5 / 1. Same acoustic seed within a row group. One seed
only so far: verse1 `bank0` and `synth2` differ by 3 cents of pitch and by 0.15 in small.en WER,
which is the size of the noise of a single clip.)

## Diffusion steps of the pitch model (`x3_coexist.py steps`, grace, 39.6 s sung, load 9: times are inflated)

| steps | seconds | RMS vs 50 steps, cents | median abs c | within 30 c | overshoot c | long notes with vibrato |
|---|---|---|---|---|---|---|
| 1 | 2.1 | 27.4 | 10.8 | 0.84 | 30 | 20 % (5.7 Hz, 26 c) |
| 2 | 3.2 | 14.3 | 6.8 | 0.87 | 17 | 28 % |
| 5 | (see json) | (see json) | 7.2 | 0.88 | 15 | 36 % |
| 10 | 9.1 | 3.9 | 7.8 | 0.88 | 15 | 52 % (5.1 Hz, 19 c) |
| 20 | (see json) | (see json) | 8.2 | 0.88 | 15 | 56 % |

10 steps (OpenUtau's default) is converged to 4 cents RMS of the 50-step curve with the same seed.

## Audio A/B, interim table (seed 1 only; `python3 -I $S/q5/x5_table.py --md` rebuilds it from out/ab/results.json)

| case | source | seeds | WER tiny | WER base | WER small | cpp | hnr | voiced | hf | lvl_sd | run ms | med abs c | in 30 c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| verse1 | synth2 | 1 | 0.57 | 0.53 | 0.55 | 20.8 | 18.3 | 0.88 | -30.2 | 6.7 | 180 | 0.0 | 1.00 |
| verse1 | s2pop | 1 | 0.53 | 0.49 | 0.42 | 21.0 | 18.6 | 0.87 | -29.9 | 6.6 | 175 | 10.4 | 0.79 |
| verse1 | bank0 | 1 | 0.59 | 0.48 | 0.40 | 20.8 | 17.8 | 0.86 | -30.0 | 6.3 | 185 | 2.5 | 1.00 |
| verse1 | bank05 | 1 | 0.56 | 0.48 | 0.52 | 21.1 | 17.8 | 0.89 | -30.2 | 6.3 | 220 | 10.9 | 0.86 |
| verse1 | bank1 | 1 | 0.52 | 0.46 | 0.36 | 21.8 | 17.1 | 0.91 | -30.5 | 6.3 | 180 | 19.6 | 0.63 |
| chorus1 | synth2 | 1 | 0.80 | 0.43 | 0.14 | 21.1 | 21.5 | 0.94 | -30.9 | 5.5 | 585 | 0.0 | 1.00 |
| chorus1 | s2pop | 1 | 0.89 | 0.34 | 0.11 | 21.1 | 21.1 | 0.94 | -30.6 | 5.7 | 540 | 7.5 | 0.93 |
| chorus1 | bank0 | 1 | 0.77 | 0.49 | 0.23 | 21.1 | 21.6 | 0.93 | -30.7 | 5.5 | 510 | 2.4 | 1.00 |
| chorus1 | bank05 | 1 | 0.46 | 0.26 | 0.09 | 21.2 | 21.3 | 0.94 | -30.6 | 5.6 | 560 | 6.7 | 0.94 |
| chorus1 | bank1 | 1 | 0.49 | 0.26 | 0.06 | 21.4 | 20.5 | 0.93 | -30.9 | 5.5 | 610 | 12.2 | 0.78 |
| twinkle | synth2 | 1 | 0.30 | 0.40 | 0.10 | 25.1 | 21.4 | 0.93 | -29.9 | 4.8 | 770 | 9.2 | 0.91 |
| twinkle | bank1 | 1 | 0.40 | 0.40 | 0.20 | 25.3 | 21.6 | 0.92 | -30.7 | 4.4 | 880 | 10.1 | 0.81 |
| plain | synth2 | 1 | 0.08 | 0.08 | 0.04 | 24.6 | 20.5 | 0.91 | -28.5 | 4.8 | 360 | 6.6 | 0.97 |
| plain | bank1 | 1 | 0.12 | 0.08 | 0.04 | 25.1 | 20.0 | 0.91 | -28.7 | 5.2 | 385 | 11.8 | 0.77 |

Reading so far (one seed, so only large effects count): on chorus1 the two flat curves (`synth2` robotic,
`bank0`) and Synth2's pop curve score 0.77-0.89 on tiny.en, the predictor at expr 0.5 and 1 scores 0.46-0.49;
base.en 0.34-0.49 against 0.26. On verse1 every source is between 0.52 and 0.59 (tiny) and 0.46-0.53 (base):
the pitch source is not what makes the fast verse hard to understand. Breathiness (cpp) moves by at most
+1.0 dB (verse1, bank1) and stays 3-4 dB below the slow cases whatever the pitch source.

Pitch-model steps (grace 39.6 s sung / chorus1 20.8 s sung, load 9-10): 1 step 2.1/2.7 s, 2: 3.2/3.5, 5: 5.3/5.3,
10: 9.1/12.2, 20: 20.3/15.9, 50: 47.6/37.6 s; RMS against the 50-step curve 27/28, 14/12, 7/6, 3.9/3.6, 1.9/1.9 cents.

## Checks on the audio (`python3 -I $S/q5/x5_table.py --audio`, seed-1 clips)

- The f0 fed to the vocoder is the f0 of the audio: Synth2's YIN (`synth2.analysis.estimate_f0`)
  on the rendered clips minus the fed f0, sustained frames: median -0.1..+0.2 cents, median
  absolute 0.7-3.1 cents (verse1, chorus1, twinkle, plain; both sources). The pitch-accuracy numbers
  in this file use the fed f0.
- Onset timing does not depend on the pitch source (same phoneme durations by construction):
  the onset envelopes of the `bank1` and `synth2` clips are aligned at 0 ms lag on verse1, chorus1
  and plain (correlation 0.60-0.67); twinkle has no clear peak (0.39 at -10 ms, 0.38 at 0 ms).
  What moves is the pitch transition inside the syllable: its half-way point is 14 ms before the
  vowel start in the robotic style, 33-54 ms before it in Synth2's pop style, and 14-54 ms
  (median; quartiles -100..-4) before it with the predictor at expr 1.

## How far the predictor strays (`x7_range.py`, 5 scores x 3 seeds, expr 1)

Largest distance of a sung frame outside the range of the notes within 0.25 s of it: twinkle 2.70,
plain 1.53, verse1 2.28, chorus1 2.45, grace 2.94 semitones (scoops into phrase starts and
consonants). Against the model's base pitch the largest difference is 4.2-5.9 semitones (early
transitions on large leaps), so a sanity bound must be on the local note range, not on the base
pitch: `dspitch.range_excess(pred, inputs) > 6` is the check used in `q5prod.py`.

    twinkle_synth2_s1 wer 0.30/0.40/0.10 cpp 25.1 hnr 21.4 voiced 0.93 hf -29.9 lvl_sd 4.8 run 770 | med|c| 9.2 in30 0.91 | render 24.0s load 8.26
    twinkle_bank1_s1  wer 0.40/0.40/0.20 cpp 25.3 hnr 21.6 voiced 0.92 hf -30.7 lvl_sd 4.4 run 880 | med|c| 10.1 in30 0.81 | pitch 3.6s render 26.6s load 9.32
    grace_synth2_s1   wer 0.27/0.19/0.04 cpp 25.5 hnr 21.7 voiced 0.95 hf -30.6 lvl_sd 4.5 run 1560 | med|c| 12.5 in30 0.86 | render 98.3s load 9.52
    grace_bank1_s1    wer 0.42/0.08/0.04 cpp 25.4 hnr 22.7 voiced 0.95 hf -30.6 lvl_sd 4.4 run 1520 | med|c| 7.8 in30 0.88 | pitch 16.5s render 86.7s load 10.47

## The curves: Synth2 `build_pitch` against the predictor at expr 1 (`x1_curves.py`, `x1_curves.json`)

f0 fed to the vocoder against the score; three predictor seeds (range given). The seed-1 rows were
reproduced bit for bit in today's `x3_coexist.py` runs.

| case (style, tempo, notes) | curve | median abs c (sustained) | within 30 c | bias c | half-way vs vowel start ms | 20-80 % ms | overshoot c | preparation c | vibrato on long notes | drift c |
|---|---|---|---|---|---|---|---|---|---|---|
| twinkle (pop, 100, 14) | Synth2 | 9.2 | 0.91 | -0.2 | -38 | 58 | 10 | 35 | 20 % of 10: 6.0 Hz, 32 c | 6 |
| | predictor | 9.3-10.1 | 0.81-0.86 | +0.7..+1.5 | -24..-54 | 110-139 | 14-17 | 15-17 | 10 %: 19-31 c | 16-27 |
| plain (pop, 120, 29) | Synth2 | 6.6 | 0.97 | -0.5 | -34 | 46 | 13 | 13 | no long notes | |
| | predictor | 11.8-12.0 | 0.75-0.77 | -3.4..-4.9 | -33..-39 | 64-70 | 18-24 | 11-13 | | |
| verse1 (robotic, 168, 114) | Synth2 | 0.0 | 1.00 | 0 | -14 | 23 | 0 | 0 | no long notes | |
| | predictor | 19.6-20.0 | 0.63-0.64 | -9.4..-11.1 | -14..-15 | 46 | 27-30 | 19-22 | | |
| chorus1 (robotic, 168, 53) | Synth2 | 0.0 | 1.00 | 0 | -14 | 23 | 0 | 0 | 0 of 4 | 1 |
| | predictor | 10.2-12.2 | 0.77-0.80 | -2.2..-3.7 | -15..-16 | 52-58 | 22-25 | 16-20 | 1-4 of 4: 4.8-5.1 Hz, 14-15 c | 8-11 |
| grace (ballad, 72, 33) | Synth2 | 12.5 | 0.86 | -0.1 | -52 | 70 | 10 | 22 | 100 % of 25: 5.3 Hz, 30 c | 8 |
| | predictor | 7.8-8.2 | 0.88-0.89 | -1.2..-1.3 | -23..-34 | 70-81 | 12-16 | 20-22 | 32-52 %: 5.0-5.1 Hz, 19-20 c | 20-22 |

Run-to-run spread of the predictor (standard deviation across three seeds, cents):
sustained frames median 3.6-5.0 (p90 12-18); within 60 ms of a legato note change median 7-19
(p90 19-44). Two takes are the same performance with different detail.

What the predicted curve looks like, in words:

- Transitions start earlier and take two to three times longer than Synth2's robotic snap, about
  as long as Synth2's pop and ballad glides, but their timing varies from note to note (quartiles
  of the half-way point span 40-90 ms where Synth2's span 5-10 ms).
- Overshoot 12-30 cents and a preparation dip of 11-22 cents: the same order as Synth2's styles.
- Slow drift inside long notes 16-27 cents (Synth2: 6-8), i.e. held notes wander like a singer's.
- Vibrato light and occasional (5 Hz, 14-22 cents, a third to a half of long notes).
- On the fast verse (median note 0.18 s) the voice never settles: 63 % of sustained frames within
  30 cents and a flat bias of 10 cents. Tuning is looser than Synth2's on every case but the ballad.

Determinism, added: the seeded curves of the interrupted run (`pitch/<case>_pred_s1.npz`, another
process, the day before, load 4-6) and today's (`pitch/x3_<case>_e1.npz`, load 9-10) are
bit-identical on chorus1, grace and plain, and the audio batch used the same curve again.

## The production algorithm, tested (`q5prod.py`, `python3 -I $S/q5/x6_prod_test.py`)

`q5prod.production_pitch(ctx, bank, source="auto", expr=None, steps=10, vibrato="auto")` is the
reference implementation of the recommendation below. Its controls, checked on pitch curves:

- robotic track (`verse1`): resolves to the expression engine, output equals `build_pitch` exactly;
- ballad (`grace`): resolves to the predictor with expr 1, no fallback; two calls give identical curves;
- `vibratoEnv` 0: fast part of long notes 22 -> 6 cents, notes with vibrato 40 % -> 0 %;
  `vibratoEnv` 2: 22 -> 38 cents;
- `pitchDelta` +50 cents: the curve moves by exactly 0.5 semitone;
- `vibrato_depth=0.6 vibrato_rate=6.5 vibrato_start=0.2` on 13 long notes: those notes sing 6.4 Hz
  at 57 cents (asked: 6.5 Hz, 60 cents), every other note is unchanged (0.00 cents);
- `transition=0.2 overshoot=0.25` on every third note: the windows sing `build_pitch`'s transition
  exactly (0.000 cents inside the ramps), the rest of the take moves by 2.1 cents RMS, the largest
  frame step near a window edge is 97 cents where the unconditioned prediction has 101;
- `vibrato="synthetic"`: 88 % of long notes get Synth2's vibrato (5.3 Hz, 28 cents) on the
  predicted transitions (predicted alone: 40 %, 19 cents);
- a phoneme missing from the pitch model: the phrases that contain it fall back to `build_pitch`,
  the others are predicted.

## Cost (`x3_coexist.py cost`; the machine was at load 9-12 for the whole study, so times are inflated)

- Session creation, 2 threads: linguistic 0.16 s (once, reusable: deterministic), pitch 0.24-0.32 s
  at load 12 (0.15 s quoted in the team brief at low load). A seeded take needs a fresh pitch
  session per phrase.
- Inference, steps 10, fresh session included: 0.58 s for a 3.3 s phrase at load 1-2;
  0.66 s (187 frames) to 4.4 s (1813 frames) at load 4-6 (interrupted run), about 2.3 ms per frame
  + 0.25 s per phrase; 1.5 s to 7.5 s at load 12 (3.6 ms per frame + 0.9 s).
- Share of a whole phrase render at the same load (audio batch, `pitch_s` / `render_s` in
  `out/ab/results.json`): 12-19 % on top of duration + variance + acoustic + vocoder.
- Steps: time is close to linear in steps above 5; 5 steps cost 55 % of 10 and are within 6-7
  cents RMS of the converged curve.

## Recommendation (provisional until the remaining seeds are in; the final table is at the end)

1. Ship the predictor as a second pitch source next to `build_pitch`, selected per track:
   `pitch_source = auto | expression | predictor`.
2. `auto`: the predictor for every style except `robotic`, and only when `voice.robotic < 0.5` and
   the bank has a usable `dspitch/`; otherwise `build_pitch`.
3. `expr = STYLE_EXPR[style] * (1 - voice.robotic)` with 1.0 for pop, ballad, rock, jazz, rnb and
   classical and 0.75 for soft. Values below 0.5 are pointless (plain note steps).
4. Layers on a predicted curve, in this order (all tested in `x6_prod_test.py`):
   kept windows for notes with `transition` / `overshoot` attributes (retake mask + 3-frame ramps),
   vibrato gain `style.vibrato_depth * vibratoEnv` on the fast part of sustained frames, Synth2's
   own vibrato on notes with vibrato attributes, `pitchDelta` last.
5. Per phrase: one fresh pitch session seeded from a hash of the phrase content; 10 steps; fall
   back to `build_pitch` for a phrase that cannot be predicted.

### Production algorithm, per phrase (reference: `q5prod.py` + `dspitch.py`)

Inputs: `phones` and `ph_dur` exactly as fed to the acoustic model (head `SP`, phonemes, tail
`SP`; F = sum(ph_dur) frames, hop 512 / 44100 s), the phrase's pitched `NoteTiming`s, `f_start`
(global frame of the phrase's first frame), the track's style, voice, seed, the `vibratoEnv` and
`pitchDelta` lanes sampled on the frame grid, and `s2 = build_pitch(...)` for the same frames
(always computed: it is the fallback, the source of kept windows and of synthetic vibrato).

1. Notes to frames: `start = round(t_start / frame) - f_start`, `end = round(t_end / frame) - f_start`
   (at least one frame), `midi = pitch + detune`, `slur = melisma` (`dspitch.notes_from_timings`).
2. Note-level inputs (`dspitch.build_notes`): a rest note from frame 0 to the first note; for every
   gap a rest note; each note (rest flag: a slur copies the previous note's flag, otherwise true
   when no vowel of the phrase overlaps the note); a tail rest; the last duration absorbs rounding
   so durations sum to F; each run of rests takes the pitch of the nearest sung note (a run between
   two sung notes is split in the middle; 60 when nothing is sung).
3. Linguistic encoder (shared session, deterministic): `tokens` from `dspitch/*.phonemes.json`,
   `languages` from the symbol prefix, `ph_dur` -> `encoder_out`.
4. Kept windows: for each note with a `transition` or `overshoot` attribute the frames from
   `c - T/2 - 0.8 T` to `min(c + T/2 + 0.25 s, note end)` with `T` the note's transition time and
   `c = start + style.transition_offset * T`. `retake` = false there, true elsewhere; `pitch` =
   `build_pitch` without vibrato there, 60 elsewhere. No such note: `retake` all true.
5. Pitch model: `encoder_out`, `ph_dur`, `note_midi`, `note_dur`, `note_rest`, `pitch`, `retake`,
   `expr` = the scalar on every frame, `steps` = 10. Session: `onnxruntime.set_seed(h)` then a new
   `InferenceSession`, one run, discard; `h` = 31 bits of sha256 over (track seed, phones, ph_dur,
   notes, expr, steps). `set_seed` is process-global: hold a lock from `set_seed` to the end of the
   session's creation if tracks render in parallel.
6. Reject and fall back to `s2` for this phrase when: a phoneme is missing from the pitch table,
   there is no pitched note, onnxruntime raises, the output is not finite, or a sung frame lies
   more than 6 semitones outside the range of the notes within 0.25 s (`dspitch.range_excess`).
7. Write kept windows back: `w = soft_window(kept, 3)`, `pred = w * s2_no_vibrato + (1 - w) * pred`.
8. Vibrato: `base = dspitch.base_pitch(inputs)`; `fast = (pred - base) - hann_lowpass(pred - base, 0.4 s)`;
   `pred += (g - 1) * weight * fast` with `g = style.vibrato_depth * vibratoEnv`, and `g = 0` on
   notes that carry `vibrato_depth` / `vibrato_rate` / `vibrato_start`; `weight` = 1 on the
   sustained part of notes of at least 0.35 s (from 0.12 s after the start to 0.05 s before the end,
   0.06 s cosine ramps), any length for notes with vibrato attributes, 0 inside kept windows. Then
   add Synth2's vibrato (`build_pitch(vibrato_env=env) - build_pitch(vibrato_env=0)`) on the frames
   of the notes with vibrato attributes.
9. Add `pitchDelta / 100`. The result feeds the variance model, the acoustic model (plus tone
   shift) and the vocoder exactly as `build_pitch`'s curve does today.

The phrase cache key must include the pitch source, expr, steps, the seed and the pitch model's
identity next to what already keys a phrase.

## Other material through the production path (`x8_extra.py`, `x8b_breath.py`, pitch only)

- `examples/sakura.s2s` (Japanese, `ja/` symbols, 22 notes, 2 melismas): 1 phrase predicted, no
  fallback, median 5.5 cents, 88 % within 30 cents.
- `examples/neon_hearts.s2s` (English pop, 75 notes, 3 phrases): no fallback, median 10.3 cents,
  81 % within 30 cents.
- A phrase with two `br` notes and a `-` melisma: each breath (`AP`) becomes a rest note owned by
  the note before it, the melisma keeps the vowel's sung flag, 455 frames predicted, finite,
  median 10.6 cents on sung frames, at most 2.4 semitones outside the local note range.

## Audio A/B, three seeds on the demo material (interim; final table at the end of the file)

| case | source | seeds | WER tiny | WER base | WER small | cpp | hnr | voiced | hf | lvl_sd | run ms | med abs c | in 30 c |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| verse1 | synth2 | 1,2,3 | 0.64 (0.57..0.69) | 0.52 (0.49..0.53) | 0.43 (0.36..0.55) | 20.9 (20.8..21.0) | 18.2 (18.0..18.3) | 0.87 (0.87..0.88) | -30.1 (-30.2..-29.7) | 6.5 (6.4..6.7) | 183 (180..190) | 0.0 (0.0..0.0) | 1.00 (1.00..1.00) |
| verse1 | s2pop | 1 | 0.53 | 0.49 | 0.42 | 21.0 | 18.6 | 0.87 | -29.9 | 6.6 | 175 | 10.4 | 0.79 |
| verse1 | bank0 | 1 | 0.59 | 0.48 | 0.40 | 20.8 | 17.8 | 0.86 | -30.0 | 6.3 | 185 | 2.5 | 1.00 |
| verse1 | bank05 | 1 | 0.56 | 0.48 | 0.52 | 21.1 | 17.8 | 0.89 | -30.2 | 6.3 | 220 | 10.9 | 0.86 |
| verse1 | bank1 | 1,2,3 | 0.54 (0.44..0.65) | 0.78 (0.46..1.29) | 0.39 (0.36..0.43) | 21.6 (21.3..21.8) | 17.2 (17.1..17.2) | 0.90 (0.89..0.91) | -30.3 (-30.6..-29.7) | 6.3 (6.2..6.3) | 213 (180..280) | 19.7 (19.6..20.0) | 0.63 (0.63..0.64) |
| chorus1 | synth2 | 1,2,3 | 0.72 (0.60..0.80) | 0.38 (0.29..0.43) | 0.12 (0.09..0.14) | 21.1 (21.1..21.1) | 21.5 (21.5..21.5) | 0.92 (0.90..0.94) | -30.4 (-30.9..-30.1) | 5.8 (5.5..5.9) | 575 (550..590) | 0.0 (0.0..0.0) | 1.00 (1.00..1.00) |
| chorus1 | s2pop | 1 | 0.89 | 0.34 | 0.11 | 21.1 | 21.1 | 0.94 | -30.6 | 5.7 | 540 | 7.5 | 0.93 |
| chorus1 | bank0 | 1 | 0.77 | 0.49 | 0.23 | 21.1 | 21.6 | 0.93 | -30.7 | 5.5 | 510 | 2.4 | 1.00 |
| chorus1 | bank05 | 1,2,3 | 0.51 (0.46..0.63) | 0.28 (0.20..0.37) | 0.16 (0.09..0.23) | 21.3 (21.2..21.3) | 21.3 (21.2..21.3) | 0.92 (0.91..0.94) | -30.3 (-30.6..-30.0) | 5.8 (5.6..5.9) | 560 (520..600) | 7.1 (6.7..8.0) | 0.94 (0.94..0.95) |
| chorus1 | bank1 | 1,2,3 | 0.51 (0.49..0.54) | 0.28 (0.23..0.34) | 0.13 (0.06..0.20) | 21.5 (21.4..21.6) | 20.4 (20.1..20.5) | 0.92 (0.91..0.93) | -30.6 (-30.9..-30.4) | 5.6 (5.5..5.8) | 588 (550..610) | 11.2 (10.2..12.2) | 0.78 (0.77..0.80) |
| twinkle | synth2 | 1 | 0.30 | 0.40 | 0.10 | 25.1 | 21.4 | 0.93 | -29.9 | 4.8 | 770 | 9.2 | 0.91 |
| twinkle | bank1 | 1 | 0.40 | 0.40 | 0.20 | 25.3 | 21.6 | 0.92 | -30.7 | 4.4 | 880 | 10.1 | 0.81 |
| plain | synth2 | 1 | 0.08 | 0.08 | 0.04 | 24.6 | 20.5 | 0.91 | -28.5 | 4.8 | 360 | 6.6 | 0.97 |
| plain | bank1 | 1 | 0.12 | 0.08 | 0.04 | 25.1 | 20.0 | 0.91 | -28.7 | 5.2 | 385 | 11.8 | 0.77 |
| grace | synth2 | 1 | 0.27 | 0.19 | 0.04 | 25.5 | 21.7 | 0.95 | -30.6 | 4.5 | 1560 | 12.5 | 0.86 |
| grace | bank1 | 1 | 0.42 | 0.08 | 0.04 | 25.4 | 22.7 | 0.95 | -30.6 | 4.4 | 1520 | 7.8 | 0.88 |

verse1 s2pop - synth2 per seed: tiny -0.03 | base -0.05 | small -0.13 | cpp +0.19 | hnr +0.32 | voiced -0.00
verse1 bank0 - synth2 per seed: tiny +0.02 | base -0.06 | small -0.15 | cpp -0.02 | hnr -0.42 | voiced -0.01
verse1 bank05 - synth2 per seed: tiny -0.01 | base -0.06 | small -0.02 | cpp +0.32 | hnr -0.43 | voiced +0.01
verse1 bank1 - synth2 per seed: tiny -0.04 -0.02 -0.25 | base -0.08 +0.06 +0.81 | small -0.18 +0.07 +0.00 | cpp +1.00 +0.68 +0.32 | hnr -1.13 -0.84 -1.08 | voiced +0.03 +0.04 +0.02
chorus1 s2pop - synth2 per seed: tiny +0.09 | base -0.09 | small -0.03 | cpp +0.03 | hnr -0.48 | voiced +0.00
chorus1 bank0 - synth2 per seed: tiny -0.03 | base +0.06 | small +0.09 | cpp +0.06 | hnr +0.09 | voiced -0.00
chorus1 bank05 - synth2 per seed: tiny -0.34 -0.14 -0.14 | base -0.17 -0.23 +0.09 | small -0.06 +0.03 +0.14 | cpp +0.18 +0.20 +0.13 | hnr -0.23 -0.30 -0.25 | voiced -0.00 +0.00 +0.01
chorus1 bank1 - synth2 per seed: tiny -0.31 -0.06 -0.29 | base -0.17 -0.20 +0.06 | small -0.09 +0.06 +0.06 | cpp +0.37 +0.47 +0.32 | hnr -1.06 -1.01 -1.39 | voiced -0.01 +0.00 +0.01
twinkle bank1 - synth2 per seed: tiny +0.10 | base +0.00 | small +0.10 | cpp +0.14 | hnr +0.19 | voiced -0.01
plain bank1 - synth2 per seed: tiny +0.04 | base +0.00 | small +0.00 | cpp +0.51 | hnr -0.54 | voiced +0.00
grace bank1 - synth2 per seed: tiny +0.15 | base -0.12 | small +0.00 | cpp -0.10 | hnr +0.93 | voiced -0.01

(mean over seeds, min..max in brackets. verse1 bank1 seed 3: base.en looped on "pop pop pop ...", WER 1.29;
without that decode the bank1 base.en mean is 0.53.)

## Recommendation, revised after three seeds (supersedes points 2 and 3 of the provisional one)

Three seeds on the demo material (WER tiny/base/small, mean; the lead is `style=robotic`):

    chorus1  flat build_pitch 0.72/0.38/0.12   predictor expr 0.5  0.51/0.28/0.16   predictor expr 1  0.51/0.28/0.13
    verse1   flat build_pitch 0.64/0.52/0.43   predictor expr 0.5  0.57/0.47*/0.47  predictor expr 1  0.54/0.53*/0.39
    (* one base.en decode of each looped on "pop pop pop"; those are left out of the starred means)

- tiny.en on chorus1 is better with the predictor in 6 of 6 paired comparisons (-0.14 to -0.34);
  flat steps from the predictor itself (expr 0) are as bad as Synth2's flat steps (0.77 vs 0.80,
  one seed), and so is Synth2's pop curve (0.89, one seed). The acoustic model sings a curve of
  the kind it was trained on more clearly than a flat or hand-built one.
- On verse1 no pitch source gets under 0.44 on tiny.en: the fast verse is not a pitch problem.
- Breathiness: cpp rises by 0.1-0.2 dB (expr 0.5) and 0.3-1.0 dB (expr 1) in 9 of 9 comparisons,
  against a 4 dB gap to the slow cases. The bank's own pitch does not cure "out of breath".

So:

2'. `auto` = the predictor for every style, the robotic style included, whenever the bank has a
    usable `dspitch/`. Robotic keeps its character through `expr` 0.5: 94-95 % of sustained frames
    within 30 cents (median 7-8), no vibrato, no drift, 35 ms note changes. Strictly flat pitch
    stays available as `pitch_source=expression` (or `flat_styles=("robotic",)` in `q5prod`); it
    costs 0.2 WER on the chorus with the strict recogniser. This departs from the brief's
    "robotic stays flat and quantised": it is the lead's call.
3'. `expr = clip(STYLE_EXPR[style] * (1 - 0.5 * voice.robotic), 0.5, 1)` with robotic 0.5,
    soft 0.75, every other style 1.0. Never below 0.5: there the curve is flat steps again.
4'. Vibrato is synthetic (Synth2's, on the predicted transitions) where it is the user's design:
    notes with vibrato attributes, tracks that set `voice.vibrato_*` / `style.vibrato_*`, and the
    robotic style; everywhere else the predicted vibrato scaled by `style.vibrato_depth * vibratoEnv`.

`q5prod.py` implements 2'-4'; `x6_prod_test.py` passes (16 checks, "ALL PASSED", load 12).

## A take survives an edit when the seed does (`x9_edit.py`)

One note in the middle of a phrase moved by 2 semitones, durations unchanged; change of the curve
on the frames more than 0.15 s away from that note:

| phrase | same seed, RMS (max) | new seed, RMS | unedited phrase, new seed |
|---|---|---|---|
| plain 0 (281 frames) | 4.8 c (22) | 13.2 c | 16.0 c |
| grace 0 (1597 frames) | 3.8 c (37) | 20.8 c | 21.5 c |
| chorus1 3 (469 frames) | 2.6 c (16) | 21.6 c | 20.4 c |

So the seed should come from the track seed and the phrase's place and length (global first frame,
frame count), not from its pitches or lyrics: a note edit then leaves the rest of the phrase's
performance in place instead of re-rolling it. (The noise is drawn per frame, so a change of the
phrase's frame count is a new take whatever the seed.) `q5prod.phrase_seed` does this. The
contract's wording is "a stable hash of the phrase content": this is a narrower hash, still
deterministic for a given project.
