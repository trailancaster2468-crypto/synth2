#!/bin/bash
# The demo's mixes: first chorus (bars 33-48) and first verse (bars 9-24), song 5 semitones lower, Qixuan cast.
# usage: final_mix.sh [OUTDIR]   (default $S/demo3/mix/final). Any bars: run_demo.py OUT A B with the same options.
S=/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad
OUT=${1:-$S/demo3/mix/final}
cd /home/user/synth2
export SYNTH2_NEURAL_THREADS=2
# lead EQ for the neural voice: half the 600 Hz cut, no 4.8 kHz bell, a shelf for the consonants
EQ='[["peak", 600, -2.0, 0.9], ["peak", 3150, -2.5, 1.4], ["highshelf", 6500, 3.0, 0.7]]'
# hats 1.5 dB lower and ducked under the vocals, snare top 2 dB lower than the song's setting
DR='"mix_overrides": {"drums hats": {"gain_db": -1.5, "duck": 1.5}, "drums snare": {"gain_db": -2.0, "carve": 1.0, "eq_add": [["peak", 3000, -2.0, 0.8], ["highshelf", 5500, -6.5, 0.7]]}}'
# vocals 8.0 dB over the band in 1-5 kHz (the song: 5.2)
timeout 900 python3 -I $S/demo3/run_demo.py $OUT 33 48 --lead-eq "$EQ" --set "{$DR, \"vocal_balance_db\": 8.0}" > $OUT.33.log 2>&1
echo "chorus exit $? $(grep '^master:' $OUT.33.log | cut -c1-110)"
# the verse is mastered 1.5 LU under the chorus (a section render is mastered on its own)
timeout 900 python3 -I $S/demo3/run_demo.py $OUT 9 24 --lead-eq "$EQ" --set "{$DR, \"vocal_balance_db\": 8.0, \"loudness_target\": -13.0}" > $OUT.9.log 2>&1
echo "verse exit $? $(grep '^master:' $OUT.9.log | cut -c1-110)"
for ab in "33 48" "9 24"; do timeout 120 python3 -I $S/demo3/mix_check.py $OUT $ab | cut -c1-330; done
