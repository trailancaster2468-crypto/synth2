"""Render bars of SELECT ALL INVADERS with Qixuan through the combined engine (demo 3).

usage: cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 python3 -I $S/demo3/run_demo.py OUT START_BAR END_BAR
           [--key N] [--cast FILE.json] [--boost name=dB,...] [--lead-eq formant|JSON] [--formant] [--set JSON]

OUT/preview/ gets the mix (select_all_invaders_bars_A-B.wav), the stems (stems_bars_A-B/), the build's report
and cast_bars_A-B.json (one row per sung track: character, gender, register shift, gain).
--key: semitones the whole song is moved by (default -5; 0 = the written key, A minor).
--cast: JSON {track-name prefix: {gender, shift, tone, doubles, opts, gain_db}} on top of ds_backend3.CAST.
--set: JSON merged into build3.MASTER (e.g. '{"vocal_balance_db": 6.0}').
--formant renders with the built-in formant engine instead (the song as it was before the neural engine).
"""
import argparse
import json
import os
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
HERE = S / "demo3"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("a", type=int)
ap.add_argument("b", type=int)
ap.add_argument("--key", type=int, default=-5)
ap.add_argument("--cast", default=None)
ap.add_argument("--boost", default="")
ap.add_argument("--lead-eq", default=None)
ap.add_argument("--formant", action="store_true")
ap.add_argument("--set", default=None)
a = ap.parse_args()

os.environ["DEMO3_KEY"] = str(a.key)
if a.cast:
    os.environ["DEMO3_CAST"] = a.cast
if a.lead_eq:
    os.environ["DEMO3_LEAD_EQ"] = a.lead_eq
os.environ["SYNTH2_ENGINE"] = "formant" if a.formant else ""
sys.path.insert(0, "/home/user/synth2")
sys.path.insert(0, "/home/user/synth2/examples/select_all_invaders")
sys.path.insert(0, str(HERE))

import build3  # noqa: E402

if a.set:
    for k, v in json.loads(a.set).items():
        if isinstance(v, dict) and isinstance(build3.MASTER.get(k), dict):
            build3.MASTER[k].update(v)
        else:
            build3.MASTER[k] = v
be = None
if not a.formant:
    import ds_backend3
    from synth2.engine.backend import register_backend

    boost = {k: float(v) for k, v in (kv.split("=") for kv in a.boost.split(",") if kv)}
    be = ds_backend3.DSBackend(boost_db=boost, threads=int(os.environ.get("SYNTH2_NEURAL_THREADS", "2")))
    register_backend(be)
rc = build3.main(["--start-bar", str(a.a), "--end-bar", str(a.b), "--out", a.out, "--no-mp3", "--no-pitch"])
if be is not None:
    (Path(a.out) / "preview" / f"cast_bars_{a.a}-{a.b}.json").write_text(json.dumps(be.rows, indent=1))
sys.exit(rc)
