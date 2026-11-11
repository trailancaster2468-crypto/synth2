"""Render bars of SELECT ALL INVADERS with Qixuan and the vocal production of prod/ (build4.py, ds_backend4.py).

usage: cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 python3 -I $S/prod/run4.py OUT START_BAR END_BAR
           [--verse written|relaxed] [--chain on|off] [--double adt|take] [--engine demo3|flow] [--room R]
           [--key N] [--boost name=dB,...] [--set JSON]

OUT/preview/ gets the mix (select_all_invaders_bars_A-B.wav), the stems (stems_bars_A-B/), the build's report
and cast_bars_A-B.json (one row per sung track: character, gender, register shift, gain).
--key: semitones the whole song is moved by (default -5; 0 = the written key, A minor).
--verse relaxed: prod/relaxed_verse.s2s in place of the lead's verse 1. --chain off: the vocal production of demo 3.
--double take: the double as a second take (demo 3) instead of a copy of the lead take (chain4.adt).
--set: JSON merged into build4.MASTER (e.g. '{"vocal_balance_db": 6.0}').
OUT/preview/chain_bars_A-B.json has what the chain did on every channel (gain curves, speed of the gain).
"""
import argparse
import json
import os
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
HERE = S / "prod"

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("a", type=int)
ap.add_argument("b", type=int)
ap.add_argument("--key", type=int, default=-5)
ap.add_argument("--verse", default="written")
ap.add_argument("--chain", default="on")
ap.add_argument("--double", default="adt")
ap.add_argument("--engine", default=None)
ap.add_argument("--room", default=None)
ap.add_argument("--boost", default="")
ap.add_argument("--set", default=None)
a = ap.parse_args()

os.environ["DEMO3_KEY"] = str(a.key)
os.environ["PROD_VERSE"] = a.verse
os.environ["PROD_CHAIN"] = a.chain
os.environ["PROD_DOUBLE"] = a.double if a.chain != "off" else "take"
if a.engine:
    os.environ["PROD_ENGINE"] = a.engine
if a.room:
    os.environ["PROD_ROOM"] = a.room
os.environ["SYNTH2_ENGINE"] = ""
sys.path.insert(0, "/home/user/synth2")
sys.path.insert(0, "/home/user/synth2/examples/select_all_invaders")
sys.path.insert(0, str(HERE))

import build4  # noqa: E402

if a.set:
    for k, v in json.loads(a.set).items():
        if isinstance(v, dict) and isinstance(build4.MASTER.get(k), dict):
            build4.MASTER[k].update(v)
        else:
            build4.MASTER[k] = v
import ds_backend4  # noqa: E402
from synth2.engine.backend import register_backend  # noqa: E402

boost = {k: float(v) for k, v in (kv.split("=") for kv in a.boost.split(",") if kv)}
be = ds_backend4.DSBackend(boost_db=boost, threads=int(os.environ.get("SYNTH2_NEURAL_THREADS", "2")), chain=a.chain != "off")
register_backend(be)
rc = build4.main(["--start-bar", str(a.a), "--end-bar", str(a.b), "--out", a.out, "--no-mp3", "--no-pitch"])
(Path(a.out) / "preview" / f"chain_bars_{a.a}-{a.b}.json").write_text(json.dumps(be.rows, indent=1))
sys.exit(rc)
