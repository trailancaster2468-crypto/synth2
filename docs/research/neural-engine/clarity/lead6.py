"""The lead of bars A-B in the song (the fourth demo's arrangement), alone, with a dial setting: the lead channels
through the backend (engine take, loudness lane, chain, level match, the in-channel room if the dial has one), at
their faders, then with the mixer's reverb and delay returns for the lead alone. Measured like every row of lab.py
(flow, voiceq, cm.py against the take's own phoneme times, recognisers).

usage: cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/clar/lead6.py NAME A B [--dial A|B|C]
           [--dial-json JSON] [--wer CASE] [--seed N] [--tag T] [--small] [--no-room-row]
Writes clar/lead/NAME_DIAL_chain.wav (mono: the channel as it goes to the mix bus, before the returns) and
NAME_DIAL_room.wav (stereo: with the lead's own reverb and delay returns), NAME_DIAL.dump.json (phoneme times,
relative to the clip's start) and two rows in clar/rows.jsonl (set = "lead").
--seed N replaces the lead tracks' seed (another take of the same notes).
"""
import argparse
import json
import os
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
P = S / "clar"
ap = argparse.ArgumentParser()
ap.add_argument("name")
ap.add_argument("a", type=int)
ap.add_argument("b", type=int)
ap.add_argument("--dial", default="A")
ap.add_argument("--dial-json", default=None)
ap.add_argument("--wer", default=None)
ap.add_argument("--seed", type=int, default=None)
ap.add_argument("--tag", default=None)
ap.add_argument("--small", action="store_true")
ap.add_argument("--no-room-row", action="store_true")
a = ap.parse_args()
os.environ.update({"DEMO3_KEY": "-5", "PROD_VERSE": "written", "PROD_CHAIN": "on", "PROD_DOUBLE": "adt",
                   "PROD_ENGINE": "flow", "SYNTH2_ENGINE": "", "CLAR_DIAL": a.dial})
if a.dial_json:
    os.environ["CLAR_DIAL_JSON"] = a.dial_json
sys.path.insert(0, "/home/user/synth2")
sys.path.insert(0, str(S / "mix5"))
sys.path.insert(0, str(P))
import numpy as np
import soundfile as sf

import ds_backend6 as be_mod
import cm
import lab  # clar's (build5 below puts prod/, which has a lab.py of its own, in front)
import fxlab
import build5

sys.modules["ds_backend5"] = be_mod
build5.install()
s = build5.build_session("after", log=lambda *_: None)
project = s.project
b2s = project.tempo.beat_to_sec
t0, t1 = b2s(4.0 * (a.a - 1)), b2s(4.0 * a.b)
MIX = dict(be_mod._dial.get("mix") or {})
if MIX.get("room"):
    be_mod.ROLE_FX["lead"] = dict(be_mod.ROLE_FX.get("lead") or {}, room=MIX["room"])
be = be_mod.DSBackend(threads=int(os.environ.get("SYNTH2_NEURAL_THREADS", "2")), log=lambda *_: None)
sr = be_mod.dsinfer.SR
n = int(round((t1 - t0) * sr))
chan = np.zeros(n, np.float32)
send = 0.0
for tr in project.tracks:
    if tr.name in be_mod.LEAD and any(4.0 * (a.a - 1) - 2 <= nt.onset < 4.0 * a.b for nt in tr.notes):
        if a.seed is not None:
            tr.seed = a.seed
        y, _ = be.render_track(project, tr, sr, t0, t1)
        chan += y[:n] * 10 ** (tr.gain_db / 20)
        send = float(MIX.get("reverb_send", tr.reverb_send))
room = fxlab.with_fx(chan, sr, room=project.master.reverb_room, send=send, delay=project.master.vocal_delay,
                     beats=project.master.delay_beats, fb=project.master.delay_feedback,
                     duck=project.master.delay_duck_db, small=None).astype(np.float32)
D = P / "lead"
D.mkdir(exist_ok=True)
tag = a.tag or a.dial
phones = []
for name, files in be.dumps.items():
    for f in sorted(files):
        phones += cm.dump_phones(f, t0)
phones = sorted({(round(x, 4), round(y, 4), p) for x, y, p in phones if y > 0 and x < t1 - t0})
(D / f"{a.name}_{tag}.dump.json").write_text(json.dumps(phones))
models = (("tiny.en", "base.en", "small.en") if a.small else ("tiny.en", "base.en")) if a.wer else None
for kind, x in (("chain", chan), ("room", room)):
    sf.write(D / f"{a.name}_{tag}_{kind}.wav", x, sr, subtype="PCM_16")
    if kind == "room" and a.no_room_row:
        continue
    r = lab.measure(x, sr, phones, a.wer, models)
    with open(lab.ROWS, "a") as fh:
        fh.write(json.dumps(dict(r, set="lead", name=f"{a.name} {tag} {kind}", take="song", case=a.wer, seed=a.seed,
                                 dial=a.dial, send=send, engine=be_mod.LEAD_OPTS, chain_cfg=be_mod.LEAD_CHAIN, mix=MIX,
                                 chain={k: v for k, v in be.info.items() if k in be_mod.LEAD})) + "\n")
    print(lab.line(f"{a.name} {tag} {kind}", r), flush=True)
    for k, h in (r.get("hyps") or {}).items():
        print(f"    {k}: " + h[:300], flush=True)
