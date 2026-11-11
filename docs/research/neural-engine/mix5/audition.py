"""Audition takes of the stamped "click": for a few values of the engine's `salt` option, sing the click channels
of bars 49-52 dry and count how often three recognisers write "click".

usage: cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 ASR_THREADS=2 python3 -I $S/mix5/audition.py [SALT ...]
A stamped hit repeats one take, so a weak take is weak every time; the salt gives another take of that sound.
Per channel and salt: "click" found by tiny.en / base.en / small.en in bar 52 (the run of four on eighth notes) and,
for the crowd, in bars 49-51 (the three pairs). Rows are written to audition.json. AUD_CH=name,name limits the channels."""
import copy
import json
import os
import re
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
os.environ.update({"PROD_ENGINE": "flow", "SYNTH2_ENGINE": ""})
for p in (str(S / "mix5"), str(S / "tools")):
    sys.path.insert(0, p)
import build5  # noqa: E402
import ds_backend5 as be_mod  # noqa: E402
import asr  # noqa: E402

SR = 44100
salts = [int(x) for x in sys.argv[1:]] or [0, 1, 2, 3]
s = build5.build_session("after", log=lambda *a: None)
p = s.project
b2s = p.tempo.beat_to_sec
bar = lambda b: b2s(4.0 * (b - 1))  # noqa: E731
rows = []
for salt in salts:
    cast = copy.deepcopy(be_mod.CAST)
    if salt:
        for k in ("AI Click", "Alien Click", "Crowd Click", "Crowd Click 2", "Crowd Click 3"):
            c = cast[k] = copy.deepcopy(cast[k])
            c["opts"]["adlib"] = dict(c["opts"]["adlib"], salt={"click": salt})
    if os.environ.get("AUD_OPTS"):  # JSON merged into the ad-lib options of the auditioned channels
        for k in ("AI Click", "Alien Click", "Crowd Click", "Crowd Click 2", "Crowd Click 3"):
            c = cast[k] = copy.deepcopy(cast[k])
            c["opts"]["adlib"] = dict(c["opts"]["adlib"], **json.loads(os.environ["AUD_OPTS"]))
    be = be_mod.DSBackend(cast=cast, threads=int(os.environ.get("SYNTH2_NEURAL_THREADS", "2")), log=lambda *a: None)
    for name in [c for c in os.environ.get("AUD_CH", "AI Click,Alien Click,Crowd Click,Crowd Click 2,Crowd Click 3").split(",")]:
        tr = p.track(name)
        t0, t1 = bar(49) - 0.3, bar(53) + 0.3
        x, _ = be.sing(p, tr, window=(t0, t1))
        row = {"salt": salt, "channel": name, "opts": os.environ.get("AUD_OPTS", "")}
        for tag, a, b in (("bar52", bar(52) - 0.25, bar(52.5) + 0.35), ("pairs", bar(49) - 0.25, bar(51.5) + 0.35)):
            seg = x[int(a * SR): int(b * SR)]
            if not seg.any():
                continue
            hyp = {m: asr.transcribe(seg, SR, m) for m in ("tiny.en", "base.en", "small.en")}
            row[tag] = [len(re.findall(r"\bclicks?\b", h.lower())) for h in hyp.values()]
            row[tag + "_hyp"] = hyp
        rows.append(row)
        print(salt, name, "bar52", row.get("bar52"), "pairs", row.get("pairs"), "|", (row.get("bar52_hyp") or {}).get("base.en"),
              "|", (row.get("bar52_hyp") or {}).get("small.en"), flush=True)
    f = S / "mix5" / "audition.json"
    done = {(x["salt"], x["channel"], x.get("opts", "")) for x in rows}
    old = [r for r in (json.loads(f.read_text()) if f.exists() else []) if (r["salt"], r["channel"], r.get("opts", "")) not in done]
    f.write_text(json.dumps(old + rows, indent=1, ensure_ascii=False))
