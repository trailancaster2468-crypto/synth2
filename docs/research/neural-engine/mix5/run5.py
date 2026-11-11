"""The song of the fourth demo: prod/build4's render with mix5/build5 (one kind per track, roles) and mix5/ds_backend5.

usage: cd /home/user/synth2 && SYNTH2_NEURAL_THREADS=2 python3 -I $S/mix5/run5.py OUT A B [--arr before|after]
           [--verse written|relaxed] [--set JSON]
A, B: first and last bar; `A B` = 1 134 renders the whole song (OUT/, with mix_levels.json), any other range a
section (OUT/preview/). --arr before = the third demo's build and backend (prod/build4.py, prod/ds_backend4.py; its
takes are read from prod/cache and new ones written to mix5/cache4).
OUT[/preview] gets the mix, the stems (post-fader, pre-reverb), the build's report and tracks_bars_A-B.json (every
vocal channel with its role, fader, pan and sends)."""
import argparse
import json
import os
import sys
from pathlib import Path

S = Path("/tmp/claude-0/-home-user-synth2/b5be9e31-3e71-50d3-9169-dace57471e89/scratchpad")
ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("a", type=int)
ap.add_argument("b", type=int)
ap.add_argument("--arr", default="after")
ap.add_argument("--verse", default="written")
ap.add_argument("--set", default=None, help='JSON merged into the master settings, e.g. {"loudness_target": -13.0}')
a = ap.parse_args()
os.environ.update({"DEMO3_KEY": "-5", "PROD_VERSE": a.verse, "PROD_CHAIN": "on", "PROD_DOUBLE": "adt", "PROD_ENGINE": "flow",
                   "SYNTH2_ENGINE": ""})
sys.path.insert(0, str(S / "mix5"))
import build5  # noqa: E402

build4 = build5.build4
if a.set:
    build4.MASTER.update(json.loads(a.set))
if a.arr == "before":
    import ds_backend4 as be_mod  # noqa: E402  (prod/)
    be_mod.CACHE = S / "mix5" / "cache4"
    be_mod.READ_CACHES = [be_mod.CACHE, S / "prod" / "cache"]
else:
    import ds_backend5 as be_mod  # noqa: E402
from synth2.engine.backend import register_backend  # noqa: E402

be = be_mod.DSBackend(threads=int(os.environ.get("SYNTH2_NEURAL_THREADS", "2")))
register_backend(be)
tracks = []
_build = build4.build_session


def build_session(log=print):
    s = build5.build_session(a.arr, log=log)
    mo = s.project.master.mix_overrides
    for t in s.project.tracks:
        o = mo.get(t.name, {})
        tracks.append({"track": t.name, "role": build5.role_of(t.name), "notes": len(t.notes), "seed": t.seed,
                       "gain_db": t.gain_db, "pan": t.pan, "reverb_send": t.reverb_send, "delay": o.get("delay", 1.0),
                       "voice": t.voice})
    return s


if a.arr != "before":
    build5.install()
    build4.build_session, _inner = build_session, build4.build_session
    build5.build4 = type("B4", (), {"build_session": staticmethod(_inner)})  # build5 calls the original
else:
    build4.build_session, _inner = build_session, build4.build_session
    build5.build4 = type("B4", (), {"build_session": staticmethod(_inner)})
# stems as FLAC (vocals 24 bit, band 16 bit), silent stems left out: the disk is small
import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

_render = build4.render_project


def render_project(*args, **kw):
    res = _render(*args, **kw)

    def save_stems(folder, bits=24):
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        from synth2.engine.renderer import _stem_name
        names = {t["id"]: t["name"] for t in res.report.get("tracks", [])}
        used: set = set()
        for tid, x in res.stems.items():
            name = _stem_name(names.get(tid, tid), tid, used)
            if not np.any(x):
                continue
            vocal = name.split("_")[0] in ("AI", "Harmony", "Aliens", "Alien", "Crowd")
            sf.write(folder / (name + ".flac"), np.clip(x, -1.0, 1.0), res.sample_rate,
                     format="FLAC", subtype="PCM_24" if vocal else "PCM_16")

    res.save_stems = save_stems
    return res


build4.render_project = render_project
whole = (a.a, a.b) == (1, 134)
if whole:
    # 47 vocal channels and 21 band stems of 191 s: the mix of the whole song was killed for lack of memory with
    # four mixer threads. One thread, and the singer's models are dropped before the mix starts.
    import gc
    from synth2.engine import mixing as _mx
    os.environ["SYNTH2_MIX_THREADS"] = "1"
    _mix = _mx.mix_sources

    def mix_sources(*args, **kw):
        be.ds = None
        getattr(be, "_env", {}).clear()
        gc.collect()
        return _mix(*args, **kw)

    _mx.mix_sources = mix_sources
    # resident memory every 2 s in OUT.rss (the last line is the latest value and the peak so far, GB)
    import threading
    import time

    def _watch(path=str(a.out) + ".rss"):
        peak = 0.0
        while True:
            with open("/proc/self/status") as fh:
                rss = next(int(line.split()[1]) for line in fh if line.startswith("VmRSS")) / 1e6
            peak = max(peak, rss)
            with open(path, "w") as fh:
                fh.write(f"{rss:.2f} GB now, {peak:.2f} GB peak\n")
            time.sleep(2.0)

    threading.Thread(target=_watch, daemon=True).start()
argv = ["--out", a.out, "--no-mp3", "--no-pitch"] + ([] if whole else ["--start-bar", str(a.a), "--end-bar", str(a.b)])
rc = build4.main(argv)
dest = Path(a.out) if whole else Path(a.out) / "preview"
(dest / f"chain_bars_{a.a}-{a.b}.json").write_text(json.dumps(be.rows, indent=1))
(dest / f"tracks_bars_{a.a}-{a.b}.json").write_text(json.dumps(tracks, indent=1))
sys.exit(rc)
