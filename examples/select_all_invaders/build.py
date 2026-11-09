"""Build, mix and render SELECT ALL INVADERS (セレクト・オール・インベーダー).

The project is put together through the studio's tool layer, the way an AI producer would drive it
(``synth2.ai.tools.Session`` + ``call_tool``):

1. ``new_project`` from the score text in ``vocals.s2s`` (the four hand-written vocal tracks),
2. ``vocals.apply_vocal_production``: AI Double, Harmony, Crowd 2 / 3 and the aliens' fan voice,
3. ``set_backing`` with ``backing.json`` (the arranger's band), then ``set_backing(merge=true)`` for the
   section tweaks of the mix (:data:`BAND_TWEAKS`),
4. ``duplicate_track`` / ``add_notes`` / ``delete_notes`` to put the verse and breakdown lead and the
   aliens' final-chorus hook on their own mixer channels (:data:`SPLITS`),
5. ``update_track`` for every vocal fader, pan and reverb send (:data:`TRACK_MIX`),
6. ``update_project`` for the master: loudness, delay, width, master EQ, band balance and the
   per-track / per-stem overrides (:data:`MASTER`),
7. ``save_project``, then ``render_project`` at 44.1 kHz.

The mix decisions are documented next to the settings below; every one of them came from a measurement
(section loudness, 1-5 kHz balance of each element against the lead, long-term spectrum, limiter activity).

It writes ``out/select_all_invaders/``: ``select_all_invaders.wav`` (24-bit), ``stems/`` (one
24-bit WAV per vocal track and band stem; post-fader, pre-reverb), ``select_all_invaders.s2p.json``,
``select_all_invaders.mp3`` (320 kbps, when ffmpeg is installed) and ``select_all_invaders.report.json``
(the render report plus the analysis printed at the end: loudness per section, true peak, the
vocal-to-band balance per section, the low-end balance, the long-term spectrum slope and the
pitch accuracy of every vocal track).

Usage (from the repository root)::

    python examples/select_all_invaders/build.py                          # the whole song
    python examples/select_all_invaders/build.py --start-bar 33 --end-bar 48   # a quick section render

A section render goes to ``out/select_all_invaders/preview/``. After a full render the mix levels
(band gain, compressor thresholds, master gain) are saved in ``mix_levels.json``, and a section render
of the same mix reuses them, so it is mastered like that part of the full song.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
try:
    import synth2  # noqa: F401
except ImportError:  # run from a checkout without installing
    sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from synth2.ai.tools import Session, call_tool  # noqa: E402
from synth2.analysis import pitch_accuracy  # noqa: E402
from synth2.engine import mixing, render_project  # noqa: E402
from synth2.engine import renderer as _renderer  # noqa: E402
from synth2.io.wav import write_wav  # noqa: E402

import vocals  # noqa: E402  (examples/select_all_invaders/vocals.py)

NAME = "select_all_invaders"
OUT = ROOT / "out" / NAME
BEATS_PER_BAR = 4

# SPEC.md: (section, first bar, last bar)
SECTIONS = [
    ("Intro", 1, 8), ("Verse 1", 9, 24), ("Pre-chorus 1", 25, 32), ("Chorus 1", 33, 48),
    ("Post-chorus", 49, 52), ("Verse 2", 53, 68), ("Pre-chorus 2", 69, 76), ("Chorus 2", 77, 92),
    ("Bridge", 93, 108), ("Breakdown", 109, 112), ("Final chorus", 113, 128), ("Outro", 129, 134),
]

# ------------------------------------------------------------------------------------------- the mix
# Some parts get their own mixer channel for a section (a "track split", as in any DAW), so they can sit
# differently there without fader automation:
# * "AI Verses": the lead rides 2 dB lower in the two verses, where the half-time band is thin (it stuck
#   out of it by ~9 dB in 1-5 kHz), which also gives the choruses more lift. In verse 2 the AI and the
#   aliens then trade lines at about the same level.
# * "AI Breakdown": the AI's unguarded "loading humanity..." moment (bars 109-112) sits lower, in more
#   reverb, so the breakdown drops away before the final chorus. The pickup "select" stays on the AI.
# * "Aliens Hook": in the final chorus the aliens sing the hook two octaves under the AI; on its own
#   channel it supports the lead instead of competing with it. Their answers ("click!", ワレワレハ) stay
#   on the main Aliens channel, at full level.
SPLITS: list[dict] = [
    {"source": "AI", "name": "AI Verses", "start": 32.0, "end": 96.0, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Verses", "start": 208.0, "end": 272.0, "skip_lyrics": ()},
    {"source": "AI", "name": "AI Breakdown", "start": 432.0, "end": 447.25, "skip_lyrics": ()},
    {"source": "Aliens", "name": "Aliens Hook", "start": 447.0, "end": 500.0, "skip_lyrics": ("click",)},
]

# Vocal faders (dB), pans and reverb sends. The AI lead is the anchor at 0 dB, dead centre; everything
# else supports it from the sides.
TRACK_MIX: dict[str, dict] = {
    "AI":           {"gain_db": 0.0, "pan": 0.0, "reverb_send": 0.22},
    "AI Verses":    {"gain_db": -2.0, "pan": 0.0, "reverb_send": 0.26},
    "AI Breakdown": {"gain_db": -2.5, "pan": 0.0, "reverb_send": 0.38},
    "AI Double":    {"gain_db": -6.5, "pan": -0.35, "reverb_send": 0.22},
    "Harmony":      {"gain_db": -6.5, "pan": 0.35, "reverb_send": 0.30},
    "Aliens":       {"gain_db": -1.0, "pan": 0.25, "reverb_send": 0.30},
    "Aliens Hook":  {"gain_db": -2.5, "pan": 0.25, "reverb_send": 0.30},
    "Aliens 2":     {"gain_db": -7.0, "pan": -0.35, "reverb_send": 0.40},
    "Crowd":        {"gain_db": -6.0, "pan": -0.60, "reverb_send": 0.35},
    "Crowd 2":      {"gain_db": -6.0, "pan": 0.60, "reverb_send": 0.35},
    "Crowd 3":      {"gain_db": -8.0, "pan": 0.25, "reverb_send": 0.35},
}

# Master / mix settings (MasterSettings fields, set with update_project). mix_overrides are keyed by
# vocal track name or backing stem name ("drums snare" is the snare sub-stem of the "drums" kit).
LEAD_EQ = [["peak", 600, -4.0, 0.9], ["peak", 3150, -2.5, 1.4], ["highshelf", 4500, 2.0, 0.7]]
MASTER: dict = {
    "loudness_target": -11.0,
    "ceiling_db": -1.0,
    "glue": 0.45,              # a little lighter master glue keeps the verse / chorus contrast
    "vocal_balance_db": 4.5,
    "band_width": 1.3,         # the band was narrow (side 9-11 dB under mid): open the L/R guitars and synths
    # master EQ (folded into every source): the default subsonic high-pass, a little 250 Hz warmth (every role
    # preset scoops 250-450 Hz, which left a 4 dB hole in the long-term spectrum) and a softer air shelf
    "master_eq": [["hp", 25, 0, 0.7071, 2], ["peak", 250, 1.5, 1.2], ["highshelf", 12000, 0.5, 0.7]],
    "vocal_delay": 0.15,       # dotted-8th ping-pong on the vocals, ducked while they sing
    "delay_beats": 0.75,
    "delay_feedback": 0.3,
    "mix_overrides": {
        # lead EQ on top of the channel strip: its fundamental (A4-E5 = 440-660 Hz) piled up as a honky
        # bump in the long-term spectrum, the strip's 3.2 kHz presence bell stuck out of it as a narrow
        # peak, and the voice falls off a cliff above 4 kHz, so: less 600 Hz, a softer 3.2 kHz, more sheen
        "AI": {"eq_add": LEAD_EQ}, "AI Verses": {"eq_add": LEAD_EQ},
        "AI Breakdown": {"eq_add": LEAD_EQ, "delay": 1.4},
        "AI Double": {"eq_add": LEAD_EQ, "delay": 0.4},
        # delay sends (1 = default): the lead gets the full send, the gang shouts stay dry and punchy.
        # crest_db: the aliens' buzzy low pulse train peaks 16 dB over its loud level (the AI: 10 dB), and those
        # spikes made the master limiter duck the whole mix by up to 5 dB; the shouts are peaky too
        "Harmony": {"delay": 0.5}, "Aliens 2": {"delay": 0.8},
        "Aliens": {"delay": 0.6, "crest_db": 10.0}, "Aliens Hook": {"delay": 0.3, "crest_db": 10.0},
        "Crowd": {"delay": 0.0, "crest_db": 12.0}, "Crowd 2": {"delay": 0.0, "crest_db": 12.0},
        "Crowd 3": {"delay": 0.0, "crest_db": 12.0},
        # band balance: the snare was the main competitor of the lead in 1-5 kHz and the brightest thing in
        # the top octave; the half-time verses were thin, so their plucks, slap bass and chugs come up
        "drums snare": {"gain_db": -2.0, "eq_add": [["peak", 3000, -2.0, 0.8], ["highshelf", 8000, -2.0, 0.7]]},
        "pluck": {"gain_db": 6.0, "crest_db": 11.0},
        "slap": {"gain_db": 1.5},
        "mute": {"gain_db": 2.0},
        # the chip-lead riff was level with the voices in the riff sections (the aliens' spam in the intro, the
        # post-chorus calls, the aliens' one-star review in the outro), and the intro's radio static with it
        "lead": {"gain_db": -3.5},
        "fx": {"gain_db": -1.5},
        "mbox": {"gain_db": -2.0},
        # the final chorus is the only chorus with the pad and the low chip lead: a little more of both
        "pad": {"gain_db": 3.0},
        "leadlo": {"gain_db": 2.0},
        # the kick's 50-60 Hz was a bit boomy against the 80-250 Hz body of the mix
        "drums kick": {"eq_add": [["lowshelf", 60, -1.5, 0.7]]},
    },
}


# Section-level band tweaks, applied with set_backing(merge=true) on top of backing.json: velocity
# multipliers for the parts of a section. The final chorus used the same band levels as choruses 1 and 2,
# so it was barely louder after the master chain; its guitars, supersaws, drums and bass dig in harder.
BAND_TWEAKS: dict[str, dict[str, float]] = {
    "Final chorus": {"gtrL": 1.12, "gtrR": 1.12, "saws": 1.15, "drums": 1.06, "bass": 1.06},
}


def tweak_sections(arrangement: dict, tweaks: dict[str, dict[str, float]]) -> list[dict]:
    """Copies of the arrangement's sections named in ``tweaks`` with the parts' velocities scaled."""
    out = []
    for sec in arrangement.get("sections", []):
        if sec.get("name") not in tweaks:
            continue
        sec = json.loads(json.dumps(sec))
        for part, mult in tweaks[sec["name"]].items():
            spec = sec.get("parts", {}).get(part)
            if spec is None:
                raise KeyError(f"section {sec['name']!r} has no part {part!r}")
            segs = spec if isinstance(spec, list) else [spec]
            segs = [{"pattern": g} if isinstance(g, str) else g for g in segs]
            for g in segs:
                g["velocity"] = round(float(g.get("velocity", 1.0)) * mult, 3)
            sec["parts"][part] = segs if isinstance(spec, list) else segs[0]
        out.append(sec)
    return out


def split_track(s: Session, source: str, name: str, start: float, end: float, skip_lyrics=()) -> int:
    """Move the notes of ``source`` with onsets in [start, end) to a new track ``name`` (same voice,
    settings and curves). Returns the number of notes moved."""
    src = s.project.track(source)
    pick = lambda n: start <= n.onset < end and (n.lyric or "").lower() not in skip_lyrics  # noqa: E731
    moved = [n for n in src.notes if pick(n)]
    if not moved:
        return 0
    if any(t.name == name for t in s.project.tracks):  # a second range for an existing split channel
        call_tool(s, "add_notes", {"track": name, "notes": [
            {"onset": n.onset, "duration": n.duration, "pitch": n.pitch, "lyric": n.lyric,
             **({"phonemes": n.phonemes} if n.phonemes else {}), **({"attrs": n.attrs} if n.attrs else {})}
            for n in moved]})
    else:
        call_tool(s, "duplicate_track", {"track": source, "name": name})
        dst = s.project.track(name)
        keep = {(n.onset, n.pitch) for n in moved}
        call_tool(s, "delete_notes", {"track": name, "ids": [n.id for n in dst.notes
                                                             if (n.onset, n.pitch) not in keep]})
        tracks = s.project.tracks  # put the split channel right after its source in the mixer
        tracks.remove(dst)
        tracks.insert(tracks.index(src) + 1, dst)
    call_tool(s, "delete_notes", {"track": source, "ids": [n.id for n in moved]})
    return len(moved)


# ------------------------------------------------------------------------------------------- build
def build_session(log=print) -> Session:
    """The finished project in a tool session (vocals, production, backing, mixer, master)."""
    s = Session()
    r = call_tool(s, "new_project", {"score": (HERE / "vocals.s2s").read_text(encoding="utf-8")})
    for w in r.get("warnings", []) or []:
        log(f"  score warning: {w}")
    prod = vocals.apply_vocal_production(s)
    log(f"  vocal production: {len(prod['tracks'])} tracks, AI Double {prod['ai_double_notes']} notes, "
        f"Harmony {prod['harmony_notes']} notes, fan-voice regions {prod['fan_chop_regions']}")
    arrangement = json.loads((HERE / "backing.json").read_text(encoding="utf-8"))
    r = call_tool(s, "set_backing", {"arrangement": arrangement})
    for w in r.get("warnings", []):
        log(f"  backing {w}")
    if BAND_TWEAKS:
        r = call_tool(s, "set_backing", {"arrangement": {"sections": tweak_sections(arrangement, BAND_TWEAKS)},
                                         "merge": True})
        for w in r.get("warnings", []):
            log(f"  backing {w}")
        log(f"  band tweaks: {BAND_TWEAKS}")
    for sp in SPLITS:
        moved = split_track(s, sp["source"], sp["name"], sp["start"], sp["end"], sp["skip_lyrics"])
        log(f"  split {moved} notes of {sp['source']} (beats {sp['start']:g}-{sp['end']:g}) to {sp['name']}")
    for name, mix in TRACK_MIX.items():
        call_tool(s, "update_track", {"track": name, **mix})
    call_tool(s, "update_project", {"master": MASTER})
    return s


# ------------------------------------------------------------------------------------------- level cache
def _mix_key(project, sr: int) -> str:
    """A fingerprint of the mix that is stable across builds (the renderer's own cache key uses track ids,
    which are new on every build): master settings, backing, mixer and notes by track name."""
    import hashlib

    tracks = [(t.name, t.voice, t.gain_db, t.pan, t.mute, t.solo, t.reverb_send,
               [(round(n.onset, 4), n.pitch, n.lyric) for n in t.notes]) for t in project.tracks]
    blob = json.dumps({"master": project.master.to_dict(), "backing": project.backing, "tracks": tracks, "sr": sr,
                       "tempo": project.tempo.to_list()}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()


def save_levels(out: Path, project, sr: int, report: dict) -> None:
    """Remember the full render's level decisions, so later section renders are mastered alike."""
    levels = {k: report["mix"].get(k) for k in _renderer._LEVEL_KEYS}
    (out / "mix_levels.json").write_text(json.dumps({"key": _mix_key(project, sr), "levels": levels}, indent=1))


def load_levels(out: Path, project, sr: int) -> bool:
    """Prime the renderer's level cache from ``mix_levels.json`` when it matches this mix."""
    try:
        d = json.loads((out / "mix_levels.json").read_text())
    except (OSError, ValueError):
        return False
    if d.get("key") != _mix_key(project, sr) or any(v is None for v in d.get("levels", {}).values()):
        return False
    _renderer._LEVEL_CACHE[_renderer._level_key(project, sr)] = d["levels"]
    return True


# ------------------------------------------------------------------------------------------- analysis
def _mono(x: np.ndarray) -> np.ndarray:
    return x.mean(axis=1) if x.ndim == 2 else x


def _band_power(x: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    """Grid (1 ms) power of ``x`` band-limited to lo..hi Hz."""
    y = mixing.eq(_mono(x).astype(np.float32), [("hp", lo, 0, 0.7071, 4), ("lp", hi, 0, 0.7071, 4)], sr,
                  threads=mixing.mix_threads())
    return mixing.block_power(y, mixing.grid_hop(sr))


def _ratio_db(pa: np.ndarray, pb: np.ndarray, mask: np.ndarray, overlap: bool = True) -> float | None:
    """Power ratio (dB) of ``pa`` over ``pb`` on the grid blocks in ``mask`` (and, with ``overlap``,
    only where ``pb`` is not silent)."""
    m = mask & (pb > 1e-14) if overlap else mask
    if m.sum() < 200:  # under 0.2 s
        return None
    return round(float(10 * np.log10(max(pa[m].sum(), 1e-30) / max(pb[m].sum(), 1e-30))), 1)


def _act(x: np.ndarray, sr: int, below_db: float = 30.0) -> np.ndarray:
    """Grid activity (bool) of a stem: 10 ms RMS within ``below_db`` of its loud (95th pct) level."""
    lvl = mixing.detector_db(x, sr, "rms", 0.01)
    on = lvl[lvl > -90]
    if not len(on):
        return np.zeros(len(lvl), dtype=bool)
    return lvl > np.percentile(on, 95) - below_db


def ltas(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Long-term average spectrum in 1/3-octave bands (mean PSD in dB) and its slope in dB/octave
    (least squares over 100 Hz - 10 kHz)."""
    m = _mono(x).astype(np.float64)
    nfft = 8192
    win = np.hanning(nfft)
    hop = nfft // 2
    k = max(1, (len(m) - nfft) // hop)
    acc = np.zeros(nfft // 2 + 1)
    for i in range(0, k, 2):  # every other frame is plenty
        fr = m[i * hop:i * hop + nfft]
        if len(fr) < nfft:
            break
        acc += np.abs(np.fft.rfft(fr * win)) ** 2
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    centres = 1000.0 * 2.0 ** (np.arange(-17, 14) / 3.0)  # 20 Hz .. 20 kHz
    psd = []
    for c in centres:
        b = (f >= c * 2 ** (-1 / 6)) & (f < c * 2 ** (1 / 6))
        psd.append(10 * np.log10(acc[b].mean() + 1e-30) if b.any() else np.nan)
    psd = np.array(psd)
    sel = (centres >= 100) & (centres <= 10000) & np.isfinite(psd)
    slope = float(np.polyfit(np.log2(centres[sel]), psd[sel], 1)[0])
    return centres, psd, slope


LEAD_TRACKS = ("AI", "AI Verses", "AI Breakdown")  # the lead vocal (on two mixer channels)


def analyse(project, res, start_bar: int, end_bar: int, with_pitch: bool = True) -> dict:
    """Section loudness and balance, low end, spectrum slope and pitch accuracy of a render.

    Per section: integrated loudness, the vocal-to-band ratio in 1-5 kHz while the vocals sing (all vocals vs
    the band, as the mixer measures it), the same for the lead alone (while the lead sings), the lead against
    the sum of the other vocals, and the lead's margin over the loudest single other element (any vocal track
    or band stem) in 1-5 kHz, with that element's name.
    """
    sr = res.sample_rate
    hop = mixing.grid_hop(sr)
    rep = res.report
    groups = {s["key"]: s["group"] for s in rep["stem_levels"]}
    names = {t.id: t.name for t in project.tracks}
    n = len(res.audio)
    vocal = np.zeros((n, 2), dtype=np.float32)
    band = np.zeros((n, 2), dtype=np.float32)
    lead = np.zeros((n, 2), dtype=np.float32)
    lead_keys = {t.id for t in project.tracks if t.name in LEAD_TRACKS}
    for k, v in res.stems.items():
        (vocal if groups.get(k) == "vocal" else band)[: len(v)] += v[:n]
        if k in lead_keys:
            lead[: len(v)] += v[:n]
    lo_hz, hi_hz = mixing.VBR_BAND
    pv, pb, pl = (_band_power(x, sr, lo_hz, hi_hz) for x in (vocal, band, lead))
    each = {names.get(k, k): _band_power(v, sr, lo_hz, hi_hz) for k, v in res.stems.items() if k not in lead_keys}
    other_vox = sum(p for k, p in each.items() if k in names.values())
    act_v, act_l = _act(vocal, sr), _act(lead, sr)
    nb = min(len(pv), len(act_v), len(act_l), *(len(p) for p in each.values()))
    b2s = project.tempo.beat_to_sec
    t0 = res.start_seconds
    sections = []
    for name, a, b in SECTIONS:
        if b < start_bar or a > end_bar:
            continue
        s0 = max(b2s((a - 1) * BEATS_PER_BAR), b2s((start_bar - 1) * BEATS_PER_BAR)) - t0
        s1 = min(b2s(b * BEATS_PER_BAR), b2s(end_bar * BEATS_PER_BAR)) - t0
        i0, i1 = int(s0 * sr), min(n, int(s1 * sr))
        if i1 - i0 < int(0.5 * sr):
            continue
        loud = mixing.loudness_stats(res.audio[i0:i1], sr)
        sl = slice(i0 // hop, min(nb, i1 // hop))
        mv, ml = act_v[sl], act_l[sl]
        margin, rival = None, None
        if ml.sum() >= 200:
            lead_p = pl[sl][ml].sum()
            rival, rp = max(((k, p[sl][ml].sum()) for k, p in each.items()), key=lambda kv: kv[1])
            margin = round(float(10 * np.log10(max(lead_p, 1e-30) / max(rp, 1e-30))), 1)
        sections.append({
            "section": name, "bars": f"{a}-{b}",
            "lufs": mixing._r(loud["integrated"], 1),
            "short_term_max_lufs": mixing._r(loud["short_term_max"], 1),
            "vocal_to_band_db": _ratio_db(pv[sl], pb[sl], mv),
            "lead_to_band_db": _ratio_db(pl[sl], pb[sl], ml),
            "lead_to_other_vocals_db": (_ratio_db(pl[sl], other_vox[sl], ml, overlap=False)
                                        if np.ndim(other_vox) else None),
            "lead_margin_db": margin, "next_loudest": rival,
            "lead_sings_pct": round(100.0 * float(ml.mean()), 0),
        })
    # low end (whole render): 40-120 Hz vs 120-500 Hz energy of the master
    lo = _band_power(res.audio, sr, 40, 120).sum()
    mid = _band_power(res.audio, sr, 120, 500).sum()
    centres, psd, slope = ltas(res.audio, sr)
    out = {
        "sections": sections,
        "low_40_120_vs_120_500_db": round(float(10 * np.log10(lo / max(mid, 1e-30))), 1),
        "ltas_slope_db_per_oct": round(slope, 2),
        "ltas_third_octave_db": {f"{c:.0f}": (round(float(p), 1) if np.isfinite(p) else None)
                                 for c, p in zip(centres, psd)},
        "clipped_samples": int(np.sum(np.abs(res.audio) >= 1.0)),
    }
    if with_pitch:
        pa = pitch_accuracy(project, res, max_list=5)
        out["pitch"] = [{"track": t["name"], "notes_checked": t["notes_checked"],
                         "mean_abs_cents": t["mean_abs_cents"], "max_abs_cents": t["max_abs_cents"],
                         "within_25_cents": t["within_25_cents"], "worst": t["worst"][:3],
                         "unvoiced_notes": len(t["unvoiced_notes"])} for t in pa["tracks"]]
    return out


def print_summary(rep: dict, ana: dict, wall: float) -> None:
    m = rep["mix"]
    print(f"\nrender: {rep['duration_s']:.1f} s of audio in {rep['render_time_s']:.1f} s "
          f"(mix {rep['mix_time_s']:.1f} s, {rep['realtime_factor']}x realtime; wall {wall:.1f} s)")
    print(f"master: {rep['loudness_lufs']} LUFS integrated, true peak {rep['true_peak_db']} dBTP, "
          f"short-term max {rep['short_term_max_lufs']} LUFS, LRA {rep['loudness_range_lu']} LU, "
          f"clipped samples {ana['clipped_samples']}")
    print(f"mix: vocal-to-band {rep['vocal_to_band_db']} dB (1-5 kHz), band gain {m.get('band_gain_db')} dB, "
          f"glue GR {m.get('glue_gr_max_db')} dB, limiter max {m.get('limiter_max_db')} dB / mean "
          f"{m.get('limiter_mean_db')} dB, master gain {m.get('master_gain_db')} dB, "
          f"levels from full render: {m.get('levels_from_full_render')}")
    print(f"spectrum: LTAS slope {ana['ltas_slope_db_per_oct']} dB/oct (100 Hz-10 kHz), "
          f"40-120 Hz vs 120-500 Hz {ana['low_40_120_vs_120_500_db']} dB")
    print(f"\n{'section':<14}{'bars':>8}{'LUFS':>7}{'ST max':>8}{'voc/band':>10}{'lead/band':>11}"
          f"{'lead/vox':>10}{'lead%':>7}   lead margin over the next loudest element (1-5 kHz)")
    f = lambda v: "-" if v is None else f"{v:+.1f}" if isinstance(v, float) else str(v)  # noqa: E731
    for s in ana["sections"]:
        print(f"{s['section']:<14}{s['bars']:>8}{f(s['lufs']):>7}{f(s['short_term_max_lufs']):>8}"
              f"{f(s['vocal_to_band_db']):>10}{f(s['lead_to_band_db']):>11}{f(s['lead_to_other_vocals_db']):>10}"
              f"{s['lead_sings_pct']:>7.0f}   {f(s['lead_margin_db'])} ({s['next_loudest'] or '-'})")
    if ana.get("pitch"):
        print(f"\n{'vocal track':<12}{'notes':>7}{'mean c':>8}{'max c':>8}{'<=25c':>8}")
        for t in ana["pitch"]:
            mean = "-" if t["mean_abs_cents"] is None else f"{t['mean_abs_cents']:.1f}"
            mx = "-" if t["max_abs_cents"] is None else f"{t['max_abs_cents']:.1f}"
            w = "-" if t["within_25_cents"] is None else f"{100 * t['within_25_cents']:.0f}%"
            print(f"{t['track']:<12}{t['notes_checked']:>7}{mean:>8}{mx:>8}{w:>8}")
    if rep.get("warnings"):
        print(f"\n{len(rep['warnings'])} engine warnings, e.g.: " + "; ".join(rep["warnings"][:3]))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sr", type=int, default=44100, help="sample rate (default 44100)")
    ap.add_argument("--start-bar", type=int, default=None, help="first bar of a section render (1-based)")
    ap.add_argument("--end-bar", type=int, default=None, help="last bar of a section render (inclusive)")
    ap.add_argument("--out", type=Path, default=OUT, help=f"output folder (default {OUT})")
    ap.add_argument("--no-stems", action="store_true", help="do not write the stems")
    ap.add_argument("--no-mp3", action="store_true", help="do not encode the mp3")
    ap.add_argument("--no-pitch", action="store_true", help="skip the pitch-accuracy analysis")
    args = ap.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    t_wall = time.perf_counter()

    print("building the project through the tool layer ...")
    s = build_session()
    project = s.project
    last_bar = len(vocals.CHORDS)
    partial = args.start_bar is not None or args.end_bar is not None
    a = max(1, args.start_bar or 1)
    b = min(last_bar, args.end_bar or last_bar)
    if b < a:
        ap.error("--end-bar must not be before --start-bar")

    if not partial:
        r = call_tool(s, "save_project", {"path": str(out / f"{NAME}.s2p.json")})
        print(f"  project: {r.get('path', out / (NAME + '.s2p.json'))}")
        start_beat = end_beat = None
    else:
        start_beat, end_beat = float((a - 1) * BEATS_PER_BAR), float(b * BEATS_PER_BAR)
        reused = load_levels(out, project, args.sr)
        print(f"  section render: bars {a}-{b} (beats {start_beat:g}-{end_beat:g}); "
              + ("mastered with the levels of the last full render" if reused else
                 "no matching full render yet, so this section is mastered on its own"))

    print(f"rendering at {args.sr} Hz ...")
    t_render = time.perf_counter()

    def progress(frac: float, msg: str) -> None:
        print(f"  [{100 * frac:3.0f}% {time.perf_counter() - t_render:5.1f} s] {msg}", flush=True)

    res = render_project(project, sample_rate=args.sr, start_beat=start_beat, end_beat=end_beat, progress=progress)
    rep = res.report

    if partial:
        dest = out / "preview"
        wav = dest / f"{NAME}_bars_{a}-{b}.wav"
    else:
        dest = out
        wav = out / f"{NAME}.wav"
        save_levels(out, project, args.sr, rep)
    dest.mkdir(parents=True, exist_ok=True)
    write_wav(wav, res.audio, res.sample_rate, 24)
    print(f"  wrote {wav}")
    if not args.no_stems:
        sd = dest / ("stems" if not partial else f"stems_bars_{a}-{b}")
        if sd.exists():
            shutil.rmtree(sd)
        res.save_stems(sd, bits=24)
        print(f"  wrote {len(res.stems)} stems to {sd}/")
    if not args.no_mp3 and not partial:
        if shutil.which("ffmpeg"):
            mp3 = wav.with_suffix(".mp3")
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-codec:a", "libmp3lame",
                            "-b:a", "320k", "-metadata", "title=SELECT ALL INVADERS",
                            "-metadata", "artist=Synth2 Vocal Studio", str(mp3)], check=True)
            print(f"  wrote {mp3}")
        else:
            print("  ffmpeg not found: no mp3")

    print("analysing ...")
    ana = analyse(project, res, a, b, with_pitch=not args.no_pitch)
    wall = time.perf_counter() - t_wall
    rep_out = {k: v for k, v in rep.items() if k != "tracks"}
    rep_out["tracks"] = [{k: v for k, v in t.items() if k != "phonemes"} for t in rep.get("tracks", [])]
    (wav.with_suffix(".report.json")).write_text(json.dumps({"report": rep_out, "analysis": ana,
                                                             "wall_time_s": round(wall, 1)},
                                                            indent=1, ensure_ascii=False, default=str))
    print_summary(rep, ana, wall)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
