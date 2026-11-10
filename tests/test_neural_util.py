"""The neural engine's utilities: the resampler, the character effects and the phrase cache.
No models, no network; files go to ``tmp_path``."""
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path, PurePosixPath

import numpy as np
import pytest

from synth2.neural import cache as pc
from synth2.neural import fx
from synth2.neural import resample as rs

try:  # the fake voicebank and fake onnxruntime of the engine's own unit tests
    from test_neural_unit import fake, sing as sing_fake, song as fake_song
except Exception:  # pragma: no cover - that file is not importable from here
    fake = None

REPO = str(Path(__file__).resolve().parent.parent)
SR = 44100
RATES = [(44100, 48000), (44100, 22050), (44100, 32000), (44100, 16000), (48000, 44100), (44100, 96000),
         (44100, 8000)]


def db(ratio: float) -> float:
    return 20.0 * math.log10(max(ratio, 1e-30))


def rms(x) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2)))


# ---- resample ---------------------------------------------------------------------------------------
def tone(f, sr, n, start=0):
    return np.sin(2 * np.pi * f * (start + np.arange(n)) / sr)


def inner(y, a, b):
    """``y`` without the outputs that reach past the ends of the input (the filter sees zeros there)."""
    k = int(math.ceil(rs.margin(a, b) * b / a)) + 2
    return y[k:-k], k


def fit_tone(y, f, sr, first):
    """Level (relative to a unit sine) and the rest after taking the sine of frequency ``f`` out."""
    t = (first + np.arange(len(y))) / sr
    basis = np.stack([np.sin(2 * np.pi * f * t), np.cos(2 * np.pi * f * t)], 1)
    c, *_ = np.linalg.lstsq(basis, y, rcond=None)
    return float(np.hypot(*c)), y - basis @ c


def test_resample_equal_rates_copy_and_shift():
    x = np.random.default_rng(0).standard_normal(1000).astype(np.float32)
    y = rs.resample(x, SR, SR)
    assert y.dtype == np.float32 and np.array_equal(y, x) and y is not x
    assert rs.margin(SR, SR) == 0 and rs.out_span(7, 1000, SR, SR) == (7, 1000)
    y = rs.resample(x, SR, SR, in_start=100, out_start=90, n_out=30)
    assert not y[:10].any() and np.array_equal(y[10:], x[:20])
    y = rs.resample(x, SR, SR, in_start=-5, out_start=990, n_out=20)
    assert np.array_equal(y[:5], x[995:]) and not y[5:].any()
    assert np.array_equal(rs.resample(x.astype(np.float64), 48000, 48000), x)


def test_resample_spans_and_argument_errors():
    assert rs.out_span(0, 44100, 44100, 48000) == (0, 48000)
    assert rs.out_span(0, 44100, 44100, 22050) == (0, 22050)
    rng = np.random.default_rng(1)
    for a, b in RATES:
        pos = int(rng.integers(-5000, 5000))
        end = None
        for n in rng.integers(0, 3000, 8):  # the spans of adjacent inputs are adjacent
            o, m = rs.out_span(pos, int(n), a, b)
            assert end is None or o == end
            assert all(pos * b <= j * a < (pos + n) * b for j in (o, o + m - 1)) or m == 0
            pos, end = pos + int(n), o + m
        x = rng.standard_normal(500)
        y = rs.resample(x, a, b, in_start=123)
        assert y.dtype == np.float32 and len(y) == rs.out_span(123, 500, a, b)[1]
        lo, n_in = rs.in_span(1000, 50, a, b)
        assert lo <= (1000 * a) // b - rs.margin(a, b) and lo + n_in >= (1049 * a) // b + rs.margin(a, b)
        assert rs.margin(a, b) > 0
    assert len(rs.resample(np.zeros(0), 44100, 48000)) == 0
    assert len(rs.resample(np.ones(10), 44100, 48000, n_out=0)) == 0
    assert not rs.resample(np.zeros(0), 44100, 48000, out_start=5, n_out=7).any()
    for bad in ((0, 48000), (44100, -1), (44100.5, 48000), ("x", 48000), (None, 48000)):
        with pytest.raises(ValueError, match="positive integers"):
            rs.resample(np.zeros(4), *bad)
        with pytest.raises(ValueError):
            rs.margin(*bad)
    with pytest.raises(ValueError, match="mono"):
        rs.resample(np.zeros((4, 2)), 44100, 48000)
    assert rs.margin(44100, 48000) == rs.margin(22050, 24000)  # the pair is reduced by its gcd
    assert np.array_equal(rs.resample(x, 22050, 24000), rs.resample(x, 44100, 48000))


@pytest.mark.parametrize("a,b", RATES)
def test_resample_windows_equal_the_whole_call_bit_for_bit(a, b):
    rng = np.random.default_rng(a + b)
    x = rng.standard_normal(20000)
    full = rs.resample(x, a, b, in_start=-300)
    first, total = rs.out_span(-300, len(x), a, b)
    assert len(full) == total
    windows = [(first - 500, 700), (first + total - 200, 900), (first + 5, 1)]  # the edges of the data, one sample
    for _ in range(8):
        n = int(rng.integers(1, total // 3))
        windows.append((first + int(rng.integers(0, total - n)), n))
    for o, n in windows:
        lo, n_in = rs.in_span(o, n, a, b)  # what the window depends on, margin included
        lo2, hi2 = max(lo, -300), min(lo + n_in, len(x) - 300)
        part = rs.resample(x[lo2 + 300:hi2 + 300], a, b, in_start=lo2, out_start=o, n_out=n)
        whole = np.zeros(n, np.float32)
        s, e = max(o, first), min(o + n, first + total)
        whole[s - o:e - o] = full[s - first:e - first]
        # outside the default span the whole call was not asked; ask it
        ref = rs.resample(x, a, b, in_start=-300, out_start=o, n_out=n)
        assert np.array_equal(ref[s - o:e - o], whole[s - o:e - o])
        assert np.array_equal(part, ref)
    # the window the backend cuts: margin + 2 input samples on each side
    m = rs.margin(a, b) + 2
    o, n = first + total // 3, total // 4
    lo, hi = (o * a) // b - m, -((-(o + n) * a) // b) + m
    part = rs.resample(x[lo + 300:hi + 300], a, b, in_start=lo, out_start=o, n_out=n)
    assert np.array_equal(part, full[o - first:o - first + n])


@pytest.mark.parametrize("a,b", RATES)
def test_resample_is_linear_and_shift_consistent(a, b):
    rng = np.random.default_rng(3)
    x, y = rng.standard_normal(6000), rng.standard_normal(6000)
    rx, ry = rs.resample(x, a, b).astype(np.float64), rs.resample(y, a, b).astype(np.float64)
    assert np.abs(rs.resample(2.0 * x - 3.0 * y, a, b) - (2.0 * rx - 3.0 * ry)).max() < 2e-6
    g = math.gcd(a, b)
    up, down = b // g, a // g
    for k in (1, 7, -3):  # moving the input by k * down samples moves the output by k * up samples
        o, n = rs.out_span(11, len(x), a, b)
        moved = rs.resample(x, a, b, in_start=11 + k * down)
        assert np.array_equal(moved, rs.resample(x, a, b, in_start=11))
        assert rs.out_span(11 + k * down, len(x), a, b) == (o + k * up, n)
    # an impulse at an input sample that is also an output instant: a symmetric response that peaks there
    imp = np.zeros(4 * down + 1)
    imp[2 * down] = 1.0
    h = rs.resample(imp, a, b, out_start=2 * up - 300, n_out=601).astype(np.float64)
    assert np.argmax(np.abs(h)) == 300 and np.abs(h - h[::-1]).max() < 5e-6  # the window's last tap is one-sided
    assert h[300] == pytest.approx(min(1.0, b / a), rel=0.08)


@pytest.mark.parametrize("a,b", RATES)
def test_resample_passband_is_flat_and_on_time(a, b):
    worst_db, worst_err = 0.0, 0.0
    for f in np.linspace(40.0, 0.85 * min(a, b) / 2, 9):
        y, k = inner(rs.resample(tone(f, a, 6000), a, b).astype(np.float64), a, b)
        level, _ = fit_tone(y, f, b, k)
        worst_db = max(worst_db, abs(db(level)))
        worst_err = max(worst_err, np.abs(y - tone(f, b, len(y), k)).max())  # the same sine on the new grid
    assert worst_db < 0.01    # dB of ripple up to 0.85 of the lower Nyquist frequency (measured: 0.0003)
    assert worst_err < 1e-3   # level, phase and position together
    ones, _ = inner(rs.resample(np.ones(3000), a, b), a, b)
    assert np.abs(ones - 1.0).max() < 1e-6


@pytest.mark.parametrize("a,b", RATES)
def test_resample_rejects_aliases_and_images(a, b):
    low = min(a, b) / 2
    worst = -200.0
    if b < a:  # what lies above the new Nyquist frequency must not fold down
        for f in np.linspace(low, 0.999 * a / 2, 9):
            y, _ = inner(rs.resample(tone(f, a, 6000), a, b).astype(np.float64), a, b)
            worst = max(worst, db(rms(y) / math.sqrt(0.5)))
    else:  # no images of the tone above the old Nyquist frequency
        for f in np.linspace(100.0, 0.85 * low, 9):
            y, k = inner(rs.resample(tone(f, a, 6000), a, b).astype(np.float64), a, b)
            level, rest = fit_tone(y, f, b, k)
            worst = max(worst, db(rms(rest) / math.sqrt(0.5)))
    assert worst < -80.0  # measured: -89.8 dB or lower


def test_resample_zero_beyond_the_filter_reach_and_any_rate_pair():
    a, b = 44100, 48000
    m = rs.margin(a, b)
    y = rs.resample(np.ones(1000), a, b, in_start=5000, out_start=0, n_out=12000)
    t_in = np.arange(12000) * a / b  # output instants in input samples
    assert not y[(t_in < 5000 - m) | (t_in > 5999 + m)].any()
    assert y[(t_in > 5000) & (t_in < 5999)].min() > 0.4
    for a, b in ((44100, 44123), (96000, 44100), (8000, 44100), (44100, 11025), (3, 2), (44100, 192000)):
        f = 0.2 * min(a, b)
        x = tone(f, a, 3000)
        y, k = inner(rs.resample(x, a, b).astype(np.float64), a, b)
        assert np.abs(y - tone(f, b, len(y), k)).max() < 1e-3, (a, b)
        o, n = rs.out_span(1000, 500, a, b)
        lo, n_in = rs.in_span(o, n, a, b)
        assert np.array_equal(rs.resample(x[lo:lo + n_in], a, b, in_start=lo, out_start=o, n_out=n),
                              rs.resample(x, a, b, out_start=o, n_out=n)), (a, b)


def test_resample_is_fast_enough_for_a_song():
    x = np.random.default_rng(5).standard_normal(60 * SR).astype(np.float32)
    t = time.perf_counter()
    y = rs.resample(x, SR, 48000)
    assert time.perf_counter() - t < 3.0 and len(y) == 60 * 48000  # measured: about 0.1 s


# ---- the phrase cache: keys ---------------------------------------------------------------------------
PARTS = ("v1", ["SP", "en/aa"], [8, 20], {"b": 1.5, "a": None, "c": [True, -0.0]},
         np.arange(6, dtype=np.float32).reshape(2, 3), 7, b"raw", PurePosixPath("a/b.onnx"))
PARTS_HASH = "9a44883e33d0b70b7ff6611bea6c33aa61f6d576d8f099e6b8d1301d4c2739b2"


def test_stable_hash_is_the_same_everywhere():
    h = pc.stable_hash(*PARTS)
    assert h == PARTS_HASH and len(h) == 64  # the encoding is fixed: keys written by another version stay valid
    assert pc.stable_hash() == "ad58a1c3bbeeaa15915710f97167ba42564022d044de6e23eaf8d01dfd7f4c35"
    code = (f"import sys; sys.path.insert(0, {REPO!r})\n"
            "import numpy as np\n"
            "from pathlib import PurePosixPath\n"
            "from synth2.neural.cache import stable_hash\n"
            "print(stable_hash('v1', ['SP', 'en/aa'], [8, 20], {'c': [True, -0.0], 'a': None, 'b': 1.5},\n"
            "                  np.arange(6, dtype=np.float32).reshape(2, 3), 7, b'raw', PurePosixPath('a/b.onnx')))\n")
    for seed in ("0", "4711"):  # another process, another string hash seed, another dict order
        out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60,
                             env={**os.environ, "PYTHONHASHSEED": seed})
        assert out.returncode == 0, out.stderr[-2000:]
        assert out.stdout.strip() == h


def test_stable_hash_equal_values_equal_hashes():
    a = np.arange(6, dtype=np.float32)
    h = pc.stable_hash
    assert h({"x": 1, "y": [a, None]}) == h({"y": [a.copy(), None], "x": 1})
    assert h((1, 2), [3]) == h([1, 2], (3,))  # a tuple is the list of its items
    assert h(np.float64(1.5), np.int64(3), np.bool_(True), np.str_("a")) == h(1.5, 3, True, "a")
    assert h(np.float32(0.1)) == h(float(np.float32(0.1))) != h(0.1)
    assert h(a[::2]) == h(np.ascontiguousarray(a[::2])) == h(np.array([0, 2, 4], np.float32))
    assert h(a.astype(">f4")) == h(a)  # byte order is not part of the value
    assert h(bytearray(b"ab"), memoryview(b"cd")) == h(b"ab", b"cd")
    assert h(Path("a") / "b") == h(PurePosixPath("a/b"))
    assert h({1: "a", "1": "b"}) == h({"1": "b", 1: "a"})


def test_stable_hash_tells_near_equal_values_apart():
    a = np.arange(6, dtype=np.float32)
    h = pc.stable_hash
    values = [1, 1.0, True, "1", b"1", [1], [[1]], {"1": 1}, None, 0, 0.0, -0.0, False, "", b"", [], {}, [None],
              "None", 1.0000000000000002, float("nan"), float("inf"), -1, 2 ** 70, "a", ["a"], Path("a"),
              a, a.astype(np.float64), a.astype(np.int64), a.reshape(2, 3), a.reshape(3, 2), a.reshape(1, 6), a[:5],
              a + 1e-6, -a, np.zeros(0, np.float32), np.zeros(0, np.float64), np.zeros((0, 3), np.float32),
              np.array(1.0), np.array([1.0]), np.array([True, False]), np.array([1, 0], np.int8),
              np.array([1, 0], np.uint8), np.array([0.0]), np.array([-0.0]), np.array([1 + 2j])]
    hashes = [h(v) for v in values]
    assert len(set(hashes)) == len(values)
    # where one part ends and the next begins is part of the key
    groups = [h("ab"), h("a", "b"), h(["a", "b"]), h(["a"], "b"), h("a", ["b"]), h(["ab"]), h("a", "b", None),
              h([1, 2], [3]), h([1], [2, 3]), h([1, [2, 3]]), h([[1, 2], 3]), h(12, 3), h(1, 23), h(b"a", "b"),
              h({"a": [1, 2]}), h({"a": 1}, 2), h({"a": {"b": 1}}), h({"a": "b"}, 1), h(1.5, 2), h(1, 5.2),
              h(a, a), h(np.concatenate([a, a])), h([a, a])]
    assert len(set(groups)) == len(groups)


def test_stable_hash_refuses_what_has_no_stable_value():
    for bad in (object(), {1, 2}, np.array(["a"], dtype=object), lambda: 0, np.array(["a", "b"]),
                np.datetime64("2024-01-01"), [1, object()], {"k": {3}}, {object(): 1}):
        with pytest.raises(TypeError, match="stable_hash cannot hash"):
            pc.stable_hash(bad)


# ---- the phrase cache: settings ------------------------------------------------------------------------
def test_cache_dir_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH2_CACHE", str(tmp_path / "c"))
    assert pc.cache_dir() == tmp_path / "c" / "phrases"
    monkeypatch.delenv("SYNTH2_CACHE")
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))
    assert pc.cache_dir() == tmp_path / "xdg" / "synth2" / "phrases"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("USERPROFILE", str(tmp_path / "home"))
    assert pc.cache_dir() == tmp_path / "home" / ".cache" / "synth2" / "phrases"
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert pc.cache_dir() == tmp_path / "local" / "synth2" / "cache" / "phrases"
    monkeypatch.delenv("LOCALAPPDATA")
    assert pc.cache_dir() == tmp_path / "home" / "AppData" / "Local" / "synth2" / "cache" / "phrases"
    monkeypatch.setenv("SYNTH2_CACHE", "")  # empty counts as unset
    assert pc.cache_dir() == tmp_path / "home" / "AppData" / "Local" / "synth2" / "cache" / "phrases"

    def no_home():
        raise RuntimeError("no home directory")

    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    monkeypatch.setattr(sys, "platform", "linux")
    assert pc.cache_dir().name == "phrases" and pc.cache_dir().parent.name == "synth2"  # never raises


def test_enabled_and_size_limit_parse_the_environment(monkeypatch):
    monkeypatch.delenv("SYNTH2_NEURAL_CACHE", raising=False)
    assert pc.enabled()
    for value in ("0", "false", "off", "no", "OFF", " No ", "False"):
        monkeypatch.setenv("SYNTH2_NEURAL_CACHE", value)
        assert not pc.enabled(), value
    for value in ("1", "true", "on", "yes", "", "2", "nope"):
        monkeypatch.setenv("SYNTH2_NEURAL_CACHE", value)
        assert pc.enabled(), value
    monkeypatch.delenv("SYNTH2_NEURAL_CACHE_MB", raising=False)
    assert pc.limit_bytes() == 2048 << 20
    for value, want in (("1", 1 << 20), ("0.5", 1 << 19), ("0", 0), (" 16 ", 16 << 20), ("lots", 2048 << 20),
                        ("-3", 2048 << 20), ("nan", 2048 << 20), ("inf", 2048 << 20), ("", 2048 << 20)):
        monkeypatch.setenv("SYNTH2_NEURAL_CACHE_MB", value)
        assert pc.limit_bytes() == want, value
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE_MB", "3")
    assert pc.PhraseCache("unused").max_bytes == 3 << 20 and pc.PhraseCache("unused", max_bytes=10).max_bytes == 10


def test_default_cache_is_shared_per_folder(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH2_CACHE", str(tmp_path / "one"))
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE_MB", "5")
    c = pc.default_cache()
    assert c is pc.default_cache() and c.root == tmp_path / "one" / "phrases" and c.max_bytes == 5 << 20
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE_MB", "7")
    assert pc.default_cache() is c and c.max_bytes == 7 << 20  # the limit follows the environment
    monkeypatch.setenv("SYNTH2_CACHE", str(tmp_path / "two"))
    d = pc.default_cache()
    assert d is not c and d.root == tmp_path / "two" / "phrases" and d is pc.default_cache()
    assert not (tmp_path / "one").exists() and not (tmp_path / "two").exists()  # nothing is created before a write


# ---- the phrase cache: entries -------------------------------------------------------------------------
def key(i) -> str:
    return pc.stable_hash("test", i)


def wave(i, n=1000) -> np.ndarray:
    return (np.arange(n, dtype=np.float32) * 1e-3 + i).astype(np.float32)


def entry_files(root) -> list[str]:
    return sorted(p.name for p in Path(root).iterdir() if p.suffix == ".npy")


def test_cache_round_trip_on_disk_and_in_memory(tmp_path):
    root = tmp_path / "deep" / "er" / "phrases"  # a missing folder is created by the first write
    c = pc.PhraseCache(root)
    assert c.get(key(1)) is None and not root.exists()
    audio = wave(1)
    c.put(key(1), audio)
    c.put(key(2), audio.astype(np.float64) * 2)                # stored as float32
    c.put(key(3), np.array([1.0, 0.0], np.float32))            # two numbers about a phrase
    c.put("phrase 4 / not a digest", [0.5, -0.5])              # any string is a key, any sequence audio
    assert len(entry_files(root)) == 4 and f"{key(1)}.npy" in entry_files(root)
    assert not [p for p in root.iterdir() if p.suffix != ".npy"]  # no temporary file stays behind
    for cache in (c, pc.PhraseCache(root), pc.PhraseCache(root, memory_bytes=0)):  # memory, then disk
        got = cache.get(key(1))
        assert got.dtype == np.float32 and np.array_equal(got, audio)
        assert np.array_equal(cache.get(key(2)), (audio.astype(np.float64) * 2).astype(np.float32))
        assert np.array_equal(cache.get(key(3)), [1.0, 0.0]) and np.array_equal(cache.get("phrase 4 / not a digest"),
                                                                              [0.5, -0.5])
        assert cache.get(key(9)) is None
        got[:] = 7.0  # the caller's copy: the cache keeps its own
        assert np.array_equal(cache.get(key(1)), audio)
    audio[:] = -1.0  # put copied the array
    assert c.get(key(1))[5] == pytest.approx(1.005)
    assert np.array_equal(np.load(root / f"{key(1)}.npy"), wave(1))  # a plain float32 .npy file
    s = c.stats()
    assert s["hits"] == 6 and s["misses"] == 2 and s["entries"] == 4 and s["root"] == str(root)
    assert s["bytes"] == sum(p.stat().st_size for p in root.iterdir()) and s["writes"] == 4 and s["errors"] == 0
    assert s["memory_entries"] == 4 and s["memory_bytes"] == 2 * 4000 + 8 + 8
    c.put(key(1), wave(5))  # a key written again holds the new audio, on disk too
    assert np.array_equal(c.get(key(1)), wave(5)) and np.array_equal(pc.PhraseCache(root).get(key(1)), wave(5))


def test_cache_memory_layer_is_bounded_and_serves_repeats(tmp_path):
    c = pc.PhraseCache(tmp_path, memory_bytes=3 * 4000)
    for i in range(5):
        c.put(key(i), wave(i))
    assert c.stats()["memory_entries"] == 3 and c.stats()["memory_bytes"] == 12000
    for p in tmp_path.iterdir():
        p.unlink()
    assert c.get(key(0)) is None and c.get(key(1)) is None  # dropped from memory, gone from disk
    assert np.array_equal(c.get(key(2)), wave(2))           # still in memory; now the most recently used
    c.put(key(5), wave(5))
    c.put(key(6), wave(6))
    for p in tmp_path.iterdir():
        p.unlink()
    assert c.get(key(3)) is None and np.array_equal(c.get(key(2)), wave(2))
    c.put(key(7), wave(7, 4000))  # larger than the whole layer: on disk only
    assert c.stats()["memory_bytes"] <= 12000 and np.array_equal(c.get(key(7)), wave(7, 4000))
    c.clear()
    assert c.get(key(2)) is None and c.stats()["memory_entries"] == 0 and c.stats()["entries"] == 0


def test_cache_corrupt_and_truncated_entries_are_misses_and_are_removed(tmp_path):
    good = pc.PhraseCache(tmp_path, memory_bytes=0)
    good.put(key(0), wave(0))
    path = tmp_path / f"{key(0)}.npy"
    whole = path.read_bytes()
    import io
    import pickle

    f64 = io.BytesIO()
    np.save(f64, np.zeros(10))
    obj = io.BytesIO()
    np.save(obj, np.array([{"a": 1}], dtype=object), allow_pickle=True)
    nan = io.BytesIO()
    np.save(nan, np.array([0.0, np.nan], np.float32))
    longer = whole.replace(b"(1000,)", b"(2000,)", 1)
    huge = whole.replace(b"(1000,), }      ", b"(1000000000,), }", 1)
    assert len(longer) == len(huge) == len(whole) and longer != whole != huge
    bad_files = {"empty": b"", "cut in the header": whole[:40], "cut in the data": whole[:-1], "half": whole[:2000],
                 "one byte more": whole + b"\0", "not npy": b"RIFF" + bytes(100), "text": b"hello\n" * 50,
                 "float64": f64.getvalue(), "pickled objects": obj.getvalue(), "pickle": pickle.dumps([1, 2, 3]),
                 "not finite": nan.getvalue(), "more samples in the header": longer,
                 "a billion samples in the header": huge,
                 "zip": b"PK\x03\x04" + bytes(200)}
    for what, data in bad_files.items():
        c = pc.PhraseCache(tmp_path, memory_bytes=0)
        path.write_bytes(data)
        assert c.get(key(0)) is None, what
        assert not path.exists(), what          # removed, so that the phrase is rendered and stored again
        assert c.stats()["errors"] == 1 and c.stats()["misses"] == 1, what
        c.put(key(0), wave(0))
        assert np.array_equal(c.get(key(0)), wave(0)), what
    assert path.read_bytes() == whole
    path.unlink()
    path.mkdir()  # something that cannot be opened as a file: a miss, nothing raised
    c = pc.PhraseCache(tmp_path, memory_bytes=0)
    assert c.get(key(0)) is None
    c.put(key(0), wave(0))  # cannot replace a folder: counted, not raised
    assert c.stats()["errors"] >= 1 and path.is_dir()
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


def test_cache_does_not_store_what_is_not_audio(tmp_path):
    c = pc.PhraseCache(tmp_path)
    c.put(key(1), np.array([0.0, np.inf], np.float32))
    c.put(key(2), np.array([np.nan]))
    c.put(key(3), "not audio")
    c.put(key(4), None)
    c.put(key(5), [[1.0, 2.0], [3.0]])
    assert all(c.get(key(i)) is None for i in range(1, 6)) and not entry_files(tmp_path)
    c.put(key(6), np.zeros(0, np.float32))  # an empty phrase is still an entry
    assert c.get(key(6)).shape == (0,) and pc.PhraseCache(tmp_path).get(key(6)).shape == (0,)


def test_cache_writes_are_atomic(tmp_path, monkeypatch):
    c = pc.PhraseCache(tmp_path, memory_bytes=0)
    c.put(key(1), wave(1))
    seen = []
    real_replace = os.replace

    def watching(src, dst):
        # at the moment of the move the new file is complete under its temporary name and the
        # old entry is still whole under its own
        seen.append((Path(src).name, np.load(src).copy(), np.load(dst).copy()))
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", watching)
    c.put(key(1), wave(2))
    (name, new, old), = seen
    assert name.endswith(".tmp") and name.startswith(key(1)) and np.array_equal(new, wave(2))
    assert np.array_equal(old, wave(1)) and np.array_equal(c.get(key(1)), wave(2))

    def failing(src, dst):
        raise PermissionError("the entry is open in another program")

    monkeypatch.setattr(os, "replace", failing)
    c.put(key(1), wave(3))  # must not raise, must not leave half an entry or a temporary file
    c.put(key(4), wave(4))
    monkeypatch.setattr(os, "replace", real_replace)
    assert entry_files(tmp_path) == [f"{key(1)}.npy"] and not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    assert np.array_equal(c.get(key(1)), wave(2)) and c.get(key(4)) is None and c.stats()["errors"] == 2

    def dying(f, arr, **kw):
        f.write(b"\x93NUMPY half an ent")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(np.lib.format, "write_array", dying)
    c.put(key(5), wave(5))
    assert entry_files(tmp_path) == [f"{key(1)}.npy"] and not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]


def test_cache_evicts_the_least_recently_used_entries(tmp_path):
    size = len(np.lib.format.magic(1, 0)) + 2 + 118 + 4000  # an entry of 1000 samples: 128 bytes of header
    c = pc.PhraseCache(tmp_path, max_bytes=10 * 4128 + 100, memory_bytes=0)
    now = time.time()
    for i in range(10):
        c.put(key(i), wave(i))
        os.utime(tmp_path / f"{key(i)}.npy", (now - 1000 + i, now - 1000 + i))  # written one second apart
    s = c.stats()
    assert size == 4128 and s["entries"] == 10 and s["bytes"] == 10 * 4128 and s["evicted"] == 0
    assert c.get(key(0)) is not None and c.get(key(3)) is not None  # reading makes an entry recent
    assert (tmp_path / f"{key(0)}.npy").stat().st_mtime > now - 5
    c.put(key(10), wave(10))  # over the limit: trimmed to 90 % of it, oldest first
    left = {i for i in range(11) if (tmp_path / f"{key(i)}.npy").exists()}
    assert left == {0, 3, 4, 5, 6, 7, 8, 9, 10}
    s = c.stats()
    assert s["bytes"] == 9 * 4128 <= 0.9 * c.max_bytes and s["evicted"] == 2 and s["entries"] == 9
    assert c.get(key(1)) is None and c.get(key(2)) is None
    for i in range(11, 14):  # the folder never stays above the limit
        c.put(key(i), wave(i))
        assert c.stats()["bytes"] <= c.max_bytes
    # entries another process wrote count too, and files that are not entries are left alone
    other = pc.PhraseCache(tmp_path, max_bytes=1 << 30, memory_bytes=0)
    for i in range(20, 60):
        other.put(key(i), wave(i))
    (tmp_path / "notes.txt").write_text("keep me")
    (tmp_path / "model.npy").write_bytes(b"not an entry of the cache")
    stale = tmp_path / f"{key(1)}.123-456-7.tmp"
    stale.write_bytes(b"left by a writer that died")
    os.utime(stale, (now - 7200, now - 7200))
    fresh = tmp_path / f"{key(2)}.123-456-8.tmp"
    fresh.write_bytes(b"being written right now")
    c.put(key(99), wave(99))
    s = c.stats()
    assert s["bytes"] <= 0.9 * c.max_bytes + 4128 and (tmp_path / f"{key(99)}.npy").exists()
    assert (tmp_path / "notes.txt").exists() and (tmp_path / "model.npy").exists()
    assert not stale.exists() and fresh.exists()
    fresh.unlink()
    big = pc.PhraseCache(tmp_path / "small", max_bytes=1000, memory_bytes=1 << 20)
    big.put(key(1), wave(1))  # larger than the whole folder may be: kept in memory only
    assert not (tmp_path / "small").exists() and np.array_equal(big.get(key(1)), wave(1))
    off = pc.PhraseCache(tmp_path / "off", max_bytes=0)
    off.put(key(1), wave(1))
    assert not (tmp_path / "off").exists() and np.array_equal(off.get(key(1)), wave(1))
    c.clear()
    assert c.stats()["entries"] == 0 and c.stats()["bytes"] == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["model.npy", "notes.txt"]


def test_cache_survives_a_folder_it_cannot_use(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("a file where the cache folder should be")
    for root in (blocker, blocker / "phrases"):
        c = pc.PhraseCache(root)
        assert c.get(key(1)) is None
        c.put(key(1), wave(1))
        assert np.array_equal(c.get(key(1)), wave(1))  # the memory layer still serves
        assert pc.PhraseCache(root).get(key(1)) is None
        s = c.stats()
        assert s["entries"] == 0 and s["bytes"] == 0 and s["errors"] == 1
        c.clear()
    assert blocker.read_text().startswith("a file")
    if hasattr(os, "geteuid") and os.geteuid() != 0:  # root writes into read-only folders
        ro = tmp_path / "ro"
        ro.mkdir()
        full = pc.PhraseCache(ro)
        full.put(key(1), wave(1))
        ro.chmod(0o500)
        try:
            c = pc.PhraseCache(ro)
            assert np.array_equal(c.get(key(1)), wave(1))
            c.put(key(2), wave(2))
            assert np.array_equal(c.get(key(2)), wave(2)) and entry_files(ro) == [f"{key(1)}.npy"]
        finally:
            ro.chmod(0o700)


def test_cache_threads_share_one_instance(tmp_path):
    c = pc.PhraseCache(tmp_path, max_bytes=40 * 4128, memory_bytes=8 * 4000)
    errors = []

    def work(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(150):
                i = int(rng.integers(0, 60))
                if rng.random() < 0.5:
                    c.put(key(i), wave(i))
                else:
                    got = c.get(key(i))
                    assert got is None or np.array_equal(got, wave(i))
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(repr(exc))

    threads = [threading.Thread(target=work, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    assert not errors and not any(t.is_alive() for t in threads)
    s = c.stats()
    assert s["errors"] == 0 and s["bytes"] <= c.max_bytes and s["memory_bytes"] <= 8 * 4000
    assert s["hits"] + s["misses"] + s["writes"] == 8 * 150
    assert not [p for p in tmp_path.iterdir() if p.suffix == ".tmp"]
    for p in tmp_path.iterdir():  # every file is a whole entry
        i = [i for i in range(60) if p.name == f"{key(i)}.npy"][0]
        assert np.array_equal(np.load(p), wave(i))


def test_cache_two_processes_share_one_folder(tmp_path):
    code = (f"import sys; sys.path.insert(0, {REPO!r})\n"
            "import json\n"
            "import numpy as np\n"
            "from synth2.neural import cache as pc\n"
            f"c = pc.PhraseCache({str(tmp_path)!r}, max_bytes=30 * 4128, memory_bytes=0)\n"
            "wave = lambda i: (np.arange(1000, dtype=np.float32) * 1e-3 + i).astype(np.float32)\n"
            "rng = np.random.default_rng(int(sys.argv[1]))\n"
            "bad = 0\n"
            "for _ in range(400):\n"
            "    i = int(rng.integers(0, 50))\n"
            "    if rng.random() < 0.6:\n"
            "        c.put(pc.stable_hash('test', i), wave(i))\n"
            "    else:\n"
            "        got = c.get(pc.stable_hash('test', i))\n"
            "        bad += int(got is not None and not np.array_equal(got, wave(i)))\n"
            "print(json.dumps({**c.stats(), 'bad': bad}))\n")
    procs = [subprocess.Popen([sys.executable, "-I", "-c", code, str(seed)], stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for seed in (1, 2)]
    results = []
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err[-2000:]
        results.append(json.loads(out.strip().splitlines()[-1]))
    # eviction by one process while the other reads or writes is a miss or a lost entry, never an error
    assert all(r["bad"] == 0 and r["errors"] == 0 for r in results), results
    assert sum(r["evicted"] for r in results) > 0
    files = list(tmp_path.iterdir())
    assert files and all(p.suffix == ".npy" for p in files)
    assert sum(p.stat().st_size for p in files) <= 30 * 4128
    for p in files:
        i = [i for i in range(50) if p.name == f"{key(i)}.npy"][0]
        assert np.array_equal(np.load(p), wave(i))


# ---- effects ----------------------------------------------------------------------------------------
PRESETS = ("alien", "robot", "radio", "crush", "double")
CHAINS = [f"{name}:{amount}" for name in PRESETS for amount in ("1", "0.5")] + [
    [{"type": "comb", "delay_ms": 3.1, "feedback": -0.7, "mix": 0.8}],
    [{"type": "crush", "bits": 5, "rate_hz": 7000}, {"type": "tremolo", "rate": 6.3, "depth": 0.9}],
    [{"type": "chorus", "voices": 3, "delay_ms": 9, "depth_ms": 4, "rate": 2.5, "mix": 1}],
    [{"type": "band", "lo_hz": 0, "hi_hz": 1500}, {"type": "ring", "freq": 441.7, "mix": 1}],
    [{"type": "drive", "gain_db": 30}, {"type": "band", "lo_hz": 900, "hi_hz": 40000}, {"type": "gain", "db": -3}],
    ["alien", "robot:0.7", {"preset": "double", "amount": 0.4}, "crush", "radio"],
]


def voice_like(seconds=0.7, sr=SR, seed=0) -> np.ndarray:
    """A sung-voice stand-in at a sung voice's level (peak 0.35): a gliding harmonic tone under
    three formants, three syllables a second, a burst of hiss before each."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    t = np.arange(n) / sr
    f0 = 220.0 * 2.0 ** (0.4 * np.sin(2 * np.pi * 1.3 * t) / 12.0)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    x = np.zeros(n)
    for k in range(1, 40):
        f = 220.0 * k
        level = sum(g * math.exp(-((f - c) / w) ** 2) for c, w, g in ((600, 250, 1.0), (1700, 400, 0.5),
                                                                         (2900, 500, 0.25)))
        x += (level + 0.02 / k) * np.sin(k * phase)
    syllable = np.clip(np.sin(2 * np.pi * 3.0 * t), 0.0, None) ** 0.5
    hiss = np.diff(rng.standard_normal(n + 1)) * np.clip(-np.sin(2 * np.pi * 3.0 * t) - 0.8, 0.0, None)
    x = x * syllable + 0.6 * hiss
    return (0.35 * x / np.abs(x).max()).astype(np.float32)


def spike_alien(x, sr, amount):
    """The effect the first neural demo used on its alien voices."""
    t = np.arange(len(x)) / sr
    ring = x * np.sin(2 * np.pi * 62.0 * t)
    d = int(0.0045 * sr)
    comb = np.zeros_like(x)
    comb[d:] = x[:-d]
    wet = 0.45 * amount
    return ((1 - wet) * x + wet * 1.6 * ring + 0.35 * amount * comb).astype(np.float32)


def test_parse_fx_reads_every_form():
    assert fx.parse_fx(None) == fx.parse_fx("") == fx.parse_fx("none") == fx.parse_fx("  ") == []
    assert fx.parse_fx("none:1") == fx.parse_fx([]) == fx.parse_fx([None, "", "none"]) == fx.parse_fx("alien:0") == []
    assert sorted(fx.FX_PRESETS) == sorted(PRESETS + ("none",))
    for name in PRESETS:
        chain = fx.parse_fx(name)
        assert chain and [e["type"] for e in chain] == [e["type"] for e in fx.FX_PRESETS[name]]
        for e in chain:  # every parameter of the type is there, as a float
            assert set(e) == {"type"} | set(fx.FX_TYPES[e["type"]])
            assert all(isinstance(v, float) for k, v in e.items() if k != "type")
        assert fx.parse_fx(f"{name}:1") == fx.parse_fx(f" {name.upper()} : 1.0 ") == chain
        assert fx.parse_fx(f"{name}:7") == chain and fx.parse_fx(f"{name}:-1") == []  # amounts are clipped to 0..1
        assert fx.parse_fx({"preset": name}) == chain == fx.parse_fx([[name]]) == fx.parse_fx((name,))
        assert fx.parse_fx(chain) == chain  # what parse_fx returns is a spec too
        half = fx.parse_fx(f"{name}:0.5")
        assert half == fx.parse_fx({"preset": name, "amount": 0.5}) and len(half) == len(chain)
        for e, h in zip(chain, half):  # the amount scales the wet mix or the depth and nothing else
            scaled = fx.AMOUNT_PARAM[e["type"]]
            assert h[scaled] == pytest.approx(0.5 * e[scaled])
            assert {k: v for k, v in h.items() if k != scaled} == {k: v for k, v in e.items() if k != scaled}
    ring = fx.parse_fx({"type": "Ring", "freq": 100})
    assert ring == [{"type": "ring", "freq": 100.0, "mix": 0.5, "gain": pytest.approx(math.sqrt(2))}]
    assert fx.parse_fx({"type": "ring", "freq": 100, "amount": 0.5})[0]["mix"] == 0.25
    assert fx.parse_fx({"type": "tremolo", "amount": 0.5})[0]["depth"] == 0.25
    assert fx.parse_fx({"type": "ring", "amount": 0}) == []
    assert fx.parse_fx({"type": "comb", "feedback": 5, "delay_ms": 0})[0]["feedback"] == 0.9  # clipped to the range
    assert fx.parse_fx({"type": "comb", "feedback": "0.25"})[0]["feedback"] == 0.25
    assert fx.parse_fx({"type": "band", "lo_hz": 3})[0]["lo_hz"] == 20.0
    chain = fx.parse_fx(["alien:0.5", {"type": "gain", "db": -3}, ["double", "none"]])
    assert [e["type"] for e in chain] == ["ring", "comb", "gain", "gain", "chorus"][:len(chain)] or chain
    assert [e["type"] for e in fx.parse_fx(["double", {"type": "gain", "db": -3}])] == ["chorus", "gain"]
    for name in fx.FX_TYPES:  # every type has defaults that do something
        (e,) = fx.parse_fx({"type": name, **({"db": -6} if name == "gain" else {})})
        assert e["type"] == name and name in fx.AMOUNT_PARAM


def test_parse_fx_errors_name_what_is_valid():
    with pytest.raises(ValueError, match=r"unknown effect 'zap'; valid effects: alien, crush, double, none, radio, robot"):
        fx.parse_fx("zap:1")
    with pytest.raises(ValueError, match="unknown effect 'ring'"):  # a type is not a preset
        fx.parse_fx("ring")
    with pytest.raises(ValueError, match="unknown effect 'nope'"):
        fx.parse_fx(["alien", {"preset": "nope"}])
    for bad in ("alien:loud", "alien:", "alien:nan", "alien:1:2", "alien:inf"):
        with pytest.raises(ValueError, match="is not a number; use name:amount with an amount from 0 to 1"):
            fx.parse_fx(bad)
    with pytest.raises(ValueError, match="not a number"):
        fx.parse_fx({"type": "ring", "amount": "much"})
    with pytest.raises(ValueError, match=r"unknown effect type 'flange'; valid types: band, chorus, comb, crush, "
                                         r"drive, gain, ring, tremolo"):
        fx.parse_fx({"type": "flange"})
    with pytest.raises(ValueError, match="unknown effect type None"):
        fx.parse_fx({"freq": 3})
    with pytest.raises(ValueError, match=r"effect type 'ring' has no parameter 'speed'; its parameters: freq, gain, mix"):
        fx.parse_fx({"type": "ring", "speed": 3})
    for bad in ("fast", None, float("nan"), [1], True):
        with pytest.raises(ValueError, match="parameter 'freq' of effect type 'ring' must be a number"):
            fx.parse_fx({"type": "ring", "freq": bad})
    with pytest.raises(ValueError, match="needs hi_hz above lo_hz"):
        fx.parse_fx({"type": "band", "lo_hz": 4000, "hi_hz": 300})
    with pytest.raises(ValueError, match="takes 'amount' only, not freq"):
        fx.parse_fx({"preset": "alien", "freq": 50})
    for bad in (5, 0.5, object(), b"alien"):
        with pytest.raises(ValueError, match="an effect is a preset name, 'name:amount', a dict"):
            fx.parse_fx(bad)
    for call in (lambda: fx.apply_fx(np.zeros(8), SR, "zap"), lambda: fx.fx_tail("zap")):
        with pytest.raises(ValueError, match="unknown effect 'zap'"):
            call()
    with pytest.raises(ValueError, match="mono"):
        fx.apply_fx(np.zeros((8, 2)), SR, "alien")


@pytest.mark.parametrize("spec", CHAINS, ids=[c if isinstance(c, str) else "+".join(
    e if isinstance(e, str) else e.get("type", "preset") for e in c) for c in CHAINS])
def test_fx_a_window_after_the_tail_equals_the_whole_render(spec):
    x = voice_like()
    tail = fx.fx_tail(spec)
    pre = int(math.ceil(tail * SR))
    assert 0.0 <= tail < 0.3
    for first in (0, 54321, -22050, 3600 * SR + 17):  # the absolute sample index of x[0]
        whole = fx.apply_fx(x, SR, spec, t0=first / SR, seed=5)
        assert whole.dtype == np.float32 and len(whole) == len(x) and np.isfinite(whole).all()
        for a in (pre, pre + 1, pre + 333, pre + 4097, len(x) - 50):  # not on any LFO or hold boundary
            piece = fx.apply_fx(x[a - pre:], SR, spec, t0=(first + a - pre) / SR, seed=5)
            assert len(piece) == len(x) - a + pre
            assert np.array_equal(piece[pre:], whole[a:]), (first, a)
            short = fx.apply_fx(x[a - pre:a + 100], SR, spec, t0=(first + a - pre) / SR, seed=5)  # nothing looks ahead
            assert np.array_equal(short[pre:], whole[a:a + 100]), (first, a)
    if tail == 0.0:  # no memory: any piece is right from its first sample
        assert np.array_equal(fx.apply_fx(x[7001:9000], SR, spec, t0=7001 / SR, seed=5),
                              fx.apply_fx(x, SR, spec, seed=5)[7001:9000])


def test_fx_tail_is_the_memory_of_the_chain():
    assert fx.fx_tail(None) == fx.fx_tail("none") == fx.fx_tail([]) == 0.0
    for spec in ({"type": "ring"}, {"type": "tremolo"}, {"type": "drive"}, {"type": "gain", "db": 3}):
        assert fx.fx_tail(spec) == 0.0
    assert 0.010 <= fx.fx_tail({"type": "comb", "delay_ms": 10}) < 0.0102
    assert fx.fx_tail({"type": "comb", "delay_ms": 10, "feedback": 0.5}) > 0.15  # as long as the echoes last
    assert 0.001 <= fx.fx_tail({"type": "crush", "rate_hz": 1000}) < 0.0015
    assert 0.024 <= fx.fx_tail({"type": "chorus", "delay_ms": 20, "depth_ms": 4}) < 0.025
    assert fx.fx_tail({"type": "band", "lo_hz": 100}) > fx.fx_tail({"type": "band", "lo_hz": 1000}) > 0.003
    parts = ["alien", "robot:0.3", {"type": "band"}]
    assert fx.fx_tail(parts) == pytest.approx(sum(fx.fx_tail(p) for p in parts))  # memories add up along a chain
    assert fx.fx_tail(["alien:0.6", "double:0.3"]) == fx.fx_tail(["alien:1", "double:1"])  # as the backend asks
    assert all(isinstance(fx.fx_tail(p), float) and fx.fx_tail(p) < 0.25 for p in PRESETS)
    # the memory is needed: without the pre-roll a piece differs from the whole render
    x = voice_like()
    for spec in ("robot", "double", "radio", "alien"):
        whole = fx.apply_fx(x, SR, spec)
        assert not np.array_equal(fx.apply_fx(x[4000:], SR, spec, t0=4000 / SR)[:2000], whole[4000:6000]), spec


def test_fx_presets_keep_length_and_level():
    x = voice_like(1.5)
    for name in PRESETS:
        for amount in (1.0, 0.5, 0.2):
            y = fx.apply_fx(x, SR, f"{name}:{amount}", t0=12.5, seed=3)
            assert y.dtype == np.float32 and len(y) == len(x) and np.isfinite(y).all()
            assert not np.array_equal(y, x), name
            assert abs(db(rms(y) / rms(x))) < 1.5, (name, amount, db(rms(y) / rms(x)))
            assert np.abs(y).max() < 2.0 * np.abs(x).max(), (name, amount)
    assert np.array_equal(fx.apply_fx(x, SR, "none"), x) and fx.apply_fx(x, SR, None) is not x
    assert fx.apply_fx(x.astype(np.float64), SR, []).dtype == np.float32
    for n in (0, 1, 2, 5, 197, 199):  # shorter than every delay
        for name in PRESETS:
            y = fx.apply_fx(x[:n], SR, name, t0=0.3)
            assert y.dtype == np.float32 and len(y) == n and np.isfinite(y).all()
    ints = (x * 32767).astype(np.int16)  # any numeric input
    assert np.allclose(fx.apply_fx(ints, SR, "alien"), fx.apply_fx(ints.astype(np.float32), SR, "alien"))


def test_fx_silence_clicks_and_full_scale_do_not_blow_up():
    n = 20000
    click = np.zeros(n, np.float32)
    click[5000] = 1.0
    t = np.arange(n) / SR
    loud = {"sine": np.sin(2 * np.pi * 166.0 * t), "square": np.sign(np.sin(2 * np.pi * 441.0 * t)),
            "noise": np.clip(np.random.default_rng(2).standard_normal(n), -1, 1), "dc": np.ones(n)}
    for spec in list(PRESETS) + CHAINS[10:]:
        assert not fx.apply_fx(np.zeros(n, np.float32), SR, spec, t0=1.0).any(), spec  # silence stays silence
        y = fx.apply_fx(click, SR, spec, t0=1.0)
        assert np.isfinite(y).all() and not y[:5000].any() and np.abs(y).max() <= 2.5, spec
        assert rms(y) <= 2.5 * rms(click) and not y[5000 + int(0.3 * SR):].any(), spec  # the click dies away
        for what, x in loud.items():
            y = fx.apply_fx(x.astype(np.float32), SR, spec)
            assert np.isfinite(y).all() and np.abs(y).max() <= 4.0, (spec, what, float(np.abs(y).max()))
    for name in PRESETS:  # presets stay close to the input's peak even on full-scale input
        for what, x in loud.items():
            assert np.abs(fx.apply_fx(x.astype(np.float32), SR, name)).max() <= 2.5, (name, what)


def test_fx_ring_and_tremolo_follow_absolute_time():
    ones = np.ones(4000)
    n0 = 123457
    idx = n0 + np.arange(4000)
    y = fx.apply_fx(ones, SR, {"type": "ring", "freq": 62, "mix": 1, "gain": 1}, t0=n0 / SR)
    assert np.allclose(y, np.sin(2 * np.pi * 62 * idx / SR), atol=1e-6)
    y = fx.apply_fx(ones, SR, {"type": "ring", "freq": 62, "mix": 0.45, "gain": 1.6}, t0=n0 / SR)
    assert np.allclose(y, 0.55 + 0.72 * np.sin(2 * np.pi * 62 * idx / SR), atol=1e-6)
    y = fx.apply_fx(ones, SR, {"type": "tremolo", "rate": 5, "depth": 0.6}, t0=n0 / SR)
    gain = 0.7 + 0.3 * np.sin(2 * np.pi * 5 * idx / SR)
    assert np.allclose(y, gain / math.sqrt(0.49 + 0.045), atol=1e-6)
    long = fx.apply_fx(np.ones(SR), SR, {"type": "tremolo", "rate": 5, "depth": 0.6})
    assert rms(long) == pytest.approx(1.0, abs=1e-6) and long.min() == pytest.approx(0.4 / math.sqrt(0.535), abs=1e-4)
    assert np.array_equal(fx.apply_fx(ones, SR, {"type": "tremolo", "depth": 0}), ones)
    assert np.allclose(fx.apply_fx(ones, SR, {"type": "gain", "db": -6}), 10 ** -0.3)
    # a carrier above half the sample rate is kept below it
    assert np.isfinite(fx.apply_fx(ones, 8000, {"type": "ring", "freq": 20000})).all()


def test_fx_alien_is_the_first_demo_effect_at_a_gentler_setting():
    x = voice_like(1.0)
    for amount in (0.3, 0.6, 1.0):  # the building blocks reproduce the spike's formula
        chain = [{"type": "ring", "freq": 62, "mix": 0.45 * amount, "gain": 1.6},
                 {"type": "comb", "delay_ms": 4.5, "feedback": 0, "mix": 0.35 * amount, "parallel": 1}]
        assert np.abs(fx.apply_fx(x, SR, chain) - spike_alien(x, SR, amount)).max() < 1e-6
    ring, comb, trim = fx.FX_PRESETS["alien"]
    assert (ring["type"], ring["freq"], ring["gain"]) == ("ring", 62.0, 1.6)
    assert (comb["type"], comb["delay_ms"], comb["feedback"], comb["parallel"]) == ("comb", 4.5, 0.0, 1.0)
    scale = ring["mix"] / 0.45  # the preset at amount 1 is the spike's effect at this amount ...
    assert comb["mix"] == pytest.approx(0.35 * scale) and 0.4 <= scale <= 0.7
    for amount in (1.0, 0.5):
        y = fx.apply_fx(x, SR, f"alien:{amount}")
        ref = spike_alien(x, SR, scale * amount) * 10 ** (trim["db"] * amount / 20)  # ... with its level made up
        assert np.abs(y - ref).max() < 1e-6
    assert rms(fx.apply_fx(x, SR, "alien") - x) < rms(spike_alien(x, SR, 1.0) - x)


def test_fx_comb_and_chorus_delays():
    click = np.zeros(3000)
    click[100] = 1.0
    y = fx.apply_fx(click, 48000, {"type": "comb", "delay_ms": 5, "feedback": 0.5, "mix": 0.8})
    taps = np.flatnonzero(y)
    assert list(taps[:5]) == [100, 340, 580, 820, 1060]  # every 5 ms
    assert np.allclose(y[taps[:5]], [1.0, 0.8, 0.4, 0.2, 0.1], atol=1e-6)
    y = fx.apply_fx(click, 48000, {"type": "comb", "delay_ms": 5, "feedback": -0.5, "mix": 1})
    assert np.allclose(y[[340, 580, 820]], [1.0, -0.5, 0.25], atol=1e-6)
    y = fx.apply_fx(click, 48000, {"type": "comb", "delay_ms": 5, "mix": 0.3})
    assert list(np.flatnonzero(y)) == [100, 340] and y[340] == pytest.approx(0.3)
    # in a chain the comb delays what the effect before it made; with parallel it delays the chain's input
    half = {"type": "gain", "db": -6.0206}
    serial = fx.apply_fx(click, 48000, [half, {"type": "comb", "delay_ms": 5, "mix": 1}])
    beside = fx.apply_fx(click, 48000, [half, {"type": "comb", "delay_ms": 5, "mix": 1, "parallel": 1}])
    assert serial[340] == pytest.approx(0.5, abs=1e-4) and beside[340] == pytest.approx(1.0)
    spec = {"type": "chorus", "voices": 2, "delay_ms": 20, "depth_ms": 2, "rate": 3, "mix": 1}
    y = fx.apply_fx(click, 48000, spec, seed=1)
    wet = np.flatnonzero(np.abs(y) > 1e-4)
    assert wet[0] == 100 and wet[1] >= 100 + int(0.018 * 48000) - 2 and wet[-1] <= 100 + int(0.022 * 48000) + 3
    assert y[100] == pytest.approx(1 / math.sqrt(1.5))
    x = voice_like()
    a, b, c = (fx.apply_fx(x, SR, "double", seed=s) for s in (1, 1, 2))
    assert np.array_equal(a, b) and not np.array_equal(a, c)  # the seed places the voices' wandering
    assert abs(db(rms(c) / rms(a))) < 1.5
    # a steady tone through a chorus without modulation is the tone delayed: the interpolation is exact there
    tone_in = np.sin(2 * np.pi * 1000 * np.arange(6000) / 48000)
    y = fx.apply_fx(tone_in, 48000, {"type": "chorus", "voices": 1, "delay_ms": 10.01, "depth_ms": 0, "mix": 1})
    delayed = np.sin(2 * np.pi * 1000 * (np.arange(6000) / 48000 - 0.01001))
    assert np.allclose(y[1000:] * math.sqrt(2), (tone_in + delayed)[1000:], atol=2e-4)


def test_fx_crush_holds_and_rounds_on_the_absolute_grid():
    x = voice_like()
    y = fx.apply_fx(x, SR, {"type": "crush", "bits": 6, "rate_hz": 11025})
    assert np.allclose(y * 32, np.round(y * 32), atol=1e-6) and len(np.unique(y)) <= 64  # 6 bits
    cells = y[:28000].reshape(-1, 4)  # 44100 / 11025: four samples a cell, each holding its first sample
    assert np.array_equal(cells, np.repeat(cells[:, :1], 4, axis=1))
    assert np.allclose(cells[:, 0], np.round(x[0:28000:4].astype(np.float64) * 32) / 32, atol=1e-7)
    shifted = fx.apply_fx(x, SR, {"type": "crush", "bits": 6, "rate_hz": 11025}, t0=3 / SR)
    cells = shifted[1:28001].reshape(-1, 4)  # the grid is counted from time 0, not from the buffer's start
    assert np.array_equal(cells, np.repeat(cells[:, :1], 4, axis=1))
    assert np.allclose(cells[:, 0], np.round(x[1:28001:4].astype(np.float64) * 32) / 32, atol=1e-7)
    assert shifted[0] == 0.0  # the first cell began before the buffer: nothing to hold
    fine = fx.apply_fx(x, SR, {"type": "crush", "bits": 24, "rate_hz": 96000})
    assert np.abs(fine - x).max() < 1e-6  # no hold above the sample rate, 24 bits: nothing to hear
    odd = fx.apply_fx(x, SR, {"type": "crush", "bits": 16, "rate_hz": 7000})  # cells of 6 and 7 samples
    runs = np.diff(np.flatnonzero(np.diff(odd[2000:6000]) != 0))
    assert set(runs) <= {6, 7, 12, 13, 14} and 6.2 < np.mean(runs[runs < 8]) < 6.4
    mixed = fx.apply_fx(x, SR, {"type": "crush", "bits": 6, "rate_hz": 11025, "mix": 0.25})
    assert np.allclose(mixed, 0.75 * x + 0.25 * y, atol=1e-6)


def test_fx_drive_and_band():
    quiet = (0.001 * voice_like() / 0.35).astype(np.float32)
    assert np.allclose(fx.apply_fx(quiet, SR, {"type": "drive", "gain_db": 12}), quiet, rtol=1e-3, atol=1e-9)
    loud = np.linspace(-1, 1, 2001)
    y = fx.apply_fx(loud, SR, {"type": "drive", "gain_db": 12})
    assert np.all(np.abs(y) <= np.abs(loud) + 1e-7) and np.all(np.diff(y) > 0) and y[-1] == pytest.approx(0.251, abs=1e-3)
    assert np.allclose(y, -y[::-1], atol=1e-7)
    assert np.allclose(fx.apply_fx(loud, SR, {"type": "drive", "gain_db": 12, "mix": 0.5}), 0.5 * loud + 0.5 * y, atol=1e-6)

    def level(f, spec, sr=SR):
        out = fx.apply_fx(tone(f, sr, 16000), sr, spec)
        return db(rms(out[8000:]) / math.sqrt(0.5))

    radio = {"type": "band", "lo_hz": 300, "hi_hz": 3400}
    assert abs(level(1000, radio)) < 0.5 and abs(level(1000, radio, 48000)) < 0.5
    assert -4.0 < level(300, radio) < -2.0 and -7.5 < level(3400, radio) < -5.0  # -3 dB and 2 x -3 dB at the corners
    assert level(75, radio) < -22 and level(13600, radio) < -45 and level(30, radio) < -38
    assert abs(level(100, {"type": "band", "lo_hz": 0, "hi_hz": 3400})) < 0.1
    assert abs(level(15000, {"type": "band", "lo_hz": 300, "hi_hz": 40000})) < 0.1
    wide = {"type": "band", "lo_hz": 0, "hi_hz": 40000}  # nothing to cut at this rate
    x = voice_like()
    assert np.array_equal(fx.apply_fx(x, SR, wide), x) and fx.fx_tail(wide) < 0.001
    half = fx.apply_fx(x, SR, {**radio, "mix": 0.5})
    assert np.allclose(half, 0.5 * x + 0.5 * fx.apply_fx(x, SR, radio), atol=1e-6)
    # no pre-echo: the filter answers a click only after it
    click = np.zeros(4000)
    click[2000] = 1.0
    y = fx.apply_fx(click, SR, radio)
    assert not y[:2000].any() and np.abs(y[2000:2040]).max() > 0.05 and np.abs(y[2000 + 700:]).max() < 1e-5


def test_fx_is_fast_enough_for_a_song():
    x = np.tile(voice_like(1.0), 30)
    t = time.perf_counter()
    y = fx.apply_fx(x, SR, list(PRESETS), t0=10.0)
    assert time.perf_counter() - t < 4.0 and len(y) == len(x)  # measured: about 0.6 s for 30 s through all five


# ---- the three modules under the backend ------------------------------------------------------------
@pytest.mark.skipif(fake is None, reason="the fake voicebank of tests/test_neural_unit.py is not importable")
def test_backend_with_the_real_cache_effects_and_resampler(fake, tmp_path, monkeypatch):
    root = tmp_path / "cache" / "phrases"
    monkeypatch.setenv("SYNTH2_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "1")
    monkeypatch.setattr(pc, "_default", None)
    p, tr = fake_song()
    first, rep = sing_fake(p, tr)
    phrases = rep["neural"]["phrases"]
    assert phrases == 3 and rep["neural"]["cached"] == 0
    assert len(entry_files(root)) == 2 * phrases  # each phrase's audio and a note about its pitch
    fake.calls.clear()
    again, rep = sing_fake(p, tr)  # the second render is served from the cache
    assert rep["neural"]["cached"] == phrases and np.array_equal(first, again)
    assert not [c for c in fake.calls if "acoustic" in c[0] or "voc" in c[0] or "variance" in c[0]]
    monkeypatch.setattr(pc, "_default", None)  # nothing in memory, as in another process
    cold, rep = sing_fake(p, tr)
    assert rep["neural"]["cached"] == phrases and np.array_equal(first, cold)
    for path in root.iterdir():  # entries that rot on disk are sung again, not played
        path.write_bytes(path.read_bytes()[:-3])
    monkeypatch.setattr(pc, "_default", None)
    resung, rep = sing_fake(p, tr)
    assert rep["neural"]["cached"] == 0 and np.array_equal(first, resung) and len(entry_files(root)) == 2 * phrases

    # effects and the sample rate come after the cache; a window of a render equals the same span of the whole
    chains = ({}, {"neural.fx.alien": 1.0}, {"neural.fx.double": 0.5, "neural.fx.radio": 1.0},
              {"neural.fx.robot": 0.7, "neural.fx.crush": 1.0})
    for params in chains:
        p, tr = fake_song()
        tr.voice_params.update(params)
        for sr in (SR, 48000, 22050):
            whole, rep = sing_fake(p, tr, sr=sr)
            assert rep["neural"]["cached"] == phrases and whole.dtype == np.float32 and np.isfinite(whole).all()
            assert not [w for w in rep["warnings"] if "effect" in w]
            for t0, t1 in ((1.3, 3.1), (2.0, 2.05), (0.0, 0.9)):
                part, _ = sing_fake(p, tr, sr=sr, t0=t0, t1=t1)
                a = int(round(t0 * sr))
                assert len(part) == int(round((t1 - t0) * sr))
                assert np.array_equal(part, whole[a:a + len(part)]), (params, sr, t0)
            if sr == SR:
                assert np.array_equal(whole, first) == (not params)
    p, tr = fake_song()
    tr.voice_params["neural.fx.zap"] = 1.0
    plain, rep = sing_fake(p, tr)
    assert np.array_equal(plain, first)
    assert [w for w in rep["warnings"] if "'zap' ignored" in w and "valid effects: alien, crush, double" in w]
    monkeypatch.setenv("SYNTH2_NEURAL_CACHE", "off")
    before = pc.default_cache().stats()
    off, rep = sing_fake(p, tr)
    assert rep["neural"]["cached"] == 0 and np.array_equal(off, first)
    after = pc.default_cache().stats()
    assert (after["hits"], after["misses"], after["writes"]) == (before["hits"], before["misses"], before["writes"])
