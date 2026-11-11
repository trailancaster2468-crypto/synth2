"""Sample data for the sample player: WAV through the standard library, FLAC/Ogg through
``soundfile`` when it is importable.

A :class:`Sample` knows its format and reads any span of frames as float32 on demand, so
a large sound set costs no memory until a note needs one of its samples. Nothing here
keeps a file open between reads.
"""
from __future__ import annotations

import os
import re
import struct
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

AUDIO_EXTS = (".wav", ".wave", ".flac", ".ogg", ".oga", ".aif", ".aiff")
_SOUNDFILE_EXTS = (".flac", ".ogg", ".oga", ".aif", ".aiff")


class SampleError(Exception):
    """A sample file or sound set cannot be read; the message says why."""


@dataclass
class Sample:
    """PCM audio of one sample, read lazily.

    ``read(a, b)`` returns frames ``[a, b)`` as float32 ``(b - a, channels)`` (zero-padded
    past the end). ``loop`` is the loop the file itself declares (``(start, end)`` with
    ``end`` exclusive), ``root`` and ``tune`` its unity note and fine tuning in cents.
    """

    key: str
    rate: int
    frames: int
    channels: int
    read: Callable[[int, int], np.ndarray] = field(repr=False, compare=False)
    name: str = ""
    loop: tuple[int, int] | None = None
    root: int | None = None
    tune: float = 0.0

    def all(self) -> np.ndarray:
        return self.read(0, self.frames)


def from_array(data: np.ndarray, rate: int, name: str = "array", loop: tuple[int, int] | None = None,
               root: int | None = None, key: str | None = None) -> Sample:
    """A :class:`Sample` over an array in memory (mono ``(n,)`` or ``(n, channels)``)."""
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]
    arr = np.ascontiguousarray(arr[:, :2])
    n = len(arr)

    def read(a: int, b: int) -> np.ndarray:
        a, b = max(0, int(a)), max(0, int(b))
        out = np.zeros((max(0, b - a), arr.shape[1]), np.float32)
        m = max(0, min(b, n) - a)
        if m:
            out[:m] = arr[a:a + m]
        return out

    return Sample(key=key or f"mem:{name}:{id(arr)}", rate=int(rate), frames=n, channels=arr.shape[1], read=read,
                  name=name, loop=loop, root=root)


# ------------------------------------------------------------------------------------------- WAV

_WAVE_PCM, _WAVE_FLOAT, _WAVE_EXTENSIBLE = 1, 3, 0xFFFE


def decode_pcm(raw: bytes, fmt: int, bits: int, channels: int) -> np.ndarray:
    """Interleaved little-endian PCM or float bytes -> float32 ``(frames, channels)``."""
    width = (bits + 7) // 8
    usable = len(raw) - len(raw) % (width * channels)
    buf = np.frombuffer(raw, dtype=np.uint8, count=usable)
    if fmt == _WAVE_FLOAT:
        if width == 4:
            x = buf.view("<f4").astype(np.float32)
        elif width == 8:
            x = buf.view("<f8").astype(np.float32)
        else:
            raise SampleError(f"{bits}-bit float WAV data is not supported")
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    elif width == 1:
        x = (buf.astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        x = buf.view("<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = buf.reshape(-1, 3).astype(np.int32)
        v = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        v -= (v & 0x800000) << 1
        x = v.astype(np.float32) / 8388608.0
    elif width == 4:
        x = (buf.view("<i4").astype(np.float64) / 2147483648.0).astype(np.float32)
    else:
        raise SampleError(f"{bits}-bit PCM WAV data is not supported")
    return x.reshape(-1, channels)


def _riff_chunks(f, size_limit: int):
    """Yield (id, offset, size) of the chunks of an open RIFF/WAVE file."""
    head = f.read(12)
    if len(head) < 12 or head[8:12] != b"WAVE":
        raise SampleError("not a WAVE file")
    if head[:4] == b"RF64":
        raise SampleError("RF64 WAVE files are not supported")
    if head[:4] != b"RIFF":
        raise SampleError("not a RIFF/WAVE file (big-endian RIFX is not supported)")
    pos = 12
    while pos + 8 <= size_limit:
        f.seek(pos)
        h = f.read(8)
        if len(h) < 8:
            break
        cid, size = h[:4], struct.unpack("<I", h[4:])[0]
        yield cid, pos + 8, min(size, max(0, size_limit - pos - 8))
        pos += 8 + size + (size & 1)


def open_wav(path: str) -> Sample:
    """A WAV file as a :class:`Sample`: 8/16/24/32-bit PCM and 32/64-bit float, mono or
    stereo (further channels are dropped), with the loop and unity note of its ``smpl``
    chunk when it has one."""
    path = os.fspath(path)
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            fmt = None
            data = None
            loop = root = None
            tune = 0.0
            for cid, off, n in _riff_chunks(f, size):
                if cid == b"fmt " and n >= 16:
                    f.seek(off)
                    raw = f.read(min(n, 40))
                    tag, ch, rate, _bps, align, bits = struct.unpack("<HHIIHH", raw[:16])
                    if tag == _WAVE_EXTENSIBLE and len(raw) >= 26:
                        tag = struct.unpack("<H", raw[24:26])[0]
                    fmt = (tag, ch, rate, align, bits)
                elif cid == b"data" and data is None:
                    data = (off, n)
                elif cid == b"smpl" and n >= 36:
                    f.seek(off)
                    raw = f.read(min(n, 36 + 24))
                    unity, frac = struct.unpack("<II", raw[12:20])
                    loops = struct.unpack("<I", raw[28:32])[0]
                    if 0 <= unity <= 127:
                        root = int(unity)
                        tune = -100.0 * frac / 4294967296.0 if frac else 0.0
                    if loops >= 1 and len(raw) >= 60:
                        _cue, _typ, ls, le = struct.unpack("<IIII", raw[36:52])
                        loop = (int(ls), int(le) + 1)  # the smpl chunk's loop end is inclusive
    except OSError as exc:
        raise SampleError(f"cannot read {path}: {exc}") from None
    if fmt is None or data is None:
        raise SampleError(f"{path}: no fmt or data chunk")
    tag, ch, rate, align, bits = fmt
    if tag not in (_WAVE_PCM, _WAVE_FLOAT):
        raise SampleError(f"{path}: compressed WAV data (format tag {tag}) is not supported; "
                          "convert it to PCM or float WAV")
    width = (bits + 7) // 8
    if ch < 1 or rate < 1 or width not in (1, 2, 3, 4, 8):
        raise SampleError(f"{path}: unsupported WAV format ({ch} channels, {bits} bits, {rate} Hz)")
    block = max(align, width * ch) if align >= width * ch else width * ch
    off, nbytes = data
    frames = nbytes // block
    if loop is not None and not (0 <= loop[0] < loop[1] <= frames):
        loop = None

    def read(a: int, b: int) -> np.ndarray:
        a, b = max(0, int(a)), max(0, int(b))
        out = np.zeros((max(0, b - a), min(ch, 2)), np.float32)
        m = max(0, min(b, frames) - a)
        if m:
            with open(path, "rb") as fh:
                fh.seek(off + a * block)
                raw = fh.read(m * block)
            if block != width * ch:  # padded frames: keep the sample bytes only
                raw = np.frombuffer(raw, np.uint8)[: (len(raw) // block) * block].reshape(-1, block)[:, : width * ch]
                raw = raw.tobytes()
            x = decode_pcm(raw, tag, bits, ch)[:, :2]
            out[: len(x)] = x
        return out

    st = os.stat(path)
    return Sample(key=f"{path}|{st.st_size}|{st.st_mtime_ns}", rate=int(rate), frames=int(frames),
                  channels=min(ch, 2), read=read, name=os.path.splitext(os.path.basename(path))[0], loop=loop,
                  root=root, tune=tune)


def have_soundfile() -> bool:
    try:
        import soundfile  # noqa: F401
    except Exception:  # an installed but broken libsndfile raises OSError
        return False
    return True


def open_soundfile(path: str) -> Sample:
    """A FLAC/Ogg/AIFF file as a :class:`Sample` (needs the optional ``soundfile`` package)."""
    path = os.fspath(path)
    try:
        import soundfile as sf
    except Exception:
        ext = os.path.splitext(path)[1].lower()
        raise SampleError(f"{path}: reading {ext} samples needs the optional 'soundfile' package "
                          "(pip install soundfile); WAV samples work without it") from None
    try:
        info = sf.info(path)
    except Exception as exc:
        raise SampleError(f"cannot read {path}: {exc}") from None
    frames, ch, rate = int(info.frames), int(info.channels), int(info.samplerate)

    def read(a: int, b: int) -> np.ndarray:
        a, b = max(0, int(a)), max(0, int(b))
        out = np.zeros((max(0, b - a), min(ch, 2)), np.float32)
        m = max(0, min(b, frames) - a)
        if m:
            with sf.SoundFile(path) as fh:
                fh.seek(a)
                x = fh.read(m, dtype="float32", always_2d=True)[:, :2]
            out[: len(x)] = x
        return out

    st = os.stat(path)
    return Sample(key=f"{path}|{st.st_size}|{st.st_mtime_ns}", rate=rate, frames=frames, channels=min(ch, 2),
                  read=read, name=os.path.splitext(os.path.basename(path))[0])


def open_sample(path: str) -> Sample:
    """Open a sample file by extension. Raises :class:`SampleError` with a plain message."""
    ext = os.path.splitext(os.fspath(path))[1].lower()
    if not os.path.isfile(path):
        raise SampleError(f"sample file not found: {path}")
    if ext in (".wav", ".wave"):
        return open_wav(path)
    if ext in _SOUNDFILE_EXTS:
        return open_soundfile(path)
    raise SampleError(f"{path}: unsupported sample format {ext or '(no extension)'}")


def write_wav(path: str, data: np.ndarray, rate: int, bits: int = 16, float_format: bool = False,
              loop: tuple[int, int] | None = None, root: int | None = None) -> None:
    """Write a WAV file (PCM 8/16/24/32 or float 32/64), optionally with a ``smpl`` chunk."""
    x = np.asarray(data, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    ch = x.shape[1]
    if float_format:
        raw = x.astype("<f4" if bits == 32 else "<f8").tobytes()
        tag = _WAVE_FLOAT
    else:
        tag = _WAVE_PCM
        x = np.clip(x, -1.0, 1.0)
        if bits == 8:
            raw = np.clip(np.round(x * 127.0 + 128.0), 0, 255).astype(np.uint8).tobytes()
        elif bits == 16:
            raw = np.round(x * 32767.0).astype("<i2").tobytes()
        elif bits == 24:
            v = np.round(x * 8388607.0).astype(np.int32).reshape(-1)
            b = np.stack([v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF], axis=1).astype(np.uint8)
            raw = b.tobytes()
        elif bits == 32:
            raw = np.round(x * 2147483647.0).astype("<i4").tobytes()
        else:
            raise ValueError("bits must be 8, 16, 24 or 32")
    width = bits // 8
    chunks = [b"fmt " + struct.pack("<IHHIIHH", 16, tag, ch, int(rate), int(rate) * ch * width, ch * width, bits),
              b"data" + struct.pack("<I", len(raw)) + raw + (b"\0" if len(raw) & 1 else b"")]
    if loop is not None or root is not None:
        body = struct.pack("<IIIIIIIII", 0, 0, int(1e9 / rate), 60 if root is None else int(root), 0, 0, 0,
                           1 if loop else 0, 0)
        if loop:
            body += struct.pack("<IIIIII", 0, 0, int(loop[0]), int(loop[1]) - 1, 0, 0)
        chunks.append(b"smpl" + struct.pack("<I", len(body)) + body)
    payload = b"WAVE" + b"".join(chunks)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(payload)) + payload)


# ------------------------------------------------------------------------------- names in file names

_NOTE_RE = re.compile(r"(?<![A-Za-z])([A-Ga-g])([#sb]?)(-?\d)(?![0-9A-Za-z])")
_MIDI_RE = re.compile(r"(?<![0-9A-Za-z])(?:m|n|note|key)?[ _-]?(\d{2,3})(?![0-9A-Za-z])", re.I)
_SEMIS = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}


def note_from_name(name: str) -> int | None:
    """MIDI note named in a file name (``Piano_C#4.wav``, ``bass-Eb1``, ``A3_v2``; middle C = C4
    = 60), or a bare MIDI number 21..108 (``epiano_060``). None when there is none."""
    stem = os.path.splitext(os.path.basename(name))[0]
    hits = _NOTE_RE.findall(stem)
    if hits:
        letter, acc, octave = hits[-1]
        semi = _SEMIS[letter.lower()] + (1 if acc in ("#", "s") else -1 if acc == "b" else 0)
        midi = 12 * (int(octave) + 1) + semi
        if 0 <= midi <= 127:
            return midi
    for m in reversed(_MIDI_RE.findall(stem)):
        v = int(m)
        if 21 <= v <= 108:
            return v
    return None


_VEL_RE = re.compile(r"(?<![A-Za-z])(?:v|vel|velocity|l|layer)[ _-]?(\d{1,3})(?![0-9])", re.I)
_RR_RE = re.compile(r"(?<![A-Za-z])(?:rr|round|alt|var|take)[ _-]?(\d{1,2})(?![0-9])", re.I)
_DYN = {"ppp": 16, "pp": 33, "p": 49, "mp": 64, "mf": 80, "f": 96, "ff": 112, "fff": 127}
_DYN_RE = re.compile(r"(?<![A-Za-z])(ppp|pp|p|mp|mf|fff|ff|f)(?![A-Za-z])")


def layer_from_name(name: str) -> tuple[int | None, int | None]:
    """(velocity layer, round-robin index) hinted by a file name: ``snare_v3_rr2``,
    ``C4_mf``, ``kick-vel100``, ``kick-accent``, ``snare_soft``. Either may be None."""
    stem = os.path.splitext(os.path.basename(name))[0]
    vel = rr = None
    m = _VEL_RE.search(stem)
    if m:
        vel = int(m.group(1))
    else:
        m = _DYN_RE.search(stem)
        if m:
            vel = _DYN[m.group(1)]
    if vel is None and re.search(r"(?<![A-Za-z])(accent|acc|hard|loud)(?![A-Za-z])", stem, re.I):
        vel = 127
    elif vel is None and re.search(r"(?<![A-Za-z])(soft|quiet|ghost)(?![A-Za-z])", stem, re.I):
        vel = 40
    m = _RR_RE.search(stem)
    if m:
        rr = int(m.group(1))
    return vel, rr
