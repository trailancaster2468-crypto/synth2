"""WAV file reading/writing (16/24-bit PCM, 32-bit float) using only the stdlib + numpy."""
from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int, bits: int = 16) -> Path:
    """Write ``audio`` (shape ``(n,)`` or ``(n, channels)``, float in [-1, 1]) to ``path``."""
    path = Path(path)
    a = np.asarray(audio, dtype=np.float64)
    if a.ndim == 1:
        a = a[:, None]
    channels = a.shape[1]
    a = np.clip(a, -1.0, 1.0)
    if bits == 32:
        data = a.astype("<f4").tobytes()
        _write_float_wav(path, data, sample_rate, channels)
        return path
    if bits == 16:
        data = (a * 32767.0).round().astype("<i2").tobytes()
    elif bits == 24:
        ints = (a * 8388607.0).round().astype("<i4").reshape(-1)
        b = ints.view(np.uint8).reshape(-1, 4)[:, :3]
        data = b.tobytes()
    else:
        raise ValueError("bits must be 16, 24 or 32")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(bits // 8)
        w.setframerate(int(sample_rate))
        w.writeframes(data)
    return path


def _write_float_wav(path: Path, data: bytes, sr: int, channels: int):
    block = channels * 4
    fmt = struct.pack("<HHIIHH", 3, channels, sr, sr * block, block, 32)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data)) + b"WAVE")
        f.write(b"fmt " + struct.pack("<I", len(fmt)) + fmt)
        f.write(b"data" + struct.pack("<I", len(data)) + data)


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a PCM/float WAV file. Returns ``(audio float64 (n, channels), sample_rate)``."""
    raw = Path(path).read_bytes()
    if raw[:4] != b"RIFF" or raw[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos = 12
    fmt = None
    data = None
    while pos + 8 <= len(raw):
        cid = raw[pos:pos + 4]
        size = struct.unpack("<I", raw[pos + 4:pos + 8])[0]
        body = raw[pos + 8:pos + 8 + size]
        if cid == b"fmt ":
            fmt = struct.unpack("<HHIIHH", body[:16])
        elif cid == b"data":
            data = body
        pos += 8 + size + (size & 1)
    if fmt is None or data is None:
        raise ValueError("missing fmt or data chunk")
    tag, channels, sr, _, _, bits = fmt
    if tag == 0xFFFE:  # WAVE_FORMAT_EXTENSIBLE: assume PCM unless 32-bit float
        tag = 3 if bits == 32 else 1
    if tag == 3:
        a = np.frombuffer(data, dtype="<f4" if bits == 32 else "<f8").astype(np.float64)
    elif bits == 16:
        a = np.frombuffer(data, dtype="<i2") / 32768.0
    elif bits == 24:
        b = np.frombuffer(data[: len(data) // 3 * 3], dtype=np.uint8).reshape(-1, 3)
        ints = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8) | (b[:, 2].astype(np.int32) << 16))
        ints = np.where(ints >= 1 << 23, ints - (1 << 24), ints)
        a = ints / 8388608.0
    elif bits == 8:
        a = (np.frombuffer(data, dtype=np.uint8).astype(np.float64) - 128) / 128.0
    elif bits == 32:
        a = np.frombuffer(data, dtype="<i4") / 2147483648.0
    else:
        raise ValueError(f"unsupported bit depth {bits}")
    a = a[: len(a) // channels * channels].reshape(-1, channels)
    return a, sr
