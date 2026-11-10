"""The phrase cache of the neural engine: rendered phrases on disk, keyed by a hash of
everything their audio depends on.

:func:`stable_hash` turns nested lists, dicts, numbers, strings and numpy arrays into a key
that is the same in every process and on every platform. :class:`PhraseCache` stores one
float32 ``.npy`` file per key in one folder::

    cache = default_cache()
    key = stable_hash(engine_version, phonemes, durations, pitch, seed)
    audio = cache.get(key)
    if audio is None:
        audio = sing(...)
        cache.put(key, audio)

The cache never raises: a folder that cannot be written leaves the in-memory layer, an entry
that cannot be read is a miss. Entries are written under a temporary name and renamed, so a
reader never sees half a file, and any number of threads and processes may share a folder.
When the folder grows past its limit the entries read or written longest ago are removed.
"""
from __future__ import annotations

import hashlib
import itertools
import math
import os
import re
import sys
import tempfile
import threading
import time
from collections import OrderedDict
from pathlib import Path, PurePath

import numpy as np

DIR_ENV = "SYNTH2_CACHE"
ENABLED_ENV = "SYNTH2_NEURAL_CACHE"
SIZE_ENV = "SYNTH2_NEURAL_CACHE_MB"
OFF_VALUES = ("0", "false", "off", "no")
DEFAULT_MB = 2048.0
MEMORY_BYTES = 64 << 20
TRIM_TO = 0.9          # an eviction shrinks the folder to this fraction of the limit
RESCAN_PUTS = 64       # writes between two counts of the folder (other processes write too)
STALE_TMP = 3600.0     # s after which a temporary file left by a dead writer is removed

_HASH_PREFIX = b"synth2-stable-hash-1\n"
_ENTRY = re.compile(r"[0-9a-f]{64}\.npy")
_TMP = re.compile(r"[0-9a-f]{64}\.[0-9a-z.-]+\.tmp")
_HEX = re.compile(r"[0-9a-f]{64}")
_serial = itertools.count()


# ---- keys ------------------------------------------------------------------------------------------
class _Bytes:
    """Collects what a hash object would be fed."""

    def __init__(self):
        self.parts: list[bytes] = []

    def update(self, b) -> None:
        self.parts.append(bytes(b))


def _feed(h, v) -> None:
    """Feed the canonical encoding of ``v`` to ``h``. Every value is self-delimiting and starts
    with a type tag, so different values never share an encoding."""
    if v is None:
        h.update(b"N")
    elif isinstance(v, (bool, np.bool_)):
        h.update(b"T" if v else b"F")
    elif isinstance(v, np.ndarray):
        if v.dtype.kind not in "biufc":
            raise TypeError(f"stable_hash cannot hash an array of dtype {v.dtype}")
        if v.dtype.byteorder == ">" or (v.dtype.byteorder == "=" and sys.byteorder == "big"):
            v = v.astype(v.dtype.newbyteorder("<"))
        h.update(f"a{v.dtype.str};{v.ndim}:{','.join(str(n) for n in v.shape)};".encode())
        h.update(np.ascontiguousarray(v).tobytes())
    elif isinstance(v, np.generic):
        item = v.item()
        if isinstance(item, np.generic):
            raise TypeError(f"stable_hash cannot hash a numpy scalar of type {type(v).__name__}")
        _feed(h, item)
    elif isinstance(v, int):
        h.update(b"i%d;" % int(v))
    elif isinstance(v, float):
        h.update(b"f" + float.__repr__(float(v)).encode() + b";")
    elif isinstance(v, str):
        b = v.encode("utf-8", "surrogatepass")
        h.update(b"s%d:" % len(b))
        h.update(b)
    elif isinstance(v, (bytes, bytearray, memoryview)):
        b = bytes(v)
        h.update(b"b%d:" % len(b))
        h.update(b)
    elif isinstance(v, PurePath):
        b = v.as_posix().encode("utf-8", "surrogatepass")
        h.update(b"p%d:" % len(b))
        h.update(b)
    elif isinstance(v, (list, tuple)):
        h.update(b"l%d:" % len(v))
        for x in v:
            _feed(h, x)
    elif isinstance(v, dict):
        items = []
        for k, x in v.items():
            kb = _Bytes()
            _feed(kb, k)
            items.append((b"".join(kb.parts), x))
        items.sort(key=lambda kv: kv[0])
        h.update(b"d%d:" % len(items))
        for kb, x in items:
            h.update(kb)
            _feed(h, x)
    else:
        raise TypeError(f"stable_hash cannot hash a value of type {type(v).__name__}; it takes None, bool, int, "
                        "float, str, bytes, paths, lists, tuples, dicts and numeric numpy arrays")


def stable_hash(*parts) -> str:
    """sha256 (64 hex digits) of a canonical encoding of ``parts``.

    Takes None, bool, int, float, str, bytes, paths, numeric numpy arrays and scalars, and
    lists, tuples and dicts of these. The same values give the same hash in every process and
    on every platform. A dict's order does not matter; a tuple equals the list of its items
    and a numpy scalar equals its Python value; everything else that differs gives another
    hash: ``1``, ``1.0``, ``True`` and ``"1"``; ``0.0`` and ``-0.0``; an array's dtype, shape
    and every bit of its data. Other types raise :class:`TypeError`."""
    h = hashlib.sha256(_HASH_PREFIX)
    _feed(h, list(parts))
    return h.hexdigest()


# ---- settings ---------------------------------------------------------------------------------------
def cache_dir() -> Path:
    """The folder phrases are cached in (it may not exist yet).

    ``$SYNTH2_CACHE/phrases`` if set, else ``%LOCALAPPDATA%\\synth2\\cache\\phrases`` on Windows
    and ``$XDG_CACHE_HOME/synth2/phrases`` (default ``~/.cache/synth2/phrases``) elsewhere."""
    env = os.environ.get(DIR_ENV)
    if env:
        return Path(env).expanduser() / "phrases"
    try:
        home = Path.home()
    except (RuntimeError, KeyError, OSError):  # no home directory: a per-user folder is not available
        home = None
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or (str(home / "AppData" / "Local") if home else tempfile.gettempdir())
        return Path(base) / "synth2" / "cache" / "phrases"
    base = os.environ.get("XDG_CACHE_HOME") or (str(home / ".cache") if home else tempfile.gettempdir())
    return Path(base) / "synth2" / "phrases"


def enabled() -> bool:
    """False when ``SYNTH2_NEURAL_CACHE`` is ``0``, ``false``, ``off`` or ``no``."""
    return (os.environ.get(ENABLED_ENV) or "").strip().lower() not in OFF_VALUES


def limit_bytes() -> int:
    """The size limit of the folder in bytes: ``SYNTH2_NEURAL_CACHE_MB`` megabytes, 2048 when it
    is unset or not a number. 0 keeps nothing on disk."""
    raw = (os.environ.get(SIZE_ENV) or "").strip()
    try:
        mb = float(raw) if raw else DEFAULT_MB
    except ValueError:
        mb = DEFAULT_MB
    if not math.isfinite(mb) or mb < 0:
        mb = DEFAULT_MB
    return int(mb * (1 << 20))


# ---- the cache --------------------------------------------------------------------------------------
def _name(key) -> str:
    """The file stem of a key: the key itself when it is a sha256 hex digest, else its sha256."""
    key = str(key)
    return key if _HEX.fullmatch(key) else hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()


def _remove(path) -> bool:
    try:
        os.remove(path)
        return True
    except OSError:
        return False


def _load(path) -> np.ndarray:
    """The float32 array in an entry file. Raises FileNotFoundError when there is none and
    ValueError when the file is not a complete entry."""
    fmt = np.lib.format
    with open(path, "rb") as f:
        size = os.fstat(f.fileno()).st_size
        try:
            version = fmt.read_magic(f)
            if version == (1, 0):
                shape, fortran, dtype = fmt.read_array_header_1_0(f)
            elif version == (2, 0):
                shape, fortran, dtype = fmt.read_array_header_2_0(f)
            else:
                raise ValueError(f"unsupported .npy version {version}")
        except (ValueError, EOFError, SyntaxError, TypeError, UnicodeError) as exc:
            raise ValueError(f"not an .npy file: {exc}") from None
        count = 1
        for n in shape:
            count *= int(n)
        if dtype != np.dtype("<f4") or fortran or size - f.tell() != 4 * count:
            raise ValueError("not a complete float32 entry")
        data = f.read(4 * count)
    if len(data) != 4 * count:
        raise ValueError("the file ends early")
    arr = np.frombuffer(data, "<f4").astype(np.float32).reshape(shape)
    if not np.isfinite(arr).all():
        raise ValueError("the entry holds values that are not finite")
    return arr


class PhraseCache:
    """Float32 arrays by key, in ``root`` (default :func:`cache_dir`) and in memory.

    ``max_bytes`` limits the folder (default :func:`limit_bytes`; 0 keeps nothing on disk) and
    ``memory_bytes`` the copies kept in memory. :meth:`get` and :meth:`put` never raise."""

    def __init__(self, root=None, max_bytes=None, memory_bytes=MEMORY_BYTES):
        self.root = Path(root) if root is not None else cache_dir()
        self.max_bytes = limit_bytes() if max_bytes is None else int(max_bytes)
        self.memory_bytes = int(memory_bytes)
        self._lock = threading.Lock()
        self._mem: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._mem_used = 0
        self._disk_used: int | None = None  # bytes in the folder when last counted, plus writes since
        self._puts = 0
        self._n = {"hits": 0, "misses": 0, "writes": 0, "evicted": 0, "errors": 0}

    def _path(self, name: str) -> Path:
        return self.root / f"{name}.npy"

    def _count(self, what: str, n: int = 1) -> None:
        with self._lock:
            self._n[what] += n

    def _remember(self, name: str, arr: np.ndarray) -> None:
        """Keep ``arr`` in memory, dropping the entries used longest ago (lock held)."""
        old = self._mem.pop(name, None)
        if old is not None:
            self._mem_used -= old.nbytes
        if arr.nbytes > self.memory_bytes:
            return
        self._mem[name] = arr
        self._mem_used += arr.nbytes
        while self._mem_used > self.memory_bytes:
            _, dropped = self._mem.popitem(last=False)
            self._mem_used -= dropped.nbytes

    # ---- reading
    def get(self, key: str):
        """A copy of the array stored under ``key``, or None. Reading an entry makes it the
        most recently used one."""
        try:
            return self._get(_name(key))
        except Exception:  # the cache must never break a render
            self._count("errors")
            return None

    def _get(self, name: str):
        with self._lock:
            arr = self._mem.get(name)
            if arr is not None:
                self._mem.move_to_end(name)
                self._n["hits"] += 1
        path = self._path(name)
        if arr is None:
            try:
                arr = _load(path)
            except ValueError:  # corrupt or cut short: remove it so that it is rendered again
                _remove(path)
                self._count("errors")
            except OSError:  # no such entry, or a folder that cannot be read
                pass
            with self._lock:
                self._n["hits" if arr is not None else "misses"] += 1
                if arr is not None:
                    self._remember(name, arr)
            if arr is None:
                return None
        try:
            os.utime(path)  # eviction goes by modification time
        except OSError:
            pass
        return arr.copy()

    # ---- writing
    def put(self, key: str, audio) -> None:
        """Store ``audio`` (converted to float32) under ``key``. Arrays with values that are not
        finite are not stored."""
        try:
            arr = np.array(audio, dtype=np.float32, order="C")
            if not np.isfinite(arr).all():
                return
            self._put(_name(key), arr)
        except Exception:  # the cache must never break a render
            self._count("errors")

    def _put(self, name: str, arr: np.ndarray) -> None:
        with self._lock:
            self._remember(name, arr)
        if self.max_bytes <= 0 or arr.nbytes > self.max_bytes:
            return
        tmp = self.root / f"{name}.{os.getpid()}-{threading.get_ident()}-{next(_serial)}.tmp"
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            with open(tmp, "wb") as f:
                np.lib.format.write_array(f, arr, version=(1, 0), allow_pickle=False)
                size = f.tell()
            os.replace(tmp, self._path(name))
        except OSError:
            _remove(tmp)
            self._count("errors")
            return
        with self._lock:
            self._n["writes"] += 1
            self._puts += 1
            known = self._disk_used is not None
            if known:
                self._disk_used += size
            check = not known or self._disk_used > self.max_bytes or self._puts % RESCAN_PUTS == 0
        if check:
            self._evict()

    def _scan(self) -> list[tuple[int, int, str]]:
        """``(modification time in ns, size, path)`` of every entry in the folder. Temporary
        files of writers that died are removed on the way."""
        out = []
        now = time.time()
        try:
            with os.scandir(self.root) as it:
                for e in it:
                    try:
                        if _ENTRY.fullmatch(e.name):
                            st = e.stat()
                            out.append((st.st_mtime_ns, st.st_size, e.path))
                        elif _TMP.fullmatch(e.name) and now - e.stat().st_mtime > STALE_TMP:
                            _remove(e.path)
                    except OSError:
                        continue
        except OSError:
            pass
        return out

    def _evict(self) -> None:
        """Count the folder and, above the limit, remove the entries used longest ago until it
        is at ``TRIM_TO`` of the limit."""
        files = self._scan()
        total = sum(size for _, size, _ in files)
        removed = 0
        if total > self.max_bytes:
            target = int(self.max_bytes * TRIM_TO)
            for _, size, path in sorted(files):
                if total <= target:
                    break
                if _remove(path):
                    removed += 1
                total -= size  # gone either way: another process may have removed it first
        with self._lock:
            self._disk_used = total
            self._n["evicted"] += removed

    # ---- housekeeping
    def clear(self) -> None:
        """Remove every entry, on disk and in memory. Other files in the folder are left alone."""
        with self._lock:
            self._mem.clear()
            self._mem_used = 0
            self._disk_used = 0
        try:
            names = os.listdir(self.root)
        except OSError:
            return
        for name in names:
            if _ENTRY.fullmatch(name) or _TMP.fullmatch(name):
                _remove(os.path.join(self.root, name))

    def stats(self) -> dict:
        """Counters of this instance (``hits``, ``misses``, ``writes``, ``evicted``, ``errors``),
        what the folder holds now (``entries``, ``bytes``, ``root``, ``max_bytes``) and what is
        kept in memory (``memory_entries``, ``memory_bytes``)."""
        files = self._scan()
        with self._lock:
            return {**self._n, "entries": len(files), "bytes": sum(size for _, size, _ in files),
                    "root": str(self.root), "max_bytes": self.max_bytes, "memory_entries": len(self._mem),
                    "memory_bytes": self._mem_used}


_default: PhraseCache | None = None
_default_lock = threading.Lock()


def default_cache() -> PhraseCache:
    """The shared cache of the current :func:`cache_dir`, with the current size limit."""
    global _default
    root = cache_dir()
    with _default_lock:
        if _default is None or _default.root != root:
            _default = PhraseCache(root)
        else:
            _default.max_bytes = limit_bytes()
        return _default
