"""Shared installer for everything Synth2 downloads: neural models and sound sets.

An installed item is one folder below a base directory::

    <base>/<id>/<manifest name>     JSON manifest, written last
    <base>/<id>/files/...           the unmodified contents of the downloads

This module fetches the sources of an item (from a URL or from local files), checks each
against a pinned size and SHA-256, unpacks archives without trusting anything in them, and
moves the finished folder into place in one step. Nothing in a download is ever executed.

``synth2.neural.models`` and ``synth2.sounds.installer`` hold the registries and call into
here. Everything that touches the network or archives is imported inside the functions that
need it, so importing this module stays cheap.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

USER_AGENT = "synth2-installer"
CHUNK = 1 << 20
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
ARCHIVE_KINDS = ("zip", "tar", "deb")
Progress = Callable[[int, int, str], None]   # (bytes done, bytes expected, label)


class AssetError(Exception):
    """An install, add or remove could not be carried out."""


class DownloadError(AssetError):
    """A source could not be fetched."""


class VerificationError(AssetError):
    """A download does not have the pinned size or SHA-256."""


class UnsafeArchive(AssetError):
    """An archive holds entries that are not plain files inside the target folder, or it
    unpacks to more than the limits allow."""


class LicenseNotAccepted(AssetError):
    """Raised before anything is fetched when the licence terms were not accepted. Carries
    what the user has to read: ``name``, ``license``, ``terms``, ``credits``, ``url`` and
    the publisher's ``statement`` on commercial use."""

    def __init__(self, item_id: str, *, name: str = "", license: str = "", terms=(), credits=(),
                 url: str = "", commercial_use: bool | None = None, statement: str = ""):
        self.id = item_id
        self.name = name or item_id
        self.license = license
        self.terms = list(terms)
        self.credits = list(credits)
        self.url = url
        self.commercial_use = commercial_use
        self.statement = statement
        super().__init__(f"the licence of {self.name!r} ({license or 'see ' + url}) has not been "
                         f"accepted; pass accept_license=True (--accept-license) after reading it")

    def text(self) -> str:
        """The terms as plain text for a terminal."""
        return license_text(self.name, self.license, self.terms, self.credits, self.url,
                            self.commercial_use, self.statement)


def license_text(name, license, terms=(), credits=(), url="", commercial_use=None,
                 statement="") -> str:
    lines = [f"{name}", f"  Licence: {license or '(not stated)'}"]
    if url:
        lines.append(f"  Read it: {url}")
    lines.append("  Commercial use: " + commercial_label(commercial_use))
    if statement:
        lines.append(f"  The publisher says: \"{statement}\"")
    lines += [f"  - {t}" for t in terms]
    lines += [f"  Credit: {c}" for c in credits]
    return "\n".join(lines)


def commercial_label(value: bool | None) -> str:
    """``commercial_use`` as shown in listings."""
    return {True: "stated: yes", False: "stated: no"}.get(value, "not stated")


@dataclass(frozen=True)
class Limits:
    """Upper bounds for one unpacked item."""

    max_entries: int = 50_000
    max_total: int = 12 << 30          # bytes after unpacking
    max_ratio: float = 100.0           # unpacked bytes / archive bytes
    ratio_floor: int = 8 << 20         # the ratio is not checked below this unpacked size


DEFAULT_LIMITS = Limits()


@dataclass(frozen=True)
class Source:
    """One file to fetch. ``url`` is an http(s) URL; ``size`` and ``sha256`` pin its contents.

    ``unpack``: ``"auto"`` (zip, tar or deb by the file's own signature, else kept as a
    file), one of :data:`ARCHIVE_KINDS`, or ``"none"`` (keep the file as it is). ``dest`` is
    the folder below ``files/`` an archive is unpacked into or a plain file is stored in.
    ``members`` restricts an archive to entries below these paths. ``filename`` is the name
    a plain file is stored under and the name looked for in a local source folder (default:
    the last part of the URL).
    """

    url: str
    sha256: str
    size: int
    filename: str = ""
    unpack: str = "auto"
    dest: str = ""
    members: tuple[str, ...] = ()

    @property
    def name(self) -> str:
        if self.filename:
            return self.filename
        from urllib.parse import unquote, urlsplit

        return unquote(urlsplit(self.url).path.rsplit("/", 1)[-1]) or "download"


# ----------------------------------------------------------------------------- small helpers

def check_id(item_id: str) -> str:
    """``item_id`` if it can be used as a folder name, else :class:`AssetError`."""
    if not isinstance(item_id, str) or not ID_RE.match(item_id) or item_id.endswith("."):
        raise AssetError(f"invalid id {item_id!r}: use letters, digits, '.', '_' and '-' "
                         "(at most 64 characters, starting with a letter or digit)")
    return item_id


def slug(text: str) -> str:
    """An id made from a file or folder name."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-._").lower()
    return s[:64].rstrip("-._") or "item"


def sha256_file(path, progress: Progress | None = None, label: str = "") -> tuple[str, int]:
    """``(hex digest, size)`` of a file."""
    import hashlib

    h = hashlib.sha256()
    done = 0
    total = os.path.getsize(path)
    with open(path, "rb") as fh:
        while True:
            block = fh.read(CHUNK)
            if not block:
                break
            h.update(block)
            done += len(block)
            if progress:
                progress(done, total, label)
    return h.hexdigest(), done


def tree_size(path) -> int:
    """Total size of the regular files below ``path`` (links are not followed)."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    total = 0
    for root, _dirs, names in os.walk(path):
        for n in names:
            try:
                st = os.lstat(os.path.join(root, n))
            except OSError:
                continue
            if (st.st_mode & 0o170000) == 0o100000:
                total += st.st_size
    return total


def now_stamp() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def format_size(n: int | float | None) -> str:
    if not n:
        return "-"
    n = float(n)
    for unit in ("B", "kB", "MB"):
        if n < 1000:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.2f} GB"


def dump_json(data: dict, width: int = 100) -> str:
    """JSON text for a manifest: two-space indent, non-ASCII kept, lists of plain values on
    one line when they fit in ``width`` columns."""

    def scalar(v) -> bool:
        return v is None or isinstance(v, (str, int, float, bool))

    def render(v, indent: int, lead: int) -> str:
        pad = "  " * indent
        if isinstance(v, dict):
            if not v:
                return "{}"
            rows = []
            for k, item in v.items():
                key = json.dumps(str(k), ensure_ascii=False) + ": "
                rows.append(f"{pad}  {key}{render(item, indent + 1, len(pad) + 2 + len(key))}")
            return "{\n" + ",\n".join(rows) + f"\n{pad}}}"
        if isinstance(v, (list, tuple)):
            if not v:
                return "[]"
            if all(scalar(x) for x in v):
                flat = "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in v) + "]"
                if lead + len(flat) + 1 <= width:
                    return flat
            rows = [f"{pad}  {render(x, indent + 1, len(pad) + 2)}" for x in v]
            return "[\n" + ",\n".join(rows) + f"\n{pad}]"
        return json.dumps(v, ensure_ascii=False)

    return render(data, 0, 0) + "\n"


def write_manifest(folder, manifest_name: str, data: dict) -> Path:
    """Write ``data`` as ``<folder>/<manifest_name>`` through a temp file and a rename."""
    folder = Path(folder)
    tmp = folder / (manifest_name + ".tmp")
    tmp.write_text(dump_json(data), encoding="utf-8")
    os.replace(tmp, folder / manifest_name)
    return folder / manifest_name


def require_license(accepted: bool, item_id: str, *, name="", license="", terms=(), credits=(),
                    url="", commercial_use=None, statement="") -> None:
    """Raise :class:`LicenseNotAccepted` unless ``accepted``. Call before any network access."""
    if not accepted:
        raise LicenseNotAccepted(item_id, name=name, license=license, terms=terms,
                                 credits=credits, url=url, commercial_use=commercial_use,
                                 statement=statement)


def confirm(prompt: str, stream=None) -> bool | None:
    """Ask a yes/no question on a terminal. None when stdin is not a terminal."""
    stream = stream or sys.stderr
    try:
        if not sys.stdin.isatty():
            return None
    except (AttributeError, ValueError):
        return None
    print(prompt, end=" ", file=stream, flush=True)
    try:
        answer = sys.stdin.readline()
    except KeyboardInterrupt:
        return False
    return answer.strip().lower() in ("y", "yes")


def terminal_progress(stream=None) -> Progress:
    """A progress callback that draws one status line on ``stream`` (default stderr)."""
    stream = stream or sys.stderr
    state = {"last": -1, "label": None}

    def show(done: int, total: int, label: str) -> None:
        pct = int(100 * done / total) if total else 0
        if label == state["label"] and pct == state["last"]:
            return
        state["label"], state["last"] = label, pct
        print(f"\r  {label[:48]:48s} {pct:3d}%  {format_size(done):>10s}", end="", file=stream,
              flush=True)
        if total and done >= total:
            print(file=stream)

    return show


# ----------------------------------------------------------------------------- fetching

def ssl_context():
    """A verifying TLS context. ``SSL_CERT_FILE`` or ``REQUESTS_CA_BUNDLE`` name an extra CA
    bundle (as behind an inspecting proxy); verification is never turned off."""
    import ssl

    ctx = ssl.create_default_context()
    for var in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
        bundle = os.environ.get(var)
        if bundle and os.path.isfile(bundle):
            try:
                ctx.load_verify_locations(cafile=bundle)
            except (OSError, ssl.SSLError):
                pass
    return ctx


def _is_url(text: str) -> bool:
    return bool(re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", str(text)))


def download(url: str, dest, *, size: int | None = None, progress: Progress | None = None,
             label: str = "", timeout: float = 60.0, retries: int = 3) -> int:
    """Stream ``url`` (http or https) into the file ``dest``. Stops as soon as more than
    ``size`` bytes arrive. Returns the number of bytes written."""
    import time
    import urllib.error
    import urllib.request
    from urllib.parse import urlsplit

    scheme = urlsplit(url).scheme.lower()
    if scheme not in ("http", "https"):
        raise DownloadError(f"unsupported URL scheme in {url!r} (http and https only)")
    opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context()))
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "*/*"})
    label = label or url.rsplit("/", 1)[-1]
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        if attempt:
            time.sleep(min(8.0, 1.5 * 2 ** (attempt - 1)))
        try:
            done = 0
            with opener.open(request, timeout=timeout) as resp, open(dest, "wb") as out:
                expected = size or int(resp.headers.get("Content-Length") or 0)
                while True:
                    block = resp.read(CHUNK)
                    if not block:
                        break
                    done += len(block)
                    if size is not None and done > size:
                        raise VerificationError(
                            f"{label}: the server sent more than the expected {size} bytes")
                    out.write(block)
                    if progress:
                        progress(done, expected, label)
            if size is not None and done < size:
                raise DownloadError(f"{label}: connection closed after {done} of {size} bytes")
            return done
        except VerificationError:
            raise
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code not in (408, 425, 429, 500, 502, 503, 504):
                break
        except (urllib.error.URLError, OSError, DownloadError) as exc:
            last = exc
    reason = f"HTTP {last.code} {last.reason}" if hasattr(last, "code") else str(last)
    raise DownloadError(f"could not download {url}: {reason}")


def verify_file(path, sha256: str, size: int | None = None, label: str = "") -> None:
    """Raise :class:`VerificationError` unless the file has this size and SHA-256."""
    label = label or os.path.basename(str(path))
    actual_size = os.path.getsize(path)
    if size is not None and actual_size != size:
        raise VerificationError(f"{label}: size is {actual_size} bytes, expected {size}")
    digest, _ = sha256_file(path)
    if digest != str(sha256).lower():
        raise VerificationError(f"{label}: SHA-256 is {digest}, expected {sha256}")


def _locate_local(source: Source, local: Path, single: bool) -> Path:
    if local.is_file():
        if not single:
            raise AssetError("this item has several source files: give the folder that holds "
                             "them instead of one file")
        return local
    if local.is_dir():
        for cand in (local / source.dest / source.name, local / source.name):
            if cand.is_file():
                return cand
        raise AssetError(f"{source.name} was not found in {local}")
    raise AssetError(f"{local} does not exist")


# ----------------------------------------------------------------------------- unpacking

def _safe_parts(name: str) -> list[str]:
    """Path components of an archive entry name, or :class:`UnsafeArchive`."""
    raw = name.replace("\\", "/")
    if not raw or "\x00" in raw:
        raise UnsafeArchive(f"archive entry with an empty or invalid name: {name!r}")
    if raw.startswith("/") or re.match(r"^[A-Za-z]:", raw):
        raise UnsafeArchive(f"archive entry with an absolute path: {name!r}")
    parts = [p for p in raw.split("/") if p not in ("", ".")]
    for p in parts:
        if p == ".." or p.endswith((" ", ".")) and p.strip(". ") == "":
            raise UnsafeArchive(f"archive entry that leaves the target folder: {name!r}")
        if ":" in p or any(ord(ch) < 32 for ch in p):
            raise UnsafeArchive(f"archive entry with an invalid name: {name!r}")
    return parts


def _selected(parts: list[str], members: tuple[str, ...]) -> bool:
    if not members:
        return True
    path = "/".join(parts)
    for m in members:
        m = m.strip("/").removeprefix("./")
        if path == m or path.startswith(m + "/"):
            return True
    return False


class _Budget:
    """Counts what an item unpacks to and stops at the limits."""

    def __init__(self, limits: Limits):
        self.limits = limits
        self.entries = 0
        self.total = 0

    def entry(self) -> None:
        self.entries += 1
        if self.entries > self.limits.max_entries:
            raise UnsafeArchive(f"archive has more than {self.limits.max_entries} entries")

    def add(self, n: int, archive_size: int | None = None) -> None:
        self.total += n
        lim = self.limits
        if self.total > lim.max_total:
            raise UnsafeArchive(f"archive unpacks to more than {format_size(lim.max_total)}")
        if archive_size and self.total > lim.ratio_floor and \
                self.total > lim.max_ratio * archive_size:
            raise UnsafeArchive("archive unpacks to more than "
                                f"{lim.max_ratio:g} times its own size")


def _target(dest: Path, parts: list[str]) -> Path:
    target = dest.joinpath(*parts)
    root = os.path.realpath(dest)
    real = os.path.realpath(target)
    if real != root and not real.startswith(root + os.sep):
        raise UnsafeArchive(f"archive entry that leaves the target folder: {'/'.join(parts)!r}")
    return target


def _copy_stream(src, target: Path, declared: int, budget: _Budget, archive_size: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        raise UnsafeArchive(f"archive holds {target.name!r} twice")
    written = 0
    with open(target, "xb") as out:
        while True:
            block = src.read(CHUNK)
            if not block:
                break
            written += len(block)
            if written > declared:
                raise UnsafeArchive(f"archive entry {target.name!r} is larger than it declares")
            budget.add(len(block), archive_size)
            out.write(block)
    os.chmod(target, 0o644)


def _unzip(archive: Path, dest: Path, members, budget: _Budget) -> None:
    import stat
    import zipfile

    archive_size = os.path.getsize(archive)
    try:
        zf = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, OSError) as exc:
        raise UnsafeArchive(f"{archive.name} is not a readable zip archive ({exc})") from None
    with zf:
        infos = zf.infolist()
        plan = []
        declared = 0
        for info in infos:
            budget.entry()
            parts = _safe_parts(info.filename)
            mode = (info.external_attr >> 16) & 0xFFFF
            kind = stat.S_IFMT(mode)
            is_dir = info.is_dir() or kind == stat.S_IFDIR
            if kind not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise UnsafeArchive(f"archive entry {info.filename!r} is a link or special file")
            if info.flag_bits & 0x1:
                raise UnsafeArchive(f"archive entry {info.filename!r} is encrypted")
            if not _selected(parts, members) or not parts:
                continue
            if not is_dir:
                declared += info.file_size
            plan.append((info, parts, is_dir))
        check = _Budget(budget.limits)
        check.total = budget.total
        check.add(declared, archive_size)
        for info, parts, is_dir in plan:
            target = _target(dest, parts)
            if is_dir:
                target.mkdir(parents=True, exist_ok=True)
                continue
            try:
                with zf.open(info) as src:
                    _copy_stream(src, target, info.file_size, budget, archive_size)
            except (zipfile.BadZipFile, EOFError, NotImplementedError, RuntimeError) as exc:
                raise UnsafeArchive(f"{archive.name}: entry {info.filename!r} is damaged "
                                    f"({exc})") from None


def _untar(archive, dest: Path, members, budget: _Budget, *, archive_size: int,
           fileobj=None, skip_links: bool = False) -> None:
    import tarfile

    try:
        tf = tarfile.open(name=None if fileobj else archive, fileobj=fileobj, mode="r:*")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise UnsafeArchive(f"not a readable tar archive ({exc})") from None
    with tf:
        try:
            for member in tf:
                budget.entry()
                parts = _safe_parts(member.name)
                if not parts or not _selected(parts, members):
                    continue
                if member.isdir():
                    _target(dest, parts).mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isreg():
                    if skip_links and (member.issym() or member.islnk()):
                        continue
                    raise UnsafeArchive(f"archive entry {member.name!r} is a link or special file")
                src = tf.extractfile(member)
                if src is None:
                    raise UnsafeArchive(f"archive entry {member.name!r} cannot be read")
                with src:
                    _copy_stream(src, _target(dest, parts), member.size, budget, archive_size)
        except (tarfile.TarError, EOFError) as exc:
            raise UnsafeArchive(f"damaged tar archive ({exc})") from None


def _open_compressed(raw, name: str):
    """A decompressing reader for the ``data.tar*`` member of a Debian package."""
    if name.endswith(".xz") or name.endswith(".lzma"):
        import lzma

        return lzma.LZMAFile(raw)
    if name.endswith(".gz"):
        import gzip

        return gzip.GzipFile(fileobj=raw)
    if name.endswith(".bz2"):
        import bz2

        return bz2.BZ2File(raw)
    if name.endswith(".zst"):
        try:
            from compression import zstd  # Python 3.14+

            return zstd.ZstdFile(raw)
        except ImportError:
            pass
        try:
            import zstandard

            return zstandard.ZstdDecompressor().stream_reader(raw)
        except ImportError:
            raise AssetError("this package is compressed with zstd: it needs Python 3.14 or "
                             "the 'zstandard' package (pip install zstandard)") from None
    return raw


class _Window:
    """Read-only view of ``length`` bytes of a file starting at its current position."""

    def __init__(self, fh, length: int):
        self.fh, self.left = fh, length

    def read(self, n: int = -1) -> bytes:
        if self.left <= 0:
            return b""
        n = self.left if n is None or n < 0 else min(n, self.left)
        data = self.fh.read(n)
        self.left -= len(data)
        return data

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        pass


def _undeb(archive: Path, dest: Path, members, budget: _Budget) -> None:
    """Unpack the data archive of a Debian package (an ``ar`` file). Links in the package
    are left out; maintainer scripts are never read."""
    archive_size = os.path.getsize(archive)
    with open(archive, "rb") as fh:
        if fh.read(8) != b"!<arch>\n":
            raise UnsafeArchive(f"{archive.name} is not a Debian package")
        while True:
            header = fh.read(60)
            if len(header) < 60:
                raise UnsafeArchive(f"{archive.name} has no data archive")
            if header[58:60] != b"`\n":
                raise UnsafeArchive(f"{archive.name} is damaged")
            name = header[:16].decode("ascii", "replace").strip().rstrip("/")
            try:
                length = int(header[48:58].decode("ascii").strip())
            except ValueError:
                raise UnsafeArchive(f"{archive.name} is damaged") from None
            if name.startswith("data.tar"):
                reader = _open_compressed(_Window(fh, length), name)
                import io

                stream = reader if isinstance(reader, io.IOBase) else io.BufferedReader(reader)
                try:
                    import tarfile

                    with tarfile.open(fileobj=stream, mode="r|") as tf:
                        _untar_stream(tf, dest, members, budget, archive_size)
                except (tarfile.TarError, EOFError, OSError) as exc:
                    if isinstance(exc, OSError) and not isinstance(exc, (EOFError,)) \
                            and exc.errno is not None:
                        raise
                    raise UnsafeArchive(f"{archive.name}: damaged data archive ({exc})") from None
                return
            fh.seek(length + (length & 1), 1)


def _untar_stream(tf, dest: Path, members, budget: _Budget, archive_size: int) -> None:
    for member in tf:
        budget.entry()
        parts = _safe_parts(member.name)
        if not parts or not _selected(parts, members):
            continue
        if member.isdir():
            _target(dest, parts).mkdir(parents=True, exist_ok=True)
            continue
        if member.issym() or member.islnk():
            continue
        if not member.isreg():
            raise UnsafeArchive(f"package entry {member.name!r} is a special file")
        src = tf.extractfile(member)
        if src is None:
            raise UnsafeArchive(f"package entry {member.name!r} cannot be read")
        _copy_stream(src, _target(dest, parts), member.size, budget, archive_size)


def archive_kind(path) -> str | None:
    """``"zip"``, ``"tar"`` or ``"deb"`` by the file's signature, None for anything else."""
    path = Path(path)
    try:
        with open(path, "rb") as fh:
            head = fh.read(512)
    except OSError:
        return None
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return "zip"
    if head[:8] == b"!<arch>\n":
        return "deb" if b"debian-binary" in head[:80] else None
    if head[257:262] == b"ustar":
        return "tar"
    if head[:2] == b"\x1f\x8b" or head[:6] == b"\xfd7zXZ\x00" or head[:3] == b"BZh":
        lower = path.name.lower()
        if any(lower.endswith(e) for e in (".tar.gz", ".tgz", ".tar.xz", ".txz", ".tar.bz2",
                                           ".tbz2", ".tbz")):
            return "tar"
    return None


def extract(archive, dest, *, kind: str = "auto", members: Iterable[str] = (),
            limits: Limits = DEFAULT_LIMITS, _budget: _Budget | None = None) -> None:
    """Unpack ``archive`` into the folder ``dest``.

    Only plain files and folders are created, with fixed permissions. An entry with an
    absolute path, a drive letter, ``..``, a link or a special file, a repeated name, or an
    archive beyond ``limits`` raises :class:`UnsafeArchive` (links inside a Debian package
    are skipped instead). ``members`` restricts the entries to those below the given paths.
    """
    archive, dest = Path(archive), Path(dest)
    if kind == "auto":
        kind = archive_kind(archive) or ""
    if kind not in ARCHIVE_KINDS:
        raise AssetError(f"{archive.name} is not a zip, tar or Debian archive")
    dest.mkdir(parents=True, exist_ok=True)
    budget = _budget or _Budget(limits)
    members = tuple(members)
    if kind == "zip":
        _unzip(archive, dest, members, budget)
    elif kind == "tar":
        _untar(archive, dest, members, budget, archive_size=os.path.getsize(archive))
    else:
        _undeb(archive, dest, members, budget)


# ----------------------------------------------------------------------------- install / remove

def _rmtree(path) -> None:
    import shutil

    path = Path(path)
    if path.is_symlink() or path.is_file():
        try:
            path.unlink()
        except OSError:
            pass
        return
    shutil.rmtree(path, ignore_errors=True)


def staging_dir(base, item_id: str) -> Path:
    """A new empty folder inside ``base`` for building an install. Its name starts with a
    dot, so listings skip it."""
    import tempfile

    base = Path(base)
    base.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".install-{item_id}-", dir=base))


def clean_stale(base) -> None:
    """Delete staging folders an interrupted install left behind (older than a day)."""
    import time

    try:
        entries = list(Path(base).iterdir())
    except OSError:
        return
    for p in entries:
        if p.name.startswith((".install-", ".old-")) and p.is_dir():
            try:
                if time.time() - p.stat().st_mtime > 86400:
                    _rmtree(p)
            except OSError:
                pass


def put_in_place(stage, base, item_id: str) -> Path:
    """Move a finished staging folder to ``<base>/<id>``. An existing install is put aside
    first and restored when the move fails."""
    stage, final = Path(stage), Path(base) / item_id
    aside = None
    if final.exists() or final.is_symlink():
        aside = Path(base) / f".old-{item_id}-{os.getpid()}"
        _rmtree(aside)
        os.replace(final, aside)
    try:
        os.replace(stage, final)
    except OSError:
        if aside is not None:
            os.replace(aside, final)
        raise
    if aside is not None:
        _rmtree(aside)
    return final


def install_item(base, item_id: str, sources: Iterable[Source], manifest: dict,
                 manifest_name: str, *, local=None, progress: Progress | None = None,
                 limits: Limits = DEFAULT_LIMITS, check: Callable[[Path], None] | None = None,
                 workers: int = 1) -> Path:
    """Fetch, verify and unpack ``sources`` into ``<base>/<id>/files`` and write ``manifest``
    (plus ``installed_size`` and ``installed_at``) as ``<base>/<id>/<manifest_name>``.

    ``local`` is an archive file (for an item with one source) or a folder holding the
    source files, used instead of downloading; local files are verified like downloads.
    ``check(files_dir)`` may inspect the unpacked files and raise :class:`AssetError`.
    The item is built in a staging folder inside ``base`` and renamed into place at the end:
    a failure leaves no trace and an existing install of the same id stays untouched.
    Licence acceptance is the caller's job (:func:`require_license`) and comes first.
    """
    check_id(item_id)
    sources = list(sources)
    if not sources:
        raise AssetError(f"{item_id}: nothing to install")
    base = Path(base)
    local = Path(local).expanduser() if local is not None else None
    if local is not None and not local.exists():
        raise AssetError(f"{local} does not exist")
    clean_stale(base)
    stage = staging_dir(base, item_id)
    try:
        files = stage / "files"
        files.mkdir()
        inbox = stage / ".download"
        inbox.mkdir()
        budget = _Budget(limits)
        if workers > 1 and local is None and len(sources) > 1:
            _fetch_parallel(sources, files, inbox, budget, progress, workers)
        else:
            for n, src in enumerate(sources):
                _fetch_one(src, n, len(sources), files, inbox, budget, local, progress)
        _rmtree(inbox)
        if check is not None:
            check(files)
        data = dict(manifest)
        data.pop("installed_at", None)
        data["installed_size"] = tree_size(files)
        data["installed_at"] = now_stamp()
        write_manifest(stage, manifest_name, data)
        return put_in_place(stage, base, item_id)
    except BaseException:
        _rmtree(stage)
        raise


def _store(src: Source, path: Path, files: Path, budget: _Budget, *, move: bool) -> None:
    """Unpack a verified source file into ``files`` or store it there as it is."""
    kind = src.unpack
    if kind == "auto":
        kind = archive_kind(path) or "none"
    dest = _target(files, _safe_parts(src.dest)) if src.dest else files
    if kind == "none":
        budget.entry()
        budget.add(os.path.getsize(path))
        target = _target(dest, _safe_parts(src.name))
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            raise AssetError(f"two sources are stored as {src.name!r}")
        if move:
            os.replace(path, target)
        else:
            import shutil

            shutil.copyfile(path, target)
        os.chmod(target, 0o644)
        return
    extract(path, dest, kind=kind, members=src.members, _budget=budget)


def _fetch_one(src: Source, n: int, count: int, files: Path, inbox: Path, budget: _Budget,
               local: Path | None, progress: Progress | None) -> None:
    label = src.name if count == 1 else f"[{n + 1}/{count}] {src.name}"
    if local is not None:
        path = _locate_local(src, local, count == 1)
        verify_file(path, src.sha256, src.size, label=src.name)
        _store(src, path, files, budget, move=False)
        return
    path = inbox / f"{n:05d}"
    download(src.url, path, size=src.size, progress=progress, label=label)
    verify_file(path, src.sha256, src.size, label=src.name)
    _store(src, path, files, budget, move=True)
    if path.exists():
        path.unlink()


def _fetch_parallel(sources, files: Path, inbox: Path, budget: _Budget,
                    progress: Progress | None, workers: int) -> None:
    """Download many plain files with a few threads; storing stays on the calling thread."""
    from concurrent.futures import ThreadPoolExecutor

    total = sum(s.size for s in sources)
    done = 0

    def get(item):
        n, src = item
        path = inbox / f"{n:05d}"
        download(src.url, path, size=src.size, label=src.name)
        verify_file(path, src.sha256, src.size, label=src.name)
        return n, src, path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        try:
            for n, src, path in pool.map(get, enumerate(sources)):
                _store(src, path, files, budget, move=True)
                if path.exists():
                    path.unlink()
                done += src.size
                if progress:
                    progress(done, total, f"{n + 1}/{len(sources)} files")
        except BaseException:
            pool.shutdown(wait=True, cancel_futures=True)
            raise


def add_item(base, item_id: str, path, manifest: dict, manifest_name: str, *,
             copy: bool = False, limits: Limits = DEFAULT_LIMITS,
             check: Callable[[Path], None] | None = None) -> Path:
    """Register something the user already has as ``<base>/<id>``.

    ``path`` is a folder or a file. With ``copy=False`` a folder or plain file is registered
    in place: the manifest gets an absolute ``root`` and ``external: true`` and nothing is
    copied. With ``copy=True`` (always for an archive) the contents are copied or unpacked
    into ``<base>/<id>/files``. ``manifest`` must not contain ``root`` for the in-place
    case; a relative ``root`` is kept for copies.
    """
    check_id(item_id)
    base, path = Path(base), Path(path).expanduser()
    if not path.exists():
        raise AssetError(f"{path} does not exist")
    kind = archive_kind(path) if path.is_file() else None
    clean_stale(base)
    stage = staging_dir(base, item_id)
    try:
        data = dict(manifest)
        if kind or copy:
            files = stage / "files"
            files.mkdir()
            if kind:
                extract(path, files, kind=kind, limits=limits)
            elif path.is_dir():
                _copy_tree(path, files, _Budget(limits))
            else:
                import shutil

                shutil.copyfile(path, files / path.name)
            if check is not None:
                check(files)
            data.setdefault("root", "files")
            data["external"] = False
            data["installed_size"] = tree_size(files)
        else:
            root = path if path.is_dir() else path.parent
            if check is not None:
                check(root)
            data["root"] = str(root.resolve())
            data["external"] = True
            data["installed_size"] = tree_size(path)
        data["installed_at"] = now_stamp()
        write_manifest(stage, manifest_name, data)
        return put_in_place(stage, base, item_id)
    except BaseException:
        _rmtree(stage)
        raise


def _copy_tree(src: Path, dest: Path, budget: _Budget) -> None:
    """Copy the plain files of a folder; links and special files are left out."""
    import shutil

    for root, dirs, names in os.walk(src):
        dirs[:] = [d for d in dirs if not os.path.islink(os.path.join(root, d))]
        rel = os.path.relpath(root, src)
        out = dest if rel == "." else dest / rel
        out.mkdir(parents=True, exist_ok=True)
        for n in names:
            p = os.path.join(root, n)
            if os.path.islink(p) or not os.path.isfile(p):
                continue
            budget.entry()
            budget.add(os.path.getsize(p))
            shutil.copyfile(p, out / n)


def remove_item(base, item_id: str, manifest_name: str) -> bool:
    """Delete ``<base>/<id>``. Only a folder that holds ``manifest_name`` is touched, and for
    an item registered in place that folder holds nothing but the manifest, so the user's
    own files stay. Returns False when there is no such item."""
    try:
        check_id(item_id)
    except AssetError:
        return False
    folder = Path(base) / item_id
    if folder.is_symlink() or not (folder / manifest_name).is_file():
        return False
    aside = Path(base) / f".old-{item_id}-{os.getpid()}"
    _rmtree(aside)
    os.replace(folder, aside)
    _rmtree(aside)
    return True
