"""Tests for the model installer: ``synth2.assets``, the write side of ``synth2.neural.models``
and ``synth2 models``. Everything runs on small archives built in ``tmp_path``; the download
path is exercised against a local ``http.server``. No real network, no installed models."""
from __future__ import annotations

import ast
import hashlib
import io
import json
import os
import stat
import tarfile
import threading
import zipfile
from dataclasses import replace
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from synth2 import assets, cli, neural
from synth2.neural import models

ROOT = Path(__file__).resolve().parents[1]


# ================================================================== fixtures

def _zip(path: Path, entries: dict, *, links: dict | None = None) -> Path:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        for name, target in (links or {}).items():
            info = zipfile.ZipInfo(name)
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            zf.writestr(info, target)
    return path


def _digest(path: Path) -> tuple[str, int]:
    data = path.read_bytes()
    return hashlib.sha256(data).hexdigest(), len(data)


BANK_FILES = {
    "Bank_v1/dsconfig.yaml": "vocoder: testvoc\n",
    "Bank_v1/character.txt": "name=Test Bank\n",
    "Bank_v1/acoustic.onnx": b"\x00" * 2000,
    "Bank_v1/TERMS.txt": "be nice\n",
}
VOC_FILES = {"vocoder.yaml": "name: testvoc\nmodel: v.onnx\n", "v.onnx": b"\x01" * 1500,
             "NOTICE.txt": "notice\n"}


def _spec(archive: Path, **over) -> models.ModelSpec:
    digest, size = _digest(archive)
    base = dict(id="bank", kind="voicebank", name="Test Bank", model_version="1.0",
                root="files/Bank_v1", source="http://127.0.0.1:9/never.zip", sha256=digest,
                size=size, license="Test Licence", license_file="files/Bank_v1/TERMS.txt",
                license_url="https://example.invalid/terms", commercial_use=True,
                terms=("Be nice.",), credits=("Vocals: Test Bank.",),
                homepage="https://example.invalid", vocoder="voc", languages=("en",),
                default=True)
    base.update(over)
    return models.ModelSpec(**base)


@pytest.fixture
def env(tmp_path, monkeypatch):
    """A models directory, two fixture archives and a registry that points at them."""
    mdir = tmp_path / "models"
    monkeypatch.setenv("SYNTH2_MODELS", str(mdir))
    bank_zip = _zip(tmp_path / "bank.zip", BANK_FILES)
    voc_zip = _zip(tmp_path / "voc.oudep", VOC_FILES)
    registry = {
        "bank": _spec(bank_zip),
        "voc": _spec(voc_zip, id="voc", kind="vocoder", name="Test Vocoder", root="files",
                     license_file="files/NOTICE.txt", commercial_use=False, vocoder=None,
                     languages=(), credits=("Vocoder: Test.",)),
    }
    monkeypatch.setattr(models, "REGISTRY", registry)
    neural.refresh()
    yield type("Env", (), {"dir": mdir, "bank_zip": bank_zip, "voc_zip": voc_zip,
                           "registry": registry, "tmp": tmp_path})
    neural.refresh()


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A local HTTP server for ``tmp_path/www``; yields ``(base_url, folder)``."""
    www = tmp_path / "www"
    www.mkdir()
    for var in ("NO_PROXY", "no_proxy"):
        monkeypatch.setenv(var, "127.0.0.1,localhost")

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *a):
            pass

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(www)))
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}", www
    finally:
        httpd.shutdown()
        httpd.server_close()


def _leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


# ================================================================== registry and manifest

def test_real_registry_matches_the_reference_manifests():
    reg = models.REGISTRY
    assert set(reg) >= {"qixuan", "pc-nsf-hifigan"}
    q, v = reg["qixuan"], reg["pc-nsf-hifigan"]
    assert (q.size, q.sha256) == (420_707_627,
                                  "fe8ee7c95883d327a2b7facbbe832a7a53e1344918dc791ca83fc23b08034110")
    assert (v.size, v.sha256) == (52_760_610,
                                  "ba7d43142d41f6900c8264b5662ca7125a50feb8760bb8b9615c61a8f5e6902e")
    assert q.source.startswith("https://github.com/yqzhishen/qixuan-diffsinger/releases/download/")
    assert v.source.startswith("https://github.com/openvpi/vocoders/releases/download/")
    assert list(q.manifest()) == [
        "format", "version", "id", "kind", "name", "model_version", "engine", "root", "vocoder",
        "languages", "license", "license_file", "license_url", "commercial_use", "terms",
        "credits", "homepage", "source", "sha256", "default", "external"]
    assert list(v.manifest()) == [
        "format", "version", "id", "kind", "name", "model_version", "engine", "root", "license",
        "license_file", "license_url", "commercial_use", "terms", "credits", "homepage", "source",
        "sha256", "default", "external"]
    assert q.manifest()["commercial_use"] is True and v.manifest()["commercial_use"] is False
    assert q.vocoder == "pc-nsf-hifigan" and models.default_set() == ["qixuan", "pc-nsf-hifigan"]
    for spec in reg.values():
        assert spec.kind in models.KINDS and spec.terms and spec.credits
        assert spec.root == "files" or spec.root.startswith("files/")
        assert spec.license_file.startswith("files/")


def test_manifest_text_format():
    text = assets.dump_json({"a": ["zh", "ja", "en"], "name": "绮萱", "long": ["x" * 120],
                             "n": 1, "t": True, "none": None, "empty": [], "d": {"k": [1, 2]}})
    assert '"a": ["zh", "ja", "en"],' in text
    assert '"name": "绮萱"' in text
    assert '"long": [\n    "' + "x" * 120 + '"\n  ]' in text
    assert text.endswith("}\n")
    assert json.loads(text)["d"] == {"k": [1, 2]}


def test_models_module_imports_stay_cheap():
    """The read side is imported by everything: no network or archive module at import time."""
    for name in ("synth2/neural/models.py", "synth2/assets.py", "synth2/sounds/__init__.py"):
        tree = ast.parse((ROOT / name).read_text(encoding="utf-8"))
        top = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                top |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                top.add((node.module or "").split(".")[0])
        assert not top & {"urllib", "ssl", "zipfile", "tarfile", "hashlib", "shutil", "tempfile",
                          "http", "lzma", "numpy"}, name


# ================================================================== install

def test_install_needs_the_licence_before_anything_else(env, monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network or disk touched before the licence was accepted")

    monkeypatch.setattr(assets, "download", boom)
    monkeypatch.setattr(assets, "install_item", boom)
    with pytest.raises(models.LicenseNotAccepted) as e:
        models.install("bank")
    err = e.value
    assert (err.name, err.license, err.url) == ("Test Bank", "Test Licence",
                                                "https://example.invalid/terms")
    assert err.terms == ["Be nice."] and err.credits == ["Vocals: Test Bank."]
    assert "Test Licence" in err.text() and "Be nice." in err.text()
    assert "--accept-license" in str(err)
    assert not env.dir.exists()


def test_install_from_a_local_archive(env):
    got = models.install("bank", accept_license=True, source=env.bank_zip)
    assert [m.id for m in got] == ["bank"]
    m = got[0]
    assert m.kind == "voicebank" and m.root == env.dir / "bank" / "files" / "Bank_v1"
    assert (m.root / "acoustic.onnx").read_bytes() == b"\x00" * 2000
    data = json.loads((env.dir / "bank" / models.MANIFEST).read_text(encoding="utf-8"))
    want = env.registry["bank"].manifest()
    assert {k: v for k, v in data.items() if k != "installed_at"} == want
    assert list(data)[-1] == "installed_at" and data["installed_at"].endswith("Z")
    assert _leftovers(env.dir) == ["bank"]
    assert models.find("voicebank").id == "bank"
    assert m.license_file.is_file() and m.commercial_use is True


def test_install_downloads_the_default_set(env, server, monkeypatch):
    base, www = server
    (www / "bank.zip").write_bytes(env.bank_zip.read_bytes())
    (www / "voc.oudep").write_bytes(env.voc_zip.read_bytes())
    reg = {"bank": replace(env.registry["bank"], source=f"{base}/bank.zip"),
           "voc": replace(env.registry["voc"], source=f"{base}/voc.oudep")}
    monkeypatch.setattr(models, "REGISTRY", reg)
    seen = []
    got = models.install(accept_license=True, progress=lambda d, t, label: seen.append((d, t)))
    assert [m.id for m in got] == ["bank", "voc"]
    assert seen and seen[-1][0] == seen[-1][1] == reg["voc"].size
    assert models.vocoder_for(models.find("voicebank", "bank")).id == "voc"
    assert _leftovers(env.dir) == ["bank", "voc"]
    ok, reason = neural.available()
    assert "voicebank" not in reason  # the freshly installed bank is seen without a restart
    # an installed model is returned as it is: no licence gate, no download
    monkeypatch.setattr(assets, "download", lambda *a, **k: 1 / 0)
    assert [m.id for m in models.install()] == ["bank", "voc"]


def test_download_errors_are_reported_and_leave_nothing(env, server, monkeypatch):
    base, www = server
    monkeypatch.setattr("time.sleep", lambda s: None)
    reg = {"bank": replace(env.registry["bank"], source=f"{base}/missing.zip")}
    monkeypatch.setattr(models, "REGISTRY", reg)
    with pytest.raises(assets.DownloadError, match="404"):
        models.install("bank", accept_license=True)
    assert _leftovers(env.dir) == []
    # a server that sends more than the pinned size is cut off
    (www / "bank.zip").write_bytes(env.bank_zip.read_bytes() + b"extra")
    reg["bank"] = replace(env.registry["bank"], source=f"{base}/bank.zip")
    with pytest.raises(assets.VerificationError, match="more than the expected"):
        models.install("bank", accept_license=True)
    # and one that sends less is an incomplete download
    (www / "bank.zip").write_bytes(env.bank_zip.read_bytes()[:-10])
    with pytest.raises(assets.DownloadError):
        models.install("bank", accept_license=True)
    assert _leftovers(env.dir) == []
    with pytest.raises(assets.DownloadError, match="scheme"):
        assets.download("file:///etc/passwd", env.tmp / "x")


def test_checksum_and_size_mismatch(env, monkeypatch):
    data = bytearray(env.bank_zip.read_bytes())
    data[len(data) // 2] ^= 0xFF
    tampered = env.tmp / "tampered.zip"
    tampered.write_bytes(bytes(data))
    with pytest.raises(assets.VerificationError, match="SHA-256"):
        models.install("bank", accept_license=True, source=tampered)
    short = env.tmp / "short.zip"
    short.write_bytes(env.bank_zip.read_bytes()[:-1])
    with pytest.raises(assets.VerificationError, match="size"):
        models.install("bank", accept_license=True, source=short)
    assert _leftovers(env.dir) == []
    with pytest.raises(assets.AssetError, match="does not exist"):
        models.install("bank", accept_license=True, source=env.tmp / "nope.zip")
    with pytest.raises(assets.AssetError, match="unknown model"):
        models.install("nope", accept_license=True)


def _install_hostile(env, monkeypatch, archive: Path):
    reg = {"bank": _spec(archive)}
    monkeypatch.setattr(models, "REGISTRY", reg)
    return models.install("bank", accept_license=True, source=archive)


@pytest.mark.parametrize("name", ["../escape.txt", "Bank_v1/../../escape.txt", "/abs/escape.txt",
                                  "C:/escape.txt", "C:\\escape.txt", "..\\escape.txt",
                                  "Bank_v1/..\\..\\escape.txt"])
def test_path_traversal_is_rejected(env, monkeypatch, name):
    bad = _zip(env.tmp / "bad.zip", {**BANK_FILES, name: "x"})
    with pytest.raises(assets.UnsafeArchive):
        _install_hostile(env, monkeypatch, bad)
    assert _leftovers(env.dir) == []
    assert not (env.tmp / "escape.txt").exists() and not (env.dir.parent / "escape.txt").exists()


def test_symlink_entries_are_rejected(env, monkeypatch):
    bad = _zip(env.tmp / "bad.zip", BANK_FILES, links={"Bank_v1/link": "/etc/passwd"})
    with pytest.raises(assets.UnsafeArchive, match="link"):
        _install_hostile(env, monkeypatch, bad)
    assert _leftovers(env.dir) == []


def test_zip_bombs_are_rejected(env, monkeypatch):
    bomb = env.tmp / "bomb.zip"
    with zipfile.ZipFile(bomb, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in BANK_FILES.items():
            zf.writestr(name, data)
        zf.writestr("Bank_v1/zeros.bin", b"\x00" * (40 << 20))
    assert bomb.stat().st_size < 200_000
    with pytest.raises(assets.UnsafeArchive, match="times its own size"):
        _install_hostile(env, monkeypatch, bomb)
    assert _leftovers(env.dir) == []
    # entry count and total size limits
    many = _zip(env.tmp / "many.zip", {f"f{i}.txt": "x" for i in range(30)})
    with pytest.raises(assets.UnsafeArchive, match="entries"):
        assets.extract(many, env.tmp / "out1", limits=assets.Limits(max_entries=10))
    big = _zip(env.tmp / "big.zip", {"a.bin": os.urandom(5000), "b.bin": os.urandom(5000)})
    with pytest.raises(assets.UnsafeArchive, match="unpacks to more than"):
        assets.extract(big, env.tmp / "out2", limits=assets.Limits(max_total=6000))
    # a repeated name must not overwrite the first copy
    twice = env.tmp / "twice.zip"
    with zipfile.ZipFile(twice, "w") as zf:
        zf.writestr("a.txt", "one")
        with pytest.warns(UserWarning):
            zf.writestr("a.txt", "two")
    with pytest.raises(assets.UnsafeArchive, match="twice"):
        assets.extract(twice, env.tmp / "out3")


def test_archive_without_the_model_is_rejected(env, monkeypatch):
    wrong = _zip(env.tmp / "wrong.zip", {"readme.txt": "no model here"})
    with pytest.raises(assets.AssetError, match="dsconfig.yaml"):
        _install_hostile(env, monkeypatch, wrong)
    assert _leftovers(env.dir) == []


def test_failed_install_keeps_the_existing_one_and_force_replaces_it(env, monkeypatch):
    first = models.install("bank", accept_license=True, source=env.bank_zip)[0]
    marker = first.root / "user-note.txt"
    marker.write_text("mine")
    # without force nothing is touched
    assert models.install("bank", source=env.bank_zip)[0].path == first.path
    assert marker.exists()
    # a forced reinstall that fails leaves the old install in place
    bad = env.tmp / "bad.zip"
    bad.write_bytes(b"not a zip at all")
    with pytest.raises(assets.AssetError):
        models.install("bank", accept_license=True, source=bad, force=True)
    assert marker.exists() and _leftovers(env.dir) == ["bank"]
    # force still asks for the licence
    with pytest.raises(models.LicenseNotAccepted):
        models.install("bank", source=env.bank_zip, force=True)
    # a forced reinstall that works replaces the folder
    models.install("bank", accept_license=True, source=env.bank_zip, force=True)
    assert not marker.exists() and _leftovers(env.dir) == ["bank"]
    assert models.find("voicebank", "bank") is not None


def test_install_without_id_recognises_a_registry_archive(env):
    got = models.install(accept_license=True, source=env.voc_zip)
    assert [m.id for m in got] == ["voc"]
    other = _zip(env.tmp / "other.zip", {"x/dsconfig.yaml": "a: 1\n"})
    with pytest.raises(assets.AssetError, match="add_local"):
        models.install(accept_license=True, source=other)


# ================================================================== tar and deb archives

def _tar(path: Path, entries: dict, *, links: dict | None = None, mode="w:gz") -> Path:
    with tarfile.open(path, mode) as tf:
        for name, data in entries.items():
            info = tarfile.TarInfo(name)
            data = data.encode() if isinstance(data, str) else data
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)
    return path


def _deb(path: Path, data_tar: Path, member="data.tar.xz") -> Path:
    def entry(name: str, payload: bytes) -> bytes:
        head = f"{name:<16}{0:<12}{0:<6}{0:<6}{'100644':<8}{len(payload):<10}`\n".encode()
        return head + payload + (b"\n" if len(payload) % 2 else b"")

    path.write_bytes(b"!<arch>\n" + entry("debian-binary", b"2.0\n")
                     + entry("control.tar.xz", b"ignored")
                     + entry(member, data_tar.read_bytes()))
    return path


def test_tar_extraction_and_its_refusals(tmp_path):
    good = _tar(tmp_path / "good.tar.gz", {"./set/a.wav": b"RIFF", "set/sub/b.txt": "hi"})
    assert assets.archive_kind(good) == "tar"
    assets.extract(good, tmp_path / "out")
    assert (tmp_path / "out/set/a.wav").read_bytes() == b"RIFF"
    assert (tmp_path / "out/set/sub/b.txt").read_text() == "hi"
    assert stat.S_IMODE((tmp_path / "out/set/a.wav").stat().st_mode) == 0o644
    for n, (entries, links) in enumerate([({"../x": "1"}, None), ({"/abs/x": "1"}, None),
                                          ({"ok": "1"}, {"lnk": "/etc/passwd"})]):
        bad = _tar(tmp_path / f"bad{n}.tar.gz", entries, links=links)
        with pytest.raises(assets.UnsafeArchive):
            assets.extract(bad, tmp_path / f"bad{n}")
    hard = tmp_path / "hard.tar"
    with tarfile.open(hard, "w") as tf:
        info = tarfile.TarInfo("h")
        info.type = tarfile.LNKTYPE
        info.linkname = "/etc/passwd"
        tf.addfile(info)
        fifo = tarfile.TarInfo("fifo")
        fifo.type = tarfile.FIFOTYPE
        tf.addfile(fifo)
    with pytest.raises(assets.UnsafeArchive, match="link or special"):
        assets.extract(hard, tmp_path / "hard")
    # members restricts what is unpacked
    assets.extract(good, tmp_path / "part", members=("set/sub",))
    assert not (tmp_path / "part/set/a.wav").exists() and (tmp_path / "part/set/sub/b.txt").exists()


def test_debian_package_extraction(tmp_path):
    data = _tar(tmp_path / "data.tar.xz", {"./usr/share/sounds/sf2/x.sf2": b"RIFFsfbk",
                                           "./usr/share/doc/x/copyright": "MIT"},
                links={"./usr/share/sounds/sf2/default.sf2": "x.sf2"}, mode="w:xz")
    deb = _deb(tmp_path / "x_1.0_all.deb", data)
    assert assets.archive_kind(deb) == "deb"
    assets.extract(deb, tmp_path / "out", members=("usr/share",))
    assert (tmp_path / "out/usr/share/sounds/sf2/x.sf2").read_bytes() == b"RIFFsfbk"
    assert (tmp_path / "out/usr/share/doc/x/copyright").read_text() == "MIT"
    assert not (tmp_path / "out/usr/share/sounds/sf2/default.sf2").exists()  # links are skipped
    escape = _deb(tmp_path / "bad.deb", _tar(tmp_path / "bad.tar.xz", {"../../x": "1"},
                                              mode="w:xz"))
    with pytest.raises(assets.UnsafeArchive):
        assets.extract(escape, tmp_path / "bad")
    assert assets.archive_kind(tmp_path / "data.tar.xz") == "tar"
    plain = tmp_path / "plain.bin"
    plain.write_bytes(b"\x00" * 600)
    assert assets.archive_kind(plain) is None
    with pytest.raises(assets.AssetError, match="not a zip"):
        assets.extract(plain, tmp_path / "plain")


# ================================================================== add_local and remove

def _bank_folder(tmp: Path, name="MyBank") -> Path:
    folder = tmp / "banks" / name
    folder.mkdir(parents=True)
    (folder / "dsconfig.yaml").write_text("a: 1\n")
    (folder / "character.txt").write_text("name=My Own Bank\n")
    (folder / "acoustic.onnx").write_bytes(b"\x02" * 100)
    return folder


def test_add_local_registers_a_folder_in_place_and_remove_keeps_the_files(env):
    folder = _bank_folder(env.tmp)
    m = models.add_local(folder)
    assert (m.id, m.kind, m.name) == ("mybank", "voicebank", "My Own Bank")
    assert m.external and m.root == folder.resolve() and m.commercial_use is None
    data = json.loads((env.dir / "mybank" / models.MANIFEST).read_text(encoding="utf-8"))
    assert Path(data["root"]).is_absolute() and data["external"] is True
    assert "commercial_use" not in data
    assert "responsible" in data["terms"][0]
    assert _leftovers(env.dir / "mybank") == [models.MANIFEST]
    assert [x.id for x in models.installed("voicebank")] == ["mybank"]
    with pytest.raises(assets.AssetError, match="already installed"):
        models.add_local(folder)
    assert models.remove("mybank") is True
    assert (folder / "acoustic.onnx").exists() and not (env.dir / "mybank").exists()
    assert models.remove("mybank") is False


def test_add_local_from_a_parent_folder_a_vocoder_and_a_zip(env):
    folder = _bank_folder(env.tmp)
    m = models.add_local(folder.parent, "viaparent", name="Named")
    assert m.root == folder.resolve() and m.name == "Named"
    voc = env.tmp / "myvoc"
    voc.mkdir()
    (voc / "vocoder.yaml").write_text("name: my_vocoder\n")
    v = models.add_local(voc, copy=True)
    assert (v.kind, v.name, v.external) == ("vocoder", "my_vocoder", False)
    assert v.root == env.dir / "myvoc" / "files" and (v.root / "vocoder.yaml").is_file()
    z = models.add_local(env.bank_zip, "zipped")
    assert z.kind == "voicebank" and not z.external
    assert z.root == env.dir / "zipped" / "files" / "Bank_v1" and z.name == "Test Bank"
    assert models.remove("zipped") and not (env.dir / "zipped").exists()


def test_add_local_refusals(env):
    with pytest.raises(assets.AssetError, match="registry"):
        models.add_local(_bank_folder(env.tmp), "bank")
    empty = env.tmp / "empty"
    empty.mkdir()
    with pytest.raises(assets.AssetError, match="dsconfig.yaml or vocoder.yaml"):
        models.add_local(empty)
    with pytest.raises(assets.AssetError, match="vocoder.yaml"):
        models.add_local(_bank_folder(env.tmp, "Other"), kind="vocoder")
    with pytest.raises(assets.AssetError, match="invalid id"):
        models.add_local(_bank_folder(env.tmp, "Third"), "../evil")
    with pytest.raises(assets.AssetError, match="does not exist"):
        models.add_local(env.tmp / "nowhere")
    assert _leftovers(env.dir) == []


def test_remove_only_touches_model_folders(env):
    env.dir.mkdir(parents=True)
    (env.dir / "notamodel").mkdir()
    (env.dir / "notamodel" / "keep.txt").write_text("x")
    assert models.remove("notamodel") is False and (env.dir / "notamodel" / "keep.txt").exists()
    assert models.remove("../outside") is False and models.remove("") is False
    models.install("bank", accept_license=True, source=env.bank_zip)
    assert models.remove("bank") is True and _leftovers(env.dir) == ["notamodel"]


# ================================================================== command line

def _run(argv) -> int:
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


def test_cli_list_install_remove_path(env, capsys):
    assert _run(["models", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(env.dir)
    assert _run(["models", "list"]) == 0
    out = capsys.readouterr().out
    assert "no models are installed" in out and "bank" in out and "not installed" in out
    assert "Test Licence" in out and "commercial use stated: yes" in out
    assert "commercial use stated: no" in out and "Vocals: Test Bank." in out

    # not a terminal and no --accept-license: the terms are printed, nothing is installed
    assert _run(["models", "install", "bank", "--from-file", str(env.bank_zip)]) == 2
    err = capsys.readouterr().err
    assert "Test Licence" in err and "Be nice." in err and "--accept-license" in err
    assert _leftovers(env.dir) == []

    assert _run(["models", "install", "bank", "--from-file", str(env.bank_zip),
                 "--accept-license", "-q"]) == 0
    assert "installed bank (voicebank)" in capsys.readouterr().out
    assert _run(["models", "list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [m["id"] for m in data["installed"]] == ["bank"]
    assert [m["id"] for m in data["available"]] == ["voc"]
    assert data["installed"][0]["commercial_use"] is True

    assert _run(["models", "install", "nope", "--accept-license"]) == 2
    assert "unknown model" in capsys.readouterr().err
    assert _run(["models", "remove", "bank"]) == 0
    assert _run(["models", "remove", "bank"]) == 2
    assert "no model" in capsys.readouterr().err


def test_cli_asks_on_a_terminal(env, monkeypatch, capsys):
    answers = iter(["n\n", "y\n"])

    class Tty(io.StringIO):
        def isatty(self):
            return True

        def readline(self):
            return next(answers)

    monkeypatch.setattr("sys.stdin", Tty())
    argv = ["models", "install", "voc", "--from-file", str(env.voc_zip), "-q"]
    assert _run(argv) == 2
    assert "not accepted" in capsys.readouterr().err and _leftovers(env.dir) == []
    assert _run(argv) == 0
    assert _leftovers(env.dir) == ["voc"]


def test_cli_add_and_install_of_an_unknown_archive(env, capsys):
    folder = _bank_folder(env.tmp)
    assert _run(["models", "add", str(folder), "--id", "mine", "--name", "Mine"]) == 0
    out = capsys.readouterr().out
    assert "added mine (voicebank)" in out and "responsible" in out
    assert models.find("voicebank", "mine").external
    # an archive that is no registry model is unpacked as the user's own model
    other = _zip(env.tmp / "Other_Bank.zip", {"Other/dsconfig.yaml": "a: 1\n"})
    assert _run(["models", "install", "--from-file", str(other)]) == 0
    assert "responsible" in capsys.readouterr().out
    got = models.find("voicebank", "other_bank")
    assert got is not None and not got.external and got.commercial_use is None
    assert _run(["models", "add", str(env.tmp / "missing")]) == 2
    # a registry archive given without an id is recognised by its hash
    assert _run(["models", "install", "--from-file", str(env.voc_zip), "--accept-license",
                 "-q"]) == 0
    assert models.find("vocoder", "voc") is not None


def test_install_hint_names_a_command_that_exists():
    hint = neural.INSTALL_HINT
    assert "synth2 models install" in hint
    parser = cli.build_parser()
    args = parser.parse_args(["models", "install"])
    assert args.func.__module__ == "synth2.neural.cli" and args.ids == []
    args = parser.parse_args(["sounds", "list"])
    assert args.func.__module__ == "synth2.sounds.cli"
