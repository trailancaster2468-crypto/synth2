"""Tests for sound sets: the read side of ``synth2.sounds``, its installer and registry, and
``synth2 sounds``. Archives are built in ``tmp_path``; nothing is downloaded and no installed
set is needed (the last test looks at installed sets and skips without them)."""
from __future__ import annotations

import hashlib
import io
import json
import re
import stat
import struct
import sys
import tarfile
import types
import zipfile
from pathlib import Path

import pytest

from synth2 import assets, cli, sounds
from synth2.assets import Source
from synth2.sounds import installer, lists, registry


# ================================================================== fixtures

def _wav(frames: int = 64) -> bytes:
    data = struct.pack(f"<{frames}h", *([0, 1000, 0, -1000] * (frames // 4)))
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 44100, 88200, 2, 16) + b"data"
            + struct.pack("<I", len(data)) + data)


KIT_FILES = {
    "TestKit/kit.sfz": "<region> sample=samples/kick.wav key=36\n",
    "TestKit/samples/kick.wav": _wav(),
    "TestKit/samples/snare.wav": _wav(128),
    "TestKit/LICENSE.txt": "CC0\n",
}


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


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _spec(archive: Path, **over) -> installer.SoundSpec:
    data = archive.read_bytes()
    base = dict(id="testkit", kind="sfz", name="Test Kit", version="1.0", publisher="Tester",
                homepage="https://example.invalid/kit", license="CC0 1.0",
                license_source="LICENSE.txt in the archive", commercial_use=True,
                commercial_use_statement="Use it for anything.",
                credits=("Drums: Test Kit by Tester.",), files=("kit.sfz",),
                size=len(data), root="files/TestKit", license_file="files/TestKit/LICENSE.txt",
                license_url="https://example.invalid/cc0", tags=("drums", "acoustic"),
                presets=({"name": "Kit", "file": "kit.sfz", "category": "drums"},),
                sources=(Source(url="http://127.0.0.1:9/never.zip", sha256=_sha(data),
                                size=len(data)),))
    base.update(over)
    return installer.SoundSpec(**base)


@pytest.fixture
def env(tmp_path, monkeypatch):
    sdir = tmp_path / "sounds"
    monkeypatch.setenv("SYNTH2_SOUNDS", str(sdir))
    kit_zip = _zip(tmp_path / "kit.zip", KIT_FILES)
    reg = {"testkit": _spec(kit_zip)}
    monkeypatch.setattr(registry, "REGISTRY", reg)
    return types.SimpleNamespace(dir=sdir, kit_zip=kit_zip, registry=reg, tmp=tmp_path)


def _leftovers(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir()) if folder.exists() else []


def _run(argv) -> int:
    try:
        return cli.main(argv)
    except SystemExit as exc:
        return int(exc.code or 0)


# ================================================================== where sets live

def test_sounds_dir(monkeypatch, tmp_path):
    monkeypatch.setenv("SYNTH2_SOUNDS", str(tmp_path / "x"))
    assert sounds.sounds_dir() == tmp_path / "x"
    monkeypatch.delenv("SYNTH2_SOUNDS")
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("SYNTH2_MODELS", raising=False)
    if sys.platform != "win32":
        from synth2.neural import models

        assert sounds.sounds_dir() == tmp_path / "data" / "synth2" / "sounds"
        assert sounds.sounds_dir().parent == models.models_dir().parent
    assert sounds.installed() == [] and sounds.find("anything") is None


def test_read_side_skips_broken_folders(env):
    env.dir.mkdir()
    (env.dir / "junk").mkdir()
    (env.dir / "badjson").mkdir()
    (env.dir / "badjson" / sounds.MANIFEST).write_text("{nope")
    (env.dir / "wrongkind").mkdir()
    (env.dir / "wrongkind" / sounds.MANIFEST).write_text(json.dumps(
        {"format": "synth2-sound", "id": "wrongkind", "kind": "video"}))
    (env.dir / "noroot").mkdir()
    (env.dir / "noroot" / sounds.MANIFEST).write_text(json.dumps(
        {"format": "synth2-sound", "id": "noroot", "kind": "sfz"}))
    good = env.dir / "good"
    (good / "files").mkdir(parents=True)
    (good / "files" / "a.sf2").write_bytes(b"RIFF")
    (good / sounds.MANIFEST).write_text(json.dumps(
        {"format": "synth2-sound", "version": 1, "id": "good", "kind": "soundfont",
         "files": ["a.sf2", "gone.sf2"], "tags": ["gm"]}))
    assert [s.id for s in sounds.installed()] == ["good"]
    s = sounds.find("good")
    assert s.files == [good / "files" / "a.sf2"] and s.gm and s.commercial_use is None
    assert s.name == "good" and s.credits == [] and s.presets == []
    assert sounds.installed("drumkit") == [] and [x.id for x in sounds.installed("soundfont")] == ["good"]
    for bad in ("", "../good", "junk", "nope", ".hidden", None):
        assert sounds.find(bad) is None


# ================================================================== install

def test_install_needs_the_licence_first(env, monkeypatch):
    monkeypatch.setattr(assets, "download", lambda *a, **k: 1 / 0)
    monkeypatch.setattr(assets, "install_item", lambda *a, **k: 1 / 0)
    with pytest.raises(sounds.installer.LicenseNotAccepted) as e:
        sounds.install("testkit")
    err = e.value
    assert err.name == "Test Kit" and err.license == "CC0 1.0"
    assert err.statement == "Use it for anything." and err.commercial_use is True
    assert "Use it for anything." in err.text() and "Drums: Test Kit by Tester." in err.text()
    assert not env.dir.exists()
    with pytest.raises(assets.AssetError, match="unknown sound set"):
        sounds.install("nope", accept_license=True)


def test_install_from_a_local_archive_and_the_manifest(env):
    s = sounds.install("testkit", accept_license=True, source=env.kit_zip)
    assert (s.id, s.kind, s.name, s.version) == ("testkit", "sfz", "Test Kit", "1.0")
    assert s.path == env.dir / "testkit" and s.root == s.path / "files" / "TestKit"
    assert s.files == [s.root / "kit.sfz"]
    assert s.file("samples/kick.wav").read_bytes() == KIT_FILES["TestKit/samples/kick.wav"]
    assert s.presets == [{"name": "Kit", "file": "kit.sfz", "category": "drums"}]
    assert s.tags == ["drums", "acoustic"] and not s.gm and not s.external
    assert s.license == "CC0 1.0" and s.license_file.read_text() == "CC0\n"
    assert s.commercial_use is True and s.commercial_use_statement == "Use it for anything."
    assert s.credits == ["Drums: Test Kit by Tester."]
    assert s.size == sum(len(v) for v in KIT_FILES.values())
    data = json.loads((s.path / sounds.MANIFEST).read_text(encoding="utf-8"))
    for key in ("format", "version", "id", "kind", "name", "sound_version", "publisher", "root",
                "files", "presets", "tags", "license", "license_file", "license_url",
                "license_source", "commercial_use", "commercial_use_statement", "share_alike",
                "terms", "credits", "homepage", "source", "size", "sha256", "external",
                "installed_size", "installed_at"):
        assert key in data, key
    assert data["format"] == "synth2-sound" and data["sha256"] == env.registry["testkit"].sources[0].sha256
    assert data["size"] == env.kit_zip.stat().st_size and data["external"] is False
    assert _leftovers(env.dir) == ["testkit"]
    assert sounds.find("testkit") == s and sounds.installed("sfz") == [s]
    assert sounds.credits() == ["Drums: Test Kit by Tester."]
    assert sounds.credits(["testkit", "missing"]) == ["Drums: Test Kit by Tester."]
    # installed already: returned as it is, without the licence gate
    assert sounds.install("testkit") == s
    assert stat.S_IMODE(s.path.stat().st_mode) & 0o055 == 0o055


def test_tampered_archive_is_rejected(env):
    data = bytearray(env.kit_zip.read_bytes())
    data[len(data) // 2] ^= 0x01
    bad = env.tmp / "tampered.zip"
    bad.write_bytes(bytes(data))
    with pytest.raises(assets.VerificationError, match="SHA-256"):
        sounds.install("testkit", accept_license=True, source=bad)
    longer = env.tmp / "longer.zip"
    longer.write_bytes(env.kit_zip.read_bytes() + b"\x00")
    with pytest.raises(assets.VerificationError, match="size"):
        sounds.install("testkit", accept_license=True, source=longer)
    assert _leftovers(env.dir) == [] and sounds.installed() == []


@pytest.mark.parametrize("hostile", ["escape", "absolute", "link", "oversize", "missing"])
def test_hostile_archives_are_rejected(env, monkeypatch, hostile):
    entries, links = dict(KIT_FILES), None
    if hostile == "escape":
        entries["TestKit/../../escape.txt"] = "x"
    elif hostile == "absolute":
        entries["/tmp/escape.txt"] = "x"
    elif hostile == "link":
        links = {"TestKit/samples/link.wav": "../../../../etc/passwd"}
    elif hostile == "oversize":
        entries["TestKit/samples/zeros.wav"] = b"\x00" * (40 << 20)
    else:
        del entries["TestKit/kit.sfz"]
    bad = _zip(env.tmp / "bad.zip", entries, links=links)
    monkeypatch.setattr(registry, "REGISTRY", {"testkit": _spec(bad)})
    expected = assets.AssetError if hostile == "missing" else assets.UnsafeArchive
    with pytest.raises(expected):
        sounds.install("testkit", accept_license=True, source=bad)
    assert _leftovers(env.dir) == []
    assert not (env.tmp / "escape.txt").exists()


def test_failed_reinstall_keeps_the_installed_set(env):
    first = sounds.install("testkit", accept_license=True, source=env.kit_zip)
    note = first.root / "note.txt"
    note.write_text("mine")
    junk = env.tmp / "junk.zip"
    junk.write_bytes(b"junk")
    with pytest.raises(assets.AssetError):
        sounds.install("testkit", accept_license=True, source=junk, force=True)
    assert note.exists() and _leftovers(env.dir) == ["testkit"]
    sounds.install("testkit", accept_license=True, source=env.kit_zip, force=True)
    assert not note.exists()


def test_set_made_of_single_files_and_a_tar(env, monkeypatch):
    """Plain files are stored under their own names; several sources make one set."""
    src = env.tmp / "src"
    (src / "docs").mkdir(parents=True)
    sf2 = b"RIFF\x00\x00\x00\x00sfbk" + b"\x00" * 100
    (src / "Font.sf2").write_bytes(sf2)
    (src / "docs" / "LICENSE.txt").write_text("free\n")
    tar_path = src / "extra.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tf:
        info = tarfile.TarInfo("extra/readme.txt")
        info.size = 3
        tf.addfile(info, io.BytesIO(b"hey"))
    spec = _spec(env.kit_zip, id="font", kind="soundfont", root="files", files=("Font.sf2",),
                 license_file="files/docs/LICENSE.txt", presets=(), size=0, sources=(
        Source(url="http://127.0.0.1:9/Font.sf2", sha256=_sha(sf2), size=len(sf2)),
        Source(url="http://127.0.0.1:9/x/LICENSE.txt", sha256=_sha(b"free\n"), size=5,
               dest="docs"),
        Source(url="http://127.0.0.1:9/extra.tar.gz", sha256=_sha(tar_path.read_bytes()),
               size=tar_path.stat().st_size),
    ))
    monkeypatch.setattr(registry, "REGISTRY", {"font": spec})
    with pytest.raises(assets.AssetError, match="several source files"):
        sounds.install("font", accept_license=True, source=src / "Font.sf2")
    s = sounds.install("font", accept_license=True, source=src)
    assert s.kind == "soundfont" and s.files == [s.root / "Font.sf2"]
    assert (s.root / "docs" / "LICENSE.txt").read_text() == "free\n"
    assert (s.root / "extra" / "readme.txt").read_text() == "hey"
    data = json.loads((s.path / sounds.MANIFEST).read_text())
    assert data["sha256"] == spec.digest() and re.fullmatch(r"[0-9a-f]{64}", data["sha256"])


def test_file_list_sets(env, monkeypatch):
    """A set fetched file by file: paths, sizes and hashes come from a list module."""
    kick, snare = _wav(), _wav(256)
    mod = types.ModuleType("synth2.sounds.lists.fake_kit")
    mod.BASE = "https://example.invalid/repo/abc123/"
    mod.FILES = (f"{_sha(kick)} {len(kick)} Samples/Kick 1.wav\n"
                 f"{_sha(snare)} {len(snare)} Samples/snare.wav\n"
                 f"{_sha(b'<region> sample=Samples/snare.wav')} 33 kit.sfz\n")
    monkeypatch.setitem(sys.modules, "synth2.sounds.lists.fake_kit", mod)
    srcs = lists.sources("fake_kit")
    assert [s.url for s in srcs] == ["https://example.invalid/repo/abc123/Samples/Kick%201.wav",
                                     "https://example.invalid/repo/abc123/Samples/snare.wav",
                                     "https://example.invalid/repo/abc123/kit.sfz"]
    assert [(s.dest, s.name, s.unpack) for s in srcs][0] == ("Samples", "Kick 1.wav", "none")
    spec = _spec(env.kit_zip, id="listed", root="files", license_file="", sources=(),
                 listing="fake_kit", presets=())
    monkeypatch.setattr(registry, "REGISTRY", {"listed": spec})
    assert spec.manifest()["source"] == mod.BASE
    local = env.tmp / "clone"
    (local / "Samples").mkdir(parents=True)
    (local / "Samples" / "Kick 1.wav").write_bytes(kick)
    (local / "Samples" / "snare.wav").write_bytes(snare)
    (local / "kit.sfz").write_text("<region> sample=Samples/snare.wav")
    s = sounds.install("listed", accept_license=True, source=local)
    assert (s.root / "Samples" / "Kick 1.wav").read_bytes() == kick and s.files == [s.root / "kit.sfz"]
    (local / "Samples" / "snare.wav").write_bytes(snare[:-2] + b"\x01\x01")
    with pytest.raises(assets.VerificationError, match="snare.wav"):
        sounds.install("listed", accept_license=True, source=local, force=True)
    with pytest.raises(assets.AssetError, match="missing from this build"):
        lists.sources("no_such_list")


# ================================================================== add_local and remove

def _sample_folder(tmp: Path, name: str, names) -> Path:
    folder = tmp / name
    folder.mkdir(parents=True)
    for n in names:
        (folder / n).parent.mkdir(parents=True, exist_ok=True)
        (folder / n).write_bytes(_wav())
    return folder


def test_add_local_in_place_and_remove_keeps_the_users_files(env):
    kit = _sample_folder(env.tmp, "My Kit", ["kick.wav", "snare_01.wav", "hat closed.wav"])
    s = sounds.add_local(kit)
    assert (s.id, s.kind, s.name) == ("my-kit", "drumkit", "My Kit")
    assert s.external and s.root == kit.resolve() and s.files == [kit.resolve()]
    assert s.commercial_use is None and "responsible" in s.terms[0]
    assert s.size == 3 * len(_wav())
    assert _leftovers(env.dir / "my-kit") == [sounds.MANIFEST]
    with pytest.raises(assets.AssetError, match="already installed"):
        sounds.add_local(kit)
    assert sounds.remove("my-kit") is True
    assert sorted(p.name for p in kit.iterdir()) == ["hat closed.wav", "kick.wav", "snare_01.wav"]
    assert sounds.remove("my-kit") is False and sounds.find("my-kit") is None


def test_add_local_kinds(env):
    multi = _sample_folder(env.tmp, "flute", ["flute_C4.wav", "flute_E4.wav", "flute_G4.wav"])
    assert sounds.add_local(multi).kind == "samples"
    irs = sounds.add_local(_sample_folder(env.tmp, "rooms", ["hall.wav"]), kind="impulse",
                           license="CC0 1.0", commercial_use=True, credits=["Rooms: me."],
                           tags=["ir"])
    assert (irs.kind, irs.license, irs.commercial_use) == ("impulse", "CC0 1.0", True)
    assert irs.credits == ["Rooms: me."] and irs.terms == [] and irs.tags == ["ir"]
    # one soundfont file: registered in place, the file is the set
    font = env.tmp / "fonts" / "Nice Piano.sf2"
    font.parent.mkdir()
    font.write_bytes(b"RIFF\x00\x00\x00\x00sfbk")
    (font.parent / "other.sf2").write_bytes(b"RIFF")
    f = sounds.add_local(font)
    assert (f.id, f.kind, f.external) == ("nice-piano", "soundfont", True)
    assert f.files == [font.resolve()] and f.size == font.stat().st_size
    # a folder with an sfz, copied into the sounds directory
    inst = _sample_folder(env.tmp, "inst", ["s/a.wav"])
    (inst / "inst.sfz").write_text("<region> sample=s/a.wav")
    c = sounds.add_local(inst, "copied", copy=True)
    assert (c.kind, c.external) == ("sfz", False) and c.root == env.dir / "copied" / "files"
    assert c.files == [c.root / "inst.sfz"] and (c.root / "s" / "a.wav").is_file()
    # an archive is always unpacked
    z = sounds.add_local(env.kit_zip, "zipped", name="Zipped Kit")
    assert (z.kind, z.name, z.external) == ("sfz", "Zipped Kit", False)
    assert z.files == [z.root / "TestKit" / "kit.sfz"]
    assert sounds.remove("copied") and not (env.dir / "copied").exists()
    assert [s.id for s in sounds.installed()] == ["flute", "nice-piano", "rooms", "zipped"]


def test_add_local_refusals(env):
    with pytest.raises(assets.AssetError, match="registry"):
        sounds.add_local(_sample_folder(env.tmp, "a", ["kick.wav"]), "testkit")
    empty = env.tmp / "empty"
    empty.mkdir()
    with pytest.raises(assets.AssetError, match="holds no"):
        sounds.add_local(empty)
    text = env.tmp / "notes.txt"
    text.write_text("x")
    with pytest.raises(assets.AssetError, match="not a soundfont"):
        sounds.add_local(text)
    with pytest.raises(assets.AssetError, match="kind must be"):
        sounds.add_local(empty, kind="video")
    with pytest.raises(assets.AssetError, match="invalid id"):
        sounds.add_local(_sample_folder(env.tmp, "b", ["kick.wav"]), "a/b")
    hostile = _zip(env.tmp / "hostile.zip", {"../evil.sfz": "x"})
    with pytest.raises(assets.UnsafeArchive):
        sounds.add_local(hostile)
    assert _leftovers(env.dir) == []


def test_remove_only_touches_sound_folders(env):
    env.dir.mkdir()
    (env.dir / "stray").mkdir()
    (env.dir / "stray" / "keep.wav").write_bytes(b"x")
    assert sounds.remove("stray") is False and (env.dir / "stray" / "keep.wav").exists()
    assert sounds.remove("..") is False and sounds.remove("") is False
    sounds.install("testkit", accept_license=True, source=env.kit_zip)
    assert sounds.remove("testkit") is True and _leftovers(env.dir) == ["stray"]


# ================================================================== command line

def test_cli_listing_install_add_remove(env, capsys):
    assert _run(["sounds", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(env.dir)
    assert _run(["sounds", "list"]) == 0
    out = capsys.readouterr().out
    assert "no sound sets are installed" in out and "not installed" in out
    assert re.search(r"testkit\s+sfz\s+\d+ B", out)
    assert "licence: CC0 1.0; commercial use stated: yes" in out
    assert 'the publisher says: "Use it for anything."' in out
    assert "credit: Drums: Test Kit by Tester." in out

    assert _run(["sounds", "install", "testkit", "--from-file", str(env.kit_zip)]) == 2
    err = capsys.readouterr().err
    assert "CC0 1.0" in err and "Use it for anything." in err and "--accept-license" in err
    assert _leftovers(env.dir) == []
    assert _run(["sounds", "install"]) == 2
    assert _run(["sounds", "install", "nope", "--accept-license"]) == 2
    capsys.readouterr()

    assert _run(["sounds", "install", "testkit", "--from-file", str(env.kit_zip),
                 "--accept-license", "-q"]) == 0
    assert "installed testkit (sfz" in capsys.readouterr().out
    kit = _sample_folder(env.tmp, "Own", ["kick.wav", "snare.wav"])
    assert _run(["sounds", "add", str(kit), "--id", "own", "--license", "CC BY 4.0",
                 "--credit", "Drums: me."]) == 0
    capsys.readouterr()
    assert _run(["sounds", "list", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert [s["id"] for s in data["installed"]] == ["own", "testkit"] and data["available"] == []
    own = data["installed"][0]
    assert own["kind"] == "drumkit" and own["license"] == "CC BY 4.0" and own["external"]
    assert own["commercial_use"] is None and own["credits"] == ["Drums: me."]
    assert _run(["sounds", "list", "--kind", "drumkit"]) == 0
    out = capsys.readouterr().out
    assert "own" in out and "testkit" not in out and "commercial use not stated" in out
    assert _run(["sounds", "remove", "own"]) == 0 and kit.is_dir()
    assert _run(["sounds", "remove", "own"]) == 2
    assert _run(["sounds", "add", str(env.tmp / "missing")]) == 2


def test_cli_install_all_takes_only_plainly_commercial_sets(env, monkeypatch, capsys):
    reg = dict(env.registry)
    reg["sa"] = _spec(env.kit_zip, id="sa", share_alike=True)
    reg["nc"] = _spec(env.kit_zip, id="nc", commercial_use=False)
    reg["unknown"] = _spec(env.kit_zip, id="unknown", commercial_use=None)
    monkeypatch.setattr(registry, "REGISTRY", reg)
    monkeypatch.setattr(assets, "download",
                        lambda url, dest, **k: Path(dest).write_bytes(env.kit_zip.read_bytes()))
    assert _run(["sounds", "install", "--all", "--accept-license", "-q"]) == 0
    assert [s.id for s in sounds.installed()] == ["testkit"]
    assert _run(["sounds", "list"]) == 0
    out = capsys.readouterr().out
    assert "share-alike" in out and "commercial use stated: no" in out
    assert "commercial use not stated" in out


# ================================================================== the real registry

def test_registry_entries_are_complete():
    reg = sounds.REGISTRY
    assert reg, "the registry is empty"
    assert any("gm" in s.tags for s in reg.values()), "no General MIDI set"
    for sid, spec in reg.items():
        seen_urls = set()
        assert spec.id == sid and assets.check_id(sid) == sid
        assert spec.kind in sounds.KINDS, sid
        assert spec.name and spec.publisher and spec.license and spec.license_source, sid
        assert spec.homepage.startswith(("https://", "http://")), sid
        assert spec.files and spec.credits, sid
        assert spec.root == "files" or spec.root.startswith("files/"), sid
        assert not spec.license_file or spec.license_file.startswith("files/"), sid
        assert spec.commercial_use in (True, False, None), sid
        if spec.commercial_use is not None:
            assert spec.commercial_use_statement, f"{sid}: no publisher statement quoted"
        sources = spec.get_sources()
        assert sources, sid
        assert spec.size == sum(s.size for s in sources), sid
        stored = set()
        for src in sources:
            assert src.url.startswith("https://"), src.url
            assert re.fullmatch(r"[0-9a-f]{64}", src.sha256), src.url
            assert src.size > 0 and src.unpack in ("auto", "none", "zip", "tar", "deb")
            assert src.url not in seen_urls, f"{sid}: {src.url} listed twice"
            seen_urls.add(src.url)
            stored.add((src.dest + "/" if src.dest else "") + src.name)
        if all(s.unpack == "none" for s in sources):
            rel = spec.root[len("files"):].strip("/")
            for f in spec.files:
                want = "/".join(p for p in (rel, f) if p)
                assert f == "." or want in stored or any(x.startswith(want + "/") for x in stored), \
                    f"{sid}: {f} is not among the files of the set"
        manifest = spec.manifest()
        assert manifest["format"] == "synth2-sound" and manifest["id"] == sid
        assert json.loads(assets.dump_json(manifest)) == manifest
        for preset in spec.presets:
            assert preset.get("name") and preset.get("file"), sid
    rows = installer.available()
    assert {r["id"] for r in rows} == set(reg) and all("installed" in r for r in rows)


def test_every_registry_set_is_documented():
    text = (Path(__file__).resolve().parents[1] / "docs" / "SOUNDS.md").read_text(encoding="utf-8")
    for sid, spec in sounds.REGISTRY.items():
        assert f"`{sid}`" in text, f"{sid} is missing from docs/SOUNDS.md"
        for line in spec.credits:
            family = sid.startswith(("karoryfer-", "vsco2-"))
            assert family or line in text, f"the credit line of {sid} is not in docs/SOUNDS.md"


def test_installed_registry_sets_match_their_manifests():
    """On a machine with installed sets: every file a manifest names is there."""
    have = [s for s in sounds.installed() if s.id in sounds.REGISTRY]
    if not have:
        pytest.skip("no registry sound set is installed")
    for s in have:
        spec = sounds.REGISTRY[s.id]
        assert s.kind == spec.kind and len(s.files) == len(spec.files), s.id
        for preset in s.presets:
            assert s.file(preset["file"]).exists(), f"{s.id}: {preset['file']}"
        if s.manifest.get("license_file"):
            assert s.license_file.is_file(), s.id
