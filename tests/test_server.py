"""Tests for the web studio server (synth2.server.app).

The server is started on an ephemeral port in a background thread and driven
over real HTTP.  Endpoints that depend on other packages (tool layer, render
engine, file formats) are skipped when those packages are unavailable.
"""
from __future__ import annotations

import http.client
import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import pytest

from synth2.project import CURVE_SPECS, NOTE_ATTRIBUTES, Project
from synth2.server.app import StudioState, make_server


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def server():
    srv = make_server("127.0.0.1", 0, quiet=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def srv(server):
    """The shared server with a fresh studio state (new session, empty history)."""
    server.state = StudioState()
    return server


def call(srv, method, path, body=None, raw: bytes | None = None, headers=None):
    """Make a request; returns ``(status, headers, body bytes)`` (no exception on HTTP errors)."""
    url = srv.url.rstrip("/") + path
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    if data is not None and raw is None:
        req.add_header("Content-Type", "application/json")
    elif raw is not None and not any(k.lower() == "content-type" for k in (headers or {})):
        req.add_header("Content-Type", "application/octet-stream")  # like the studio's file upload
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.headers, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers, e.read()


def jcall(srv, method, path, body=None, **kw):
    status, headers, data = call(srv, method, path, body, **kw)
    return status, json.loads(data.decode("utf-8")) if data else None


def _short_project(title="Test song") -> dict:
    p = Project.new(title, bpm=120)
    p.tracks[0].add_note(1.0, 1.0, "A4", "la")
    return p.to_dict()


# --------------------------------------------------------------------------- static files
def test_index_and_static(srv):
    status, headers, body = call(srv, "GET", "/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert headers["Cache-Control"] == "no-store"
    assert b"Synth2" in body and b"/static/app.js" in body
    status, headers, body = call(srv, "GET", "/static/app.js")
    assert status == 200 and "javascript" in headers["Content-Type"]
    assert b"class PianoRoll" in body
    status, headers, _ = call(srv, "GET", "/static/style.css")
    assert status == 200 and headers["Content-Type"].startswith("text/css")


def test_static_rejects_traversal_and_unknown(srv):
    status, _, _ = call(srv, "GET", "/static/../app.py")
    assert status in (403, 404)
    status, _, _ = call(srv, "GET", "/static/%2e%2e/app.py")
    assert status in (403, 404)
    status, body = jcall(srv, "GET", "/api/nope")
    assert status == 404 and "error" in body
    status, body = jcall(srv, "DELETE", "/api/project")
    assert status == 405


# --------------------------------------------------------------------------- meta / project
def test_meta(srv):
    status, meta = jcall(srv, "GET", "/api/meta")
    assert status == 200
    assert meta["voices"] and all("id" in v and "name" in v for v in meta["voices"])
    assert meta["styles"] and all("name" in s for s in meta["styles"])
    assert set(meta["curves"]) == set(CURVE_SPECS)
    for name, spec in meta["curves"].items():
        assert spec["min"] <= spec["default"] <= spec["max"], name
    assert set(meta["note_attributes"]) == set(NOTE_ATTRIBUTES)
    assert meta["phonemes"] and "name" in meta["phonemes"][0]
    assert "available" in meta["director"]
    assert meta["voice_params"] and meta["voice_defaults"]


def test_get_project(srv):
    status, p = jcall(srv, "GET", "/api/project")
    assert status == 200
    assert p["format"] == "synth2-project"
    assert len(p["tracks"]) >= 1
    Project.from_dict(p)  # valid model


def test_put_project_roundtrip_and_history(srv):
    d = _short_project("Round trip")
    d["tracks"][0]["curves"] = {"loudness": {"interpolation": "linear", "points": [[0, 0], [2, -6]]}}
    d["metadata"] = {"track_colors": {d["tracks"][0]["id"]: "#ff0000"}}
    status, r = jcall(srv, "PUT", "/api/project", d)
    assert status == 200 and r["ok"]
    assert r["history"]["undo"] == 1 and r["history"]["redo"] == 0
    status, p = jcall(srv, "GET", "/api/project")
    assert p["title"] == "Round trip"
    assert p["tracks"][0]["notes"][0]["pitch"] == 69
    assert p["tracks"][0]["curves"]["loudness"]["points"] == [[0, 0], [2, -6]]
    assert p["metadata"]["track_colors"] == d["metadata"]["track_colors"]
    status, h = jcall(srv, "GET", "/api/history")
    assert h["undo"] == 1 and h["revision"] >= 1


def test_put_project_validation(srv):
    status, body = jcall(srv, "PUT", "/api/project", {"tracks": [{"notes": [{"onset": "soon", "duration": 1}]}]})
    assert status == 400 and "invalid project" in body["error"]
    status, _, data = call(srv, "PUT", "/api/project", raw=b"{not json", headers={"Content-Type": "application/json"})
    assert status == 400
    # nothing was pushed to the history by failed requests
    status, h = jcall(srv, "GET", "/api/history")
    assert h["undo"] == 0


def test_new_project(srv):
    status, r = jcall(srv, "POST", "/api/new", {"title": "Fresh", "bpm": 90})
    assert status == 200 and r["project"]["title"] == "Fresh"
    assert r["project"]["tempo"][0]["bpm"] == 90
    assert r["history"]["undo"] == 1
    assert r["history"]["path"] is None  # Save must ask for a file name


# --------------------------------------------------------------------------- undo / redo
def test_undo_redo_after_put(srv):
    status, original = jcall(srv, "GET", "/api/project")
    edited = json.loads(json.dumps(original))
    edited["title"] = "Edited"
    assert jcall(srv, "PUT", "/api/project", edited)[0] == 200
    status, r = jcall(srv, "POST", "/api/undo")
    assert status == 200 and r["project"]["title"] == original["title"]
    assert r["history"]["redo"] == 1
    status, r = jcall(srv, "POST", "/api/redo")
    assert status == 200 and r["project"]["title"] == "Edited"
    # undo past the beginning of history is a clean 400
    assert jcall(srv, "POST", "/api/undo")[0] == 200
    status, body = jcall(srv, "POST", "/api/undo")
    assert status == 400 and "undo" in body["error"]


def test_keepalive_connection_survives_ignored_bodies(srv):
    """A body the handler does not read must not corrupt the next request on the same connection."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        conn.request("POST", "/api/redo", body=b'{"x": 1}', headers={"Content-Type": "application/json"})
        r1 = conn.getresponse()
        r1.read()
        assert r1.status == 400  # nothing to redo
        conn.request("GET", "/api/history")
        r2 = conn.getresponse()
        assert r2.status == 200
        assert "undo" in json.loads(r2.read())
    finally:
        conn.close()


# --------------------------------------------------------------------------- tool layer
def test_tools_and_tool_call(srv):
    pytest.importorskip("synth2.ai.tools")
    status, tools = jcall(srv, "GET", "/api/tools")
    assert status == 200
    names = {t["name"] for t in tools}
    assert {"get_project", "add_notes", "undo", "redo"} <= names
    assert all("input_schema" in t for t in tools)

    status, r = jcall(srv, "POST", "/api/tool", {"name": "add_notes", "args": {
        "track": 0, "notes": [{"onset": 1, "duration": 1, "pitch": "C4", "lyric": "hel"},
                              {"onset": 2, "duration": 1, "pitch": "E4", "lyric": "lo"}]}})
    assert status == 200, r
    status, p = jcall(srv, "GET", "/api/project")
    assert [n["pitch"] for n in p["tracks"][0]["notes"]][-2:] == [60, 64]

    status, r = jcall(srv, "POST", "/api/tool", {"name": "undo", "args": {}})
    assert status == 200
    status, p = jcall(srv, "GET", "/api/project")
    assert all(n["pitch"] not in (60, 64) or n["lyric"] not in ("hel", "lo") for n in p["tracks"][0]["notes"])

    status, body = jcall(srv, "POST", "/api/tool", {"name": "no_such_tool", "args": {}})
    assert status == 400 and "error" in body
    status, body = jcall(srv, "POST", "/api/tool", {"args": {}})
    assert status == 400
    status, body = jcall(srv, "POST", "/api/tool", {"name": "transpose", "args": {"semitones": "lots"}})
    assert status == 400 and "error" in body


def test_put_then_tool_undo_share_history(srv):
    pytest.importorskip("synth2.ai.tools")
    assert jcall(srv, "PUT", "/api/project", _short_project("Shared"))[0] == 200
    status, r = jcall(srv, "POST", "/api/tool", {"name": "undo", "args": {}})
    assert status == 200
    status, p = jcall(srv, "GET", "/api/project")
    assert p["title"] != "Shared"


# --------------------------------------------------------------------------- AI director
def test_ai_returns_501_without_api_key(srv, monkeypatch, tmp_path):
    for var in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_FEDERATION_RULE_ID"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))  # no ~/.config/anthropic profile either
    status, body = jcall(srv, "POST", "/api/ai", {"message": "write a melody"})
    assert status == 501
    assert "ANTHROPIC_API_KEY" in body["error"]
    status, meta = jcall(srv, "GET", "/api/meta")
    assert meta["director"]["available"] is False


# --------------------------------------------------------------------------- lyrics helpers
def test_phonemes_and_split_lyrics(srv):
    d = _short_project()
    d["tracks"][0]["notes"] = [
        {"id": "n1", "onset": 1, "duration": 1, "pitch": 60, "lyric": "hello"},
        {"id": "n2", "onset": 2, "duration": 1, "pitch": 62, "lyric": "+"},
        {"id": "n3", "onset": 3, "duration": 1, "pitch": 64, "lyric": "-"},
        {"id": "n4", "onset": 4, "duration": 1, "pitch": 65, "lyric": "x", "phonemes": "zz qq"},
    ]
    assert jcall(srv, "PUT", "/api/project", d)[0] == 200
    status, r = jcall(srv, "GET", "/api/phonemes")
    assert status == 200
    tid = d["tracks"][0]["id"]
    ph = r["tracks"][tid]
    assert ph["n1"] and ph["n2"]  # "hello" + "+" split over two notes
    assert ph["n3"] == "-"  # melisma
    assert "n4" in r["errors"][tid]  # invalid explicit phonemes are reported, not fatal
    assert "n1" not in r["errors"][tid]
    status, r = jcall(srv, "POST", "/api/split_lyrics", {"text": "twinkle little star", "language": "en"})
    assert status == 200
    assert r["tokens"][0] == "twinkle" and r["tokens"][1] == "+" and r["tokens"][-1] == "star"


# --------------------------------------------------------------------------- render
def test_render_returns_wav_with_report(srv):
    pytest.importorskip("synth2.engine.renderer")
    assert jcall(srv, "PUT", "/api/project", _short_project("Render me"))[0] == 200
    status, headers, data = call(srv, "POST", "/api/render", {"start_beat": 0, "end_beat": 2})
    assert status == 200, data[:300]
    assert headers["Content-Type"] == "audio/wav"
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    report = json.loads(headers["X-Render-Report"])
    assert report["duration_s"] > 0
    assert headers["X-Render-Report"].isascii()
    status, headers2, data2 = call(srv, "GET", "/api/render/last")
    assert status == 200 and data2 == data
    # the tool layer knows about the studio render (analyze_render works on it)
    session = srv.state.session
    if hasattr(session, "last_render"):
        assert session.last_render is not None


def test_render_last_404_before_render(srv):
    status, body = jcall(srv, "GET", "/api/render/last")
    assert status == 404


# --------------------------------------------------------------------------- files
@pytest.mark.parametrize("fmt", ["json", "svp", "mid", "musicxml", "ust", "s2s"])
def test_export_formats(srv, fmt):
    if fmt != "json":
        io = pytest.importorskip("synth2.io")
        if not hasattr(io, "export_any"):
            pytest.skip("synth2.io.export_any not available")
    assert jcall(srv, "PUT", "/api/project", _short_project("Export test"))[0] == 200
    status, headers, data = call(srv, "GET", f"/api/export?format={fmt}")
    assert status == 200, data[:300]
    assert "attachment" in headers["Content-Disposition"]
    assert "Export_test" in headers["Content-Disposition"]
    assert len(data) > 10
    if fmt == "mid":
        assert data[:4] == b"MThd"


def test_export_unknown_format(srv):
    status, body = jcall(srv, "GET", "/api/export?format=docx")
    assert status == 400


def test_import_roundtrip(srv):
    io = pytest.importorskip("synth2.io")
    if not hasattr(io, "load_any"):
        pytest.skip("synth2.io.load_any not available")
    d = _short_project("Import me")
    d["tracks"][0]["notes"].append({"onset": 2, "duration": 1, "pitch": 72, "lyric": "oh"})
    assert jcall(srv, "PUT", "/api/project", d)[0] == 200
    _, _, svp = call(srv, "GET", "/api/export?format=svp")
    assert jcall(srv, "POST", "/api/new", {})[0] == 200
    status, r = jcall(srv, "POST", "/api/import?filename=song.svp", raw=svp, headers={"Content-Type": "application/octet-stream"})
    assert status == 200, r
    notes = r["project"]["tracks"][0]["notes"]
    assert [n["pitch"] for n in notes] == [69, 72]
    status, r = jcall(srv, "POST", "/api/import?filename=song.docx", raw=b"xx")
    assert status == 400
    status, r = jcall(srv, "POST", "/api/import?filename=bad.svp", raw=b"\x00\x01garbage")
    assert status == 400


def test_save(srv, tmp_path):
    status, body = jcall(srv, "POST", "/api/save", {})
    assert status == 400  # no path yet
    target = tmp_path / "saved.s2p.json"
    status, r = jcall(srv, "POST", "/api/save", {"path": str(target)})
    assert status == 200 and r["path"] == str(target)
    assert Project.from_json(target.read_text())  # valid project file
    status, h = jcall(srv, "GET", "/api/history")
    assert h["path"] == str(target)
    status, r = jcall(srv, "POST", "/api/save", {})  # now saves to the remembered path
    assert status == 200


def test_ai_chat_with_fake_director(srv, monkeypatch):
    """The endpoint runs Director.chat on the shared session and returns the updated project."""
    tools = pytest.importorskip("synth2.ai.tools")
    import synth2.server.app as app_mod

    class FakeDirector:
        def __init__(self, session):
            self.session = session
            self.messages = []

        def chat(self, message):
            self.messages.append(message)
            tools.call_tool(self.session, "add_notes", {"track": 0, "notes": [
                {"onset": 1, "duration": 2, "pitch": "G4", "lyric": "sky"}]})
            return {"reply": "Added a note.", "actions": [{"tool": "add_notes", "args": {"track": 0}, "ok": True}]}

    monkeypatch.setattr(app_mod, "director_status", lambda: {"available": True, "reason": ""})
    srv.state.director = FakeDirector(srv.state.session)
    status, r = jcall(srv, "POST", "/api/ai", {"message": "add a note"})
    assert status == 200, r
    assert r["reply"] == "Added a note." and r["actions"][0]["tool"] == "add_notes"
    assert any(n["lyric"] == "sky" for n in r["project"]["tracks"][0]["notes"])
    assert r["history"]["undo"] == 1  # the director's edit is undoable from the studio
    assert srv.state.director.messages == ["add a note"]
    status, body = jcall(srv, "POST", "/api/ai", {"message": "  "})
    assert status == 400


def test_fallback_session_without_tool_layer(server):
    """The studio keeps working (edit, undo, redo) with the built-in session when the tool layer is absent."""
    from synth2.server.app import FallbackSession

    server.state = StudioState()
    server.state.session = FallbackSession(Project.new("Fallback"))
    assert jcall(server, "PUT", "/api/project", _short_project("Edited"))[0] == 200
    status, r = jcall(server, "POST", "/api/undo")
    assert status == 200 and r["project"]["title"] == "Fallback"
    status, r = jcall(server, "POST", "/api/redo")
    assert status == 200 and r["project"]["title"] == "Edited"
    status, body = jcall(server, "POST", "/api/tool", {"name": "transpose", "args": {}})
    assert status == 501


# --------------------------------------------------------------------------- security / robustness regressions
def hcall(srv, method, path, body=None, headers=None):
    """Raw request with exactly the given headers (http.client adds no Content-Type or Origin)."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=30)
    try:
        conn.request(method, path, body=body, headers=headers or {})
        r = conn.getresponse()
        return r.status, r, r.read()
    finally:
        conn.close()


def test_cross_site_writes_are_refused(srv, tmp_path):
    """CSRF: another site's page must not save, export, replace the project or run tools."""
    host, port = srv.server_address[:2]
    victim = tmp_path / "victim.json"
    victim.write_text("original")
    save = json.dumps({"path": str(victim)}).encode()
    evil = {"Origin": "http://evil.example", "Content-Type": "text/plain"}
    assert hcall(srv, "POST", "/api/save", save, evil)[0] == 403
    assert hcall(srv, "POST", "/api/save", save, {**evil, "Content-Type": "application/json"})[0] == 403
    assert hcall(srv, "POST", "/api/save", save, {"Origin": "null", "Content-Type": "application/json"})[0] == 403
    assert hcall(srv, "POST", "/api/save", save, {"Sec-Fetch-Site": "cross-site",
                                                  "Content-Type": "application/json"})[0] == 403
    # a "simple" request (no preflight) is refused even without an Origin header
    assert hcall(srv, "POST", "/api/save", save, {"Content-Type": "text/plain"})[0] == 415
    assert hcall(srv, "POST", "/api/save", save)[0] == 415  # no Content-Type at all
    assert victim.read_text() == "original"
    pom = tmp_path / "sub" / "dir" / "pom.xml"
    export = json.dumps({"name": "export", "args": {"path": str(pom)}}).encode()
    assert hcall(srv, "POST", "/api/tool", export, evil)[0] == 403
    assert hcall(srv, "POST", "/api/tool", export, {"Content-Type": "text/plain"})[0] == 415
    assert not (tmp_path / "sub").exists()
    # the POST alias of /api/project (used by the studio's unload beacon) is just as protected
    proj = json.dumps(_short_project("Hijacked")).encode()
    assert hcall(srv, "POST", "/api/project", proj, evil)[0] == 403
    assert hcall(srv, "POST", "/api/project", proj, {"Content-Type": "text/plain"})[0] == 415
    assert hcall(srv, "POST", "/api/project", b"", {})[0] == 400  # empty body is not a project
    assert hcall(srv, "POST", "/api/import?filename=a.s2s", b"title: X", {"Content-Type": "text/plain"})[0] == 415
    status, h = jcall(srv, "GET", "/api/history")
    assert h["undo"] == 0
    # JSON endpoints want JSON
    assert hcall(srv, "POST", "/api/tool", export, {"Content-Type": "application/octet-stream"})[0] == 415
    # the studio itself (same origin) still works
    same = {"Origin": f"http://{host}:{port}", "Sec-Fetch-Site": "same-origin", "Content-Type": "application/json"}
    status, _, data = hcall(srv, "POST", "/api/project", proj, same)
    assert status == 200, data
    status, _, data = hcall(srv, "POST", "/api/save", save, same)
    assert status == 200, data
    assert Project.from_json(victim.read_text()).title == "Hijacked"


def test_foreign_host_header_is_refused(srv, tmp_path):
    """DNS rebinding: a request whose Host names another site gets 403 before any routing."""
    port = srv.server_address[1]
    secret = tmp_path / "secrets.txt"
    secret.write_text("AWS_SECRET_ACCESS_KEY=abcd1234supersecret\n")
    for host in (f"rebind.evil.example:{port}", "evil.example", f"localhost.evil.example:{port}", "a@127.0.0.1"):
        for method, path, body in (("GET", "/api/project", None), ("GET", "/", None),
                                   ("POST", "/api/tool", json.dumps({"name": "load_project",
                                                                     "args": {"path": str(secret)}}).encode())):
            headers = {"Host": host, "Content-Type": "application/json"}
            status, _, data = hcall(srv, method, path, body, headers)
            assert status == 403, (host, path)
            assert b"abcd1234" not in data
    assert srv.state.session.path is None  # load_project never ran
    for host in (f"127.0.0.1:{port}", f"localhost:{port}", f"LOCALHOST:{port}", f"[::1]:{port}", "localhost"):
        status, _, _ = hcall(srv, "GET", "/api/history", None, {"Host": host})
        assert status == 200, host
    status, r, _ = hcall(srv, "GET", "/api/project")
    assert r.getheader("X-Content-Type-Options") == "nosniff"
    assert r.getheader("Cross-Origin-Resource-Policy") == "same-origin"


def test_host_allowlist_options():
    s = make_server("127.0.0.1", 0, quiet=True, allowed_hosts=["Studio.Example"])
    try:
        assert s.host_allowed("studio.example:443") and s.host_allowed("localhost:1234")
        assert s.host_allowed("192.168.1.20:8765") and s.host_allowed("[::1]:80")
        assert not s.host_allowed("evil.example:8765") and not s.host_allowed("") and not s.host_allowed("x y")
    finally:
        s.server_close()


def test_save_as_export_does_not_become_project_path(srv, tmp_path):
    io = pytest.importorskip("synth2.io")
    if not hasattr(io, "export_any"):
        pytest.skip("synth2.io.export_any not available")
    d = _short_project("Original")
    d["tracks"][0]["curves"] = {"loudness": {"interpolation": "linear", "points": [[0, 0], [2, -6]]}}
    assert jcall(srv, "PUT", "/api/project", d)[0] == 200
    # never saved as a project: exporting does not invent a project path
    status, r = jcall(srv, "POST", "/api/save", {"path": str(tmp_path / "early.mid")})
    assert status == 200 and r["exported"] is True and r["project_path"] is None
    status, body = jcall(srv, "POST", "/api/save", {})
    assert status == 400
    native = tmp_path / "song.s2p.json"
    status, r = jcall(srv, "POST", "/api/save", {"path": str(native)})
    assert status == 200 and r["exported"] is False and r["format"] == "json"
    status, r = jcall(srv, "POST", "/api/save", {"path": str(tmp_path / "song.mid")})
    assert status == 200 and r["path"] == str(tmp_path / "song.mid")
    assert r["exported"] is True and r["format"] == "midi" and r["project_path"] == str(native)
    assert (tmp_path / "song.mid").read_bytes()[:4] == b"MThd"
    assert jcall(srv, "GET", "/api/history")[1]["path"] == str(native)
    d["title"] = "Edited"
    assert jcall(srv, "PUT", "/api/project", d)[0] == 200
    status, r = jcall(srv, "POST", "/api/save", {})  # plain Save: the lossless project file
    assert status == 200 and r["path"] == str(native)
    saved = Project.from_json(native.read_text())
    assert saved.title == "Edited"
    assert saved.to_dict()["tracks"][0]["curves"]["loudness"]["points"] == [[0, 0], [2, -6]]


def test_save_unknown_extension_appends_native_suffix(srv, tmp_path):
    from synth2.server.app import save_format

    for name in ("My Song v1.2", "take.final", "out.wav"):
        status, r = jcall(srv, "POST", "/api/save", {"path": str(tmp_path / name)})
        assert status == 200, r
        assert r["path"] == str(tmp_path / (name + ".s2p.json")) and r["exported"] is False
        assert Project.from_json(Path(r["path"]).read_text())
    assert jcall(srv, "GET", "/api/history")[1]["path"] == str(tmp_path / "out.wav.s2p.json")
    assert save_format("a.S2P.JSON") == "json" and save_format("a.s2s") == "score" and save_format("a.2") is None
    status, body = jcall(srv, "POST", "/api/save", {"path": ["x"]})
    assert status == 400


def test_project_path_header_cannot_be_split(srv, tmp_path):
    """CR/LF in a file name must never reach the response headers (header injection)."""
    from synth2.server.app import _ascii

    evil = str(tmp_path / "a\r\nSet-Cookie: pwn=1\r\nX-Evil: y.s2p.json")
    status, body = jcall(srv, "POST", "/api/save", {"path": evil})
    assert status == 400 and "control characters" in body["error"]
    srv.state.session.path = evil  # e.g. set by load_project or --project
    status, r, data = hcall(srv, "GET", "/api/project")
    assert status == 200
    names = [k.lower() for k, _ in r.getheaders()]
    assert "set-cookie" not in names and "x-evil" not in names
    value = r.getheader("X-Project-Path")
    assert "\r" not in value and "\n" not in value and "%0D%0ASet-Cookie" in value
    assert urllib.parse.unquote(value) == evil
    assert json.loads(data)["title"]
    assert _ascii('café "x"\x7f') == "caf%C3%A9 %22x%22%7F"


def test_nan_and_infinity_are_rejected(srv):
    d = _short_project("Strict")
    raw = json.dumps(d).replace('"bpm": 120.0', '"bpm": NaN').encode()
    assert b"NaN" in raw
    status, _, data = call(srv, "PUT", "/api/project", raw=raw, headers={"Content-Type": "application/json"})
    assert status == 400 and b"NaN" in data
    for bad in ('"bpm": Infinity', '"bpm": -Infinity', '"bpm": 1e999'):
        raw = json.dumps(d).replace('"bpm": 120.0', bad).encode()
        assert call(srv, "PUT", "/api/project", raw=raw, headers={"Content-Type": "application/json"})[0] == 400
    d2 = json.loads(json.dumps(d))
    d2["tempo"][0]["bpm"] = "inf"  # float("inf") would accept the string
    assert jcall(srv, "PUT", "/api/project", d2)[0] == 400
    for bpm in ("nan", "inf", "-inf"):
        assert jcall(srv, "POST", "/api/new", {"bpm": bpm})[0] == 400
    status, _, data = call(srv, "GET", "/api/project")

    def strict(name):
        raise AssertionError(f"non-standard JSON constant {name}")

    json.loads(data, parse_constant=strict)  # still valid for a browser's JSON.parse
    assert jcall(srv, "GET", "/api/history")[1]["undo"] == 0


def test_json_output_is_always_strict():
    from synth2.server.app import dumps_json

    assert dumps_json({"a": float("nan"), "b": [float("-inf"), 1.5]}) == '{"a": null, "b": [null, 1.5]}'
    assert dumps_json({"x": 1}) == '{"x": 1}'


def test_new_project_rejects_bad_bpm(srv):
    for bpm in ("abc", -50, [1], True, 0, 1000, {"x": 1}):
        status, body = jcall(srv, "POST", "/api/new", {"bpm": bpm})
        assert status == 400 and "bpm" in body["error"], bpm
    assert jcall(srv, "GET", "/api/history")[1]["undo"] == 0
    status, r = jcall(srv, "POST", "/api/new", {"bpm": "96"})
    assert status == 200 and r["project"]["tempo"][0]["bpm"] == 96


def test_render_args_validation():
    from synth2.server.app import RENDER_MARGIN_BEATS, render_args

    p = Project.from_dict(_short_project())  # one note, beats 1-2
    tid = p.tracks[0].id
    assert render_args(p) == (None, None, None)
    assert render_args(p, 0, 2, [tid]) == (0.0, 2.0, [tid])
    assert render_args(p, None, None, tid) == (None, None, [tid])  # one id, not its characters
    assert render_args(p, None, None, []) == (None, None, None)
    limit = 2.0 + RENDER_MARGIN_BEATS
    assert render_args(p, 0, 1e9) == (0.0, limit, None)
    assert render_args(p, 1e9, 2e9) == (limit, None, None)  # wholly past the song: short silence
    for kwargs in ({"start_beat": -50}, {"start_beat": -0.001, "end_beat": 2}, {"start_beat": float("nan")},
                   {"end_beat": float("inf")}, {"start_beat": "soon"}, {"start_beat": True}, {"start_beat": [1]},
                   {"start_beat": 1, "end_beat": 0.5}, {"start_beat": 1, "end_beat": 1}, {"end_beat": 0},
                   {"track_ids": "zz"}, {"track_ids": [1]}, {"track_ids": {"a": 1}}, {"start_beat": 10 ** 400}):
        with pytest.raises(ValueError):
            render_args(p, kwargs.get("start_beat"), kwargs.get("end_beat"), kwargs.get("track_ids"))


def test_render_rejects_bad_ranges_and_track_ids(srv):
    pytest.importorskip("synth2.engine.renderer")
    d = _short_project("Render checks")
    tid = d["tracks"][0]["id"]
    assert jcall(srv, "PUT", "/api/project", d)[0] == 200
    for body in ({"start_beat": -50, "end_beat": 2}, {"start_beat": -1000}, {"start_beat": 1, "end_beat": 0.5},
                 {"track_ids": "zz"}, {"track_ids": [7]}, {"start_beat": "nan"}):
        status, _, data = call(srv, "POST", "/api/render", body)
        assert status == 400, (body, data[:200])
    raw = b'{"start_beat": NaN}'
    assert call(srv, "POST", "/api/render", raw=raw, headers={"Content-Type": "application/json"})[0] == 400
    status, headers, one = call(srv, "POST", "/api/render", {"track_ids": tid, "start_beat": 0, "end_beat": 2})
    assert status == 200
    status, _, listed = call(srv, "POST", "/api/render", {"track_ids": [tid], "start_beat": 0, "end_beat": 2})
    assert status == 200 and one == listed  # a single id string means that track, not its characters
    status, headers, data = call(srv, "POST", "/api/render", {"start_beat": 0, "end_beat": 1e9})
    assert status == 200
    report = json.loads(headers["X-Render-Report"])
    assert report["end_beat"] <= 2 + 32 and report["duration_s"] < 30


def test_chunked_request_body(srv):
    """Transfer-Encoding: chunked bodies are decoded, never re-parsed as a second request."""
    host, port = srv.server_address[:2]
    conn = http.client.HTTPConnection(host, port, timeout=10)
    try:
        body = iter([b'{"title": "Chun', b'ked", "bpm": 90}'])  # http.client sends an iterable chunked
        conn.request("POST", "/api/new", body=body, headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        data = json.loads(r.read())
        assert r.status == 200, data
        assert data["project"]["title"] == "Chunked" and data["project"]["tempo"][0]["bpm"] == 90
        conn.request("GET", "/api/history")  # same keep-alive connection still in sync
        r = conn.getresponse()
        assert r.status == 200 and json.loads(r.read())["undo"] == 1
    finally:
        conn.close()
    import socket

    def raw_request(payload: bytes) -> bytes:
        with socket.create_connection((host, port), timeout=10) as s:
            s.sendall(payload)
            out = b""
            while chunk := s.recv(65536):
                out += chunk
            return out

    smuggled = b'POST /api/undo HTTP/1.1\r\nHost: x\r\n\r\n'
    out = raw_request(b"POST /api/new HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n"
                      b"Transfer-Encoding: chunked\r\n\r\nzz\r\n" + smuggled)
    assert out.startswith(b"HTTP/1.1 400") and out.count(b"HTTP/1.1") == 1  # closed, nothing smuggled
    out = raw_request(b"POST /api/new HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: gzip\r\n\r\nxx")
    assert out.startswith(b"HTTP/1.1 501")
    assert jcall(srv, "GET", "/api/history")[1]["undo"] == 1


def test_tool_file_errors_are_400(srv, tmp_path):
    pytest.importorskip("synth2.ai.tools")
    blocker = tmp_path / "file.txt"
    blocker.write_text("x")
    status, body = jcall(srv, "POST", "/api/tool", {"name": "save_project",
                                                    "args": {"path": str(blocker / "x.s2p.json")}})
    assert status == 400 and body["tool"] == "save_project" and "error" in body
    (tmp_path / "d.s2p.json").mkdir()
    status, body = jcall(srv, "POST", "/api/tool", {"name": "export",
                                                    "args": {"path": str(tmp_path / "d.s2p.json"), "format": "json"}})
    assert status == 400 and body["tool"] == "export"


def test_ai_chat_does_not_freeze_the_studio(srv, monkeypatch):
    """A long Director.chat must not hold the studio lock: other requests keep being answered."""
    tools = pytest.importorskip("synth2.ai.tools")
    import time

    import synth2.server.app as app_mod

    started, release = threading.Event(), threading.Event()

    class SlowDirector:
        def __init__(self, session):
            self.session = session

        def chat(self, message):
            started.set()
            release.wait(10)  # a slow Claude round trip
            tools.call_tool(self.session, "add_notes", {"track": 0, "notes": [
                {"onset": 4, "duration": 1, "pitch": "C5", "lyric": "late"}]})
            return {"reply": "done", "actions": []}

    monkeypatch.setattr(app_mod, "director_status", lambda: {"available": True, "reason": ""})
    srv.state.director = SlowDirector(srv.state.session)
    results = {}
    worker = threading.Thread(target=lambda: results.setdefault("ai", jcall(srv, "POST", "/api/ai", {"message": "go"})))
    worker.start()
    try:
        assert started.wait(10)
        t0 = time.perf_counter()
        assert jcall(srv, "GET", "/api/history")[0] == 200
        assert jcall(srv, "PUT", "/api/project", _short_project("During AI"))[0] == 200
        assert time.perf_counter() - t0 < 2.0
        status, body = jcall(srv, "POST", "/api/ai", {"message": "again"})
        assert status == 409  # one conversation at a time
    finally:
        release.set()
        worker.join(15)
    status, r = results["ai"]
    assert status == 200 and r["reply"] == "done"
    assert r["project"]["title"] == "During AI"  # the director's edit applied on top of the studio's
    assert any(n["lyric"] == "late" for n in r["project"]["tracks"][0]["notes"])
    assert r["history"]["undo"] == 2


def test_undo_of_new_project_or_import_restores_the_save_path():
    from synth2.ai.tools import Session
    from synth2.server.app import push_undo

    s = Session()
    s.path = "song.s2p.json"
    push_undo(s, "import x.svp", new_path=None)
    s.path = None
    s.undo()
    assert s.path == "song.s2p.json"
    s.redo()
    assert s.path is None
