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
import urllib.request

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
