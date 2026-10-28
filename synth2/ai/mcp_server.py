"""Dependency-free MCP (Model Context Protocol) server over stdio.

Exposes every tool in :data:`synth2.ai.tools.TOOLS` to MCP clients (Claude
Desktop, Claude Code, ...).  Transport: newline-delimited JSON-RPC 2.0 on
stdin/stdout; logs go to stderr only (stdout carries nothing but protocol
messages).

Run::

    python -m synth2.ai.mcp_server [--project song.s2p.json] [--autosave]

Client configuration example (Claude Desktop ``claude_desktop_config.json``)::

    {"mcpServers": {"synth2": {"command": "python", "args": ["-m", "synth2.ai.mcp_server",
                                                             "--project", "/path/song.s2p.json"]}}}
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any, BinaryIO

from .. import __version__
from .tools import MUTATING_TOOLS, TOOLS, Session, ToolError, call_tool

DEFAULT_PROTOCOL_VERSION = "2025-06-18"
SERVER_NAME = "synth2"

INSTRUCTIONS = (
    "Synth2 Vocal Studio: an AI-native singing voice synthesizer. Time is in beats (quarter notes, beat 0 = song "
    "start); pitch is MIDI or note names (C4 = 60). Typical workflow: new_project -> write_melody (score notation "
    "'C4 q la  D4 q la  E4 h la') -> set_lyrics -> update_track/apply_expression (voice, style) -> add_harmony -> "
    "check_project -> render -> analyze_render -> fix and re-render. Every edit is undoable (undo/redo)."
)

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603


def log(*parts):
    print("[synth2-mcp]", *parts, file=sys.stderr, flush=True)


class MCPServer:
    """Protocol handler. ``handle(message)`` returns the response dict, or ``None`` for notifications."""

    def __init__(self, session: Session | None = None, autosave: bool = False):
        self.session = session or Session()
        self.autosave = autosave
        self.initialized = False
        self.protocol_version = DEFAULT_PROTOCOL_VERSION

    # -- dispatch ----------------------------------------------------------------
    def handle(self, msg: Any) -> dict | list | None:
        if isinstance(msg, list):  # JSON-RPC batch (older protocol revisions)
            if not msg:
                return _error(None, INVALID_REQUEST, "empty batch")
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            return out or None
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return _error(msg.get("id") if isinstance(msg, dict) else None, INVALID_REQUEST,
                          "invalid JSON-RPC 2.0 message")
        method = msg.get("method")
        is_notification = "id" not in msg
        mid = msg.get("id")
        if method is None:
            # a response from the client (we never send requests) - ignore
            return None
        params = msg.get("params") or {}
        if not isinstance(params, dict):
            return None if is_notification else _error(mid, INVALID_PARAMS, "params must be an object")
        handler = getattr(self, "m_" + str(method).replace("/", "_"), None)
        if is_notification:
            if handler is not None and str(method).startswith("notifications/"):
                try:
                    handler(params)
                except Exception:  # pragma: no cover - notifications never answer
                    log("notification handler failed:", traceback.format_exc())
            return None
        if handler is None or str(method).startswith("notifications/"):
            return _error(mid, METHOD_NOT_FOUND, f"method not found: {method}")
        try:
            result = handler(params)
        except _RPCError as exc:
            return _error(mid, exc.code, exc.message)
        except Exception as exc:  # pragma: no cover - defensive
            log("internal error:", traceback.format_exc())
            return _error(mid, INTERNAL_ERROR, f"internal error: {exc}")
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    # -- methods -----------------------------------------------------------------
    def m_initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        self.protocol_version = requested if isinstance(requested, str) and requested else DEFAULT_PROTOCOL_VERSION
        client = (params.get("clientInfo") or {}).get("name", "unknown client")
        log(f"initialize from {client} (protocol {self.protocol_version})")
        return {
            "protocolVersion": self.protocol_version,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": __version__},
            "instructions": INSTRUCTIONS,
        }

    def m_notifications_initialized(self, params: dict):
        self.initialized = True

    def m_notifications_cancelled(self, params: dict):
        pass  # tools run synchronously; nothing to cancel

    def m_ping(self, params: dict) -> dict:
        return {}

    def m_tools_list(self, params: dict) -> dict:
        return {"tools": [{"name": t["name"], "description": t["description"], "inputSchema": t["input_schema"]}
                          for t in TOOLS]}

    def m_tools_call(self, params: dict) -> dict:
        name = params.get("name")
        args = params.get("arguments") or {}
        if not isinstance(name, str) or name not in {t["name"] for t in TOOLS}:
            raise _RPCError(INVALID_PARAMS, f"Unknown tool: {name}")
        if not isinstance(args, dict):
            return _tool_error("arguments must be a JSON object")
        try:
            result = call_tool(self.session, name, args)
        except ToolError as exc:
            return _tool_error(str(exc))
        except Exception as exc:
            log(f"tool {name} failed:", traceback.format_exc())
            return _tool_error(f"{type(exc).__name__}: {exc}")
        if self.autosave and name in MUTATING_TOOLS and self.session.path:
            try:
                call_tool(self.session, "save_project", {})
            except Exception as exc:  # pragma: no cover - disk errors
                log("autosave failed:", exc)
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}], "isError": False}


class _RPCError(Exception):
    def __init__(self, code: int, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _error(mid, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": mid, "error": {"code": code, "message": message}}


def _tool_error(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": True}


def serve(server: MCPServer, stdin: BinaryIO, stdout: BinaryIO):
    """Read newline-delimited JSON-RPC from ``stdin`` and write responses to ``stdout`` until EOF."""
    for raw in stdin:
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError as exc:
            resp: Any = _error(None, PARSE_ERROR, f"parse error: {exc}")
        else:
            resp = server.handle(msg)
        if resp is not None:
            stdout.write((json.dumps(resp, ensure_ascii=False) + "\n").encode("utf-8"))
            stdout.flush()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="synth2-mcp", description="Synth2 Vocal Studio MCP server (stdio)")
    ap.add_argument("--project", help="project file to load (created on first save if it does not exist)")
    ap.add_argument("--autosave", action="store_true", help="save the project after every editing tool call")
    args = ap.parse_args(argv)

    # Keep stdout clean: anything printed by library code goes to stderr.
    proto_out = sys.stdout.buffer
    sys.stdout = sys.stderr

    session = None
    if args.project:
        path = Path(args.project)
        if path.exists():
            try:
                session = Session.open(path)
                log(f"loaded {path}")
            except Exception as exc:
                log(f"could not load {path}: {exc}")
                return 2
        else:
            session = Session(path=path)
            log(f"new project, will save to {path}")
    server = MCPServer(session, autosave=args.autosave)
    log(f"Synth2 MCP server {__version__} ready ({len(TOOLS)} tools)")
    try:
        serve(server, sys.stdin.buffer, proto_out)
    except KeyboardInterrupt:  # pragma: no cover
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
