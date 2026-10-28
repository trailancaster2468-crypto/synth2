"""Bundled demo productions, built through the same tool layer an AI agent uses."""
from __future__ import annotations

from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"

# (example file, extra tool calls applied after loading)
DEMOS = {
    "neon_hearts": [
        ("add_harmony", {"source_track": "Lead", "steps": -2, "voice": "atlas", "name": "Harmony",
                         "start_beat": 40, "gain_db": -5, "pan": 0.3}),
        ("add_harmony", {"source_track": "Lead", "steps": 2, "voice": "seraph", "name": "High",
                         "start_beat": 40, "gain_db": -8, "pan": -0.3}),
    ],
    "amazing_grace": [
        ("add_harmony", {"source_track": "Lead", "steps": -2, "voice": "ember", "name": "Low",
                         "style": "ballad", "gain_db": -4, "pan": 0.25}),
    ],
    "sakura": [],
}


def build_demo(name: str):
    """Return a tools Session holding the finished demo project."""
    from .ai.tools import Session, call_tool

    session = Session()
    call_tool(session, "load_project", {"path": str(EXAMPLES / f"{name}.s2s")})
    for tool, args in DEMOS[name]:
        call_tool(session, tool, args)
    return session


def render_demos(out_dir: Path, names=None) -> list[Path]:
    from .ai.tools import call_tool

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in names or DEMOS:
        session = build_demo(name)
        res = call_tool(session, "render", {"path": str(out_dir / f"{name}.wav")})
        paths.append(Path(res.get("path") or out_dir / f"{name}.wav"))
        call_tool(session, "save_project", {"path": str(out_dir / f"{name}.s2p.json")})
    return paths
