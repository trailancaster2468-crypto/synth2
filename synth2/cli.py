"""Command line interface: ``synth2 <command>``."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _load(path: str):
    from .io import load_any

    return load_any(path)


def _error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def _track_ids(project, refs) -> list[str]:
    """Resolve --track names/ids/indexes exactly like the tool layer (raises ToolError for unknown tracks)."""
    from .ai.tools import Session, _track

    session = Session(project)
    return [_track(session, r).id for r in refs]


def _filter_to_range(pa: dict, start, end) -> dict:
    """Keep only notes inside the rendered beat range [start, end) (same rule as the analyze_render tool)."""
    if start is None and end is None:
        return pa
    lo = -math.inf if start is None else start
    hi = math.inf if end is None else end
    for tr in pa.get("tracks", []):
        for key in ("worst", "unvoiced_notes"):
            if isinstance(tr.get(key), list):
                tr[key] = [w for w in tr[key] if lo - 1e-9 <= float(w.get("beat", lo)) < hi]
    return pa


def cmd_render(a):
    from .ai.tools import ToolError
    from .analysis import pitch_accuracy
    from .engine import render_project

    if a.start is not None and a.end is not None and a.end <= a.start:
        return _error("--end must be greater than --start")
    project = _load(a.input)
    try:
        track_ids = _track_ids(project, a.track) if a.track else None
    except ToolError as exc:
        return _error(str(exc))
    out = Path(a.output or Path(a.input).with_suffix(".wav"))

    def progress(f, msg):
        if not a.quiet:
            print(f"\r[{int(f * 100):3d}%] {msg:40s}", end="", file=sys.stderr, flush=True)

    res = render_project(project, sample_rate=a.sample_rate, start_beat=a.start, end_beat=a.end,
                         track_ids=track_ids, progress=progress)
    if not a.quiet:
        print(file=sys.stderr)
    res.save(out, bits=a.bits)
    if a.stems:
        res.save_stems(a.stems, bits=a.bits)
    report = dict(res.report)
    for t in report.get("tracks", []):
        t.pop("phonemes", None)
    if a.check:
        report["pitch_accuracy"] = _filter_to_range(pitch_accuracy(project, res), a.start, a.end)
    print(json.dumps({"output": str(out), **report}, indent=2, ensure_ascii=False))


def cmd_new(a):
    from .project import Project

    p = Project.new(a.title, bpm=a.bpm, voice=a.voice, language=a.language)
    out = Path(a.output)
    from .io import export_any

    export_any(p, out)
    print(out)


def cmd_info(a):
    p = _load(a.input)
    print(json.dumps(p.summary(), indent=2, ensure_ascii=False))


def cmd_score(a):
    from .score import project_to_score

    print(project_to_score(_load(a.input)))


def cmd_convert(a):
    from .io import export_any

    p = _load(a.input)
    print(export_any(p, a.output, fmt=a.format))


def cmd_voices(a):
    from .voices import list_voices

    for v in list_voices():
        print(f"{v['id']:8s} {v['range']:9s} {v['default_style']:9s} {v['description']}")


def cmd_styles(a):
    from .engine.expression import list_styles

    for s in list_styles():
        print(f"{s['name']:10s} {s['description']}")


def cmd_phonemize(a):
    from .phonemes import lyric_phonemes, split_lyrics

    for tok in split_lyrics(a.text, a.language):
        if tok in ("+", "-"):
            print(f"{tok:12s} (continuation)" if tok == "+" else f"{tok:12s} (melisma)")
        else:
            print(f"{tok:12s} {' '.join(lyric_phonemes(tok, a.language))}")


def cmd_analyze(a):
    from .analysis import analyze_audio
    from .io.wav import read_wav

    audio, sr = read_wav(a.input)
    print(json.dumps(analyze_audio(audio, sr), indent=2))


def cmd_studio(a):
    from .server.app import run

    run(host=a.host, port=a.port, project_path=a.project, open_browser=a.open)


def cmd_mcp(a):
    from .ai import mcp_server

    return mcp_server.main((["--project", a.project] if a.project else []) + (["--autosave"] if a.autosave else []))


def cmd_director(a):
    from .ai.director import Director, DirectorUnavailable
    from .ai.tools import Session, ToolError

    try:
        return _director_main(a, Director, Session)
    except (DirectorUnavailable, ToolError) as exc:
        return _error(str(exc))


def _director_turn(director, msg: str) -> bool:
    """Send one message; API errors are reported and do not end the session."""
    try:
        out = director.chat(msg)
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return False
    for act in out.get("actions", []):
        print(f"  [{'ok' if act.get('ok') else 'error'}] {act.get('tool')}", file=sys.stderr)
    print(out.get("reply", ""))
    return True


def _director_main(a, Director, Session) -> int:
    from .ai.tools import call_tool

    session = Session()
    target = a.project
    create = bool(a.project) and not Path(a.project).exists()
    if a.project and not create:
        call_tool(session, "load_project", {"path": a.project})
        # the session path is the file itself for native projects and <stem>.s2p.json next to an imported
        # .svp/.mid/.musicxml/.ust, which is never overwritten with a lossy re-export
        target = session.path
        if Path(target) != Path(a.project):
            print(f"note: {a.project} is imported; changes are saved to {target}", file=sys.stderr)
    before = session.project.to_dict()
    director = Director(session, model=a.model)
    ok = True
    try:
        if a.message:
            ok = _director_turn(director, a.message)
        else:
            while True:
                try:
                    msg = input("you> ").strip()
                except (EOFError, KeyboardInterrupt):
                    print(file=sys.stderr)
                    break
                if msg in ("exit", "quit", ""):
                    break
                try:
                    _director_turn(director, msg)
                except KeyboardInterrupt:
                    print("\n(interrupted)", file=sys.stderr)
    finally:
        # always keep the session's work, also after an API error or Ctrl-C
        if a.project and (create or session.project.to_dict() != before):
            saved = call_tool(session, "save_project", {"path": target})
            print(f"saved {saved['path']}", file=sys.stderr)
    return 0 if ok else 1


def cmd_demo(a):
    from .demo import render_demos

    for path in render_demos(Path(a.output)):
        print(path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="synth2", description="Synth2 Vocal Studio - AI-native singing synthesizer")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="render a project (.json/.s2s/.svp/.mid/.musicxml/.ust) to WAV")
    r.add_argument("input")
    r.add_argument("-o", "--output")
    r.add_argument("--bits", type=int, default=16, choices=(16, 24, 32))
    r.add_argument("--sample-rate", type=int)
    r.add_argument("--start", type=float, help="start beat")
    r.add_argument("--end", type=float, help="end beat")
    r.add_argument("--track", action="append", help="render only these tracks (name/id/index)")
    r.add_argument("--stems", help="also write per-track stems into this directory")
    r.add_argument("--check", action="store_true", help="include a pitch-accuracy analysis in the report")
    r.add_argument("-q", "--quiet", action="store_true")
    r.set_defaults(func=cmd_render)

    n = sub.add_parser("new", help="create an empty project")
    n.add_argument("output")
    n.add_argument("--title", default="Untitled")
    n.add_argument("--bpm", type=float, default=120)
    n.add_argument("--voice", default="aurora")
    n.add_argument("--language", default="en")
    n.set_defaults(func=cmd_new)

    i = sub.add_parser("info", help="print a project summary")
    i.add_argument("input")
    i.set_defaults(func=cmd_info)

    s = sub.add_parser("score", help="print a project as score notation")
    s.add_argument("input")
    s.set_defaults(func=cmd_score)

    c = sub.add_parser("convert", help="convert between project formats")
    c.add_argument("input")
    c.add_argument("output")
    c.add_argument("--format")
    c.set_defaults(func=cmd_convert)

    sub.add_parser("voices", help="list voices").set_defaults(func=cmd_voices)
    sub.add_parser("styles", help="list expression styles").set_defaults(func=cmd_styles)

    ph = sub.add_parser("phonemize", help="show how lyrics will be pronounced")
    ph.add_argument("text")
    ph.add_argument("--language", default="en")
    ph.set_defaults(func=cmd_phonemize)

    an = sub.add_parser("analyze", help="analyse a WAV file (levels, pitch range, voicing)")
    an.add_argument("input")
    an.set_defaults(func=cmd_analyze)

    st = sub.add_parser("studio", help="start the web studio")
    st.add_argument("--host", default="127.0.0.1")
    st.add_argument("--port", type=int, default=8765)
    st.add_argument("--project")
    st.add_argument("--open", action="store_true", help="open a browser")
    st.set_defaults(func=cmd_studio)

    m = sub.add_parser("mcp", help="run the MCP server on stdio (for Claude and other AI clients)")
    m.add_argument("--project")
    m.add_argument("--autosave", action="store_true", help="save the project after every editing tool call")
    m.set_defaults(func=cmd_mcp)

    d = sub.add_parser("director", help="produce vocals from natural language (needs ANTHROPIC_API_KEY)")
    d.add_argument("message", nargs="?")
    d.add_argument("--project", help="project file to load and save")
    d.add_argument("--model")
    d.set_defaults(func=cmd_director)

    dm = sub.add_parser("demo", help="render the built-in demo songs")
    dm.add_argument("-o", "--output", default="out/demo")
    dm.set_defaults(func=cmd_demo)

    from .neural import cli as models_cli
    from .sounds import cli as sounds_cli

    models_cli.add_parser(sub)
    sounds_cli.add_parser(sub)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    code = args.func(args) or 0
    if code:
        sys.exit(code)
    return 0


if __name__ == "__main__":
    main()
