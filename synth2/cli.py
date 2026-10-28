"""Command line interface: ``synth2 <command>``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load(path: str):
    from .io import load_any

    return load_any(path)


def cmd_render(a):
    from .analysis import pitch_accuracy
    from .engine import render_project

    project = _load(a.input)
    out = Path(a.output or Path(a.input).with_suffix(".wav"))

    def progress(f, msg):
        if not a.quiet:
            print(f"\r[{int(f * 100):3d}%] {msg:40s}", end="", file=sys.stderr, flush=True)

    res = render_project(project, sample_rate=a.sample_rate, start_beat=a.start, end_beat=a.end,
                         track_ids=a.track or None, progress=progress)
    if not a.quiet:
        print(file=sys.stderr)
    res.save(out, bits=a.bits)
    if a.stems:
        res.save_stems(a.stems, bits=a.bits)
    report = dict(res.report)
    for t in report.get("tracks", []):
        t.pop("phonemes", None)
    if a.check:
        report["pitch_accuracy"] = pitch_accuracy(project, res)
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

    sys.argv = ["synth2-mcp"] + (["--project", a.project] if a.project else [])
    mcp_server.main()


def cmd_director(a):
    from .ai.director import Director
    from .ai.tools import Session

    session = Session()
    if a.project and Path(a.project).exists():
        from .ai.tools import call_tool

        call_tool(session, "load_project", {"path": a.project})
    director = Director(session, model=a.model)
    messages = [a.message] if a.message else []
    interactive = not messages
    while True:
        if interactive:
            try:
                msg = input("you> ").strip()
            except EOFError:
                break
            if msg in ("exit", "quit", ""):
                break
            messages = [msg]
        if not messages:
            break
        out = director.chat(messages.pop(0))
        for act in out.get("actions", []):
            print(f"  [{'ok' if act.get('ok') else 'error'}] {act.get('tool')}", file=sys.stderr)
        print(out.get("reply", ""))
        if not interactive:
            break
    if a.project:
        from .ai.tools import call_tool

        call_tool(session, "save_project", {"path": a.project})


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
    m.set_defaults(func=cmd_mcp)

    d = sub.add_parser("director", help="produce vocals from natural language (needs ANTHROPIC_API_KEY)")
    d.add_argument("message", nargs="?")
    d.add_argument("--project", help="project file to load and save")
    d.add_argument("--model")
    d.set_defaults(func=cmd_director)

    dm = sub.add_parser("demo", help="render the built-in demo songs")
    dm.add_argument("-o", "--output", default="out/demo")
    dm.set_defaults(func=cmd_demo)
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
