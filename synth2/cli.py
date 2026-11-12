"""Command line interface: ``synth2 <command>``."""
from __future__ import annotations

import argparse
import json
import math
import os
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


def _engine_arg(value: str) -> str:
    """argparse type of ``--engine``: auto, formant, neural or neural:<voicebank id>."""
    from .engine.backend import parse_engine

    try:
        parse_engine(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    return value.strip()


def _add_engine_options(parser, what: str = "this render") -> None:
    from .engine.backend import QUALITY_CHOICES

    parser.add_argument("--engine", type=_engine_arg, metavar="ENGINE",
                        help=f"singing engine for {what}: auto, formant, neural or neural:<voicebank id> "
                             "(see 'synth2 engines'); overrides the project's and the tracks' own")
    parser.add_argument("--quality", choices=QUALITY_CHOICES,
                        help="render quality of the neural engine: draft (fast), normal (default), high (final)")


def _engine_problem(engine: str | None) -> str | None:
    """Why ``engine`` cannot be used on this machine (None when it can, or when it is unset or auto)."""
    from .engine.backend import EngineUnavailable, resolve_engine

    try:
        resolve_engine(override=engine)
    except (EngineUnavailable, ValueError) as exc:
        return str(exc)
    return None


def _set_default_engine(a) -> str | None:
    """Make ``--engine`` / ``--quality`` the defaults of this process (the studio, the MCP server and the
    director render through the tool layer, which reads them from the environment). Returns an error
    message when the engine cannot run here."""
    from .engine.backend import ENGINE_ENV, QUALITY_ENV

    engine, quality = getattr(a, "engine", None), getattr(a, "quality", None)
    if engine:
        problem = _engine_problem(engine)
        if problem:
            return problem
        os.environ[ENGINE_ENV] = engine
    if quality:
        os.environ[QUALITY_ENV] = quality
    return None


def cmd_render(a):
    from .ai.tools import ToolError, render_notes, slim_report, summary_lines
    from .analysis import pitch_accuracy
    from .engine import render_project
    from .engine.backend import EngineUnavailable, current_quality

    if a.start is not None and a.end is not None and a.end <= a.start:
        return _error("--end must be greater than --start")
    try:
        project = _load(a.input)
    except (OSError, ValueError, KeyError) as exc:
        return _error(f"could not load {a.input}: {exc}")
    try:
        track_ids = _track_ids(project, a.track) if a.track else None
    except ToolError as exc:
        return _error(str(exc))
    out = Path(a.output or Path(a.input).with_suffix(".wav"))

    def progress(f, msg):
        if not a.quiet:
            print(f"\r[{int(f * 100):3d}%] {msg:40s}", end="", file=sys.stderr, flush=True)

    try:
        res = render_project(project, sample_rate=a.sample_rate, start_beat=a.start, end_beat=a.end,
                             track_ids=track_ids, progress=progress, engine=a.engine, quality=a.quality)
    except (EngineUnavailable, ValueError) as exc:  # an engine that is not installed, or a misspelt one in the file
        return _error(str(exc))
    if not a.quiet:
        print(file=sys.stderr)
    res.save(out, bits=a.bits)
    if a.stems:
        res.save_stems(a.stems, bits=a.bits)
    report = slim_report(res.report)
    notes = render_notes(project, report)
    report.update(warnings=notes["warnings"], credits=notes["credits"], licence_notes=notes["licence_notes"])
    if a.check:
        report["pitch_accuracy"] = _filter_to_range(pitch_accuracy(project, res), a.start, a.end)
    print(json.dumps({"output": str(out), "quality": a.quality or current_quality(), **report},
                     indent=2, ensure_ascii=False))
    if not a.quiet:
        print(f"wrote {out}", file=sys.stderr)
        for line in summary_lines(notes):
            print(line, file=sys.stderr)


def cmd_new(a):
    from .project import Project

    p = Project.new(a.title, bpm=a.bpm, voice=a.voice, language=a.language)
    out = Path(a.output)
    from .io import export_any

    export_any(p, out)
    print(out)


def cmd_info(a):
    from .ai.tools import resolved_engines

    p = _load(a.input)
    out = p.summary()
    out["engine"] = p.engine or "auto"
    out["engines"] = resolved_engines(p)
    for row, t in zip(out.get("tracks", []), p.tracks):
        if t.role:
            row["role"] = t.role
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_score(a):
    from .score import project_to_score

    print(project_to_score(_load(a.input)))


def cmd_convert(a):
    from .io import export_any

    p = _load(a.input)
    print(export_any(p, a.output, fmt=a.format))


def cmd_voices(a):
    from .ai.tools import voices_info

    voices = voices_info()
    if a.json:
        print(json.dumps(voices, indent=2, ensure_ascii=False))
        return
    for v in voices:
        print(f"{v['id']:8s} {v['range']:9s} {v['default_style']:9s} {v['description']}")
        if v.get("neural"):
            print(f"{'':8s} neural engine: {v['neural']['summary']}")
    if voices and voices[0].get("neural") and not voices[0]["neural"]["available"]:
        print("the neural engine is not available on this machine: see 'synth2 engines'")


def _print_model(row: dict) -> None:
    from . import assets

    version = f" {row['version']}" if row.get("version") else ""
    default = " (default)" if row.get("default") else ""
    langs = f"; languages: {', '.join(row['languages'])}" if row.get("languages") else ""
    print(f"  {row['id']:18s} {row['name']}{version}{default}{langs}")
    print(f"  {'':18s} licence: {row.get('license') or '(not stated)'}; commercial use "
          f"{assets.commercial_label(row.get('commercial_use'))}")
    for t in row.get("terms") or []:
        print(f"  {'':18s} terms: {t}")
    for c in row.get("credits") or []:
        print(f"  {'':18s} credit: {c}")


def cmd_engines(a):
    from .ai.tools import engines_info

    info = engines_info()
    if a.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return
    neural = info["neural"]
    print("Singing engines")
    for e in info["engines"]:
        state = "available" if e["available"] else "not available"
        print(f"  {e['name']:8s} {state:14s} {e['description']}")
        if not e["available"]:
            print(f"  {'':8s} reason: {e.get('reason')}")
            print(f"  {'':8s} install with: {e.get('install_hint')}")
    print("  choose with --engine auto | formant | neural | neural:<voicebank id>, 'engine:' in a score, "
          "or SYNTH2_ENGINE")
    env = "; ".join(f"{k}={v}" for k, v in info["environment"].items() if v)
    if info["default_engine"]:
        print(f"default (auto) on this machine: {info['default_engine']}" + (f"  [{env}]" if env else ""))
    else:
        print(f"default engine: unusable: {info.get('default_engine_problem')}")
    print("quality (--quality, neural engine): " + "; ".join(
        f"{q}{' (default)' if q == info['default_quality'] else ''}: {info['quality_notes'][q]}"
        for q in info["qualities"]))
    print(f"models directory: {neural.get('models_dir')}")
    print(f"onnxruntime: {neural.get('onnxruntime') or 'not installed'}")
    for title, key in (("voicebanks", "banks"), ("vocoders", "vocoders")):
        rows = neural.get(key) or []
        print(f"{title}:" + ("" if rows else " none installed"))
        for row in rows:
            _print_model(row)
    if info["credits"]:
        print("credits to show when you publish a render of the neural engine (default voicebank):")
        for c in info["credits"]:
            print(f"  - {c}")
    if info["licence_notes"]:
        print("licence notes:")
        for n in info["licence_notes"]:
            print(f"  - {n}")


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

    problem = _set_default_engine(a)
    if problem:
        return _error(problem)
    run(host=a.host, port=a.port, project_path=a.project, open_browser=a.open)


def cmd_mcp(a):
    from .ai import mcp_server

    problem = _set_default_engine(a)
    if problem:
        return _error(problem)
    return mcp_server.main((["--project", a.project] if a.project else []) + (["--autosave"] if a.autosave else []))


def cmd_director(a):
    from .ai.director import Director, DirectorUnavailable
    from .ai.tools import Session, ToolError

    problem = _set_default_engine(a)
    if problem:
        return _error(problem)
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
    from .ai.tools import ToolError, call_tool, summary_lines
    from .demo import DEMOS, build_demo

    problem = _engine_problem(a.engine)
    if problem:
        return _error(problem)
    out_dir = Path(a.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    args = {k: v for k, v in (("engine", a.engine), ("quality", a.quality)) if v}
    shown = None
    for name in DEMOS:
        session = build_demo(name)
        try:
            res = call_tool(session, "render", {"path": str(out_dir / f"{name}.wav"), **args})
        except ToolError as exc:
            return _error(str(exc))
        call_tool(session, "save_project", {"path": str(out_dir / f"{name}.s2p.json")})
        print(res.get("path") or out_dir / f"{name}.wav")
        lines = summary_lines({k: res.get(k) for k in ("engines", "voicebanks", "credits", "licence_notes")})
        if lines != shown:  # the demos share an engine: say it once
            shown = lines
            for line in lines:
                print(line, file=sys.stderr)


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
    _add_engine_options(r)
    r.add_argument("-q", "--quiet", action="store_true",
                   help="no progress and no summary on stderr (the JSON report is still printed)")
    r.set_defaults(func=cmd_render)

    n = sub.add_parser("new", help="create an empty project")
    n.add_argument("output")
    n.add_argument("--title", default="Untitled")
    n.add_argument("--bpm", type=float, default=120)
    n.add_argument("--voice", default="aurora")
    n.add_argument("--language", default="en")
    n.set_defaults(func=cmd_new)

    i = sub.add_parser("info", help="print a project summary (with the engine that sings each track here)")
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

    v = sub.add_parser("voices", help="list voices and what each becomes on the neural engine")
    v.add_argument("--json", action="store_true", help="print JSON")
    v.set_defaults(func=cmd_voices)
    e = sub.add_parser("engines", help="singing engines: which can run here, installed voicebanks, credits")
    e.add_argument("--json", action="store_true", help="print JSON")
    e.set_defaults(func=cmd_engines)
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
    _add_engine_options(st, "the studio's renders (the default when the project and the page choose none)")
    st.set_defaults(func=cmd_studio)

    m = sub.add_parser("mcp", help="run the MCP server on stdio (for Claude and other AI clients)")
    m.add_argument("--project")
    m.add_argument("--autosave", action="store_true", help="save the project after every editing tool call")
    _add_engine_options(m, "the server's renders (the default when a project or a tool call chooses none)")
    m.set_defaults(func=cmd_mcp)

    d = sub.add_parser("director", help="produce vocals from natural language (needs ANTHROPIC_API_KEY)")
    d.add_argument("message", nargs="?")
    d.add_argument("--project", help="project file to load and save")
    d.add_argument("--model")
    _add_engine_options(d, "the director's renders (the default when the project chooses none)")
    d.set_defaults(func=cmd_director)

    dm = sub.add_parser("demo", help="render the built-in demo songs")
    dm.add_argument("-o", "--output", default="out/demo")
    _add_engine_options(dm, "the demo renders")
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
