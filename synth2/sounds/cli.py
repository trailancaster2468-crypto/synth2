"""``synth2 sounds``: list, install, add, remove and locate sound sets.

``add_parser(subparsers)`` registers the command on the main parser and ``run(args)`` carries
it out and returns the exit status.
"""
from __future__ import annotations

import json
import sys

from .. import assets
from . import KINDS, installed, sounds_dir


def add_parser(subparsers):
    p = subparsers.add_parser(
        "sounds", help="list, install and remove sampled instruments, drum kits and impulses",
        description="Sound sets: soundfonts, SFZ instruments, drum kits, sample folders and "
                    "impulse responses. They are downloaded from their publishers, checked "
                    "against a pinned SHA-256 and kept in the per-user sounds directory "
                    "(synth2 sounds path). Each set keeps its own licence.")
    sub = p.add_subparsers(dest="sounds_cmd", required=True)

    ls = sub.add_parser("list", help="installed sets and the ones the registry offers")
    ls.add_argument("--json", action="store_true", help="print JSON")
    ls.add_argument("--kind", choices=KINDS)

    ins = sub.add_parser("install", help="download and install sound sets")
    ins.add_argument("ids", nargs="*", metavar="ID")
    ins.add_argument("--all", action="store_true",
                     help="every registry set whose publisher states that commercial use is "
                          "allowed and that is not share-alike")
    ins.add_argument("--accept-license", action="store_true",
                     help="accept the licence terms of the sets without being asked")
    ins.add_argument("--from-file", metavar="PATH",
                     help="use this local archive (or folder of source files) instead of "
                          "downloading")
    ins.add_argument("--force", action="store_true", help="reinstall over an existing install")
    ins.add_argument("-q", "--quiet", action="store_true")

    add = sub.add_parser("add", help="register a soundfont, an SFZ instrument or a folder of "
                                     "samples you already have")
    add.add_argument("path", metavar="PATH")
    add.add_argument("--id")
    add.add_argument("--kind", choices=KINDS)
    add.add_argument("--name")
    add.add_argument("--copy", action="store_true",
                     help="copy the files into the sounds directory instead of pointing at them")
    add.add_argument("--license", default="", help="the licence of these sounds, if you know it")
    add.add_argument("--credit", action="append", default=[],
                     help="a credit line to show when publishing (repeatable)")

    rm = sub.add_parser("remove", help="delete an installed sound set")
    rm.add_argument("id", metavar="ID")

    sub.add_parser("path", help="print the sounds directory")
    p.set_defaults(func=run)
    return p


def _error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def listing(kind: str | None = None) -> dict:
    """Installed sets and the registry sets that are not installed."""
    from . import installer

    have = installed(kind)
    return {"sounds_dir": str(sounds_dir()),
            "installed": [s.summary() for s in have],
            "available": [row for row in installer.available()
                          if not row["installed"] and (kind is None or row["kind"] == kind)]}


def _licence_line(row: dict) -> str:
    text = f"licence: {row.get('license') or '(not stated)'}; commercial use " \
           f"{assets.commercial_label(row.get('commercial_use'))}"
    if row.get("share_alike"):
        text += "; share-alike"
    return text


def _print_set(row: dict, state: str) -> None:
    version = f" {row['version']}" if row.get("version") else ""
    print(f"{row['id']:26s} {row['kind']:10s} {assets.format_size(row.get('size')):>9s}  "
          f"{state:13s} {row['name']}{version}")
    pad = " " * 26
    print(f"{pad} {_licence_line(row)}")
    if row.get("commercial_use_statement"):
        print(f"{pad} the publisher says: \"{row['commercial_use_statement']}\"")
    for c in row.get("credits") or []:
        print(f"{pad} credit: {c}")


def _list(args) -> int:
    data = listing(args.kind)
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    print(f"sounds directory: {data['sounds_dir']}")
    if not data["installed"]:
        print("no sound sets are installed")
    for row in data["installed"]:
        _print_set(row, "installed")
    for row in data["available"]:
        _print_set(row, "not installed")
    if data["available"]:
        print("install with: synth2 sounds install ID [ID ...]")
    return 0


def _install(args) -> int:
    from . import installer

    registry = installer.REGISTRY
    ids = list(args.ids)
    if args.all:
        ids += [s.id for s in registry.values()
                if s.commercial_use is True and not s.share_alike and s.id not in ids]
    if not ids:
        return _error("name the sets to install (synth2 sounds list shows them), or use --all")
    if args.from_file and len(ids) != 1:
        return _error("--from-file installs one set: give one id")
    for sid in ids:
        if sid not in registry:
            return _error(f"unknown sound set {sid!r} (synth2 sounds list shows the registry)")
    todo = [registry[s] for s in ids
            if args.force or installer.read_manifest(sounds_dir() / s) is None]
    accepted = bool(args.accept_license)
    if todo and not args.quiet:
        print(f"{len(todo)} set(s) to download, {assets.format_size(sum(s.size for s in todo))}",
              file=sys.stderr)
    if todo and not accepted:
        for spec in todo:
            print(assets.license_text(spec.name, spec.license, spec.terms, spec.credits,
                                      spec.license_url or spec.license_source,
                                      spec.commercial_use, spec.commercial_use_statement),
                  file=sys.stderr)
        answer = assets.confirm("Accept these terms and install? [y/N]")
        if answer is None:
            return _error("the licence terms above must be accepted: run again with "
                          "--accept-license")
        if not answer:
            return _error("nothing installed (terms not accepted)")
        accepted = True
    progress = None if args.quiet else assets.terminal_progress()
    status = 0
    for sid in ids:
        try:
            s = installer.install(sid, accept_license=accepted, source=args.from_file,
                                  progress=progress, force=args.force)
        except assets.AssetError as exc:
            status = _error(f"{sid}: {exc}")
            continue
        print(f"installed {s.id} ({s.kind}, {assets.format_size(s.size)}) in {s.path}")
        for c in s.credits:
            print(f"  credit: {c}")
    return status


def _add(args) -> int:
    from . import installer

    try:
        s = installer.add_local(args.path, args.id, args.kind, name=args.name, copy=args.copy,
                                license=args.license, credits=args.credit)
    except assets.AssetError as exc:
        return _error(str(exc))
    print(f"added {s.id} ({s.kind}): {s.root}")
    if not args.license:
        print(f"  {installer.LOCAL_TERMS}")
    return 0


def _remove(args) -> int:
    from . import installer

    if not installer.remove(args.id):
        return _error(f"no sound set {args.id!r} is installed in {sounds_dir()}")
    print(f"removed {args.id}")
    return 0


def run(args) -> int:
    cmd = args.sounds_cmd
    if cmd == "list":
        return _list(args)
    if cmd == "install":
        return _install(args)
    if cmd == "add":
        return _add(args)
    if cmd == "remove":
        return _remove(args)
    print(sounds_dir())
    return 0
