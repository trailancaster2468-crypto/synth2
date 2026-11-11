"""``synth2 models``: list, install, add, remove and locate neural models.

``add_parser(subparsers)`` registers the command on the main parser and ``run(args)`` carries
it out and returns the exit status.
"""
from __future__ import annotations

import json
import sys

from .. import assets
from . import models


def add_parser(subparsers):
    p = subparsers.add_parser(
        "models", help="list, install and remove neural voicebanks and vocoders",
        description="Neural voicebanks and vocoders. Weights are downloaded from their official "
                    "release pages, checked against a pinned SHA-256 and kept in the per-user "
                    "models directory (synth2 models path).")
    sub = p.add_subparsers(dest="models_cmd", required=True)

    ls = sub.add_parser("list", help="installed models and the ones the registry offers")
    ls.add_argument("--json", action="store_true", help="print JSON")

    ins = sub.add_parser("install", help="download and install models (default: the default "
                                         "voicebank and its vocoder)")
    ins.add_argument("ids", nargs="*", metavar="ID")
    ins.add_argument("--accept-license", action="store_true",
                     help="accept the licence terms of the models without being asked")
    ins.add_argument("--from-file", metavar="ARCHIVE",
                     help="use this local archive instead of downloading")
    ins.add_argument("--force", action="store_true", help="reinstall over an existing install")
    ins.add_argument("-q", "--quiet", action="store_true")

    add = sub.add_parser("add", help="register a voicebank or vocoder folder (or zip) you have")
    add.add_argument("folder", metavar="FOLDER")
    add.add_argument("--id")
    add.add_argument("--kind", choices=models.KINDS)
    add.add_argument("--name")
    add.add_argument("--copy", action="store_true",
                     help="copy the folder into the models directory instead of pointing at it")

    rm = sub.add_parser("remove", help="delete an installed model")
    rm.add_argument("id", metavar="ID")

    sub.add_parser("path", help="print the models directory")
    p.set_defaults(func=run)
    return p


def _error(msg: str) -> int:
    print(f"error: {msg}", file=sys.stderr)
    return 2


def listing() -> dict:
    """Installed models and the registry models that are not installed."""
    have = models.installed()
    ids = {m.id for m in have}
    return {"models_dir": str(models.models_dir()),
            "installed": [m.summary() for m in have],
            "available": [s.summary() for s in models.REGISTRY.values() if s.id not in ids]}


def _print_model(row: dict, state: str) -> None:
    version = f" {row['version']}" if row.get("version") else ""
    print(f"{row['id']:18s} {row['kind']:10s} {state:14s} {row['name']}{version}")
    print(f"{'':18s} licence: {row.get('license') or '(not stated)'}; commercial use "
          f"{assets.commercial_label(row.get('commercial_use'))}")
    for c in row.get("credits") or []:
        print(f"{'':18s} credit: {c}")


def _list(args) -> int:
    data = listing()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
        return 0
    print(f"models directory: {data['models_dir']}")
    if not data["installed"]:
        print("no models are installed")
    for row in data["installed"]:
        _print_model(row, "installed")
    for row in data["available"]:
        _print_model(row, f"not installed ({assets.format_size(row.get('size'))})")
    if data["available"]:
        print("install with: synth2 models install [ID ...]")
    return 0


def _install(args) -> int:
    ids = list(args.ids)
    if args.from_file and len(ids) > 1:
        return _error("--from-file installs one model: give one id")
    if args.from_file and not ids:
        spec = models.spec_for_archive(args.from_file)
        if spec is None:
            # not a registry archive: a model of the user's own, unpacked into the models folder
            try:
                model = models.add_local(args.from_file, copy=True, force=args.force)
            except assets.AssetError as exc:
                return _error(str(exc))
            print(f"installed {model.id} ({model.kind}) from {args.from_file}")
            print(f"  {models.LOCAL_TERMS}")
            return 0
        ids = [spec.id]
    ids = ids or models.default_set()
    for mid in ids:
        if mid not in models.REGISTRY:
            return _error(f"unknown model {mid!r} (the registry has: "
                          f"{', '.join(sorted(models.REGISTRY))})")
    todo = [models.REGISTRY[m] for m in ids
            if args.force or models.read_manifest(models.models_dir() / m) is None]
    accepted = bool(args.accept_license)
    if todo and not accepted:
        for spec in todo:
            print(assets.license_text(spec.name, spec.license, spec.terms, spec.credits,
                                      spec.license_url, spec.commercial_use), file=sys.stderr)
        answer = assets.confirm("Accept these terms and install? [y/N]")
        if answer is None:
            return _error("the licence terms above must be accepted: run again with "
                          "--accept-license")
        if not answer:
            return _error("nothing installed (terms not accepted)")
        accepted = True
    progress = None if args.quiet else assets.terminal_progress()
    for mid in ids:
        try:
            got = models.install(mid, accept_license=accepted, source=args.from_file,
                                 progress=progress, force=args.force)
        except assets.AssetError as exc:
            return _error(str(exc))
        for m in got:
            print(f"installed {m.id} ({m.kind}) in {m.path}")
            for c in m.credits:
                print(f"  credit: {c}")
    return 0


def _add(args) -> int:
    try:
        model = models.add_local(args.folder, args.id, args.kind, name=args.name, copy=args.copy)
    except assets.AssetError as exc:
        return _error(str(exc))
    print(f"added {model.id} ({model.kind}): {model.root}")
    print(f"  {models.LOCAL_TERMS}")
    return 0


def _remove(args) -> int:
    if not models.remove(args.id):
        return _error(f"no model {args.id!r} is installed in {models.models_dir()}")
    print(f"removed {args.id}")
    return 0


def run(args) -> int:
    cmd = args.models_cmd
    if cmd == "list":
        return _list(args)
    if cmd == "install":
        return _install(args)
    if cmd == "add":
        return _add(args)
    if cmd == "remove":
        return _remove(args)
    print(models.models_dir())
    return 0
