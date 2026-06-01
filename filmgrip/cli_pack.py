"""``film-grip pack`` — list, show, and apply named edit recipes (packs).

Deterministic packs compile to a typed EditPlan and run through the exact validate→apply pipeline
``film-grip edit`` uses — fixture mode (offline, provable) or live Resolve. Exit codes match the
rest of the CLI: 0 ok, 1 bad input / rejected plan, 2 editor unreachable, 3 unsupported.
"""
from __future__ import annotations

import os

from .core.ir import TimelineIR


def _emit(text: str) -> None:
    print(text)


def cmd_pack(args) -> int:
    action = getattr(args, "pack_action", "list")
    if action == "show":
        return _show(args)
    if action == "apply":
        return _apply(args)
    return _list()


def _list() -> int:
    from .packs import all_packs

    _emit("film-grip packs:")
    for p in all_packs():
        _emit(f"  {p.name:14} [{p.kind:13}] {p.description}  ({p.source})")
    _emit("\napply one: film-grip pack apply <name> [--fixture cut.otio --dry-run]")
    return 0


def _show(args) -> int:
    from .packs import PackError, get_pack

    if not getattr(args, "name", ""):
        _emit("error: which pack? e.g. film-grip pack show punch-up")
        return 1
    try:
        p = get_pack(args.name)
    except PackError as exc:
        _emit(f"error: {exc}")
        return 1
    _emit(f"{p.name} [{p.kind}]  ({p.source})")
    _emit(f"  {p.description}")
    if p.params:
        _emit(f"  params: {p.params}")
    if p.kind == "prompt" and p.prompt:
        _emit(f"  prompt: {p.prompt}")
    return 0


def _apply(args) -> int:
    from .packs import PackError, get_pack

    if not getattr(args, "name", ""):
        _emit("error: which pack? e.g. film-grip pack apply marker-pass --fixture cut.otio --dry-run")
        return 1
    try:
        pack = get_pack(args.name)
    except PackError as exc:
        _emit(f"error: {exc}")
        return 1
    if pack.kind != "deterministic":
        _emit(f"error: '{pack.name}' is a {pack.kind} pack — prompt packs run through the planner; "
              f"use `film-grip edit` (offline prompt-pack apply lands in D7).")
        return 3
    return _apply_fixture(args, pack) if args.fixture else _apply_live(args, pack)


def _apply_fixture(args, pack) -> int:
    from .packs import PackError
    from .packs.engine import compile_pack
    from .protocol.validate import dry_run, validate

    ir = TimelineIR.from_otio_file(args.fixture)
    ids = ([s.strip() for s in args.select.split(",") if s.strip()]
           if getattr(args, "select", None) else [c.id for c in ir.real_clips()])
    try:
        plan = compile_pack(pack, ir, ids)
    except PackError as exc:
        _emit(f"error: {exc}")
        return 1
    if not plan.ops:
        _emit(f"# pack '{pack.name}': nothing to do for this selection.")
        return 0
    if args.dry_run:
        _emit(dry_run(plan, ir))
        return 0 if validate(plan, ir).ok else 1

    from .adapters.interchange import InterchangeAdapter

    out = args.out or (os.path.splitext(args.fixture)[0] + ".edited" + os.path.splitext(args.fixture)[1])
    res = InterchangeAdapter().apply(plan, args.fixture, out_path=out)
    _emit(res.diff if res.ok else "apply failed:\n  " + "\n  ".join(res.errors))
    for w in res.warnings:
        _emit(f"  ⚠ {w}")
    return 0 if res.ok else 1


def _apply_live(args, pack) -> int:
    if getattr(args, "editor", "resolve") != "resolve":
        _emit("error: live pack apply supports --editor resolve; otherwise use --fixture FILE.")
        return 3

    from .adapters import resolve_client as rc
    from .adapters.resolve_adapter import ResolveAdapter
    from .cli_edit import _preflight_message
    from .packs.engine import compile_pack
    from .protocol.validate import dry_run, validate

    report = rc.preflight()
    if not report["app_running"]:
        _emit(_preflight_message(report))
        return 2

    adapter = ResolveAdapter()
    session = rc.connect()
    ir = adapter.snapshot(session)
    sel = adapter.get_selection(session, ir)
    plan = compile_pack(pack, ir, sel.ids)
    if not plan.ops:
        _emit(f"# pack '{pack.name}': nothing to do for the current selection "
              f"({len(sel.ids)} clip(s) [{getattr(sel, 'confidence', 'precise')}]).")
        return 0
    if args.dry_run:
        _emit(dry_run(plan, ir))
        return 0 if validate(plan, ir).ok else 1
    res = adapter.apply(plan, session)
    _emit(res.diff)
    for w in res.warnings:
        _emit(f"  ⚠ {w}")
    if not res.ok:
        _emit("apply failed:\n  " + "\n  ".join(res.errors))
        return 1
    return 0
