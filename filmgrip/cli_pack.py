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
        _emit("error: which pack? e.g. film-grip pack show marker-pass")
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
    if pack.kind == "prompt":
        return _apply_prompt(args, pack)
    if pack.kind != "deterministic":
        _emit(f"error: unknown pack kind '{pack.kind}' for '{pack.name}'.")
        return 1
    return _apply_fixture(args, pack) if args.fixture else _apply_live(args, pack)


def _apply_fixture(args, pack) -> int:
    from .cli_edit import FixtureError, load_fixture_ir
    from .packs import PackError
    from .packs.engine import compile_pack
    from .protocol.validate import dry_run, validate

    try:
        ir = load_fixture_ir(args.fixture)
    except FixtureError as exc:
        _emit(f"error: {exc}")
        return 1
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
    from .cli_edit import _emit_apply_result, run_verify

    out = args.out or (os.path.splitext(args.fixture)[0] + ".edited" + os.path.splitext(args.fixture)[1])
    res = InterchangeAdapter().apply(plan, args.fixture, out_path=out)
    if getattr(args, "verify", False) and res.ok:
        try:
            ir_after = load_fixture_ir(out)
            run_verify(res, ir, plan, ir_after)
        except Exception as exc:  # verification must never mask the apply result
            res.warnings.append(f"verify loop failed to run: {exc}")
    return _emit_apply_result(res)


def _format_prompt(pack) -> str:
    """Fill {placeholders} in a prompt pack from its params; fall back to the raw prompt."""
    try:
        return pack.prompt.format(**pack.params)
    except (KeyError, IndexError, ValueError):
        return pack.prompt


def _apply_prompt(args, pack) -> int:
    """Apply a prompt pack: hand its (parameterized) instruction to the active planner backend."""
    from .integration.backend import UnknownBackendError, get_backend
    from .integration.mcp_host import PlannerContext
    from .integration.repair import plan_with_repair
    from .protocol.validate import dry_run, validate

    prompt = _format_prompt(pack)
    try:
        transport = get_backend(getattr(args, "backend", None)).transport()
    except UnknownBackendError as exc:
        _emit(f"error: {exc}")
        return 1

    session = adapter = None
    if args.fixture:
        from .adapters.base import Selection
        from .cli_edit import FixtureError, load_fixture_ir

        try:
            ir = load_fixture_ir(args.fixture)
        except FixtureError as exc:
            _emit(f"error: {exc}")
            return 1
        ids = ([s.strip() for s in args.select.split(",") if s.strip()]
               if getattr(args, "select", None) else [c.id for c in ir.real_clips()])
        sel = Selection(ids=ids, basis="fixture")
    else:
        if getattr(args, "editor", "resolve") != "resolve":
            _emit("error: live pack apply supports --editor resolve; otherwise use --fixture FILE.")
            return 3
        from .adapters import resolve_client as rc
        from .adapters.resolve_adapter import ResolveAdapter
        from .cli_edit import _preflight_message

        report = rc.preflight()
        if not report["app_running"]:
            _emit(_preflight_message(report))
            return 2
        adapter = ResolveAdapter()
        session = rc.connect()
        ir = adapter.snapshot(session)
        sel = adapter.get_selection(session, ir)

    ctx = PlannerContext(ir=ir, selection=sel, source=session, adapter=adapter)
    result = plan_with_repair(ctx, prompt, transport)
    _emit("# " + result.cost_line())
    if result.plan is None or not result.ok:
        _emit("plan failed:\n  " + "\n  ".join(result.errors or ["no plan produced"]))
        return 1
    plan = result.plan

    if args.dry_run:
        _emit(dry_run(plan, ir))
        return 0 if validate(plan, ir).ok else 1

    if args.fixture:
        from .adapters.interchange import InterchangeAdapter

        out = args.out or (os.path.splitext(args.fixture)[0] + ".edited"
                           + os.path.splitext(args.fixture)[1])
        res = InterchangeAdapter().apply(plan, args.fixture, out_path=out)
    else:
        res = adapter.apply(plan, session)
    from .cli_edit import _emit_apply_result
    return _emit_apply_result(res)


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
    from .cli_edit import _emit_apply_result
    return _emit_apply_result(res)
