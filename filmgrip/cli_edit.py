"""``film-grip edit`` — the end-to-end pipeline.

Two ways in, one pipeline:

* **live** (default, ``--editor resolve``): connect to a running editor, snapshot the timeline,
  capture the selection, ask Claude for an EditPlan (or replay one via ``--plan``), validate, and
  apply through the editor's adapter.
* **fixture** (``--fixture cut.otio``): run the whole thing against an ``.otio`` file with no editor
  and (with ``--plan``) no network — the path the e2e test exercises so the pipeline is provable on
  any machine.

``--dry-run`` stops after validation and prints the diff. Output is plain text to stdout; the exit
code is the contract (0 ok, 1 invalid/rejected plan, 2 editor unreachable, 3 unsupported).
"""
from __future__ import annotations

import json
import os
from typing import Optional

from .adapters.base import Selection
from .core.ir import TimelineIR
from .integration.mcp_host import PlannerContext, plan_edit
from .protocol.editplan import EditPlan
from .protocol.validate import dry_run, validate


def _load_recorded_plan(path: str) -> EditPlan:
    with open(path, "r", encoding="utf-8") as fh:
        return EditPlan.parse(json.load(fh))


def _selection_from_plan(ir: TimelineIR, plan: Optional[EditPlan]) -> list[str]:
    """Pick a sensible selection for context: the clips a recorded plan touches, else clip #1."""
    if plan is not None:
        ids = []
        for op in plan.ops:
            cid = getattr(op, "clip_id", None)
            if cid and cid not in ids and ir.clip(cid) is not None:
                ids.append(cid)
        if ids:
            return ids
    reals = ir.real_clips()
    return [reals[0].id] if reals else []


def _derived_out(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}.edited{ext}"


def _emit(text: str) -> None:
    print(text)


def cmd_edit(args) -> int:
    if args.fixture:
        return _run_fixture(args)
    return _run_live(args)


# --------------------------------------------------------------------------- fixture path
def _run_fixture(args) -> int:
    ir = TimelineIR.from_otio_file(args.fixture)

    plan: Optional[EditPlan] = None
    if args.plan:
        plan = _load_recorded_plan(args.plan)
    if plan is None:
        _emit("error: --fixture mode needs --plan (offline) — live Claude planning requires a "
              "running editor. Provide a recorded EditPlan with --plan.")
        return 3

    sel = Selection(ids=_selection_from_plan(ir, plan), basis="fixture")
    _ = PlannerContext(ir=ir, selection=sel)  # context the live path would hand to Claude

    if args.dry_run:
        diff = dry_run(plan, ir)
        _emit(diff)
        return 0 if validate(plan, ir).ok else 1

    # Applying to a fixture file = the OTIO-rebuild path (mutate OTIO, write a NEW file). Never
    # overwrite the source in place — default to a derived path so the input is preserved.
    from .adapters.interchange import InterchangeAdapter

    out = args.out or _derived_out(args.fixture)
    res = InterchangeAdapter().apply(plan, args.fixture, out_path=out)
    _emit(res.diff if res.ok else "apply failed:\n  " + "\n  ".join(res.errors))
    for w in res.warnings:
        _emit(f"  ⚠ {w}")
    return 0 if res.ok else 1


# --------------------------------------------------------------------------- live path
def _run_live(args) -> int:
    if args.editor != "resolve":
        _emit(f"error: live editor '{args.editor}' not yet supported; use --editor resolve or "
              f"--fixture. (Interchange editors run via export/import — see the registry.)")
        return 3

    from .adapters import resolve_client as rc
    from .adapters.resolve_adapter import ResolveAdapter

    report = rc.preflight()
    if not report["app_running"]:
        _emit(_preflight_message(report))
        return 2

    adapter = ResolveAdapter()
    session = rc.connect()
    ir = adapter.snapshot(session)
    selection = adapter.get_selection(session, ir)

    # Build the plan: replay a recorded one, or ask Claude.
    if args.plan:
        plan = _load_recorded_plan(args.plan)
    else:
        if not args.prompt:
            _emit("error: provide an instruction, e.g. film-grip edit \"add a blue marker on the "
                  "selected clip\"")
            return 1
        from .integration.mcp_host import ClaudeAgentTransport

        ctx = PlannerContext(ir=ir, selection=selection, source=session, adapter=adapter)
        result = plan_edit(ctx, args.prompt, ClaudeAgentTransport())
        if result.plan is None:
            _emit("plan failed: " + "; ".join(result.errors))
            return 1
        if not result.ok:
            _emit("plan invalid:\n  " + "\n  ".join(result.errors))
            return 1
        plan = result.plan
        if result.cost_usd:
            _emit(f"# planned in 1 turn — ${result.cost_usd:.4f}")

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


def _preflight_message(report: dict) -> str:
    lines = ["DaVinci Resolve is not reachable for scripting:"]
    lines.append(f"  - scripting module importable: {report['module_importable']}")
    lines.append(f"  - app running: {report['app_running']}")
    if report["module_importable"] and not report["app_running"]:
        lines.append("  → Open DaVinci Resolve (Studio), open a project + timeline, and enable")
        lines.append("    Preferences ▸ System ▸ General ▸ 'External scripting using' = Local.")
    elif not report["module_importable"]:
        lines.append("  → Install DaVinci Resolve (Studio); set RESOLVE_SCRIPT_API/RESOLVE_SCRIPT_LIB.")
    return "\n".join(lines)
