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
from .integration.mcp_host import PlannerContext
from .protocol.editplan import EditPlan
from .protocol.validate import dry_run, validate


class FixtureError(Exception):
    """A --fixture path that can't be loaded — surfaced as a friendly CLI error, not a traceback."""


def load_fixture_ir(path: str) -> TimelineIR:
    """Load a fixture into the IR with actionable errors (missing file, unreadable format)."""
    if not os.path.exists(path):
        raise FixtureError(f"fixture not found: {path}")
    if not os.path.isfile(path):
        raise FixtureError(f"fixture is not a file: {path}")
    try:
        return TimelineIR.from_otio_file(path)
    except Exception as exc:  # OTIO raises various types for bad/unsupported files
        raise FixtureError(
            f"could not read '{path}' as a timeline ({type(exc).__name__}: {exc}). Expected an "
            f"OpenTimelineIO file (.otio) or a format OTIO can read."
        ) from exc


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
    try:
        ir = load_fixture_ir(args.fixture)
    except FixtureError as exc:
        _emit(f"error: {exc}")
        return 1

    # Two ways to get a plan with no live editor: replay a recorded one (--plan), or plan against
    # the fixture via the selected backend (a prompt). The latter needs no editor — only the
    # backend — so it's how a non-Claude backend (e.g. codex) is exercised, and how you can plan
    # against an .otio on any machine.
    if args.plan:
        plan: Optional[EditPlan] = _load_recorded_plan(args.plan)
    elif args.prompt:
        plan = _plan_against_fixture(args, ir)
        if plan is None:
            return 1  # the planner already emitted the failure + cost line
    else:
        _emit("error: --fixture needs either --plan <file> (replay a recorded EditPlan) or a "
              "prompt (plan against the fixture via the selected backend, e.g. "
              "film-grip edit --fixture cut.otio \"tighten the open\").")
        return 3

    if args.dry_run:
        _emit(dry_run(plan, ir))
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


def _plan_against_fixture(args, ir: TimelineIR) -> Optional[EditPlan]:
    """Plan against a fixture IR via the selected backend (no live editor). None on failure."""
    from .integration.backend import UnknownBackendError, get_backend
    from .integration.repair import plan_with_repair

    try:
        transport = get_backend(getattr(args, "backend", None)).transport()
    except UnknownBackendError as exc:
        _emit(f"error: {exc}")
        return None

    # No real selection offline → default the planning context to every clip in the fixture.
    sel = Selection(ids=[c.id for c in ir.real_clips()], basis="fixture")
    ctx = PlannerContext(ir=ir, selection=sel)
    result = plan_with_repair(ctx, args.prompt, transport)
    _emit("# " + result.cost_line())
    if result.plan is None or not result.ok:
        _emit("plan failed:\n  " + "\n  ".join(result.errors or ["no plan produced"]))
        return None
    return result.plan


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

    # Surface the selection + how confident we are in it (Resolve has no true multi-select API).
    conf = getattr(selection, "confidence", "precise")
    _emit(f"# selection: {len(selection.ids)} clip(s) [{conf}]"
          + (f" — {selection.note}" if selection.note else ""))

    # Build the plan: replay a recorded one, or ask Claude.
    if args.plan:
        plan = _load_recorded_plan(args.plan)
    else:
        if not args.prompt:
            _emit("error: provide an instruction, e.g. film-grip edit \"add a blue marker on the "
                  "selected clip\"")
            return 1
        if not selection.ids:
            _emit("error: no clips selected. Select a clip on the timeline (or a media-pool item) "
                  "in Resolve, then re-run. Resolve exposes no multi-clip selection API, so "
                  "film-grip reconstructs your selection from the current clip + media-pool "
                  "selection — give it something to work from.")
            return 1
        from .integration.backend import UnknownBackendError, get_backend
        from .integration.repair import plan_with_repair

        try:
            transport = get_backend(getattr(args, "backend", None)).transport()
        except UnknownBackendError as exc:
            _emit(f"error: {exc}")
            return 1

        ctx = PlannerContext(ir=ir, selection=selection, source=session, adapter=adapter)
        result = plan_with_repair(ctx, args.prompt, transport)
        _emit("# " + result.cost_line())
        if result.plan is None or not result.ok:
            _emit("plan failed:\n  " + "\n  ".join(result.errors or ["no plan produced"]))
            return 1
        plan = result.plan

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
