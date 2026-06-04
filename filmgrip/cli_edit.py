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


def _adapter_for_fixture(path: str):
    """Pick the adapter for a fixture by file extension (the registry's job), defaulting to the OTIO
    interchange adapter. This is what lets `--fixture x.mlt|x.wfp|capcut.json` drive the CapCut /
    Filmora / MLT adapters the `film-grip editors` matrix advertises — not just OTIO files."""
    from .adapters import registry
    from .adapters.interchange import InterchangeAdapter

    ext = os.path.splitext(path)[1].lower()
    entry = registry.for_extension(ext)
    return entry.adapter if entry is not None else InterchangeAdapter()


def load_fixture_ir(path: str, adapter=None) -> TimelineIR:
    """Load a fixture into the IR via the right per-extension adapter, with actionable errors."""
    if not os.path.exists(path):
        raise FixtureError(f"fixture not found: {path}")
    if not os.path.isfile(path):
        raise FixtureError(f"fixture is not a file: {path}")
    adapter = adapter or _adapter_for_fixture(path)
    try:
        return adapter.snapshot(path)
    except Exception as exc:  # adapters raise various types for bad/unsupported files
        raise FixtureError(
            f"could not read '{path}' via the {adapter.name} adapter ({type(exc).__name__}: {exc}). "
            f"Expected a file the {adapter.name} adapter understands."
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


def _emit_apply_result(res) -> int:
    """Render an ApplyResult honestly and map it to the documented exit code.

    Always shows what DID apply (``diff``), then ops that couldn't (``✗`` unsupported), then lossy
    annotations (``⚠`` warnings), then hard failures. Exit: 0 ok · 3 some op unsupported · 1 other.
    A partial apply (some ops landed, some unsupported) is a non-zero exit — film-grip never claims
    success for an edit it didn't fully perform.
    """
    _emit(res.diff)
    for u in res.unsupported:
        _emit(f"  ✗ {u}")
    for w in res.warnings:
        _emit(f"  ⚠ {w}")
    if res.errors:
        _emit("apply failed:\n  " + "\n  ".join(res.errors))
    if res.ok:
        return 0
    return 3 if res.unsupported and not res.errors else 1


def cmd_edit(args) -> int:
    if args.fixture:
        return _run_fixture(args)
    return _run_live(args)


# --------------------------------------------------------------------------- fixture path
def _run_fixture(args) -> int:
    adapter = _adapter_for_fixture(args.fixture)
    try:
        ir = load_fixture_ir(args.fixture, adapter)
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

    # Apply via the extension's adapter (OTIO rebuild for interchange/.otio; native rewrite for
    # CapCut/MLT; Filmora honestly refuses). Never overwrite the source — write a derived path.
    from .adapters.base import NotSupportedError

    out = args.out or _derived_out(args.fixture)
    try:
        res = adapter.apply(plan, args.fixture, out_path=out)
    except NotSupportedError as exc:
        # e.g. Filmora is read-only — say so plainly and use the "unsupported" exit code.
        _emit(f"error: {exc}")
        return 3
    return _emit_apply_result(res)


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
    return _emit_apply_result(res)


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
