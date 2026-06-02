"""``film-grip grab`` — capture the selected clips as a ``<selected_clips>`` context block.

react-grab's primary flow, for a timeline: select clips (Resolve's *native multi-select* — the
thing react-grab can't do), run ``grab``, and the compact context block is printed and copied to the
clipboard, ready to paste into any agent (Claude Code, Codex, Cursor). The same block powers the
in-Resolve panel's capture-preview and *Copy context* button.

Two ways in, mirroring ``film-grip edit``:

* **live** (default, ``--editor resolve``): connect to Resolve, snapshot, capture the real
  selection (honestly tagged ``[reconstructed]`` because Resolve exposes no multi-select API).
* **fixture** (``--fixture cut.otio [--select id1,id2]``): grab from an ``.otio`` file with no
  editor — the path the test exercises, and the way to see the block on any machine.

Exit codes match the rest of the CLI: 0 ok, 1 bad input (unknown id), 2 editor unreachable,
3 unsupported editor.
"""
from __future__ import annotations

from .core.ir import TimelineIR
from .serialize.selection_block import format_selection


def _emit(text: str) -> None:
    print(text)


def cmd_grab(args) -> int:
    return _grab_fixture(args) if args.fixture else _grab_live(args)


def _grab_fixture(args) -> int:
    from .cli_edit import FixtureError, load_fixture_ir

    try:
        ir = load_fixture_ir(args.fixture)
    except FixtureError as exc:
        _emit(f"error: {exc}")
        return 1
    if getattr(args, "select", None):
        ids = [s.strip() for s in args.select.split(",") if s.strip()]
        unknown = [s for s in ids if ir.clip(s) is None]
        if unknown:
            _emit(f"error: unknown clip id(s): {', '.join(unknown)}. Re-run "
                  f"`film-grip grab --fixture {args.fixture}` with no --select to list every clip "
                  f"and its id, then pick from those.")
            return 1
    else:
        ids = [c.id for c in ir.real_clips()]
    block = format_selection(ir, ids, confidence="precise", basis="fixture",
                             neighbors=not getattr(args, "no_neighbors", False))
    _emit(block)
    _maybe_copy(args, block)
    return 0


def _grab_live(args) -> int:
    if args.editor != "resolve":
        _emit(f"error: live editor '{args.editor}' not supported for grab; use --editor resolve "
              f"or --fixture FILE.")
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
    block = format_selection(ir, sel.ids, confidence=getattr(sel, "confidence", "precise"),
                             basis=sel.basis, neighbors=not getattr(args, "no_neighbors", False))
    _emit(block)
    _maybe_copy(args, block)
    return 0


def _maybe_copy(args, block: str) -> None:
    if getattr(args, "no_copy", False):
        return
    from .clipboard import copy

    if copy(block):
        _emit("# copied to clipboard — paste into your agent (Claude Code / Codex / Cursor)")
    else:
        _emit("# (no clipboard tool found — block printed above; copy it manually)")
