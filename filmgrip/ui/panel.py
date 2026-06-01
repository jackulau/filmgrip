"""The film-grip editor panel — structure + behaviour, decoupled from any GUI toolkit.

Resolve's Fusion UIManager is undocumented in the public scripting API and version-fragile, and an
Edit-page script has no Composition context — so we keep the panel a THIN shell. This module owns:

* :class:`PanelSpec` — a declarative widget tree (Window ▸ selection label ▸ prompt box ▸ Apply /
  Dry-run buttons ▸ output). The Resolve script just walks this tree to build native UIManager
  widgets; tests walk the same tree to assert the panel's shape.
* :class:`PanelController` — the behaviour: it owns a single injectable ``run_edit(prompt, dry_run)``
  seam, so a unit test drives Apply/Dry-run with a fake and asserts the right call, with no Resolve,
  no SDK and no network.

This is what makes "a panel inside DaVinci Resolve" testable: the only thing that needs Resolve open
is the ~50-line translation in the script, and even that imports cleanly everywhere (guards below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

# Widget ids the Resolve script and the tests both rely on (single source of truth).
ID_SELECTION = "fg_selection"
ID_PROMPT = "fg_prompt"
ID_APPLY = "fg_apply"
ID_DRYRUN = "fg_dryrun"
ID_COPY = "fg_copy"
ID_OUTPUT = "fg_output"


@dataclass
class Widget:
    kind: str                       # Window | VGroup | HGroup | Label | TextEdit | Button
    id: str = ""
    text: str = ""
    props: dict = field(default_factory=dict)
    children: list["Widget"] = field(default_factory=list)

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()


@dataclass
class PanelSpec:
    title: str
    root: Widget

    def find(self, wid: str) -> Optional[Widget]:
        return next((w for w in self.root.walk() if w.id == wid), None)

    def ids(self) -> set[str]:
        return {w.id for w in self.root.walk() if w.id}


def build_panel_spec(selection_summary: str = "(no selection)") -> PanelSpec:
    """The film-grip panel layout. Stable ids let the renderer + tests agree on structure."""
    root = Widget("Window", id="fg_window", props={"WindowTitle": "film-grip", "Geometry": [200, 200, 520, 360]},
                  children=[Widget("VGroup", children=[
                      Widget("Label", id=ID_SELECTION, text=selection_summary,
                             props={"WordWrap": True}),
                      Widget("Label", text="Describe the edit (clips are already in context):"),
                      Widget("TextEdit", id=ID_PROMPT,
                             props={"PlaceholderText": "e.g. add a whoosh as the title flies in, "
                                                       "then tighten the open by 12 frames"}),
                      Widget("HGroup", children=[
                          Widget("Button", id=ID_COPY, text="Copy context"),
                          Widget("Button", id=ID_DRYRUN, text="Dry-run"),
                          Widget("Button", id=ID_APPLY, text="Apply"),
                      ]),
                      Widget("Label", id=ID_OUTPUT, text="", props={"WordWrap": True}),
                  ])])
    return PanelSpec(title="film-grip", root=root)


@dataclass
class PanelResult:
    ok: bool
    text: str


# run_edit(prompt, dry_run) -> PanelResult : the edit seam. grab_context() -> str : the capture
# seam (the <selected_clips> block for Copy-context). The live factory below wires both to the real
# pipeline; tests inject fakes.
RunEdit = Callable[[str, bool], PanelResult]
GrabContext = Callable[[], str]


def _default_copy(text: str) -> bool:
    from ..clipboard import copy

    return copy(text)


class PanelController:
    """Panel behaviour over injectable seams — fully testable without Resolve, SDK or network."""

    def __init__(
        self,
        run_edit: RunEdit,
        selection_summary: str = "(no selection)",
        *,
        grab_context: Optional[GrabContext] = None,
        copy_fn: Callable[[str], bool] = _default_copy,
    ):
        self._run_edit = run_edit
        self._summary = selection_summary
        self._grab_context = grab_context
        self._copy_fn = copy_fn

    def selection_summary(self) -> str:
        return self._summary

    def spec(self) -> PanelSpec:
        return build_panel_spec(self._summary)

    def on_apply(self, prompt: str) -> PanelResult:
        return self._submit(prompt, dry_run=False)

    def on_dry_run(self, prompt: str) -> PanelResult:
        return self._submit(prompt, dry_run=True)

    def on_copy(self) -> PanelResult:
        """Copy the full ``<selected_clips>`` block to the clipboard (the panel's react-grab path)."""
        if self._grab_context is None:
            return PanelResult(False, "Copy unavailable — no selection context is wired.")
        try:
            block = self._grab_context()
        except Exception as exc:  # capture must never crash the panel
            return PanelResult(False, f"error: {exc}")
        if self._copy_fn(block):
            return PanelResult(True, "Copied selection context to the clipboard — paste it into "
                                     "your agent (Claude Code / Codex / Cursor).")
        # No clipboard tool available: show the block so the user can copy it from the output area.
        return PanelResult(True, block)

    def _submit(self, prompt: str, *, dry_run: bool) -> PanelResult:
        if not (prompt or "").strip():
            return PanelResult(False, "Type an instruction first.")
        try:
            return self._run_edit(prompt, dry_run)
        except Exception as exc:  # never let a planner/apply error crash the panel
            return PanelResult(False, f"error: {exc}")


def live_controller(session, *, adapter=None) -> PanelController:
    """Build a controller wired to the real Resolve pipeline (snapshot → plan → apply).

    Imported lazily by the Resolve script. Kept here (tested module) so the script stays a thin
    translator. The planner/transport are only imported when an edit is actually run.
    """
    from ..adapters.resolve_adapter import ResolveAdapter
    from ..serialize.selection_block import armed_preview, format_selection

    adapter = adapter or ResolveAdapter()
    ir = adapter.snapshot(session)
    selection = adapter.get_selection(session, ir)
    summary = armed_preview(ir, selection.ids,
                            confidence=getattr(selection, "confidence", "precise"))

    def grab_context() -> str:
        # Re-snapshot so the copied block matches the CURRENT timeline + selection, not the
        # snapshot taken when the panel opened.
        cur_ir = adapter.snapshot(session)
        cur_sel = adapter.get_selection(session, cur_ir)
        return format_selection(cur_ir, cur_sel.ids,
                                confidence=getattr(cur_sel, "confidence", "precise"),
                                basis=getattr(cur_sel, "basis", ""))

    def run_edit(prompt: str, dry_run: bool) -> PanelResult:
        from ..integration.backend import get_backend
        from ..integration.mcp_host import PlannerContext
        from ..integration.repair import plan_with_repair
        from ..protocol.validate import dry_run as render_dry_run

        # Re-snapshot so the panel always plans against the current timeline state.
        cur_ir = adapter.snapshot(session)
        cur_sel = adapter.get_selection(session, cur_ir)
        if not cur_sel.ids:
            return PanelResult(False, "Select a clip in Resolve first (a timeline clip or a "
                               "media-pool item), then type your instruction.")
        ctx = PlannerContext(ir=cur_ir, selection=cur_sel, source=session, adapter=adapter)
        result = plan_with_repair(ctx, prompt, get_backend().transport())
        if result.plan is None or not result.ok:
            return PanelResult(False, "Could not produce a valid edit:\n  "
                               + "\n  ".join(result.errors or ["no plan"]))
        if dry_run:
            return PanelResult(True, render_dry_run(result.plan, cur_ir))
        res = adapter.apply(result.plan, session)
        body = res.diff if res.ok else ("apply failed:\n  " + "\n  ".join(res.errors))
        if res.warnings:
            body += "\n  ⚠ " + "\n  ⚠ ".join(res.warnings)
        return PanelResult(res.ok, body)

    return PanelController(run_edit, selection_summary=summary, grab_context=grab_context)
