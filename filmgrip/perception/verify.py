"""Post-apply verification: prove the editor ended up with what the plan meant.

Two layers, both honest about what they can and cannot see:

* **structural** — simulate the expected timeline by running the SAME OTIO mutator on a copy of
  the pre-edit graph, then diff its geometry against a fresh post-apply snapshot. Exact for
  every rebuild-path op with zero per-op bookkeeping; ops outside the rebuild set
  (``import_audio``/``add_track``/…) are reported as *skipped*, never silently assumed.
* **visual** — contact sheets ±N seconds around every NEW cut boundary the edit introduced
  (:mod:`~filmgrip.perception.frames`), so the agent can *look* at each seam. Source-media
  based: film-grip never renders an export to check its own work.

This is the non-destructive equivalent of the render→re-watch self-eval loop pipeline tools
use: same evidence (geometry + boundary imagery), no generation-loss bake.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..adapters.interchange import REBUILD_OPS, OtioMutator
from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan

MAX_BOUNDARIES = 8


@dataclass
class VerifyReport:
    ok: bool
    verified: list[str] = field(default_factory=list)    # human-readable proof lines
    mismatches: list[str] = field(default_factory=list)  # expected-vs-actual divergences
    skipped: list[str] = field(default_factory=list)     # ops structural verify can't model
    boundaries: list[int] = field(default_factory=list)  # new cut frames (timeline)
    sheets: list = field(default_factory=list)           # SheetResult per media group
    sheet_errors: list[str] = field(default_factory=list)

    def lines(self) -> list[str]:
        out = [f"  ✓ {v}" for v in self.verified]
        out += [f"  ✗ verify mismatch: {m}" for m in self.mismatches]
        out += [f"  ⚠ not structurally verifiable: {s}" for s in self.skipped]
        for sheet in self.sheets:
            out.append(f"  ◉ boundary sheet: {sheet.png_path}")
            out += [f"      tile {e['tile']}: frame {e['timeline_frame']}" for e in sheet.legend]
        out += [f"  ⚠ {e}" for e in self.sheet_errors]
        return out


def expected_after(ir_before: TimelineIR, plan: EditPlan) -> tuple[TimelineIR, list[str]]:
    """Simulate the post-edit timeline on a deep copy. Returns (expected IR, skipped op names).

    Clip ids are deterministic functions of (track, index, start, src_ref), so the copy mints
    the SAME ids — the plan applies to the copy verbatim.
    """
    copy_ir = TimelineIR(ir_before.timeline.deepcopy())
    modeled = [op for op in plan.ops if op.op in REBUILD_OPS]
    skipped = sorted({op.op for op in plan.ops if op.op not in REBUILD_OPS})
    OtioMutator(copy_ir).apply(EditPlan(notes=plan.notes, ops=modeled))
    copy_ir.reindex()
    return copy_ir, skipped


def geometry_rows(ir: TimelineIR) -> dict[str, list[tuple]]:
    """Per-track structural rows: (kind, start, duration, source_start, src_ref, enabled, fx).

    Trailing gaps are dropped and adjacent gaps merged per track — different writers express
    empty time differently without it meaning a different edit.
    """
    tracks: dict[str, list[tuple]] = {}
    for c in ir.clips:
        if c.kind == "transition":
            continue
        code = f"{c.track_kind[0]}{c.track_index}"
        enabled = bool(getattr(c.otio, "enabled", True)) if c.kind == "clip" else True
        fx = ()
        if c.kind == "clip":
            effects = getattr(c.otio, "effects", None) or []
            fx = tuple(sorted(
                (type(e).__name__, round(float(getattr(e, "time_scalar", 0.0) or 0.0), 4))
                for e in effects))
        tracks.setdefault(code, []).append(
            (c.kind, c.start, c.duration, c.source_start, c.src_ref, enabled, fx))
    for code, rows in tracks.items():
        merged: list[tuple] = []
        for row in rows:
            if row[0] == "gap" and merged and merged[-1][0] == "gap":
                prev = merged.pop()
                row = ("gap", prev[1], prev[2] + row[2], 0, prev[4], True, ())
            merged.append(row)
        while merged and merged[-1][0] == "gap":
            merged.pop()
        tracks[code] = merged
    return {k: v for k, v in tracks.items() if v}


def diff_geometry(expected: TimelineIR, actual: TimelineIR) -> tuple[list[str], list[str]]:
    """(mismatches, verified-summaries) comparing expected vs actual rows per track."""
    exp, act = geometry_rows(expected), geometry_rows(actual)
    mismatches: list[str] = []
    verified: list[str] = []
    for code in sorted(set(exp) | set(act)):
        e_rows, a_rows = exp.get(code, []), act.get(code, [])
        if e_rows == a_rows:
            if e_rows:
                verified.append(f"{code}: {len(e_rows)} item(s) match the expected geometry")
            continue
        if len(e_rows) != len(a_rows):
            mismatches.append(f"{code}: expected {len(e_rows)} item(s), found {len(a_rows)}")
        for i, (er, ar) in enumerate(zip(e_rows, a_rows)):
            if er != ar:
                mismatches.append(
                    f"{code} item {i}: expected {_fmt(er)}, got {_fmt(ar)}")
    return mismatches, verified


def _fmt(row: tuple) -> str:
    kind, start, dur, src, ref, enabled, fx = row
    bits = [f"{kind} @{start} dur {dur}"]
    if kind == "clip":
        bits.append(f"src {ref}@{src}")
        if not enabled:
            bits.append("disabled")
        if fx:
            bits.append("fx=" + ",".join(f"{n}:{s:g}" for n, s in fx))
    return " ".join(bits)


def new_boundaries(ir_before: TimelineIR, expected: TimelineIR) -> list[int]:
    """Cut points the edit introduced: clip edges in the expected timeline that didn't exist."""
    def edges(ir: TimelineIR) -> set[int]:
        out: set[int] = set()
        for c in ir.real_clips():
            out.add(c.start)
            out.add(c.end)
        return out

    fresh = sorted(edges(expected) - edges(ir_before))
    return fresh[:MAX_BOUNDARIES]


def verify_apply(ir_before: TimelineIR, plan: EditPlan, ir_after: TimelineIR, *,
                 sheets: bool = False, window_s: float = 1.5) -> VerifyReport:
    """The whole loop: simulate → diff → (optionally) render boundary sheets on the result."""
    expected, skipped = expected_after(ir_before, plan)
    mismatches, verified = diff_geometry(expected, ir_after)
    boundaries = new_boundaries(ir_before, expected)
    report = VerifyReport(ok=not mismatches, verified=verified, mismatches=mismatches,
                          skipped=skipped, boundaries=boundaries)
    if sheets and boundaries:
        from .frames import timeline_sheet

        win = max(1, int(round(window_s * ir_after.rate)))
        frames: list[int] = []
        for b in boundaries:
            frames += [max(0, b - win), max(0, b - 1), b, b + win]
        seen: set[int] = set()
        frames = [f for f in frames if not (f in seen or seen.add(f))]
        report.sheets, report.sheet_errors = timeline_sheet(ir_after, frames)
    return report
