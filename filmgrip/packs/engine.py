"""Compile a deterministic pack into an EditPlan — then the normal pipeline validates/applies it."""
from __future__ import annotations

from typing import Optional

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan
from . import Pack, PackError


def compile_pack(pack: Pack, ir: TimelineIR, ids: list, params: Optional[dict] = None) -> EditPlan:
    """Compile a DETERMINISTIC pack to an EditPlan over ``ids`` (unvalidated — caller validates).

    Merges ``params`` over the pack's defaults, runs the recipe's compiler, and wraps the resulting
    ops in an EditPlan via the same ``EditPlan.parse`` the planner uses — so a recipe that emits a
    malformed op fails at parse, and an inapplicable one fails at ``validate``. No bypass.
    """
    if pack.kind != "deterministic" or pack.compile is None:
        raise PackError(
            f"pack '{pack.name}' is a {pack.kind} pack; it can't be compiled to ops directly "
            f"(prompt packs run through the planner backend)."
        )
    merged = {**pack.params, **(params or {})}
    ops = pack.compile(ir, list(ids or []), merged)
    return EditPlan.parse({"notes": f"pack: {pack.name}", "ops": ops})
