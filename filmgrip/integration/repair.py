"""Self-heal repair loop + token/cost accounting.

When Claude returns a well-formed but *invalid* EditPlan (a clip id that doesn't exist, a trim out
of bounds), film-grip doesn't guess a fix — it hands the exact, machine-readable validation errors
back to Claude and asks it to repair, resuming the same session so the model keeps its context.
The loop is capped, and it stops immediately on a terminal transport failure
(``error_max_structured_output_retries`` or a hard error) rather than looping forever.

It also accounts for the thing the product promises: tokens. Each turn's cost/usage is summed, and
the FGX context is measured against a raw timeline dump so the savings are reported, not asserted.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

from ..serialize import fgx
from .mcp_host import PlannerContext, PlanResult, Transport, plan_edit


@dataclass
class RepairResult:
    ok: bool
    plan: Optional[object] = None
    attempts: int = 0
    total_cost_usd: float = 0.0
    total_usage: dict = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    session_id: Optional[str] = None
    savings: dict = field(default_factory=dict)
    history: list[PlanResult] = field(default_factory=list)

    def cost_line(self) -> str:
        out = self.total_usage.get("output_tokens", 0)
        s = self.savings
        saved = f"; FGX {s.get('fgx_tokens', '?')} tok vs raw {s.get('raw_tokens', '?')} " \
                f"({s.get('savings_pct', '?')}% saved)" if s else ""
        status = "ok" if self.ok else "FAILED"
        return (f"[{status}] {self.attempts} turn(s), ${self.total_cost_usd:.4f}, "
                f"{out} output tok{saved}")


def _repair_prompt(errors: list[str]) -> str:
    bullets = "\n".join(f"- {e}" for e in errors)
    return ("Your previous EditPlan was invalid and was rejected by validation. Fix EXACTLY these "
            f"errors and return a corrected EditPlan:\n{bullets}\n"
            "Only reference clip ids you have seen via get_context/get_selection. Do not invent ids "
            "or frames. Return the full corrected plan.")


def token_report(ctx: PlannerContext) -> dict:
    """Measure the FGX context against a raw timeline dump (what 'paste the project' would cost)."""
    import opentimelineio as otio

    raw = otio.adapters.write_to_string(ctx.ir.timeline, "otio_json")
    raw_tokens = fgx.estimate_tokens(raw)
    bundle = fgx.bundle(ctx.ir, list(ctx.selection.ids), hops=1)
    fgx_tokens = fgx.estimate_tokens(bundle)
    savings_pct = round(100.0 * (1.0 - fgx_tokens / raw_tokens), 1) if raw_tokens else 0.0
    return {"fgx_tokens": fgx_tokens, "raw_tokens": raw_tokens, "savings_pct": savings_pct}


def plan_with_repair(ctx: PlannerContext, user_prompt: str, transport: Transport, *,
                     max_retries: int = 2, model: Optional[str] = None) -> RepairResult:
    """Plan, and if the plan is invalid, feed the errors back and retry via session resume."""
    usage: Counter = Counter()
    total_cost = 0.0
    session_id: Optional[str] = None
    prompt = user_prompt
    history: list[PlanResult] = []
    last: Optional[PlanResult] = None

    for attempt in range(1, max_retries + 2):  # 1 initial + max_retries repairs
        result = plan_edit(ctx, prompt, transport, session_id=session_id, model=model)
        history.append(result)
        last = result
        total_cost += result.cost_usd
        for k, v in (result.usage or {}).items():
            if isinstance(v, (int, float)):
                usage[k] += v
        session_id = result.session_id or session_id

        if result.ok:
            break
        if result.terminal:
            break  # model gave up / hard error — repairing won't help
        prompt = _repair_prompt(result.errors)  # resume next turn with the exact errors

    return RepairResult(
        ok=bool(last and last.ok),
        plan=last.plan if last else None,
        attempts=len(history),
        total_cost_usd=total_cost,
        total_usage=dict(usage),
        errors=last.errors if last else [],
        session_id=session_id,
        savings=token_report(ctx),
        history=history,
    )
