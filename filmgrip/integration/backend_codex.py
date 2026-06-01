"""Codex / GPT planner backend — the seam, ready for an implementation.

film-grip's planner is provider-agnostic (see :mod:`filmgrip.integration.backend`): anything that
can turn a prompt + JSON schema into a ``PlanResponse`` is a valid transport. Claude is the live
flagship; this module is the registered ``codex`` slot so an OpenAI/Codex/GPT backend can be added
*without touching* the CLI, the panel, the repair loop, or the validator.

What's intentionally NOT here yet: the actual API calls. Implementing this means filling in
:meth:`CodexTransport.run` to satisfy the contract in ``docs/BACKENDS.md`` — chiefly, coercing a
model that lacks Claude's native ``json_schema`` output mode into a schema-valid EditPlan and
running its own bounded repair. Until then the backend resolves and fails *honestly* (a clear
message, never a crash or a silent wrong result), so the seam is testable today.
"""
from __future__ import annotations

from typing import Optional

from .mcp_host import PlanResponse, PlannerContext, Transport

NOT_IMPLEMENTED = (
    "codex backend not yet implemented — the PlannerBackend seam is ready (see docs/BACKENDS.md); "
    "only the Claude backend can plan today. Use --backend claude (the default) for now."
)


class CodexTransport(Transport):
    """Placeholder transport: resolves like a real one but returns an honest not-implemented error.

    Returning an error ``PlanResponse`` (rather than raising) means it flows through the normal
    ``plan_edit`` path and surfaces as a clean "plan failed" message with the right exit code.
    """

    def run(self, *, system_prompt: str, user_prompt: str, schema: dict,
            ctx: PlannerContext, session_id: Optional[str] = None,
            model: Optional[str] = None) -> PlanResponse:
        return PlanResponse(subtype="error", errors=[NOT_IMPLEMENTED])


class CodexBackend:
    """The ``codex`` backend slot. ``transport()`` returns the not-yet-implemented placeholder."""

    name = "codex"

    def transport(self) -> Transport:
        return CodexTransport()
