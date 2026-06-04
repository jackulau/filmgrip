"""D7 (goal 004) — the live LLM glue: ClaudeAgentTransport's ResultMessage -> PlanResponse mapping.

Every other planner test injects a fake Transport, so the REAL Claude path (subtype/structured_output/
cost/usage/errors mapping) had zero coverage — a regression in reading the SDK's ResultMessage would
ship silently. `_to_plan_response` is pure (no SDK, no network), so we exercise it with a fake
ResultMessage-shaped object. The async `query()` loop around it stays the only un-coverable glue.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from filmgrip.integration.mcp_host import SUBTYPE_SUCCESS, ClaudeAgentTransport, PlanResponse

_map = ClaudeAgentTransport._to_plan_response


@dataclass
class FakeResultMessage:
    """Shaped like claude_agent_sdk.ResultMessage — only the fields the mapping reads."""
    subtype: Optional[str] = None
    is_error: bool = False
    structured_output: Optional[dict] = None
    session_id: Optional[str] = None
    total_cost_usd: Optional[float] = None
    usage: Optional[dict] = None
    errors: Optional[list] = None
    result: Any = None


def test_maps_a_successful_result_through():
    plan = {"ops": [{"op": "add_marker", "clip_id": "c1", "frame": 0}]}
    r = _map(FakeResultMessage(subtype="success", structured_output=plan, session_id="s1",
                               total_cost_usd=0.42, usage={"input_tokens": 10}, result="done"))
    assert isinstance(r, PlanResponse)
    assert r.subtype == "success"
    assert r.structured_output == plan
    assert r.session_id == "s1"
    assert r.cost_usd == 0.42
    assert r.usage == {"input_tokens": 10}
    assert r.errors == []
    assert r.raw == "done"


def test_subtype_falls_back_to_error_when_is_error():
    r = _map(FakeResultMessage(subtype=None, is_error=True, errors=["boom"]))
    assert r.subtype == "error"
    assert r.errors == ["boom"]


def test_subtype_falls_back_to_success_when_not_error():
    r = _map(FakeResultMessage(subtype=None, is_error=False))
    assert r.subtype == SUBTYPE_SUCCESS


def test_none_totals_coerce_to_safe_defaults():
    # The SDK can return None for cost/usage/errors — the caller must never see a stray None.
    r = _map(FakeResultMessage(subtype="success", total_cost_usd=None, usage=None, errors=None))
    assert r.cost_usd == 0.0
    assert r.usage == {}
    assert r.errors == []


def test_no_result_message_is_an_honest_error():
    r = _map(None)
    assert r.subtype == "error"
    assert r.errors == ["no ResultMessage from SDK"]
