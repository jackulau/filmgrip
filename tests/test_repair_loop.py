"""D16 — self-heal repair loop + token/cost accounting."""
from __future__ import annotations

import pytest

from filmgrip.adapters.base import Selection
from filmgrip.core.ir import TimelineIR
from filmgrip.integration import mcp_host as mh
from filmgrip.integration.repair import plan_with_repair, token_report


@pytest.fixture
def ctx(fixtures_dir):
    ir = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    sel = [next(c for c in ir.real_clips() if c.name == "intro").id]
    return mh.PlannerContext(ir=ir, selection=Selection(ids=sel, basis="test"))


class SequenceTransport(mh.Transport):
    """Returns a queued list of PlanResponses in order; records each call's prompt + session."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def run(self, *, system_prompt, user_prompt, schema, ctx, session_id=None, model=None):
        self.calls.append({"user_prompt": user_prompt, "session_id": session_id})
        return self.responses[min(len(self.calls) - 1, len(self.responses) - 1)]


def _valid_plan_json(ctx):
    intro = ctx.selection.ids[0]
    return {"ops": [{"op": "add_marker", "clip_id": intro, "frame": 0, "color": "Blue"}]}


def test_repairs_invalid_then_succeeds_via_resume(ctx):
    invalid = mh.PlanResponse(subtype=mh.SUBTYPE_SUCCESS, session_id="s1",
                              structured_output={"ops": [{"op": "delete", "clip_id": "ghost"}]},
                              cost_usd=0.01, usage={"output_tokens": 20})
    valid = mh.PlanResponse(subtype=mh.SUBTYPE_SUCCESS, session_id="s1",
                            structured_output=_valid_plan_json(ctx),
                            cost_usd=0.02, usage={"output_tokens": 25})
    transport = SequenceTransport([invalid, valid])

    res = plan_with_repair(ctx, "flag the intro", transport, max_retries=2)
    assert res.ok
    assert res.attempts == 2
    # the second call fed the validation error back AND resumed the session
    second = transport.calls[1]
    assert "UNKNOWN_CLIP" in second["user_prompt"]
    assert second["session_id"] == "s1"
    # cost + usage accumulated across both turns
    assert abs(res.total_cost_usd - 0.03) < 1e-9
    assert res.total_usage["output_tokens"] == 45
    assert "$0.0300" in res.cost_line()


def test_caps_retries_when_always_invalid(ctx):
    invalid = mh.PlanResponse(subtype=mh.SUBTYPE_SUCCESS, session_id="s2",
                              structured_output={"ops": [{"op": "delete", "clip_id": "ghost"}]},
                              cost_usd=0.01)
    transport = SequenceTransport([invalid])  # always invalid
    res = plan_with_repair(ctx, "x", transport, max_retries=2)
    assert res.ok is False
    assert res.attempts == 3            # 1 initial + 2 repairs, then stop
    assert any("UNKNOWN_CLIP" in e for e in res.errors)


def test_terminal_failure_is_not_retried(ctx):
    terminal = mh.PlanResponse(subtype=mh.SUBTYPE_MAX_RETRIES, session_id="s3")
    transport = SequenceTransport([terminal])
    res = plan_with_repair(ctx, "x", transport, max_retries=5)
    assert res.ok is False
    assert res.attempts == 1            # gave up immediately, no wasted repair turns


def test_token_report_shows_savings(ctx):
    rep = token_report(ctx)
    assert rep["fgx_tokens"] < rep["raw_tokens"]
    assert rep["savings_pct"] > 50.0    # FGX subgraph is a fraction of a raw dump


def test_cost_line_includes_savings(ctx):
    valid = mh.PlanResponse(subtype=mh.SUBTYPE_SUCCESS, structured_output=_valid_plan_json(ctx),
                            cost_usd=0.0, usage={"output_tokens": 10})
    res = plan_with_repair(ctx, "x", SequenceTransport([valid]))
    line = res.cost_line()
    assert "saved" in line and "%" in line
    assert line.startswith("[ok]")
