"""D9 — MCP host + Agent SDK integration (transport stubbed)."""
from __future__ import annotations

import pytest

from filmgrip.adapters.base import Selection
from filmgrip.core.ir import TimelineIR
from filmgrip.integration import mcp_host as mh
from filmgrip.protocol import editplan as ep


@pytest.fixture
def ctx(fixtures_dir):
    ir = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    sel_ids = [next(c for c in ir.real_clips() if c.name == "broll_1").id]
    selection = Selection(ids=sel_ids, basis="test")
    return mh.PlannerContext(ir=ir, selection=selection)


class RecordingTransport(mh.Transport):
    """Captures what plan_edit sends and returns a canned PlanResponse."""

    def __init__(self, response: mh.PlanResponse):
        self.response = response
        self.seen = {}

    def run(self, *, system_prompt, user_prompt, schema, ctx, session_id=None, model=None):
        self.seen = {"system_prompt": system_prompt, "user_prompt": user_prompt,
                     "schema": schema, "session_id": session_id}
        return self.response


def test_tools_return_fgx_payloads(ctx):
    hdr = mh.payload_get_selection(ctx)
    assert set(hdr) >= {"seq", "r", "sel"}
    assert hdr["sel"] == ctx.selection.ids

    bundle = mh.payload_get_context(ctx, hops=1)
    assert bundle["cols"] == ["id", "t", "s", "d", "src", "si"]
    srcs = {row[4] for row in bundle["clips"]}
    assert "broll_1.mov" in srcs

    q = mh.payload_query_clips(ctx, name="broll")
    names = {row[4] for row in q["clips"]}
    assert names == {"broll_1.mov", "broll_2.mov"}

    qt = mh.payload_query_clips(ctx, track="a1")
    assert all(row[1] == "a1" for row in qt["clips"])


def test_system_prompt_declares_fgx_columns_once(ctx):
    sp = mh.build_system_prompt(ctx)
    assert "id,t,s,d,src,si" in sp
    assert "set_property" in sp and "Never invent ids" in sp


def test_system_prompt_teaches_color_ops_and_advisory(ctx):
    sp = mh.build_system_prompt(ctx)
    # every color op is advertised (ops list is derived from the schema, so this can't drift)
    for op in ("set_cdl", "apply_lut", "color_group", "color_version", "apply_grade"):
        assert op in sp, f"{op} not taught to the planner"
    # the honest advisory: non-scriptable color tools are named and forbidden
    assert "not scriptable" in sp.lower() or "NOT scriptable" in sp
    assert "Magic Mask" in sp and "Power Windows" in sp
    assert "get_scopes" in sp


def test_ops_list_is_derived_from_schema(ctx):
    sp = mh.build_system_prompt(ctx)
    for op in ep.all_op_names():
        assert op in sp


def test_get_scopes_tool_is_registered():
    assert "mcp__filmgrip__get_scopes" in mh._FG_TOOLS


def test_payload_get_scopes_handles_offline_media(ctx):
    # the cut.otio fixture references media that isn't on disk → honest per-clip errors, no fabrication
    out = mh.payload_get_scopes(ctx)
    assert "clips" in out and "errors" in out
    assert out["clips"] == [] or all("clip_id" in c for c in out["clips"])


def test_plan_edit_success_parses_and_validates(ctx):
    broll = ctx.selection.ids[0]
    plan_json = {"notes": "flag it", "ops": [
        {"op": "add_marker", "clip_id": broll, "frame": 0, "color": "Blue", "name": "x"},
    ]}
    transport = RecordingTransport(mh.PlanResponse(
        subtype=mh.SUBTYPE_SUCCESS, structured_output=plan_json,
        session_id="sess-1", cost_usd=0.012, usage={"output_tokens": 30}))
    result = mh.plan_edit(ctx, "flag the b-roll", transport)

    assert result.ok
    assert result.plan is not None and len(result.plan.ops) == 1
    assert result.session_id == "sess-1"
    assert result.cost_usd == 0.012
    # The EditPlan JSON Schema was handed to the transport as the structured-output contract.
    assert transport.seen["schema"] == ep.schema()


def test_plan_edit_invalid_plan_is_not_terminal(ctx):
    # Well-formed but references a clip that doesn't exist -> validation fails, repairable.
    transport = RecordingTransport(mh.PlanResponse(
        subtype=mh.SUBTYPE_SUCCESS,
        structured_output={"ops": [{"op": "delete", "clip_id": "ghost"}]},
        session_id="sess-2"))
    result = mh.plan_edit(ctx, "delete the ghost", transport)
    assert result.ok is False
    assert result.terminal is False                 # can be repaired via resume
    assert any("UNKNOWN_CLIP" in e for e in result.errors)
    assert result.session_id == "sess-2"


def test_plan_edit_handles_max_structured_output_retries(ctx):
    transport = RecordingTransport(mh.PlanResponse(
        subtype=mh.SUBTYPE_MAX_RETRIES, session_id="sess-3"))
    result = mh.plan_edit(ctx, "do something impossible", transport)
    assert result.ok is False
    assert result.terminal is True                  # model gave up; don't loop forever
    assert any("error_max_structured_output_retries" in e for e in result.errors)


def test_plan_edit_handles_generic_error_subtype(ctx):
    transport = RecordingTransport(mh.PlanResponse(subtype="error", errors=["boom"]))
    result = mh.plan_edit(ctx, "x", transport)
    assert result.ok is False and result.terminal is True
    assert "boom" in result.errors


def test_real_mcp_server_constructs_against_the_sdk(ctx):
    """The real in-process MCP server must build against the installed Agent SDK (no network)."""
    pytest.importorskip("claude_agent_sdk")
    server = mh.build_mcp_server(ctx)
    assert server["type"] == "sdk"
    assert server["name"] == "filmgrip"
    assert server["instance"] is not None
