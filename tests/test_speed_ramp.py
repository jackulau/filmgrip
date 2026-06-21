"""D11 (goal 008) — the `speed_ramp` op: a validated, intentionally apply-UNSUPPORTED primitive.

speed_ramp is the variable-speed sibling of `retime`. retime is a single constant time_scalar (an
OTIO ``LinearTimeWarp``) and lands everywhere; a speed RAMP is a continuous speed curve, and there
is **no portable mechanism** for one — OTIO carries only a constant warp, no interchange format
film-grip writes (FCPXML/EDL/AAF/MLT/CapCut) round-trips a speed curve, and Resolve's scripting API
exposes no retime-curve method. So the HONEST contract is: the op is fully parsed + validated (the
decision is captured precisely, surfaced to the planner as a manual step), but EVERY apply path
REFUSES it via the capability-unsupported channel rather than silently faking a flat retime.

These tests are written so they FAIL the moment any adapter starts pretending it applied a ramp
(e.g. returns ok=True, or stops listing speed_ramp as unsupported) — proving there is no silent
fabrication. They run on the built-in ``otio_json`` / offline fixtures — no live editor required.
"""
from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

from filmgrip.adapters.capcut import CapcutAdapter
from filmgrip.adapters.interchange import REBUILD_OPS, InterchangeAdapter, OtioMutator
from filmgrip.adapters.mlt import MltAdapter
from filmgrip.adapters.resolve_adapter import LIVE_EXTRA_OPS, LIVE_OPS
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import SCHEMA_VERSION, EditPlan, all_op_names, schema
from filmgrip.protocol.validate import validate

OTIO = "tests/fixtures/cut.otio"
ROOT = Path(__file__).resolve().parent.parent


def _first_id(ir):
    return ir.real_clips()[0].id


def _ramp(cid, start=0, end=24, start_speed=1.0, end_speed=2.0, easing="smooth"):
    return {"op": "speed_ramp", "clip_id": cid, "start_frame": start, "end_frame": end,
            "start_speed": start_speed, "end_speed": end_speed, "easing": easing}


# --------------------------------------------------------------- registration / schema version
def test_speed_ramp_is_a_registered_op():
    assert "speed_ramp" in all_op_names()


def test_schema_version_was_bumped():
    # v4 was the color-grading bump; speed_ramp is v5.
    assert SCHEMA_VERSION >= 5
    assert schema()["properties"]["version"]["default"] == SCHEMA_VERSION


def test_speed_ramp_is_NOT_in_any_apply_set():
    # The honesty crux: it must be in none of the apply sets, so every adapter refuses it. (If a real
    # portable speed-curve ever lands, that change moves it INTO a set and this test is updated then.)
    assert "speed_ramp" not in REBUILD_OPS
    assert "speed_ramp" not in LIVE_OPS
    assert "speed_ramp" not in LIVE_EXTRA_OPS


# --------------------------------------------------------------- parse-time validation (pydantic)
def test_valid_speed_ramp_parses():
    p = EditPlan.parse({"ops": [_ramp("x")]})
    assert p.ops[0].op == "speed_ramp"
    assert p.ops[0].start_speed == 1.0 and p.ops[0].end_speed == 2.0
    assert p.ops[0].easing == "smooth"


@pytest.mark.parametrize("bad,why", [
    (dict(start_speed=0.0), "speed must be > 0 (0 is a freeze — that's retime, not a ramp)"),
    (dict(start_speed=-1.0), "negative speed (reverse) is retime's job, not a ramp"),
    (dict(end_speed=0.0), "end speed must be > 0"),
    (dict(start=24, end=24), "end_frame must be > start_frame"),
    (dict(start=30, end=10), "end_frame must be > start_frame"),
    (dict(start_speed=2.0, end_speed=2.0), "equal speeds = constant; use retime"),
    (dict(easing="warpspeed"), "unknown easing keyword"),
    (dict(start_speed=10000.0), "speed above the bound"),
])
def test_bad_speed_ramp_params_are_rejected_at_parse(bad, why):
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [_ramp("x", **bad)]})


# --------------------------------------------------------------- validator (range vs the live clip)
def test_speed_ramp_unknown_clip_is_rejected():
    ir = TimelineIR.from_otio_file(OTIO)
    res = validate(EditPlan.parse({"ops": [_ramp("nope")]}), ir)
    assert not res.ok and "UNKNOWN_CLIP" in res.codes()


def test_speed_ramp_window_inside_clip_validates():
    ir = TimelineIR.from_otio_file(OTIO)
    clip = ir.real_clips()[0]
    res = validate(EditPlan.parse(
        {"ops": [_ramp(clip.id, start=clip.start, end=clip.start + 1)]}), ir)
    assert res.ok, [str(e) for e in res.errors]


def test_speed_ramp_window_outside_clip_is_rejected():
    ir = TimelineIR.from_otio_file(OTIO)
    clip = ir.real_clips()[0]
    # end runs one frame past the clip → OUT_OF_BOUNDS, exactly like cut_range
    res = validate(EditPlan.parse(
        {"ops": [_ramp(clip.id, start=clip.start, end=clip.end + 1)]}), ir)
    assert not res.ok and "OUT_OF_BOUNDS" in res.codes()


# --------------------------------------------------------------- JSON Schema (the checked-in contract)
def _json_schema():
    return json.loads((ROOT / "editplan.schema.json").read_text())


def test_checked_in_json_schema_accepts_a_valid_speed_ramp():
    s = _json_schema()
    assert "speed_ramp" in json.dumps(s)  # the op made it into the published schema
    inst = {"ops": [_ramp("clipA", start=0, end=24, start_speed=1.0, end_speed=2.0)]}
    jsonschema.validate(instance=inst, schema=s)  # must not raise


@pytest.mark.parametrize("bad", [
    {"ops": [_ramp("c", start_speed=0.0)]},           # speed must be > 0 (exclusiveMinimum)
    {"ops": [_ramp("c", start_speed=-1.0)]},          # negative speed
    {"ops": [_ramp("c", easing="warpspeed")]},        # easing enum
    {"ops": [{"op": "speed_ramp", "clip_id": "c", "start_frame": 0, "end_frame": 24,
              "start_speed": 1.0, "end_speed": 2.0, "bogus": 1}]},  # extra=forbid
])
def test_checked_in_json_schema_rejects_invalid_speed_ramp(bad):
    s = _json_schema()
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(instance=bad, schema=s)


# --------------------------------------------------------------- CAPABILITY HONESTY (no silent fake)
# Every write path must REFUSE speed_ramp — never report it applied. These are the load-bearing tests.

def test_interchange_refuses_speed_ramp_no_silent_fake(tmp_path):
    a = InterchangeAdapter()
    ir = a.snapshot(OTIO)
    out = tmp_path / "o.otio"
    res = a.apply(EditPlan.parse({"ops": [_ramp(_first_id(ir))]}), OTIO, out_path=str(out))
    assert res.ok is False                      # NOT a buried-warning success
    assert res.applied == []                    # nothing was applied
    assert any("speed_ramp" in u for u in res.unsupported)
    assert any("speed curve" in u or "variable-speed" in u for u in res.unsupported)
    assert not out.exists()                     # nothing applied → no file written


def test_otio_mutator_does_not_attach_any_warp_for_speed_ramp(tmp_path):
    # The strongest anti-fabrication assertion: the OTIO graph is UNCHANGED — no LinearTimeWarp,
    # no FreezeFrame is sneaked on (which would be collapsing the ramp to a flat retime).
    import opentimelineio as otio
    a = InterchangeAdapter()
    ir = a.snapshot(OTIO)
    cid = _first_id(ir)
    applied, unsupported = OtioMutator(ir).apply(EditPlan.parse({"ops": [_ramp(cid)]}))
    assert applied == []
    assert any("speed_ramp" in u for u in unsupported)
    effects = list(getattr(ir.clip(cid).otio, "effects", []) or [])
    assert not any(isinstance(e, (otio.schema.LinearTimeWarp, otio.schema.FreezeFrame))
                   for e in effects), "speed_ramp must NOT fabricate a constant time-warp"


def test_capcut_refuses_speed_ramp(tmp_path):
    a = CapcutAdapter()
    src = "tests/fixtures/capcut_draft.json"
    ir = a.snapshot(src)
    out = tmp_path / "o.json"
    res = a.apply(EditPlan.parse({"ops": [_ramp(_first_id(ir))]}), src, out_path=str(out))
    assert res.ok is False
    assert any("speed_ramp" in u for u in res.unsupported)
    assert not out.exists()


def test_mlt_refuses_speed_ramp(tmp_path):
    a = MltAdapter()
    src = "tests/fixtures/sample.mlt"
    ir = a.snapshot(src)
    out = tmp_path / "o.mlt"
    res = a.apply(EditPlan.parse({"ops": [_ramp(_first_id(ir))]}), src, out_path=str(out))
    assert res.ok is False
    assert any("speed_ramp" in u for u in res.unsupported)
    assert not out.exists()


def test_filmora_refuses_speed_ramp_read_only():
    # Filmora has no write-back at all — it raises NotSupportedError for ANY op, speed_ramp included.
    from filmgrip.adapters.base import NotSupportedError
    from filmgrip.adapters.filmora import FilmoraAdapter
    a = FilmoraAdapter()
    with pytest.raises(NotSupportedError):
        a.apply(EditPlan.parse({"ops": [_ramp("anything")]}), "tests/fixtures/sample.wfp")


def test_resolve_refuses_speed_ramp_live_and_rebuild():
    # Resolve has neither a live retime-curve API nor a rebuild path for a speed curve, so it refuses.
    from filmgrip.adapters import resolve_client as rc
    from filmgrip.adapters.resolve_adapter import ResolveAdapter
    from tests.fakes import make_two_track_resolve
    adapter = ResolveAdapter()
    session = rc.connect(make_two_track_resolve())
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [_ramp(_first_id(ir))]}), session)
    assert res.ok is False
    assert any("speed_ramp" in u for u in res.unsupported)


def test_partial_plan_with_speed_ramp_is_ok_false_but_keeps_the_real_op(tmp_path):
    # A retime (real) + a speed_ramp (unsupported): the retime lands, the ramp is refused, ok=False.
    a = InterchangeAdapter()
    ir = a.snapshot(OTIO)
    cid = _first_id(ir)
    out = tmp_path / "o.otio"
    plan = EditPlan.parse({"ops": [
        {"op": "retime", "clip_id": cid, "speed_percent": 200},
        _ramp(cid),
    ]})
    res = a.apply(plan, OTIO, out_path=str(out))
    assert res.ok is False                                    # not everything asked for happened
    assert len(res.applied) == 1 and "retime" in res.applied[0]
    assert any("speed_ramp" in u for u in res.unsupported)
    assert out.exists()                                       # the retime that DID apply was written


# --------------------------------------------------------------- planner is told the honest truth
def test_planner_prompt_advertises_speed_ramp_as_manual():
    from filmgrip.adapters.base import Selection
    from filmgrip.integration.mcp_host import PlannerContext, build_system_prompt
    ir = TimelineIR.from_otio_file(OTIO)
    sel = Selection(ids=[_first_id(ir)], basis="t", note="", confidence="precise")
    prompt = build_system_prompt(PlannerContext(ir=ir, selection=sel))
    assert "speed_ramp" in prompt                 # the op is advertised
    assert "manual" in prompt.lower()             # honestly described as a manual step
