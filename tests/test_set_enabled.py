"""D2 (goal 005) — the `set_enabled` op: disable/enable a clip without deleting it.

set_enabled lands live in Resolve (``TimelineItem.SetClipEnabled``) AND via the OTIO rebuild
(the clip's ``enabled`` flag), so it shows yes/yes/yes in the honest capability matrix. These tests
exercise the rebuild/interchange path on a built-in ``otio_json`` timeline (no editor needed) plus
the op's protocol wiring.
"""
from __future__ import annotations

import opentimelineio as otio
import pytest

from filmgrip.adapters.interchange import REBUILD_OPS, OtioMutator
from filmgrip.adapters.resolve_adapter import LIVE_OPS
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import EditPlan, all_op_names
from filmgrip.protocol.validate import validate


def _single_clip_ir(tmp_path):
    rate = 24.0

    def rt(f):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="t")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v.append(otio.schema.Clip(
        name="A", media_reference=otio.schema.ExternalReference(target_url="/m/a.mov"),
        source_range=otio.opentime.TimeRange(rt(0), rt(48))))
    tl.tracks.append(v)
    path = str(tmp_path / "t.otio")
    otio.adapters.write_to_file(tl, path)
    return TimelineIR.from_otio_file(path), path


def test_set_enabled_is_registered_op():
    assert "set_enabled" in all_op_names()
    # both paths: live in Resolve (SetClipEnabled) and via the OTIO rebuild / interchange
    assert "set_enabled" in LIVE_OPS
    assert "set_enabled" in REBUILD_OPS


def test_disable_sets_clip_enabled_false(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    assert ir.clip(cid).otio.enabled is True  # clips start enabled
    applied, unsupported = OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "set_enabled", "clip_id": cid, "enabled": False}]}))
    assert not unsupported and len(applied) == 1
    assert ir.clip(cid).otio.enabled is False


def test_enable_sets_clip_enabled_true(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    ir.clip(cid).otio.enabled = False
    OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "set_enabled", "clip_id": cid, "enabled": True}]}))
    assert ir.clip(cid).otio.enabled is True


def test_set_enabled_survives_otio_roundtrip(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "set_enabled", "clip_id": cid, "enabled": False}]}))
    out = str(tmp_path / "se.otio")
    ir.to_otio_file(out)
    ir2 = TimelineIR.from_otio_file(out)
    assert ir2.real_clips()[0].otio.enabled is False


def test_set_enabled_unknown_clip_is_rejected(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    res = validate(
        EditPlan.parse({"ops": [{"op": "set_enabled", "clip_id": "nope", "enabled": False}]}), ir)
    assert not res.ok
    assert "UNKNOWN_CLIP" in res.codes()


def test_set_enabled_requires_bool(tmp_path):
    # the schema forbids extra/garbage and requires the bool field
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [{"op": "set_enabled", "clip_id": "x"}]})  # missing 'enabled'


def test_set_enabled_advertised_to_planner(tmp_path):
    from filmgrip.adapters.base import Selection
    from filmgrip.integration.mcp_host import PlannerContext, build_system_prompt
    ir, _ = _single_clip_ir(tmp_path)
    sel = Selection(ids=[ir.real_clips()[0].id], basis="t", note="", confidence="precise")
    prompt = build_system_prompt(PlannerContext(ir=ir, selection=sel))
    assert "set_enabled" in prompt
