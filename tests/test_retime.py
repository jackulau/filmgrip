"""D1 (goal 005) — the `retime` op: real speed change via an OTIO time-warp.

retime is the highest-value pro primitive the OTIO IR can carry truthfully: it attaches a
``LinearTimeWarp`` (``FreezeFrame`` at 0) to the clip, which survives the OTIO round-trip and so
lands on every rebuild/interchange editor and in Resolve via the rebuild. These tests run on a
built-in ``otio_json`` timeline — no editor and no interchange extra required.
"""
from __future__ import annotations

import opentimelineio as otio
import pytest

from filmgrip.adapters.interchange import REBUILD_OPS, OtioMutator
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import EditPlan, all_op_names
from filmgrip.protocol.validate import validate


def _single_clip_ir(tmp_path, dur=48):
    rate = 24.0

    def rt(f):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="t")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v.append(otio.schema.Clip(
        name="A", media_reference=otio.schema.ExternalReference(target_url="/m/a.mov"),
        source_range=otio.opentime.TimeRange(rt(0), rt(dur))))
    tl.tracks.append(v)
    path = str(tmp_path / "t.otio")
    otio.adapters.write_to_file(tl, path)
    return TimelineIR.from_otio_file(path), path


def _warps(clip):
    return [e for e in clip.otio.effects if isinstance(e, otio.schema.LinearTimeWarp)]


def test_retime_is_a_registered_op():
    assert "retime" in all_op_names()
    assert "retime" in REBUILD_OPS  # applies via the OTIO rebuild / interchange path


def test_retime_2x_attaches_linear_timewarp(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    applied, unsupported = OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": 200}]}))
    assert not unsupported and len(applied) == 1
    warps = _warps(ir.clip(cid))
    assert len(warps) == 1
    assert warps[0].time_scalar == pytest.approx(2.0)
    # the warp changes playback, NOT the clip's timeline span — neighbours are untouched
    assert ir.clip(cid).duration == 48


def test_retime_half_speed(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": 50}]}))
    assert _warps(ir.clip(cid))[0].time_scalar == pytest.approx(0.5)


def test_retime_reverse_is_negative_scalar(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": -100}]}))
    assert _warps(ir.clip(cid))[0].time_scalar == pytest.approx(-1.0)


def test_retime_zero_is_freeze_frame(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": 0}]}))
    warps = _warps(ir.clip(cid))
    assert len(warps) == 1 and isinstance(warps[0], otio.schema.FreezeFrame)
    assert warps[0].time_scalar == pytest.approx(0.0)


def test_reapplying_retime_replaces_not_stacks(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    mut = OtioMutator(ir)
    mut.apply(EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": 200}]}))
    mut.apply(EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": 50}]}))
    warps = _warps(ir.clip(cid))
    assert len(warps) == 1  # replaced, not stacked
    assert warps[0].time_scalar == pytest.approx(0.5)


def test_retime_survives_otio_roundtrip(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    cid = ir.real_clips()[0].id
    OtioMutator(ir).apply(
        EditPlan.parse({"ops": [{"op": "retime", "clip_id": cid, "speed_percent": 200}]}))
    out = str(tmp_path / "rt.otio")
    ir.to_otio_file(out)
    ir2 = TimelineIR.from_otio_file(out)
    warps = _warps(ir2.real_clips()[0])
    assert len(warps) == 1 and warps[0].time_scalar == pytest.approx(2.0)


def test_retime_unknown_clip_is_rejected(tmp_path):
    ir, _ = _single_clip_ir(tmp_path)
    res = validate(EditPlan.parse({"ops": [{"op": "retime", "clip_id": "nope", "speed_percent": 200}]}), ir)
    assert not res.ok
    assert "UNKNOWN_CLIP" in res.codes()


def test_retime_out_of_range_rejected_at_parse():
    # speed_percent is bounded so a hallucinated absurd value can't survive parsing.
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [{"op": "retime", "clip_id": "x", "speed_percent": 999999}]})


def test_retime_advertised_to_planner(tmp_path):
    from filmgrip.adapters.base import Selection
    from filmgrip.integration.mcp_host import PlannerContext, build_system_prompt
    ir, _ = _single_clip_ir(tmp_path)
    sel = Selection(ids=[ir.real_clips()[0].id], basis="t", note="", confidence="precise")
    prompt = build_system_prompt(PlannerContext(ir=ir, selection=sel))
    assert "retime" in prompt
    assert "freeze-frame" in prompt and "reverse" in prompt
