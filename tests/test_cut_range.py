"""D3 — the cut_range op: parse/validate rules + OTIO mutator surgery.

Geometry under test (24fps, one video track): clipA frames [0,480) (source 0..480),
clipB frames [480,720). Every assertion re-indexes the mutated OTIO graph through TimelineIR
so what's checked is the real resulting timeline, not mutator bookkeeping.
"""
from __future__ import annotations

import opentimelineio as otio
import pytest

from filmgrip.adapters.interchange import REBUILD_OPS, OtioMutator
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import EditPlan, all_op_names
from filmgrip.protocol.validate import (
    CLIP_TIMEWARPED,
    CUT_RANGE_ORDER,
    OUT_OF_BOUNDS,
    validate,
)


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


def _clip(name: str, src_in: int, dur: int) -> otio.schema.Clip:
    return otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(target_url=f"/footage/{name}.mov"),
        source_range=otio.opentime.TimeRange(_rt(src_in), _rt(dur)),
    )


@pytest.fixture
def ir() -> TimelineIR:
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("clipA", 0, 480))
    track.append(_clip("clipB", 0, 240))
    return TimelineIR(tl)


def _ids(ir):
    return [c.id for c in ir.real_clips()]


def _plan(ops) -> EditPlan:
    return EditPlan.parse({"ops": ops})


def _apply(ir, ops):
    plan = _plan(ops)
    res = validate(plan, ir)
    assert res.ok, [str(e) for e in res.errors]
    applied, unsupported = OtioMutator(ir).apply(plan)
    assert unsupported == []
    ir.reindex()
    return applied


def _geometry(ir):
    """[(name, start, duration, source_start)] for real clips, in track order."""
    return [(c.name, c.start, c.duration, c.source_start) for c in ir.real_clips()]


# --------------------------------------------------------------------------- protocol
def test_cut_range_is_a_schema_op():
    assert "cut_range" in all_op_names()
    with open("editplan.schema.json", "r", encoding="utf-8") as fh:
        assert '"cut_range"' in fh.read()


def test_cut_range_in_rebuild_ops():
    assert "cut_range" in REBUILD_OPS


def test_backwards_range_rejected_at_parse():
    with pytest.raises(Exception, match="start_frame < end_frame"):
        _plan([{"op": "cut_range", "clip_id": "x", "start_frame": 100, "end_frame": 100}])


# --------------------------------------------------------------------------- validation
def test_out_of_clip_bounds_rejected(ir):
    a = _ids(ir)[0]
    res = validate(_plan([{"op": "cut_range", "clip_id": a,
                           "start_frame": 400, "end_frame": 500}]), ir)  # clipA ends at 480
    assert OUT_OF_BOUNDS in res.codes()


def test_timewarped_clip_refused(ir):
    clip = ir.real_clips()[0]
    clip.otio.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    res = validate(_plan([{"op": "cut_range", "clip_id": clip.id,
                           "start_frame": 100, "end_frame": 148}]), ir)
    assert CLIP_TIMEWARPED in res.codes()


def test_rippling_cuts_must_be_last_to_first(ir):
    a = _ids(ir)[0]
    ascending = [
        {"op": "cut_range", "clip_id": a, "start_frame": 100, "end_frame": 124},
        {"op": "cut_range", "clip_id": a, "start_frame": 300, "end_frame": 348},
    ]
    res = validate(_plan(ascending), ir)
    assert CUT_RANGE_ORDER in res.codes()
    assert validate(_plan(list(reversed(ascending))), ir).ok


def test_non_rippling_cuts_may_be_any_order(ir):
    a = _ids(ir)[0]
    ascending = [
        {"op": "cut_range", "clip_id": a, "start_frame": 100, "end_frame": 124, "ripple": False},
        {"op": "cut_range", "clip_id": a, "start_frame": 300, "end_frame": 348, "ripple": False},
    ]
    assert validate(_plan(ascending), ir).ok


# --------------------------------------------------------------------------- mutation
def test_interior_cut_with_ripple(ir):
    a = _ids(ir)[0]
    _apply(ir, [{"op": "cut_range", "clip_id": a, "start_frame": 100, "end_frame": 148}])
    assert _geometry(ir) == [
        ("clipA", 0, 100, 0),       # left half keeps the head
        ("clipA", 100, 332, 148),   # tail resumes at source 148, pulled left (ripple)
        ("clipB", 432, 240, 0),     # clipB shifted left by the 48 removed frames
    ]


def test_interior_cut_leaving_a_gap(ir):
    a = _ids(ir)[0]
    _apply(ir, [{"op": "cut_range", "clip_id": a, "start_frame": 100, "end_frame": 148,
                 "ripple": False}])
    assert _geometry(ir) == [
        ("clipA", 0, 100, 0),
        ("clipA", 148, 332, 148),   # tail stays put; the hole is a gap
        ("clipB", 480, 240, 0),     # clipB untouched
    ]
    gaps = [c for c in ir.clips if c.kind == "gap"]
    assert any(g.start == 100 and g.duration == 48 for g in gaps)


def test_head_cut_is_a_trim_in(ir):
    a = _ids(ir)[0]
    _apply(ir, [{"op": "cut_range", "clip_id": a, "start_frame": 0, "end_frame": 48}])
    assert _geometry(ir) == [
        ("clipA", 0, 432, 48),
        ("clipB", 432, 240, 0),
    ]


def test_tail_cut_is_a_trim_out(ir):
    a = _ids(ir)[0]
    _apply(ir, [{"op": "cut_range", "clip_id": a, "start_frame": 432, "end_frame": 480}])
    assert _geometry(ir) == [
        ("clipA", 0, 432, 0),
        ("clipB", 432, 240, 0),
    ]


def test_whole_clip_cut_deletes(ir):
    a = _ids(ir)[0]
    _apply(ir, [{"op": "cut_range", "clip_id": a, "start_frame": 0, "end_frame": 480}])
    assert _geometry(ir) == [("clipB", 0, 240, 0)]


def test_whole_clip_cut_without_ripple_leaves_gap(ir):
    a = _ids(ir)[0]
    _apply(ir, [{"op": "cut_range", "clip_id": a, "start_frame": 0, "end_frame": 480,
                 "ripple": False}])
    assert _geometry(ir) == [("clipB", 480, 240, 0)]
    assert any(c.kind == "gap" and c.start == 0 and c.duration == 480 for c in ir.clips)


def test_two_descending_cuts_on_one_clip(ir):
    a = _ids(ir)[0]
    _apply(ir, [
        {"op": "cut_range", "clip_id": a, "start_frame": 300, "end_frame": 348},
        {"op": "cut_range", "clip_id": a, "start_frame": 100, "end_frame": 124},
    ])
    assert _geometry(ir) == [
        ("clipA", 0, 100, 0),
        ("clipA", 100, 176, 124),   # 124..300 of source
        ("clipA", 276, 132, 348),   # 348..480 of source
        ("clipB", 408, 240, 0),     # shifted by 48+24 removed frames
    ]


def test_markers_partition_and_cut_span_markers_drop(ir):
    a_clip = ir.real_clips()[0]
    for src_frame, name in ((50, "keep-left"), (120, "inside-cut"), (200, "keep-tail")):
        a_clip.otio.markers.append(otio.schema.Marker(
            name=name, marked_range=otio.opentime.TimeRange(_rt(src_frame), _rt(1))))
    _apply(ir, [{"op": "cut_range", "clip_id": a_clip.id,
                 "start_frame": 100, "end_frame": 148}])
    names = {m.name for c in ir.real_clips() for m in c.otio.markers}
    assert names == {"keep-left", "keep-tail"}
    left, tail = ir.real_clips()[0], ir.real_clips()[1]
    assert [m.name for m in left.otio.markers] == ["keep-left"]
    assert [m.name for m in tail.otio.markers] == ["keep-tail"]
