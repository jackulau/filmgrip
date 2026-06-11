"""D6 — plan validator + dry-run diff."""
from __future__ import annotations

import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.protocol import validate as V
from filmgrip.protocol.editplan import EditPlan


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def _id(cut, name):
    return next(c for c in cut.real_clips() if c.name == name).id


def test_valid_plan_passes(cut):
    intro = _id(cut, "intro")
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": intro, "edge": "out", "delta": -12},
        {"op": "add_marker", "clip_id": intro, "frame": 4, "color": "Green"},
        {"op": "set_property", "clip_id": intro, "key": "Opacity", "value": 80},
    ]})
    res = V.validate(plan, cut)
    assert res.ok
    assert res.errors == []


def test_unknown_id_and_out_of_bounds_trim_rejected_atomically(cut):
    intro = _id(cut, "intro")  # dur 48
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": "deadbeef", "edge": "out", "delta": -5},     # unknown
        {"op": "trim", "clip_id": intro, "edge": "out", "delta": -1000},        # dur -> negative
    ]})
    res = V.validate(plan, cut)
    assert not res.ok
    assert V.UNKNOWN_CLIP in res.codes()
    assert V.OUT_OF_BOUNDS in res.codes()
    # atomic: dry_run reports rejection, does not pretend any op applied
    diff = V.dry_run(plan, cut)
    assert "PLAN REJECTED" in diff
    assert "UNKNOWN_CLIP" in diff and "OUT_OF_BOUNDS" in diff


def test_marker_outside_clip_rejected(cut):
    intro = _id(cut, "intro")  # dur 48
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": intro, "frame": 100}]})
    res = V.validate(plan, cut)
    assert V.OUT_OF_BOUNDS in res.codes()


def test_move_overlap_rejected_but_free_slot_ok(cut):
    outro = _id(cut, "outro")  # V1 300..360, dur 60
    # Move onto dialogue_a (48..120) -> overlap.
    bad = EditPlan.parse({"ops": [{"op": "move", "clip_id": outro, "to_start": 60}]})
    assert V.ILLEGAL_OVERLAP in V.validate(bad, cut).codes()
    # Move far past the end (free) -> ok.
    good = EditPlan.parse({"ops": [{"op": "move", "clip_id": outro, "to_start": 1000}]})
    assert V.validate(good, cut).ok


def test_move_to_missing_track_rejected(cut):
    intro = _id(cut, "intro")
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": intro, "to_start": 500, "to_track": "v9"}]})
    assert V.TRACK_NOT_FOUND in V.validate(plan, cut).codes()


def test_transition_needs_adjacent_clip(cut):
    outro = _id(cut, "outro")  # last clip on V1 -> no 'out' neighbor
    plan = EditPlan.parse({"ops": [
        {"op": "add_transition", "clip_id": outro, "edge": "out", "type": "cross_dissolve"},
    ]})
    assert V.TRANSITION_NOT_ADJACENT in V.validate(plan, cut).codes()
    # 'in' edge of outro has broll_2 before it -> fine
    ok = EditPlan.parse({"ops": [
        {"op": "add_transition", "clip_id": outro, "edge": "in", "type": "cross_dissolve"},
    ]})
    assert V.validate(ok, cut).ok


def test_split_must_be_inside_clip(cut):
    intro = _id(cut, "intro")  # 0..48
    assert V.SPLIT_OUT_OF_CLIP in V.validate(
        EditPlan.parse({"ops": [{"op": "split", "clip_id": intro, "at_frame": 200}]}), cut).codes()
    assert V.validate(
        EditPlan.parse({"ops": [{"op": "split", "clip_id": intro, "at_frame": 24}]}), cut).ok


def test_targeting_a_gap_is_not_a_clip(cut):
    # find a gap id
    gap = next(c for c in cut.clips if c.kind == "gap")
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": gap.id}]})
    assert V.NOT_A_CLIP in V.validate(plan, cut).codes()


def test_dry_run_reads_like_a_diff(cut):
    intro = _id(cut, "intro")
    plan = EditPlan.parse({"notes": "tighten open", "ops": [
        {"op": "trim", "clip_id": intro, "edge": "out", "delta": -12},
        {"op": "add_marker", "clip_id": intro, "frame": 0, "color": "Red", "name": "hook"},
    ]})
    diff = V.dry_run(plan, cut)
    assert "PLAN OK" in diff
    assert "tighten open" in diff
    assert "trim intro out -12" in diff
    assert "0..48 ⇒ 0..36" in diff
    assert "marker Red on intro" in diff


def test_dry_run_shows_rationale(cut):
    """v3: reason/quote ride along into the human-readable diff."""
    cid = cut.real_clips()[0].id
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": cid, "edge": "in", "delta": 5,
         "reason": "tighten the open", "quote": "Hey everyone"},
    ]})
    text = V.dry_run(plan, cut)
    assert '"Hey everyone"' in text
    assert "— tighten the open" in text
