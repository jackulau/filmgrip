"""D8 — Resolve adapter apply (live write, compensating rollback)."""
from __future__ import annotations

import pytest

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.resolve_adapter import ResolveAdapter
from filmgrip.protocol.editplan import EditPlan
from tests.fakes import (
    FakeMediaPool,
    FakeProject,
    FakeResolve,
    FakeTimeline,
    FakeTimelineItem,
    make_two_track_resolve,
)


def _session(resolve):
    return rc.connect(resolve)


def _build(fail_set_on=None):
    """Two-clip V1 timeline; optionally make one clip's SetProperty fail silently."""
    a = FakeTimelineItem("intro", 0, 48)
    b = FakeTimelineItem("midshot", 48, 72, fail_set=(fail_set_on == "midshot"))
    tl = FakeTimeline("T")
    tl.add_track("video", 1, [a, b])
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())
    return FakeResolve(project=proj), a, b, tl


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def test_add_marker_maps_to_native_call():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "add_marker", "clip_id": _id(ir, "intro"), "frame": 4, "color": "Blue", "name": "hook"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok
    assert ("AddMarker", 4, "Blue", "hook", "", 1, "") in a.calls
    assert 4.0 in a.GetMarkers()


def test_set_property_and_rename_map_to_native_calls():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "set_property", "clip_id": _id(ir, "intro"), "key": "ZoomX", "value": 1.5},
        {"op": "set_property", "clip_id": _id(ir, "midshot"), "key": "name", "value": "hero"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok
    assert ("SetProperty", "ZoomX", 1.5) in a.calls
    assert ("SetName", "hero") in b.calls
    assert b.GetName() == "hero"


def test_silent_false_aborts_and_rolls_back():
    # Plan: add a marker on intro (succeeds), then set a property on midshot (returns False).
    # The whole transaction must abort AND the intro marker must be rolled back.
    resolve, a, b, tl = _build(fail_set_on="midshot")
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "add_marker", "clip_id": _id(ir, "intro"), "frame": 2, "color": "Red"},
        {"op": "set_property", "clip_id": _id(ir, "midshot"), "key": "ZoomX", "value": 2.0},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("rolled back" in e for e in res.errors)
    # marker was added then removed -> net zero markers on intro
    assert a.GetMarkers() == {}
    assert ("DeleteMarkerAtFrame", 2) in a.calls


def test_delete_maps_to_timeline_deleteclips():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": _id(ir, "midshot")}]})
    res = adapter.apply(plan, session)
    assert res.ok
    assert any(call[0] == "DeleteClips" for call in tl.calls)


def test_non_live_ops_are_reported_for_rebuild_not_applied():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    # a move is valid but not live-applicable -> must surface as a rebuild warning, not error
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200}]})
    res = adapter.apply(plan, session)
    assert res.ok
    assert any("OTIO-rebuild" in w for w in res.warnings)


def test_apply_rejects_invalid_plan_before_touching_anything():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": "nope"}]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("UNKNOWN_CLIP" in e for e in res.errors)
    assert a.calls == [] and b.calls == []


@pytest.mark.live
def test_live_add_marker_roundtrip():
    """Real integration: add a marker and read it back. Skips unless Resolve is open."""
    try:
        rc.load_module()
    except rc.ResolveUnavailable:
        pytest.skip("DaVinciResolveScript not importable")
    session = rc.connect()
    if session is None or session.current_timeline() is None:
        pytest.skip("Resolve not running or no timeline open")
    adapter = ResolveAdapter()
    ir = adapter.snapshot(session)
    clips = ir.real_clips()
    if not clips:
        pytest.skip("timeline has no clips")
    plan = EditPlan.parse({"ops": [
        {"op": "add_marker", "clip_id": clips[0].id, "frame": 0, "color": "Blue", "name": "filmgrip"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
