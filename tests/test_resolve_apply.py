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
    # rename routes through the MediaPoolItem 'Clip Name' (Resolve has no TimelineItem.SetName)
    assert b.GetMediaPoolItem().GetClipProperty("Clip Name") == "hero"


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


def _media_pool(resolve):
    return resolve.GetProjectManager().GetCurrentProject().GetMediaPool()


def _rebuilt_ir(resolve):
    """Read the OTIO the adapter handed to ImportTimelineFromFile — the actual rebuilt timeline."""
    from filmgrip.core.ir import TimelineIR
    path = _media_pool(resolve).imported_timelines[-1]
    return TimelineIR.from_otio_file(path)


def test_move_applies_live_via_otio_rebuild():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    # A move is NOT live-applicable, so it must route through export→mutate→import and ACTUALLY
    # take effect (the old behaviour merely warned + no-op'd).
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert any(c[0] == "Export" for c in tl.calls)              # timeline was exported
    assert _media_pool(resolve).imported_timelines                # a rebuilt timeline was imported
    assert any("rebuild" in w for w in res.warnings)             # lossy-rebuild surfaced honestly
    assert _start_of(_rebuilt_ir(resolve), "midshot") == 200     # the move really landed


def test_trim_applies_live_via_otio_rebuild():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "trim", "clip_id": _id(ir, "midshot"), "edge": "out", "delta": -24}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    rebuilt = _rebuilt_ir(resolve)
    assert next(c for c in rebuilt.real_clips() if c.name == "midshot").duration == 48


def test_split_applies_live_via_otio_rebuild():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "split", "clip_id": _id(ir, "midshot"), "at_frame": 84}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    rebuilt = _rebuilt_ir(resolve)
    halves = [c for c in rebuilt.real_clips() if c.name == "midshot"]
    assert len(halves) == 2 and {c.duration for c in halves} == {36}


def test_rebuild_import_failure_leaves_original_intact():
    resolve, a, b, tl = _build()
    # Make ImportTimelineFromFile fail silently (Resolve's falsy-on-failure behaviour).
    _media_pool(resolve).ImportTimelineFromFile = lambda path, options=None: None
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200}]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("original timeline intact" in e for e in res.errors)


def _start_of(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).start


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
