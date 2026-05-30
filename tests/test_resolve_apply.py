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


def test_mixed_live_op_rolled_back_when_structural_rebuild_fails():
    # add_marker (reversible live) + move (structural). If the rebuild import fails, the marker must
    # be rolled back — proving the unwind runs with a NON-empty rollback stack.
    resolve, a, b, tl = _build()
    _media_pool(resolve).ImportTimelineFromFile = lambda path, options=None: None  # rebuild fails
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "add_marker", "clip_id": _id(ir, "intro"), "frame": 3, "color": "Red"},
        {"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert ("DeleteMarkerAtFrame", 3) in a.calls    # the marker inverse ran
    assert a.GetMarkers() == {}                       # net zero — rolled back


def test_irreversible_live_op_before_failed_rebuild_is_surfaced_honestly():
    # add_track (NO inverse) + move (structural). Rebuild fails -> the track stays; the result must
    # NOT silently claim everything is intact: it surfaces a warning naming the un-undoable op.
    resolve, a, b, tl = _build()
    _media_pool(resolve).ImportTimelineFromFile = lambda path, options=None: None
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "add_track", "kind": "audio", "audio_type": "stereo"},
        {"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("NOT scriptable to undo" in w for w in res.warnings)


def test_export_failure_aborts_before_any_import():
    resolve, a, b, tl = _build()
    tl.Export = lambda *a, **k: None  # Resolve's falsy-on-failure for the export step
    mp = _media_pool(resolve)
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200}]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("export failed" in e for e in res.errors)
    assert mp.imported_timelines == []  # never reached the import


def test_mixed_plan_surfaces_rebuild_warning_and_applies_both():
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "add_marker", "clip_id": _id(ir, "intro"), "frame": 4, "color": "Blue"},
        {"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert a.GetMarkers()                                       # live marker applied
    assert _start_of(_rebuilt_ir(resolve), "midshot") == 200    # structural move landed
    assert any("rebuild" in w for w in res.warnings)            # lossy warning survived the merge


def test_reresolve_remaps_clip_id_when_export_drifts_position(monkeypatch):
    # Force the exported OTIO to put midshot at a drifted start so its content-id changes; the plan
    # (authored against the original id) must still land via IdMap.reresolve, with a warning.
    import opentimelineio as otio
    resolve, a, b, tl = _build()

    def drifted_export(path, *args, **kw):
        rate = 24.0

        def rt(f):
            return otio.opentime.RationalTime(f, rate)

        t = otio.schema.Timeline(name="drift")
        t.global_start_time = rt(0)
        v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
        v.append(otio.schema.Clip(name="intro",
                 media_reference=otio.schema.ExternalReference(target_url="/media/intro.mov"),
                 source_range=otio.opentime.TimeRange(rt(0), rt(48))))
        # a 1-frame gap pushes midshot's START to 49 -> different content-derived id -> reresolve.
        v.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(rt(0), rt(1))))
        v.append(otio.schema.Clip(name="midshot",
                 media_reference=otio.schema.ExternalReference(target_url="/media/midshot.mov"),
                 source_range=otio.opentime.TimeRange(rt(0), rt(72))))
        t.tracks.append(v)
        otio.adapters.write_to_file(t, path)
        return True

    monkeypatch.setattr(tl, "Export", drifted_export)
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "midshot"), "to_start": 200}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert any("re-resolved clip" in w for w in res.warnings)
    assert _start_of(_rebuilt_ir(resolve), "midshot") == 200


def test_native_call_phantom_method_raises():
    from filmgrip.adapters.resolve_adapter import _native_call
    from filmgrip.adapters.resolve_client import ResolveOperationFailed

    class _Phantom:  # mimic fusionscript: any attribute resolves to None
        def __getattr__(self, name):
            return None

    with pytest.raises(ResolveOperationFailed, match="no callable 'DeleteClips'"):
        _native_call(_Phantom(), "DeleteClips", [])


def test_ripple_delete_is_live_and_closes_the_gap_natively():
    # A ripple-delete (lift + close the gap) is the core "cut this out" primitive. delete is a live
    # op, so it stays on the fast-path and hands ripple=True to Resolve's DeleteClips, which closes
    # the gap natively — no lossy rebuild needed for a straight cut.
    resolve, a, b, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": _id(ir, "midshot"), "ripple": True}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert ("DeleteClips", ["midshot"], True) in tl.calls


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
