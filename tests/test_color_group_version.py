"""Color deliverable 4 — color_group + color_version: Resolve-live color organization, honestly
declared as having no interchange path."""
from __future__ import annotations

import opentimelineio as otio

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.interchange import REBUILD_OPS, InterchangeAdapter
from filmgrip.adapters.resolve_adapter import LIVE_OPS, ResolveAdapter
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import EditPlan, all_op_names
from filmgrip.protocol.validate import validate
from tests.fakes import FakeMediaPool, FakeProject, FakeResolve, FakeTimeline, FakeTimelineItem


def _session(resolve):
    return rc.connect(resolve)


def _build(fail_set_on=None):
    a = FakeTimelineItem("intro", 0, 48)
    b = FakeTimelineItem("midshot", 48, 72, fail_set=(fail_set_on == "midshot"))
    tl = FakeTimeline("T")
    tl.add_track("video", 1, [a, b])
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())
    return FakeResolve(project=proj), a, b, tl, proj


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def _two_clip_otio(tmp_path):
    rate = 24.0
    rt = lambda f: otio.opentime.RationalTime(f, rate)  # noqa: E731
    tl = otio.schema.Timeline(name="t")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for nm, st in (("A", 0), ("B", 48)):
        v.append(otio.schema.Clip(
            name=nm, media_reference=otio.schema.ExternalReference(target_url=f"/m/{nm}.mov"),
            source_range=otio.opentime.TimeRange(rt(0), rt(48))))
    tl.tracks.append(v)
    p = str(tmp_path / "t.otio")
    otio.adapters.write_to_file(tl, p)
    return p


# ----------------------------------------------------------------------- registration / honesty
def test_ops_registered_live_only():
    for op in ("color_group", "color_version"):
        assert op in all_op_names()
        assert op in LIVE_OPS            # live in Resolve
        assert op not in REBUILD_OPS     # honestly: no interchange/rebuild path


def test_interchange_reports_color_group_unsupported(tmp_path):
    src = _two_clip_otio(tmp_path)
    ir = TimelineIR.from_otio_file(src)
    cid = ir.real_clips()[0].id
    res = InterchangeAdapter().apply(
        EditPlan.parse({"ops": [{"op": "color_group", "clip_id": cid, "group": "skin"}]}),
        src, out_path=str(tmp_path / "o.otio"))
    assert not res.ok and res.unsupported     # no interchange path — never silently dropped


# ----------------------------------------------------------------------- color_group (live)
def test_color_group_assign_creates_and_assigns():
    resolve, a, _b, _tl, proj = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "color_group", "clip_id": _id(ir, "intro"), "group": "skintone"}]}), session)
    assert res.ok, res.errors
    assert ("AssignToColorGroup", "skintone") in a.calls
    assert [g.GetName() for g in proj.GetColorGroupsList()] == ["skintone"]


def test_color_group_is_reused_across_clips():
    resolve, a, b, _tl, proj = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "color_group", "clip_id": _id(ir, "intro"), "group": "night"},
        {"op": "color_group", "clip_id": _id(ir, "midshot"), "group": "night"}]}), session)
    assert res.ok, res.errors
    # one group, both clips assigned to it
    assert len(proj.GetColorGroupsList()) == 1
    assert ("AssignToColorGroup", "night") in a.calls
    assert ("AssignToColorGroup", "night") in b.calls


def test_color_group_remove():
    resolve, a, _b, _tl, _proj = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "color_group", "clip_id": _id(ir, "intro"), "group": "x", "action": "remove"}]}),
        session)
    assert res.ok
    assert ("RemoveFromColorGroup",) in a.calls


# ----------------------------------------------------------------------- color_version (live)
def test_color_version_add_local():
    resolve, a, _b, _tl, _proj = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "color_version", "clip_id": _id(ir, "intro"), "name": "client"}]}), session)
    assert res.ok, res.errors
    assert ("AddVersion", "client", 0) in a.calls


def test_color_version_load_remote_maps_type_1():
    resolve, a, _b, _tl, _proj = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "color_version", "clip_id": _id(ir, "intro"), "name": "hero",
         "action": "load", "version_type": "remote"}]}), session)
    assert res.ok, res.errors
    assert ("LoadVersionByName", "hero", 1) in a.calls


def test_color_version_add_rollback_deletes():
    resolve, a, _b, _tl, _proj = _build(fail_set_on="midshot")
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "color_version", "clip_id": _id(ir, "intro"), "name": "v1"},
        {"op": "set_property", "clip_id": _id(ir, "midshot"), "key": "ZoomX", "value": 2.0}]}),
        session)
    assert not res.ok
    assert ("DeleteVersionByName", "v1", 0) in a.calls   # rolled back


def test_validate_unknown_clip_rejected():
    resolve, _a, _b, _tl, _proj = _build()
    ir = ResolveAdapter().snapshot(_session(resolve))
    res = validate(EditPlan.parse({"ops": [
        {"op": "color_group", "clip_id": "nope", "group": "x"}]}), ir)
    assert not res.ok
