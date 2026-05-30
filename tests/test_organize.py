"""D7 — organize ops: add_track / rename_track / create_bin / move_to_bin."""
from __future__ import annotations

import pytest

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.resolve_adapter import ResolveAdapter
from filmgrip.protocol import validate as V
from filmgrip.protocol.editplan import EditPlan
from filmgrip.core.ir import TimelineIR
from tests.fakes import FakeMediaPool, FakeProject, FakeResolve, FakeTimeline, FakeTimelineItem


def _session(resolve):
    return rc.connect(resolve)


def _build():
    v1 = [FakeTimelineItem("intro", 0, 48), FakeTimelineItem("midshot", 48, 72)]
    a1 = [FakeTimelineItem("music", 0, 120, src_path="/media/music.wav")]
    tl = FakeTimeline("T")
    tl.add_track("video", 1, v1)
    tl.add_track("audio", 1, a1)
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())
    return FakeResolve(project=proj), tl


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def test_add_track_video():
    resolve, tl = _build()
    session = _session(resolve)
    before = tl.GetTrackCount("video")
    res = ResolveAdapter().apply(EditPlan.parse({"ops": [{"op": "add_track", "kind": "video"}]}), session)
    assert res.ok, res.errors
    assert ("AddTrack", "video", None) in tl.calls
    assert tl.GetTrackCount("video") == before + 1


def test_rename_track_maps_to_settrackname_and_is_reversible():
    resolve, tl = _build()
    session = _session(resolve)
    res = ResolveAdapter().apply(
        EditPlan.parse({"ops": [{"op": "rename_track", "track": "v1", "name": "B-ROLL"}]}), session)
    assert res.ok, res.errors
    assert ("SetTrackName", "video", 1, "B-ROLL") in tl.calls
    assert tl.GetTrackName("video", 1) == "B-ROLL"


def test_create_bin_at_root():
    resolve, tl = _build()
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    res = ResolveAdapter().apply(EditPlan.parse({"ops": [{"op": "create_bin", "name": "selects"}]}), session)
    assert res.ok, res.errors
    assert any(f.GetName() == "selects" for f in mp.GetRootFolder().GetSubFolderList())


def test_create_bin_under_existing_parent():
    resolve, tl = _build()
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    plan = EditPlan.parse({"ops": [
        {"op": "create_bin", "name": "footage"},
        {"op": "create_bin", "name": "drone", "parent": "footage"},
    ]})
    res = ResolveAdapter().apply(plan, session)
    assert res.ok, res.errors
    footage = next(f for f in mp.GetRootFolder().GetSubFolderList() if f.GetName() == "footage")
    assert any(f.GetName() == "drone" for f in footage.GetSubFolderList())


def test_create_bin_missing_parent_falls_back_to_root_with_warning():
    resolve, tl = _build()
    session = _session(resolve)
    res = ResolveAdapter().apply(
        EditPlan.parse({"ops": [{"op": "create_bin", "name": "x", "parent": "ghost"}]}), session)
    assert res.ok
    assert any("not found" in w for w in res.warnings)


def test_move_to_bin_moves_media_and_autocreates_bin():
    resolve, tl = _build()
    session = _session(resolve)
    mp = resolve.GetProjectManager().GetCurrentProject().GetMediaPool()
    ir = ResolveAdapter().snapshot(session)
    plan = EditPlan.parse({"ops": [{"op": "move_to_bin", "clip_id": _id(ir, "intro"), "bin": "keepers"}]})
    res = ResolveAdapter().apply(plan, session)
    assert res.ok, res.errors
    assert mp.moved and mp.moved[-1][1] == "keepers"
    keepers = next(f for f in mp.GetRootFolder().GetSubFolderList() if f.GetName() == "keepers")
    assert len(keepers.GetClipList()) == 1


def test_move_to_bin_unknown_clip_rejected_at_validation():
    resolve, tl = _build()
    session = _session(resolve)
    res = ResolveAdapter().apply(
        EditPlan.parse({"ops": [{"op": "move_to_bin", "clip_id": "nope", "bin": "x"}]}), session)
    assert res.ok is False
    assert any("UNKNOWN_CLIP" in e for e in res.errors)


def test_rename_track_unknown_track_rejected():
    ir = TimelineIR.from_otio_file("tests/fixtures/cut.otio")
    plan = EditPlan.parse({"ops": [{"op": "rename_track", "track": "v9", "name": "x"}]})
    res = V.validate(plan, ir)
    assert not res.ok and V.TRACK_NOT_FOUND in res.codes()


def test_interchange_warns_organize_ops_not_applied(fixtures_dir, tmp_path):
    # Organize ops have no meaning in a flat interchange file — must warn, not crash.
    from filmgrip.adapters.interchange import InterchangeAdapter
    src = str(fixtures_dir / "sample.fcpxml")
    a = InterchangeAdapter()
    plan = EditPlan.parse({"ops": [{"op": "create_bin", "name": "selects"}]})
    res = a.apply(plan, src, out_path=str(tmp_path / "o.fcpxml"))
    assert res.ok
    assert any("create_bin" in w and "not applied" in w for w in res.warnings)
