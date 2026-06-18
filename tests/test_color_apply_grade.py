"""Color deliverable 5 — apply_grade: the match-this-shot primitive (CopyGrades from a hero clip,
or ApplyGradeFromDRX for a saved PowerGrade)."""
from __future__ import annotations

import pytest

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.resolve_adapter import LIVE_OPS, ResolveAdapter
from filmgrip.protocol import validate as V
from filmgrip.protocol.editplan import EditPlan, all_op_names
from tests.fakes import FakeMediaPool, FakeProject, FakeResolve, FakeTimeline, FakeTimelineItem


def _session(resolve):
    return rc.connect(resolve)


def _build():
    a = FakeTimelineItem("hero", 0, 48)
    b = FakeTimelineItem("shot_b", 48, 48)
    c = FakeTimelineItem("shot_c", 96, 48)
    tl = FakeTimeline("T")
    tl.add_track("video", 1, [a, b, c])
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())
    return FakeResolve(project=proj), a, b, c, tl


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


# ----------------------------------------------------------------------- op + parse validation
def test_apply_grade_registered_live_only():
    assert "apply_grade" in all_op_names()
    assert "apply_grade" in LIVE_OPS


def test_apply_grade_needs_exactly_one_source():
    with pytest.raises(Exception):                       # neither source
        EditPlan.parse({"ops": [{"op": "apply_grade", "to_clips": ["x"]}]})
    with pytest.raises(Exception):                       # both sources
        EditPlan.parse({"ops": [{"op": "apply_grade", "to_clips": ["x"],
                                 "from_clip": "y", "drx_path": "/g.drx"}]})


def test_apply_grade_needs_targets():
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [{"op": "apply_grade", "to_clips": [], "from_clip": "y"}]})


# ----------------------------------------------------------------------- CopyGrades (hero → others)
def test_copy_grade_from_hero():
    resolve, a, b, c, _tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "apply_grade", "from_clip": _id(ir, "hero"),
         "to_clips": [_id(ir, "shot_b"), _id(ir, "shot_c")]}]}), session)
    assert res.ok, res.errors
    copy = [call for call in a.calls if call[0] == "CopyGrades"]
    assert copy and copy[0][1] == ["shot_b", "shot_c"]
    assert any("not reversible" in w for w in res.warnings)


def test_copy_grade_unknown_target_rejected():
    resolve, _a, _b, _c, _tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "apply_grade", "from_clip": _id(ir, "hero"), "to_clips": ["deadbeef"]}]}), session)
    assert not res.ok and V.UNKNOWN_CLIP in " ".join(res.errors) or not res.ok


# ----------------------------------------------------------------------- ApplyGradeFromDRX
def test_apply_drx_grade(tmp_path):
    drx = tmp_path / "look.drx"
    drx.write_text("<drx/>")
    resolve, _a, _b, _c, tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "apply_grade", "drx_path": str(drx), "grade_mode": 1,
         "to_clips": [_id(ir, "shot_b"), _id(ir, "shot_c")]}]}), session)
    assert res.ok, res.errors
    drx_calls = [call for call in tl.calls if call[0] == "ApplyGradeFromDRX"]
    assert drx_calls and drx_calls[0][1] == str(drx) and drx_calls[0][2] == 1
    assert drx_calls[0][3] == ["shot_b", "shot_c"]


def test_apply_drx_missing_file_rejected():
    resolve, _a, _b, _c, _tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    res = adapter.apply(EditPlan.parse({"ops": [
        {"op": "apply_grade", "drx_path": "/tmp/nope.drx", "to_clips": [_id(ir, "shot_b")]}]}),
        session)
    assert not res.ok


def test_apply_drx_non_drx_extension_rejected(tmp_path):
    f = tmp_path / "look.cube"
    f.write_text("x")
    resolve, _a, _b, _c, _tl = _build()
    ir = ResolveAdapter().snapshot(_session(resolve))
    res = V.validate(EditPlan.parse({"ops": [
        {"op": "apply_grade", "drx_path": str(f), "to_clips": [_id(ir, "shot_b")]}]}), ir)
    assert not res.ok and V.DRX_NOT_FOUND in res.codes()


def test_dry_run_describes_apply_grade():
    resolve, _a, _b, _c, _tl = _build()
    ir = ResolveAdapter().snapshot(_session(resolve))
    out = V.dry_run(EditPlan.parse({"ops": [
        {"op": "apply_grade", "from_clip": _id(ir, "hero"),
         "to_clips": [_id(ir, "shot_b"), _id(ir, "shot_c")]}]}), ir)
    assert "apply grade of hero → 2 clip(s)" in out
