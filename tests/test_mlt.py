"""D12 — Kdenlive/Shotcut MLT XML adapter."""
from __future__ import annotations

import pytest
from lxml import etree

from filmgrip.adapters.mlt import MltAdapter
from filmgrip.protocol.editplan import EditPlan


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


@pytest.mark.parametrize("ext", [".mlt", ".kdenlive"])
def test_parses_both_mlt_dialects(fixtures_dir, ext):
    ir = MltAdapter().snapshot(str(fixtures_dir / f"sample{ext}"))
    names = [c.name for c in ir.real_clips()]
    assert names == ["alpha", "bravo", "charlie"]
    by = {c.name: c for c in ir.real_clips()}
    assert (by["alpha"].start, by["alpha"].duration) == (0, 48)
    assert by["bravo"].duration == 72  # out=71 inclusive -> 72 frames


def test_move_reorders_and_stays_well_formed(fixtures_dir, tmp_path):
    a = MltAdapter()
    src = str(fixtures_dir / "sample.mlt")
    ir = a.snapshot(src)
    # alpha -> the free slot at the end (start 156); validator allows it, MLT repacks order
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "alpha"), "to_start": 156}]})
    out = str(tmp_path / "out.mlt")
    res = a.apply(plan, src, out_path=out)
    assert res.ok, res.errors

    # still valid XML, and the order changed (alpha is now last)
    etree.parse(out)  # raises if malformed
    ir2 = a.snapshot(out)
    assert [c.name for c in ir2.real_clips()] == ["bravo", "charlie", "alpha"]


def test_trim_adjusts_entry_inout(fixtures_dir, tmp_path):
    a = MltAdapter()
    src = str(fixtures_dir / "sample.mlt")
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [{"op": "trim", "clip_id": _id(ir, "bravo"), "edge": "out", "delta": -12}]})
    out = str(tmp_path / "out.mlt")
    assert a.apply(plan, src, out_path=out).ok
    ir2 = a.snapshot(out)
    assert next(c for c in ir2.real_clips() if c.name == "bravo").duration == 60


def test_delete_ripple_removes_clip(fixtures_dir, tmp_path):
    a = MltAdapter()
    src = str(fixtures_dir / "sample.mlt")
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": _id(ir, "bravo"), "ripple": True}]})
    out = str(tmp_path / "out.mlt")
    assert a.apply(plan, src, out_path=out).ok
    ir2 = a.snapshot(out)
    assert [c.name for c in ir2.real_clips()] == ["alpha", "charlie"]


def test_unsupported_op_is_rejected_not_faked(fixtures_dir, tmp_path):
    a = MltAdapter()
    src = str(fixtures_dir / "sample.mlt")
    ir = a.snapshot(src)
    out = tmp_path / "o.mlt"
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": _id(ir, "alpha"), "frame": 0}]})
    res = a.apply(plan, src, out_path=str(out))
    # add_marker is not representable in MLT — report it unsupported (ok=False), don't fake success.
    assert res.ok is False
    assert any("add_marker" in u for u in res.unsupported)
    assert not out.exists()


def test_capabilities():
    cap = MltAdapter().capabilities()
    assert cap.role == "interchange"
    assert cap.requires_app_running is False
    assert cap.write_back is True
