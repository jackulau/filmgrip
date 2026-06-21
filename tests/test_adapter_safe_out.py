"""D1 — offline write-back adapters are non-destructive by default.

The three offline adapters used to default ``out = out_path or source``, silently overwriting the
user's project when ``apply()`` ran without an explicit destination. They now route through
``base.safe_out_path``, which defaults to a ``<stem>.edited<ext>`` sibling. These tests pin both the
helper and the end-to-end guarantee: applying with no ``out_path`` leaves the source bytes untouched
and writes the edit to the sibling instead.
"""
from __future__ import annotations

import shutil

import pytest

from filmgrip.adapters import base
from filmgrip.adapters.capcut import CapcutAdapter
from filmgrip.adapters.interchange import InterchangeAdapter
from filmgrip.adapters.mlt import MltAdapter
from filmgrip.protocol.editplan import EditPlan


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


# -- helper -----------------------------------------------------------------------
def test_safe_out_path_defaults_to_sibling_on_none():
    out = base.safe_out_path("/tmp/proj.mlt", None)
    assert out == "/tmp/proj.edited.mlt"
    assert out != "/tmp/proj.mlt"  # never the source


def test_safe_out_path_returns_explicit_unchanged():
    # An explicit destination always wins, untouched (even one that looks like the source).
    assert base.safe_out_path("/tmp/proj.mlt", "/tmp/x.mlt") == "/tmp/x.mlt"
    assert base.safe_out_path("/tmp/proj.mlt", "/tmp/proj.mlt") == "/tmp/proj.mlt"


def test_safe_out_path_custom_suffix():
    assert base.safe_out_path("/a/b.json", None, suffix=".out") == "/a/b.out.json"


# -- per-adapter: no out_path must NOT touch the source ---------------------------
def test_mlt_apply_without_out_path_does_not_overwrite_source(fixtures_dir, tmp_path):
    a = MltAdapter()
    src = tmp_path / "proj.mlt"
    shutil.copy(fixtures_dir / "sample.mlt", src)
    before = src.read_bytes()
    ir = a.snapshot(str(src))
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": _id(ir, "bravo"), "edge": "out", "delta": -12}]})
    res = a.apply(plan, str(src))  # no out_path
    assert res.ok, res.errors
    assert src.read_bytes() == before  # source untouched
    assert (tmp_path / "proj.edited.mlt").exists()  # edit landed on the sibling


def test_capcut_apply_without_out_path_does_not_overwrite_source(fixtures_dir, tmp_path):
    a = CapcutAdapter()
    src = tmp_path / "draft.json"
    shutil.copy(fixtures_dir / "capcut_draft.json", src)
    before = src.read_bytes()
    ir = a.snapshot(str(src))
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": _id(ir, "clip_two")}]})
    res = a.apply(plan, str(src))  # no out_path
    assert res.ok, res.errors
    assert src.read_bytes() == before  # source untouched
    sibling = tmp_path / "draft.edited.json"
    assert sibling.exists()
    assert "clip_two" not in {c.name for c in a.snapshot(str(sibling)).real_clips()}


def test_interchange_apply_without_out_path_does_not_overwrite_source(tmp_path):
    # otio_json is built into OpenTimelineIO, so this path needs no interchange extra.
    otio = pytest.importorskip("opentimelineio")
    rate = 24.0

    def rt(f):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="t")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v.append(otio.schema.Clip(
        name="A", media_reference=otio.schema.ExternalReference(target_url="/m/a.mov"),
        source_range=otio.opentime.TimeRange(rt(0), rt(48))))
    tl.tracks.append(v)
    src = tmp_path / "proj.otio"
    otio.adapters.write_to_file(tl, str(src))

    a = InterchangeAdapter()
    before = src.read_bytes()
    ir = a.snapshot(str(src))
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": _id(ir, "A"), "edge": "out", "delta": -12}]})
    res = a.apply(plan, str(src))  # no out_path
    assert res.ok, res.errors
    assert src.read_bytes() == before  # source untouched
    sibling = tmp_path / "proj.edited.otio"
    assert sibling.exists()
    assert next(c for c in a.snapshot(str(sibling)).real_clips() if c.name == "A").duration == 36
