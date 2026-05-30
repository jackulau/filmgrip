"""D11 — OTIO interchange adapter (FCPXML/EDL/AAF in+out)."""
from __future__ import annotations

import pytest

otio = pytest.importorskip("opentimelineio")
# Interchange adapters live in OpenTimelineIO-Plugins; skip cleanly if absent.
_have = set(otio.adapters.available_adapter_names())
pytestmark = pytest.mark.skipif(
    not {"fcp_xml", "cmx_3600"} <= _have,
    reason="OpenTimelineIO-Plugins (fcp_xml/cmx_3600) not installed",
)

from filmgrip.adapters.interchange import InterchangeAdapter  # noqa: E402
from filmgrip.protocol.editplan import EditPlan  # noqa: E402


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def test_reads_fcpxml_and_edl_into_ir(fixtures_dir):
    a = InterchangeAdapter()
    ir_fcp = a.snapshot(str(fixtures_dir / "sample.fcpxml"))
    assert {c.name for c in ir_fcp.real_clips()} == {"shotA", "shotB", "shotC"}
    ir_edl = a.snapshot(str(fixtures_dir / "sample.edl"))
    assert len(ir_edl.real_clips()) == 3


def test_trim_survives_fcpxml_roundtrip_with_lossy_warning(fixtures_dir, tmp_path):
    a = InterchangeAdapter()
    src = str(fixtures_dir / "sample.fcpxml")
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": _id(ir, "shotB"), "edge": "out", "delta": -20},
        {"op": "add_marker", "clip_id": _id(ir, "shotA"), "frame": 0, "color": "Blue", "name": "x"},
    ]})
    out = str(tmp_path / "out.fcpxml")
    res = a.apply(plan, src, out_path=out)
    assert res.ok
    # lossy warning fired because the source timeline has a dissolve transition
    assert any("transition" in w for w in res.warnings)

    # re-read the written FCPXML: the trim survived (shotB 80 -> 60)
    ir2 = a.snapshot(out)
    shotB = next(c for c in ir2.real_clips() if c.name == "shotB")
    assert shotB.duration == 60


def test_non_rebuild_ops_are_warned_not_applied(fixtures_dir, tmp_path):
    a = InterchangeAdapter()
    src = str(fixtures_dir / "sample.fcpxml")
    ir = a.snapshot(src)
    # a move is valid but the interchange rebuild path doesn't reposition — must warn, not crash
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "shotC"), "to_start": 500}]})
    res = a.apply(plan, src, out_path=str(tmp_path / "o.fcpxml"))
    assert res.ok
    assert any("move" in w and "not applied" in w for w in res.warnings)


def test_edl_target_refuses_multitrack_cleanly(fixtures_dir, tmp_path):
    a = InterchangeAdapter()
    # cut.otio has multiple video tracks; EDL is single-track cuts-only -> refuse, don't crash.
    from filmgrip.core.ir import TimelineIR
    ir = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": _id(ir, "intro"), "frame": 0}]})
    res = a.apply(plan, str(fixtures_dir / "cut.otio"), out_path=str(tmp_path / "o.edl"))
    assert res.ok is False
    assert any("cannot write cmx_3600" in e or "single" in e.lower() for e in res.errors)
    # the cuts-only warning was still surfaced
    assert any("cuts-only" in w or "EDL" in w for w in res.warnings)


def test_edl_single_track_roundtrips(fixtures_dir, tmp_path):
    a = InterchangeAdapter()
    src = str(fixtures_dir / "sample.edl")  # single track -> writable
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [{"op": "trim", "clip_id": _id(ir, "b"), "edge": "out", "delta": -12}]})
    res = a.apply(plan, src, out_path=str(tmp_path / "out.edl"))
    assert res.ok, res.errors
    ir2 = a.snapshot(str(tmp_path / "out.edl"))
    assert next(c for c in ir2.real_clips() if c.name == "b").duration == 60


def test_capabilities_and_invalid_plan(fixtures_dir, tmp_path):
    a = InterchangeAdapter()
    cap = a.capabilities()
    assert cap.role == "interchange"
    assert cap.requires_app_running is False
    assert cap.write_back is True
    bad = EditPlan.parse({"ops": [{"op": "delete", "clip_id": "nope"}]})
    res = a.apply(bad, str(fixtures_dir / "sample.fcpxml"), out_path=str(tmp_path / "x.fcpxml"))
    assert res.ok is False
    assert any("UNKNOWN_CLIP" in e for e in res.errors)
