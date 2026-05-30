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


def test_add_transition_is_warned_not_applied(fixtures_dir, tmp_path):
    a = InterchangeAdapter()
    src = str(fixtures_dir / "sample.fcpxml")
    ir = a.snapshot(src)
    # add_transition is the one structural op the rebuild path leaves to the live editor — must
    # warn, not crash. (move/insert/ripple now DO apply via OTIO — see test_structural_ops_*.)
    plan = EditPlan.parse({"ops": [
        {"op": "add_transition", "clip_id": _id(ir, "shotB"), "edge": "out", "type": "cross_dissolve"},
    ]})
    res = a.apply(plan, src, out_path=str(tmp_path / "o.fcpxml"))
    assert res.ok
    assert any("add_transition" in w and "not applied" in w for w in res.warnings)


def _build_otio(tmp_path, tracks):
    """Write a controlled single-rate OTIO timeline; ``tracks`` = {code: [(name|None, start, dur)]}.

    A ``None`` name makes a gap; a string makes a clip. Positions are absolute frames so the
    fixture's gaps are explicit and the move/insert/ripple maths is unambiguous.
    """
    import opentimelineio as otio
    rate = 24.0

    def rt(f):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="ctl")
    tl.global_start_time = rt(0)
    for code, items in tracks.items():
        kind = otio.schema.TrackKind.Video if code[0] == "v" else otio.schema.TrackKind.Audio
        tr = otio.schema.Track(name=code.upper(), kind=kind)
        for name, _start, dur in items:
            if name is None:
                tr.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(rt(0), rt(dur))))
            else:
                tr.append(otio.schema.Clip(
                    name=name,
                    media_reference=otio.schema.ExternalReference(target_url=f"/m/{name}.mov"),
                    source_range=otio.opentime.TimeRange(rt(0), rt(dur))))
        tl.tracks.append(tr)
    path = str(tmp_path / "ctl.otio")
    otio.adapters.write_to_file(tl, path)
    return path


def _start_of(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).start


def test_structural_move_applies_via_otio(tmp_path):
    a = InterchangeAdapter()
    # clipA 0..48, gap 48..120, clipB 120..180 — move clipB into the gap at 48.
    src = _build_otio(tmp_path, {"v1": [("clipA", 0, 48), (None, 48, 72), ("clipB", 120, 60)]})
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [{"op": "move", "clip_id": _id(ir, "clipB"), "to_start": 48}]})
    out = str(tmp_path / "moved.otio")
    res = a.apply(plan, src, out_path=out)
    assert res.ok, res.errors
    ir2 = a.snapshot(out)
    assert _start_of(ir2, "clipB") == 48
    assert _start_of(ir2, "clipA") == 0  # source neighbour held position


def test_structural_insert_applies_via_otio(tmp_path):
    a = InterchangeAdapter()
    src = _build_otio(tmp_path, {"v1": [("clipA", 0, 48), (None, 48, 152), ("clipC", 200, 60)]})
    plan = EditPlan.parse({"ops": [
        {"op": "insert", "src_ref": "broll.mov", "track": "v1", "at_start": 100, "duration": 30},
    ]})
    out = str(tmp_path / "inserted.otio")
    res = a.apply(plan, src, out_path=out)
    assert res.ok, res.errors
    ir2 = a.snapshot(out)
    ins = next(c for c in ir2.real_clips() if c.name == "broll.mov")
    assert ins.start == 100 and ins.duration == 30
    assert _start_of(ir2, "clipC") == 200  # untouched


def test_structural_ripple_applies_via_otio(tmp_path):
    a = InterchangeAdapter()
    # clipA 0..48, clipB 48..96 — ripple everything at/after 48 by +24.
    src = _build_otio(tmp_path, {"v1": [("clipA", 0, 48), ("clipB", 48, 48)]})
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [{"op": "ripple", "from_frame": 48, "delta": 24, "track": "v1"}]})
    out = str(tmp_path / "rippled.otio")
    res = a.apply(plan, src, out_path=out)
    assert res.ok, res.errors
    ir2 = a.snapshot(out)
    assert _start_of(ir2, "clipA") == 0    # before the ripple point — unmoved
    assert _start_of(ir2, "clipB") == 72   # pushed right by 24


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
