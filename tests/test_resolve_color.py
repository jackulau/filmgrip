"""Color deliverable 3 — Resolve live color wiring: set_cdl → SetCDL, apply_lut → SetLUT
(Graph-object v19+ with TimelineItem fallback), plus write-only-CDL rollback and LUT restore.

All fake-backed (tests/fakes), so they run without DaVinci Resolve installed."""
from __future__ import annotations

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.resolve_adapter import LIVE_OPS, ResolveAdapter
from filmgrip.protocol.editplan import EditPlan
from tests.fakes import FakeMediaPool, FakeProject, FakeResolve, FakeTimeline, FakeTimelineItem


def _session(resolve):
    return rc.connect(resolve)


def _build(fail_set_on=None, color_graph=True):
    a = FakeTimelineItem("intro", 0, 48, color_graph=color_graph)
    b = FakeTimelineItem("midshot", 48, 72, fail_set=(fail_set_on == "midshot"),
                         color_graph=color_graph)
    tl = FakeTimeline("T")
    tl.add_track("video", 1, [a, b])
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())
    return FakeResolve(project=proj), a, b, tl


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def test_color_ops_are_live():
    assert {"set_cdl", "apply_lut"} <= set(LIVE_OPS)


def test_set_cdl_maps_to_native_setcdl():
    resolve, a, _b, _tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": _id(ir, "intro"),
         "slope": [1.2, 1.0, 0.9], "offset": [0.0, 0.0, 0.0], "power": [1.0, 1.0, 1.0],
         "saturation": 1.15, "node_index": 2}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    setcdl = [c for c in a.calls if c[0] == "SetCDL"]
    assert setcdl, a.calls
    payload = setcdl[0][1]
    assert payload["NodeIndex"] == "2"
    assert payload["Slope"] == "1.2 1 0.9"
    assert payload["Saturation"] == "1.15"


def test_apply_lut_uses_graph_setlut_on_v19():
    resolve, a, _b, _tl = _build(color_graph=True)
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(ir, "intro"), "path": "Kodak2383", "node_index": 3}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    graph = a.GetNodeGraph()
    assert ("SetLUT", 3, "Kodak2383") in graph.calls
    # the LUT went to the Graph, NOT the item, on a v19+ build
    assert not any(c[0] == "SetLUT" for c in a.calls)


def test_apply_lut_falls_back_to_item_setlut_on_older_build():
    resolve, a, _b, _tl = _build(color_graph=False)
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(ir, "intro"), "path": "Kodak2383", "node_index": 1}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert ("SetLUT", 1, "Kodak2383") in a.calls


def test_set_cdl_is_write_only_with_warning():
    resolve, a, _b, _tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": _id(ir, "intro"), "saturation": 1.2}]})
    res = adapter.apply(plan, session)
    assert res.ok
    assert any("write-only" in w for w in res.warnings)


def test_set_cdl_rollback_resets_node_to_identity():
    # set_cdl on intro (succeeds) then a failing set_property on midshot → abort + rollback.
    resolve, a, _b, _tl = _build(fail_set_on="midshot")
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": _id(ir, "intro"), "slope": [1.5, 1.5, 1.5], "node_index": 1},
        {"op": "set_property", "clip_id": _id(ir, "midshot"), "key": "ZoomX", "value": 2.0}]})
    res = adapter.apply(plan, session)
    assert not res.ok                                   # the whole plan aborted
    setcdls = [c for c in a.calls if c[0] == "SetCDL"]
    assert len(setcdls) == 2                            # applied grade, then the rollback reset
    assert setcdls[-1][1]["Slope"] == "1 1 1"          # rolled back to identity
    assert setcdls[-1][1]["Saturation"] == "1"


def test_apply_lut_rollback_restores_prior_lut():
    resolve, a, _b, _tl = _build(fail_set_on="midshot")
    # seed a prior LUT on node 2 so the inverse has something to restore.
    a.GetNodeGraph().SetLUT(2, "/luts/old.cube")
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(ir, "intro"), "path": "/luts/new.cube", "node_index": 2},
        {"op": "set_property", "clip_id": _id(ir, "midshot"), "key": "ZoomX", "value": 2.0}]})
    res = adapter.apply(plan, session)
    assert not res.ok
    graph = a.GetNodeGraph()
    setluts = [c for c in graph.calls if c[0] == "SetLUT"]
    # seed, apply new, rollback to old
    assert setluts[-1] == ("SetLUT", 2, "/luts/old.cube")


def test_color_plan_end_to_end_applies_both():
    resolve, a, _b, _tl = _build()
    adapter = ResolveAdapter()
    session = _session(resolve)
    ir = adapter.snapshot(session)
    intro = _id(ir, "intro")
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": intro, "saturation": 1.1, "node_index": 1},
        {"op": "apply_lut", "clip_id": intro, "path": "Kodak2383", "node_index": 2}]})
    res = adapter.apply(plan, session)
    assert res.ok, res.errors
    assert any(c[0] == "SetCDL" for c in a.calls)
    assert ("SetLUT", 2, "Kodak2383") in a.GetNodeGraph().calls
