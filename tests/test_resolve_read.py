"""D7 — Resolve adapter read side (snapshot + selection)."""
from __future__ import annotations

import pytest

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.resolve_adapter import ResolveAdapter
from filmgrip.core.ir import TimelineIR
from tests.fakes import make_two_track_resolve


@pytest.fixture
def session():
    return rc.connect(make_two_track_resolve())


def test_snapshot_builds_a_valid_ir(session):
    ir = ResolveAdapter().snapshot(session)
    assert isinstance(ir, TimelineIR)
    names = [c.name for c in ir.real_clips()]
    assert names == ["intro", "midshot", "outro", "music"]
    by_name = {c.name: c for c in ir.real_clips()}
    assert (by_name["intro"].start, by_name["intro"].duration) == (0, 48)
    assert (by_name["midshot"].start, by_name["midshot"].duration) == (48, 72)
    assert by_name["music"].track_kind == "audio"
    # media references survived (basename)
    assert by_name["midshot"].src_ref == "midshot.mov"


def test_snapshot_attaches_native_handles(session):
    ir = ResolveAdapter().snapshot(session)
    for c in ir.real_clips():
        assert c.native is not None
        assert c.native.GetName() == c.name


def test_get_selection_returns_current_item_and_media_matches(session):
    adapter = ResolveAdapter()
    ir = adapter.snapshot(session)
    sel = adapter.get_selection(session, ir)
    names = {ir.clip(i).name for i in sel.ids}
    # current video item is 'intro'; selected media is midshot's -> 'midshot'
    assert "intro" in names
    assert "midshot" in names
    assert sel.basis == "current_video_item+media_pool"
    assert "no true multi-clip" in sel.note


def test_selection_header_is_compact(session):
    adapter = ResolveAdapter()
    ir = adapter.snapshot(session)
    sel = adapter.get_selection(session, ir)
    hdr = sel.as_header(ir)
    assert set(hdr) >= {"seq", "r", "sel", "basis"}
    assert isinstance(hdr["sel"], list)


def test_capabilities_are_honest():
    cap = ResolveAdapter().capabilities()
    assert cap.write_back is True
    assert cap.live_selection is False         # the honest part
    assert cap.requires_app_running is True
    assert cap.role == "flagship-native"


@pytest.mark.live
def test_live_snapshot_when_resolve_open():
    """Real integration: only runs when DaVinci Resolve is open with a project + timeline."""
    try:
        rc.load_module()
    except rc.ResolveUnavailable:
        pytest.skip("DaVinciResolveScript not importable")
    session = rc.connect()
    if session is None or session.current_timeline() is None:
        pytest.skip("Resolve not running or no timeline open")
    ir = ResolveAdapter().snapshot(session)
    assert ir.duration >= 0
    sel = ResolveAdapter().get_selection(session, ir)
    assert isinstance(sel.ids, list)
