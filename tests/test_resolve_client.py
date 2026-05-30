"""D2 — Resolve connection module + offline-mockable façade."""
from __future__ import annotations

import pytest

from filmgrip.adapters import resolve_client as rc
from tests.fakes import make_two_track_resolve


def test_connect_with_injected_fake_returns_session_and_reads_clips():
    fake = make_two_track_resolve()
    session = rc.connect(fake)
    assert session is not None
    assert session.product_name() == "DaVinci Resolve Studio"
    assert session.is_studio() is True
    proj = session.current_project()
    assert proj is not None and proj.GetName() == "Demo"
    tl = session.current_timeline()
    assert tl is not None and tl.GetName() == "Demo Timeline"
    v1 = session.timeline_items("video", 1)
    assert [it.GetName() for it in v1] == ["intro", "midshot", "outro"]
    a1 = session.timeline_items("audio", 1)
    assert [it.GetName() for it in a1] == ["music"]


def test_connect_with_falsy_handle_is_none():
    # scriptapp('Resolve') returns a falsy value when the app is closed; connect must
    # degrade to None rather than raise or wrap it.
    assert rc.connect(0) is None
    assert rc.connect(False) is None
    assert rc.connect("") is None


def test_require_raises_on_falsy():
    with pytest.raises(rc.ResolveOperationFailed):
        rc.require(None, "boom")
    with pytest.raises(rc.ResolveOperationFailed):
        rc.require(False, "boom")
    assert rc.require("ok", "boom") == "ok"
    assert rc.require([1], "boom") == [1]


def test_configure_env_returns_keys():
    env = rc.configure_env()
    assert set(env) == {"RESOLVE_SCRIPT_API", "RESOLVE_SCRIPT_LIB", "PYTHONPATH"}


def test_live_connect_degrades_gracefully():
    """On a machine with Resolve installed but closed, the real connect() returns None.

    Skips entirely when the scripting module is not importable (Resolve not installed), so
    this is safe in CI. The point: connect() never raises on a closed app.
    """
    try:
        rc.load_module()
    except rc.ResolveUnavailable:
        pytest.skip("DaVinciResolveScript not importable (Resolve not installed)")
    result = rc.connect()  # must not raise; None if app closed, ResolveSession if open
    assert result is None or isinstance(result, rc.ResolveSession)


def test_preflight_reports_structure():
    report = rc.preflight()
    for key in ("module_importable", "app_running", "project_open", "timeline_open", "env"):
        assert key in report
