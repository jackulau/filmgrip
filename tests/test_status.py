"""D9 — `film-grip status` doctor command + next-step guidance."""
from __future__ import annotations

from filmgrip.cli import main
from filmgrip.cli_status import render_status, status_guidance

READY = {"module_importable": True, "app_running": True, "is_studio": True,
         "product": "DaVinci Resolve Studio", "version": "20.0", "project_open": True,
         "timeline_open": True}


def _set_preflight(monkeypatch, report):
    monkeypatch.setattr("filmgrip.adapters.resolve_client.preflight", lambda: report)


def test_guidance_orders_most_blocking_first():
    assert "Install DaVinci Resolve" in status_guidance({})[0]
    assert "Open DaVinci Resolve" in status_guidance({"module_importable": True})[0]
    assert "Studio" in status_guidance(
        {"module_importable": True, "app_running": True, "is_studio": False})[0]
    assert "project" in status_guidance(
        {"module_importable": True, "app_running": True, "is_studio": True})[0].lower()
    assert "timeline" in status_guidance(
        {"module_importable": True, "app_running": True, "is_studio": True,
         "project_open": True})[0].lower()
    assert "Ready" in status_guidance(READY)[0]


def test_render_includes_sfx_and_editors():
    out = render_status(READY, "~/.filmgrip/sfx — 3 effect(s)", 8)
    assert "Ready" in out and "SFX library" in out and "8 supported" in out


def test_status_command_exits_zero_when_closed(monkeypatch, capsys, tmp_path):
    _set_preflight(monkeypatch, {"module_importable": True, "app_running": False})
    code = main(["status", "--sfx-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Resolve app running      : no" in out
    assert "Open DaVinci Resolve" in out


def test_status_command_reports_ready(monkeypatch, capsys, tmp_path):
    _set_preflight(monkeypatch, READY)
    code = main(["status", "--sfx-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0 and "Ready" in out
