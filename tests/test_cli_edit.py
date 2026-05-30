"""D9 — precise, actionable CLI errors on the live edit path + selection confidence surfaced."""
from __future__ import annotations

from filmgrip.adapters.base import Selection
from filmgrip.adapters.resolve_adapter import ResolveAdapter
from filmgrip.cli import main
from filmgrip.core.ir import TimelineIR

FIX = "tests/fixtures/cut.otio"


def test_live_edit_reports_preflight_when_resolve_closed(monkeypatch, capsys):
    monkeypatch.setattr("filmgrip.adapters.resolve_client.preflight",
                        lambda: {"module_importable": True, "app_running": False})
    code = main(["edit", "--editor", "resolve", "add a marker"])
    out = capsys.readouterr().out
    assert code == 2
    assert "not reachable" in out and "External scripting" in out


def test_live_edit_no_prompt_explains(monkeypatch, capsys):
    monkeypatch.setattr("filmgrip.adapters.resolve_client.preflight",
                        lambda: {"module_importable": True, "app_running": True,
                                 "project_open": True, "timeline_open": True})
    monkeypatch.setattr("filmgrip.adapters.resolve_client.connect", lambda *a, **k: object())
    monkeypatch.setattr(ResolveAdapter, "snapshot", lambda self, s: TimelineIR.from_otio_file(FIX))
    monkeypatch.setattr(ResolveAdapter, "get_selection",
                        lambda self, s, ir=None: Selection(ids=["c1"], basis="x"))
    code = main(["edit", "--editor", "resolve"])  # no prompt
    out = capsys.readouterr().out
    assert code == 1 and "provide an instruction" in out


def test_live_edit_no_selection_is_actionable(monkeypatch, capsys):
    monkeypatch.setattr("filmgrip.adapters.resolve_client.preflight",
                        lambda: {"module_importable": True, "app_running": True,
                                 "project_open": True, "timeline_open": True})
    monkeypatch.setattr("filmgrip.adapters.resolve_client.connect", lambda *a, **k: object())
    monkeypatch.setattr(ResolveAdapter, "snapshot", lambda self, s: TimelineIR.from_otio_file(FIX))
    monkeypatch.setattr(ResolveAdapter, "get_selection",
                        lambda self, s, ir=None: Selection(ids=[], basis="x", confidence="reconstructed"))
    code = main(["edit", "--editor", "resolve", "add a blue marker"])
    out = capsys.readouterr().out
    assert code == 1
    assert "no clips selected" in out
    # confidence surfaced even when empty
    assert "[reconstructed]" in out


def test_live_edit_surfaces_selection_confidence(monkeypatch, capsys, tmp_path):
    # A recorded plan avoids the network; we only assert the confidence line prints.
    import json
    ir = TimelineIR.from_otio_file(FIX)
    cid = ir.real_clips()[0].id
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps({"ops": [{"op": "add_marker", "clip_id": cid, "frame": 0}]}))

    monkeypatch.setattr("filmgrip.adapters.resolve_client.preflight",
                        lambda: {"module_importable": True, "app_running": True,
                                 "project_open": True, "timeline_open": True})
    monkeypatch.setattr("filmgrip.adapters.resolve_client.connect", lambda *a, **k: object())
    monkeypatch.setattr(ResolveAdapter, "snapshot", lambda self, s: ir)
    monkeypatch.setattr(ResolveAdapter, "get_selection",
                        lambda self, s, ir=None: Selection(ids=[cid], basis="x", confidence="reconstructed"))
    code = main(["edit", "--editor", "resolve", "--plan", str(plan), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "selection: 1 clip(s) [reconstructed]" in out
