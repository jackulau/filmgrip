"""D9 — `film-grip status` doctor command + next-step guidance.

D12 — perception-toolchain doctor (ffmpeg/ffprobe/numpy/ASR) + ``--json`` + opt-in ``--exit-code``.
"""
from __future__ import annotations

import json

from filmgrip.cli import main
from filmgrip.cli_status import (
    actionable,
    perception_checks,
    render_status,
    status_guidance,
)

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
    assert "select" in status_guidance(READY)[0].lower()  # tells the user to select a clip


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


# --------------------------------------------------------------------------- D12 perception doctor
def test_perception_checks_report_the_toolchain():
    """Each perception dep is reported (don't assume present/absent — assert the CHECK exists)."""
    checks = perception_checks()
    names = {c["name"] for c in checks}
    assert {"ffmpeg", "ffprobe", "numpy", "asr_backend"} <= names
    for c in checks:
        assert c["status"] in ("ok", "warn", "fail")
        # A failing/warning row always carries a fix command; an ok row never claims one is needed.
        if c["status"] == "ok":
            assert c["fix"] == ""
        else:
            assert c["fix"], f"{c['name']} is {c['status']} but has no fix command"


def test_human_output_shows_perception_lines(monkeypatch, capsys, tmp_path):
    _set_preflight(monkeypatch, READY)
    code = main(["status", "--sfx-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Perception toolchain" in out
    # The checks are reported regardless of whether ffmpeg/numpy happen to be installed here.
    for name in ("ffmpeg", "numpy", "asr_backend"):
        assert name in out


def test_status_json_is_valid_and_carries_dep_keys(monkeypatch, capsys, tmp_path):
    _set_preflight(monkeypatch, READY)
    code = main(["status", "--json", "--sfx-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0  # --json alone never changes the exit code
    data = json.loads(out)  # must be valid JSON an agent/CI can parse
    blob = json.dumps(data).lower()
    for dep in ("ffmpeg", "numpy", "asr_backend"):
        assert dep in blob
    assert {"name", "status", "ok"} <= set(data["perception"][0])
    assert "resolve" in data and "editors" in data and "ok" in data


def test_exit_code_flag_gates_on_hard_blocker(monkeypatch, tmp_path):
    """--exit-code returns non-zero iff a perception check is a hard 'fail'; default stays 0."""
    _set_preflight(monkeypatch, READY)

    monkeypatch.setattr(
        "filmgrip.cli_status.perception_checks",
        lambda: [{"name": "ffmpeg", "ok": False, "status": "fail",
                  "detail": "not on PATH", "fix": "brew install ffmpeg"}],
    )
    assert main(["status", "--sfx-dir", str(tmp_path)]) == 0           # default: always 0
    assert main(["status", "--exit-code", "--sfx-dir", str(tmp_path)]) == 1  # opt-in: blocker => 1

    monkeypatch.setattr(
        "filmgrip.cli_status.perception_checks",
        lambda: [{"name": "ffmpeg", "ok": True, "status": "ok", "detail": "/usr/bin/ffmpeg",
                  "fix": ""}],
    )
    assert main(["status", "--exit-code", "--sfx-dir", str(tmp_path)]) == 0  # all green => 0


def test_actionable_what_why_fix_see_format():
    lines = actionable("composition analysis needs OpenCV + numpy",
                       why="neither is importable", fix="pip install 'film-grip[vision]'",
                       see="film-grip status")
    assert lines[0].startswith("error: ")
    assert any(line.strip().startswith("why:") for line in lines)
    assert any("fix:" in line for line in lines)
    assert any("see:" in line for line in lines)
    # Optional fields are omitted, not blank-filled.
    assert actionable("x") == ["error: x"]
