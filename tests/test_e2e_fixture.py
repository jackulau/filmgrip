"""D10 — MVP end-to-end CLI, provable offline via --fixture/--plan/--dry-run."""
from __future__ import annotations

import json

from filmgrip.cli import main

FIX = "tests/fixtures/cut.otio"
PLAN = "tests/fixtures/plan.json"


def test_edit_fixture_dry_run_renders_expected_diff(capsys):
    code = main(["edit", "--fixture", FIX, "--plan", PLAN, "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PLAN OK — 3 op(s)" in out
    assert "tighten the open and flag the b-roll" in out
    assert "trim intro out -12  →  0..48 ⇒ 0..36" in out
    assert "marker Blue on dialogue_a @+0 'review'" in out
    assert "set broll_1.ZoomX = 1.2" in out


def test_edit_fixture_rejects_invalid_plan(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"ops": [{"op": "delete", "clip_id": "ghost"}]}))
    code = main(["edit", "--fixture", FIX, "--plan", str(bad), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 1
    assert "PLAN REJECTED" in out
    assert "UNKNOWN_CLIP" in out


def test_edit_fixture_without_plan_explains_itself(capsys):
    code = main(["edit", "--fixture", FIX])  # no --plan, no editor
    out = capsys.readouterr().out
    assert code == 3
    assert "needs --plan" in out


def test_version_exits_zero():
    import pytest

    # --version is handled by argparse (raises SystemExit) and must not import the heavy pipeline.
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
