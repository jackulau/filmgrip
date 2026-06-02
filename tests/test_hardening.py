"""D8 — bounded integration hardening: friendly errors + complete, honest surfaces."""
from __future__ import annotations

from filmgrip.cli import main

FIX = "tests/fixtures/cut.otio"


# --- friendly fixture errors (no raw tracebacks across edit / grab / pack) ----
def test_edit_missing_fixture_is_friendly(capsys):
    code = main(["edit", "--fixture", "/nope/missing.otio", "--plan", "x"])
    out = capsys.readouterr().out
    assert code == 1 and "fixture not found" in out


def test_grab_missing_fixture_is_friendly(capsys):
    code = main(["grab", "--fixture", "/nope/missing.otio", "--no-copy"])
    out = capsys.readouterr().out
    assert code == 1 and "fixture not found" in out


def test_pack_apply_missing_fixture_is_friendly(capsys):
    code = main(["pack", "apply", "marker-pass", "--fixture", "/nope/missing.otio", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 1 and "fixture not found" in out


def test_unreadable_fixture_is_friendly(tmp_path, capsys):
    bad = tmp_path / "bad.otio"
    bad.write_text("this is not otio json")
    code = main(["grab", "--fixture", str(bad), "--no-copy"])
    out = capsys.readouterr().out
    assert code == 1 and "could not read" in out


# --- complete, honest surfaces ----------------------------------------------
def test_editors_lists_cross_editor_features(capsys):
    code = main(["editors"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Resolve" in out                     # the per-editor matrix still renders
    assert "pack" in out and "grab" in out and "backend" in out  # cross-editor features surfaced


def test_status_always_succeeds_and_reports_auth(capsys):
    code = main(["status"])
    out = capsys.readouterr().out
    assert code == 0
    assert "film-grip — status" in out and "LLM auth" in out
