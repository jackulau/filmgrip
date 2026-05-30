"""D10 — in-Resolve panel: testable PanelSpec/PanelController seam + `film-grip panel install`."""
from __future__ import annotations

import py_compile
from pathlib import Path

from filmgrip.adapters import resolve_client as rc
from filmgrip.cli import main
from filmgrip.cli_panel import default_scripts_dir, panel_source_path
from filmgrip.ui.panel import (
    ID_APPLY,
    ID_DRYRUN,
    ID_OUTPUT,
    ID_PROMPT,
    ID_SELECTION,
    PanelController,
    PanelResult,
    build_panel_spec,
    live_controller,
)
from tests.fakes import make_two_track_resolve

ROOT = Path(__file__).resolve().parent.parent


# -- spec structure ---------------------------------------------------------
def test_panel_spec_has_prompt_buttons_and_output():
    spec = build_panel_spec("3 clip(s) selected [reconstructed]")
    ids = spec.ids()
    assert {ID_SELECTION, ID_PROMPT, ID_APPLY, ID_DRYRUN, ID_OUTPUT} <= ids
    assert spec.find(ID_SELECTION).text == "3 clip(s) selected [reconstructed]"
    # Apply + Dry-run are buttons; prompt is an editable field
    assert spec.find(ID_APPLY).kind == "Button" and spec.find(ID_DRYRUN).kind == "Button"
    assert spec.find(ID_PROMPT).kind == "TextEdit"


# -- controller behaviour (the Apply/Dry-run seam) --------------------------
def test_apply_invokes_run_edit_with_dry_run_false():
    calls = []
    ctrl = PanelController(lambda prompt, dry: calls.append((prompt, dry)) or PanelResult(True, "ok"))
    res = ctrl.on_apply("add a marker")
    assert res.ok and res.text == "ok"
    assert calls == [("add a marker", False)]


def test_dry_run_invokes_run_edit_with_dry_run_true():
    calls = []
    ctrl = PanelController(lambda prompt, dry: calls.append((prompt, dry)) or PanelResult(True, "diff"))
    ctrl.on_dry_run("trim 12")
    assert calls == [("trim 12", True)]


def test_empty_prompt_does_not_call_run_edit():
    called = []
    ctrl = PanelController(lambda p, d: called.append(1) or PanelResult(True, ""))
    res = ctrl.on_apply("   ")
    assert not called and res.ok is False and "instruction" in res.text


def test_run_edit_exception_becomes_panel_error_not_crash():
    def boom(prompt, dry):
        raise RuntimeError("planner exploded")

    res = PanelController(boom).on_apply("do it")
    assert res.ok is False and "planner exploded" in res.text


def test_live_controller_summarizes_selection_without_network():
    # Build the live controller against a fake Resolve; it must summarize the selection (and NOT
    # call the planner — that only happens on Apply).
    session = rc.connect(make_two_track_resolve())
    ctrl = live_controller(session)
    summary = ctrl.selection_summary()
    assert "clip(s) selected" in summary and "reconstructed" in summary
    assert {ID_PROMPT, ID_APPLY} <= ctrl.spec().ids()


# -- install command --------------------------------------------------------
def test_panel_script_byte_compiles():
    py_compile.compile(str(ROOT / "scripts" / "film_grip_resolve_panel.py"), doraise=True)


def test_panel_source_path_points_at_the_script():
    assert panel_source_path().name == "film_grip_resolve_panel.py"


def test_panel_install_dry_run_prints_target(capsys):
    code = main(["panel", "install", "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "would install" in out and "Workspace" in out


def test_panel_install_writes_stamped_copy(tmp_path, capsys):
    code = main(["panel", "install", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    installed = tmp_path / "film-grip.py"
    assert installed.is_file()
    text = installed.read_text()
    # the filmgrip site path was baked in so the installed copy can import the package
    assert 'FILMGRIP_SITE = ""' not in text and "FILMGRIP_SITE = " in text
    py_compile.compile(str(installed), doraise=True)


def test_default_scripts_dir_is_resolve_edit_folder():
    assert "Scripts" in str(default_scripts_dir()) and "Edit" in str(default_scripts_dir())
