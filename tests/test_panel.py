"""D10 — in-Resolve panel: testable PanelSpec/PanelController seam + `film-grip panel install`."""
from __future__ import annotations

import py_compile
from pathlib import Path

from filmgrip.adapters import resolve_client as rc
from filmgrip.cli import main
from filmgrip.cli_panel import default_scripts_dir, panel_source_path
from filmgrip.ui.panel import (
    ID_APPLY,
    ID_COPY,
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


def test_live_controller_blocks_apply_when_nothing_selected():
    from filmgrip.adapters.base import Selection as _Sel  # noqa: F401  (kept for parity)
    from tests.fakes import FakeMediaPool, FakeProject, FakeResolve, FakeTimeline, FakeTimelineItem

    tl = FakeTimeline("T")
    tl.add_track("video", 1, [FakeTimelineItem("solo", 0, 48)])
    tl._current_video_item = None  # nothing selected on the timeline
    proj = FakeProject("P", timeline=tl, media_pool=FakeMediaPool())  # empty media selection
    session = rc.connect(FakeResolve(project=proj))
    ctrl = live_controller(session)
    # No selection -> Apply returns the actionable guard WITHOUT touching the planner/network.
    res = ctrl.on_apply("add a marker")
    assert res.ok is False and "Select a clip" in res.text


def test_live_controller_capture_preview_lists_clips_without_network():
    # Build the live controller against a fake Resolve; the summary is now a capture-preview that
    # lists each armed clip (name · track), not a bare count — and it must NOT call the planner.
    session = rc.connect(make_two_track_resolve())
    ctrl = live_controller(session)
    summary = ctrl.selection_summary()
    assert "armed" in summary and "reconstructed" in summary   # honest capture-preview
    assert "midshot" in summary and "v1" in summary            # the selected clip, with its track
    assert {ID_PROMPT, ID_APPLY, ID_COPY} <= ctrl.spec().ids()


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


# -- D2: capture-preview HUD + Copy-context ---------------------------------
def test_panel_spec_has_copy_context_button():
    spec = build_panel_spec()
    btn = spec.find(ID_COPY)
    assert btn is not None and btn.kind == "Button" and btn.text == "Copy context"


def test_armed_preview_is_a_per_clip_capture_list(fixtures_dir):
    from filmgrip.core.ir import TimelineIR
    from filmgrip.serialize.selection_block import armed_preview

    ir = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    ids = [c.id for c in ir.real_clips()[:2]]
    preview = armed_preview(ir, ids, confidence="reconstructed")
    assert preview.startswith("2 clip(s) armed [reconstructed]:")
    assert "•" in preview          # bulleted per-clip lines (the capture boundary)
    assert "f" in preview          # frame ranges shown


def test_armed_preview_empty_is_explicit(fixtures_dir):
    from filmgrip.core.ir import TimelineIR
    from filmgrip.serialize.selection_block import armed_preview

    ir = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    assert "no clips armed" in armed_preview(ir, [])


def test_on_copy_grabs_context_and_copies_to_clipboard():
    copied = []
    block = "<selected_clips>\n## clips:\n- [c1] x\n</selected_clips>"
    ctrl = PanelController(
        lambda p, d: PanelResult(True, ""),
        grab_context=lambda: block,
        copy_fn=lambda text: copied.append(text) or True,
    )
    res = ctrl.on_copy()
    assert res.ok and "Copied" in res.text
    assert copied == [block]


def test_on_copy_without_clipboard_tool_shows_block():
    block = "<selected_clips>...</selected_clips>"
    ctrl = PanelController(lambda p, d: PanelResult(True, ""),
                           grab_context=lambda: block, copy_fn=lambda text: False)
    res = ctrl.on_copy()
    assert res.ok and res.text == block   # degrades to showing the block for manual copy


def test_on_copy_without_grab_context_is_graceful():
    res = PanelController(lambda p, d: PanelResult(True, "")).on_copy()
    assert res.ok is False and "unavailable" in res.text


def test_on_copy_grab_exception_does_not_crash():
    def boom():
        raise RuntimeError("snapshot failed")

    res = PanelController(lambda p, d: PanelResult(True, ""), grab_context=boom).on_copy()
    assert res.ok is False and "snapshot failed" in res.text
