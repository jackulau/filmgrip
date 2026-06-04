"""D6 — edit-pack engine + deterministic built-in recipes (compile → validate → apply)."""
from __future__ import annotations

import pytest

from filmgrip.cli import main
from filmgrip.core.ir import TimelineIR
from filmgrip.packs import Pack, PackError, all_packs, get_pack
from filmgrip.packs.engine import compile_pack
from filmgrip.protocol.validate import validate

FIX = "tests/fixtures/cut.otio"


@pytest.fixture
def cut():
    return TimelineIR.from_otio_file(FIX)


def _all_ids(ir):
    return [c.id for c in ir.real_clips()]


# --- registry ----------------------------------------------------------------
def test_builtins_registered():
    names = {p.name for p in all_packs()}
    assert "marker-pass" in names
    # punch-up / dissolves were removed — they emitted add_transition, which no adapter can apply,
    # so they reported success while changing nothing. A deterministic pack must only emit applicable
    # ops (see test_packs_applicable.py).
    assert "punch-up" not in names and "dissolves" not in names


def test_get_unknown_pack_raises():
    with pytest.raises(PackError):
        get_pack("does-not-exist")


# --- deterministic recipes compile to VALID plans ----------------------------
def test_marker_pass_marks_every_selected_clip(cut):
    ids = _all_ids(cut)
    plan = compile_pack(get_pack("marker-pass"), cut, ids)
    assert len(plan.ops) == len(ids)
    assert all(op.op == "add_marker" for op in plan.ops)
    assert validate(plan, cut).ok


def test_marker_pass_respects_param_override(cut):
    plan = compile_pack(get_pack("marker-pass"), cut, _all_ids(cut), {"color": "Red"})
    assert all(op.color == "Red" for op in plan.ops)
    assert validate(plan, cut).ok


def test_compile_prompt_pack_is_rejected(cut):
    p = Pack("p", "a prompt pack", kind="prompt", prompt="do the thing")
    with pytest.raises(PackError):
        compile_pack(p, cut, _all_ids(cut))


def test_empty_selection_yields_empty_plan(cut):
    plan = compile_pack(get_pack("marker-pass"), cut, [])
    assert plan.ops == []
    assert validate(plan, cut).ok


# --- CLI ---------------------------------------------------------------------
def test_cli_pack_list(capsys):
    code = main(["pack", "list"])
    out = capsys.readouterr().out
    assert code == 0
    assert "marker-pass" in out


def test_cli_pack_show(capsys):
    code = main(["pack", "show", "marker-pass"])
    out = capsys.readouterr().out
    assert code == 0
    assert "marker-pass" in out and "marker" in out


def test_cli_pack_apply_marker_pass_dry_run(capsys):
    code = main(["pack", "apply", "marker-pass", "--fixture", FIX, "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PLAN OK" in out and "marker" in out


def test_cli_pack_apply_unknown_is_clean(capsys):
    code = main(["pack", "apply", "nope", "--fixture", FIX])
    out = capsys.readouterr().out
    assert code == 1
    assert "unknown pack" in out


def test_cli_pack_apply_writes_edited_file(tmp_path, capsys):
    import shutil

    src = tmp_path / "cut.otio"
    shutil.copy(FIX, src)
    code = main(["pack", "apply", "marker-pass", "--fixture", str(src)])
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "cut.edited.otio").is_file()
