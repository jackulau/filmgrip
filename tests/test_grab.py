"""D1 — `film-grip grab`: selection -> <selected_clips> context block + best-effort clipboard."""
from __future__ import annotations

import argparse

import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.serialize.selection_block import format_selection, frames_to_tc


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def _id(cut, name):
    return next(c for c in cut.real_clips() if c.name == name).id


# --- formatter ---------------------------------------------------------------
def test_frames_to_tc_basic():
    assert frames_to_tc(0, 24) == "00:00:00:00"
    assert frames_to_tc(24, 24) == "00:00:01:00"
    assert frames_to_tc(49, 24) == "00:00:02:01"
    assert frames_to_tc(-5, 24) == "00:00:00:00"  # clamps, never negative


def test_block_has_tags_and_header(cut):
    block = format_selection(cut, [_id(cut, "broll_1")])
    assert block.startswith("<selected_clips>")
    assert block.rstrip().endswith("</selected_clips>")
    assert "fps:" in block
    assert "selected: 1" in block
    assert "[precise]" in block


def test_block_lists_clip_fields(cut):
    cid = _id(cut, "broll_1")
    block = format_selection(cut, [cid])
    assert cid in block
    assert "broll_1" in block
    assert "broll_1.mov" in block        # source ref carried
    assert "track v1" in block           # track code
    assert "dur " in block               # duration field present


def test_block_shows_neighbor_context(cut):
    block = format_selection(cut, [_id(cut, "broll_1")])
    assert "context" in block
    assert "next=broll_2" in block       # broll_2 immediately follows broll_1 on V1


def test_block_can_omit_neighbors(cut):
    block = format_selection(cut, [_id(cut, "broll_1")], neighbors=False)
    assert "context" not in block


def test_block_multi_select_ordered_by_track_then_start(cut):
    ids = [_id(cut, "broll_2"), _id(cut, "intro")]  # deliberately reversed
    block = format_selection(cut, ids)
    assert "selected: 2" in block
    assert block.index("intro") < block.index("broll_2")  # intro (start 0) listed first


def test_block_drops_unknown_ids(cut):
    block = format_selection(cut, ["nope", _id(cut, "intro")])
    assert "selected: 1" in block


def test_block_empty_selection_is_explicit(cut):
    block = format_selection(cut, [])
    assert "selected: 0" in block
    assert "none" in block
    assert block.rstrip().endswith("</selected_clips>")


def test_block_surfaces_confidence_and_basis(cut):
    block = format_selection(cut, [_id(cut, "intro")], confidence="reconstructed", basis="live")
    assert "[reconstructed]" in block
    assert "basis: live" in block


# --- CLI ---------------------------------------------------------------------
def _grab_args(**kw):
    ns = argparse.Namespace(editor="resolve", fixture=None, select=None,
                            no_neighbors=False, no_copy=True)
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns


def test_cmd_grab_fixture_prints_block(fixtures_dir, capsys):
    from filmgrip.cli_grab import cmd_grab

    rc = cmd_grab(_grab_args(fixture=str(fixtures_dir / "cut.otio")))
    out = capsys.readouterr().out
    assert rc == 0
    assert "<selected_clips>" in out
    assert "</selected_clips>" in out


def test_cmd_grab_fixture_select_subset(cut, fixtures_dir, capsys):
    from filmgrip.cli_grab import cmd_grab

    cid = _id(cut, "broll_1")
    rc = cmd_grab(_grab_args(fixture=str(fixtures_dir / "cut.otio"), select=cid))
    out = capsys.readouterr().out
    assert rc == 0
    assert "selected: 1" in out
    assert cid in out


def test_cmd_grab_unknown_id_errors(fixtures_dir, capsys):
    from filmgrip.cli_grab import cmd_grab

    rc = cmd_grab(_grab_args(fixture=str(fixtures_dir / "cut.otio"), select="bogus99"))
    out = capsys.readouterr().out
    assert rc == 1
    assert "unknown clip id" in out


# --- clipboard (best-effort, never fatal) ------------------------------------
def test_clipboard_copy_returns_bool_never_raises():
    from filmgrip import clipboard

    result = clipboard.copy("hello")
    assert isinstance(result, bool)
