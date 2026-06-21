"""D6 — musical-beat perception exposed to BOTH the planner (MCP ``get_beats``) and the user
(``film-grip beats`` CLI), mirroring the existing ``get_scopes`` / ``film-grip scopes`` surfaces.

All offline: no real ffmpeg/librosa. The decode path is stubbed by monkeypatching
:func:`filmgrip.perception.music.beats_for_media` (exactly as the scopes tests stub frame access),
and the real offline-fixture honesty wall is exercised directly (its offline branch is ffmpeg-free).
The load-bearing assertion throughout: a retimed/offline/dep-missing clip becomes an ``errors`` entry
with NO fabricated tempo or beat list.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from types import SimpleNamespace

import pytest

from filmgrip.adapters.base import Selection
from filmgrip.core.ir import TimelineIR
from filmgrip.integration import mcp_host as mh
from filmgrip.perception.transcribe import PerceptionUnavailable


# --------------------------------------------------------------------------- shared fakes
def _good_report(clip=None):
    """A successful (numpy-tier) beat read — the shape beats_for_media returns on real audio."""
    return {
        "engine": "numpy", "tier": "advisory", "rate": "24",
        "frames": "timeline" if clip is not None else "media",
        "tempo_bpm": 120.0, "beats": [0, 12, 24, 36], "downbeats": [],
        "onsets": [0, 6, 12], "source": "x.mov", "errors": [],
    }


def _offline_report(clip=None, msg="source media not found ('/media/x.mov') — offline"):
    """The honesty-wall payload: an errors entry, an empty grid, tempo 0.0 — never a fake read."""
    return {
        "engine": "numpy", "tier": "advisory", "rate": "24",
        "frames": "timeline" if clip is not None else "media",
        "tempo_bpm": 0.0, "beats": [], "downbeats": [], "onsets": [], "errors": [msg],
    }


@pytest.fixture
def ctx(fixtures_dir):
    ir = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    sel_ids = [next(c for c in ir.real_clips() if c.name == "broll_1").id]
    return mh.PlannerContext(ir=ir, selection=Selection(ids=sel_ids, basis="test"))


# --------------------------------------------------------------------------- MCP wiring parity
def test_get_beats_tool_is_registered():
    assert "mcp__filmgrip__get_beats" in mh._FG_TOOLS


def test_system_prompt_teaches_get_beats(ctx):
    sp = mh.build_system_prompt(ctx)
    assert "get_beats" in sp


def test_payload_get_beats_assembles_clips_and_errors(ctx, monkeypatch):
    # A real read joins clips (keyed by clip_id); the {clips,errors} shape mirrors get_scopes exactly.
    monkeypatch.setattr("filmgrip.perception.music.beats_for_media",
                        lambda media, clip=None, **kw: _good_report(clip))
    out = mh.payload_get_beats(ctx)
    assert set(out) == {"clips", "errors"}
    assert out["errors"] == []
    assert len(out["clips"]) == 1
    entry = out["clips"][0]
    assert entry["clip_id"] == ctx.selection.ids[0]
    assert entry["tempo_bpm"] == 120.0 and entry["beats"] == [0, 12, 24, 36]


def test_payload_get_beats_offline_clip_is_an_error_not_a_fake_grid(ctx):
    # The cut.otio fixture references media that isn't on disk → real offline path, ffmpeg-free.
    out = mh.payload_get_beats(ctx)
    assert out["clips"] == []                                  # nothing fabricated joined clips
    assert out["errors"] and any("broll_1" in e for e in out["errors"])


def test_payload_get_beats_retimed_clip_passes_error_through(ctx, monkeypatch):
    # A retimed clip comes back from beats_for_media as an errors entry; it must NOT join clips.
    monkeypatch.setattr(
        "filmgrip.perception.music.beats_for_media",
        lambda media, clip=None, **kw: _offline_report(clip, msg="retimed clip — refused"))
    out = mh.payload_get_beats(ctx)
    assert out["clips"] == []
    assert any("retimed" in e for e in out["errors"])


def test_payload_get_beats_missing_dep_surfaces_per_clip(ctx, monkeypatch):
    # ffmpeg/numpy absent → PerceptionUnavailable per clip, surfaced as an error, never swallowed.
    def _boom(media, clip=None, **kw):
        raise PerceptionUnavailable("music analysis needs numpy")

    monkeypatch.setattr("filmgrip.perception.music.beats_for_media", _boom)
    out = mh.payload_get_beats(ctx)
    assert out["clips"] == []
    assert any("needs numpy" in e for e in out["errors"])


# --------------------------------------------------------------------------- CLI parity
def _run_beats(args) -> tuple[int, str]:
    from filmgrip.cli_beats import cmd_beats

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = cmd_beats(args)
    return code, buf.getvalue()


def test_beats_subcommand_is_wired_into_the_parser():
    import filmgrip.cli as c

    choices = c.build_parser()._subparsers._group_actions[0].choices
    assert "beats" in choices


def test_cli_beats_media_mode_formats_a_result(monkeypatch):
    monkeypatch.setattr("filmgrip.perception.music.beats_for_media",
                        lambda media, clip=None, **kw: _good_report(clip))
    code, out = _run_beats(SimpleNamespace(media="x.mov", fixture=None, select=None))
    assert code == 0
    report = json.loads(out)
    assert report["tempo_bpm"] == 120.0
    assert report["beats"] == [0, 12, 24, 36]


def test_cli_beats_media_offline_is_nonzero_with_errors(monkeypatch):
    # An offline/undecodable media file: an errors entry, no fake grid, non-zero exit.
    monkeypatch.setattr("filmgrip.perception.music.beats_for_media",
                        lambda media, clip=None, **kw: _offline_report(clip))
    code, out = _run_beats(SimpleNamespace(media="x.mov", fixture=None, select=None))
    assert code == 2
    report = json.loads(out)
    assert report["beats"] == [] and report["tempo_bpm"] == 0.0
    assert report["errors"]


def test_cli_beats_fixture_mode_per_clip(monkeypatch, fixtures_dir):
    monkeypatch.setattr("filmgrip.perception.music.beats_for_media",
                        lambda media, clip=None, **kw: _good_report(clip))
    code, out = _run_beats(SimpleNamespace(
        media=None, fixture=str(fixtures_dir / "sample.fcpxml"), select=None))
    assert code == 0
    rows = json.loads(out)
    assert len(rows) == 3
    assert all("clip_id" in r and "clip_name" in r for r in rows)
    assert all(r["tempo_bpm"] == 120.0 for r in rows)


def test_cli_beats_fixture_offline_clip_yields_error_not_fake(monkeypatch, fixtures_dir):
    # Honesty wall through the CLI fixture path: an offline clip is reported with errors, partial exit 3.
    monkeypatch.setattr("filmgrip.perception.music.beats_for_media",
                        lambda media, clip=None, **kw: _offline_report(clip))
    code, out = _run_beats(SimpleNamespace(
        media=None, fixture=str(fixtures_dir / "sample.fcpxml"), select=None))
    assert code == 3
    rows = json.loads(out)
    assert all(r["beats"] == [] and r["errors"] for r in rows)
