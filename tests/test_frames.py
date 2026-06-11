"""D4 — visual perception: contact sheets from real (tiny, ffmpeg-generated) media.

Real ffmpeg runs here: a 2s 64x48 test pattern with a 440Hz tone, and one video-only twin.
Skips cleanly when ffmpeg isn't installed. Pure-math helpers are tested without media.
"""
from __future__ import annotations

import shutil
import subprocess

import opentimelineio as otio
import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.perception.frames import (
    SheetResult,
    clip_sheet,
    compose_sheet,
    media_time_at,
    timeline_sheet,
)
from filmgrip.perception.transcribe import PerceptionUnavailable

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

PNG_MAGIC = b"\x89PNG"


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """2s of testsrc + sine tone, 24fps, 64x48 — a few KB, generated once per module."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)
    return path


@pytest.fixture(scope="module")
def media_mute(tmp_path_factory):
    path = tmp_path_factory.mktemp("media") / "mute.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=2:size=64x48:rate=24",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)],
        check=True, capture_output=True)
    return path


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


def _ir_with(media_path, *, src_in=0, dur=48, retimed=False) -> TimelineIR:
    clip = otio.schema.Clip(
        name="take1",
        media_reference=otio.schema.ExternalReference(target_url=media_path.as_uri()),
        source_range=otio.opentime.TimeRange(_rt(src_in), _rt(dur)),
    )
    if retimed:
        clip.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(clip)
    return TimelineIR(tl)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMGRIP_CACHE_DIR", str(tmp_path / "cache"))


# --------------------------------------------------------------------------- pure math
def test_media_time_at_maps_through_source_offset():
    ir = _ir_with_pathless(src_in=24, start_gap=0)
    clip = ir.real_clips()[0]
    # timeline frame 10 → source frame 24+10 → 34/24 s
    assert media_time_at(clip, 10, ir.rate) == pytest.approx(34 / 24)


def _ir_with_pathless(*, src_in=0, start_gap=0) -> TimelineIR:
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    if start_gap:
        track.append(otio.schema.Gap(
            source_range=otio.opentime.TimeRange(_rt(0), _rt(start_gap))))
    track.append(otio.schema.Clip(
        name="x", media_reference=otio.schema.ExternalReference(target_url="/nope.mov"),
        source_range=otio.opentime.TimeRange(_rt(src_in), _rt(48))))
    return TimelineIR(tl)


# --------------------------------------------------------------------------- sheets
def test_clip_sheet_renders_png_with_legend(media):
    ir = _ir_with(media)
    cid = ir.real_clips()[0].id
    sheet = clip_sheet(ir, cid, count=4)
    with open(sheet.png_path, "rb") as fh:
        assert fh.read(4) == PNG_MAGIC
    assert len(sheet.legend) == 4
    clip = ir.real_clips()[0]
    for entry in sheet.legend:
        assert clip.start <= entry["timeline_frame"] < clip.end
        assert 0.0 <= entry["media_s"] <= 2.0
    assert sheet.notes == []                      # tone present → waveform included


def test_clip_sheet_video_only_media_omits_waveform_honestly(media_mute):
    ir = _ir_with(media_mute)
    sheet = clip_sheet(ir, ir.real_clips()[0].id, count=2)
    with open(sheet.png_path, "rb") as fh:
        assert fh.read(4) == PNG_MAGIC
    assert any("waveform omitted" in n for n in sheet.notes)


def test_timeline_sheet_exact_frames_and_gap_errors(media):
    ir = _ir_with(media)                          # clip occupies [0,48)
    sheets, errors = timeline_sheet(ir, [10, 30, 100])
    assert len(sheets) == 1
    assert [e["timeline_frame"] for e in sheets[0].legend] == [10, 30]
    assert any("frame 100" in e for e in errors)


def test_retimed_clip_refused(media):
    ir = _ir_with(media, retimed=True)
    with pytest.raises(PerceptionUnavailable, match="retimed"):
        clip_sheet(ir, ir.real_clips()[0].id)
    sheets, errors = timeline_sheet(ir, [10])
    assert sheets == [] and any("retimed" in e for e in errors)


def test_offline_media_is_honest(tmp_path):
    ir = _ir_with_pathless()
    with pytest.raises(PerceptionUnavailable, match="not on disk"):
        clip_sheet(ir, ir.real_clips()[0].id)


def test_compose_sheet_single_tile(media, tmp_path):
    out = tmp_path / "one.png"
    compose_sheet(str(media), [0.5], str(out))
    assert out.read_bytes()[:4] == PNG_MAGIC


def test_no_ffmpeg_is_an_actionable_error(media, monkeypatch):
    import filmgrip.perception.transcribe as tr

    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    ir = _ir_with(media)
    with pytest.raises(PerceptionUnavailable, match="brew install ffmpeg"):
        clip_sheet(ir, ir.real_clips()[0].id)


# --------------------------------------------------------------------------- MCP payload
def test_payload_view_frames(media):
    from filmgrip.adapters.base import Selection
    from filmgrip.integration import mcp_host as mh

    ir = _ir_with(media)
    cid = ir.real_clips()[0].id
    ctx = mh.PlannerContext(ir=ir, selection=Selection(ids=[cid], basis="fixture"))
    payload = mh.payload_view_frames(ctx, count=3)
    assert payload["errors"] == []
    assert len(payload["sheets"]) == 1
    assert len(payload["sheets"][0]["legend"]) == 3

    payload = mh.payload_view_frames(ctx, frames=[10, 999])
    assert len(payload["sheets"]) == 1
    assert payload["errors"]


# --------------------------------------------------------------------------- CLI
def test_cli_frames_fixture(media, tmp_path, capsys):
    ir = _ir_with(media)
    fixture = tmp_path / "seq.otio"
    otio.adapters.write_to_file(ir.timeline, str(fixture))
    out_png = tmp_path / "sheet.png"
    from filmgrip.cli import main

    rc = main(["frames", "--fixture", str(fixture), "--count", "3", "--out", str(out_png)])
    out = capsys.readouterr().out
    assert rc == 0
    assert out_png.read_bytes()[:4] == PNG_MAGIC
    assert "tile 0" in out and "frame" in out
