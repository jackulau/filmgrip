"""D14 — batched contact-sheet extraction: one ffmpeg decode emits N tiles (was N spawns).

Real ffmpeg runs here (lavfi test pattern), so these skip cleanly when ffmpeg is absent. They lock
the three properties D14 must hold: (1) the sheet has the SAME tile count the contract promises and
is a valid PNG; (2) the cache key tracks source content (mtime/size) so a changed source regenerates
the sheet instead of serving a stale one; (3) a real ffmpeg failure (bogus media) raises
``PerceptionUnavailable`` — never a silent empty/short sheet.
"""
from __future__ import annotations

import os
import shutil
import subprocess

import opentimelineio as otio
import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.perception import frames as F
from filmgrip.perception.frames import (
    _extract_frames_batch,
    _source_sig,
    clip_sheet,
    compose_sheet,
)
from filmgrip.perception.transcribe import PerceptionUnavailable

FFMPEG = shutil.which("ffmpeg")
pytestmark = pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")

PNG_MAGIC = b"\x89PNG"


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    """3s testsrc + 440Hz tone, 24fps, 64x48 — enough frames for an 8-tile sheet."""
    path = tmp_path_factory.mktemp("media") / "clip.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=3:size=64x48:rate=24",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=3",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-shortest", str(path)],
        check=True, capture_output=True)
    return path


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


def _ir_with(media_path, *, src_in=0, dur=48) -> TimelineIR:
    clip = otio.schema.Clip(
        name="take1",
        media_reference=otio.schema.ExternalReference(target_url=media_path.as_uri()),
        source_range=otio.opentime.TimeRange(_rt(src_in), _rt(dur)),
    )
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(clip)
    return TimelineIR(tl)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMGRIP_CACHE_DIR", str(tmp_path / "cache"))


# --------------------------------------------------------------------------- batching
def test_batch_emits_exactly_one_tile_per_timestamp(media, tmp_path):
    """The single-decode helper returns the requested number of tiles, in request order."""
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    times = [0.1, 0.7, 1.3, 1.9, 2.5]
    tiles = _extract_frames_batch(str(media), times, str(out_dir))
    assert len(tiles) == len(times)
    for tile in tiles:
        assert os.path.isfile(tile)
        with open(tile, "rb") as fh:
            assert fh.read(4) == PNG_MAGIC


def test_clip_sheet_tile_count_matches_contract_and_png(media):
    """End-to-end: the contact sheet has the SAME tile count requested + valid PNG magic bytes."""
    ir = _ir_with(media)
    cid = ir.real_clips()[0].id
    for count in (1, 4, 8):
        sheet = clip_sheet(ir, cid, count=count, waveform=False)
        assert len(sheet.legend) == count                  # contract: tile count == count
        with open(sheet.png_path, "rb") as fh:
            assert fh.read(4) == PNG_MAGIC


def test_single_invocation_for_multi_tile_sheet(media, monkeypatch):
    """A multi-tile sheet must trigger ONE decode call, not one per tile (the D14 regression)."""
    real_run = F._run
    decode_calls = {"n": 0}

    def counting_run(cmd, **kw):
        # Count ffmpeg calls that decode the source via an explicit ``select=`` filter (the batch
        # extraction), distinct from the hstack/scale-probe/waveform composition calls.
        if any(isinstance(a, str) and a.startswith("select='eq(n") for a in cmd):
            decode_calls["n"] += 1
        return real_run(cmd, **kw)

    monkeypatch.setattr(F, "_run", counting_run)
    sheet = clip_sheet(_ir_with(media), _ir_with(media).real_clips()[0].id,
                       count=8, waveform=False)
    assert len(sheet.legend) == 8
    assert decode_calls["n"] == 1, "expected a single batched decode, not one ffmpeg per tile"


# --------------------------------------------------------------------------- cache staleness
def test_source_sig_changes_when_file_changes(media, tmp_path):
    """The cache signature must change when the source bytes/mtime change."""
    copy = tmp_path / "copy.mp4"
    shutil.copyfile(media, copy)
    sig1 = _source_sig(str(copy))
    # Re-render the same logical clip with different content → different size/mtime.
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc2=duration=3:size=64x48:rate=24",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(copy)],
        check=True, capture_output=True)
    sig2 = _source_sig(str(copy))
    assert sig1 != sig2


def test_changed_source_regenerates_sheet_not_stale(media, tmp_path):
    """Editing the source invalidates the cached sheet: a fresh, different PNG is produced."""
    src = tmp_path / "src.mp4"
    shutil.copyfile(media, src)
    ir = _ir_with(src)
    cid = ir.real_clips()[0].id

    sheet1 = clip_sheet(ir, cid, count=4, waveform=False)
    path1 = sheet1.png_path
    bytes1 = open(path1, "rb").read()
    assert os.path.isfile(path1)

    # Mutate the source content (and therefore mtime/size). Same logical clip & params.
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "mandelbrot=size=64x48:rate=24",
         "-t", "3", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(src)],
        check=True, capture_output=True)
    ir2 = _ir_with(src)
    sheet2 = clip_sheet(ir2, ir2.real_clips()[0].id, count=4, waveform=False)

    # Cache key folds in the source signature → a NEW cache path (no stale-sheet reuse).
    assert sheet2.png_path != path1, "changed source must not reuse the old cache key"
    assert open(sheet2.png_path, "rb").read(4) == PNG_MAGIC
    # And the picture actually differs (testsrc → mandelbrot).
    assert open(sheet2.png_path, "rb").read() != bytes1


# --------------------------------------------------------------------------- honest failure
def test_bogus_media_raises_not_silent_empty_sheet(tmp_path):
    """A bogus media path is an honest PerceptionUnavailable, never a silent empty/short sheet."""
    out = tmp_path / "sheet.png"
    with pytest.raises(PerceptionUnavailable):
        compose_sheet(str(tmp_path / "does_not_exist.mp4"), [0.1, 0.5, 0.9],
                      str(out), waveform=False)
    assert not out.exists()                                # nothing pretending to be evidence


def test_truncated_source_short_frames_is_honest(media, tmp_path):
    """Asking for a media-second past the source's end yields an honest error, not a short sheet."""
    out_dir = tmp_path / "tiles"
    out_dir.mkdir()
    # Source is 3s @ 24fps (~72 frames). 99.0s maps to a frame index far past the end → fewer
    # frames come back than requested, which must raise rather than silently drop a tile.
    with pytest.raises(PerceptionUnavailable, match="of 2 requested|could not batch-extract"):
        _extract_frames_batch(str(media), [0.5, 99.0], str(out_dir))
