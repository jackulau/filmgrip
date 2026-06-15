"""Color deliverable 6 — color perception. analyze_rgb is pure numpy, so its correctness is proved
on synthetic frames with known statistics (no editor, no ffmpeg). The ffmpeg extraction path is
exercised when ffmpeg is present, else skipped."""
from __future__ import annotations

import json
import subprocess

import numpy as np
import pytest

from filmgrip.perception import scopes
from filmgrip.perception.scopes import analyze_rgb, report_json


def _solid(r, g, b, h=16, w=16):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = r, g, b
    return a


# ----------------------------------------------------------------------- luma / exposure flags
def test_black_frame_is_crushed_and_under():
    r = analyze_rgb(np.zeros((8, 8, 3), dtype=np.uint8))
    assert r["luma"]["crushed"] is True
    assert r["luma"]["clipped"] is False
    assert r["exposure"]["verdict"] == "under"
    assert r["luma"]["median"] == 0.0


def test_white_frame_is_clipped_and_over():
    r = analyze_rgb(_solid(255, 255, 255))
    assert r["luma"]["clipped"] is True
    assert r["luma"]["crushed"] is False
    assert r["exposure"]["verdict"] == "over"


def test_mid_grey_is_neutral_and_ok():
    r = analyze_rgb(_solid(128, 128, 128))
    assert r["white_balance"]["cast"] == "neutral"
    assert r["exposure"]["verdict"] == "ok"
    assert r["parade"]["g"]["p50"] == 128.0
    # greyscale → essentially zero chroma on the vectorscope
    assert r["vectorscope"]["saturation"] < 0.01


# ----------------------------------------------------------------------- white-balance cast
def test_warm_cast_detected():
    assert analyze_rgb(_solid(200, 100, 50))["white_balance"]["cast"] == "warm"


def test_cool_cast_detected():
    assert analyze_rgb(_solid(40, 80, 200))["white_balance"]["cast"] == "cool"


def test_green_cast_detected():
    assert analyze_rgb(_solid(60, 200, 70))["white_balance"]["cast"] == "green"


# ----------------------------------------------------------------------- parade percentiles
def test_parade_percentiles_on_gradient():
    # a horizontal 0..255 ramp → median near 128, p1 near 0, p99 near 255 on every channel.
    ramp = np.tile(np.linspace(0, 255, 256, dtype=np.uint8), (4, 1))
    arr = np.dstack([ramp, ramp, ramp])
    r = analyze_rgb(arr)
    assert 120 <= r["parade"]["r"]["p50"] <= 136
    assert r["parade"]["r"]["p1"] <= 10 and r["parade"]["r"]["p99"] >= 245


# ----------------------------------------------------------------------- vectorscope
def test_vectorscope_saturation_rises_with_chroma():
    grey = analyze_rgb(_solid(128, 128, 128))["vectorscope"]["saturation"]
    saturated = analyze_rgb(_solid(220, 30, 30))["vectorscope"]["saturation"]
    assert saturated > grey


def test_skin_tone_delta_is_a_number_in_range():
    d = analyze_rgb(_solid(210, 150, 120))["vectorscope"]["skin_tone_delta_deg"]
    assert 0.0 <= d <= 180.0


# ----------------------------------------------------------------------- robustness / shape
def test_wrong_shape_raises():
    with pytest.raises(ValueError):
        analyze_rgb(np.zeros((8, 8), dtype=np.uint8))


def test_report_is_json_serializable():
    r = analyze_rgb(_solid(100, 120, 140))
    assert json.loads(report_json(r))["white_balance"]["cast"] in {"neutral", "warm", "cool", "green", "magenta"}


# ----------------------------------------------------------------------- ffmpeg extraction (guarded)
def _ffmpeg():
    from filmgrip.perception.transcribe import ffmpeg_path
    return ffmpeg_path()


@pytest.mark.skipif(not _ffmpeg(), reason="ffmpeg not installed")
def test_frame_rgb_extracts_known_color(tmp_path):
    # render a solid mid-grey PNG via ffmpeg's lavfi color source, then read it back.
    png = tmp_path / "grey.png"
    subprocess.run([_ffmpeg(), "-y", "-v", "error", "-f", "lavfi",
                    "-i", "color=c=gray:s=160x90", "-frames:v", "1", str(png)],
                   check=True, capture_output=True)
    arr = scopes.frame_rgb(str(png))
    assert arr.shape == (90, 160, 3)
    rep = analyze_rgb(arr)
    # ffmpeg "gray" ≈ 128 → neutral, ok exposure
    assert rep["white_balance"]["cast"] == "neutral"
    assert rep["exposure"]["verdict"] == "ok"
