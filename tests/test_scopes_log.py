"""Color deliverable 9 — log/ACES color-depth awareness + the corrected skin-tone line.

Pure-numpy (no ffmpeg, no editor), mirroring tests/test_scopes.py conventions:
synthetic frames with known statistics, per-PATCH vectorscope assertions (the research warns that a
whole-frame vectorscope mean averages opposite hues to a meaningless angle — color-science.md §1.3).
Everything here is validated against docs/research/color-science.md.
"""
from __future__ import annotations

import inspect

import numpy as np

from filmgrip.perception import scopes
from filmgrip.perception.scopes import analyze_rgb, detect_log_footage


def _solid(r, g, b, h=32, w=32):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = r, g, b
    return a


def _angle_diff(a, b):
    d = abs((a - b) % 360.0)
    return min(d, 360.0 - d)


# ----------------------------------------------------------------------- color_space kwarg present
def test_analyze_rgb_accepts_color_space_kwarg():
    sig = inspect.signature(analyze_rgb)
    assert "color_space" in sig.parameters
    # default keeps every existing caller byte-identical (rec709, no decode).
    assert sig.parameters["color_space"].default == "rec709"
    rep = analyze_rgb(_solid(128, 128, 128), color_space="rec709")
    assert rep["color_space"] == "rec709"
    assert rep["decoded"] is False


def test_default_call_is_unchanged_and_not_decoded():
    # No kwarg => identical report shape to the legacy path, decoded flag False.
    rep = analyze_rgb(_solid(128, 128, 128))
    assert rep["decoded"] is False
    assert rep["color_space"] == "rec709"
    # legacy keys still present
    assert rep["exposure"]["verdict"] == "ok"
    assert rep["white_balance"]["cast"] == "neutral"


def test_unknown_space_falls_through_to_display_path():
    rep = analyze_rgb(_solid(100, 120, 140), color_space="totally_made_up")
    assert rep["decoded"] is False  # unknown => treated as already-709, no transform


# ----------------------------------------------------------------------- per-patch vectorscope angles
def test_red75_patch_vectorscope_angle():
    # 75% red bar plots at ~102.9° in scopes.py's atan2(Cr,Cb) convention (color-science.md §1.3).
    rep = analyze_rgb(_solid(191, 0, 0))
    assert _angle_diff(rep["vectorscope"]["hue_deg"], 102.91) <= 2.0


def test_skin_patch_lands_near_corrected_iline():
    # Real skin sample (210,150,120) measures ~124.9° and the corrected I-line reference is 123.0,
    # so the skin-tone delta must be small (~2°), NOT the ~10° the old 115° reference produced.
    rep = analyze_rgb(_solid(210, 150, 120))
    assert _angle_diff(rep["vectorscope"]["hue_deg"], 123.0) <= 5.0
    assert rep["vectorscope"]["skin_tone_delta_deg"] <= 5.0


def test_skin_tone_reference_angle_is_corrected():
    # The confirmed-bug fix: 115.0 -> 123.0 (color-science.md §1.4).
    assert scopes.SKIN_TONE_ANGLE_DEG == 123.0


# ----------------------------------------------------------------------- detect_log_footage advisory
def test_detect_log_trips_on_lifted_lowcontrast_frame():
    # Lifted-black, low-contrast, desaturated, bunched-mid frame ~ a flat/log look: >=3 signals fire.
    rng = np.random.default_rng(0)
    log_like = (rng.uniform(28, 150, (64, 64, 1)) * np.ones((1, 1, 3))).astype(np.uint8)
    out = detect_log_footage(log_like)
    assert out["likely_log"] is True
    assert out["score"] >= 3
    assert out["signals"]["low_contrast"] and out["signals"]["lifted_blacks"]
    assert "ADVISORY" in out["advice"]


def test_detect_log_quiet_on_normal_graded_frame():
    # A well-graded 709 frame: shadows near 0, highlights near 255, saturated, good contrast.
    rng = np.random.default_rng(1)
    base = rng.uniform(5, 250, (64, 64, 1)) * np.ones((1, 1, 3))
    base[:, :32, 0] *= 1.3   # warm half
    base[:, 32:, 2] *= 1.3   # cool half  -> real chroma
    normal = np.clip(base, 0, 255).astype(np.uint8)
    out = detect_log_footage(normal)
    assert out["likely_log"] is False
    assert out["score"] < 3


def test_detect_log_is_advisory_never_transforms():
    # Contract: it returns advice only — never a transformed array, never a hard claim.
    out = detect_log_footage(_solid(64, 64, 64))
    assert set(out) == {"likely_log", "score", "signals", "advice"}
    assert isinstance(out["likely_log"], bool)


# ----------------------------------------------------------------------- log decode shifts exposure
def test_slog3_decode_shifts_exposure_verdict():
    # A uniform S-Log3 highlight at code 0.60 (153/255) reads "ok" as raw 709 (mid), but decoding to
    # display reveals it as a blown highlight -> "over". This is the whole point: scopes lie on raw
    # log (color-science.md §2.3).
    patch = _solid(153, 153, 153)
    raw = analyze_rgb(patch, color_space="rec709")
    decoded = analyze_rgb(patch, color_space="sony_slog3")
    assert raw["decoded"] is False and decoded["decoded"] is True
    assert raw["exposure"]["verdict"] == "ok"
    assert decoded["exposure"]["verdict"] == "over"
    assert decoded["exposure"]["verdict"] != raw["exposure"]["verdict"]


def test_log_decode_expands_18pct_gray_to_display_midgray():
    # S-Log3 18% mid-gray sits at code ~0.4106 (105/255). Decoded to display it should land near
    # mid-gray code ~125 (0.18 linear ^ (1/2.4) * 255), proving the decode hits the documented anchor.
    patch = _solid(105, 105, 105)
    decoded = analyze_rgb(patch, color_space="sony_slog3")
    assert 118 <= decoded["luma"]["median"] <= 132


def test_logc3_and_vlog_decode_paths_engage():
    # Other known log spaces also decode (decoded flag True) and differ from the raw 709 read.
    patch = _solid(140, 140, 140)
    for cs in ("arri_logc3", "panasonic_vlog", "acescct"):
        decoded = analyze_rgb(patch, color_space=cs)
        raw = analyze_rgb(patch, color_space="rec709")
        assert decoded["decoded"] is True, cs
        assert decoded["luma"]["median"] != raw["luma"]["median"], cs


def test_linear_space_reencodes_to_display():
    # 'linear'/'aces' carry no log curve but are not directly viewable -> re-encoded to display gamma.
    decoded = analyze_rgb(_solid(46, 46, 46), color_space="linear")  # 0.18 linear ~ code 46
    assert decoded["decoded"] is True
    # 0.18 linear -> display ~code 125
    assert 118 <= decoded["luma"]["median"] <= 132
