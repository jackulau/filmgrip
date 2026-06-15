"""Color deliverable 7 — the color verify loop: apply a CDL to a frame (predictive engine),
compute scope deltas, and judge a grade against a target/reference within tolerance."""
from __future__ import annotations

import numpy as np

from filmgrip.color import CDL, lgg_to_cdl
from filmgrip.perception import verify as V
from filmgrip.perception.scopes import (analyze_rgb, apply_cdl_array, predict_grade_reading)


def _solid(r, g, b, h=16, w=16):
    a = np.zeros((h, w, 3), dtype=np.uint8)
    a[:, :, 0], a[:, :, 1], a[:, :, 2] = r, g, b
    return a


# ----------------------------------------------------------------------- predictive engine
def test_identity_cdl_leaves_frame_unchanged():
    src = _solid(100, 120, 140)
    out = apply_cdl_array(src, CDL.identity())
    assert np.allclose(out, src, atol=1)


def test_slope_doubles_value():
    out = apply_cdl_array(_solid(64, 64, 64), CDL(slope=(2.0, 2.0, 2.0)))
    assert abs(int(out[0, 0, 0]) - 128) <= 1


def test_predict_desaturation_drops_vectorscope_saturation():
    src = _solid(220, 40, 40)
    base = analyze_rgb(src)["vectorscope"]["saturation"]
    pred = predict_grade_reading(src, CDL(saturation=0.0))["vectorscope"]["saturation"]
    assert pred < base
    assert pred < 0.01


def test_predict_gain_raises_exposure():
    src = _solid(100, 100, 100)
    pred = predict_grade_reading(src, lgg_to_cdl(gain=1.4))
    assert pred["luma"]["median"] > analyze_rgb(src)["luma"]["median"]


# ----------------------------------------------------------------------- delta + verdict
def test_grade_delta_signs():
    before = analyze_rgb(_solid(100, 100, 100))
    after = analyze_rgb(_solid(150, 100, 100))   # redder + brighter
    d = V.grade_delta(before, after)
    assert d["d_luma_median"] > 0
    assert d["d_wb"]["r"] > 0


def test_verify_grade_matches_itself():
    reading = analyze_rgb(_solid(120, 120, 120))
    verdict = V.verify_grade(reading, reading)
    assert verdict.matched is True and verdict.failed == []


def test_verify_grade_flags_out_of_tolerance():
    after = analyze_rgb(_solid(40, 40, 40))
    target = analyze_rgb(_solid(200, 200, 200))
    verdict = V.verify_grade(after, target)
    assert verdict.matched is False
    assert "luma_median" in verdict.failed


# ----------------------------------------------------------------------- the full predictive loop
def test_predicted_grade_matches_target_within_tolerance():
    # PERCEIVE source, PROPOSE a gain to hit the target's brightness, PREDICT, VERIFY.
    source = _solid(100, 100, 100)
    target = analyze_rgb(_solid(150, 150, 150))     # the look we want
    proposed = lgg_to_cdl(gain=1.5)                 # 100 * 1.5 = 150
    predicted = predict_grade_reading(source, proposed)
    verdict = V.verify_grade(predicted, target)
    assert verdict.matched is True, verdict.failed


def test_wrong_grade_fails_then_corrected_passes():
    source = _solid(100, 100, 100)
    target = analyze_rgb(_solid(150, 150, 150))
    too_weak = V.verify_grade(predict_grade_reading(source, lgg_to_cdl(gain=1.1)), target)
    assert too_weak.matched is False                # 110 ≠ 150
    corrected = V.verify_grade(predict_grade_reading(source, lgg_to_cdl(gain=1.5)), target)
    assert corrected.matched is True                # iterate → match


def test_module_exposes_verify_api():
    assert hasattr(V, "verify_grade") and hasattr(V, "grade_delta")
