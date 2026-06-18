"""Color deliverable 1 — the ASC CDL primitive: value math, ergonomic compiler, the `set_cdl`
op, its validation, and its portable OTIO-metadata interchange path."""
from __future__ import annotations

import math

import pytest

from filmgrip.color import CDL, lgg_to_cdl
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol import validate as V
from filmgrip.protocol.editplan import SCHEMA_VERSION, EditPlan, all_op_names


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


# --------------------------------------------------------------------- CDL value math
def test_identity_is_a_noop():
    c = CDL.identity()
    assert c.is_identity
    for sample in ([0.0, 0.0, 0.0], [0.5, 0.25, 0.9], [1.0, 1.0, 1.0]):
        out = c.apply(sample)
        assert all(abs(a - b) < 1e-9 for a, b in zip(out, sample))


def test_slope_scales_offset_lifts_power_gammas():
    # slope 2x doubles (then clamps), offset adds, power<1 brightens midtones.
    assert CDL(slope=(2.0, 2.0, 2.0)).apply([0.25, 0.25, 0.25])[0] == pytest.approx(0.5)
    assert CDL(offset=(0.1, 0.1, 0.1)).apply([0.2, 0.2, 0.2])[0] == pytest.approx(0.3)
    got = CDL(power=(0.5, 0.5, 0.5)).apply([0.25, 0.25, 0.25])[0]
    assert got == pytest.approx(math.sqrt(0.25))  # x**0.5


def test_negative_base_is_clamped_before_power():
    # in*slope+offset goes negative; must clamp to 0 BEFORE the (fractional) power, not NaN.
    out = CDL(offset=(-1.0, -1.0, -1.0), power=(0.5, 0.5, 0.5)).apply([0.2, 0.2, 0.2])
    assert out == (0.0, 0.0, 0.0)
    assert not any(math.isnan(v) for v in out)


def test_saturation_zero_is_greyscale_rec709():
    out = CDL(saturation=0.0).apply([0.8, 0.2, 0.1])
    assert out[0] == pytest.approx(out[1]) == pytest.approx(out[2])
    luma = 0.2126 * 0.8 + 0.7152 * 0.2 + 0.0722 * 0.1
    assert out[0] == pytest.approx(luma)


def test_invalid_cdl_values_raise():
    with pytest.raises(ValueError):
        CDL(power=(0.0, 1.0, 1.0))      # power must be > 0
    with pytest.raises(ValueError):
        CDL(slope=(-1.0, 1.0, 1.0))     # slope must be >= 0
    with pytest.raises(ValueError):
        CDL(saturation=-0.5)            # sat must be >= 0


def test_to_resolve_shape():
    d = CDL(slope=(1.1, 1.0, 0.9), saturation=1.2).to_resolve(node_index=2)
    assert d["NodeIndex"] == "2"
    assert d["Slope"] == "1.1 1 0.9"
    assert d["Saturation"] == "1.2"


def test_to_from_dict_roundtrips():
    c = CDL(slope=(1.2, 1.0, 0.8), offset=(0.0, 0.01, -0.02),
            power=(1.0, 0.95, 1.05), saturation=1.1, color_space="arri_logc4")
    assert CDL.from_dict(c.to_dict()) == c


# ------------------------------------------------------------- ergonomic lift/gamma/gain compiler
def test_lgg_defaults_are_identity():
    assert lgg_to_cdl().is_identity


def test_lgg_maps_controls_to_sop():
    c = lgg_to_cdl(gain=1.5, lift=0.05, gamma=2.0)
    assert c.slope[0] == pytest.approx(1.5)          # gain -> slope
    assert c.offset[0] == pytest.approx(0.05)        # lift -> offset
    assert c.power[0] == pytest.approx(0.5)          # power = 1/gamma


def test_lgg_per_channel_triples():
    c = lgg_to_cdl(gain=(1.0, 1.0, 1.2))
    assert c.slope[2] == pytest.approx(1.2) and c.slope[0] == pytest.approx(1.0)


def test_lgg_temp_warms_red_cools_blue():
    c = lgg_to_cdl(temp=1.0)
    assert c.slope[0] > 1.0 > c.slope[2]             # +R, -B
    assert c.slope[1] == pytest.approx(1.0)


def test_lgg_contrast_pivots():
    # at the pivot, contrast leaves the value unchanged (out = (in-p)*c + p == in when in==p).
    p = 0.435
    c = lgg_to_cdl(contrast=1.5, pivot=p)
    assert c.apply([p, p, p])[0] == pytest.approx(p, abs=1e-6)


# ------------------------------------------------------------------------- the set_cdl op
def test_set_cdl_in_schema_and_versioned():
    assert "set_cdl" in all_op_names()
    assert SCHEMA_VERSION >= 4


def test_set_cdl_parses_with_defaults_and_explicit():
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": "c00000"},   # all-identity defaults
        {"op": "set_cdl", "clip_id": "c00001", "slope": [1.1, 1.0, 0.9],
         "offset": [0, 0.01, 0], "power": [1, 1, 1], "saturation": 1.2,
         "color_space": "rec709"},
    ]})
    assert plan.ops[0].slope == [1.0, 1.0, 1.0]
    assert plan.ops[1].color_space == "rec709"


def test_set_cdl_rejects_bad_shape_and_space():
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [{"op": "set_cdl", "clip_id": "x", "slope": [1.0, 1.0]}]})
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [{"op": "set_cdl", "clip_id": "x", "power": [0.0, 1.0, 1.0]}]})
    with pytest.raises(Exception):
        EditPlan.parse({"ops": [{"op": "set_cdl", "clip_id": "x", "color_space": "klingon"}]})


# ------------------------------------------------------------------------- validation
def test_validate_set_cdl_ok(cut):
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": _id(cut, "intro"), "saturation": 1.2}]})
    assert V.validate(plan, cut).ok


def test_validate_set_cdl_unknown_clip_rejected(cut):
    plan = EditPlan.parse({"ops": [{"op": "set_cdl", "clip_id": "deadbeef"}]})
    res = V.validate(plan, cut)
    assert not res.ok and V.UNKNOWN_CLIP in res.codes()


def test_dry_run_describes_grade(cut):
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": _id(cut, "intro"), "saturation": 0.5,
         "reason": "desaturate for the flashback"}]})
    out = V.dry_run(plan, cut)
    assert "grade" in out and "CDL" in out and "flashback" in out


# ----------------------------------------------------- portable interchange (OTIO metadata) path
def test_set_cdl_writes_portable_metadata_and_survives_roundtrip(cut, tmp_path):
    from filmgrip.adapters.interchange import InterchangeAdapter, OtioMutator

    intro = _id(cut, "intro")
    plan = EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": intro, "slope": [1.2, 1.0, 0.85],
         "offset": [0.0, 0.0, 0.02], "power": [1.0, 1.0, 1.0], "saturation": 1.15,
         "color_space": "rec709"}]})
    applied, unsupported = OtioMutator(cut).apply(plan)
    assert applied and not unsupported
    # OTIO wraps stored lists in its AnyVector container, so coerce with list() before comparing.
    meta = cut.clip(intro).otio.metadata["filmgrip"]["cdl"]
    assert list(meta["slope"]) == [1.2, 1.0, 0.85] and meta["saturation"] == 1.15
    assert meta["color_space"] == "rec709"

    # Round-trip through an .otio file: the grade must still be there after re-import.
    src = tmp_path / "in.otio"
    cut.to_otio_file(str(src))
    out = tmp_path / "graded.otio"
    res = InterchangeAdapter().apply(plan, str(src), out_path=str(out))
    assert res.ok, res.errors
    reloaded = TimelineIR.from_otio_file(str(out))
    rmeta = reloaded.clip(_id(reloaded, "intro")).otio.metadata["filmgrip"]["cdl"]
    assert list(rmeta["slope"]) == [1.2, 1.0, 0.85] and rmeta["saturation"] == 1.15
