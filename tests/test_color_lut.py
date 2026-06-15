"""Color deliverable 2 — the apply_lut *look* primitive: .cube/.3dl parsing + sanity, the op,
its validation (missing/malformed rejected, bare name warned), and the interchange path."""
from __future__ import annotations

import pytest

from filmgrip.color.lut import (LutError, inspect_lut, looks_like_bare_reference,
                                parse_3dl, parse_cube)
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol import validate as V
from filmgrip.protocol.editplan import EditPlan, all_op_names

# A minimal valid 3D cube, size 2 → 2**3 = 8 rows.
CUBE_3D = """TITLE "tiny"
LUT_3D_SIZE 2
0 0 0
1 0 0
0 1 0
1 1 0
0 0 1
1 0 1
0 1 1
1 1 1
"""

CUBE_1D = """# a 1D curve
LUT_1D_SIZE 2
0.0 0.0 0.0
1.0 1.0 1.0
"""


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def _write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


# ------------------------------------------------------------------------ .cube parsing
def test_parse_cube_3d_ok():
    info = parse_cube(CUBE_3D)
    assert info.kind == "3D" and info.size == 2 and info.points == 8 and info.title == "tiny"


def test_parse_cube_1d_ok():
    info = parse_cube(CUBE_1D)
    assert info.kind == "1D" and info.size == 2 and info.points == 2


def test_parse_cube_rejects_no_size():
    with pytest.raises(LutError):
        parse_cube("0 0 0\n1 1 1\n")


def test_parse_cube_rejects_both_sizes():
    with pytest.raises(LutError):
        parse_cube("LUT_1D_SIZE 2\nLUT_3D_SIZE 2\n")


def test_parse_cube_rejects_wrong_row_count():
    with pytest.raises(LutError):
        parse_cube("LUT_3D_SIZE 2\n0 0 0\n1 1 1\n")   # declares 8, has 2


def test_parse_cube_rejects_nonnumeric_and_bad_width():
    with pytest.raises(LutError):
        parse_cube("LUT_1D_SIZE 2\nfoo bar baz\n0 0 0\n")
    with pytest.raises(LutError):
        parse_cube("LUT_1D_SIZE 2\n0 0\n1 1\n")        # 2 values, not 3


# ------------------------------------------------------------------------ .3dl + inspect
def test_parse_3dl_perfect_cube():
    rows = "\n".join("0 0 0" for _ in range(8))        # 8 = 2**3
    assert parse_3dl(rows).size == 2


def test_parse_3dl_rejects_non_cube():
    rows = "\n".join("0 0 0" for _ in range(5))
    with pytest.raises(LutError):
        parse_3dl(rows)


def test_inspect_lut_missing_and_unsupported(tmp_path):
    with pytest.raises(LutError):
        inspect_lut(str(tmp_path / "nope.cube"))
    bad = _write(tmp_path, "look.xyz", "whatever")
    with pytest.raises(LutError):
        inspect_lut(bad)


def test_inspect_lut_reads_cube(tmp_path):
    info = inspect_lut(_write(tmp_path, "look.cube", CUBE_3D))
    assert info.kind == "3D"


def test_bare_reference_detection():
    assert looks_like_bare_reference("Kodak2383")
    assert not looks_like_bare_reference("looks/Kodak2383.cube")
    assert not looks_like_bare_reference("Kodak2383.cube")


# ------------------------------------------------------------------------ the op + validation
def test_apply_lut_in_schema():
    assert "apply_lut" in all_op_names()


def test_validate_apply_lut_existing_file_ok(cut, tmp_path):
    lut = _write(tmp_path, "look.cube", CUBE_3D)
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(cut, "intro"), "path": lut, "node_index": 2}]})
    assert V.validate(plan, cut).ok


def test_validate_apply_lut_missing_file_rejected(cut):
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(cut, "intro"), "path": "/tmp/does_not_exist.cube"}]})
    res = V.validate(plan, cut)
    assert not res.ok and V.LUT_NOT_FOUND in res.codes()


def test_validate_apply_lut_malformed_rejected(cut, tmp_path):
    bad = _write(tmp_path, "broken.cube", "LUT_3D_SIZE 2\n0 0 0\n")  # too few rows
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(cut, "intro"), "path": bad}]})
    res = V.validate(plan, cut)
    assert not res.ok and V.LUT_INVALID in res.codes()


def test_validate_apply_lut_bare_name_warns_not_errors(cut):
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(cut, "intro"), "path": "Kodak2383"}]})
    res = V.validate(plan, cut)
    assert res.ok                                   # allowed (editor resolves it) ...
    assert any(w.code == V.LUT_NOT_FOUND for w in res.warnings)   # ... but warned


def test_dry_run_describes_apply_lut(cut, tmp_path):
    lut = _write(tmp_path, "teal_orange.cube", CUBE_3D)
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": _id(cut, "intro"), "path": lut}]})
    out = V.dry_run(plan, cut)
    assert "apply LUT teal_orange.cube" in out


# ------------------------------------------------------------------------ interchange path
def test_apply_lut_writes_metadata_and_roundtrips(cut, tmp_path):
    from filmgrip.adapters.interchange import InterchangeAdapter, OtioMutator

    lut = _write(tmp_path, "look.cube", CUBE_3D)
    intro = _id(cut, "intro")
    plan = EditPlan.parse({"ops": [
        {"op": "apply_lut", "clip_id": intro, "path": lut, "node_index": 3}]})
    applied, unsupported = OtioMutator(cut).apply(plan)
    assert applied and not unsupported
    luts = cut.clip(intro).otio.metadata["filmgrip"]["luts"]
    assert list(luts)[0]["path"] == lut and list(luts)[0]["node_index"] == 3

    src = tmp_path / "in.otio"
    cut.to_otio_file(str(src))
    out = tmp_path / "out.otio"
    res = InterchangeAdapter().apply(plan, str(src), out_path=str(out))
    assert res.ok, res.errors
    reloaded = TimelineIR.from_otio_file(str(out))
    rluts = reloaded.clip(_id(reloaded, "intro")).otio.metadata["filmgrip"]["luts"]
    assert list(rluts)[0]["path"] == lut
