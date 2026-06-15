"""Color deliverable 9 — grade packs / look presets. Deterministic look packs compile to portable,
valid set_cdl ops; prompt grade packs are registered and parameterized."""
from __future__ import annotations

import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.packs import all_packs, get_pack
from filmgrip.packs.engine import compile_pack
from filmgrip.protocol.validate import validate

LOOKS = ["teal-orange", "filmstock-warm", "bleach-bypass", "day-for-night"]


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def _ids(ir):
    return [c.id for c in ir.real_clips()]


def test_grade_packs_registered():
    names = {p.name for p in all_packs()}
    for n in LOOKS + ["neutral-balance", "grade-match"]:
        assert n in names, f"{n} not registered"


def test_look_packs_compile_to_valid_set_cdl(cut):
    ids = _ids(cut)
    for name in LOOKS:
        plan = compile_pack(get_pack(name), cut, ids)
        assert plan.ops, f"{name} emitted nothing"
        assert {op.op for op in plan.ops} == {"set_cdl"}
        # one grade per selected clip
        assert len(plan.ops) == len(ids)
        res = validate(plan, cut)
        assert res.ok, f"{name} invalid: {res.codes()}"


def test_bleach_bypass_lowers_saturation(cut):
    plan = compile_pack(get_pack("bleach-bypass"), cut, _ids(cut))
    assert all(op.saturation < 1.0 for op in plan.ops)


def test_teal_orange_warms_red_over_blue(cut):
    plan = compile_pack(get_pack("teal-orange"), cut, _ids(cut))
    op = plan.ops[0]
    assert op.slope[0] > op.slope[2]            # red gain above blue gain → warm


def test_day_for_night_is_cool_and_darker(cut):
    op = compile_pack(get_pack("day-for-night"), cut, _ids(cut)).ops[0]
    assert op.slope[2] > op.slope[0]            # blue above red → cool
    assert op.saturation < 1.0


def test_grade_packs_only_emit_applicable_ops(cut):
    # mirrors the project's applicability contract for the new look packs specifically
    from filmgrip.adapters.interchange import REBUILD_OPS
    from filmgrip.adapters.resolve_adapter import LIVE_OPS
    applicable = LIVE_OPS | REBUILD_OPS
    for name in LOOKS:
        plan = compile_pack(get_pack(name), cut, _ids(cut))
        assert {op.op for op in plan.ops} <= applicable


def test_prompt_grade_packs_are_prompt_kind():
    for name in ("neutral-balance", "grade-match"):
        p = get_pack(name)
        assert p.kind == "prompt" and p.prompt
    assert "{ref}" in get_pack("grade-match").prompt
