"""D5 — audio EditPlan ops: import_audio, audio-aware insert, honest no-volume-scripting."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from filmgrip.core.ir import TimelineIR
from filmgrip.protocol import editplan as ep
from filmgrip.protocol import validate as V

FIX = "tests/fixtures/cut.otio"  # has audio tracks a1 (vo) and a2 (music_bed, gap, stinger)


def _ir():
    return TimelineIR.from_otio_file(FIX)


# -- parse ------------------------------------------------------------------
def test_import_audio_parses_with_sfx_name():
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a1", "at_start": 120, "sfx": "whoosh"},
    ]})
    assert plan.ops[0].op == "import_audio"
    assert plan.ops[0].sfx == "whoosh"
    assert plan.ops[0].duration is None  # full length by default


def test_import_audio_parses_with_explicit_src_ref():
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a2", "at_start": 0, "src_ref": "/sfx/boom.wav", "duration": 30},
    ]})
    assert plan.ops[0].src_ref.endswith("boom.wav") and plan.ops[0].duration == 30


def test_import_audio_requires_a_source():
    with pytest.raises(ValidationError) as exc:
        ep.EditPlan.parse({"ops": [{"op": "import_audio", "track": "a1", "at_start": 0}]})
    assert "sfx" in str(exc.value) or "src_ref" in str(exc.value)


def test_import_audio_rejects_non_audio_track_at_parse():
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [{"op": "import_audio", "track": "v1", "at_start": 0, "sfx": "x"}]})


def test_insert_media_type_defaults_auto_and_accepts_audio():
    plan = ep.EditPlan.parse({"ops": [
        {"op": "insert", "src_ref": "x.mov", "track": "v1", "at_start": 0, "duration": 24},
        {"op": "insert", "src_ref": "x.wav", "track": "a1", "at_start": 0, "duration": 24,
         "media_type": "audio"},
    ]})
    assert plan.ops[0].media_type == "auto"
    assert plan.ops[1].media_type == "audio"


def test_audio_volume_is_not_a_settable_property():
    # "lower the volume" must not silently succeed — volume isn't in the allowlist (Fairlight only).
    assert "volume" in ep.AUDIO_PROPS_UNSUPPORTED
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [
            {"op": "set_property", "clip_id": "c1", "key": "volume", "value": 0.5},
        ]})


# -- schema -----------------------------------------------------------------
def test_schema_includes_audio_and_organize_ops():
    defs = set(ep.schema()["$defs"])
    for model in ("ImportAudio", "AddTrack", "RenameTrack", "CreateBin", "MoveToBin"):
        assert model in defs


def test_schema_version_bumped():
    assert ep.SCHEMA_VERSION >= 2


# -- validate ---------------------------------------------------------------
def test_import_audio_valid_on_existing_audio_track_gap():
    ir = _ir()
    # a2 has a gap 100..260 -> place a 50f effect at 120, no overlap.
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a2", "at_start": 120, "sfx": "stinger", "duration": 50},
    ]})
    assert V.validate(plan, ir).ok


def test_import_audio_overlap_rejected_when_duration_known():
    ir = _ir()
    # a2 music_bed occupies 0..100 -> placing 0..50 overlaps.
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a2", "at_start": 0, "sfx": "boom", "duration": 50},
    ]})
    res = V.validate(plan, ir)
    assert not res.ok and V.ILLEGAL_OVERLAP in res.codes()


def test_import_audio_unknown_track_rejected():
    ir = _ir()
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a9", "at_start": 0, "sfx": "x"},
    ]})
    res = V.validate(plan, ir)
    assert not res.ok and V.TRACK_NOT_FOUND in res.codes()


def test_import_audio_unknown_duration_skips_overlap_check():
    ir = _ir()
    # No duration -> can't pre-check overlap; must validate OK (adapter places at apply time).
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a2", "at_start": 0, "sfx": "x"},
    ]})
    assert V.validate(plan, ir).ok


def test_import_audio_dry_run_describes():
    ir = _ir()
    plan = ep.EditPlan.parse({"ops": [
        {"op": "import_audio", "track": "a2", "at_start": 120, "sfx": "whoosh", "duration": 18},
    ]})
    diff = V.dry_run(plan, ir)
    assert "import_audio" in diff and "sfx:whoosh" in diff and "a2" in diff
