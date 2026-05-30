"""D5 — typed EditPlan protocol + JSON Schema export."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from filmgrip.protocol import editplan as ep

ROOT = Path(__file__).resolve().parent.parent


def test_parses_a_valid_multi_op_plan():
    plan = ep.EditPlan.parse({
        "notes": "tighten the open and flag the b-roll",
        "ops": [
            {"op": "trim", "clip_id": "c1", "edge": "out", "delta": -12},
            {"op": "add_marker", "clip_id": "c2", "frame": 0, "color": "blue", "name": "check"},
            {"op": "set_property", "clip_id": "c3", "key": "ZoomX", "value": 1.5},
            {"op": "move", "clip_id": "c4", "to_start": 240, "to_track": "v2"},
            {"op": "delete", "clip_id": "c5", "ripple": True},
        ],
    })
    assert len(plan.ops) == 5
    assert plan.ops[0].op == "trim" and plan.ops[0].delta == -12
    # marker color normalized to canonical casing
    assert plan.ops[1].color == "Blue"
    assert plan.version == ep.SCHEMA_VERSION


def test_rejects_disallowed_property_key():
    with pytest.raises(ValidationError) as exc:
        ep.EditPlan.parse({"ops": [
            {"op": "set_property", "clip_id": "c1", "key": "RmRfSlash", "value": 1},
        ]})
    assert "allowlist" in str(exc.value)


def test_rejects_unknown_op_and_extra_fields():
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [{"op": "format_drive", "clip_id": "c1"}]})
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [
            {"op": "delete", "clip_id": "c1", "ripple": False, "sneaky": "x"},
        ]})


def test_rejects_unknown_marker_color_and_transition():
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": "c1", "color": "Octarine"}]})
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [{"op": "add_transition", "clip_id": "c1", "type": "teleport"}]})


def test_numeric_bounds_enforced():
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [
            {"op": "insert", "src_ref": "x.mov", "track": "v1", "at_start": 0, "duration": 0},
        ]})  # duration must be > 0
    with pytest.raises(ValidationError):
        ep.EditPlan.parse({"ops": [{"op": "move", "clip_id": "c1", "to_start": -5}]})


def test_schema_emitted_and_is_valid_json_schema(tmp_path):
    out = ep.write_schema(str(tmp_path / "editplan.schema.json"))
    assert Path(out).exists()
    doc = json.loads(Path(out).read_text())
    # structural validity of the JSON Schema
    assert doc.get("type") == "object"
    assert "ops" in doc["properties"]
    assert "$defs" in doc
    op_defs = set(doc["$defs"])
    for model in ("Trim", "Move", "Insert", "Delete", "SetProperty", "AddMarker",
                  "AddTransition", "Split", "Ripple"):
        assert model in op_defs
    # If jsonschema is available, assert the schema itself is a valid draft.
    try:
        import jsonschema  # type: ignore

        jsonschema.Draft202012Validator.check_schema(doc)
    except ImportError:
        pass


def test_repo_schema_file_is_current():
    """The checked-in editplan.schema.json must match the live models (no drift)."""
    repo_schema = ROOT / "editplan.schema.json"
    assert repo_schema.exists(), "run python -m filmgrip.protocol.editplan to generate it"
    assert json.loads(repo_schema.read_text()) == ep.schema()
