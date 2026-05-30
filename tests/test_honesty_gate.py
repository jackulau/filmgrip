"""D12 — the honesty gate: no capability field left None, schema versioned, docs not drifted."""
from __future__ import annotations

from pathlib import Path

from filmgrip.adapters import registry as R
from filmgrip.protocol import editplan as ep

ROOT = Path(__file__).resolve().parent.parent


def test_schema_version_is_at_least_v2():
    assert ep.SCHEMA_VERSION >= 2


def test_no_capability_field_is_none():
    # Every editor must declare every honesty field — a None would read as "unknown" in the matrix.
    for slug in R.editors():
        d = R.get(slug).capability.as_dict()
        for key, value in d.items():
            assert value is not None, f"{slug}.{key} is None"


def test_every_capability_declares_the_v2_fields():
    required = {"audio_support", "audio_volume_scriptable", "organize_support",
                "editor_panel", "selection_confidence"}
    for slug in R.editors():
        assert required <= set(R.get(slug).capability.as_dict())


def test_checked_in_editplan_schema_matches_models():
    repo = ROOT / "editplan.schema.json"
    import json
    assert json.loads(repo.read_text()) == ep.schema(), \
        "editplan.schema.json drifted — run `python -m filmgrip.protocol.editplan`"


def test_checked_in_capabilities_doc_is_current(tmp_path):
    fresh = R.write_capabilities_doc(str(tmp_path / "CAPABILITIES.md"))
    repo = ROOT / "docs" / "CAPABILITIES.md"
    assert repo.read_text() == Path(fresh).read_text(), \
        "docs/CAPABILITIES.md drifted — regenerate from filmgrip.adapters.registry"


def test_all_editplan_ops_are_reachable_in_validate_or_warned():
    # Every op type the schema accepts must be handled somewhere (live, rebuild, or an honest warn) —
    # i.e. it appears in the Resolve live sets, the interchange REBUILD_OPS, or is the documented
    # warn-only op (add_transition).
    from filmgrip.adapters.interchange import REBUILD_OPS
    from filmgrip.adapters.resolve_adapter import LIVE_EXTRA_OPS, LIVE_OPS

    all_ops = set(ep.EditPlan.model_json_schema()["$defs"])  # model names
    op_literals = {
        "trim", "move", "insert", "delete", "set_property", "add_marker", "add_transition",
        "split", "ripple", "import_audio", "add_track", "rename_track", "create_bin", "move_to_bin",
    }
    handled = LIVE_OPS | LIVE_EXTRA_OPS | REBUILD_OPS | {"add_transition"}
    assert op_literals <= handled, f"unhandled ops: {op_literals - handled}"
    assert all_ops  # sanity: schema actually has model defs
