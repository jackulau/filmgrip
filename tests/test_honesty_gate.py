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
    # Every op type the schema accepts must be handled somewhere (live, rebuild, or honestly rejected)
    # — i.e. it appears in the Resolve live sets, the interchange REBUILD_OPS, or is the documented
    # reject-only op (add_transition). Op list comes from the schema itself (no hand-coded list).
    from filmgrip.adapters.interchange import REBUILD_OPS
    from filmgrip.adapters.resolve_adapter import LIVE_EXTRA_OPS, LIVE_OPS

    op_literals = set(ep.all_op_names())
    assert op_literals, "schema must declare ops"
    handled = LIVE_OPS | LIVE_EXTRA_OPS | REBUILD_OPS | {"add_transition"}
    assert op_literals <= handled, f"unhandled ops: {op_literals - handled}"


def _parse_op_table() -> dict:
    """Parse the rendered op-support markdown into {op: (live, rebuild, inter)} (suffixes stripped)."""
    rows = {}
    for line in R.op_support_markdown().splitlines():
        if not line.startswith("| ") or line.startswith("| Op"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 4:
            continue
        rows[cells[0]] = (cells[1], cells[2], cells[3].split()[0])  # drop "(metadata)" etc.
    return rows


def test_op_support_table_is_complete_against_schema():
    # The table's op list must cover EXACTLY the schema's ops — a new op can't be silently omitted.
    assert set(R._OP_DISPLAY_ORDER) == set(ep.all_op_names())


def test_op_support_table_columns_are_derived_from_op_sets():
    # Self-enforcing: parse the PUBLISHED table and confirm each column matches the real op sets. If
    # anyone re-hand-codes the table with a wrong claim (e.g. delete "live: no"), this fails — the
    # published capability can't drift from what the adapters actually do.
    from filmgrip.adapters.interchange import REBUILD_OPS
    from filmgrip.adapters.resolve_adapter import LIVE_EXTRA_OPS, LIVE_OPS

    rows = _parse_op_table()
    assert set(rows) == set(ep.all_op_names())
    live_yes = {op for op, (live, _r, _i) in rows.items() if live == "yes"}
    rebuild_yes = {op for op, (_l, rebuild, _i) in rows.items() if rebuild == "yes"}
    inter_yes = {op for op, (_l, _r, inter) in rows.items() if inter == "yes"}
    assert live_yes == (LIVE_OPS | LIVE_EXTRA_OPS)
    assert rebuild_yes == REBUILD_OPS
    assert inter_yes == REBUILD_OPS
