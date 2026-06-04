"""D1 (goal 004) — the honest apply contract.

film-grip must NEVER report success for an edit it didn't perform. When an adapter can't apply a
requested op, that op is ``unsupported`` and ``ApplyResult.ok`` is False — not ok=True with a buried
warning (the warn-and-noop the project forbids). A partial apply (some ops land, some can't) is also
ok=False, with the file holding only what actually applied. The CLI maps a non-empty ``unsupported``
to exit code 3 (its documented "unsupported" code), and writes nothing when nothing applied.

These tests would all pass against the OLD dishonest code only if the fake-success were real — i.e.
they fail loudly the moment any adapter goes back to returning ok=True for a dropped op.
"""
from __future__ import annotations

import json

from filmgrip.adapters import resolve_client as rc
from filmgrip.adapters.capcut import CapcutAdapter
from filmgrip.adapters.interchange import InterchangeAdapter
from filmgrip.adapters.mlt import MltAdapter
from filmgrip.adapters.resolve_adapter import ResolveAdapter
from filmgrip.cli import main
from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import EditPlan
from tests.fakes import make_two_track_resolve

OTIO = "tests/fixtures/cut.otio"


def _first_id(ir):
    return ir.real_clips()[0].id


# --------------------------------------------------------------- per-adapter contract
def test_interchange_unsupported_op_is_ok_false(tmp_path):
    a = InterchangeAdapter()
    ir = a.snapshot(OTIO)
    out = tmp_path / "o.otio"
    plan = EditPlan.parse({"ops": [
        {"op": "add_transition", "clip_id": _first_id(ir), "edge": "out", "type": "cross_dissolve"},
    ]})
    res = a.apply(plan, OTIO, out_path=str(out))
    assert res.ok is False
    assert res.applied == []
    assert any("add_transition" in u for u in res.unsupported)
    assert not out.exists()  # nothing applied → no file written


def test_capcut_unsupported_op_is_ok_false(tmp_path):
    a = CapcutAdapter()
    src = "tests/fixtures/capcut_draft.json"
    ir = a.snapshot(src)
    out = tmp_path / "o.json"
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": _first_id(ir), "frame": 0}]})
    res = a.apply(plan, src, out_path=str(out))
    assert res.ok is False
    assert any("add_marker" in u for u in res.unsupported)
    assert not out.exists()


def test_mlt_unsupported_op_is_ok_false(tmp_path):
    a = MltAdapter()
    src = "tests/fixtures/sample.mlt"
    ir = a.snapshot(src)
    out = tmp_path / "o.mlt"
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": _first_id(ir), "frame": 0}]})
    res = a.apply(plan, src, out_path=str(out))
    assert res.ok is False
    assert any("add_marker" in u for u in res.unsupported)
    assert not out.exists()


def test_resolve_unsupported_op_is_ok_false():
    adapter = ResolveAdapter()
    session = rc.connect(make_two_track_resolve())
    ir = adapter.snapshot(session)
    plan = EditPlan.parse({"ops": [
        {"op": "add_transition", "clip_id": _first_id(ir), "edge": "out", "type": "cross_dissolve"},
    ]})
    res = adapter.apply(plan, session)
    assert res.ok is False
    assert any("add_transition" in u for u in res.unsupported)


# --------------------------------------------------------------- partial apply
def test_partial_apply_is_ok_false_but_keeps_applied_op(tmp_path):
    """One op lands (add_marker), one can't (add_transition): ok=False, file has the marker."""
    a = InterchangeAdapter()
    ir = a.snapshot(OTIO)
    cid = _first_id(ir)
    out = tmp_path / "o.otio"
    plan = EditPlan.parse({"ops": [
        {"op": "add_marker", "clip_id": cid, "frame": 0},
        {"op": "add_transition", "clip_id": cid, "edge": "out", "type": "cross_dissolve"},
    ]})
    res = a.apply(plan, OTIO, out_path=str(out))
    assert res.ok is False                       # not everything the user asked for happened
    assert len(res.applied) == 1 and "marker" in res.applied[0]   # but the marker DID apply
    assert any("add_transition" in u for u in res.unsupported)
    assert out.exists()                          # the applied op was written


# --------------------------------------------------------------- CLI exit codes
def test_cli_exits_3_on_unsupported(tmp_path, capsys):
    ir = TimelineIR.from_otio_file(OTIO)
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps({"ops": [
        {"op": "add_transition", "clip_id": ir.real_clips()[0].id, "edge": "out",
         "type": "cross_dissolve"},
    ]}))
    out = tmp_path / "o.otio"
    code = main(["edit", "--fixture", OTIO, "--plan", str(plan), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 3                             # documented "unsupported" exit code
    assert "✗" in printed and "add_transition" in printed
    assert not out.exists()


def test_cli_exits_0_on_full_apply(tmp_path, capsys):
    ir = TimelineIR.from_otio_file(OTIO)
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps({"ops": [
        {"op": "add_marker", "clip_id": ir.real_clips()[0].id, "frame": 0},
    ]}))
    out = tmp_path / "o.otio"
    code = main(["edit", "--fixture", OTIO, "--plan", str(plan), "--out", str(out)])
    assert code == 0
    assert out.exists()
