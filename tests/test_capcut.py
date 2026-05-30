"""D13 — CapCut offline draft adapter (best-effort, encryption-gated)."""
from __future__ import annotations

import json

import pytest

from filmgrip.adapters.base import NotSupportedError
from filmgrip.adapters.capcut import CapcutAdapter
from filmgrip.protocol.editplan import EditPlan


def _id(ir, name):
    return next(c for c in ir.real_clips() if c.name == name).id


def test_parses_unencrypted_draft(fixtures_dir):
    ir = CapcutAdapter().snapshot(str(fixtures_dir / "capcut_draft.json"))
    by = {c.name: c for c in ir.real_clips()}
    assert {"clip_one", "clip_two", "clip_three", "song"} <= set(by)
    assert (by["clip_one"].start, by["clip_one"].duration) == (0, 60)  # 2.4Ms @25fps
    assert by["clip_two"].track_kind == "video"
    assert by["song"].track_kind == "audio"


def test_trim_roundtrips_microseconds(fixtures_dir, tmp_path):
    a = CapcutAdapter()
    src = str(fixtures_dir / "capcut_draft.json")
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [
        {"op": "trim", "clip_id": _id(ir, "clip_two"), "edge": "out", "delta": -10},
    ]})
    out = str(tmp_path / "out.json")
    res = a.apply(plan, src, out_path=out)
    assert res.ok, res.errors
    ir2 = a.snapshot(out)
    assert next(c for c in ir2.real_clips() if c.name == "clip_two").duration == 50
    # the JSON is still valid + still microseconds
    doc = json.loads(open(out).read())
    seg = doc["tracks"][0]["segments"][1]
    assert seg["target_timerange"]["duration"] == 2000000  # 50 frames * 40000us


def test_delete_removes_segment(fixtures_dir, tmp_path):
    a = CapcutAdapter()
    src = str(fixtures_dir / "capcut_draft.json")
    ir = a.snapshot(src)
    plan = EditPlan.parse({"ops": [{"op": "delete", "clip_id": _id(ir, "clip_two")}]})
    out = str(tmp_path / "out.json")
    assert a.apply(plan, src, out_path=out).ok
    ir2 = a.snapshot(out)
    assert "clip_two" not in {c.name for c in ir2.real_clips()}


def test_encrypted_draft_is_refused_clearly(fixtures_dir):
    a = CapcutAdapter()
    with pytest.raises(NotSupportedError) as exc:
        a.snapshot(str(fixtures_dir / "capcut_encrypted.json"))
    assert "encrypted" in str(exc.value).lower()


def test_apply_on_encrypted_does_not_corrupt(fixtures_dir, tmp_path):
    a = CapcutAdapter()
    plan = EditPlan.parse({"ops": []})
    with pytest.raises(NotSupportedError):
        a.apply(plan, str(fixtures_dir / "capcut_encrypted.json"), out_path=str(tmp_path / "x.json"))


def test_capabilities_are_honest():
    cap = CapcutAdapter().capabilities()
    assert cap.role == "best-effort"
    assert cap.live_selection is False
    assert cap.requires_app_running is False
