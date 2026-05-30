"""D14 — Filmora .wfp read-only parser (no write-back, honestly)."""
from __future__ import annotations

import pytest

from filmgrip.adapters.base import NotSupportedError
from filmgrip.adapters.filmora import FilmoraAdapter
from filmgrip.protocol.editplan import EditPlan


def test_reads_wfp_tracks_and_clips_readonly(fixtures_dir):
    ir = FilmoraAdapter().snapshot(str(fixtures_dir / "sample.wfp"))
    by = {c.name: c for c in ir.real_clips()}
    assert {"opening", "interview", "bgm"} <= set(by)
    assert (by["opening"].start, by["opening"].duration) == (0, 90)
    assert by["interview"].start == 90
    assert by["bgm"].track_kind == "audio"


def test_selection_announces_readonly(fixtures_dir):
    a = FilmoraAdapter()
    sel = a.get_selection(str(fixtures_dir / "sample.wfp"))
    assert sel.basis == "filmora_readonly"
    assert "read-only" in sel.note.lower()


def test_apply_raises_not_supported(fixtures_dir):
    a = FilmoraAdapter()
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": "c1", "frame": 0}]})
    with pytest.raises(NotSupportedError) as exc:
        a.apply(plan, str(fixtures_dir / "sample.wfp"))
    assert "no automation" in str(exc.value).lower() or "read-only" in str(exc.value).lower()


def test_capabilities_say_no_writeback():
    cap = FilmoraAdapter().capabilities()
    assert cap.write_back is False
    assert cap.role == "read-only"
    assert cap.live_selection is False


def test_non_zip_wfp_is_refused(tmp_path):
    bad = tmp_path / "old.wfp"
    bad.write_bytes(b"not a zip, an older Filmora binary format")
    with pytest.raises(NotSupportedError):
        FilmoraAdapter().snapshot(str(bad))
