"""D5 — post-apply verification: expected-state simulation, geometry diff, boundary sheets.

The loop under test: simulate the plan on a copy (same OTIO mutator) → diff against a fresh
post-apply snapshot → render contact sheets at the new cut boundaries. Honest paths: ops the
rebuild can't model are 'skipped', sheet failures are warnings, a geometry mismatch is exit 1.
"""
from __future__ import annotations

import json
import shutil
import subprocess

import opentimelineio as otio
import pytest

from filmgrip.adapters.interchange import OtioMutator
from filmgrip.core.ir import TimelineIR
from filmgrip.perception.verify import (
    diff_geometry,
    expected_after,
    geometry_rows,
    new_boundaries,
    verify_apply,
)
from filmgrip.protocol.editplan import EditPlan


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


def _clip(name: str, src_in: int, dur: int, url: str = "") -> otio.schema.Clip:
    return otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(
            target_url=url or f"/footage/{name}.mov"),
        source_range=otio.opentime.TimeRange(_rt(src_in), _rt(dur)),
    )


def _ir(url: str = "") -> TimelineIR:
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("clipA", 0, 480, url))
    track.append(_clip("clipB", 0, 240, url))
    return TimelineIR(tl)


def _cut_plan(ir, start=100, end=148) -> EditPlan:
    a = ir.real_clips()[0].id
    return EditPlan.parse({"ops": [{"op": "cut_range", "clip_id": a,
                                    "start_frame": start, "end_frame": end}]})


# --------------------------------------------------------------------------- simulation
def test_expected_after_simulates_on_a_copy():
    ir = _ir()
    plan = _cut_plan(ir)
    expected, skipped = expected_after(ir, plan)
    assert skipped == []
    assert ir.duration == 720                      # original untouched
    assert expected.duration == 720 - 48           # simulation took the cut


def test_expected_after_reports_unmodelable_ops():
    ir = _ir()
    plan = EditPlan.parse({"ops": [
        {"op": "add_track", "kind": "audio"},
        {"op": "import_audio", "track": "a1", "src_ref": "/sfx/whoosh.wav", "at_start": 0},
    ]})
    _, skipped = expected_after(ir, plan)
    assert skipped == ["add_track", "import_audio"]


# --------------------------------------------------------------------------- geometry
def test_geometry_rows_merge_adjacent_and_drop_trailing_gaps():
    tl = otio.schema.Timeline(name="t")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("a", 0, 48))
    track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(_rt(0), _rt(10))))
    track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(_rt(0), _rt(14))))
    track.append(_clip("b", 0, 48))
    track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(_rt(0), _rt(99))))
    rows = geometry_rows(TimelineIR(tl))["v1"]
    assert [r[0] for r in rows] == ["clip", "gap", "clip"]
    assert rows[1][2] == 24                        # 10+14 merged


def test_diff_geometry_flags_divergence_with_location():
    ir = _ir()
    plan = _cut_plan(ir)
    expected, _ = expected_after(ir, plan)
    actual, _ = expected_after(ir, plan)
    # tamper: pretend the editor only removed half the range
    item = actual.real_clips()[1].otio
    sr = item.source_range
    item.source_range = otio.opentime.TimeRange(sr.start_time, _rt(360))
    actual.reindex()
    mismatches, verified = diff_geometry(expected, actual)
    assert mismatches and "v1" in mismatches[0]
    assert any("dur 332" in m and "dur 360" in m for m in mismatches)


def test_diff_geometry_catches_enabled_and_retime_state():
    ir = _ir()
    plan = EditPlan.parse({"ops": [
        {"op": "set_enabled", "clip_id": ir.real_clips()[0].id, "enabled": False},
        {"op": "retime", "clip_id": ir.real_clips()[1].id, "speed_percent": 200.0},
    ]})
    expected, _ = expected_after(ir, plan)
    mismatches, _ = diff_geometry(expected, ir)    # actual = the unedited timeline
    joined = " ".join(mismatches)
    assert "disabled" in joined
    assert "LinearTimeWarp" in joined


# --------------------------------------------------------------------------- boundaries
def test_new_boundaries_are_the_fresh_cut_edges():
    ir = _ir()
    expected, _ = expected_after(ir, _cut_plan(ir))
    assert new_boundaries(ir, expected) == [100, 432, 672]
    # 100 = the cut point; 432/672 = clipB & end shifted left by 48 (new edges); 0/480 existed


# --------------------------------------------------------------------------- whole loop
def test_verify_apply_green_path():
    ir = _ir()
    plan = _cut_plan(ir)
    actual = TimelineIR(ir.timeline.deepcopy())
    OtioMutator(actual).apply(plan)
    actual.reindex()
    report = verify_apply(ir, plan, actual)
    assert report.ok
    assert any("match the expected geometry" in v for v in report.verified)
    assert report.mismatches == []
    assert report.boundaries == [100, 432, 672]


def test_verify_apply_red_path():
    ir = _ir()
    plan = _cut_plan(ir)
    report = verify_apply(ir, plan, ir)            # editor "did nothing"
    assert not report.ok
    assert report.mismatches


FFMPEG = shutil.which("ffmpeg")


@pytest.mark.skipif(FFMPEG is None, reason="ffmpeg not installed")
def test_verify_apply_renders_boundary_sheets(tmp_path):
    media = tmp_path / "real.mp4"
    subprocess.run(
        [FFMPEG, "-y", "-v", "error",
         "-f", "lavfi", "-i", "testsrc=duration=31:size=64x48:rate=24",
         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(media)],
        check=True, capture_output=True)
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("real", 0, 720, media.as_uri()))
    ir = TimelineIR(tl)
    plan = _cut_plan(ir, 100, 148)
    actual = TimelineIR(ir.timeline.deepcopy())
    OtioMutator(actual).apply(plan)
    actual.reindex()
    import os
    os.environ["FILMGRIP_CACHE_DIR"] = str(tmp_path / "cache")
    report = verify_apply(ir, plan, actual, sheets=True)
    assert report.ok
    assert report.sheets, report.sheet_errors
    assert report.sheets[0].png_path.endswith(".png")


# --------------------------------------------------------------------------- CLI + panel
def test_cli_edit_verify_flag(tmp_path, capsys):
    ir = _ir()
    fixture = tmp_path / "seq.otio"
    otio.adapters.write_to_file(ir.timeline, str(fixture))
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(
        {"ops": [{"op": "cut_range", "clip_id": ir.real_clips()[0].id,
                  "start_frame": 100, "end_frame": 148}]}))
    from filmgrip.cli import main

    rc = main(["edit", "--fixture", str(fixture), "--plan", str(plan_file), "--verify",
               "--out", str(tmp_path / "out.otio")])
    out = capsys.readouterr().out
    assert rc == 0
    assert "match the expected geometry" in out


def test_cli_pack_silence_cut_with_verify(tmp_path, capsys, monkeypatch):
    words = [{"t": "Hey", "s": 1.0, "e": 1.4}, {"t": "back", "s": 6.0, "e": 6.4}]
    asr_json = tmp_path / "asr.json"
    asr_json.write_text(json.dumps({"words": words}))
    monkeypatch.setenv("FILMGRIP_ASR_BACKEND", "fake")
    monkeypatch.setenv("FILMGRIP_FAKE_ASR_JSON", str(asr_json))
    monkeypatch.setenv("FILMGRIP_CACHE_DIR", str(tmp_path / "cache"))
    media = tmp_path / "interview.mov"
    media.write_bytes(b"\x00" * 32)                # fake media: sheets will warn, not fail
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("take1", 0, 480, media.as_uri()))
    fixture = tmp_path / "seq.otio"
    otio.adapters.write_to_file(tl, str(fixture))
    from filmgrip.cli import main

    rc = main(["pack", "apply", "silence-cut", "--fixture", str(fixture), "--verify",
               "--out", str(tmp_path / "out.otio")])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "match the expected geometry" in out


def test_panel_shows_verify_evidence():
    from filmgrip.adapters.base import ApplyResult
    from filmgrip.ui.panel import format_apply_body

    res = ApplyResult(ok=True, diff="PLAN OK", verified=["v1: 3 item(s) match"],
                      mismatches=["v1 item 0: expected clip @0 dur 100, got dur 90"])
    body = format_apply_body(res)
    assert "✓ v1: 3 item(s) match" in body
    assert "✗ verify mismatch" in body


def test_emit_apply_result_maps_mismatch_to_exit_1(capsys):
    from filmgrip.adapters.base import ApplyResult
    from filmgrip.cli_edit import _emit_apply_result

    res = ApplyResult(ok=True, diff="PLAN OK", mismatches=["v1: drift"])
    assert _emit_apply_result(res) == 1
    assert "verify mismatch" in capsys.readouterr().out
