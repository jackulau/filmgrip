"""D10 — MVP end-to-end CLI, provable offline via --fixture/--plan/--dry-run."""
from __future__ import annotations

import json

from filmgrip.cli import main

FIX = "tests/fixtures/cut.otio"
PLAN = "tests/fixtures/plan.json"


def test_edit_fixture_dry_run_renders_expected_diff(capsys):
    code = main(["edit", "--fixture", FIX, "--plan", PLAN, "--dry-run"])
    out = capsys.readouterr().out
    assert code == 0
    assert "PLAN OK — 3 op(s)" in out
    assert "tighten the open and flag the b-roll" in out
    assert "trim intro out -12  →  0..48 ⇒ 0..36" in out
    assert "marker Blue on dialogue_a @+0 'review'" in out
    assert "set broll_1.ZoomX = 1.2" in out


def test_edit_fixture_rejects_invalid_plan(tmp_path, capsys):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"ops": [{"op": "delete", "clip_id": "ghost"}]}))
    code = main(["edit", "--fixture", FIX, "--plan", str(bad), "--dry-run"])
    out = capsys.readouterr().out
    assert code == 1
    assert "PLAN REJECTED" in out
    assert "UNKNOWN_CLIP" in out


def test_edit_fixture_without_plan_or_prompt_explains_itself(capsys):
    # Fixture mode now accepts EITHER --plan (replay) OR a prompt (plan against the fixture via the
    # backend). With neither, it explains both offline routes.
    code = main(["edit", "--fixture", FIX])  # no --plan, no prompt, no editor
    out = capsys.readouterr().out
    assert code == 3
    assert "--plan" in out and "prompt" in out


def test_edit_fixture_with_prompt_plans_through_backend(capsys):
    # A prompt + --fixture plans with no live editor; codex surfaces its honest not-implemented.
    code = main(["edit", "--fixture", FIX, "--backend", "codex", "make the open punchier"])
    out = capsys.readouterr().out
    assert code == 1
    assert "not yet implemented" in out


def _otio_three(tmp_path):
    """A1-clip-per-48f V1 timeline A|B|C — clean ground for cut/split assertions."""
    import opentimelineio as otio
    rate = 24.0

    def rt(f):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="cutdemo")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    for nm in ("A", "B", "C"):
        v.append(otio.schema.Clip(
            name=nm, media_reference=otio.schema.ExternalReference(target_url=f"/m/{nm}.mov"),
            source_range=otio.opentime.TimeRange(rt(0), rt(48))))
    tl.tracks.append(v)
    p = str(tmp_path / "cutdemo.otio")
    otio.adapters.write_to_file(tl, p)
    return p


def _clip_id(src, name):
    from filmgrip.core.ir import TimelineIR
    return next(c for c in TimelineIR.from_otio_file(src).real_clips() if c.name == name).id


def test_ripple_delete_cut_closes_gap_via_cli(tmp_path):
    from filmgrip.core.ir import TimelineIR
    src = _otio_three(tmp_path)  # A 0..48, B 48..96, C 96..144
    plan = tmp_path / "cut.json"
    plan.write_text(json.dumps({"notes": "cut B out",
                                "ops": [{"op": "delete", "clip_id": _clip_id(src, "B"), "ripple": True}]}))
    out = str(tmp_path / "cut.edited.otio")
    code = main(["edit", "--fixture", src, "--plan", str(plan), "--out", out])
    assert code == 0
    ir2 = TimelineIR.from_otio_file(out)
    assert [c.name for c in ir2.real_clips()] == ["A", "C"]
    assert next(c for c in ir2.real_clips() if c.name == "C").start == 48  # gap closed by ripple


def test_split_razor_cut_via_cli(tmp_path):
    from filmgrip.core.ir import TimelineIR
    src = _otio_three(tmp_path)
    plan = tmp_path / "split.json"
    # B is 48..96 — razor at 72 yields two 24f halves.
    plan.write_text(json.dumps({"ops": [{"op": "split", "clip_id": _clip_id(src, "B"), "at_frame": 72}]}))
    out = str(tmp_path / "split.edited.otio")
    code = main(["edit", "--fixture", src, "--plan", str(plan), "--out", out])
    assert code == 0
    ir2 = TimelineIR.from_otio_file(out)
    halves = [c for c in ir2.real_clips() if c.name == "B"]
    assert len(halves) == 2 and {c.duration for c in halves} == {24}


def test_version_exits_zero():
    import pytest

    # --version is handled by argparse (raises SystemExit) and must not import the heavy pipeline.
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0
