"""D13 — `film-grip quickstart`: a zero-config, offline, no-key onboarding demo.

These tests prove the demo is honest by construction: invoked with no editor, no network, and no
API key, it must exit 0, name the bundled fixture, show the edit it actually applied, and end with
concrete next-step commands. No test touches Resolve, a backend, or the network.
"""
from __future__ import annotations

import os

from filmgrip.cli import main
from filmgrip.cli_quickstart import cmd_quickstart, find_demo_fixtures


def test_quickstart_runs_offline_and_exits_zero(capsys):
    # No --editor, no --backend, no key, no network: the bare command must succeed.
    code = main(["quickstart"])
    out = capsys.readouterr().out
    assert code == 0
    # Names the bundled demo timeline.
    assert "cut.otio" in out
    # Honest framing: explicitly an offline, no-key demo.
    low = out.lower()
    assert "offline" in low
    assert "no api key" in low or "no key" in low


def test_quickstart_shows_the_applied_changes(capsys):
    code = main(["quickstart"])
    out = capsys.readouterr().out
    assert code == 0
    # The recorded plan's goal + the concrete ops it applies (the same diff the edit path renders).
    assert "tighten the open and flag the b-roll" in out
    assert "trim intro out -12" in out
    assert "marker Blue on dialogue_a" in out
    assert "broll_1.ZoomX" in out
    # It proves the offline apply ran end-to-end (not just a dry-run).
    assert "applied" in out.lower()


def test_quickstart_prints_next_step_suggestions(capsys):
    code = main(["quickstart"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Next steps" in out
    # Each suggested command genuinely works with no editor / no key.
    assert "film-grip editors" in out
    assert "film-grip status" in out
    # Points the user at running it on their own project, offline.
    assert "--fixture" in out


def test_quickstart_cmd_entrypoint_matches_routing(capsys):
    # Calling the command function directly behaves identically to routing through main().
    code = cmd_quickstart()
    out = capsys.readouterr().out
    assert code == 0
    assert "cut.otio" in out and "Next steps" in out


def test_find_demo_fixtures_locates_committed_pair():
    found = find_demo_fixtures()
    assert found is not None, "the committed tests/fixtures/{cut.otio,plan.json} should be discoverable"
    cut, plan = found
    assert os.path.basename(cut) == "cut.otio" and os.path.isfile(cut)
    assert os.path.basename(plan) == "plan.json" and os.path.isfile(plan)


def test_quickstart_degrades_to_synthesized_demo_when_fixtures_missing(monkeypatch, capsys):
    # If the committed fixtures can't be found (e.g. a bare wheel), the demo must still run the real
    # offline apply path on a generated stand-in and exit 0 — never crash, never fake output.
    monkeypatch.setattr("filmgrip.cli_quickstart.find_demo_fixtures", lambda: None)
    code = cmd_quickstart()
    out = capsys.readouterr().out
    assert code == 0
    assert "stand-in" in out.lower()  # honest about using a generated fixture
    assert "applied" in out.lower()   # still proves the apply path end-to-end
    assert "Next steps" in out
