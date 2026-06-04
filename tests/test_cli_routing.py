"""D5 (goal 004) — `film-grip edit --fixture` routes to the adapter the extension implies.

`film-grip editors` advertises CapCut / Filmora / Kdenlive / Shotcut, but the CLI only ever read OTIO
and applied via the interchange adapter — those editors were unreachable from the command line. Now
the fixture path resolves the adapter from the file extension (via registry.for_extension), so the
advertised editors are actually drivable, and Filmora honestly refuses to write.
"""
from __future__ import annotations

import json

from lxml import etree

from filmgrip.adapters.capcut import CapcutAdapter
from filmgrip.adapters.filmora import FilmoraAdapter
from filmgrip.adapters.interchange import InterchangeAdapter
from filmgrip.adapters.mlt import MltAdapter
from filmgrip.cli import main
from filmgrip.cli_edit import _adapter_for_fixture


# --------------------------------------------------------------- unit: extension -> adapter
def test_adapter_resolved_by_extension():
    assert isinstance(_adapter_for_fixture("x.mlt"), MltAdapter)
    assert isinstance(_adapter_for_fixture("x.kdenlive"), MltAdapter)
    assert isinstance(_adapter_for_fixture("draft.json"), CapcutAdapter)
    assert isinstance(_adapter_for_fixture("x.wfp"), FilmoraAdapter)
    # .otio (and unknown) fall back to the OTIO interchange adapter
    assert isinstance(_adapter_for_fixture("x.otio"), InterchangeAdapter)
    assert isinstance(_adapter_for_fixture("x.fcpxml"), InterchangeAdapter)


# --------------------------------------------------------------- e2e: an .mlt fixture really applies via MLT
def test_cli_drives_mlt_fixture(tmp_path, capsys):
    src = "tests/fixtures/sample.mlt"
    ir = MltAdapter().snapshot(src)
    bravo = next(c for c in ir.real_clips() if c.name == "bravo").id
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps({"ops": [{"op": "trim", "clip_id": bravo, "edge": "out", "delta": -12}]}))
    out = tmp_path / "out.mlt"
    code = main(["edit", "--fixture", src, "--plan", str(plan), "--out", str(out)])
    assert code == 0
    # output is real MLT XML (proves the MLT adapter ran, not the OTIO path — OTIO can't read .mlt)
    root = etree.parse(str(out)).getroot()
    assert root.tag == "mlt"
    assert next(c for c in MltAdapter().snapshot(str(out)).real_clips() if c.name == "bravo").duration == 60


# --------------------------------------------------------------- Filmora is read-only -> honest exit 3
def test_cli_filmora_fixture_refuses_write(tmp_path, capsys):
    src = "tests/fixtures/sample.wfp"
    ir = FilmoraAdapter().snapshot(src)
    cid = ir.real_clips()[0].id
    plan = tmp_path / "p.json"
    plan.write_text(json.dumps({"ops": [{"op": "add_marker", "clip_id": cid, "frame": 0}]}))
    out = tmp_path / "out.wfp"
    code = main(["edit", "--fixture", src, "--plan", str(plan), "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 3                       # documented "unsupported" exit
    assert "error:" in printed
    assert not out.exists()
