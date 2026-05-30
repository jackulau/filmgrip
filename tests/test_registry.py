"""D15 — adapter registry + capability matrix."""
from __future__ import annotations

from pathlib import Path

from filmgrip.adapters import registry as R
from filmgrip.adapters.base import Capabilities

EXPECTED_EDITORS = {
    "DaVinci Resolve (Studio)", "Final Cut Pro", "Premiere Pro", "Avid Media Composer",
    "Kdenlive", "Shotcut", "CapCut (International)", "Wondershare Filmora",
}


def test_every_entry_declares_a_capability_record():
    assert len(R.editors()) == 8
    for slug in R.editors():
        entry = R.get(slug)
        assert isinstance(entry.capability, Capabilities)
        assert entry.adapter is not None
        assert entry.capability.editor


def test_eight_editors_with_correct_roles():
    by_editor = {c.editor: c for c in R.capability_matrix()}
    assert set(by_editor) == EXPECTED_EDITORS
    assert by_editor["DaVinci Resolve (Studio)"].role == "flagship-native"
    assert by_editor["Final Cut Pro"].role == "interchange"
    assert by_editor["Avid Media Composer"].role == "best-effort"
    assert by_editor["CapCut (International)"].role == "best-effort"
    assert by_editor["Wondershare Filmora"].role == "read-only"


def test_filmora_write_back_is_false_resolve_true():
    assert R.get("filmora").capability.write_back is False
    assert R.get("resolve").capability.write_back is True
    # only Resolve requires the app running; everything else is file-based
    assert R.get("resolve").capability.requires_app_running is True
    assert R.get("finalcut").capability.requires_app_running is False


def test_markdown_table_lists_all_editors_and_roles():
    md = R.capability_markdown()
    for editor in EXPECTED_EDITORS:
        assert editor in md
    assert "flagship-native" in md
    assert "read-only" in md
    # Filmora's write-back shows the emphatic NO
    assert "| Wondershare Filmora | read-only | no | NO |" in md


def test_extension_routing():
    assert R.for_extension(".wfp").slug == "filmora"
    assert R.for_extension(".kdenlive").slug == "kdenlive"
    assert R.for_extension(".mlt").slug == "shotcut"
    assert R.for_extension(".fcpxml").slug == "finalcut"
    assert R.for_extension(".nope") is None


def test_capabilities_doc_generation(tmp_path):
    out = R.write_capabilities_doc(str(tmp_path / "docs" / "CAPABILITIES.md"))
    text = Path(out).read_text()
    assert "capability matrix" in text
    for editor in EXPECTED_EDITORS:
        assert editor in text
