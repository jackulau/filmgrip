"""D4 — FGX compact context serializer."""
from __future__ import annotations

import opentimelineio as otio
import pytest

from filmgrip.core.ir import TimelineIR
from filmgrip.serialize import fgx


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def _sel(cut, name):
    return next(c for c in cut.real_clips() if c.name == name).id


def test_track_code_roundtrip():
    assert fgx.track_code("video", 1) == "v1"
    assert fgx.track_code("audio", 2) == "a2"
    assert fgx.parse_track_code("v1") == ("video", 1)
    assert fgx.parse_track_code("a2") == ("audio", 2)


def test_subgraph_is_one_hop_and_excludes_far_clips(cut):
    sel = _sel(cut, "broll_1")
    b = fgx.bundle(cut, [sel], hops=1)
    names_present = {row[4] for row in b["clips"]}
    # broll_1 + its 1-hop V1 neighbors (the gap before, broll_2 after). NOT intro/outro/V2/A1.
    srcs = {row[4] for row in b["clips"]}
    assert "broll_1.mov" in srcs
    assert "broll_2.mov" in srcs
    assert "GAP" in srcs                       # the gap before broll_1 is the other neighbor
    assert "intro.mov" not in srcs             # 5 hops away -> excluded
    assert "outro.mov" not in srcs             # 2 hops away -> excluded
    assert "lower_third.mov" not in names_present  # different track, no vertical -> excluded


def test_all_times_are_integers(cut):
    b = fgx.bundle(cut, [_sel(cut, "dialogue_a")], hops=2)
    for row in b["clips"]:
        assert isinstance(row[2], int)  # start
        assert isinstance(row[3], int)  # dur
        assert isinstance(row[5], int)  # source-in


def test_gaps_and_transitions_are_marked_not_clip_ids(cut):
    b = fgx.bundle(cut, [_sel(cut, "dialogue_b")], hops=2)
    marker_rows = [r for r in b["clips"] if r[4] in ("GAP", "XFADE")]
    assert marker_rows, "expected a gap/transition marker in dialogue_b's neighborhood"
    for r in marker_rows:
        assert r[0] == "-"  # markers carry no targetable clip id


def test_selection_header_is_tiny(cut):
    hdr = fgx.selection_header(cut, [_sel(cut, "intro")])
    assert set(hdr) == {"seq", "r", "sel"}
    assert fgx.estimate_tokens(hdr) < 40


def test_fgx_is_far_smaller_than_a_raw_timeline_dump(cut):
    """The core token win: send a selected subgraph, not the editor's full serialization.

    Baseline = OTIO's own JSON dump of the whole timeline (what 'just paste the project' would
    cost). FCPXML/AAF are comparably verbose. FGX for a small selection must be < 1/8 of it.
    """
    raw = otio.adapters.write_to_string(cut.timeline, "otio_json")
    raw_tokens = fgx.estimate_tokens(raw)
    b = fgx.bundle(cut, [_sel(cut, "broll_1")], hops=1)
    fgx_tokens = fgx.estimate_tokens(b)
    assert fgx_tokens * 8 < raw_tokens, f"fgx={fgx_tokens} raw={raw_tokens}"


def test_include_vertical_adds_overlapping_other_track_clips(cut):
    # logo_bug (V2, 200..260) temporally overlaps broll_2 (V1, 240..300) and stinger? no.
    sel = _sel(cut, "broll_2")  # V1 240..300
    flat = fgx.bundle(cut, [sel], hops=0, include_vertical=False)
    vert = fgx.bundle(cut, [sel], hops=0, include_vertical=True)
    assert len(vert["clips"]) > len(flat["clips"])
    vsrcs = {r[4] for r in vert["clips"]}
    assert "logo_bug.mov" in vsrcs  # V2 clip overlapping in time


def test_delta_emits_only_changed_rows(cut, fixtures_dir):
    sel_name = "broll_1"
    sel = _sel(cut, sel_name)
    first = fgx.bundle(cut, [sel], hops=1)
    prev_rows = {r[0]: r for r in first["clips"] if r[0] != "-"}

    # Shift the timeline by extending the V1 gap; rebuild IR.
    tl2 = otio.adapters.read_from_file(str(fixtures_dir / "cut.otio"))
    for ch in tl2.tracks[0]:
        if isinstance(ch, otio.schema.Gap):
            sr = ch.source_range
            ch.source_range = otio.opentime.TimeRange(
                sr.start_time, sr.duration + otio.opentime.RationalTime(10, 24))
            break
    new = TimelineIR.from_otio(tl2)
    new_sel = next(c for c in new.real_clips() if c.name == sel_name).id

    d = fgx.delta(new, [new_sel], prev_rows, hops=1)
    assert "changed" in d
    # broll_1 moved 180->190 so it (and broll_2) changed; nothing unchanged is resent.
    changed_starts = {r[0]: r[2] for r in d["changed"] if r[0] != "-"}
    assert any(start == 190 for start in changed_starts.values())
