"""D3 — OTIO core IR + stable clip-ID map."""
from __future__ import annotations

import opentimelineio as otio
import pytest

from filmgrip.core.idmap import stable_clip_id, to_base36
from filmgrip.core.ir import TimelineIR


@pytest.fixture
def cut(fixtures_dir):
    return TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))


def test_base36_roundtrip():
    assert to_base36(0) == "0"
    assert to_base36(35) == "z"
    assert to_base36(36) == "10"


def test_stable_id_is_deterministic_and_pure():
    a = stable_clip_id("video", 1, 48, "dialogue_a.mov")
    b = stable_clip_id("video", 1, 48, "dialogue_a.mov")
    c = stable_clip_id("video", 1, 49, "dialogue_a.mov")
    assert a == b
    assert a != c
    assert a.startswith("c")


def test_ir_parses_fixture_shape(cut):
    assert cut.rate == 24.0
    assert len(cut.real_clips()) == 12
    assert cut.track_count("video") == 2
    assert cut.track_count("audio") == 2
    assert cut.duration == 360
    # The dissolve and gaps are represented but distinct from clips.
    kinds = {c.kind for c in cut.clips}
    assert {"clip", "gap", "transition"} <= kinds


def test_clip_positions(cut):
    by_name = {c.name: c for c in cut.real_clips()}
    assert (by_name["intro"].start, by_name["intro"].duration) == (0, 48)
    assert (by_name["dialogue_a"].start, by_name["dialogue_a"].duration) == (48, 72)
    assert by_name["dialogue_a"].source_start == 100  # source in-point preserved
    assert by_name["broll_1"].start == 180


def test_ids_are_deterministic_across_reloads(fixtures_dir):
    a = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    b = TimelineIR.from_otio_file(str(fixtures_dir / "cut.otio"))
    assert [c.id for c in a.clips] == [c.id for c in b.clips]
    assert all(a.clip(c.id) is not None for c in a.clips)


def test_neighbors_are_track_local(cut):
    intro = next(c for c in cut.real_clips() if c.name == "intro")
    names = {c.name for c in cut.neighbors(intro.id, hops=1)}
    assert names == {"dialogue_a"}  # only the adjacent V1 item, not V2/A1
    two = {c.name for c in cut.neighbors(intro.id, hops=2)}
    assert "dialogue_a" in two and "dissolve" in two


def test_reresolution_survives_a_ripple_shift(cut, fixtures_dir):
    target_old = next(c for c in cut.real_clips() if c.name == "broll_1")
    old_id = target_old.id
    assert target_old.start == 180

    # Simulate a ripple: extend the V1 gap by 10 frames so everything after shifts +10.
    tl2 = otio.adapters.read_from_file(str(fixtures_dir / "cut.otio"))
    v1 = tl2.tracks[0]
    for ch in v1:
        if isinstance(ch, otio.schema.Gap):
            sr = ch.source_range
            ch.source_range = otio.opentime.TimeRange(
                sr.start_time, sr.duration + otio.opentime.RationalTime(10, 24)
            )
            break
    new = TimelineIR.from_otio(tl2)

    target_new = next(c for c in new.real_clips() if c.name == "broll_1")
    assert target_new.start == 190                    # it really moved
    assert target_new.id != old_id                    # deterministic ID changed with position

    resolved = cut.reresolve(old_id, new)             # ...but the OLD id still finds it
    assert resolved is not None
    assert resolved.name == "broll_1"
    assert resolved.start == 190


def test_reresolution_returns_none_when_clip_is_gone(cut):
    empty = TimelineIR.from_otio(otio.schema.Timeline(name="empty"))
    some_id = cut.real_clips()[0].id
    assert cut.reresolve(some_id, empty) is None
