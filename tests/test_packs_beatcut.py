"""D15 — the deterministic ``beat-cut`` pack: beat grid → on-beat ``cut_range``, honest failures.

Zero LLM, zero network. The musical beat grid is injected (a stub :func:`beats_for_media` returning
known TIMELINE frames), exactly as ``test_speech.py`` injects a fake transcriber — so the pack's
compile logic (snap-to-beat head/tail cuts, descending-per-track order, REBUILD_OPS-only,
PackError honesty) is proven without ffmpeg/numpy/librosa or any real audio.

Scenario (24fps): one video clip on V1 at timeline frames [0, 120) whose detected beats are
[6, 30, 54, 78, 102] — first in-beat 6, last in-beat 102 — so the pack drops the off-beat lead-in
[0, 6) and the off-beat tail [102, 120), snapping both boundaries onto the grid.
"""
from __future__ import annotations

import opentimelineio as otio
import pytest

from filmgrip.adapters.interchange import REBUILD_OPS, OtioMutator
from filmgrip.core.ir import TimelineIR
from filmgrip.packs import PackError, all_packs, get_pack
from filmgrip.packs.engine import compile_pack
from filmgrip.perception.transcribe import PerceptionUnavailable
from filmgrip.protocol.validate import validate


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


@pytest.fixture
def media(tmp_path):
    p = tmp_path / "song.mov"
    p.write_bytes(b"\x00" * 32)
    return p


@pytest.fixture
def ir(media) -> TimelineIR:
    """One video clip [0,120) backed by an on-disk media file (so media_path_of resolves)."""
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(otio.schema.Clip(
        name="hero",
        media_reference=otio.schema.ExternalReference(target_url=media.as_uri()),
        source_range=otio.opentime.TimeRange(_rt(0), _rt(120)),
    ))
    return TimelineIR(tl)


def _stub_beats(monkeypatch, beats_by_src, *, downbeats_by_src=None):
    """Patch music.beats_for_media to return a known TIMELINE-frame grid keyed by media basename."""
    import os

    import filmgrip.perception.music as music

    def fake(media_path, clip=None, **kw):
        key = os.path.basename(media_path)
        return {
            "engine": "stub", "tier": "advisory", "rate": "24", "frames": "timeline",
            "tempo_bpm": 120.0,
            "beats": list(beats_by_src.get(key, [])),
            "downbeats": list((downbeats_by_src or {}).get(key, [])),
            "onsets": [], "source": key, "errors": [],
        }

    monkeypatch.setattr(music, "beats_for_media", fake)


# --------------------------------------------------------------------------- registration
def test_beat_cut_registered_and_named():
    names = {p.name for p in all_packs()}
    assert "beat-cut" in names
    pack = get_pack("beat-cut")
    assert pack.name == "beat-cut" and pack.kind == "deterministic"


def test_beat_cut_declares_requirements():
    assert get_pack("beat-cut").requires  # honest about needing ffmpeg/numpy + media on disk


# --------------------------------------------------------------------------- happy path
def test_beat_cut_snaps_head_and_tail_to_beats(ir, media, monkeypatch):
    _stub_beats(monkeypatch, {media.name: [6, 30, 54, 78, 102]})
    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("beat-cut"), ir, [cid])

    # head [0,6) drops the off-beat lead-in; tail [102,120) drops the off-beat tail.
    assert [(op.start_frame, op.end_frame) for op in plan.ops] == [(102, 120), (0, 6)]
    assert {op.op for op in plan.ops} == {"cut_range"}
    assert {op.op for op in plan.ops} <= REBUILD_OPS              # applicability honesty


def test_beat_cut_emits_only_rebuild_ops_and_validates(ir, media, monkeypatch):
    _stub_beats(monkeypatch, {media.name: [6, 30, 54, 78, 102]})
    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("beat-cut"), ir, [cid])
    res = validate(plan, ir)
    assert res.ok, [str(e) for e in res.errors]                  # in-bounds + descending contract
    # and it actually applies through the same pipeline as everything else.
    OtioMutator(ir).apply(plan)
    ir.reindex()
    # removed 6 (head) + 18 (tail) of 120 → 96 frames of content remain, snapped to the grid.
    assert sum(c.duration for c in ir.real_clips()) == 96


def test_beat_cut_descending_per_track(media, tmp_path, monkeypatch):
    """Two clips per track over two tracks: ops must be strictly descending within each track."""
    tl = otio.schema.Timeline(name="seq")
    srcs = {}
    for tk, kind, names in (("V1", otio.schema.TrackKind.Video, ("a", "b")),
                            ("V2", otio.schema.TrackKind.Video, ("c", "d"))):
        track = otio.schema.Track(name=tk, kind=kind)
        tl.tracks.append(track)
        for nm in names:
            f = tmp_path / f"{tk}_{nm}.mov"
            f.write_bytes(b"\x00" * 32)
            track.append(otio.schema.Clip(
                name=nm, media_reference=otio.schema.ExternalReference(target_url=f.as_uri()),
                source_range=otio.opentime.TimeRange(_rt(0), _rt(60))))
            srcs[f.name] = [6, 30, 54]                            # in-beats 6..54 inside [start,start+60)
    ir = TimelineIR.from_otio(tl)
    _stub_beats(monkeypatch, srcs)

    plan = compile_pack(get_pack("beat-cut"), ir, [c.id for c in ir.real_clips()])
    res = validate(plan, ir)
    assert res.ok, [str(e) for e in res.errors]                  # validator enforces per-track order

    by_track: dict[str, list[int]] = {}
    for op in plan.ops:
        clip = ir.clip(op.clip_id)
        by_track.setdefault(f"{clip.track_kind}{clip.track_index}", []).append(op.start_frame)
    for code, starts in by_track.items():
        assert starts == sorted(starts, reverse=True), f"track {code} not descending: {starts}"


def test_beat_cut_downbeat_unit_uses_detected_downbeats(ir, media, monkeypatch):
    _stub_beats(monkeypatch, {media.name: [6, 30, 54, 78, 102]},
                downbeats_by_src={media.name: [12, 84]})
    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("beat-cut"), ir, [cid], params={"unit": "downbeat"})
    # snap to the downbeat grid: head [0,12), tail [84,120).
    assert [(op.start_frame, op.end_frame) for op in plan.ops] == [(84, 120), (0, 12)]


def test_beat_cut_no_beats_inside_clip_emits_nothing(ir, media, monkeypatch):
    # an out-of-range grid (or clip edges already on the only beats) yields no honest cut, not a lie.
    _stub_beats(monkeypatch, {media.name: [0, 120, 500]})
    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("beat-cut"), ir, [cid])
    assert plan.ops == []


# --------------------------------------------------------------------------- honesty: PackError
def test_beat_cut_raises_when_deps_missing(ir, media, monkeypatch):
    """ffmpeg/numpy absent → beats_for_media raises PerceptionUnavailable → pack raises PackError
    (proves it never turns a missing engine into an empty 'success' plan)."""
    import filmgrip.perception.music as music

    def boom(media_path, clip=None, **kw):
        raise PerceptionUnavailable("ffmpeg is required to decode audio")

    monkeypatch.setattr(music, "beats_for_media", boom)
    with pytest.raises(PackError, match="cannot read beats"):
        compile_pack(get_pack("beat-cut"), ir, [ir.real_clips()[0].id])


def test_beat_cut_raises_on_offline_media(ir, media, monkeypatch):
    """An ``errors`` entry from the reader (offline/undecodable media) → PackError, never a silent
    under-edit."""
    import filmgrip.perception.music as music

    def offline(media_path, clip=None, **kw):
        return {"engine": "numpy", "tier": "advisory", "rate": "24", "frames": "timeline",
                "tempo_bpm": 0.0, "beats": [], "downbeats": [], "onsets": [],
                "errors": [f"{getattr(clip, 'id', '?')}: source media not found — offline"]}

    monkeypatch.setattr(music, "beats_for_media", offline)
    with pytest.raises(PackError, match="could not analyze every selected clip"):
        compile_pack(get_pack("beat-cut"), ir, [ir.real_clips()[0].id])


def test_beat_cut_raises_on_retimed_clip(ir, media):
    """A retimed clip makes beat→frame mapping a lie; the real beats_for_media returns it as an
    ``errors`` entry, so the pack raises PackError (no monkeypatch — exercises the real reader's
    retime guard)."""
    clip = ir.real_clips()[0]
    clip.otio.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    with pytest.raises(PackError, match="could not analyze every selected clip"):
        compile_pack(get_pack("beat-cut"), ir, [clip.id])


def test_beat_cut_raises_on_unknown_media_path():
    """A clip with no resolvable media path → PackError (offline reference, no file URL)."""
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(otio.schema.Clip(  # no media_reference target_url → media_path_of returns None
        name="orphan", source_range=otio.opentime.TimeRange(_rt(0), _rt(120))))
    ir = TimelineIR.from_otio(tl)
    with pytest.raises(PackError, match="source media path unknown"):
        compile_pack(get_pack("beat-cut"), ir, [ir.real_clips()[0].id])
