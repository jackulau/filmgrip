"""D4 — cinematography MVP: pacing/rhythm read from the IR alone. pacing_metrics is a pure
function (timeline structure in → numbers out), so correctness is proved on synthetic timelines
with known shot counts — no editor, no ffmpeg, no numpy. One case uses the cut.otio fixture for a
realistic timeline."""
from __future__ import annotations

import opentimelineio as otio

from filmgrip.core.ir import TimelineIR
from filmgrip.perception.pacing import (
    FAST_MAX_S,
    SLOW_MIN_S,
    pacing_metrics,
    rhythm_verdict,
)

FIX = "tests/fixtures/cut.otio"


def _ir(track_specs: dict[str, list[int]], rate: float = 24.0) -> TimelineIR:
    """Build a TimelineIR from {track_name: [clip_frame_durations]} (video tracks, contiguous)."""
    def rt(f):
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name="pacedemo")
    tl.global_start_time = rt(0)
    for tname, durs in track_specs.items():
        kind = otio.schema.TrackKind.Audio if tname.startswith("A") else otio.schema.TrackKind.Video
        track = otio.schema.Track(name=tname, kind=kind)
        for i, d in enumerate(durs):
            track.append(otio.schema.Clip(
                name=f"{tname}_{i}",
                media_reference=otio.schema.ExternalReference(target_url=f"/m/{tname}_{i}.mov"),
                source_range=otio.opentime.TimeRange(rt(0), rt(d))))
        tl.tracks.append(track)
    return TimelineIR.from_otio(tl)


# ---------------------------------------------------------------- equal-length shots: ASL & cadence
def test_equal_length_shots_asl_cadence_and_verdict():
    # 5 shots of 48f @ 24fps → each shot is exactly 2.0s. ASL == clip length; the 2.0s boundary is
    # 'medium' (verdict is < FAST_MAX_S for 'fast', so == 2.0 is NOT fast).
    ir = _ir({"V1": [48, 48, 48, 48, 48]})
    m = pacing_metrics(ir)["overall"]
    assert m["shot_count"] == 5
    assert m["asl_frames"] == 48.0
    assert m["asl_seconds"] == 2.0
    # 4 cuts over 240f = 10.0s = 1/6 min → 24 cuts/min.
    assert m["cuts_per_minute"] == 24.0
    assert m["verdict"] == "medium"
    # equal lengths → every distribution stat is the shot length.
    assert m["distribution"] == {"min": 48, "median": 48.0, "p90": 48.0, "max": 48}


def test_fast_and_slow_verdicts_from_asl():
    # short shots (24f = 1.0s) → fast; long shots (240f = 10.0s) → slow.
    assert pacing_metrics(_ir({"V1": [24, 24, 24]}))["overall"]["verdict"] == "fast"
    assert pacing_metrics(_ir({"V1": [240, 240]}))["overall"]["verdict"] == "slow"


def test_rhythm_verdict_thresholds_are_documented_bands():
    assert rhythm_verdict(FAST_MAX_S - 0.01) == "fast"
    assert rhythm_verdict(FAST_MAX_S) == "medium"        # 2.0s boundary → medium (inclusive low)
    assert rhythm_verdict(SLOW_MIN_S) == "medium"        # 5.0s boundary → medium (inclusive high)
    assert rhythm_verdict(SLOW_MIN_S + 0.01) == "slow"
    assert rhythm_verdict(0.0) == "none"


# ---------------------------------------------------------------- mixed lengths: distribution stats
def test_mixed_length_distribution_stats():
    # durations sorted: [10, 20, 30, 40, 100] → min 10, max 100, median 30.
    # p90 (inclusive deciles, 9th point) of a 5-sample set is the 4th gap interpolated: 76.0.
    ir = _ir({"V1": [40, 10, 100, 20, 30]})
    dist = pacing_metrics(ir)["overall"]["distribution"]
    assert dist["min"] == 10
    assert dist["max"] == 100
    assert dist["median"] == 30.0
    assert dist["p90"] == 76.0
    # sanity: min <= median <= p90 <= max.
    assert dist["min"] <= dist["median"] <= dist["p90"] <= dist["max"]


# ---------------------------------------------------------------- edge cases
def test_empty_timeline_is_zeros_not_a_crash():
    tl = otio.schema.Timeline(name="empty")
    tl.global_start_time = otio.opentime.RationalTime(0, 24.0)
    m = pacing_metrics(TimelineIR.from_otio(tl))
    assert m["tracks"] == []
    o = m["overall"]
    assert o["shot_count"] == 0
    assert o["asl_frames"] == 0.0 and o["asl_seconds"] == 0.0
    assert o["cuts_per_minute"] == 0.0
    assert o["verdict"] == "none"
    assert o["distribution"] == {"min": 0, "median": 0.0, "p90": 0.0, "max": 0}


def test_single_clip_has_zero_cadence():
    m = pacing_metrics(_ir({"V1": [72]}))["overall"]
    assert m["shot_count"] == 1
    assert m["cuts_per_minute"] == 0.0           # one shot → no cuts
    assert m["asl_frames"] == 72.0
    assert m["distribution"] == {"min": 72, "median": 72.0, "p90": 72.0, "max": 72}


def test_audio_only_timeline_reports_no_video_shots():
    m = pacing_metrics(_ir({"A1": [100, 100]}))
    assert m["tracks"] == []                      # no video tracks
    assert m["overall"]["shot_count"] == 0
    assert m["overall"]["verdict"] == "none"


def test_per_track_metrics_independent_of_overall():
    # two video tracks with different rhythms; overall pools all video shots.
    ir = _ir({"V1": [24, 24, 24, 24], "V2": [240]})
    m = pacing_metrics(ir)
    tracks = {t["track_index"]: t for t in m["tracks"]}
    assert set(tracks) == {1, 2}
    assert tracks[1]["shot_count"] == 4 and tracks[1]["verdict"] == "fast"
    assert tracks[2]["shot_count"] == 1 and tracks[2]["cuts_per_minute"] == 0.0
    # overall = 5 shots across both tracks.
    assert m["overall"]["shot_count"] == 5


# ---------------------------------------------------------------- realistic fixture
def test_cut_fixture_pacing_is_realistic():
    ir = TimelineIR.from_otio_file(FIX)
    m = pacing_metrics(ir)
    by_track = {t["track_index"]: t for t in m["tracks"]}
    # cut.otio: V1 has 6 shots (durations 48,72,36,60,60,60 = 336f), V2 has 2 shots. Audio excluded.
    assert set(by_track) == {1, 2}
    v1 = by_track[1]
    assert v1["shot_count"] == 6
    assert v1["asl_frames"] == 56.0               # 336 / 6
    assert v1["asl_seconds"] == round(56.0 / 24.0, 3)
    assert v1["verdict"] == "medium"              # ~2.33s
    assert v1["distribution"]["min"] == 36 and v1["distribution"]["max"] == 72
    assert v1["distribution"]["median"] == 60.0
    # 5 cuts over 336f = 14.0s → 5 / (14/60) ≈ 21.429 cuts/min.
    assert v1["cuts_per_minute"] == round(5 / (336 / 24.0 / 60.0), 3)
    # overall pools V1 (6) + V2 (2) = 8 video shots; audio tracks are not counted.
    assert m["overall"]["shot_count"] == 8


def test_payload_is_json_serializable():
    import json
    m = pacing_metrics(TimelineIR.from_otio_file(FIX))
    again = json.loads(json.dumps(m))
    assert again["advisory"] is True
    assert again["overall"]["shot_count"] == 8
