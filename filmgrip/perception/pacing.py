"""Pacing & rhythm — the cheapest cinematography read, computed from the IR graph alone.

Every other perception module decodes media (ASR audio, ffmpeg frames). This one reads only the
*timeline structure* — clip starts and durations that the IR already holds — so it has NO IO
layer and ALWAYS runs: pure stdlib, no ffmpeg, no numpy, no media on disk. It is to *editing
rhythm* what :mod:`~filmgrip.perception.scopes` is to color: it reduces the cut structure to the
numbers an editor reads off a timeline (average shot length, cut cadence, the shot-length
distribution) so the planner can reason about — and verify against — the pace of an edit.

Like the scopes, this is an **advisory editorial read, a suggestion surface — not an applied
edit.** It mutates nothing and returns no ops. The rhythm verdict ('fast'/'medium'/'slow') is a
coarse heuristic on average shot length; pacing is contextual (a tense dialogue scene and a
montage live at opposite ends of "good"), so the thresholds below are deliberately advisory
rules of thumb, not absolutes — the editor (human or model) owns the final call.

``pacing_metrics`` is a pure function (IR in → JSON-able dict out), so its correctness is
provable offline on synthetic timelines with known shot counts.
"""
from __future__ import annotations

import statistics

from ..core.ir import Clip, TimelineIR

# Rhythm thresholds on AVERAGE SHOT LENGTH in seconds. Advisory editorial rules of thumb, not
# absolutes — fast-cut action/montage runs well under 2s, conventional dialogue/narrative coverage
# sits in the ~2-5s band, and contemplative/long-take styles run past 5s. Boundaries are inclusive
# at the low end of each band (see :func:`rhythm_verdict`).
FAST_MAX_S = 2.0   # ASL < 2s   → "fast"
SLOW_MIN_S = 5.0   # ASL > 5s   → "slow"; the [2, 5] band is "medium"


def rhythm_verdict(asl_seconds: float) -> str:
    """A coarse, ADVISORY pace label from average shot length in seconds.

    ``< FAST_MAX_S`` → 'fast', ``> SLOW_MIN_S`` → 'slow', otherwise 'medium'. An empty track
    (no shots, ``asl_seconds == 0``) reads 'none'. These are editorial heuristics, not a quality
    judgement — context decides whether a given pace is right.
    """
    if asl_seconds <= 0:
        return "none"
    if asl_seconds < FAST_MAX_S:
        return "fast"
    if asl_seconds > SLOW_MIN_S:
        return "slow"
    return "medium"


def _distribution(durations: list[int]) -> dict:
    """Shot-length distribution {min, median, p90, max} in frames from a list of clip durations.

    Uses stdlib :mod:`statistics`. ``p90`` is the 90th percentile via the inclusive method (it lies
    within the observed range and is defined for n==2). With a single shot every stat collapses to
    that shot's length; an empty list yields zeros.
    """
    if not durations:
        return {"min": 0, "median": 0.0, "p90": 0.0, "max": 0}
    if len(durations) == 1:
        only = durations[0]
        return {"min": only, "median": float(only), "p90": float(only), "max": only}
    # quantiles(n=10) returns the 9 deciles; the 9th (index 8) is the 90th percentile.
    p90 = statistics.quantiles(durations, n=10, method="inclusive")[8]
    return {
        "min": min(durations),
        "median": round(statistics.median(durations), 3),
        "p90": round(p90, 3),
        "max": max(durations),
    }


def _metrics(durations: list[int], rate: float) -> dict:
    """Core pacing numbers for one set of shots: count, ASL (frames + seconds), cut cadence,
    distribution, and the rhythm verdict. ``rate`` is fps (``ir.rate``).

    Cut cadence is *cuts per minute* = ``(count - 1) / total_minutes`` — a single shot has zero
    cuts (cadence 0), and an empty track has no shots (everything zero).
    """
    rate = float(rate) or 24.0
    count = len(durations)
    total_frames = sum(durations)
    total_seconds = total_frames / rate if total_frames else 0.0
    asl_frames = total_frames / count if count else 0.0
    asl_seconds = asl_frames / rate if count else 0.0
    cuts = max(count - 1, 0)
    cadence = (cuts / (total_seconds / 60.0)) if total_seconds > 0 else 0.0
    return {
        "shot_count": count,
        "asl_frames": round(asl_frames, 3),
        "asl_seconds": round(asl_seconds, 3),
        "cuts_per_minute": round(cadence, 3),
        "total_frames": total_frames,
        "total_seconds": round(total_seconds, 3),
        "distribution": _distribution(durations),
        "verdict": rhythm_verdict(asl_seconds),
    }


def _video_track_indices(clips: list[Clip]) -> list[int]:
    return sorted({c.track_index for c in clips if c.track_kind == "video"})


def pacing_metrics(ir: TimelineIR) -> dict:
    """The ``pacing`` payload: an ADVISORY rhythm read of a timeline, from its cut structure alone.

    For every video track AND for all video shots combined, reports shot count, average shot length
    (ASL) in frames *and* seconds, cut cadence (cuts per minute), the shot-length distribution
    ({min, median, p90, max} frames), and a coarse rhythm verdict ('fast'/'medium'/'slow', see
    :func:`rhythm_verdict`).

    Reads ``ir.real_clips()`` only — gaps and transitions are excluded, since pace is a property of
    the shots themselves. Edge cases degrade to sensible zeros, never a crash: an empty timeline and
    an audio-only timeline both report ``shot_count: 0`` with no tracks; a single shot reports
    ``cuts_per_minute: 0``.

    This is a suggestion surface — it mutates nothing and returns no edit ops. Pace is contextual,
    so the verdict is a rule of thumb, not a verdict on quality.
    """
    rate = float(ir.rate) or 24.0
    video = [c for c in ir.real_clips() if c.track_kind == "video"]

    tracks: list[dict] = []
    for ti in _video_track_indices(video):
        durations = [c.duration for c in video if c.track_index == ti]
        track = {"track_kind": "video", "track_index": ti}
        track.update(_metrics(durations, rate))
        tracks.append(track)

    overall = _metrics([c.duration for c in video], rate)
    return {
        "rate": round(rate, 3),
        "frames": "timeline",
        "advisory": True,           # editorial suggestion surface, not an applied edit
        "overall": overall,
        "tracks": tracks,
    }
