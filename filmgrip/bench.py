"""Importable, dependency-light benchmark core for film-grip's pure-CPU hot paths.

Every agent turn pays a fixed CPU cost before any network round-trip: it rebuilds the IR from the
OTIO graph (:meth:`TimelineIR.from_otio`), projects the token-frugal FGX view of the selection
(:func:`serialize.fgx.bundle` + :func:`~serialize.fgx.to_text`), and validates the EditPlan it
emits (:func:`protocol.validate.validate`). This module is the *single source of truth* for timing
those paths: both the human harness (``scripts/bench.py``), the Rust-spike ceiling tool
(``scripts/bench_core.py``), and the regression tripwire (``tests/test_bench_guard.py``) import it,
so the numbers a test guards are the numbers the scripts print — no parallel implementations to
drift.

Everything here is deterministic and media-free: timelines (and the optional frames/transcripts used
by ``scripts/bench.py``) are synthesized in-memory, so timings are reproducible and no ffmpeg or real
media is touched. numpy is required only by :func:`make_frame`; the hot paths in :func:`hot_paths`
are pure-Python and never import it.
"""
from __future__ import annotations

import statistics
import time
from typing import Callable

import opentimelineio as otio

from .core.ir import TimelineIR
from .protocol.editplan import EditPlan
from .protocol.validate import validate
from .serialize import fgx

try:  # numpy is an optional extra; only make_frame needs it.
    import numpy as np
except Exception:  # pragma: no cover - exercised only where numpy is absent
    np = None

# Fixture knobs, shared so every caller times the same synthetic shape.
RATE = 24.0
CLIP_FRAMES = 48


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, RATE)


def median_ms(fn: Callable[[], object], *, repeats: int = 7, warmup: int = 1) -> float:
    """Median wall-clock milliseconds of ``fn`` over ``repeats`` runs (after ``warmup`` discarded).

    Median (not mean) so a single GC pause or scheduler hiccup doesn't dominate; the warmup runs
    pay one-time import/JIT/allocation costs that would otherwise skew the first sample.
    """
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def build_timeline(n: int, *, tracks: int = 1) -> otio.schema.Timeline:
    """A synthetic timeline: ``tracks`` video tracks, each carrying ``n`` back-to-back clips.

    ``n`` is the per-track clip count, so the total item count is ``n * tracks``. Each clip points at
    a distinct fake media URL (so ``media_ref_string`` does real work) but no file is touched.
    """
    tl = otio.schema.Timeline(name="bench")
    tl.global_start_time = _rt(0)
    for t in range(tracks):
        v = otio.schema.Track(name=f"V{t + 1}", kind=otio.schema.TrackKind.Video)
        for i in range(n):
            v.append(otio.schema.Clip(
                name=f"c{t}_{i}",
                media_reference=otio.schema.ExternalReference(target_url=f"/media/{t}_{i}.mov"),
                source_range=otio.opentime.TimeRange(_rt(0), _rt(CLIP_FRAMES)),
            ))
        tl.tracks.append(v)
    return tl


def make_frame(h: int, w: int):
    """A deterministic H×W×3 uint8 gradient frame (no randomness, so timings are reproducible).

    Requires numpy. Used by ``scripts/bench.py`` for the ``analyze_rgb`` rows; not on the
    :func:`hot_paths` path.
    """
    if np is None:  # pragma: no cover - guarded by callers that check np first
        raise RuntimeError("make_frame requires numpy")
    ramp_x = (np.linspace(0, 255, w, dtype=np.float64))[None, :]
    ramp_y = (np.linspace(0, 255, h, dtype=np.float64))[:, None]
    r = np.broadcast_to(ramp_x, (h, w))
    g = np.broadcast_to(ramp_y, (h, w))
    b = ((ramp_x + ramp_y) / 2.0)
    return np.stack([r, np.broadcast_to(g, (h, w)), np.broadcast_to(b, (h, w))], axis=2).astype(np.uint8)


def make_transcript(n_words: int):
    """A synthetic transcript of ``n_words`` evenly-spaced words (for ``pack_transcript`` timing)."""
    from .perception.transcribe import Transcript, Word
    words = [Word(text=f"word{i}", start=i * 0.4, end=i * 0.4 + 0.35) for i in range(n_words)]
    return Transcript(media_path="/media/clip.mov", backend="fake", words=words,
                      duration_s=n_words * 0.4)


# --------------------------------------------------------------------------- hot paths
def _hot_plan(ir: TimelineIR, count: int = 50) -> EditPlan:
    """A trivial but fully valid EditPlan that exercises validate's per-op clip-resolution loop.

    ``add_marker`` is the cheapest op that targets an existing clip, so the plan validates clean
    (every clip id is real) while still walking the op loop ``count`` times — what validate actually
    costs on a real turn.
    """
    reals = ir.real_clips()
    targets = reals[: min(count, len(reals))]
    return EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": c.id, "frame": 0}
                                   for c in targets]})


def hot_paths(n_clips: int = 2000, *, repeats: int = 7, warmup: int = 1) -> dict[str, float]:
    """Median ms for each pure-CPU hot path at ``n_clips`` clips.

    Returns ``{"ir_build", "fgx_bundle_to_text", "validate"}`` — the three paths every agent turn
    pays before the network. All pure-Python; no ffmpeg, media, or numpy. The timeline is built once
    (its construction is not what we're measuring) and reused across the three measurements.
    """
    tl = build_timeline(n_clips)
    out: dict[str, float] = {}

    out["ir_build"] = median_ms(lambda: TimelineIR.from_otio(tl), repeats=repeats, warmup=warmup)

    ir = TimelineIR.from_otio(tl)
    sel = [c.id for c in ir.real_clips()[: min(5, n_clips)]]
    out["fgx_bundle_to_text"] = median_ms(
        lambda: fgx.to_text(fgx.bundle(ir, sel, hops=1)), repeats=repeats, warmup=warmup)

    plan = _hot_plan(ir)
    out["validate"] = median_ms(lambda: validate(plan, ir), repeats=repeats, warmup=warmup)

    return out
