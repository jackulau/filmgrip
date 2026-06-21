#!/usr/bin/env python3
"""film-grip performance benchmark — measures the hot paths the agent hits every turn.

The hot paths, in the order a turn exercises them:

1. **IR build** (`TimelineIR.from_otio`) — OTIO graph → flattened, ID-minted clip list. Runs on
   every snapshot of the timeline.
2. **FGX projection** (`serialize.fgx.bundle` + `to_text`) — the token-frugal view pulled into the
   model's context. Runs on every `get_context`.
3. **Color scope synthesis** (`perception.scopes.analyze_rgb`) — pure-numpy pixel statistics, the
   per-frame cost of the grading read.
4. **Transcript packing** (`perception.transcribe.pack_transcript`) — word JSON → compact phrase
   string the model greps.

Run it::

    python scripts/bench.py              # table of median ms per op × size
    python scripts/bench.py --json out.json
    python scripts/bench.py --profile    # cProfile the largest IR build + bundle (find hot spots)

Deterministic and dependency-light: timelines + frames + transcripts are synthesized in-memory, no
media or ffmpeg required. numpy is needed only for the scope rows (skipped with a note if absent).
"""
from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import statistics
import sys
import time
from typing import Callable

import opentimelineio as otio

from filmgrip.core.ir import TimelineIR
from filmgrip.serialize import fgx

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is an optional extra
    np = None

RATE = 24.0
CLIP_FRAMES = 48


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, RATE)


def make_timeline(n_clips: int, *, tracks: int = 1) -> otio.schema.Timeline:
    """A synthetic timeline: ``tracks`` video tracks each carrying ``n_clips`` back-to-back clips."""
    tl = otio.schema.Timeline(name="bench")
    tl.global_start_time = _rt(0)
    for t in range(tracks):
        v = otio.schema.Track(name=f"V{t + 1}", kind=otio.schema.TrackKind.Video)
        for i in range(n_clips):
            v.append(otio.schema.Clip(
                name=f"c{t}_{i}",
                media_reference=otio.schema.ExternalReference(target_url=f"/media/{t}_{i}.mov"),
                source_range=otio.opentime.TimeRange(_rt(0), _rt(CLIP_FRAMES)),
            ))
        tl.tracks.append(v)
    return tl


def make_frame(h: int, w: int):
    """A deterministic H×W×3 uint8 gradient frame (no randomness, so timings are reproducible)."""
    ramp_x = (np.linspace(0, 255, w, dtype=np.float64))[None, :]
    ramp_y = (np.linspace(0, 255, h, dtype=np.float64))[:, None]
    r = np.broadcast_to(ramp_x, (h, w))
    g = np.broadcast_to(ramp_y, (h, w))
    b = ((ramp_x + ramp_y) / 2.0)
    return np.stack([r, np.broadcast_to(g, (h, w)), np.broadcast_to(b, (h, w))], axis=2).astype(np.uint8)


def make_transcript(n_words: int):
    from filmgrip.perception.transcribe import Transcript, Word
    words = [Word(text=f"word{i}", start=i * 0.4, end=i * 0.4 + 0.35) for i in range(n_words)]
    return Transcript(media_path="/media/clip.mov", backend="fake", words=words,
                      duration_s=n_words * 0.4)


def timeit(fn: Callable[[], object], *, repeats: int = 7, warmup: int = 1) -> float:
    """Median wall-clock milliseconds over ``repeats`` runs (after ``warmup`` discarded runs)."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def bench() -> list[dict]:
    rows: list[dict] = []

    # 1. IR build — across timeline sizes.
    for n in (10, 100, 500, 2000):
        tl = make_timeline(n)
        ms = timeit(lambda: TimelineIR.from_otio(tl))
        rows.append({"op": "ir_build", "size": f"{n} clips", "ms": ms})

    # 2. FGX projection — bundle a 5-clip selection (hops=1) + serialize, across sizes.
    for n in (10, 100, 500, 2000):
        ir = TimelineIR.from_otio(make_timeline(n))
        sel = [c.id for c in ir.real_clips()[: min(5, n)]]
        rows.append({"op": "fgx_bundle(hops=1,sel=5)", "size": f"{n} clips",
                     "ms": timeit(lambda: fgx.bundle(ir, sel, hops=1))})
        rows.append({"op": "fgx_bundle+to_text", "size": f"{n} clips",
                     "ms": timeit(lambda: fgx.to_text(fgx.bundle(ir, sel, hops=1)))})

    # 2b. include_vertical on a 3-track timeline — exercises the temporal-overlap O(n) pass.
    for n in (100, 2000):
        ir = TimelineIR.from_otio(make_timeline(n, tracks=3))
        sel = [c.id for c in ir.real_clips()[:5]]
        rows.append({"op": "fgx_bundle(include_vertical,3trk)", "size": f"{n}×3 clips",
                     "ms": timeit(lambda: fgx.bundle(ir, sel, hops=1, include_vertical=True))})

    # 3. Color scope synthesis (numpy).
    if np is not None:
        for (h, w) in ((120, 160), (240, 320), (480, 640)):
            frame = make_frame(h, w)
            from filmgrip.perception.scopes import analyze_rgb
            rows.append({"op": "analyze_rgb", "size": f"{w}x{h}",
                         "ms": timeit(lambda: analyze_rgb(frame))})
    else:
        rows.append({"op": "analyze_rgb", "size": "—", "ms": float("nan"), "note": "numpy absent"})

    # 4. Transcript packing.
    from filmgrip.perception.transcribe import pack_transcript
    for n in (100, 1000, 5000):
        tr = make_transcript(n)
        rows.append({"op": "pack_transcript", "size": f"{n} words",
                     "ms": timeit(lambda: pack_transcript(tr))})

    return rows


def print_table(rows: list[dict]) -> None:
    w_op = max(len(r["op"]) for r in rows)
    w_sz = max(len(str(r["size"])) for r in rows)
    print(f"{'op'.ljust(w_op)}  {'size'.ljust(w_sz)}  {'median ms':>10}  note")
    print(f"{'-' * w_op}  {'-' * w_sz}  {'-' * 10}  ----")
    for r in rows:
        ms = r["ms"]
        ms_s = "   nan" if ms != ms else f"{ms:10.4f}"
        print(f"{r['op'].ljust(w_op)}  {str(r['size']).ljust(w_sz)}  {ms_s}  {r.get('note', '')}")


def profile() -> None:
    """cProfile the largest IR build + a bundle — surfaces the real hot spots, not guesses."""
    tl = make_timeline(2000, tracks=3)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(20):
        ir = TimelineIR.from_otio(tl)
        sel = [c.id for c in ir.real_clips()[:5]]
        fgx.to_text(fgx.bundle(ir, sel, hops=1, include_vertical=True))
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(25)
    print(s.getvalue())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", metavar="PATH", help="write machine-readable results to PATH")
    ap.add_argument("--profile", action="store_true", help="cProfile the largest build+bundle and exit")
    args = ap.parse_args(argv)

    if args.profile:
        profile()
        return 0

    rows = bench()
    print_table(rows)
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"rate": RATE, "clip_frames": CLIP_FRAMES, "rows": rows}, f, indent=2)
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
