#!/usr/bin/env python
"""Benchmark film-grip's pure-compute core — to size a Rust spike on data, not faith.

A "rewrite the core in Rust" idea is only worth it if the CPU paths it would speed up are a
material fraction of a real edit's wall-clock. This measures those paths — IR indexing, FGX
serialization, EditPlan validation — across timeline sizes, and compares their total against the
I/O floor that actually dominates an edit: a single Claude planning turn (seconds of network).

Prints a table + a ceiling estimate and exits 0. Run: ``python scripts/bench_core.py [N ...]``.
"""
from __future__ import annotations

import statistics
import sys
import time

import opentimelineio as otio

from filmgrip.core.ir import TimelineIR
from filmgrip.protocol.editplan import EditPlan
from filmgrip.protocol.validate import validate
from filmgrip.serialize import fgx

# A single Claude planning turn is seconds of network round-trip; this is the I/O floor a compiled
# core can NEVER remove. Conservative (turns are often slower); the ceiling below only gets smaller
# with a bigger floor.
TYPICAL_LLM_TURN_MS = 4000.0


def build_timeline(n_clips: int, tracks: int = 2) -> otio.schema.Timeline:
    rate = 24.0

    def rt(f: int) -> otio.opentime.RationalTime:
        return otio.opentime.RationalTime(f, rate)

    tl = otio.schema.Timeline(name=f"bench-{n_clips}")
    tl.global_start_time = rt(0)
    per = max(1, n_clips // tracks)
    for t in range(tracks):
        trk = otio.schema.Track(name=f"V{t + 1}", kind=otio.schema.TrackKind.Video)
        for i in range(per):
            trk.append(otio.schema.Clip(
                name=f"c{t}_{i}",
                media_reference=otio.schema.ExternalReference(target_url=f"/m/c{t}_{i}.mov"),
                source_range=otio.opentime.TimeRange(rt(0), rt(48))))
        tl.tracks.append(trk)
    return tl


def _median_ms(fn, iters: int = 5) -> float:
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def bench(n: int) -> dict:
    tl = build_timeline(n)
    ir_ms = _median_ms(lambda: TimelineIR(tl))
    ir = TimelineIR(tl)
    reals = ir.real_clips()
    sel = [c.id for c in reals[:10]]
    fgx_ms = _median_ms(lambda: fgx.to_text(fgx.bundle(ir, sel, hops=1)))
    plan = EditPlan.parse({"ops": [{"op": "add_marker", "clip_id": c.id, "frame": 0}
                                   for c in reals[:50]]})
    val_ms = _median_ms(lambda: validate(plan, ir))
    return {"n": len(reals), "ir": ir_ms, "fgx": fgx_ms, "val": val_ms,
            "cpu": ir_ms + fgx_ms + val_ms}


def main(argv=None) -> int:
    sizes = [int(a) for a in (argv or sys.argv[1:])] or [500, 2000, 5000]
    print("film-grip core benchmark — CPU paths a Rust core could speed up")
    print(f"(I/O floor: one Claude planning turn ≈ {TYPICAL_LLM_TURN_MS:.0f} ms of network)\n")
    print(f"{'clips':>7} | {'IR index':>10} | {'FGX ser':>10} | {'validate':>10} | {'CPU total':>10}")
    print("-" * 60)
    rows = []
    for n in sizes:
        r = bench(n)
        rows.append(r)
        print(f"{r['n']:>7} | {r['ir']:>8.2f}ms | {r['fgx']:>8.3f}ms | {r['val']:>8.3f}ms | "
              f"{r['cpu']:>8.2f}ms")

    worst = max(rows, key=lambda r: r["cpu"])
    ceiling = 100.0 * worst["cpu"] / (worst["cpu"] + TYPICAL_LLM_TURN_MS)
    print("\nCeiling estimate (largest timeline):")
    print(f"  CPU total at {worst['n']} clips: {worst['cpu']:.2f} ms")
    print(f"  An edit's wall-clock ≈ CPU + one LLM turn ≈ {worst['cpu'] + TYPICAL_LLM_TURN_MS:.0f} ms")
    print(f"  A perfect (zero-cost) Rust core could remove AT MOST {ceiling:.2f}% of that wall-clock.")
    print("\nSee docs/RUST_SPIKE.md for the verdict.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
