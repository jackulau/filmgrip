#!/usr/bin/env python
"""Benchmark film-grip's pure-compute core — to size a Rust spike on data, not faith.

A "rewrite the core in Rust" idea is only worth it if the CPU paths it would speed up are a
material fraction of a real edit's wall-clock. This is a THIN CLI over
:func:`filmgrip.bench.hot_paths`: it times the three pure-CPU hot paths every agent turn pays — IR
indexing, FGX serialization, EditPlan validation — across timeline sizes, and compares their total
against the I/O floor that actually dominates an edit: a single Claude planning turn (seconds of
network).

The timing + fixtures live in :mod:`filmgrip.bench` so the numbers here match what
``scripts/bench.py`` prints and what ``tests/test_bench_guard.py`` guards — no parallel
implementation to drift.

Prints a table + a ceiling estimate and exits 0. Run: ``python scripts/bench_core.py [N ...]``
(default sizes 500 / 2000 / 5000).
"""
from __future__ import annotations

import sys

from filmgrip.bench import hot_paths

# A single Claude planning turn is seconds of network round-trip; this is the I/O floor a compiled
# core can NEVER remove. Conservative (turns are often slower); the ceiling below only gets smaller
# with a bigger floor.
TYPICAL_LLM_TURN_MS = 4000.0


def bench(n: int) -> dict:
    """Median ms for the three hot paths at ``n`` clips, plus their CPU total."""
    hp = hot_paths(n)
    ir_ms, fgx_ms, val_ms = hp["ir_build"], hp["fgx_bundle_to_text"], hp["validate"]
    return {"n": n, "ir": ir_ms, "fgx": fgx_ms, "val": val_ms,
            "cpu": ir_ms + fgx_ms + val_ms}


def main(argv=None) -> int:
    sizes = [int(a) for a in (argv if argv is not None else sys.argv[1:])] or [500, 2000, 5000]
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
