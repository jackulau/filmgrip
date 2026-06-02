# Rust core spike — measured go/no-go

**Question.** film-grip's core is Python. Would rewriting the hot compute paths (IR indexing, FGX
serialization, EditPlan validation) in a compiled language — Rust via PyO3 — make the tool
meaningfully faster?

**Method.** `scripts/bench_core.py` builds synthetic timelines of 500 / 2000 / 5000 clips and times
each CPU path (median of 5 runs), then compares the total against the I/O floor that dominates a
real edit: one Claude planning turn (≈ 4000 ms of network round-trip — conservative; turns are often
slower, which only shrinks the ceiling). Reproduce with `python scripts/bench_core.py`.

## Measured numbers

| clips | IR index | FGX serialize | validate | CPU total |
|------:|---------:|--------------:|---------:|----------:|
|   500 |  2.86 ms |     0.176 ms  | 0.028 ms |  3.07 ms  |
|  2000 | 13.90 ms |     0.602 ms  | 0.086 ms | 14.59 ms  |
|  5000 | 45.06 ms |     1.461 ms  | 0.200 ms | 46.72 ms  |

*(Numbers from a dev run; rerun the script for your machine — the ratio, not the absolute, is the
point.)*

Observations:

- **The CPU paths are tiny.** Even at 5000 clips — a very large timeline — the whole compute core
  is ~47 ms, and **all but ~2 ms of that is IR indexing**, which already calls into
  OpenTimelineIO's **C++** core for the timing math. FGX serialization and validation are
  sub-2 ms and effectively free.
- **The pipeline is I/O-bound, exactly as predicted.** A single planning turn is ~4000 ms of
  network; the Resolve scripting API and OTIO file round-trips are the other big costs. CPU is
  noise next to them.

## The ceiling

An edit's wall-clock ≈ (CPU core) + (one LLM turn) ≈ 47 ms + 4000 ms ≈ **4047 ms**.

A **perfect, zero-cost** Rust core — one that made IR indexing, FGX, and validation literally
instantaneous — could remove **at most ~1.15%** of that wall-clock. A realistic Rust port (say 5×
faster on the CPU parts) would shave ~37 ms: **under 1%**, unnoticeable, in exchange for a PyO3 FFI
boundary, a dual toolchain, and porting ~100 tests. And it could not touch the flagship at all:
live Resolve scripting is Python/Lua-only, OTIO's adapters are Python, and the Agent SDK is Python.

## Verdict / recommendation

**NO-GO on a Rust rewrite of the core — now or as currently scoped.** The data says a compiled core
buys < 1% of user-facing time while adding real complexity and an FFI seam. The bottleneck is, and
will remain, LLM latency + the editor/file I/O.

**Where the actual speed wins are** (all Python, already in scope or cheap to add):

1. **Token & latency reduction** — FGX already cuts ~96% of context tokens vs a raw dump
   (measured live in this build: `204 tok vs raw 6050`). Tightening it, prompt-cache warmth, and
   streaming results shave far more wall-clock than any CPU rewrite.
2. **Snapshot caching** — re-snapshotting Resolve is I/O; cache between turns when the timeline is
   unchanged.
3. **Parallel / batched validation** when a turn produces independent sub-plans.

**Revisit only if** a profile on a *real* project ever shows the CPU core as a material fraction of
wall-clock — e.g. a >50,000-clip timeline indexed many times per second with no LLM in the loop.
That is not film-grip's workload. Until then, Python is the right call, and the spike paid for
itself by replacing a guess with a number.
