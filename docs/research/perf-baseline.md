# Performance baseline (goal 008, workstream 2)

Measured with `scripts/bench.py` on Python 3.13.7, numpy 2.4.6, Apple Silicon. All timelines /
frames / transcripts are synthesized in-memory (deterministic, no ffmpeg/media), so the numbers are
reproducible: `python scripts/bench.py`. Profiler: `python scripts/bench.py --profile`.

## Baseline numbers (median ms)

| op | size | median ms |
|---|---|---|
| `ir_build` | 10 clips | 0.17 |
| `ir_build` | 100 clips | 0.66 |
| `ir_build` | 500 clips | 3.32 |
| `ir_build` | **2000 clips** | **19.14** |
| `fgx_bundle(hops=1,sel=5)` | 2000 clips | 0.44 |
| `fgx_bundle+to_text` | 2000 clips | 0.45 |
| `fgx_bundle(include_vertical,3trk)` | 2000×3 clips | 0.97 |
| `analyze_rgb` | 160×120 (**real path**) | 0.79 |
| `analyze_rgb` | 320×240 | 3.31 |
| `analyze_rgb` | 640×480 | 21.65 |
| `pack_transcript` | 5000 words | 0.63 |

## Reading the numbers honestly

- **film-grip is already fast for typical projects.** A ≤500-clip timeline builds in ~3ms; FGX
  projection and transcript packing are sub-millisecond. Nothing here is a user-visible problem at
  normal scale.
- **FGX projection is cheap and scales fine.** `bundle` is O(selected × clips) via
  `ir.neighbors → clips_on` (a full O(n) scan + sort per selected clip), but with a typical handful
  of selected clips the constant is tiny — 0.44ms at 2000 clips. Not worth touching yet; noted for
  the record in case selection sizes grow.
- **`analyze_rgb` is NOT a real hot path.** `frame_rgb` downscales every frame to width 160
  (`_SCOPE_WIDTH`) before analysis, so the production cost is the 160px row: **0.79ms**. The 21ms
  640×480 row only happens if a caller hands `analyze_rgb` a full-res frame, which the codebase
  never does. The algorithm does have redundant work (5 separate `np.percentile`/`np.median`
  passes over the pixel vector — each an O(n log n) sort) — a single-pass refactor is a *correctness-
  neutral micro-win*, low priority.

## The one clear target: IR build at scale

`--profile` (2000×3 clips, 20 iters) attributes ~59% of wall-clock to the per-clip indexing chain:

```
ncalls  tottime  cumtime  function
120000   0.043    0.520   ir.py:132(_add)
120000   0.073    0.467   idmap.py:59(mint)
120000   0.116    0.298   idmap.py:35(stable_clip_id)   # blake2b + to_base36
120000   0.047    0.231   ir.py:52(media_ref_string)    # getattr chain + posixpath.basename
360000   0.135    0.135   builtins.getattr
120000   0.089    0.128   idmap.py:25(to_base36)        # divmod loop + rjust
120000   0.050    0.050   <string>:2(__init__)          # Clip dataclass __init__
```

`from_otio` is called on **every snapshot** of the timeline, so for a 1000–4000-clip project (a
feature edit, a long multicam) the rebuild cost is the thing that adds up across an agent's many
turns. Optimization hypotheses, each independently measurable with `bench.py` and guarded by the
507-test suite:

1. **`@dataclass(slots=True)` on `Clip`** — faster construction + attribute access, lower memory.
   Expected: ~10–20% off `ir_build`. *Risk:* breaks any code that sets a non-declared attribute on
   a `Clip`; must grep first.
2. **Inline basename in `media_ref_string`** — replace `os.path.basename` (posixpath overhead,
   0.23s cumulative) with a direct `rfind`/`rsplit`. Expected: small but free.
3. **Cheaper `to_base36`** — the digest is a fixed 32 bits, so the base36 string is ≤7 chars;
   a table/format-based conversion avoids the per-char `divmod`+`rjust`. Expected: small.
4. **(Investigate) avoid re-minting unchanged clips on `reindex`** — `reindex` rebuilds the whole
   IdMap after an in-place mutation; only touched clips changed. A diff-aware reindex would cut the
   post-edit rebuild. Larger change; needs the decompose to scope.

**Done = a `bench.py` before/after table showing a real reduction in `ir_build` at 2000 clips, with
the full suite still green and a regression test pinning IR-build correctness on a large timeline.**
"Faster" is only a claim when the table proves it — no vibes.

## What this workstream will NOT do

- Chase the 640×480 `analyze_rgb` number — it isn't the real path (frames are 160px).
- Micro-optimize FGX projection — already sub-ms; would add complexity for no user-visible win.
- Rewrite the IR in another language — measured and rejected previously (≤1.15% wall-clock); the
  costs above are Python-object churn, addressable in Python (slots, inlining).

## Measured result (2026-06-21)

Landed hypotheses 1 + 2 (`@dataclass(slots=True)` on `Clip`; inline basename in `media_ref_string`).
Clean back-to-back A/B on an idle machine — `TimelineIR.from_otio`, 25-repeat median:

| clips | before | after | delta |
|------:|-------:|------:|------:|
| 100  | 0.641 ms | 0.600 ms | ~6% |
| 500  | 3.214 ms | 2.989 ms | ~7% |
| 2000 | 17.098 ms | 16.603 ms | ~3% |

Honest reading: the win is **small and shrinks with size** — a confirmation re-run measured
0.604 / 3.176 / 16.812 ms, so the 500- and 2000-clip deltas sit near measurement noise. That is
expected, not disappointing: the profile above puts the IR-build hot spot in the blake2b + base36 ID
**mint** (hypothesis 3 territory), which slots does not touch. `Clip.__init__` is only ~0.050s of the
total, so slotting it can only move that slice.

What slots buys unconditionally is **lower memory per `Clip`** (no per-instance `__dict__`), real on
the 1000–4000-clip projects this path targets, plus it is the idiomatic shape for a hot value object.
Kept on those grounds; correctness-neutral (full suite green; `test_bench_guard` builds + validates
a 2000-clip timeline, pinning large-timeline IR-build correctness). Hypotheses 3 (table-based
`to_base36`) and 4 (diff-aware `reindex`) are the remaining real levers if IR-build ever becomes a
user-visible cost — not pursued now: absolute time is already sub-20ms at 2000 clips, and
honesty-tier discipline says do not add complexity chasing a number the user cannot feel.
