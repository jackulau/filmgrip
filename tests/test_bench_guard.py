"""D2 — performance regression tripwire for the pure-CPU hot paths.

This is NOT a perf target and NOT a microbenchmark assertion — it is a *tripwire*. The bounds below
are deliberately ~10–15× the measured medians in ``docs/research/perf-baseline.md`` (ir_build ≈
19ms, fgx_bundle+to_text ≈ 0.45ms at 2000 clips), so a green run says nothing about whether the code
got faster — only that nothing made it catastrophically (5–10×) slower. They are loose enough never
to flake on a slow/loaded CI box (where the same paths already clock ~2× the doc baseline) yet tight
enough to catch the kind of accidental O(n²) blow-up or per-clip I/O a refactor could introduce.

If you legitimately make a hot path much faster, these bounds do not need touching; if you must
*raise* one, that is a signal a regression slipped in — investigate before loosening.
"""
from __future__ import annotations

from filmgrip.bench import hot_paths

# Fixed clip count so the guard is comparable run-to-run; large enough that an O(n²) regression in
# any hot path would blow well past the bounds, small enough to stay fast in the suite.
GUARD_CLIPS = 2000

# Generous upper bounds (ms). ~15× the documented ir_build median and ~100× the fgx median — a
# regression tripwire, not a perf target. See the module docstring.
IR_BUILD_MAX_MS = 300.0
FGX_BUNDLE_TO_TEXT_MAX_MS = 50.0
VALIDATE_MAX_MS = 100.0


def test_hot_paths_stay_under_regression_tripwire():
    hp = hot_paths(GUARD_CLIPS)

    # The runner reports exactly the three pure-CPU paths every agent turn pays.
    assert set(hp) == {"ir_build", "fgx_bundle_to_text", "validate"}

    assert hp["ir_build"] < IR_BUILD_MAX_MS, (
        f"ir_build regression tripwire: {hp['ir_build']:.2f}ms at {GUARD_CLIPS} clips "
        f"exceeded {IR_BUILD_MAX_MS}ms (~15× the {19}ms baseline)")
    assert hp["fgx_bundle_to_text"] < FGX_BUNDLE_TO_TEXT_MAX_MS, (
        f"fgx_bundle+to_text regression tripwire: {hp['fgx_bundle_to_text']:.3f}ms at "
        f"{GUARD_CLIPS} clips exceeded {FGX_BUNDLE_TO_TEXT_MAX_MS}ms")
    assert hp["validate"] < VALIDATE_MAX_MS, (
        f"validate regression tripwire: {hp['validate']:.3f}ms at {GUARD_CLIPS} clips "
        f"exceeded {VALIDATE_MAX_MS}ms")
