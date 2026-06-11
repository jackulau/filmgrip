"""D2 (goal 004) — a deterministic built-in pack may only emit ops some adapter can actually apply.

A pack that emits an op no editor automation can perform would make `film-grip pack <p>` report
success while changing nothing — the warn-and-noop the project forbids. This is what removed the old
`punch-up`/`dissolves` packs (they emitted `add_transition`). This test guards against it returning.
"""
from __future__ import annotations

from filmgrip.adapters.interchange import REBUILD_OPS
from filmgrip.adapters.resolve_adapter import LIVE_EXTRA_OPS, LIVE_OPS
from filmgrip.core.ir import TimelineIR
from filmgrip.packs import all_packs
from filmgrip.packs.engine import compile_pack

FIX = "tests/fixtures/cut.otio"

# Every op SOME adapter can apply (Resolve is the most capable). add_transition is deliberately NOT
# here — no adapter applies it; the planner may propose it, and adapters then reject it honestly.
APPLICABLE_ANYWHERE = LIVE_OPS | LIVE_EXTRA_OPS | REBUILD_OPS


def test_add_transition_is_not_applicable_anywhere():
    # documents the premise: this is the one schema op no adapter can apply.
    assert "add_transition" not in APPLICABLE_ANYWHERE


def test_deterministic_builtin_packs_emit_only_applicable_ops():
    ir = TimelineIR.from_otio_file(FIX)
    ids = [c.id for c in ir.real_clips()]
    # Packs with declared runtime requirements (e.g. silence-cut needs an ASR backend + media on
    # disk) can't compile against this bare fixture; their applicability is covered with a fake
    # backend in tests/test_speech.py. Everything requirement-free must compile here.
    builtins = [p for p in all_packs()
                if p.source == "built-in" and p.kind == "deterministic" and not p.requires]
    assert builtins, "expected at least the marker-pass built-in"
    for pack in builtins:
        plan = compile_pack(pack, ir, ids)
        emitted = {op.op for op in plan.ops}
        unappliable = emitted - APPLICABLE_ANYWHERE
        assert not unappliable, (
            f"pack '{pack.name}' emits {unappliable}, which no adapter can apply — it would report "
            f"success while changing nothing"
        )
