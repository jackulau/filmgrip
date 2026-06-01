"""Edit packs — named, reusable edit recipes.

A *pack* turns a one-word intent ("punch-up", "marker-pass") into edits, so common operations don't
need a fresh natural-language prompt every time. Two kinds:

* **deterministic** — an IR-aware recipe that compiles straight to a typed
  :class:`~filmgrip.protocol.editplan.EditPlan` and runs through the SAME validate→apply pipeline as
  everything else. No LLM, no cost, repeatable. (D6)
* **prompt** — a saved, parameterized instruction handed to the active planner backend. (D7)

Built-ins live in :mod:`filmgrip.packs.builtins`; user packs (D7) are discovered from
``~/.filmgrip/packs``. Packs never bypass validation — a recipe that would emit an invalid op is
rejected exactly like a hallucinated LLM plan, because it goes through the same ``validate``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..core.ir import Clip, TimelineIR

# A deterministic pack's compiler: (ir, selected_ids, params) -> list of op dicts.
CompileFn = Callable[[TimelineIR, list, dict], list]


class PackError(RuntimeError):
    """Raised for an unknown pack, or a misuse (e.g. compiling a prompt pack to ops)."""


@dataclass
class Pack:
    name: str
    description: str
    kind: str = "deterministic"            # "deterministic" | "prompt"
    compile: Optional[CompileFn] = None    # deterministic: builds op dicts from ir + selection
    prompt: str = ""                       # prompt kind (D7): the parameterized instruction
    params: dict = field(default_factory=dict)
    source: str = "built-in"               # "built-in" | a file path (user packs, D7)


_REGISTRY: dict[str, Pack] = {}


def register(pack: Pack) -> None:
    """Register a built-in pack (idempotent — last registration wins)."""
    _REGISTRY[pack.name] = pack


def get_pack(name: str) -> Pack:
    pack = _REGISTRY.get(name)
    if pack is None:
        raise PackError(f"unknown pack '{name}'. Run `film-grip pack list` to see available packs.")
    return pack


def all_packs() -> list[Pack]:
    """All packs, sorted by name. (D7 layers user packs over these.)"""
    return [p for _, p in sorted(_REGISTRY.items())]


def selected_clips(ir: TimelineIR, ids) -> list[Clip]:
    """The real (editable) clips for ``ids``, in track→start order. Gaps/transitions are dropped."""
    out = [c for c in (ir.clip(i) for i in (ids or [])) if c is not None and c.kind == "clip"]
    return sorted(out, key=lambda c: (c.track_kind, c.track_index, c.start))


from . import builtins as _builtins  # noqa: E402,F401  (import registers the built-ins)
