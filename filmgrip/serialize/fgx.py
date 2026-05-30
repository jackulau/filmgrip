"""FGX — film-grip's token-frugal projection of the timeline for the LLM.

Claude is a planner that *pulls* context, never gets force-fed a project dump. FGX is the compact
view it pulls. Design choices, each a deliberate token win:

* **integer frames + one rate header** — ``[..,48,72,..]`` tokenizes far cheaper than
  ``"01:00:02:00"`` and is exact.
* **positional rows** — ``[id, t, s, d, src, si]`` with the column order declared *once* (in the
  system prompt / ``COLS``), so per-clip key names are never repeated.
* **selected subgraph + an N-hop neighbor ring** — never the whole project; deeper context only
  when Claude calls ``get_context(hops)``.
* **media by short reference** — a basename, never inlined bytes/waveforms.
* **gaps/transitions as terse markers** — ``"GAP"`` / ``"XFADE"`` rows keep structure without
  pretending they are editable clips (their id column is ``"-"``).
* **deltas, not dumps** — after turn 1, :func:`delta` emits only changed/new rows.

The full OTIO is always retained host-side, so this lossy view is losslessly re-expandable.
"""
from __future__ import annotations

import json
from typing import Iterable, Optional

from ..core.ir import Clip, TimelineIR

# Column legend for the positional clip rows. Declared once; sent to the model in the system
# prompt rather than repeated per row.
COLS = ["id", "t", "s", "d", "src", "si"]  # id, track, start, dur, srcRef, sourceIn

_KIND_MARK = {"gap": "GAP", "transition": "XFADE"}


def track_code(kind: str, index: int) -> str:
    """('video', 1) -> 'v1' ; ('audio', 2) -> 'a2'."""
    return f"{kind[0].lower()}{index}"


def parse_track_code(code: str) -> tuple[str, int]:
    kind = {"v": "video", "a": "audio", "s": "subtitle"}[code[0].lower()]
    return kind, int(code[1:])


def clip_row(clip: Clip) -> list:
    """Project one Clip to a positional FGX row matching :data:`COLS`."""
    if clip.kind in _KIND_MARK:
        return ["-", track_code(clip.track_kind, clip.track_index), clip.start, clip.duration,
                _KIND_MARK[clip.kind], 0]
    return [clip.id, track_code(clip.track_kind, clip.track_index), clip.start, clip.duration,
            clip.src_ref, clip.source_start]


def _subgraph(ir: TimelineIR, selected_ids: Iterable[str], hops: int, include_vertical: bool) -> list[Clip]:
    selected = [ir.clip(s) for s in selected_ids if ir.clip(s) is not None]
    chosen: dict[str, Clip] = {}
    for c in selected:
        chosen[c.id] = c
        for nb in ir.neighbors(c.id, hops):
            chosen[nb.id] = nb
    if include_vertical and selected:
        lo = min(c.start for c in selected)
        hi = max(c.end for c in selected)
        for c in ir.clips:
            if c.id in chosen:
                continue
            if c.start < hi and c.end > lo:  # temporal overlap on any track
                chosen[c.id] = c
    # Stable display order: track then start.
    return sorted(chosen.values(), key=lambda c: (c.track_kind, c.track_index, c.start))


def selection_header(ir: TimelineIR, selected_ids: Iterable[str]) -> dict:
    """The tiny ``get_selection`` payload — a few dozen tokens, no clip detail."""
    return {"seq": ir.timeline.name or "seq", "r": _rate_str(ir.rate),
            "sel": [s for s in selected_ids if ir.clip(s) is not None]}


def bundle(
    ir: TimelineIR,
    selected_ids: Iterable[str],
    *,
    hops: int = 1,
    include_vertical: bool = False,
) -> dict:
    """Build the FGX context bundle for a selection.

    Includes the selected clips plus an N-hop same-track neighbor ring (and, when
    ``include_vertical``, temporally-overlapping clips on other tracks).
    """
    selected_ids = list(selected_ids)
    rows = [clip_row(c) for c in _subgraph(ir, selected_ids, hops, include_vertical)]
    return {
        "seq": ir.timeline.name or "seq",
        "r": _rate_str(ir.rate),
        "cols": COLS,
        "sel": [s for s in selected_ids if ir.clip(s) is not None],
        "clips": rows,
    }


def delta(
    ir: TimelineIR,
    selected_ids: Iterable[str],
    prev_rows: dict[str, list],
    *,
    hops: int = 1,
    include_vertical: bool = False,
) -> dict:
    """Emit only rows that are new or changed since ``prev_rows`` ({id: row}).

    Used on multi-turn ``--resume`` so turn 2+ costs only the delta, not the whole subgraph.
    """
    full = bundle(ir, selected_ids, hops=hops, include_vertical=include_vertical)
    changed = []
    for row in full["clips"]:
        cid = row[0]
        if cid == "-":  # gaps/transitions are positional context; resend if anything changed
            changed.append(row)
            continue
        if prev_rows.get(cid) != row:
            changed.append(row)
    return {
        "seq": full["seq"],
        "r": full["r"],
        "cols": COLS,
        "sel": full["sel"],
        "changed": changed,
    }


def to_text(obj: dict) -> str:
    """Serialize an FGX bundle to the most compact JSON form (no whitespace)."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def estimate_tokens(obj) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token). Good enough for budgeting/asserts."""
    text = obj if isinstance(obj, str) else to_text(obj)
    return max(1, round(len(text) / 4))


def _rate_str(rate: float) -> str:
    # Emit "24" for integer rates; keep the NDF rational form otherwise.
    if abs(rate - round(rate)) < 1e-6:
        return str(int(round(rate)))
    if abs(rate - 24000 / 1001) < 1e-3:
        return "24000/1001"
    if abs(rate - 30000 / 1001) < 1e-3:
        return "30000/1001"
    return f"{rate:g}"
