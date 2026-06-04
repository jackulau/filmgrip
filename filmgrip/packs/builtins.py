"""Built-in deterministic edit packs.

Each recipe is IR-aware so it only ever emits ops that validate AND that some adapter can actually
apply: ``marker-pass`` puts markers at frame 0 (always in-bounds, and ``add_marker`` is live in
Resolve / rebuildable in interchange). A deterministic pack must NOT emit an op no adapter can apply —
that would make ``film-grip pack <p>`` report success while changing nothing (the honesty contract,
enforced by tests/test_packs_applicable.py). That is why there is no ``punch-up``/``dissolves`` pack:
they emitted ``add_transition``, which no editor automation can apply (Resolve scripting can't add a
timeline transition; the OTIO round-trip doesn't carry it reliably). ``add_transition`` stays a valid
op the *planner* may propose — adapters then reject it honestly (ok=False, exit 3) — but no built-in
pack auto-emits an op the tool can't perform.
"""
from __future__ import annotations

from . import Pack, register, selected_clips


def _marker_pass(ir, ids, params) -> list:
    color, note = params.get("color", "Blue"), params.get("note", "review")
    return [{"op": "add_marker", "clip_id": c.id, "frame": 0, "color": color, "note": note}
            for c in selected_clips(ir, ids)]


register(Pack("marker-pass", "Drop a review marker on each selected clip.",
              compile=_marker_pass, params={"color": "Blue", "note": "review"}))


# --- prompt packs (D7): a saved instruction handed to the active planner backend ---------------
# Parameters fill {placeholders} in the prompt at apply time; users can add their own prompt packs
# as data files in ~/.filmgrip/packs (see filmgrip.packs.loader).
register(Pack("podcast-cleanup", "Tighten a talking-head/podcast cut: close long silences and snug "
                                 "the edits.", kind="prompt",
              prompt="Remove silent gaps longer than {min_silence} on the selected clips and "
                     "ripple the timeline closed so there are no holes. Do not alter clip content "
                     "or audio levels.",
              params={"min_silence": "0.5s"}))
register(Pack("tighten-open", "Tighten the opening: trim slack off the head of the first selected "
                              "clip.", kind="prompt",
              prompt="Tighten the open: trim up to {frames} frames off the head of the first "
                     "selected clip so it starts on the action. Keep everything else untouched.",
              params={"frames": 12}))
