"""Built-in deterministic edit packs.

Each recipe is IR-aware so it only ever emits ops that validate: ``marker-pass`` puts markers at
frame 0 (always in-bounds); ``punch-up`` uses fades (which need no neighbor); ``dissolves`` checks
adjacency before emitting a cross dissolve. The result still goes through ``validate`` — these just
never give it anything to reject.
"""
from __future__ import annotations

from . import Pack, register, selected_clips


def _marker_pass(ir, ids, params) -> list:
    color, note = params.get("color", "Blue"), params.get("note", "review")
    return [{"op": "add_marker", "clip_id": c.id, "frame": 0, "color": color, "note": note}
            for c in selected_clips(ir, ids)]


def _punch_up(ir, ids, params) -> list:
    fade = int(params.get("fade", 12))
    clips = selected_clips(ir, ids)
    if not clips:
        return []
    first, last = clips[0], clips[-1]
    return [
        {"op": "add_transition", "clip_id": first.id, "edge": "in", "type": "fade_in", "duration": fade},
        {"op": "add_transition", "clip_id": last.id, "edge": "out", "type": "fade_out", "duration": fade},
    ]


def _dissolves(ir, ids, params) -> list:
    dur = int(params.get("duration", 12))
    ops = []
    for c in selected_clips(ir, ids):
        track = sorted(ir.clips_on(c.track_kind, c.track_index), key=lambda x: x.start)
        pos = next((j for j, x in enumerate(track) if x.id == c.id), None)
        if pos is not None and pos + 1 < len(track):  # a following item exists → cross_dissolve is legal
            ops.append({"op": "add_transition", "clip_id": c.id, "edge": "out",
                        "type": "cross_dissolve", "duration": dur})
    return ops


register(Pack("marker-pass", "Drop a review marker on each selected clip.",
              compile=_marker_pass, params={"color": "Blue", "note": "review"}))
register(Pack("punch-up", "Top-and-tail the selection: fade in on the first clip, fade out on the last.",
              compile=_punch_up, params={"fade": 12}))
register(Pack("dissolves", "Add a cross dissolve on the out-edge of each selected clip that has a "
                           "following clip.",
              compile=_dissolves, params={"duration": 12}))


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
