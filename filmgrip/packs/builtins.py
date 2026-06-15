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


def _silence_cut(ir, ids, params) -> list:
    """Transcript-driven silence/filler removal — zero LLM calls.

    Pulls word-level transcripts (perception layer), finds silences ≥ min_silence and filler
    words, and compiles descending ``cut_range`` ops. Honest failure: no ASR backend, offline
    media, or per-clip analysis errors raise PackError with the real reasons — the pack never
    silently edits less than it was asked to.
    """
    from ..perception.speech import analyze_clips
    from ..perception.transcribe import PerceptionUnavailable, detect_backend
    from . import PackError

    try:
        backend = detect_backend()
    except PerceptionUnavailable as exc:
        raise PackError(f"silence-cut needs a transcription backend:\n{exc}") from exc
    min_silence_s = _seconds(params.get("min_silence", "0.4s"))
    include_fillers = str(params.get("fillers", "true")).lower() not in ("false", "0", "no")
    analysis = analyze_clips(
        ir, [c.id for c in selected_clips(ir, ids)], backend=backend,
        min_silence_s=min_silence_s, include_fillers=include_fillers)
    if analysis["errors"]:
        raise PackError("silence-cut could not analyze every selected clip:\n  "
                        + "\n  ".join(analysis["errors"]))
    return analysis["candidates"]


def _seconds(value) -> float:
    """'0.4s' / '400ms' / 0.4 → seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().lower()
    if text.endswith("ms"):
        return float(text[:-2]) / 1000.0
    return float(text[:-1] if text.endswith("s") else text)


register(Pack("silence-cut", "Remove silences (and umm/uh fillers) from the selected clips, "
                             "driven by word-level transcripts — deterministic, no LLM call.",
              compile=_silence_cut, params={"min_silence": "0.4s", "fillers": "true"},
              requires=("an ASR backend (pip install 'film-grip[transcribe]', whisper.cpp, or "
                        "ELEVENLABS_API_KEY)", "source media files on disk")))


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


# --- grade packs (color look presets) ----------------------------------------------------------
# Deterministic creative looks compile to a portable `set_cdl` per selected clip — zero LLM, always
# applicable (set_cdl is live in Resolve + carried in interchange). These are CDL PRIMARY
# approximations of looks; a true split-tone look wants a 3D LUT (apply_lut), but a primary grade is
# portable, re-editable, and gets most of the way. The ergonomic lift/gamma/gain controls compile to
# CDL via filmgrip.color.lgg_to_cdl, so the recipe reads in colorist terms.

def _grade_compiler(cdl):
    def _compile(ir, ids, params):
        d = cdl.to_dict()
        return [{"op": "set_cdl", "clip_id": c.id, "slope": d["slope"], "offset": d["offset"],
                 "power": d["power"], "saturation": d["saturation"]}
                for c in selected_clips(ir, ids)]
    return _compile


def _register_look(name: str, description: str, cdl) -> None:
    register(Pack(name, description, compile=_grade_compiler(cdl)))


def _looks():
    from ..color import lgg_to_cdl
    _register_look(
        "teal-orange",
        "Cinematic teal-orange: warm highlights, cool shadows, punchy contrast (CDL primary "
        "approximation — a 3D LUT does split-tone better).",
        lgg_to_cdl(gain=(1.08, 1.0, 0.94), contrast=1.12, saturation=1.12))
    _register_look(
        "filmstock-warm",
        "Warm filmstock emulation: lifted shadows, gentle warmth, mild saturation.",
        lgg_to_cdl(lift=(0.02, 0.01, 0.0), gain=(1.05, 1.0, 0.96), saturation=1.05))
    _register_look(
        "bleach-bypass",
        "High-contrast, low-saturation bleach-bypass look.",
        lgg_to_cdl(contrast=1.4, saturation=0.45))
    _register_look(
        "day-for-night",
        "Cool, darker day-for-night: blue cast, lowered mids and saturation.",
        lgg_to_cdl(gain=(0.85, 0.95, 1.18), gamma=0.8, saturation=0.7))


_looks()


# Prompt grade packs — the planner reads the clip scopes (film-grip scopes) and emits CDL.
register(Pack("neutral-balance",
              "Neutralize color cast and set a clean mid exposure, read from the clip's scopes.",
              kind="prompt",
              prompt="Read the color scopes for the selected clips. Neutralize any color cast so "
                     "neutrals are neutral and place mid-grey near {target_mid} (0–1). Emit a "
                     "conservative set_cdl per clip; do not crush shadows or clip highlights, and "
                     "keep saturation natural.",
              params={"target_mid": "0.45"}))
register(Pack("grade-match",
              "Match the selected clips' color to a reference / hero clip using CDL.",
              kind="prompt",
              prompt="Match the color of the selected clips to the reference clip {ref}. Compare "
                     "both clips' scopes (exposure, white balance, saturation, skin-tone), then emit "
                     "set_cdl per clip (or apply_grade to copy the hero grade) so they match the "
                     "reference within tolerance. Verify with the scopes after.",
              params={"ref": ""}))
