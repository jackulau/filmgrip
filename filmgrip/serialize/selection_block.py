"""``<selected_clips>`` — the human- and LLM-readable view of a grabbed selection.

This is film-grip's analog of react-grab's ``<selected_element>`` block. One gesture
(``film-grip grab``, or the panel's *Copy context*) turns "the clips I have selected" into a
compact, labeled context block that carries everything an agent needs to act:

* each clip's stable ID, name, track, in/out (frames + timecode), source media + source-in,
* the same-track neighbors, so the *sequence relationship* — the thing that actually matters in
  editing — is explicit rather than implied.

Where FGX (:mod:`filmgrip.serialize.fgx`) is the token-frugal JSON the in-process planner *pulls*,
this block is the prose-ish projection a human skims in the panel and that pastes cleanly into an
external agent (Claude Code, Codex, Cursor) via the clipboard. Same IR, two surfaces.
"""
from __future__ import annotations

from typing import Iterable

from ..core.ir import Clip, TimelineIR
from .fgx import _rate_str, track_code


def frames_to_tc(frames: int, rate: float) -> str:
    """Non-drop ``HH:MM:SS:FF`` timecode for a frame count. Display only (uses the integer rate)."""
    fps = max(1, int(round(rate)))
    f = max(0, int(frames))
    ff = f % fps
    secs = f // fps
    return f"{secs // 3600:02d}:{(secs // 60) % 60:02d}:{secs % 60:02d}:{ff:02d}"


def _clip_line(ir: TimelineIR, clip: Clip) -> str:
    t = track_code(clip.track_kind, clip.track_index)
    return (f"- [{clip.id}] {clip.name}  track {t}  "
            f"in {frames_to_tc(clip.start, ir.rate)} ({clip.start}f)  "
            f"out {frames_to_tc(clip.end, ir.rate)} ({clip.end}f)  dur {clip.duration}f  "
            f"src {clip.src_ref} @ {clip.source_start}f")


def _context_line(ir: TimelineIR, clip: Clip) -> str | None:
    """One line of same-track sequence context: the items immediately before/after ``clip``."""
    track = sorted(ir.clips_on(clip.track_kind, clip.track_index), key=lambda c: c.start)
    pos = next((i for i, c in enumerate(track) if c.id == clip.id), None)
    if pos is None:
        return None
    parts = []
    if pos > 0:
        prev = track[pos - 1]
        parts.append(f"prev={prev.name}(ends {prev.end}f)")
    if pos + 1 < len(track):
        nxt = track[pos + 1]
        parts.append(f"next={nxt.name}(starts {nxt.start}f)")
    if not parts:
        return None
    return f"- [{clip.id}] {track_code(clip.track_kind, clip.track_index)}: " + "  ".join(parts)


def format_selection(
    ir: TimelineIR,
    selected_ids: Iterable[str],
    *,
    confidence: str = "precise",
    basis: str = "",
    neighbors: bool = True,
) -> str:
    """Render a selection as a ``<selected_clips>`` block.

    ``confidence`` / ``basis`` come from the adapter's :class:`~filmgrip.adapters.base.Selection`
    so the block stays honest about *how* the selection was derived (Resolve has no true multi-clip
    selection API — it surfaces ``reconstructed``). Unknown IDs are silently dropped; an empty
    selection renders a clear "nothing selected" block rather than a bare shell.
    """
    clips = sorted(
        (c for s in selected_ids if (c := ir.clip(s)) is not None),
        key=lambda c: (c.track_kind, c.track_index, c.start),
    )
    head = f"## timeline: {ir.timeline.name or 'seq'}  fps: {_rate_str(ir.rate)}  selected: {len(clips)} [{confidence}]"
    if basis:
        head += f"  basis: {basis}"
    lines = ["<selected_clips>", head]
    if not clips:
        lines += ["## clips: (none — select clips on the timeline, then grab)", "</selected_clips>"]
        return "\n".join(lines)
    lines.append("## clips:")
    lines += [_clip_line(ir, c) for c in clips]
    if neighbors:
        ctx = [ln for c in clips if (ln := _context_line(ir, c)) is not None]
        if ctx:
            lines.append("## context (same-track neighbors):")
            lines += ctx
    lines.append("</selected_clips>")
    return "\n".join(lines)


def armed_preview(
    ir: TimelineIR,
    selected_ids: Iterable[str],
    *,
    confidence: str = "precise",
    limit: int = 8,
) -> str:
    """A compact HUD list of the clips currently "armed" for a grab — the panel capture-preview.

    react-grab's purple hover-highlight tells you *exactly what you'll grab before you commit*; a
    Resolve script can't repaint timeline clips, so this is the honest equivalent: a terse,
    per-clip list (name · track · in–out) the panel shows so the editor sees the capture boundary
    rather than grabbing blind. Truncates past ``limit`` so a huge selection can't blow up the label.
    """
    clips = sorted(
        (c for s in selected_ids if (c := ir.clip(s)) is not None),
        key=lambda c: (c.track_kind, c.track_index, c.start),
    )
    if not clips:
        return f"(no clips armed [{confidence}]) — select clips in Resolve and they appear here"
    lines = [f"{len(clips)} clip(s) armed [{confidence}]:"]
    for c in clips[:limit]:
        lines.append(f"  • {c.name}  {track_code(c.track_kind, c.track_index)}  "
                     f"{c.start}–{c.end}f")
    if len(clips) > limit:
        lines.append(f"  … +{len(clips) - limit} more")
    return "\n".join(lines)
