"""Validate an EditPlan against a live IR — the safety gate before anything is applied.

Parsing (D5) guarantees an op is *well-formed*; validation guarantees it is *applicable to this
timeline right now*: the clip exists, the target track exists, frames are in range, trims stay
inside the source media, moves/inserts don't illegally overlap, transitions touch an adjacent
clip, splits fall inside the clip. A plan is accepted only if **every** op validates — partial
application of an editing plan is worse than none, so rejection is atomic. Each failure carries a
machine-readable code so the D16 self-heal loop can feed precise errors back to Claude for repair.
``dry_run`` renders the same plan as a human-readable diff without touching the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..core.ir import Clip, TimelineIR
from ..serialize.fgx import parse_track_code
from .editplan import EditPlan

# --- error codes (stable identifiers; surfaced to Claude on repair turns) ------------------
UNKNOWN_CLIP = "UNKNOWN_CLIP"
TRACK_NOT_FOUND = "TRACK_NOT_FOUND"
OUT_OF_BOUNDS = "OUT_OF_BOUNDS"
TRIM_EXCEEDS_SOURCE = "TRIM_EXCEEDS_SOURCE"
ILLEGAL_OVERLAP = "ILLEGAL_OVERLAP"
TRANSITION_NOT_ADJACENT = "TRANSITION_NOT_ADJACENT"
SPLIT_OUT_OF_CLIP = "SPLIT_OUT_OF_CLIP"
NOT_A_CLIP = "NOT_A_CLIP"
NOT_AUDIO_TRACK = "NOT_AUDIO_TRACK"
NO_AUDIO_SOURCE = "NO_AUDIO_SOURCE"
CUT_RANGE_ORDER = "CUT_RANGE_ORDER"
CLIP_TIMEWARPED = "CLIP_TIMEWARPED"

_TRANSITIONS_NEED_NEIGHBOR = {"cross_dissolve", "dip_to_color", "smooth_cut", "wipe"}


@dataclass
class OpError:
    code: str
    op_index: int
    op: str
    message: str
    clip_id: Optional[str] = None

    def __str__(self) -> str:
        loc = f"[{self.op_index}] {self.op}"
        cid = f" ({self.clip_id})" if self.clip_id else ""
        return f"{self.code}{cid} {loc}: {self.message}"


@dataclass
class ValidationResult:
    errors: list[OpError] = field(default_factory=list)
    warnings: list[OpError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def __bool__(self) -> bool:
        return self.ok

    def codes(self) -> list[str]:
        return [e.code for e in self.errors]


def _track_exists(ir: TimelineIR, code: str) -> bool:
    try:
        kind, idx = parse_track_code(code)
    except Exception:
        return False
    return 1 <= idx <= ir.track_count(kind)


def _clips_on_code(ir: TimelineIR, code: str, *, exclude: Optional[str] = None) -> list[Clip]:
    kind, idx = parse_track_code(code)
    return [c for c in ir.clips_on(kind, idx)
            if c.kind == "clip" and c.id != exclude]


def _overlaps(start: int, dur: int, others: list[Clip]) -> Optional[Clip]:
    end = start + dur
    for c in others:
        if c.start < end and c.end > start:
            return c
    return None


def _source_available(clip: Clip):
    """Return (in, out) source bounds in frames if known from the OTIO media reference, else None."""
    mr = getattr(clip.otio, "media_reference", None)
    rng = getattr(mr, "available_range", None) if mr is not None else None
    if rng is None:
        return None
    start = int(round(rng.start_time.to_frames()))
    return start, start + int(round(rng.duration.to_frames()))


def _is_timewarped(clip: Clip) -> bool:
    """True when the clip carries a retime effect — frame surgery on it would be wrong."""
    import opentimelineio as otio

    effects = getattr(clip.otio, "effects", None) or []
    return any(isinstance(e, (otio.schema.LinearTimeWarp, otio.schema.FreezeFrame))
               for e in effects)


def validate(plan: EditPlan, ir: TimelineIR) -> ValidationResult:
    res = ValidationResult()
    seq_dur = ir.duration

    def err(code, i, name, msg, cid=None):
        res.errors.append(OpError(code, i, name, msg, cid))

    def warn(code, i, name, msg, cid=None):
        res.warnings.append(OpError(code, i, name, msg, cid))

    # A plan can add_track then target it, so track-existence is evaluated against the timeline
    # PLUS tracks created by earlier ops in the same plan.
    added = {"video": 0, "audio": 0, "subtitle": 0}

    # Ordering contract for rippling cut_range ops: per track, strictly descending start_frame
    # (last-to-first), so a later op's frames aren't invalidated by an earlier op's ripple.
    last_ripple_cut_start: dict[str, int] = {}

    def track_exists(code: str) -> bool:
        try:
            kind, idx = parse_track_code(code)
        except Exception:
            return False
        return 1 <= idx <= ir.track_count(kind) + added.get(kind, 0)

    for i, op in enumerate(plan.ops):
        name = op.op
        if name == "add_track":
            added[op.kind] = added.get(op.kind, 0) + 1
            continue
        # ops that target an existing clip
        if name in ("trim", "move", "delete", "set_property", "add_marker", "add_transition",
                    "split", "move_to_bin", "retime", "set_enabled", "cut_range"):
            clip = ir.clip(op.clip_id)
            if clip is None:
                err(UNKNOWN_CLIP, i, name, f"no clip with id '{op.clip_id}' in current timeline", op.clip_id)
                continue
            if clip.kind != "clip":
                err(NOT_A_CLIP, i, name, f"'{op.clip_id}' is a {clip.kind}, not an editable clip", op.clip_id)
                continue

        if name == "trim":
            if op.edge == "out":
                new_start, new_dur = clip.start, clip.duration + op.delta
            else:  # "in"
                new_start, new_dur = clip.start + op.delta, clip.duration - op.delta
            if new_dur <= 0:
                err(OUT_OF_BOUNDS, i, name, f"trim leaves duration {new_dur} (<= 0)", op.clip_id)
            if new_start < 0:
                err(OUT_OF_BOUNDS, i, name, f"trim moves start to {new_start} (< 0)", op.clip_id)
            bounds = _source_available(clip)
            if bounds is not None:
                src_in = clip.source_start + (op.delta if op.edge == "in" else 0)
                src_out = clip.source_start + new_dur + (op.delta if op.edge == "in" else 0)
                if src_in < bounds[0] or src_out > bounds[1]:
                    err(TRIM_EXCEEDS_SOURCE, i, name,
                        f"trim exceeds source media bounds {bounds}", op.clip_id)
            else:
                warn(TRIM_EXCEEDS_SOURCE, i, name,
                     "source media length unknown; cannot verify trim stays in media", op.clip_id)

        elif name == "move":
            track_code = op.to_track or _code_of(clip)
            if op.to_track and not track_exists(op.to_track):
                err(TRACK_NOT_FOUND, i, name, f"track '{op.to_track}' does not exist", op.clip_id)
            else:
                others = _clips_on_code(ir, track_code, exclude=op.clip_id)
                hit = _overlaps(op.to_start, clip.duration, others)
                if hit is not None:
                    err(ILLEGAL_OVERLAP, i, name,
                        f"moving to {op.to_start} overlaps '{hit.name}' ({hit.start}..{hit.end})",
                        op.clip_id)

        elif name == "insert":
            if not track_exists(op.track):
                err(TRACK_NOT_FOUND, i, name, f"track '{op.track}' does not exist", None)
            else:
                others = _clips_on_code(ir, op.track)
                hit = _overlaps(op.at_start, op.duration, others)
                if hit is not None:
                    err(ILLEGAL_OVERLAP, i, name,
                        f"insert at {op.at_start} overlaps '{hit.name}' ({hit.start}..{hit.end})", None)

        elif name == "add_marker":
            if not (0 <= op.frame < clip.duration):
                err(OUT_OF_BOUNDS, i, name,
                    f"marker frame {op.frame} outside clip 0..{clip.duration}", op.clip_id)

        elif name == "add_transition":
            if op.type in _TRANSITIONS_NEED_NEIGHBOR:
                track = sorted(ir.clips_on(clip.track_kind, clip.track_index), key=lambda c: c.start)
                pos = next((j for j, c in enumerate(track) if c.id == op.clip_id), None)
                has_neighbor = (
                    (op.edge == "out" and pos is not None and pos + 1 < len(track))
                    or (op.edge == "in" and pos is not None and pos - 1 >= 0)
                )
                if not has_neighbor:
                    err(TRANSITION_NOT_ADJACENT, i, name,
                        f"'{op.type}' on the '{op.edge}' edge needs an adjacent clip", op.clip_id)

        elif name == "split":
            if not (clip.start < op.at_frame < clip.end):
                err(SPLIT_OUT_OF_CLIP, i, name,
                    f"split frame {op.at_frame} not inside clip {clip.start}..{clip.end}", op.clip_id)

        elif name == "cut_range":
            if not (clip.start <= op.start_frame < op.end_frame <= clip.end):
                err(OUT_OF_BOUNDS, i, name,
                    f"range {op.start_frame}..{op.end_frame} not inside clip "
                    f"{clip.start}..{clip.end}", op.clip_id)
            elif _is_timewarped(clip):
                err(CLIP_TIMEWARPED, i, name,
                    f"'{op.clip_id}' is retimed — frame ranges inside a time-warped clip don't "
                    f"map linearly; remove the retime first or address whole clips", op.clip_id)
            elif op.ripple:
                code = _code_of(clip)
                prev = last_ripple_cut_start.get(code)
                if prev is not None and op.start_frame >= prev:
                    err(CUT_RANGE_ORDER, i, name,
                        f"rippling cut_range ops on track {code} must be ordered last-to-first "
                        f"(this start {op.start_frame} ≥ previous {prev})", op.clip_id)
                last_ripple_cut_start[code] = op.start_frame

        elif name == "ripple":
            if op.track and not track_exists(op.track):
                err(TRACK_NOT_FOUND, i, name, f"track '{op.track}' does not exist", None)
            affected = ir.clips if not op.track else _clips_on_code(ir, op.track)
            for c in affected:
                if c.start >= op.from_frame and c.start + op.delta < 0:
                    err(OUT_OF_BOUNDS, i, name,
                        f"ripple would move '{c.name}' to {c.start + op.delta} (< 0)", c.id)
                    break

        elif name == "import_audio":
            if not (op.sfx or op.src_ref):
                err(NO_AUDIO_SOURCE, i, name, "import_audio needs 'sfx' or 'src_ref'", None)
            elif not track_exists(op.track):
                err(TRACK_NOT_FOUND, i, name, f"audio track '{op.track}' does not exist "
                    f"(add it with add_track first)", None)
            else:
                kind, _ = parse_track_code(op.track)
                if kind != "audio":
                    err(NOT_AUDIO_TRACK, i, name, f"'{op.track}' is not an audio track", None)
                elif op.duration is not None:
                    hit = _overlaps(op.at_start, op.duration, _clips_on_code(ir, op.track))
                    if hit is not None:
                        err(ILLEGAL_OVERLAP, i, name,
                            f"audio at {op.at_start} overlaps '{hit.name}' ({hit.start}..{hit.end})",
                            None)
                # duration None -> the file's full length is unknown until it's resolved, so the
                # overlap pre-check can't run here; the adapter places the full clip and warns.

        elif name == "rename_track":
            if not track_exists(op.track):
                err(TRACK_NOT_FOUND, i, name, f"track '{op.track}' does not exist", None)

        # add_track / create_bin / move_to_bin: shape is guaranteed at parse time; their live
        # preconditions (media-pool state) are checked by the adapter at apply (D7).

    return res


def _code_of(clip: Clip) -> str:
    return f"{clip.track_kind[0]}{clip.track_index}"


# --------------------------------------------------------------------------- dry-run diff
def dry_run(plan: EditPlan, ir: TimelineIR) -> str:
    """Render the plan as a human-readable diff. Validates first; shows errors if invalid."""
    res = validate(plan, ir)
    lines: list[str] = []
    if plan.notes:
        lines.append(f"# {plan.notes}")
    if not res.ok:
        lines.append(f"PLAN REJECTED — {len(res.errors)} error(s):")
        lines.extend(f"  ✗ {e}" for e in res.errors)
        return "\n".join(lines)

    lines.append(f"PLAN OK — {len(plan.ops)} op(s):")
    for op in plan.ops:
        desc = _describe(op, ir)
        if getattr(op, "quote", ""):
            desc += f' "{op.quote}"'
        if getattr(op, "reason", ""):
            desc += f" — {op.reason}"
        lines.append("  ✓ " + desc)
    if res.warnings:
        lines.append(f"  ({len(res.warnings)} warning(s))")
        lines.extend(f"    ⚠ {w}" for w in res.warnings)
    return "\n".join(lines)


def _describe(op, ir: TimelineIR) -> str:
    def nm(cid):
        c = ir.clip(cid)
        return f"{c.name}" if c else cid

    if op.op == "trim":
        c = ir.clip(op.clip_id)
        if op.op == "trim" and c:
            if op.edge == "out":
                new = f"{c.start}..{c.end + op.delta}"
            else:
                new = f"{c.start + op.delta}..{c.end}"
            return f"trim {nm(op.clip_id)} {op.edge} {op.delta:+d}  →  {c.start}..{c.end} ⇒ {new}"
    if op.op == "move":
        tr = f" track {op.to_track}" if op.to_track else ""
        return f"move {nm(op.clip_id)} → start {op.to_start}{tr}"
    if op.op == "insert":
        return f"insert {op.src_ref} on {op.track} @ {op.at_start} (dur {op.duration})"
    if op.op == "delete":
        return f"delete {nm(op.clip_id)}{' (ripple)' if op.ripple else ''}"
    if op.op == "set_property":
        return f"set {nm(op.clip_id)}.{op.key} = {op.value!r}"
    if op.op == "add_marker":
        return f"marker {op.color} on {nm(op.clip_id)} @+{op.frame} {op.name!r}".rstrip()
    if op.op == "add_transition":
        return f"{op.type} ({op.duration}f) on {nm(op.clip_id)} {op.edge} edge"
    if op.op == "split":
        return f"split {nm(op.clip_id)} @ {op.at_frame}"
    if op.op == "cut_range":
        dur = op.end_frame - op.start_frame
        how = "ripple" if op.ripple else "leave gap"
        return f"cut {nm(op.clip_id)} [{op.start_frame}..{op.end_frame}) {dur}f ({how})"
    if op.op == "ripple":
        scope = op.track or "all tracks"
        return f"ripple {scope} from {op.from_frame} by {op.delta:+d}"
    if op.op == "import_audio":
        src = op.sfx and f"sfx:{op.sfx}" or op.src_ref
        dur = f" (dur {op.duration})" if op.duration else ""
        return f"import_audio {src} on {op.track} @ {op.at_start}{dur}"
    if op.op == "add_track":
        extra = f" ({op.audio_type})" if op.kind == "audio" else ""
        return f"add_track {op.kind}{extra}"
    if op.op == "rename_track":
        return f"rename_track {op.track} → {op.name!r}"
    if op.op == "create_bin":
        loc = f" under {op.parent!r}" if op.parent else ""
        return f"create_bin {op.name!r}{loc}"
    if op.op == "move_to_bin":
        return f"move_to_bin {nm(op.clip_id)} → {op.bin!r}"
    if op.op == "retime":
        pct = op.speed_percent
        how = ("freeze-frame" if pct == 0
               else f"{pct:g}% speed" + (" (reverse)" if pct < 0 else ""))
        return f"retime {nm(op.clip_id)} → {how}"
    if op.op == "set_enabled":
        return f"{'enable' if op.enabled else 'disable'} {nm(op.clip_id)}"
    return op.op
