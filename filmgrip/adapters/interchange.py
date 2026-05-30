"""OTIO interchange adapter — the universal OTIO-rebuild write path.

This is how film-grip reaches every file-based editor (Final Cut Pro, Premiere via FCP7 XML,
Avid via AAF, and any NLE that imports OTIO/EDL) *without a live API*: read an exported
interchange file into the IR, mutate the OTIO graph with a validated EditPlan, and write a NEW
interchange file the editor re-imports. It's also the rebuild path Resolve-live falls back to for
ops its scripting API can't do precisely (trim/delete/split).

Interchange is lossy by a documented amount. The adapter applies only ops that survive the OTIO
round-trip (trim/delete/split/add_marker/set_property) and flags everything else — and warns when
the target format degrades existing features (transitions in EDL/FCP7 XML, effects in AAF).
"""
from __future__ import annotations

import os
from typing import Any, Optional

import opentimelineio as otio

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan
from ..protocol.validate import validate
from ..serialize.fgx import parse_track_code
from .base import ApplyResult, Capabilities, GrabAdapter, Selection

# extension -> OTIO adapter name
FORMAT_BY_EXT = {
    ".fcpxml": "fcp_xml", ".xml": "fcp_xml",
    ".edl": "cmx_3600",
    ".aaf": "AAF",
    ".otio": "otio_json", ".otioz": "otioz",
}

# Ops the OTIO-rebuild path applies on the timeline graph. This is the universal write path —
# it's also what the live Resolve adapter falls back to (export EXPORT_OTIO -> mutate -> import),
# where structural repositioning (move/insert/ripple) MUST take effect, not just warn. The OTIO
# model expresses absolute position as ordered items + gaps, so move/insert/ripple are exact here.
# Only add_transition stays out — transition fidelity is format-dependent and best done live.
REBUILD_OPS = frozenset({
    "trim", "delete", "split", "add_marker", "set_property", "move", "insert", "ripple",
})

_MARKER_COLOR = {
    "red": otio.schema.MarkerColor.RED, "green": otio.schema.MarkerColor.GREEN,
    "blue": otio.schema.MarkerColor.BLUE, "cyan": otio.schema.MarkerColor.CYAN,
    "magenta": otio.schema.MarkerColor.MAGENTA, "yellow": otio.schema.MarkerColor.YELLOW,
    "pink": otio.schema.MarkerColor.PINK, "purple": otio.schema.MarkerColor.PURPLE,
    "orange": otio.schema.MarkerColor.ORANGE, "white": otio.schema.MarkerColor.WHITE,
}

# Features the IR may contain that degrade when written to a given target format.
_FORMAT_LOSSY = {
    "cmx_3600": ["transitions", "multiple video tracks", "effects", "markers"],
    "fcp_xml": ["transitions (fidelity varies by importer)", "speed/retime"],
    "AAF": ["effects", "transitions"],
}


def _rt(frames: int, rate: float) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, rate)


class OtioMutator:
    """Apply REBUILD_OPS to an IR's backing OTIO graph in place, via each clip's otio handle."""

    def __init__(self, ir: TimelineIR):
        self.ir = ir
        self.rate = ir.rate

    def apply(self, plan: EditPlan) -> tuple[list[str], list[str]]:
        applied: list[str] = []
        warnings: list[str] = []
        for op in plan.ops:
            if op.op not in REBUILD_OPS:
                warnings.append(f"op '{op.op}' not applied in interchange rebuild "
                                f"(needs a live editor or repositioning support)")
                continue
            applied.append(getattr(self, f"_{op.op}")(op))
        return applied, warnings

    def _clip_and_item(self, op):
        clip = self.ir.clip(op.clip_id)
        return clip, clip.otio

    # -- placement helpers (shared by move/insert/ripple) -----------------------
    def _gap(self, dur: int) -> otio.schema.Gap:
        return otio.schema.Gap(
            source_range=otio.opentime.TimeRange(_rt(0, self.rate), _rt(dur, self.rate)))

    @staticmethod
    def _positions(track) -> list[tuple]:
        """``(child, start_frame, duration_frame)`` per item; transitions carry ``None`` (no time)."""
        out: list[tuple] = []
        for ch in track:
            if isinstance(ch, otio.schema.Transition):
                out.append((ch, None, None))
                continue
            rng = ch.range_in_parent()
            out.append((ch, int(round(rng.start_time.to_frames())),
                        int(round(rng.duration.to_frames()))))
        return out

    def _otio_track(self, code: str):
        """Resolve a track code ('v1'/'a2') to its OTIO track in timeline order."""
        kind, idx = parse_track_code(code)
        n = 0
        for tr in self.ir.timeline.tracks:
            is_video = (tr.kind or "Video").lower().startswith("v")
            if (kind == "video") == is_video:
                n += 1
                if n == idx:
                    return tr
        raise ValueError(f"track '{code}' not found")

    def _place_at(self, track, item, at_frame: int, item_dur: int) -> None:
        """Place ``item`` so it starts at ``at_frame`` — splitting a free gap or padding past end.

        The validator guarantees the destination span doesn't overlap a clip, so the target region
        is covered by a single gap or lies at/after the track end. Anything else is a real
        inconsistency and is raised rather than written wrong.
        """
        positions = self._positions(track)
        track_end = max((s + d for _, s, d in positions if s is not None), default=0)
        for i, (ch, start, dur) in enumerate(positions):
            if start is None:
                continue
            if isinstance(ch, otio.schema.Gap) and start <= at_frame \
                    and at_frame + item_dur <= start + dur:
                left = at_frame - start
                right = (start + dur) - (at_frame + item_dur)
                pieces = ([self._gap(left)] if left > 0 else []) + [item] \
                    + ([self._gap(right)] if right > 0 else [])
                del track[i]
                for off, piece in enumerate(pieces):
                    track.insert(i + off, piece)
                return
        if at_frame >= track_end:
            if at_frame > track_end:
                track.append(self._gap(at_frame - track_end))
            track.append(item)
            return
        raise ValueError(
            f"cannot place item at frame {at_frame}: no single free gap of {item_dur} frames")

    def _move(self, op) -> str:
        clip, item = self._clip_and_item(op)
        item_dur = clip.duration
        src_track = item.parent()
        sidx = list(src_track).index(item)
        del src_track[sidx]
        src_track.insert(sidx, self._gap(item_dur))  # leave a gap so source neighbours hold position
        code = op.to_track or f"{clip.track_kind[0]}{clip.track_index}"
        self._place_at(self._otio_track(code), item, op.to_start, item_dur)
        return f"move {clip.name} -> {op.to_start}{(' ' + op.to_track) if op.to_track else ''}"

    def _insert(self, op) -> str:
        new = otio.schema.Clip(
            name=op.src_ref,
            media_reference=otio.schema.ExternalReference(target_url=op.src_ref),
            source_range=otio.opentime.TimeRange(
                _rt(op.source_in, self.rate), _rt(op.duration, self.rate)))
        self._place_at(self._otio_track(op.track), new, op.at_start, op.duration)
        return f"insert {op.src_ref} on {op.track} @ {op.at_start} (dur {op.duration})"

    def _ripple(self, op) -> str:
        targets = [self._otio_track(op.track)] if op.track else list(self.ir.timeline.tracks)
        for tr in targets:
            positions = self._positions(tr)
            first = next((i for i, (ch, s, d) in enumerate(positions)
                          if s is not None and s >= op.from_frame), None)
            if first is None:
                continue
            if op.delta > 0:
                tr.insert(first, self._gap(op.delta))
            elif op.delta < 0:
                j = first - 1
                gdur = positions[j][2] if j >= 0 else 0
                if j >= 0 and isinstance(positions[j][0], otio.schema.Gap) and gdur >= -op.delta:
                    if gdur == -op.delta:
                        del tr[j]
                    else:
                        tr[j] = self._gap(gdur + op.delta)  # delta < 0 -> shrink the gap
                else:
                    raise ValueError(
                        f"ripple {op.delta:+d} needs {-op.delta} frames of gap before frame "
                        f"{op.from_frame}; none available")
        return f"ripple {op.track or 'all'} from {op.from_frame} by {op.delta:+d}"

    def _trim(self, op) -> str:
        clip, item = self._clip_and_item(op)
        sr = item.source_range
        if op.edge == "out":
            item.source_range = otio.opentime.TimeRange(
                sr.start_time, _rt(clip.duration + op.delta, self.rate))
        else:
            item.source_range = otio.opentime.TimeRange(
                sr.start_time + _rt(op.delta, self.rate), _rt(clip.duration - op.delta, self.rate))
        return f"trim {clip.name} {op.edge} {op.delta:+d}"

    def _delete(self, op) -> str:
        clip, item = self._clip_and_item(op)
        track = item.parent()
        idx = list(track).index(item)
        if op.ripple:
            del track[idx]
        else:
            track[idx] = otio.schema.Gap(
                source_range=otio.opentime.TimeRange(_rt(0, self.rate), _rt(clip.duration, self.rate)))
        return f"delete {clip.name}{' (ripple)' if op.ripple else ''}"

    def _split(self, op) -> str:
        clip, item = self._clip_and_item(op)
        track = item.parent()
        idx = list(track).index(item)
        off = op.at_frame - clip.start
        sr = item.source_range
        item.source_range = otio.opentime.TimeRange(sr.start_time, _rt(off, self.rate))
        tail = item.deepcopy()
        tail.source_range = otio.opentime.TimeRange(
            sr.start_time + _rt(off, self.rate), _rt(clip.duration - off, self.rate))
        track.insert(idx + 1, tail)
        return f"split {clip.name} @ {op.at_frame}"

    def _add_marker(self, op) -> str:
        clip, item = self._clip_and_item(op)
        color = _MARKER_COLOR.get(op.color.lower(), otio.schema.MarkerColor.RED)
        item.markers.append(otio.schema.Marker(
            name=op.name or "marker", color=color,
            marked_range=otio.opentime.TimeRange(
                _rt(clip.source_start + op.frame, self.rate), _rt(op.duration, self.rate))))
        return f"marker {op.color} on {clip.name} @+{op.frame}"

    def _set_property(self, op) -> str:
        clip, item = self._clip_and_item(op)
        meta = item.metadata.setdefault("filmgrip", {})
        meta[op.key] = op.value
        return f"set {clip.name}.{op.key} = {op.value!r}"


class InterchangeAdapter(GrabAdapter):
    name = "interchange"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            editor="Interchange (FCPXML / EDL / AAF / OTIO)",
            role="interchange",
            mechanism="OpenTimelineIO adapters (read file -> mutate -> write new file)",
            live_selection=False,
            write_back=True,
            requires_app_running=False,
            lossy_features=["transitions", "effects", "speed (format-dependent)"],
        )

    @staticmethod
    def _fmt_for(path: str, override: Optional[str] = None) -> str:
        if override:
            return override
        ext = os.path.splitext(path)[1].lower()
        if ext not in FORMAT_BY_EXT:
            raise ValueError(f"unknown interchange extension '{ext}'")
        return FORMAT_BY_EXT[ext]

    def snapshot(self, source: Any, *, fmt: Optional[str] = None) -> TimelineIR:
        timeline = otio.adapters.read_from_file(source, adapter_name=self._fmt_for(source, fmt))
        return TimelineIR.from_otio(timeline)

    def get_selection(self, source: Any, ir: Optional[TimelineIR] = None) -> Selection:
        if ir is None:
            ir = self.snapshot(source)
        return Selection(
            ids=[c.id for c in ir.real_clips()],
            basis="interchange_export",
            note="interchange has no live selection; the whole exported timeline is the context.")

    def apply(self, plan: EditPlan, source: Any, *, out_path: Optional[str] = None,
              out_format: Optional[str] = None, **kw) -> ApplyResult:
        ir = self.snapshot(source)
        res = validate(plan, ir)
        if not res.ok:
            return ApplyResult(ok=False, errors=[str(e) for e in res.errors])

        applied, warnings = OtioMutator(ir).apply(plan)

        out = out_path or source
        fmt = self._fmt_for(out, out_format)
        warnings.extend(self._lossy_warnings(ir, fmt))
        try:
            otio.adapters.write_to_file(ir.timeline, out, adapter_name=fmt)
        except Exception as exc:
            # The target format genuinely can't represent this timeline (e.g. EDL is single-track
            # cuts-only). Refuse cleanly rather than writing a corrupt/garbage file.
            return ApplyResult(ok=False, applied=applied, warnings=warnings,
                               errors=[f"cannot write {fmt}: {exc}"])

        diff = "\n".join(f"  ✓ {d}" for d in applied) or "  (no rebuild ops)"
        diff += f"\n  → wrote {fmt} to {out}"
        return ApplyResult(ok=True, applied=applied, diff=diff, warnings=warnings)

    @staticmethod
    def _lossy_warnings(ir: TimelineIR, fmt: str) -> list[str]:
        warnings: list[str] = []
        lossy = _FORMAT_LOSSY.get(fmt, [])
        has_transition = any(c.kind == "transition" for c in ir.clips)
        if has_transition and any("transition" in f for f in lossy):
            warnings.append(f"timeline has transition(s); fidelity to '{fmt}' varies by importer "
                            f"(lossy per OTIO feature matrix)")
        if fmt == "cmx_3600" and ir.track_count("video") > 1:
            warnings.append("EDL (cmx_3600) is cuts-only on a single track; extra tracks/effects "
                            "will not survive")
        return warnings
