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
from .base import ApplyResult, Capabilities, GrabAdapter, Selection

# extension -> OTIO adapter name
FORMAT_BY_EXT = {
    ".fcpxml": "fcp_xml", ".xml": "fcp_xml",
    ".edl": "cmx_3600",
    ".aaf": "AAF",
    ".otio": "otio_json", ".otioz": "otioz",
}

# Ops that survive an OTIO interchange round-trip cleanly. The rest are surfaced as warnings
# (move/insert/ripple need absolute repositioning the sequential model handles poorly here;
# add_transition fidelity is format-dependent — better done in the live editor).
REBUILD_OPS = frozenset({"trim", "delete", "split", "add_marker", "set_property"})

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
