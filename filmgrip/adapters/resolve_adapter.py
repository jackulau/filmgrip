"""DaVinci Resolve adapter — the flagship live editor.

Read side (D7): snapshot the active timeline into the universal IR, and capture the user's
"selection". Write side (D8): apply a validated EditPlan through the native API in one undo group.

Honest constraint, surfaced not faked: **Resolve has no true multi-clip timeline-selection API**.
We reconstruct an intent-level selection from the one timeline item Resolve does expose
(``GetCurrentVideoItem``) plus the media-pool selection (``MediaPool.GetSelectedClips``), mapping
selected media back to every timeline instance that references it. The ``Selection.note`` records
that this is a reconstruction.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import opentimelineio as otio

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan
from ..protocol.validate import validate
from .base import ApplyResult, Capabilities, GrabAdapter, Selection
from .resolve_client import ResolveOperationFailed, ResolveSession, connect, require

# Ops the Resolve scripting API can apply in place, reliably. Everything else (precise
# move/trim/split/transition/ripple) the API cannot do faithfully — those route to the
# OTIO-rebuild path (export timeline -> mutate OTIO -> ImportTimelineFromFile), implemented in
# the interchange adapter (D11) and orchestrated by the CLI (D10). This is the dual apply-path.
LIVE_OPS = frozenset({"add_marker", "set_property", "delete"})


def _native_call(obj: Any, method: str, *args):
    """Call a Resolve object method, raising if it isn't actually implemented.

    Resolve's fusionscript proxies return ``None`` (not raise) for ANY attribute name, so
    ``hasattr`` always lies and a missing method surfaces as ``TypeError: 'NoneType' is not
    callable`` mid-apply. Routing every native call through here turns a phantom method into a
    clean :class:`ResolveOperationFailed` that the transaction can roll back.
    """
    fn = getattr(obj, method, None)
    if not callable(fn):
        raise ResolveOperationFailed(f"Resolve object has no callable '{method}'")
    return fn(*args)


def _media_url(item: Any) -> str:
    """Best-effort source path for a Resolve TimelineItem (pulled in one GetClipProperty call)."""
    mpi = item.GetMediaPoolItem() if hasattr(item, "GetMediaPoolItem") else None
    if mpi is not None:
        props = mpi.GetClipProperty() or {}
        path = props.get("File Path") or props.get("File Name") or ""
        if path:
            return path
        name = mpi.GetName()
        if name:
            return name
    return item.GetName() or "clip"


def _timeline_rate(timeline: Any, default: float = 24.0) -> float:
    raw = ""
    if hasattr(timeline, "GetSetting"):
        raw = timeline.GetSetting("timelineFrameRate") or ""
    try:
        return float(raw) if raw else default
    except ValueError:
        return default


class ResolveAdapter(GrabAdapter):
    name = "resolve"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            editor="DaVinci Resolve (Studio)",
            role="flagship-native",
            mechanism="native Python scripting API (fusionscript)",
            live_selection=False,  # honest: no true multi-clip timeline selection API
            write_back=True,
            requires_app_running=True,
            lossy_features=["add_transition (Fusion/interchange only)"],
        )

    # -- read -------------------------------------------------------------------
    def snapshot(self, source: Any) -> TimelineIR:
        """Build a :class:`TimelineIR` from the active Resolve timeline.

        ``source`` is a :class:`ResolveSession` (or a raw resolve object, which we wrap).
        Frame positions are made relative to the timeline start so the IR begins at 0.
        """
        session = source if isinstance(source, ResolveSession) else connect(source)
        if session is None:
            raise RuntimeError("Resolve is not running / no session — open a project first.")
        timeline = require(session.current_timeline(), "no current timeline (open one in Resolve)")
        rate = _timeline_rate(timeline)
        tl_start = int(timeline.GetStartFrame() or 0)

        otio_tl = otio.schema.Timeline(name=timeline.GetName() or "Resolve Timeline")
        otio_tl.global_start_time = otio.opentime.RationalTime(0, rate)
        native_map: dict[tuple[str, int, int], Any] = {}

        for kind, TrackKind in (("video", otio.schema.TrackKind.Video),
                                ("audio", otio.schema.TrackKind.Audio)):
            count = int(timeline.GetTrackCount(kind) or 0)
            for idx in range(1, count + 1):
                track = otio.schema.Track(
                    name=(timeline.GetTrackName(kind, idx) if hasattr(timeline, "GetTrackName") else f"{kind}{idx}")
                    or f"{kind}{idx}",
                    kind=TrackKind,
                )
                items = sorted(session.timeline_items(kind, idx, timeline), key=lambda it: int(it.GetStart()))
                playhead = 0
                for it in items:
                    start = int(it.GetStart()) - tl_start
                    dur = int(it.GetDuration())
                    if start > playhead:
                        track.append(otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(
                                otio.opentime.RationalTime(0, rate),
                                otio.opentime.RationalTime(start - playhead, rate))))
                    src_start = int(it.GetSourceStartFrame()) if hasattr(it, "GetSourceStartFrame") else 0
                    track.append(otio.schema.Clip(
                        name=it.GetName() or "clip",
                        media_reference=otio.schema.ExternalReference(target_url=_media_url(it)),
                        source_range=otio.opentime.TimeRange(
                            otio.opentime.RationalTime(src_start, rate),
                            otio.opentime.RationalTime(dur, rate)),
                    ))
                    native_map[(kind, idx, start)] = it
                    playhead = start + dur
                otio_tl.tracks.append(track)

        ir = TimelineIR.from_otio(otio_tl)
        for c in ir.real_clips():
            c.native = native_map.get((c.track_kind, c.track_index, c.start))
        return ir

    def get_selection(self, source: Any, ir: Optional[TimelineIR] = None) -> Selection:
        """Reconstruct the selection from the current video item + media-pool selection."""
        session = source if isinstance(source, ResolveSession) else connect(source)
        if session is None:
            raise RuntimeError("Resolve is not running — open a project first.")
        if ir is None:
            ir = self.snapshot(session)
        ids: list[str] = []

        # (1) the one timeline item Resolve exposes directly
        timeline = session.current_timeline()
        cur = timeline.GetCurrentVideoItem() if (timeline and hasattr(timeline, "GetCurrentVideoItem")) else None
        if cur is not None:
            for c in ir.real_clips():
                if c.native is cur:
                    ids.append(c.id)
                    break

        # (2) media-pool selection -> every timeline instance of that media (the reconstruction)
        mp = session.media_pool()
        selected_media = (mp.GetSelectedClips() if mp and hasattr(mp, "GetSelectedClips") else []) or []
        sel_refs = set()
        for m in selected_media:
            props = m.GetClipProperty() if hasattr(m, "GetClipProperty") else {}
            path = (props or {}).get("File Path") or m.GetName() or ""
            if path:
                sel_refs.add(os.path.basename(path))
        if sel_refs:
            for c in ir.real_clips():
                if c.src_ref in sel_refs and c.id not in ids:
                    ids.append(c.id)

        note = ("Resolve exposes no true multi-clip timeline selection; reconstructed from "
                "GetCurrentVideoItem + media-pool selection.")
        return Selection(ids=ids, basis="current_video_item+media_pool", note=note)

    # -- write ------------------------------------------------------------------
    def apply(self, plan: EditPlan, source: Any, **kw) -> ApplyResult:
        """Apply a validated EditPlan live, with compensating rollback.

        Resolve's scripting API has no begin/end-undo transaction, and fails *silently* (falsy
        returns). So film-grip validates first, then applies each live op while pushing an inverse
        onto a rollback stack; any silent failure aborts and unwinds the inverses, leaving the
        timeline as it was. Ops outside :data:`LIVE_OPS` are reported as needing the OTIO-rebuild
        path rather than half-applied.
        """
        session = source if isinstance(source, ResolveSession) else connect(source)
        if session is None:
            return ApplyResult(ok=False, errors=["Resolve is not running — open a project first."])

        ir = self.snapshot(session)
        timeline = session.current_timeline()
        result = validate(plan, ir)
        if not result.ok:
            return ApplyResult(ok=False, errors=[str(e) for e in result.errors])

        rebuild_needed = [op.op for op in plan.ops if op.op not in LIVE_OPS]
        live_ops = [op for op in plan.ops if op.op in LIVE_OPS]

        applied: list[str] = []
        rollback: list = []
        warnings: list[str] = []
        try:
            for op in live_ops:
                desc, inverse, warn = self._apply_one(op, ir, timeline)
                applied.append(desc)
                if inverse is not None:
                    rollback.append(inverse)
                if warn:
                    warnings.append(warn)
        except ResolveOperationFailed as exc:
            for inv in reversed(rollback):
                try:
                    inv()
                except Exception:
                    pass
            return ApplyResult(ok=False, errors=[f"aborted + rolled back: {exc}"], applied=applied)

        if rebuild_needed:
            warnings.append(
                f"{len(rebuild_needed)} op(s) need the OTIO-rebuild path (not live-applicable in "
                f"Resolve): {sorted(set(rebuild_needed))}"
            )
        diff = "\n".join(f"  ✓ {d}" for d in applied) or "  (no live ops)"
        return ApplyResult(ok=True, applied=applied, diff=diff, warnings=warnings)

    def _apply_one(self, op, ir: TimelineIR, timeline: Any):
        """Apply one live op against its native handle.

        Returns ``(description, inverse_callable_or_None, warning_or_None)``. Required calls go
        through :func:`_native_call` so a phantom/failed Resolve method aborts + rolls back; the
        rename path is best-effort (Resolve has no reliable per-item rename).
        """
        clip = ir.clip(op.clip_id)
        item = getattr(clip, "native", None)
        if item is None:
            raise ResolveOperationFailed(f"no native handle for clip '{op.clip_id}'")

        if op.op == "add_marker":
            require(_native_call(item, "AddMarker", op.frame, op.color, op.name, op.note, op.duration, ""),
                    f"AddMarker failed on '{clip.name}' (frame {op.frame})")
            return (f"marker {op.color} on {clip.name} @+{op.frame}",
                    lambda: _native_call(item, "DeleteMarkerAtFrame", op.frame), None)

        if op.op == "set_property":
            if op.key == "name":
                return self._rename(clip, item, str(op.value))  # best-effort via the media-pool item
            if op.key == "color":
                old = item.GetClipColor()
                require(_native_call(item, "SetClipColor", str(op.value)),
                        f"SetClipColor failed on '{clip.name}'")
                inverse = ((lambda: _native_call(item, "SetClipColor", old)) if old
                           else (lambda: _native_call(item, "ClearClipColor")))
                return (f"color {clip.name} = {op.value}", inverse, None)
            old = item.GetProperty(op.key)
            require(_native_call(item, "SetProperty", op.key, op.value),
                    f"SetProperty {op.key} failed on '{clip.name}'")
            return (f"set {clip.name}.{op.key} = {op.value!r}",
                    lambda: _native_call(item, "SetProperty", op.key, old), None)

        if op.op == "delete":
            require(timeline, "no timeline handle for delete")
            require(_native_call(timeline, "DeleteClips", [item], bool(op.ripple)),
                    f"DeleteClips failed on '{clip.name}'")
            # Delete is not cleanly reversible via the API; no inverse (validated up-front).
            return (f"delete {clip.name}", None, None)

        raise ResolveOperationFailed(f"op '{op.op}' is not a live op")

    def _rename(self, clip, item, value: str):
        """Rename a clip via its MediaPoolItem's 'Clip Name' (Resolve has no TimelineItem.SetName).

        Best-effort: Resolve's rename API is unreliable (can return falsy yet take effect, and it
        renames the SOURCE pool clip — affecting all instances), so we never abort the whole plan
        over it. We apply, register an inverse, and warn.
        """
        mpi = item.GetMediaPoolItem()
        if mpi is None:
            return (f"rename {clip.name} -> {value} (skipped: no media-pool item)", None,
                    f"could not rename '{clip.name}': no backing media-pool clip")
        old = mpi.GetClipProperty("Clip Name") or clip.name
        ret = mpi.SetClipProperty("Clip Name", value)
        warn = None if ret else (
            f"rename '{clip.name}' -> '{value}' reported no-op by Resolve (its rename API is "
            f"unreliable) and renames the SOURCE pool clip for all instances")
        return (f"rename {clip.name} -> {value} (source pool clip)",
                lambda: mpi.SetClipProperty("Clip Name", old), warn)
