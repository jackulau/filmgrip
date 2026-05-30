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
from .base import Capabilities, GrabAdapter, Selection
from .resolve_client import ResolveSession, connect, require


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
