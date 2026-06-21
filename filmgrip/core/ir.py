"""The universal intermediate representation: an OpenTimelineIO timeline + a stable-ID map.

Every editor's timeline is normalized into an OTIO graph; :class:`TimelineIR` is the single
authoritative in-memory object and everything else (FGX serialization, the EditPlan validator,
adapters) is a projection of it. The IR keeps the raw OTIO timeline so a compact view is always
losslessly re-expandable, and assigns every clip/gap/transition a stable :mod:`~filmgrip.core.idmap`
ID so Claude can reference clips without indices or names that drift.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import opentimelineio as otio

from .idmap import IdMap


@dataclass(slots=True)
class Clip:
    """A flattened view of one OTIO item with film-grip's stable ID attached.

    ``slots=True``: a Clip is minted once per item on every timeline snapshot (``from_otio`` runs each
    turn), so the per-instance dict overhead is real churn on large timelines. Slots cut construction +
    attribute-access cost and memory; no code sets an attribute outside the declared fields.
    """

    track_kind: str            # "video" | "audio"
    track_index: int           # 1-based, in track order
    name: str
    start: int                 # timeline frame, relative to timeline start
    duration: int              # frames
    source_start: int          # frame offset into the source media
    src_ref: str               # media basename / id (never inlined bytes)
    kind: str = "clip"         # "clip" | "gap" | "transition"
    id: str = ""
    otio: Any = field(default=None, repr=False)   # the backing OTIO item (for mutation)
    native: Any = field(default=None, repr=False)  # editor handle (Resolve TimelineItem), if live

    @property
    def end(self) -> int:
        return self.start + self.duration


def _rate_of(timeline: otio.schema.Timeline) -> float:
    if timeline.global_start_time is not None:
        return float(timeline.global_start_time.rate)
    for track in timeline.tracks:
        for child in track:
            sr = getattr(child, "source_range", None)
            if sr is not None:
                return float(sr.duration.rate)
    return 24.0


def media_ref_string(item: Any) -> str:
    """A short, stable media reference: source basename, else media name, else item name."""
    mr = getattr(item, "media_reference", None)
    if mr is not None:
        url = getattr(mr, "target_url", None)
        if url:
            # Inline basename: media URLs are posix/file paths; rsplit avoids posixpath.basename's
            # call overhead, which the profile flagged as a per-clip hot spot in the IR build.
            return url.rsplit("/", 1)[-1]
        if getattr(mr, "name", None):
            return mr.name
    return getattr(item, "name", "") or ""


class TimelineIR:
    """Wraps an OTIO ``Timeline`` and indexes its items by stable ID."""

    def __init__(self, timeline: otio.schema.Timeline):
        self.timeline = timeline
        self.rate: float = _rate_of(timeline)
        self.idmap = IdMap()
        self.clips: list[Clip] = []
        self._index()

    # -- construction -----------------------------------------------------------
    @classmethod
    def from_otio(cls, timeline: otio.schema.Timeline) -> "TimelineIR":
        return cls(timeline)

    @classmethod
    def from_otio_file(cls, path: str) -> "TimelineIR":
        timeline = otio.adapters.read_from_file(path)
        return cls(timeline)

    def _index(self) -> None:
        self.clips.clear()
        video_n = 0
        audio_n = 0
        for track in self.timeline.tracks:
            kind = (track.kind or "Video").lower()
            if kind.startswith("v"):
                video_n += 1
                tk, ti = "video", video_n
            else:
                audio_n += 1
                tk, ti = "audio", audio_n
            frame = 0
            for child in track:
                if isinstance(child, otio.schema.Transition):
                    in_f = int(round(child.in_offset.to_frames())) if child.in_offset else 0
                    out_f = int(round(child.out_offset.to_frames())) if child.out_offset else 0
                    clip = Clip(
                        track_kind=tk, track_index=ti, name=child.name or "transition",
                        start=frame, duration=in_f + out_f, source_start=0,
                        src_ref=child.transition_type or "transition", kind="transition", otio=child,
                    )
                    self._add(clip)
                    continue  # transitions overlap neighbors; don't advance the playhead
                try:
                    rng = child.range_in_parent()
                    start = int(round(rng.start_time.to_frames()))
                    dur = int(round(rng.duration.to_frames()))
                except Exception:
                    start = frame
                    dur = int(round(child.duration().to_frames())) if hasattr(child, "duration") else 0
                if isinstance(child, otio.schema.Gap):
                    clip = Clip(
                        track_kind=tk, track_index=ti, name=child.name or "gap",
                        start=start, duration=dur, source_start=0, src_ref="gap", kind="gap", otio=child,
                    )
                else:  # Clip
                    src_start = 0
                    if getattr(child, "source_range", None) is not None:
                        src_start = int(round(child.source_range.start_time.to_frames()))
                    clip = Clip(
                        track_kind=tk, track_index=ti, name=child.name or "clip",
                        start=start, duration=dur, source_start=src_start,
                        src_ref=media_ref_string(child), kind="clip", otio=child,
                    )
                self._add(clip)
                frame = start + dur

    def _add(self, clip: Clip) -> None:
        self.idmap.mint(clip)
        self.clips.append(clip)

    # -- queries ----------------------------------------------------------------
    def clip(self, cid: str) -> Optional[Clip]:
        return self.idmap.get(cid)

    def reresolve(self, cid: str, new_ir: "TimelineIR") -> Optional[Clip]:
        return self.idmap.reresolve(cid, new_ir)

    def clips_on(self, track_kind: str, track_index: int) -> list[Clip]:
        return [c for c in self.clips if c.track_kind == track_kind and c.track_index == track_index]

    def real_clips(self) -> list[Clip]:
        """Clips only (no gaps/transitions)."""
        return [c for c in self.clips if c.kind == "clip"]

    def neighbors(self, cid: str, hops: int = 1) -> list[Clip]:
        """Clips within ``hops`` positions of ``cid`` on the same track (excluding ``cid``)."""
        target = self.clip(cid)
        if target is None:
            return []
        track = sorted(self.clips_on(target.track_kind, target.track_index), key=lambda c: c.start)
        try:
            pos = next(i for i, c in enumerate(track) if c.id == cid)
        except StopIteration:
            return []
        lo, hi = max(0, pos - hops), min(len(track), pos + hops + 1)
        return [c for c in track[lo:hi] if c.id != cid]

    @property
    def duration(self) -> int:
        return max((c.end for c in self.clips), default=0)

    def track_count(self, track_kind: str) -> int:
        return max((c.track_index for c in self.clips if c.track_kind == track_kind), default=0)

    # -- output -----------------------------------------------------------------
    def to_otio_file(self, path: str) -> None:
        otio.adapters.write_to_file(self.timeline, path)

    def reindex(self) -> None:
        """Re-derive the flattened clip list + IDs after the OTIO graph was mutated in place."""
        self.idmap = IdMap()
        self._index()
