"""In-memory fakes that mirror the DaVinci Resolve scripting object graph.

These duck-type the real Resolve API method names (``GetItemListInTrack``, ``GetCurrentVideoItem``,
``GetClipProperty``, ``AddMarker``, ``SetProperty`` ...) closely enough that the Resolve client,
read adapter (D7) and apply adapter (D8) can be exercised with **no app installed**. Each fake
records the calls made against it so apply tests can assert the exact native call sequence and
that a silent-``False`` return aborts a transaction.
"""
from __future__ import annotations

from typing import Any, Optional


class FakeMediaPoolItem:
    def __init__(self, name: str, props: Optional[dict] = None, media_id: str = "", unique_id: str = ""):
        self._name = name
        self._props = dict(props or {})
        self._media_id = media_id or f"mid-{name}"
        self._unique_id = unique_id or f"uid-{name}"

    def GetName(self) -> str:
        return self._name

    def GetClipProperty(self, name: Optional[str] = None):
        if name is None:
            return dict(self._props)
        return self._props.get(name, "")

    def SetClipProperty(self, name: str, value: Any) -> bool:
        self._props[name] = value
        return True

    def GetMediaId(self) -> str:
        return self._media_id

    def GetUniqueId(self) -> str:
        return self._unique_id

    def GetMetadata(self, key: Optional[str] = None):
        return {} if key is None else ""


class FakeColorGroup:
    """Resolve ColorGroup (``Project.AddColorGroup`` → assigned via ``TimelineItem``)."""

    def __init__(self, name: str):
        self._name = name
        self.calls: list[tuple] = []

    def GetName(self) -> str:
        return self._name


class FakeColorGraph:
    """Resolve 19+ Graph object (``TimelineItem.GetNodeGraph()``) — SetLUT/GetLUT live HERE on
    newer builds, not on the TimelineItem. Mirrors the version split the adapter branches on."""

    def __init__(self, num_nodes: int = 2):
        self._luts: dict[int, str] = {}
        self._num_nodes = num_nodes
        self.calls: list[tuple] = []

    def GetNumNodes(self) -> int:
        return self._num_nodes

    def SetLUT(self, node_index, path) -> bool:
        self.calls.append(("SetLUT", node_index, path))
        self._luts[int(node_index)] = path
        return True

    def GetLUT(self, node_index):
        return self._luts.get(int(node_index), "")


class FakeTimelineItem:
    """A clip on a timeline track. Frame values are timeline-relative ints, like Resolve."""

    def __init__(
        self,
        name: str,
        start: int,
        duration: int,
        *,
        source_start: int = 0,
        src_path: str = "",
        media_item: Optional[FakeMediaPoolItem] = None,
        properties: Optional[dict] = None,
        fail_set: bool = False,
        fail_marker: bool = False,
        color_graph: bool = True,
    ):
        self._name = name
        self._start = int(start)
        self._duration = int(duration)
        self._source_start = int(source_start)
        self._src_path = src_path or f"/media/{name}.mov"
        self._media = media_item or FakeMediaPoolItem(name, {"File Path": self._src_path})
        self._props: dict[str, Any] = dict(properties or {})
        self._markers: dict[float, dict] = {}
        self._color = ""
        self._fail_set = fail_set
        self._fail_marker = fail_marker
        self.calls: list[tuple] = []
        # color: v19+ exposes SetLUT on a Graph object; older builds expose it on the item.
        self._has_graph = color_graph
        self._graph: Optional[FakeColorGraph] = None
        self._cdl: Optional[dict] = None
        self._item_luts: dict[int, str] = {}
        self._color_group: Any = None
        self._versions: list[tuple] = []
        self._current_version: Optional[tuple] = None

    # -- reads ------------------------------------------------------------------
    def GetName(self) -> str:
        return self._name

    def GetStart(self, subframe: bool = False):
        return self._start

    def GetEnd(self, subframe: bool = False):
        return self._start + self._duration

    def GetDuration(self, subframe: bool = False):
        return self._duration

    def GetLeftOffset(self, subframe: bool = False):
        return self._source_start

    def GetSourceStartFrame(self) -> int:
        return self._source_start

    def GetSourceEndFrame(self) -> int:
        return self._source_start + self._duration

    def GetMediaPoolItem(self) -> FakeMediaPoolItem:
        return self._media

    def GetProperty(self, key: Optional[str] = None):
        if key is None:
            return dict(self._props)
        return self._props.get(key)

    def GetMarkers(self) -> dict:
        return dict(self._markers)

    def GetTrackTypeAndIndex(self):
        return [getattr(self, "_track_type", "video"), getattr(self, "_track_index", 1)]

    # -- writes (record + honor silent-failure flags) ---------------------------
    def SetProperty(self, key: str, value: Any) -> bool:
        self.calls.append(("SetProperty", key, value))
        if self._fail_set:
            return False
        self._props[key] = value
        return True

    def AddMarker(self, frame, color, name, note, duration, customData="") -> bool:
        self.calls.append(("AddMarker", frame, color, name, note, duration, customData))
        if self._fail_marker:
            return False
        self._markers[float(frame)] = {
            "color": color,
            "name": name,
            "note": note,
            "duration": duration,
            "customData": customData,
        }
        return True

    def DeleteMarkerAtFrame(self, frame) -> bool:
        self.calls.append(("DeleteMarkerAtFrame", frame))
        return self._markers.pop(float(frame), None) is not None

    def GetClipColor(self) -> str:
        return self._color

    def SetClipColor(self, color: str) -> bool:
        self.calls.append(("SetClipColor", color))
        self._color = color
        return True

    def ClearClipColor(self) -> bool:
        self.calls.append(("ClearClipColor",))
        self._color = ""
        return True

    # -- color: set_cdl / apply_lut ---------------------------------------------
    def SetCDL(self, cdl_map) -> bool:
        # Resolve's SetCDL is write-only (there is intentionally no GetCDL on the fake either).
        self.calls.append(("SetCDL", dict(cdl_map)))
        if self._fail_set:
            return False
        self._cdl = dict(cdl_map)
        return True

    def GetNodeGraph(self, layer_index: int = 0):
        if not self._has_graph:
            return None      # older build: no Graph object → adapter falls back to item.SetLUT
        if self._graph is None:
            self._graph = FakeColorGraph()
        return self._graph

    # older-build fallback path: SetLUT/GetLUT directly on the TimelineItem
    def SetLUT(self, node_index, path) -> bool:
        self.calls.append(("SetLUT", node_index, path))
        if self._fail_set:
            return False
        self._item_luts[int(node_index)] = path
        return True

    def GetLUT(self, node_index):
        return self._item_luts.get(int(node_index), "")

    # color groups + versions (Resolve-live color organization)
    def AssignToColorGroup(self, group) -> bool:
        self.calls.append(("AssignToColorGroup", group.GetName() if hasattr(group, "GetName") else group))
        self._color_group = group
        return True

    def GetColorGroup(self):
        return self._color_group

    def RemoveFromColorGroup(self) -> bool:
        self.calls.append(("RemoveFromColorGroup",))
        self._color_group = None
        return True

    def AddVersion(self, name, version_type) -> bool:
        self.calls.append(("AddVersion", name, version_type))
        self._versions.append((name, int(version_type)))
        return True

    def LoadVersionByName(self, name, version_type) -> bool:
        self.calls.append(("LoadVersionByName", name, version_type))
        self._current_version = (name, int(version_type))
        return True

    def DeleteVersionByName(self, name, version_type) -> bool:
        self.calls.append(("DeleteVersionByName", name, version_type))
        self._versions = [v for v in self._versions if v != (name, int(version_type))]
        return True

    def GetVersionNameList(self, version_type):
        return [n for (n, t) in self._versions if t == int(version_type)]

    # NOTE: real Resolve TimelineItem has NO SetName — renaming goes through the MediaPoolItem's
    # "Clip Name" property. The fake omits SetName deliberately so tests match the live API.


class FakeTimeline:
    def __init__(self, name: str = "Timeline 1", start_frame: int = 0):
        self._name = name
        self._start = start_frame
        self._tracks: dict[tuple[str, int], list[FakeTimelineItem]] = {}
        self._track_names: dict[tuple[str, int], str] = {}
        self._markers: dict[float, dict] = {}
        self._current_video_item: Optional[FakeTimelineItem] = None
        self.calls: list[tuple] = []

    def add_track(self, track_type: str, index: int, items: list[FakeTimelineItem], name: str = "") -> None:
        for it in items:
            it._track_type = track_type
            it._track_index = index
        self._tracks[(track_type, index)] = items
        self._track_names[(track_type, index)] = name or f"{track_type[0].upper()}{index}"
        if track_type == "video" and self._current_video_item is None and items:
            self._current_video_item = items[0]

    # -- reads ------------------------------------------------------------------
    def GetName(self) -> str:
        return self._name

    def GetStartFrame(self) -> int:
        return self._start

    def GetEndFrame(self) -> int:
        ends = [it.GetEnd() for items in self._tracks.values() for it in items]
        return max(ends) if ends else self._start

    def GetStartTimecode(self) -> str:
        return "01:00:00:00"

    def GetTrackCount(self, track_type: str) -> int:
        return max([idx for (t, idx) in self._tracks if t == track_type], default=0)

    def GetItemListInTrack(self, track_type: str, index: int) -> list[FakeTimelineItem]:
        return list(self._tracks.get((track_type, index), []))

    def GetTrackName(self, track_type: str, index: int) -> str:
        return self._track_names.get((track_type, index), "")

    def GetCurrentVideoItem(self) -> Optional[FakeTimelineItem]:
        return self._current_video_item

    def GetMarkers(self) -> dict:
        return dict(self._markers)

    def GetSetting(self, name: str) -> str:
        return {"timelineFrameRate": "24"}.get(name, "")

    def GetCurrentTimecode(self) -> str:
        return "01:00:00:00"

    # -- writes -----------------------------------------------------------------
    def AddMarker(self, frame, color, name, note, duration, customData="") -> bool:
        self.calls.append(("AddMarker", frame, color, name, note, duration, customData))
        self._markers[float(frame)] = {
            "color": color,
            "name": name,
            "note": note,
            "duration": duration,
            "customData": customData,
        }
        return True

    def SetName(self, name: str) -> bool:
        self._name = name
        return True

    def DeleteClips(self, items, ripple=False) -> bool:
        self.calls.append(("DeleteClips", [i.GetName() for i in items], ripple))
        for (key, lst) in self._tracks.items():
            self._tracks[key] = [i for i in lst if i not in items]
        return True

    def AddTrack(self, track_type: str, sub_track_type=None) -> bool:
        self.calls.append(("AddTrack", track_type, sub_track_type))
        idx = self.GetTrackCount(track_type) + 1
        self._tracks[(track_type, idx)] = []
        self._track_names[(track_type, idx)] = f"{track_type[0].upper()}{idx}"
        return True

    def SetTrackName(self, track_type: str, index: int, name: str) -> bool:
        self.calls.append(("SetTrackName", track_type, index, name))
        if (track_type, index) not in self._tracks:
            return False
        self._track_names[(track_type, index)] = name
        return True

    def Export(self, path, export_type=None, export_subtype=None) -> bool:
        """Serialize this fake timeline to OTIO, mirroring ResolveAdapter.snapshot's layout.

        Producing OTIO whose clips match the snapshot (same names/positions/src basenames) is what
        lets a plan's content-stable clip ids resolve against the exported IR in the rebuild path.
        """
        self.calls.append(("Export", path, export_type, export_subtype))
        import opentimelineio as otio
        rate = 24.0

        def rt(f):
            return otio.opentime.RationalTime(int(f), rate)

        tl = otio.schema.Timeline(name=self._name)
        tl.global_start_time = rt(0)
        for kind, TrackKind in (("video", otio.schema.TrackKind.Video),
                                ("audio", otio.schema.TrackKind.Audio)):
            for (ttype, idx) in sorted(k for k in self._tracks if k[0] == kind):
                tr = otio.schema.Track(name=self._track_names.get((ttype, idx), ""), kind=TrackKind)
                cursor = 0
                for it in sorted(self._tracks[(ttype, idx)], key=lambda i: i.GetStart()):
                    start = it.GetStart() - self._start
                    if start > cursor:
                        tr.append(otio.schema.Gap(
                            source_range=otio.opentime.TimeRange(rt(0), rt(start - cursor))))
                    tr.append(otio.schema.Clip(
                        name=it.GetName(),
                        media_reference=otio.schema.ExternalReference(target_url=it._src_path),
                        source_range=otio.opentime.TimeRange(
                            rt(it.GetSourceStartFrame()), rt(it.GetDuration()))))
                    cursor = start + it.GetDuration()
                tl.tracks.append(tr)
        otio.adapters.write_to_file(tl, path)
        return True


class FakeFolder:
    """A media-pool bin: a name, nested subfolders, and the clips it holds."""

    def __init__(self, name: str = "Master"):
        self._name = name
        self._subs: list["FakeFolder"] = []
        self._clips: list[Any] = []

    def GetName(self) -> str:
        return self._name

    def GetSubFolderList(self) -> list["FakeFolder"]:
        return list(self._subs)

    def GetClipList(self) -> list:
        return list(self._clips)


class FakeMediaPool:
    def __init__(self, selected: Optional[list[FakeMediaPoolItem]] = None):
        self._selected = list(selected or [])
        self.append_log: list[Any] = []
        self.imported_timelines: list[str] = []
        self.imported_media: list[Any] = []
        self._root = FakeFolder("Master")
        self.moved: list[tuple] = []

    def GetSelectedClips(self) -> list[FakeMediaPoolItem]:
        return list(self._selected)

    def SetSelectedClip(self, item) -> bool:
        self._selected = [item]
        return True

    def AppendToTimeline(self, *args) -> list:
        self.append_log.append(args)
        # Return one fake item per appended clip-info, mimicking the real return.
        flat = args[0] if len(args) == 1 and isinstance(args[0], list) else list(args)
        return [FakeTimelineItem(f"appended-{i}", 0, 1) for i, _ in enumerate(flat)]

    def ImportMedia(self, items) -> list:
        """Record imported file paths; return one media-pool item per path (truthy, like Resolve)."""
        paths = items if isinstance(items, list) else [items]
        out = []
        for p in paths:
            path = p.get("FilePath") if isinstance(p, dict) else p
            self.imported_media.append(path)
            out.append(FakeMediaPoolItem(str(path), {"File Path": str(path)}))
        return out

    def ImportTimelineFromFile(self, path, options=None):
        """Record the rebuilt-timeline path and return a truthy handle (a loaded FakeTimeline)."""
        self.imported_timelines.append(path)
        return _fake_timeline_from_otio(path)

    def GetRootFolder(self) -> FakeFolder:
        return self._root

    def AddSubFolder(self, folder: FakeFolder, name: str) -> FakeFolder:
        sub = FakeFolder(name)
        folder._subs.append(sub)
        return sub

    def MoveClips(self, clips, target: FakeFolder) -> bool:
        self.moved.append(([getattr(c, "GetName", lambda: c)() for c in clips], target.GetName()))
        target._clips.extend(clips)
        return True


class FakeProject:
    def __init__(self, name: str = "Project", timeline: Optional[FakeTimeline] = None,
                 media_pool: Optional[FakeMediaPool] = None):
        self._name = name
        self._timeline = timeline
        self._media_pool = media_pool or FakeMediaPool()
        self._color_groups: list[FakeColorGroup] = []

    def GetName(self) -> str:
        return self._name

    # color groups (Resolve-live color organization)
    def AddColorGroup(self, name: str) -> FakeColorGroup:
        g = FakeColorGroup(name)
        self._color_groups.append(g)
        return g

    def GetColorGroupsList(self) -> list:
        return list(self._color_groups)

    def DeleteColorGroup(self, group) -> bool:
        if group in self._color_groups:
            self._color_groups.remove(group)
            return True
        return False

    def GetCurrentTimeline(self) -> Optional[FakeTimeline]:
        return self._timeline

    def SetCurrentTimeline(self, tl) -> bool:
        self._timeline = tl
        return True

    def GetMediaPool(self) -> FakeMediaPool:
        return self._media_pool

    def GetTimelineCount(self) -> int:
        return 1 if self._timeline else 0

    def GetTimelineByIndex(self, idx: int) -> Optional[FakeTimeline]:
        return self._timeline if idx == 1 else None


class FakeProjectManager:
    def __init__(self, project: Optional[FakeProject] = None):
        self._project = project

    def GetCurrentProject(self) -> Optional[FakeProject]:
        return self._project


class FakeResolve:
    def __init__(self, project: Optional[FakeProject] = None,
                 product: str = "DaVinci Resolve Studio", version: str = "20.0.0.001"):
        self._pm = FakeProjectManager(project)
        self._product = product
        self._version = version
        self.pages_opened: list[str] = []

    def GetProductName(self) -> str:
        return self._product

    def GetVersionString(self) -> str:
        return self._version

    def GetProjectManager(self) -> FakeProjectManager:
        return self._pm

    def OpenPage(self, page: str) -> bool:
        self.pages_opened.append(page)
        return True

    def Fusion(self):
        return None

    # Timeline-export type constants (the adapter reads resolve.EXPORT_OTIO off this handle).
    EXPORT_OTIO = "EXPORT_OTIO"
    EXPORT_NONE = 0


# --------------------------------------------------------------------------- builders
def _fake_timeline_from_otio(path: str) -> "FakeTimeline":
    """Load an OTIO file into a FakeTimeline so an imported/rebuilt timeline is a usable handle."""
    import opentimelineio as otio

    timeline = otio.adapters.read_from_file(path)
    tl = FakeTimeline(timeline.name or "Imported", start_frame=0)
    v_idx = a_idx = 0
    for track in timeline.tracks:
        is_video = (track.kind or "Video").lower().startswith("v")
        if is_video:
            v_idx += 1
            ttype, idx = "video", v_idx
        else:
            a_idx += 1
            ttype, idx = "audio", a_idx
        items = []
        for child in track:
            if isinstance(child, (otio.schema.Gap, otio.schema.Transition)):
                continue
            rng = child.range_in_parent()
            src_start = 0
            if getattr(child, "source_range", None) is not None:
                src_start = int(round(child.source_range.start_time.to_frames()))
            url = ""
            mr = getattr(child, "media_reference", None)
            if mr is not None:
                url = getattr(mr, "target_url", "") or ""
            items.append(FakeTimelineItem(
                child.name or "clip",
                int(round(rng.start_time.to_frames())),
                int(round(rng.duration.to_frames())),
                source_start=src_start, src_path=url))
        tl.add_track(ttype, idx, items)
    return tl
def make_two_track_resolve() -> FakeResolve:
    """A small populated graph: V1 with 3 clips, A1 with 1 clip, one selected media item."""
    v1 = [
        FakeTimelineItem("intro", start=0, duration=48, src_path="/media/intro.mov"),
        FakeTimelineItem("midshot", start=48, duration=72, src_path="/media/midshot.mov"),
        FakeTimelineItem("outro", start=120, duration=36, src_path="/media/outro.mov"),
    ]
    a1 = [FakeTimelineItem("music", start=0, duration=156, src_path="/media/music.wav")]
    tl = FakeTimeline("Demo Timeline", start_frame=0)
    tl.add_track("video", 1, v1)
    tl.add_track("audio", 1, a1)
    selected_media = [v1[1].GetMediaPoolItem()]
    project = FakeProject("Demo", timeline=tl, media_pool=FakeMediaPool(selected=selected_media))
    return FakeResolve(project=project)
