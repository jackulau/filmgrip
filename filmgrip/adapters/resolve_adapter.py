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
from ..serialize.fgx import parse_track_code
from .base import ApplyResult, Capabilities, GrabAdapter, Selection
from .interchange import REBUILD_OPS, OtioMutator
from .resolve_client import (
    ResolveOperationFailed,
    ResolveSession,
    connect,
    export_timeline_otio,
    import_timeline_from_file,
    require,
)

# Ops the Resolve scripting API can apply in place, reliably. Everything else (precise
# move/trim/split/transition/ripple) the API cannot do faithfully — those route to the
# OTIO-rebuild path (export timeline -> mutate OTIO -> ImportTimelineFromFile), implemented in
# the interchange adapter (D11) and orchestrated by the CLI (D10). This is the dual apply-path.
LIVE_OPS = frozenset({"add_marker", "set_property", "delete"})

# Live ops that ADD media or structure rather than mutate an existing clip: import_audio (media-pool
# import + AppendToTimeline) and add_track. They apply live (no lossy rebuild) and, because they only
# add, are applied BEFORE any structural rebuild so the export captures them. (D7 grows this set with
# rename_track / create_bin / move_to_bin.)
LIVE_EXTRA_OPS = frozenset({
    "import_audio", "add_track", "rename_track", "create_bin", "move_to_bin",
})


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

    def __init__(self, sfx_library: Any = None):
        # Optional injected SfxLibrary so import_audio can resolve a name -> file. Loaded lazily from
        # the default dir ($FILMGRIP_SFX_DIR / ~/.filmgrip/sfx) when not supplied.
        self._sfx_library = sfx_library

    def capabilities(self) -> Capabilities:
        return Capabilities(
            editor="DaVinci Resolve (Studio)",
            role="flagship-native",
            mechanism="native Python scripting API (fusionscript)",
            live_selection=False,  # honest: no true multi-clip timeline selection API
            write_back=True,
            requires_app_running=True,
            lossy_features=[
                "add_transition (Fusion/interchange only)",
                "structural edits via OTIO rebuild are lossy (drop grades/Fusion/some transitions)",
                "audio volume/gain/fades are NOT scriptable (Fairlight-only) — placement only",
            ],
            audio_support="live",                 # ImportMedia + AppendToTimeline(mediaType=2)
            audio_volume_scriptable=False,        # honest: Fairlight levels aren't scriptable
            organize_support="live",              # add_track/rename_track/create_bin/move_to_bin
            editor_panel="native",                # Scripts-menu UIManager floating panel (D10)
            selection_confidence="reconstructed",  # current item + media-pool, not a true multi-select
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
        return Selection(ids=ids, basis="current_video_item+media_pool", note=note,
                         confidence="reconstructed")

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

        # Three op classes:
        #  • live    (marker/property/delete) — mutate an existing clip in place, reversible.
        #  • extra   (import_audio/add_track)  — ADD media/structure live; reversible best-effort.
        #  • struct  (trim/move/split/insert/ripple) — need the lossy OTIO rebuild.
        # Apply live + extra IN PLACE first (so the export captures added audio/tracks), then route
        # the structural ops — structural-only — through the rebuild. Pure live/extra plans never
        # trigger a rebuild, so a plain "add a whoosh" stays lossless.
        live_ops = [op for op in plan.ops if op.op in LIVE_OPS or op.op in LIVE_EXTRA_OPS]
        structural = [op for op in plan.ops if op.op in REBUILD_OPS and op.op not in LIVE_OPS]
        not_applicable = [op.op for op in plan.ops if op.op not in LIVE_OPS
                          and op.op not in LIVE_EXTRA_OPS and op.op not in REBUILD_OPS]

        applied: list[str] = []
        rollback: list = []
        warnings: list[str] = []
        try:
            for op in live_ops:
                desc, inverse, warn = self._apply_one(op, ir, timeline, session)
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

        if structural:
            sub = EditPlan(notes=plan.notes, ops=structural)
            res = self._apply_via_rebuild(sub, session, ir, timeline)
            if not res.ok:
                reversed_count = 0
                for inv in reversed(rollback):  # unwind the reversible in-place live ops
                    try:
                        inv()
                        reversed_count += 1
                    except Exception:
                        pass
                # Be honest: add_track/create_bin/move_to_bin have no inverse, so any of those that
                # already ran are NOT undone. Don't claim "timeline intact" when they did.
                irreversible = len(applied) - reversed_count
                fail_warnings = list(warnings)
                if irreversible > 0:
                    fail_warnings.append(
                        f"structural rebuild failed AFTER {len(applied)} live op(s) were applied; "
                        f"{reversed_count} reversible one(s) were rolled back, but {irreversible} "
                        f"(track/bin additions, media-pool moves) are NOT scriptable to undo — "
                        f"review them in Resolve: {applied}")
                return ApplyResult(ok=False, applied=applied, errors=res.errors,
                                   warnings=fail_warnings)
            applied += res.applied
            warnings += res.warnings

        unsupported = [
            f"op '{name}' has no live or rebuild path in Resolve — do it in the editor "
            f"(e.g. add a transition on the Edit page)"
            for name in sorted(set(not_applicable))
        ]
        diff = "\n".join(f"  ✓ {d}" for d in applied) or "  (no live ops)"
        # ok=False if any requested op had no path — never report success for an edit we didn't make,
        # even when other ops in the same plan applied live.
        return ApplyResult(ok=not unsupported, applied=applied, diff=diff,
                           warnings=warnings, unsupported=unsupported)

    # -- rebuild apply path -----------------------------------------------------
    def _apply_via_rebuild(self, plan: EditPlan, session: ResolveSession,
                           native_ir: TimelineIR, timeline: Any) -> ApplyResult:
        """Apply structural ops by export→mutate→import (the dual apply-path's rebuild side).

        Export the live timeline to OTIO, mutate the OTIO graph with the validated plan, and import
        the result as a NEW timeline. Resolve imports rather than mutating in place, so a failure at
        any step leaves the user's original timeline intact — no half-applied state to roll back.
        """
        import os
        import tempfile

        tmpdir = tempfile.mkdtemp(prefix="filmgrip-rebuild-")
        export_path = os.path.join(tmpdir, "export.otio")
        import_path = os.path.join(tmpdir, "rebuilt.otio")
        try:
            export_timeline_otio(session, export_path, timeline)
        except ResolveOperationFailed as exc:
            return ApplyResult(ok=False, errors=[f"rebuild export failed (timeline intact): {exc}"])

        rebuild_ir = TimelineIR.from_otio_file(export_path)
        remap_warnings = self._reresolve_plan(plan, native_ir, rebuild_ir)
        vres = validate(plan, rebuild_ir)
        if not vres.ok:
            return ApplyResult(
                ok=False,
                errors=[f"rebuild re-validation failed (timeline intact): {e}" for e in vres.errors])

        applied, unsupported = OtioMutator(rebuild_ir).apply(plan)
        rebuild_ir.to_otio_file(import_path)
        try:
            import_timeline_from_file(session, import_path)
        except ResolveOperationFailed as exc:
            return ApplyResult(ok=False, applied=applied,
                               errors=[f"rebuild import failed (original timeline intact): {exc}"])

        warnings = remap_warnings + [
            "applied via OTIO rebuild — created a NEW timeline; color grades, Fusion comps and some "
            "transitions are NOT carried over (lossy). Your original timeline is left intact."
        ]
        diff = "\n".join(f"  ✓ {d}" for d in applied) or "  (no rebuild ops)"
        # The structural sub-plan is pre-filtered to REBUILD_OPS, so `unsupported` is normally empty;
        # propagate it (ok=False) if a future op ever slips through unhandled.
        return ApplyResult(ok=not unsupported, applied=applied, diff=diff,
                           warnings=warnings, unsupported=unsupported)

    @staticmethod
    def _reresolve_plan(plan: EditPlan, native_ir: TimelineIR,
                        rebuild_ir: TimelineIR) -> list[str]:
        """Rewrite each op's clip_id to the exported IR's ids (content-stable, but drift-tolerant).

        IDs are content-derived, so the exported OTIO usually mints identical ids — but if Resolve's
        export drifts a position, ``IdMap.reresolve`` matches by fingerprint so the plan still lands.
        """
        warnings: list[str] = []
        for op in plan.ops:
            cid = getattr(op, "clip_id", None)
            if cid is None or rebuild_ir.clip(cid) is not None:
                continue
            match = native_ir.reresolve(cid, rebuild_ir)
            if match is not None and match.id != cid:
                warnings.append(f"re-resolved clip {cid} → {match.id} across the OTIO export")
                op.clip_id = match.id
        return warnings

    # -- live media / structure ops (import_audio, add_track) -------------------
    def _sfx_path(self, op) -> str:
        """Resolve an import_audio op to a concrete audio file path (explicit src_ref or SFX name)."""
        if op.src_ref:
            return op.src_ref
        from ..audio.library import SfxLibrary
        lib = self._sfx_library or SfxLibrary.load()
        entry = lib.get(op.sfx) or lib.resolve(op.sfx)
        if entry is None:
            raise ResolveOperationFailed(
                f"sound effect '{op.sfx}' not found in the SFX library at {lib.base} "
                f"(check `film-grip sfx list`)")
        if not entry.exists(lib.base):
            raise ResolveOperationFailed(
                f"sound effect '{op.sfx}' resolves to '{entry.file}', but that file is missing "
                f"under {lib.base} — fix the manifest or restore the file (`film-grip sfx list`)")
        return str(entry.path(lib.base))

    def _import_audio(self, op, timeline: Any, session: ResolveSession):
        """Import an audio file into the media pool and place it on an audio track."""
        path = self._sfx_path(op)
        mp = require(session.media_pool(), "no media pool (open a project first)")
        imported = require(_native_call(mp, "ImportMedia", [path]),
                           f"ImportMedia({path}) failed")
        mpi = imported[0] if isinstance(imported, list) and imported else imported
        require(mpi, f"ImportMedia({path}) returned nothing")

        _, idx = parse_track_code(op.track)
        if idx > int(timeline.GetTrackCount("audio") or 0):
            raise ResolveOperationFailed(
                f"audio track '{op.track}' does not exist — add one first (add_track audio)")
        clip_info: dict[str, Any] = {"mediaPoolItem": mpi, "mediaType": 2,
                                     "trackIndex": idx, "recordFrame": op.at_start}
        if op.duration is not None:
            clip_info["startFrame"] = op.source_in
            clip_info["endFrame"] = op.source_in + op.duration
        appended = require(_native_call(mp, "AppendToTimeline", [clip_info]),
                           f"AppendToTimeline failed for '{path}' on {op.track}")
        inverse = None
        if isinstance(appended, list) and appended:
            placed = appended[0]
            inverse = lambda: _native_call(timeline, "DeleteClips", [placed], False)  # noqa: E731
        label = op.sfx or path
        warn = (None if op.duration is not None else
                f"placed the FULL length of '{label}' (its duration wasn't known at validation, so "
                f"overlap with later audio wasn't pre-checked) — trim/split it if it runs long")
        return (f"import_audio {label} → {op.track} @ {op.at_start}", inverse, warn)

    def _add_track(self, op, timeline: Any):
        if op.kind == "audio":
            require(_native_call(timeline, "AddTrack", "audio", op.audio_type),
                    f"AddTrack audio ({op.audio_type}) failed")
            desc = f"add_track audio ({op.audio_type})"
        else:
            require(_native_call(timeline, "AddTrack", op.kind), f"AddTrack {op.kind} failed")
            desc = f"add_track {op.kind}"
        # Resolve has no reliable scripted track-removal, so this is not auto-reversible.
        return (desc, None, "added a track (track removal is not scriptable — undo in Resolve if needed)")

    def _rename_track(self, op, timeline: Any):
        kind, idx = parse_track_code(op.track)
        old = timeline.GetTrackName(kind, idx) if hasattr(timeline, "GetTrackName") else ""
        require(_native_call(timeline, "SetTrackName", kind, idx, op.name),
                f"SetTrackName on {op.track} failed")
        inverse = ((lambda: _native_call(timeline, "SetTrackName", kind, idx, old)) if old else None)
        return (f"rename_track {op.track} → {op.name!r}", inverse, None)

    @staticmethod
    def _find_folder(root: Any, name: str):
        subs = root.GetSubFolderList() if hasattr(root, "GetSubFolderList") else []
        for f in subs or []:
            if (f.GetName() if hasattr(f, "GetName") else "") == name:
                return f
        return None

    def _create_bin(self, op, session: ResolveSession):
        mp = require(session.media_pool(), "no media pool (open a project first)")
        root = require(_native_call(mp, "GetRootFolder"), "GetRootFolder failed")
        parent, warn = root, None
        if op.parent:
            found = self._find_folder(root, op.parent)
            if found is None:
                warn = f"parent bin '{op.parent}' not found — created '{op.name}' at the media-pool root"
            else:
                parent = found
        require(_native_call(mp, "AddSubFolder", parent, op.name), f"AddSubFolder '{op.name}' failed")
        loc = f" under '{op.parent}'" if (op.parent and warn is None) else ""
        # Bin deletion isn't reliably scriptable -> no inverse.
        return (f"create_bin '{op.name}'{loc}", None, warn)

    def _move_to_bin(self, op, ir: TimelineIR, session: ResolveSession):
        clip = ir.clip(op.clip_id)
        item = getattr(clip, "native", None) if clip is not None else None
        mpi = item.GetMediaPoolItem() if item is not None and hasattr(item, "GetMediaPoolItem") else None
        if mpi is None:
            raise ResolveOperationFailed(f"no media-pool item backing clip '{op.clip_id}'")
        mp = require(session.media_pool(), "no media pool (open a project first)")
        root = require(_native_call(mp, "GetRootFolder"), "GetRootFolder failed")
        target = self._find_folder(root, op.bin) \
            or require(_native_call(mp, "AddSubFolder", root, op.bin),
                       f"could not create destination bin '{op.bin}'")
        require(_native_call(mp, "MoveClips", [mpi], target), f"MoveClips to '{op.bin}' failed")
        return (f"move_to_bin {clip.name} → '{op.bin}'", None,
                "media moved between media-pool bins (not auto-reversible)")

    def _apply_one(self, op, ir: TimelineIR, timeline: Any, session: ResolveSession = None):
        """Apply one live op against its native handle.

        Returns ``(description, inverse_callable_or_None, warning_or_None)``. Required calls go
        through :func:`_native_call` so a phantom/failed Resolve method aborts + rolls back; the
        rename path is best-effort (Resolve has no reliable per-item rename).
        """
        # Ops that add media/structure/organization don't target an existing clip handle.
        if op.op == "import_audio":
            return self._import_audio(op, timeline, session)
        if op.op == "add_track":
            return self._add_track(op, timeline)
        if op.op == "rename_track":
            return self._rename_track(op, timeline)
        if op.op == "create_bin":
            return self._create_bin(op, session)
        if op.op == "move_to_bin":
            return self._move_to_bin(op, ir, session)

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
