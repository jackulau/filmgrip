"""Wondershare Filmora adapter — read-only, by honest necessity.

Filmora has **no** scripting API, plugin SDK, or supported automation surface, and its ``.wfp``
project format is proprietary and version-fragile (newer versions are a ZIP of JSON/XML). film-grip
will *read* a ``.wfp`` to enumerate tracks/clips so a user can pull Filmora context into a prompt —
but it will never write a ``.wfp`` back. :meth:`apply` raises :class:`NotSupportedError`. Promising
editing here would be a lie; the conservative posture is the correct one.
"""
from __future__ import annotations

import json
import zipfile
from typing import Any, Optional

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan
from .base import ApplyResult, Capabilities, GrabAdapter, NotSupportedError, Selection

# JSON entries Filmora versions have been observed to use; we read the first that parses with a
# "tracks" array. Order = preference.
_CANDIDATE_ENTRIES = ("project.json", "MainCut.json", "Config.json", "project_info.json")


class FilmoraAdapter(GrabAdapter):
    name = "filmora"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            editor="Wondershare Filmora",
            role="read-only",
            mechanism="offline .wfp (ZIP of JSON/XML) READ-ONLY parse",
            live_selection=False,
            write_back=False,           # the whole point — Filmora cannot be automated
            requires_app_running=False,
            lossy_features=["EVERYTHING on write — no write-back path exists"],
            audio_support="read-only",       # audio is parsed, never written
            organize_support="none",
            editor_panel="read-only",
            selection_confidence="readonly",
        )

    # -- read -------------------------------------------------------------------
    def _read_project_json(self, source: str) -> dict:
        if not zipfile.is_zipfile(source):
            raise NotSupportedError(
                f"{source} is not a ZIP-based .wfp (older Filmora formats are unsupported and "
                "version-fragile). film-grip can only read newer ZIP-based projects.")
        with zipfile.ZipFile(source) as zf:
            names = zf.namelist()
            ordered = [n for n in _CANDIDATE_ENTRIES if n in names] + \
                      [n for n in names if n.endswith(".json") and n not in _CANDIDATE_ENTRIES]
            for name in ordered:
                try:
                    doc = json.loads(zf.read(name).decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError):
                    continue
                if isinstance(doc, dict) and "tracks" in doc:
                    return doc
        raise NotSupportedError(
            "No parseable project JSON with a 'tracks' array inside the .wfp — unsupported Filmora "
            "version/layout. (Filmora has no documented format; this is read-only best-effort.)")

    def snapshot(self, source: Any, *, _doc: Optional[dict] = None) -> TimelineIR:
        import opentimelineio as otio

        doc = _doc if _doc is not None else self._read_project_json(source)
        rate = float(doc.get("fps") or 30.0)
        tl = otio.schema.Timeline(name=doc.get("name") or "Filmora Project")
        tl.global_start_time = otio.opentime.RationalTime(0, rate)
        for tr in doc.get("tracks", []):
            kind = (otio.schema.TrackKind.Audio if tr.get("type") == "audio"
                    else otio.schema.TrackKind.Video)
            track = otio.schema.Track(name=tr.get("type", "video"), kind=kind)
            playhead = 0
            for cl in sorted(tr.get("clips", []), key=lambda c: int(c.get("start", 0))):
                start = int(cl.get("start", 0))
                dur = int(cl.get("duration", 0))
                if start > playhead:
                    track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(0, rate),
                        otio.opentime.RationalTime(start - playhead, rate))))
                path = cl.get("path") or cl.get("name") or "clip"
                track.append(otio.schema.Clip(
                    name=cl.get("name") or "clip",
                    media_reference=otio.schema.ExternalReference(target_url=path),
                    source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(int(cl.get("in", 0)), rate),
                        otio.opentime.RationalTime(dur, rate))))
                playhead = start + dur
            tl.tracks.append(track)
        return TimelineIR.from_otio(tl)

    def get_selection(self, source: Any, ir: Optional[TimelineIR] = None) -> Selection:
        if ir is None:
            ir = self.snapshot(source)
        return Selection(
            ids=[c.id for c in ir.real_clips()], basis="filmora_readonly",
            note="Filmora is READ-ONLY in film-grip — context only, no edits can be applied back.")

    # -- write: impossible, and we say so --------------------------------------
    def apply(self, plan: EditPlan, source: Any, **kw) -> ApplyResult:
        raise NotSupportedError(
            "Filmora has no automation/write-back path. This adapter is read-only — use it to pull "
            "Filmora context into a prompt, then make the edit in an editor film-grip can drive "
            "(DaVinci Resolve live, or any FCPXML/EDL/MLT interchange editor).")
