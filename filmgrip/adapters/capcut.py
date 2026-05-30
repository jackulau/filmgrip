"""CapCut (International) offline draft adapter — best-effort.

CapCut International saves a project as ``draft_content.json``: a tracks/segments/materials graph
with all times in **microseconds**. There is no live API and no plugin surface, so film-grip edits
the draft file directly, **only when the app is closed**, and refuses anything it can't safely
touch:

* **encryption gate** — newer JianYing (CapCut's Chinese sibling, v6+) encrypts the draft. If the
  file isn't plain JSON we refuse with a clear message rather than corrupt a project.
* **no live selection** — never claimed; selection is the whole draft.

Supported ops are the ones the JSON represents cleanly (trim/delete/move on segment timeranges);
the rest warn. This is deliberately conservative per the research: a real working round-trip for
unencrypted International drafts, and an honest "no" everywhere else.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan
from ..protocol.validate import validate
from .base import ApplyResult, Capabilities, GrabAdapter, NotSupportedError, Selection

SUPPORTED_OPS = frozenset({"trim", "delete", "move"})


class CapcutAdapter(GrabAdapter):
    name = "capcut"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            editor="CapCut (International)",
            role="best-effort",
            mechanism="offline draft_content.json rewrite (microsecond timeranges)",
            live_selection=False,
            write_back=True,                # offline only, app must be closed
            requires_app_running=False,
            lossy_features=["effects/filters/text", "encrypted JianYing v6+ drafts (refused)"],
            audio_support="offline",         # audio segments rewritten in the offline draft JSON
            organize_support="none",
            editor_panel="none",
            selection_confidence="precise",  # the draft JSON IS the exact project
        )

    # -- locate + load ----------------------------------------------------------
    @staticmethod
    def _resolve_path(source: str) -> str:
        if os.path.isdir(source):
            cand = os.path.join(source, "draft_content.json")
            if os.path.exists(cand):
                return cand
            raise FileNotFoundError(f"no draft_content.json in {source}")
        return source

    def _load(self, source: str) -> dict:
        path = self._resolve_path(source)
        with open(path, "rb") as fh:
            raw = fh.read()
        head = raw.lstrip()[:1]
        if head not in (b"{", b"["):
            raise NotSupportedError(
                "CapCut draft is not plain JSON — likely an encrypted JianYing (v6+) draft. "
                "film-grip refuses to edit encrypted drafts rather than corrupt them.")
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise NotSupportedError(f"CapCut draft is not valid JSON (encrypted?): {exc}") from exc

    @staticmethod
    def _fps(doc: dict) -> float:
        return float(doc.get("fps") or 30.0)

    def _us_per_frame(self, doc: dict) -> float:
        return 1_000_000.0 / self._fps(doc)

    # -- parse ------------------------------------------------------------------
    def snapshot(self, source: Any, *, _doc: Optional[dict] = None) -> TimelineIR:
        import opentimelineio as otio

        doc = _doc if _doc is not None else self._load(source)
        rate = self._fps(doc)
        upf = self._us_per_frame(doc)

        mats = {}
        materials = doc.get("materials", {})
        for cat in ("videos", "audios", "images"):
            for m in materials.get(cat, []) or []:
                mats[m.get("id")] = m.get("path") or m.get("material_name") or m.get("id")

        def fr(us: float) -> int:
            return int(round(us / upf))

        tl = otio.schema.Timeline(name=doc.get("name") or "CapCut Draft")
        tl.global_start_time = otio.opentime.RationalTime(0, rate)
        for tr in doc.get("tracks", []):
            kind = (otio.schema.TrackKind.Audio if tr.get("type") == "audio"
                    else otio.schema.TrackKind.Video)
            track = otio.schema.Track(name=tr.get("type", "video"), kind=kind)
            playhead = 0
            for seg in sorted(tr.get("segments", []), key=lambda s: s["target_timerange"]["start"]):
                tgt = seg["target_timerange"]
                src = seg.get("source_timerange", tgt)
                start = fr(tgt["start"])
                dur = fr(tgt["duration"])
                if start > playhead:
                    track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(0, rate),
                        otio.opentime.RationalTime(start - playhead, rate))))
                res = mats.get(seg.get("material_id"), seg.get("material_id") or "clip")
                track.append(otio.schema.Clip(
                    name=os.path.splitext(os.path.basename(res))[0] or "clip",
                    media_reference=otio.schema.ExternalReference(target_url=res),
                    source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(fr(src["start"]), rate),
                        otio.opentime.RationalTime(dur, rate))))
                playhead = start + dur
            tl.tracks.append(track)
        return TimelineIR.from_otio(tl)

    def get_selection(self, source: Any, ir: Optional[TimelineIR] = None) -> Selection:
        if ir is None:
            ir = self.snapshot(source)
        return Selection(ids=[c.id for c in ir.real_clips()], basis="capcut_draft",
                         note="CapCut has no live selection; whole draft is context. App must be closed to write.")

    # -- apply ------------------------------------------------------------------
    def apply(self, plan: EditPlan, source: Any, *, out_path: Optional[str] = None, **kw) -> ApplyResult:
        doc = self._load(source)
        ir = self.snapshot(source, _doc=doc)
        res = validate(plan, ir)
        if not res.ok:
            return ApplyResult(ok=False, errors=[str(e) for e in res.errors])

        upf = self._us_per_frame(doc)

        # Map clip id -> segment dict, in track+start order.
        seg_for: dict[str, dict] = {}
        ti = 0
        for tr in doc.get("tracks", []):
            ti += 1
            segs = sorted(tr.get("segments", []), key=lambda s: s["target_timerange"]["start"])
            clips = [c for c in ir.real_clips() if c.track_index == ti]
            for clip, seg in zip(clips, segs):
                seg_for[clip.id] = seg

        applied: list[str] = []
        warnings: list[str] = []
        for op in plan.ops:
            if op.op not in SUPPORTED_OPS:
                warnings.append(f"op '{op.op}' not represented in CapCut draft rewrite")
                continue
            clip = ir.clip(op.clip_id)
            seg = seg_for[op.clip_id]
            tgt = seg["target_timerange"]
            srcr = seg.setdefault("source_timerange", dict(tgt))
            if op.op == "trim":
                d_us = int(round(op.delta * upf))
                if op.edge == "out":
                    tgt["duration"] += d_us
                    srcr["duration"] += d_us
                else:
                    tgt["start"] += d_us
                    tgt["duration"] -= d_us
                    srcr["start"] += d_us
                    srcr["duration"] -= d_us
                applied.append(f"trim {clip.name} {op.edge} {op.delta:+d}")
            elif op.op == "move":
                tgt["start"] = int(round(op.to_start * upf))
                applied.append(f"move {clip.name} -> {op.to_start}")
            elif op.op == "delete":
                for tr in doc.get("tracks", []):
                    if seg in tr.get("segments", []):
                        tr["segments"].remove(seg)
                applied.append(f"delete {clip.name}")

        out = self._resolve_path(out_path) if out_path else self._resolve_path(source)
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, ensure_ascii=False)
        diff = "\n".join(f"  ✓ {d}" for d in applied) or "  (no CapCut-applicable ops)"
        diff += f"\n  → wrote CapCut draft to {out}"
        return ApplyResult(ok=True, applied=applied, diff=diff, warnings=warnings)
