"""Kdenlive + Shotcut adapter (MLT XML).

Both editors save the MLT framework's XML: ``<producer>`` elements describe media, ``<playlist>``
elements are tracks (a sequence of ``<entry>`` clips and ``<blank>`` gaps), and a ``<tractor>``
combines playlists into the timeline. There is no OTIO MLT adapter, so film-grip parses and
rewrites the XML directly with lxml — a near-free interchange win, since the same engine backs
both editors and the format is open and offline-editable.

Supported ops are the ones MLT represents cleanly: ``move`` (reorder/reposition an entry on its
track), ``trim`` (adjust an entry's in/out) and ``delete`` (drop the entry or blank it). MLT's
in/out are **inclusive** frame indices, so duration = out − in + 1.
"""
from __future__ import annotations

import os
from typing import Any, Optional

from lxml import etree

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan
from ..protocol.validate import validate
from .base import ApplyResult, Capabilities, GrabAdapter, Selection

SUPPORTED_OPS = frozenset({"move", "trim", "delete"})


def _profile_rate(root) -> float:
    prof = root.find("profile")
    if prof is not None:
        num = prof.get("frame_rate_num")
        den = prof.get("frame_rate_den") or "1"
        try:
            return float(num) / float(den) if num else 24.0
        except (TypeError, ValueError, ZeroDivisionError):
            return 24.0
    return 24.0


def _to_frames(value: str, rate: float) -> int:
    """MLT in/out may be a frame integer or a 'HH:MM:SS.mmm' timecode."""
    value = (value or "0").strip()
    if ":" in value:
        h, m, rest = value.split(":")
        s = float(rest)
        return int(round((int(h) * 3600 + int(m) * 60 + s) * rate))
    try:
        return int(round(float(value)))
    except ValueError:
        return 0


def _timeline_playlists(root) -> list:
    """Playlists used as timeline tracks (referenced by the tractor, excluding background)."""
    tractor = root.find("tractor")
    if tractor is None:
        return [pl for pl in root.findall("playlist") if pl.findall("entry")]
    ids = []
    for tr in tractor.findall("track"):
        pid = tr.get("producer")
        if pid and "background" not in pid.lower():
            ids.append(pid)
    by_id = {pl.get("id"): pl for pl in root.findall("playlist")}
    return [by_id[i] for i in ids if i in by_id]


class MltAdapter(GrabAdapter):
    name = "mlt"

    def capabilities(self) -> Capabilities:
        return Capabilities(
            editor="Kdenlive / Shotcut (MLT XML)",
            role="interchange",
            mechanism="native MLT XML parse/rewrite via lxml",
            live_selection=False,
            write_back=True,
            requires_app_running=False,
            lossy_features=["effects/filters", "transitions (kept verbatim, not edited)"],
        )

    # -- parse ------------------------------------------------------------------
    def _parse(self, path: str):
        tree = etree.parse(path)
        return tree, tree.getroot()

    def snapshot(self, source: Any, *, _root=None) -> TimelineIR:
        import opentimelineio as otio

        root = _root if _root is not None else self._parse(source)[1]
        rate = _profile_rate(root)
        producers = {}
        for p in root.findall("producer"):
            res = ""
            for prop in p.findall("property"):
                if prop.get("name") == "resource":
                    res = prop.text or ""
            producers[p.get("id")] = res

        tl = otio.schema.Timeline(name=root.get("title") or "MLT Timeline")
        tl.global_start_time = otio.opentime.RationalTime(0, rate)
        for pl in _timeline_playlists(root):
            track = otio.schema.Track(name=pl.get("id") or "V", kind=otio.schema.TrackKind.Video)
            for el in pl:
                if el.tag == "blank":
                    length = _to_frames(el.get("length", "0"), rate)
                    track.append(otio.schema.Gap(source_range=otio.opentime.TimeRange(
                        otio.opentime.RationalTime(0, rate), otio.opentime.RationalTime(length, rate))))
                elif el.tag == "entry":
                    cin = _to_frames(el.get("in", "0"), rate)
                    cout = _to_frames(el.get("out", "0"), rate)
                    dur = cout - cin + 1
                    res = producers.get(el.get("producer"), el.get("producer") or "clip")
                    track.append(otio.schema.Clip(
                        name=os.path.splitext(os.path.basename(res))[0] or "clip",
                        media_reference=otio.schema.ExternalReference(target_url=res),
                        source_range=otio.opentime.TimeRange(
                            otio.opentime.RationalTime(cin, rate),
                            otio.opentime.RationalTime(dur, rate))))
            tl.tracks.append(track)
        return TimelineIR.from_otio(tl)

    def get_selection(self, source: Any, ir: Optional[TimelineIR] = None) -> Selection:
        if ir is None:
            ir = self.snapshot(source)
        return Selection(ids=[c.id for c in ir.real_clips()], basis="mlt_file",
                         note="MLT has no live selection; whole project is context.")

    # -- apply ------------------------------------------------------------------
    def apply(self, plan: EditPlan, source: Any, *, out_path: Optional[str] = None, **kw) -> ApplyResult:
        tree, root = self._parse(source)
        rate = _profile_rate(root)
        ir = self.snapshot(source, _root=root)
        res = validate(plan, ir)
        if not res.ok:
            return ApplyResult(ok=False, errors=[str(e) for e in res.errors])

        # Map each IR clip id -> (playlist element, entry element), in track+start order.
        entry_for: dict[str, tuple] = {}
        playlists = _timeline_playlists(root)
        for ti, pl in enumerate(playlists, start=1):
            entries = [el for el in pl if el.tag == "entry"]
            clips = [c for c in ir.real_clips() if c.track_index == ti]
            for clip, entry in zip(clips, entries):
                entry_for[clip.id] = (pl, entry)

        applied: list[str] = []
        warnings: list[str] = []
        for op in plan.ops:
            if op.op not in SUPPORTED_OPS:
                warnings.append(f"op '{op.op}' not represented in MLT rewrite (kept project as-is)")
                continue
            clip = ir.clip(op.clip_id)
            pl, entry = entry_for[op.clip_id]
            if op.op == "trim":
                cin = _to_frames(entry.get("in", "0"), rate)
                if op.edge == "out":
                    entry.set("out", str(cin + (clip.duration + op.delta) - 1))
                else:
                    entry.set("in", str(cin + op.delta))
                applied.append(f"trim {clip.name} {op.edge} {op.delta:+d}")
            elif op.op == "delete":
                pl.remove(entry)
                if not op.ripple:
                    blank = etree.SubElement(pl, "blank")
                    blank.set("length", str(clip.duration))
                    pl.insert(list(pl).index(blank), blank)  # keep order best-effort
                applied.append(f"delete {clip.name}{' (ripple)' if op.ripple else ''}")
            elif op.op == "move":
                pl.remove(entry)
                others = [c for c in ir.real_clips()
                          if c.track_index == clip.track_index and c.id != op.clip_id]
                insert_idx = sum(1 for c in others if c.start < op.to_start)
                # translate clip index among entries to a child index in the playlist
                real_entries = [el for el in pl if el.tag == "entry"]
                ref = real_entries[insert_idx] if insert_idx < len(real_entries) else None
                if ref is None:
                    pl.append(entry)
                else:
                    pl.insert(list(pl).index(ref), entry)
                applied.append(f"move {clip.name} -> ~{op.to_start}")

        out = out_path or source
        tree.write(out, xml_declaration=True, encoding="utf-8", pretty_print=True)
        diff = "\n".join(f"  ✓ {d}" for d in applied) or "  (no MLT-applicable ops)"
        diff += f"\n  → wrote MLT to {out}"
        return ApplyResult(ok=True, applied=applied, diff=diff, warnings=warnings)
