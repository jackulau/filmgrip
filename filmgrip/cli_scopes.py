"""``film-grip scopes`` — synthesize color scopes (the perception Resolve's API won't give an
agent) for a media file or a timeline's clips, as JSON (+ an optional visual scope PNG).

This is the read side of agentic color: the numbers an agent reasons over to PROPOSE a grade and,
after applying, re-reads to VERIFY it landed. Honest failure modes (no ffmpeg, no numpy, offline
media) surface as clear messages, never an empty result pretending to be a reading.
"""
from __future__ import annotations

import json
import sys


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_scopes(args) -> int:
    from .perception.scopes import analyze_frame, render_scopes_png
    from .perception.transcribe import PerceptionUnavailable

    try:
        if args.media:
            report = analyze_frame(args.media, float(args.at or 0.0))
            if args.png:
                report["scope_png"] = render_scopes_png(args.media, float(args.at or 0.0), args.png)
            _emit(report)
            return 0

        if not args.fixture:
            print("scopes: pass --media <file> or --fixture <timeline> [--select ids]",
                  file=sys.stderr)
            return 2

        # Timeline mode: read each selected clip's source media at the clip's source-in point.
        from .adapters.registry import for_extension
        from .perception.align import media_path_of
        import os

        entry = for_extension(os.path.splitext(args.fixture)[1])
        if entry is None:
            print(f"scopes: no adapter for '{args.fixture}'", file=sys.stderr)
            return 2
        ir = entry.adapter.snapshot(args.fixture)
        want = set((args.select or "").split(",")) if args.select else None
        clips = [c for c in ir.real_clips() if (want is None or c.id in want)]
        if not clips:
            print("scopes: no matching clips", file=sys.stderr)
            return 2
        out = []
        for c in clips:
            try:
                media = media_path_of(c)
                at_s = c.source_start / ir.rate if ir.rate else 0.0
                rep = analyze_frame(media, at_s)
                rep["clip_id"] = c.id
                rep["clip_name"] = c.name
                out.append(rep)
            except PerceptionUnavailable as exc:
                out.append({"clip_id": c.id, "clip_name": c.name, "error": str(exc)})
        _emit(out)
        return 0
    except PerceptionUnavailable as exc:
        print(f"scopes unavailable: {exc}", file=sys.stderr)
        return 2
