"""``film-grip beats`` — synthesize a musical beat grid + tempo (the rhythm perception Resolve's API
won't give an agent) for a media file or a timeline's clips, as JSON.

The rhythm sibling of ``film-grip scopes``: the numbers an agent reasons over to PROPOSE a
music-synced edit ("cut on the beat", "land each clip on the drop") and, after applying, re-reads to
VERIFY it landed. Honest failure modes (no ffmpeg, no numpy, offline media, a retimed clip) surface
as clear ``errors`` entries, never an empty result pretending to be a reading.
"""
from __future__ import annotations

import json
import sys


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True))


def cmd_beats(args) -> int:
    from .perception.music import beats_for_media
    from .perception.transcribe import PerceptionUnavailable

    try:
        if args.media:
            report = beats_for_media(args.media)
            _emit(report)
            # An offline/undecodable media file comes back as an errors entry — honest, not a fake grid.
            return 0 if not report.get("errors") else 2

        if not args.fixture:
            print("beats: pass --media <file> or --fixture <timeline> [--select ids]",
                  file=sys.stderr)
            return 2

        # Timeline mode: read each selected clip's source audio over the clip's source window so beats
        # come back in timeline frames (and the retimed/offline honesty wall applies per clip).
        from .adapters.registry import for_extension
        from .perception.align import media_path_of
        import os

        entry = for_extension(os.path.splitext(args.fixture)[1])
        if entry is None:
            print(f"beats: no adapter for '{args.fixture}'", file=sys.stderr)
            return 2
        ir = entry.adapter.snapshot(args.fixture)
        want = set((args.select or "").split(",")) if args.select else None
        clips = [c for c in ir.real_clips() if (want is None or c.id in want)]
        if not clips:
            print("beats: no matching clips", file=sys.stderr)
            return 2
        out = []
        any_errors = False
        for c in clips:
            try:
                rep = beats_for_media(media_path_of(c), c)
                rep["clip_id"] = c.id
                rep["clip_name"] = c.name
                if rep.get("errors"):
                    any_errors = True
                out.append(rep)
            except PerceptionUnavailable as exc:
                any_errors = True
                out.append({"clip_id": c.id, "clip_name": c.name, "error": str(exc)})
        _emit(out)
        # 0 = every clip read; 3 = partial (some clips errored) — mirrors the perception CLIs' contract.
        return 0 if not any_errors else 3
    except PerceptionUnavailable as exc:
        print(f"beats unavailable: {exc}", file=sys.stderr)
        return 2
