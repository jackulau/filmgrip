"""``film-grip frames`` — contact-sheet PNGs (filmstrip + waveform) from the timeline.

Per selected clip (evenly sampled, ``--count`` tiles) or at exact timeline frames (``--at``).
Prints the PNG path plus a tile legend (tile → timeline frame → media seconds) so the picture
and the numbers stay together. Exit codes: 0 all rendered · 3 partial · 1 nothing rendered ·
2 editor unreachable.
"""
from __future__ import annotations

import shutil


def _emit(text: str) -> None:
    print(text)


def cmd_frames(args) -> int:
    from .perception.transcribe import PerceptionUnavailable

    try:
        if getattr(args, "fixture", None):
            from .cli_edit import FixtureError, load_fixture_ir

            try:
                ir = load_fixture_ir(args.fixture)
            except FixtureError as exc:
                _emit(f"error: {exc}")
                return 1
            return _render(args, ir, None)
        return _run_live(args)
    except PerceptionUnavailable as exc:
        _emit(f"error: {exc}")
        return 1


def _run_live(args) -> int:
    if args.editor != "resolve":
        _emit(f"error: live editor '{args.editor}' not yet supported; use --editor resolve or "
              f"--fixture.")
        return 3

    from .adapters import resolve_client as rc
    from .adapters.resolve_adapter import ResolveAdapter
    from .cli_edit import _preflight_message

    report = rc.preflight()
    if not report["app_running"]:
        _emit(_preflight_message(report))
        return 2

    adapter = ResolveAdapter()
    session = rc.connect()
    ir = adapter.snapshot(session)
    selection = adapter.get_selection(session, ir)
    return _render(args, ir, selection.ids or None)


def _render(args, ir, live_selection) -> int:
    from .perception.frames import clip_sheet, timeline_sheet
    from .perception.transcribe import PerceptionUnavailable

    sheets, errors = [], []
    if getattr(args, "at", None):
        frames = [int(f.strip()) for f in args.at.split(",") if f.strip()]
        results, errors = timeline_sheet(ir, frames)
        sheets = list(results)
    else:
        if getattr(args, "select", None):
            ids = [s.strip() for s in args.select.split(",") if s.strip()]
        else:
            ids = live_selection or [c.id for c in ir.real_clips()]
        for cid in ids:
            try:
                sheets.append(clip_sheet(ir, cid, count=args.count))
            except PerceptionUnavailable as exc:
                errors.append(str(exc))

    if getattr(args, "out", None) and len(sheets) == 1:
        shutil.copyfile(sheets[0].png_path, args.out)
        sheets[0].png_path = args.out

    for sheet in sheets:
        _emit(sheet.png_path)
        for entry in sheet.legend:
            _emit(f"  tile {entry['tile']}: frame {entry['timeline_frame']} "
                  f"({entry['media_s']}s media, {entry['clip_id']})")
        for note in sheet.notes:
            _emit(f"  ⚠ {note}")
    for err in errors:
        _emit(f"  ✗ {err}")
    if sheets and not errors:
        return 0
    return 3 if sheets else 1
