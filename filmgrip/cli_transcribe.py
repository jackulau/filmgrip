"""``film-grip transcribe`` — word-level transcripts for media files or timeline clips.

Three ways in:

* ``--media path.mov`` — transcribe one file directly; phrases print in MEDIA seconds.
* ``--fixture cut.otio`` — snapshot a fixture, align each selected clip's words to TIMELINE
  frames (what the planner consumes via the ``get_transcript`` MCP tool).
* live (default ``--editor resolve``) — same, against the running editor's current timeline.

``--srt out.srt`` writes captions instead of phrases (media-time for ``--media``, timeline-time
otherwise — importable into the NLE). Exit codes: 0 all good · 3 partial (some clips failed,
errors printed) · 1 nothing transcribed · 2 editor unreachable.
"""
from __future__ import annotations

from typing import Optional


def _emit(text: str) -> None:
    print(text)


def _backend(args):
    from .perception.transcribe import detect_backend

    return detect_backend(getattr(args, "asr", None))


def _print_payload(payload: dict) -> int:
    clips = payload.get("clips", [])
    errors = payload.get("errors", [])
    for entry in clips:
        _emit(f"## {entry['id']} {entry['t']} {entry['src']} "
              f"[{entry['span'][0]}-{entry['span'][1]}]")
        for line in entry["phrases"]:
            _emit(line)
        if not entry["phrases"]:
            _emit("(no speech recognized in this clip's source window)")
    for err in errors:
        _emit(f"  ✗ {err}")
    if clips and not errors:
        return 0
    if clips and errors:
        return 3
    return 1


def cmd_transcribe(args) -> int:
    from .perception.transcribe import PerceptionUnavailable

    try:
        if getattr(args, "media", None):
            return _run_media(args)
        if getattr(args, "fixture", None):
            return _run_ir(args, _fixture_ir(args))
        return _run_live(args)
    except PerceptionUnavailable as exc:
        _emit(f"error: {exc}")
        return 1


# --------------------------------------------------------------------------- media file mode
def _run_media(args) -> int:
    from .perception.transcribe import pack_transcript, to_srt, transcribe_media

    transcript = transcribe_media(args.media, backend=_backend(args))
    if getattr(args, "srt", None):
        with open(args.srt, "w", encoding="utf-8") as fh:
            fh.write(to_srt(transcript))
        _emit(f"wrote {args.srt} ({len(transcript.words)} words, backend={transcript.backend})")
        return 0
    packed = pack_transcript(transcript)
    _emit(f"# {args.media} — {len(transcript.words)} words via {transcript.backend} "
          f"(media seconds)")
    _emit(packed if packed else "(no speech recognized)")
    return 0 if transcript.words else 1


# --------------------------------------------------------------------------- timeline modes
def _fixture_ir(args):
    from .cli_edit import FixtureError, load_fixture_ir

    try:
        return load_fixture_ir(args.fixture)
    except FixtureError as exc:
        raise SystemExit(_die(f"error: {exc}", 1))


def _die(msg: str, code: int) -> int:
    _emit(msg)
    return code


def _selected_ids(args, ir) -> list[str]:
    if getattr(args, "select", None):
        return [s.strip() for s in args.select.split(",") if s.strip()]
    return [c.id for c in ir.real_clips()]


def _run_ir(args, ir, ids: Optional[list[str]] = None) -> int:
    from .perception.align import aligned_srt, transcript_for_clips

    ids = ids if ids is not None else _selected_ids(args, ir)
    backend = _backend(args)
    if getattr(args, "srt", None):
        srt, errors = aligned_srt(ir, ids, backend=backend)
        with open(args.srt, "w", encoding="utf-8") as fh:
            fh.write(srt)
        _emit(f"wrote {args.srt} (timeline-time captions)")
        for err in errors:
            _emit(f"  ✗ {err}")
        return 0 if not errors else 3
    payload = transcript_for_clips(ir, ids, backend=backend)
    _emit(f"# transcript — timeline frames @ {payload['r']}")
    return _print_payload(payload)


def _run_live(args) -> int:
    if args.editor != "resolve":
        _emit(f"error: live editor '{args.editor}' not yet supported; use --editor resolve, "
              f"--fixture, or --media.")
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
    ids = selection.ids or [c.id for c in ir.real_clips()]
    if not selection.ids:
        _emit("# no selection — transcribing every clip on the timeline")
    return _run_ir(args, ir, ids)
