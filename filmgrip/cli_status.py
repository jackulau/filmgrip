"""``film-grip status`` — the doctor command.

One place a user can run to see whether film-grip can actually reach their editor *and* whether the
perception toolchain (ffmpeg, ffprobe, numpy, an ASR backend) the ``scopes``/``frames``/``transcribe``
subcommands depend on is present — with the exact command to fix each gap. It never raises and always
exits 0 by default — it's a diagnostic, not an operation (opt into ``--exit-code`` to gate CI on a
hard blocker).
"""
from __future__ import annotations

import json as _json

#: Glyphs per severity (flutter-doctor style): ok / hard-blocker / warning.
GLYPH = {"ok": "✓", "fail": "✗", "warn": "!"}


def _yn(v) -> str:
    if v is None:
        return "?"
    return "yes" if v else "no"


def actionable(what: str, *, why: str = "", fix: str = "", see: str = "") -> list[str]:
    """The house actionable-message format (WHAT / WHY / FIX / SEE), as lines.

    Used by the doctor for failing rows; factored here so the live-edit error paths can speak with
    the same voice (per the cinematography-ux research's "one identical, fix-carrying message" rule).
    """
    lines = [f"error: {what}"]
    if why:
        lines.append(f"  why: {why}")
    if fix:
        lines.append(f"  fix: {fix}")
    if see:
        lines.append(f"  see: {see}")
    return lines


def status_guidance(report: dict) -> list[str]:
    """The single most useful next step given the preflight state (ordered, most-blocking first)."""
    if not report.get("module_importable"):
        return ["Install DaVinci Resolve (Studio), or set RESOLVE_SCRIPT_API / RESOLVE_SCRIPT_LIB "
                "to your install's Developer/Scripting paths."]
    if not report.get("app_running"):
        return ["Open DaVinci Resolve (Studio) and enable Preferences ▸ System ▸ General ▸ "
                "'External scripting using' = Local."]
    if report.get("is_studio") is False:
        return ["This looks like the free edition — Resolve scripting needs the Studio version."]
    if not report.get("project_open"):
        return ["Open (or create) a project in Resolve."]
    if not report.get("timeline_open"):
        return ["Open a timeline in the project."]
    return ["Ready — select a clip on the timeline (or a media-pool item), then try: "
            "film-grip edit \"add a blue marker on the selected clip\""]


def _check(name: str, status: str, detail: str, fix: str = "") -> dict:
    """One doctor row: ``status`` is 'ok' | 'warn' | 'fail'; ``fix`` is the exact command (if any)."""
    return {"name": name, "ok": status == "ok", "status": status, "detail": detail, "fix": fix}


def perception_checks() -> list[dict]:
    """Probe the perception toolchain (ffmpeg, ffprobe, numpy, ASR backend), most-blocking first.

    Honest by construction: every result comes from a real ``shutil.which`` / import / backend probe,
    never an assumption. Ordered most-blocking-first — ffmpeg gates audio extraction + contact sheets
    + scopes; numpy gates scopes; the ASR backend gates only ``transcribe``.
    """
    import shutil

    from .perception import transcribe as tx

    checks: list[dict] = []

    # ffmpeg — most blocking: audio extraction, contact sheets, scopes all shell out to it.
    ffmpeg = tx.ffmpeg_path()
    checks.append(_check(
        "ffmpeg", "ok" if ffmpeg else "fail",
        ffmpeg if ffmpeg else "not on PATH (needed for scopes, frames, transcribe)",
        "" if ffmpeg else "brew install ffmpeg"))

    # ffprobe — duration probing; degrades (falls back to last-word end) so it is a warning, not fatal.
    ffprobe = shutil.which("ffprobe")
    checks.append(_check(
        "ffprobe", "ok" if ffprobe else "warn",
        ffprobe if ffprobe else "not on PATH (media duration falls back to last cue)",
        "" if ffprobe else "brew install ffmpeg"))

    # numpy — required by scopes (and the proposed composition/pacing readers).
    try:
        import numpy  # noqa: F401
        numpy_detail, numpy_status = getattr(numpy, "__version__", "present"), "ok"
    except ImportError:
        numpy_detail, numpy_status = "not importable (needed for scopes / color analysis)", "fail"
    checks.append(_check(
        "numpy", numpy_status, numpy_detail,
        "" if numpy_status == "ok" else "pip install 'film-grip[color]'"))

    # ASR backend — gates `transcribe` only; reuse transcribe.detect_backend's honest auto-detect.
    try:
        backend = tx.detect_backend()
        checks.append(_check("asr_backend", "ok", f"{backend.name} available"))
    except tx.PerceptionUnavailable:
        checks.append(_check(
            "asr_backend", "warn",
            "none of faster-whisper / whisper-cpp / elevenlabs available (transcribe only)",
            "pip install 'film-grip[transcribe]'"))

    return checks


def render_status(report: dict, sfx_summary: str, editor_count: int, auth=None,
                  checks: list[dict] | None = None) -> str:
    lines = ["film-grip — status", "=" * 42]
    lines.append(f"Resolve scripting module : {_yn(report.get('module_importable'))}")
    lines.append(f"Resolve app running      : {_yn(report.get('app_running'))}")
    if report.get("app_running"):
        lines.append(f"  product / version      : {report.get('product')} {report.get('version')}")
        lines.append(f"  Studio (scriptable)    : {_yn(report.get('is_studio'))}")
        lines.append(f"  project open           : {_yn(report.get('project_open'))}")
        lines.append(f"  timeline open          : {_yn(report.get('timeline_open'))}")
    for msg in status_guidance(report):
        lines.append(f"  → {msg}")
    lines.append("")

    if checks:
        lines.append("Perception toolchain (scopes / frames / transcribe):")
        for c in checks:
            glyph = GLYPH.get(c["status"], "?")
            lines.append(f"  {glyph} {c['name']:<12} {c['detail']}")
            if c.get("fix"):
                lines.append(f"      fix: {c['fix']}")
        lines.append("")

    if auth is not None:
        lines.append(f"LLM auth    : {auth.method} — {auth.detail}")
        if auth.hint:
            lines.append(f"  → {auth.hint}")
    lines.append(f"SFX library : {sfx_summary}")
    lines.append(f"Editors     : {editor_count} supported (run `film-grip editors` for the matrix)")
    return "\n".join(lines)


def build_report(args) -> dict:
    """Assemble the full machine-readable doctor report (editor preflight + perception + sfx + auth)."""
    from .adapters import registry
    from .adapters import resolve_client as rc
    from .audio.library import SfxLibrary
    from .integration.auth import detect_auth

    report = rc.preflight()
    lib = SfxLibrary.load(getattr(args, "sfx_dir", None) or None)
    auth = detect_auth()
    checks = perception_checks()
    return {
        "resolve": report,
        "guidance": status_guidance(report),
        "perception": checks,
        "sfx": {"base": str(lib.base), "effects": len(lib.entries), "missing": len(lib.missing)},
        "auth": {"method": auth.method, "detail": auth.detail, "hint": auth.hint},
        "editors": len(registry.editors()),
        # True only when nothing is a hard blocker (warnings are fine). Lets CI gate via --exit-code.
        "ok": not any(c["status"] == "fail" for c in checks),
    }


def cmd_status(args) -> int:
    data = build_report(args)

    if getattr(args, "json", False):
        print(_json.dumps(data, indent=2))
    else:
        sfx = data["sfx"]
        missing = f", {sfx['missing']} missing file(s)" if sfx["missing"] else ""
        sfx_summary = f"{sfx['base']} — {sfx['effects']} effect(s){missing}"
        from .integration.auth import AuthStatus

        auth = AuthStatus(**data["auth"])
        print(render_status(data["resolve"], sfx_summary, data["editors"], auth=auth,
                            checks=data["perception"]))

    # Diagnostic: exit 0 by default. Only --exit-code makes a hard blocker non-zero (CI gate).
    if getattr(args, "exit_code", False) and not data["ok"]:
        return 1
    return 0
