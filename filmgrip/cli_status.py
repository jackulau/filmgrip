"""``film-grip status`` — the doctor command.

One place a user can run to see whether film-grip can actually reach their editor, what's wrong if
not, and what it can do. It never raises and always exits 0 — it's a diagnostic, not an operation.
"""
from __future__ import annotations


def _yn(v) -> str:
    if v is None:
        return "?"
    return "yes" if v else "no"


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
    return ["Ready — try: film-grip edit \"add a blue marker on the selected clip\""]


def render_status(report: dict, sfx_summary: str, editor_count: int) -> str:
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
    lines.append(f"SFX library : {sfx_summary}")
    lines.append(f"Editors     : {editor_count} supported (run `film-grip editors` for the matrix)")
    return "\n".join(lines)


def cmd_status(args) -> int:
    from .adapters import registry
    from .adapters import resolve_client as rc
    from .audio.library import SfxLibrary

    report = rc.preflight()
    lib = SfxLibrary.load(getattr(args, "sfx_dir", None) or None)
    missing = f", {len(lib.missing)} missing file(s)" if lib.missing else ""
    sfx_summary = f"{lib.base} — {len(lib.entries)} effect(s){missing}"

    print(render_status(report, sfx_summary, len(registry.editors())))
    return 0  # diagnostic: always succeeds
