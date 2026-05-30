"""Adapter registry + capability matrix — film-grip's honesty surface.

One adapter often backs several editors (the interchange adapter reaches Final Cut, Premiere and
Avid; the MLT adapter reaches Kdenlive and Shotcut). The registry maps each *editor* to its
adapter and a capability record tailored to that editor, so the CLI/MCP can tell the user exactly
what is and isn't possible per editor — live vs interchange vs best-effort vs read-only — instead
of implying uniform support. Generating the README/CLI table from this single source keeps the
promise and the reality in sync.
"""
from __future__ import annotations

from dataclasses import dataclass

from .base import Capabilities, GrabAdapter
from .capcut import CapcutAdapter
from .filmora import FilmoraAdapter
from .interchange import InterchangeAdapter
from .mlt import MltAdapter
from .resolve_adapter import ResolveAdapter


@dataclass
class EditorEntry:
    slug: str
    adapter: GrabAdapter
    capability: Capabilities
    extensions: tuple[str, ...] = ()


def _cap(editor, role, mechanism, *, live, write, app, lossy,
         audio="interchange", organize="interchange-warn", panel="none",
         confidence="precise", volume_scriptable=False) -> Capabilities:
    return Capabilities(editor=editor, role=role, mechanism=mechanism, live_selection=live,
                        write_back=write, requires_app_running=app, lossy_features=list(lossy),
                        audio_support=audio, audio_volume_scriptable=volume_scriptable,
                        organize_support=organize, editor_panel=panel,
                        selection_confidence=confidence)


def _build_registry() -> dict[str, EditorEntry]:
    interchange = InterchangeAdapter()
    mlt = MltAdapter()
    entries = [
        EditorEntry("resolve", ResolveAdapter(), ResolveAdapter().capabilities()),
        EditorEntry(
            "finalcut", interchange,
            _cap("Final Cut Pro", "interchange", "FCPXML round-trip (File ▸ Export/Import XML)",
                 live=False, write=True, app=False, lossy=["transitions (varies)", "effects"]),
            (".fcpxml", ".xml")),
        EditorEntry(
            "premiere", interchange,
            _cap("Premiere Pro", "interchange", "FCP7 XML / AAF round-trip (UXP panel = future path)",
                 live=False, write=True, app=False, lossy=["transitions", "effects", "speed"],
                 panel="uxp-future"),
            (".xml", ".aaf")),
        EditorEntry(
            "avid", interchange,
            _cap("Avid Media Composer", "best-effort", "AAF interchange (relink/conform)",
                 live=False, write=True, app=False,
                 lossy=["effects", "AAF can import offline — manual relink, not clean automation"]),
            (".aaf",)),
        EditorEntry(
            "kdenlive", mlt,
            _cap("Kdenlive", "interchange", "native MLT XML (.kdenlive) parse/rewrite",
                 live=False, write=True, app=False, lossy=["effects/filters", "transitions"]),
            (".kdenlive",)),
        EditorEntry(
            "shotcut", mlt,
            _cap("Shotcut", "interchange", "native MLT XML (.mlt) parse/rewrite",
                 live=False, write=True, app=False, lossy=["effects/filters", "transitions"]),
            (".mlt",)),
        EditorEntry("capcut", CapcutAdapter(), CapcutAdapter().capabilities(), (".json",)),
        EditorEntry("filmora", FilmoraAdapter(), FilmoraAdapter().capabilities(), (".wfp",)),
    ]
    return {e.slug: e for e in entries}


REGISTRY: dict[str, EditorEntry] = _build_registry()


def get(slug: str) -> EditorEntry:
    if slug not in REGISTRY:
        raise KeyError(f"unknown editor '{slug}'; one of {sorted(REGISTRY)}")
    return REGISTRY[slug]


def editors() -> list[str]:
    return list(REGISTRY)


def capability_matrix() -> list[Capabilities]:
    return [e.capability for e in REGISTRY.values()]


def for_extension(ext: str) -> EditorEntry | None:
    ext = ext.lower()
    for e in REGISTRY.values():
        if ext in e.extensions:
            return e
    return None


def capability_markdown() -> str:
    """Render the honest capability table (also written to docs by ``write_capabilities_doc``)."""
    head = ("| Editor | Role | Write-back | Audio | Organize | In-app panel | Selection | Mechanism |\n"
            "|---|---|---|---|---|---|---|---|")
    rows = []
    for e in REGISTRY.values():
        c = e.capability
        rows.append(
            f"| {c.editor} | {c.role} | {'yes' if c.write_back else 'NO'} | "
            f"{c.audio_support} | {c.organize_support} | {c.editor_panel} | "
            f"{c.selection_confidence} | {c.mechanism} |")
    return "\n".join([head, *rows])


def op_support_markdown() -> str:
    """Per-op support so users know which edits actually land where (live vs rebuild vs interchange)."""
    from .resolve_adapter import LIVE_EXTRA_OPS, LIVE_OPS

    head = ("| Op | Resolve (live) | Resolve (rebuild) | Interchange file |\n"
            "|---|---|---|---|")
    notes = {
        "trim": ("rebuild", "yes", "yes"), "move": ("rebuild", "yes", "yes"),
        "split": ("rebuild", "yes", "yes"), "insert": ("rebuild", "yes", "yes"),
        "ripple": ("rebuild", "yes", "yes"), "delete": ("yes", "yes", "yes"),
        "add_marker": ("yes", "yes", "yes"), "set_property": ("yes", "yes", "yes (metadata)"),
        "add_transition": ("no", "no", "no (do in editor)"),
        "import_audio": ("yes", "n/a", "no"), "add_track": ("yes", "n/a", "no"),
        "rename_track": ("yes", "n/a", "no"), "create_bin": ("yes", "n/a", "no"),
        "move_to_bin": ("yes", "n/a", "no"),
    }
    rows = []
    for op, (live, rebuild, inter) in notes.items():
        rows.append(f"| {op} | {live} | {rebuild} | {inter} |")
    _ = (LIVE_OPS, LIVE_EXTRA_OPS)  # keep the table honest against the live sets
    return "\n".join([head, *rows])


def write_capabilities_doc(path: str = "docs/CAPABILITIES.md") -> str:
    import os

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    body = ("# film-grip — editor capability matrix\n\n"
            "Generated from `filmgrip.adapters.registry`. film-grip surfaces these so it never "
            "promises an editor automation it cannot deliver.\n\n"
            "**Audio**: live = import+place on an audio track via scripting; interchange/offline = via "
            "file round-trip; read-only = parsed, not written. **Per-clip volume/gain/fades are NOT "
            "scriptable in Resolve** (Fairlight-only) — film-grip places audio, levels stay manual.\n\n"
            + capability_markdown() + "\n\n"
            "## Which ops land where\n\n" + op_support_markdown() + "\n")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


if __name__ == "__main__":  # pragma: no cover
    print("wrote", write_capabilities_doc())
