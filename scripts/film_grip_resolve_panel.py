#!/usr/bin/env python
"""film-grip - in-Resolve panel.

Install into Resolve's ``Fusion/Scripts/Edit`` folder (``film-grip panel install``) and launch it from
**Workspace > Scripts > film-grip** inside DaVinci Resolve. It opens a floating panel where you see
your current selection, type a natural-language instruction, and hit Apply (or Dry-run) - film-grip
plans the edit and applies it through the tested core.

This script is a THIN translator: all behaviour lives in ``filmgrip.ui.panel`` (unit-tested with no
Resolve). The module imports cleanly anywhere (the Resolve globals ``resolve``/``bmd`` and the Fusion
UIManager are looked up lazily and guarded), so it can be byte-compiled in CI without Resolve open.
"""
from __future__ import annotations

import os
import sys

# `film-grip panel install` bakes the filmgrip site path here so the installed copy (which lives
# outside the repo, in Resolve's Scripts folder) can import the package.
FILMGRIP_SITE = ""


def _ensure_filmgrip_importable() -> bool:
    for candidate in (FILMGRIP_SITE, os.environ.get("FILMGRIP_SITE", "")):
        if candidate and candidate not in sys.path and os.path.isdir(candidate):
            sys.path.insert(0, candidate)
    try:
        import filmgrip  # noqa: F401
        return True
    except Exception:
        return False


def _get_resolve():
    """Resolve injects a `resolve` global into Scripts; fall back to the scripting module."""
    r = globals().get("resolve")
    if r is not None:
        return r
    try:
        import DaVinciResolveScript as dvr  # type: ignore
        return dvr.scriptapp("Resolve")
    except Exception:
        return None


def _get_ui():
    """Return (UIManager, UIDispatcher) or (None, None) when not running inside Resolve/Fusion."""
    bmd = globals().get("bmd")
    if bmd is None:
        try:
            import BlackmagicFusion as bmd  # type: ignore
        except Exception:
            return None, None
    fu = bmd.scriptapp("Fusion") if hasattr(bmd, "scriptapp") else None
    if fu is None or not getattr(fu, "UIManager", None):
        return None, None
    return fu.UIManager, bmd.UIDispatcher(fu.UIManager)


def _build_and_run(ui, disp, controller) -> None:
    from filmgrip.ui.panel import ID_APPLY, ID_DRYRUN, ID_OUTPUT, ID_PROMPT, ID_SELECTION

    win = disp.AddWindow(
        {"ID": "fg_window", "WindowTitle": "film-grip", "Geometry": [200, 200, 540, 380]},
        ui.VGroup([
            ui.Label({"ID": ID_SELECTION, "Text": controller.selection_summary(), "WordWrap": True}),
            ui.Label({"Text": "Describe the edit (your selected clips are already in context):"}),
            ui.TextEdit({"ID": ID_PROMPT, "PlaceholderText":
                         "e.g. add a whoosh as the title flies in, then tighten the open by 12 frames"}),
            ui.HGroup([
                ui.Button({"ID": ID_APPLY, "Text": "Apply"}),
                ui.Button({"ID": ID_DRYRUN, "Text": "Dry-run"}),
            ]),
            ui.Label({"ID": ID_OUTPUT, "Text": "", "WordWrap": True}),
        ]),
    )
    items = win.GetItems()

    def _prompt_text() -> str:
        box = items[ID_PROMPT]
        return getattr(box, "PlainText", None) or getattr(box, "Text", "") or ""

    def _run(dry: bool):
        result = controller.on_dry_run(_prompt_text()) if dry else controller.on_apply(_prompt_text())
        items[ID_OUTPUT].Text = result.text

    win.On["fg_window"].Close = lambda ev: disp.ExitLoop()
    win.On[ID_APPLY].Clicked = lambda ev: _run(False)
    win.On[ID_DRYRUN].Clicked = lambda ev: _run(True)

    win.Show()
    disp.RunLoop()
    win.Hide()


def main() -> int:
    if not _ensure_filmgrip_importable():
        print("film-grip is not importable. Re-run `film-grip panel install` so the panel knows "
              "where the package lives, or set $FILMGRIP_SITE.")
        return 1

    from filmgrip.adapters import resolve_client as rc
    from filmgrip.ui.panel import live_controller

    resolve = _get_resolve()
    session = rc.connect(resolve)
    if session is None or session.current_timeline() is None:
        print("Open DaVinci Resolve (Studio) with a project + timeline, then launch the panel.")
        return 2

    ui, disp = _get_ui()
    if ui is None:
        print("Fusion UIManager is unavailable - launch this from inside Resolve "
              "(Workspace > Scripts > film-grip).")
        return 3

    _build_and_run(ui, disp, live_controller(session))
    return 0


if __name__ == "__main__":
    sys.exit(main())
