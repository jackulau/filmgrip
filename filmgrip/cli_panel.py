"""``film-grip panel install`` — drop the in-Resolve panel into Resolve's Scripts menu.

Copies ``scripts/film_grip_resolve_panel.py`` into Resolve's ``Fusion/Scripts/Edit`` folder (so it
appears under Workspace ▸ Scripts), baking in the filmgrip site path so the installed copy — which
lives outside the repo — can import the package. ``--dry-run`` shows the target without writing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Resolve's per-page script folders (Edit page) by platform — from the Scripting README.
_SCRIPTS_DIRS = {
    "darwin": "~/Library/Application Support/Blackmagic Design/DaVinci Resolve/Fusion/Scripts/Edit",
    "win32": r"%APPDATA%\Blackmagic Design\DaVinci Resolve\Support\Fusion\Scripts\Edit",
    "linux": "~/.local/share/DaVinciResolve/Fusion/Scripts/Edit",
}


def default_scripts_dir() -> Path:
    key = "linux" if sys.platform.startswith("linux") else sys.platform
    raw = _SCRIPTS_DIRS.get(key, _SCRIPTS_DIRS["darwin"])
    return Path(os.path.expandvars(os.path.expanduser(raw)))


def filmgrip_site() -> str:
    """The directory to put on sys.path so ``import filmgrip`` works (parent of the package)."""
    import filmgrip
    return str(Path(filmgrip.__file__).resolve().parent.parent)


def panel_source_path() -> Path:
    """Locate the repo's panel script (sibling ``scripts/`` of the installed/edited package)."""
    return Path(filmgrip_site()) / "scripts" / "film_grip_resolve_panel.py"


def _stamped_source(src_text: str, site: str) -> str:
    """Bake the filmgrip site path into the FILMGRIP_SITE constant of the installed copy."""
    out_lines = []
    for line in src_text.splitlines():
        if line.startswith("FILMGRIP_SITE = "):
            out_lines.append(f"FILMGRIP_SITE = {site!r}")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + "\n"


def install_panel(*, dry_run: bool = False, dest_dir: str | None = None) -> int:
    src = panel_source_path()
    target_dir = Path(dest_dir).expanduser() if dest_dir else default_scripts_dir()
    target = target_dir / "film-grip.py"
    site = filmgrip_site()

    if not src.is_file():
        print(f"error: panel script not found at {src} "
              f"(install film-grip from a source checkout to get scripts/).")
        return 1

    if dry_run:
        print(f"would install film-grip panel → {target}")
        print(f"  source     : {src}")
        print(f"  filmgrip at: {site}")
        print("  then launch it in Resolve: Workspace ▸ Scripts ▸ film-grip")
        return 0

    target_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(_stamped_source(src.read_text(encoding="utf-8"), site), encoding="utf-8")
    print(f"installed film-grip panel → {target}")
    print("  launch it in Resolve: Workspace ▸ Scripts ▸ film-grip")
    return 0


def cmd_panel(args) -> int:
    action = getattr(args, "panel_action", None) or "install"
    if action == "install":
        return install_panel(dry_run=bool(getattr(args, "dry_run", False)),
                             dest_dir=getattr(args, "dir", None))
    print(f"unknown panel action '{action}'")
    return 2
