"""film-grip command-line entry point.

D1 ships only the argument skeleton so the ``film-grip`` console script resolves at install
time and ``film-grip --version`` works. The real ``edit`` pipeline is wired in D10
(:func:`filmgrip.cli.cmd_edit`), which imports the adapters/protocol/integration layers
lazily so a partial install still exposes ``--help``/``--version``.
"""
from __future__ import annotations

import argparse
import sys

from . import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="film-grip",
        description="react-grab for video editors — select clips, prompt, Claude edits.",
    )
    parser.add_argument("--version", action="version", version=f"film-grip {__version__}")
    sub = parser.add_subparsers(dest="command")

    edit = sub.add_parser("edit", help="apply a natural-language edit to the active timeline")
    edit.add_argument("prompt", nargs="?", default="", help="natural-language edit instruction")
    edit.add_argument("--editor", default="resolve", help="target editor (default: resolve)")
    edit.add_argument("--fixture", help="run the pipeline against a .otio fixture instead of a live editor")
    edit.add_argument("--plan", help="apply a recorded EditPlan JSON instead of calling Claude (offline e2e)")
    edit.add_argument("--backend", help="planner backend (default: claude; or $FILMGRIP_BACKEND)")
    edit.add_argument("--out", help="output path for fixture/interchange apply (default: <name>.edited.<ext>)")
    edit.add_argument("--dry-run", action="store_true", help="validate + print the diff without applying")

    grab = sub.add_parser(
        "grab",
        help="capture the selected clips as a compact <selected_clips> context block (react-grab "
             "flow) and copy it to the clipboard",
    )
    grab.add_argument("--editor", default="resolve", help="target editor (default: resolve)")
    grab.add_argument("--fixture", help="grab from a .otio fixture instead of a live editor")
    grab.add_argument("--select", help="comma-separated clip IDs to grab in fixture mode "
                                        "(default: all clips)")
    grab.add_argument("--no-neighbors", action="store_true",
                      help="omit the same-track neighbor context")
    grab.add_argument("--no-copy", action="store_true",
                      help="print only; do not copy to the clipboard")

    pack = sub.add_parser("pack", help="list / show / apply named edit recipes (packs)")
    pack.add_argument("pack_action", nargs="?", choices=["list", "show", "apply"], default="list",
                      help="list packs (default), show one, or apply one")
    pack.add_argument("name", nargs="?", default="", help="pack name (for show / apply)")
    pack.add_argument("--editor", default="resolve", help="target editor for live apply (default: resolve)")
    pack.add_argument("--fixture", help="apply against a .otio fixture instead of a live editor")
    pack.add_argument("--select", help="comma-separated clip IDs in fixture mode (default: all clips)")
    pack.add_argument("--dry-run", action="store_true", help="validate + print the diff without applying")
    pack.add_argument("--out", help="output path for fixture apply (default: <name>.edited.<ext>)")

    sub.add_parser("editors", help="print the editor capability matrix (what film-grip can/can't do)")

    st = sub.add_parser("status", help="diagnose whether film-grip can reach your editor (the doctor)")
    st.add_argument("--sfx-dir", dest="sfx_dir", help="SFX folder to report (default: ~/.filmgrip/sfx)")

    panel = sub.add_parser("panel", help="install the in-Resolve panel (Workspace ▸ Scripts ▸ film-grip)")
    panel.add_argument("panel_action", nargs="?", choices=["install"], default="install")
    panel.add_argument("--dry-run", action="store_true", help="show the install target without writing")
    panel.add_argument("--dir", help="override Resolve's Edit-page Scripts folder")

    sfx = sub.add_parser("sfx", help="inspect the sound-effects library Claude can pull from")
    sfx.add_argument("sfx_action", nargs="?", choices=["list", "scan", "resolve"], default="list",
                     help="list effects (default), scan the folder into a manifest, or resolve a description")
    sfx.add_argument("query", nargs="?", default="", help="description to resolve (with 'resolve')")
    sfx.add_argument("--dir", help="SFX folder (default: $FILMGRIP_SFX_DIR or ~/.filmgrip/sfx)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "edit":
        # Imported lazily — keeps --version/--help working on a core-only install.
        from .cli_edit import cmd_edit

        return cmd_edit(args)
    if args.command == "grab":
        from .cli_grab import cmd_grab

        return cmd_grab(args)
    if args.command == "pack":
        from .cli_pack import cmd_pack

        return cmd_pack(args)
    if args.command == "editors":
        from .adapters.registry import capability_markdown

        print(capability_markdown())
        return 0
    if args.command == "sfx":
        from .audio.library import cmd_sfx

        return cmd_sfx(args)
    if args.command == "status":
        from .cli_status import cmd_status

        return cmd_status(args)
    if args.command == "panel":
        from .cli_panel import cmd_panel

        return cmd_panel(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
