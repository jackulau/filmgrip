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
    edit.add_argument("--dry-run", action="store_true", help="validate + print the diff without applying")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "edit":
        # Imported lazily — keeps --version/--help working on a core-only install.
        from .cli_edit import cmd_edit

        return cmd_edit(args)
    parser.print_help()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
