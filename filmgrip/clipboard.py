"""Best-effort clipboard copy — a convenience, never a contract.

``film-grip grab`` and the in-Resolve panel's *Copy context* put the ``<selected_clips>`` block on
the OS clipboard so it pastes straight into an external agent. If no clipboard tool is present
(headless CI, a stripped container), we degrade silently: the block is *always* printed to stdout,
and :func:`copy` simply returns ``False``. Copying must never break the actual capture, so every
failure path is swallowed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def _candidates() -> list[list[str]]:
    """Clipboard-writer commands to try, in order, for the current platform."""
    if sys.platform == "darwin":
        return [["pbcopy"]]
    if sys.platform.startswith("win"):
        return [["clip"]]
    # Linux/BSD — Wayland first, then the common X tools.
    return [["wl-copy"], ["xclip", "-selection", "clipboard"], ["xsel", "--clipboard", "--input"]]


def copy(text: str) -> bool:
    """Copy ``text`` to the OS clipboard. Return ``True`` on success, ``False`` if nothing worked.

    Never raises — a missing clipboard tool or a write error is reported as ``False``, not an
    exception, because the grabbed block is already on stdout regardless.
    """
    for cmd in _candidates():
        if shutil.which(cmd[0]) is None:
            continue
        try:
            done = subprocess.run(cmd, input=text.encode("utf-8"), check=False)
            if done.returncode == 0:
                return True
        except Exception:
            continue
    return False
