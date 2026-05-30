"""Shared pytest configuration for film-grip.

Sets the three DaVinci Resolve scripting environment variables (macOS paths) before any
test imports the Resolve client, so live-integration tests can connect when the app is
running. The variables are harmless when Resolve is closed — ``scriptapp('Resolve')``
simply returns ``None`` and the live tests skip themselves.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the repo's tests/ importable as a package root (for tests.fakes, fixtures helpers).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

_MAC_RESOLVE_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
_MAC_RESOLVE_LIB = (
    "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"
)


def _default_env() -> None:
    """Populate Resolve env vars if unset and the default macOS paths exist."""
    if not os.environ.get("RESOLVE_SCRIPT_API") and Path(_MAC_RESOLVE_API).exists():
        os.environ["RESOLVE_SCRIPT_API"] = _MAC_RESOLVE_API
    if not os.environ.get("RESOLVE_SCRIPT_LIB") and Path(_MAC_RESOLVE_LIB).exists():
        os.environ["RESOLVE_SCRIPT_LIB"] = _MAC_RESOLVE_LIB
    api = os.environ.get("RESOLVE_SCRIPT_API")
    if api:
        modules = str(Path(api) / "Modules")
        current = os.environ.get("PYTHONPATH", "")
        if modules not in current.split(os.pathsep):
            os.environ["PYTHONPATH"] = (current + os.pathsep + modules) if current else modules


_default_env()


@pytest.fixture(scope="session")
def fixtures_dir() -> Path:
    return Path(__file__).resolve().parent / "fixtures"
