"""DaVinci Resolve connection + a thin, injectable façade.

The Resolve scripting module (``DaVinciResolveScript`` / ``fusionscript``) is a native CPython
extension loaded from the Resolve install. Two facts drive this module's design, both verified
against the local install:

1. The module imports fine even when Resolve is **not running**; ``scriptapp('Resolve')`` then
   returns ``None``. So :func:`connect` must degrade gracefully to ``None`` rather than raise.
2. The Resolve API fails **silently** — bad keys / failed ops return falsy values, not
   exceptions. Callers must check returns; helpers here make that explicit.

Every Resolve call goes through :class:`ResolveSession`, and :func:`connect` accepts an injected
``resolve`` object, so the whole façade is drivable by a fake in tests with no app installed.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

# Default scripting paths per platform (the env-var values from the Resolve README).
_DEFAULTS: dict[str, dict[str, str]] = {
    "darwin": {
        "api": "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting",
        "lib": "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so",
    },
    "win32": {
        "api": r"%PROGRAMDATA%\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting",
        "lib": r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll",
    },
    "linux": {
        "api": "/opt/resolve/Developer/Scripting",
        "lib": "/opt/resolve/libs/Fusion/fusionscript.so",
    },
}


class ResolveUnavailable(RuntimeError):
    """Raised when the Resolve scripting module cannot be imported at all."""


class ResolveOperationFailed(RuntimeError):
    """Raised when a Resolve API call returns a falsy value (Resolve fails silently)."""


def _platform_key() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    return sys.platform


def configure_env(*, force: bool = False) -> dict[str, str]:
    """Populate RESOLVE_SCRIPT_API / RESOLVE_SCRIPT_LIB / PYTHONPATH if unset.

    Returns the resulting values (whatever is in the environment afterward). Only fills a
    variable when it is unset (or ``force=True``) and the default path exists, so a user's
    explicit configuration is never clobbered.
    """
    defaults = _DEFAULTS.get(_platform_key(), {})
    api = os.environ.get("RESOLVE_SCRIPT_API")
    if (force or not api) and defaults.get("api") and Path(os.path.expandvars(defaults["api"])).exists():
        api = os.path.expandvars(defaults["api"])
        os.environ["RESOLVE_SCRIPT_API"] = api
    lib = os.environ.get("RESOLVE_SCRIPT_LIB")
    if (force or not lib) and defaults.get("lib") and Path(os.path.expandvars(defaults["lib"])).exists():
        lib = os.path.expandvars(defaults["lib"])
        os.environ["RESOLVE_SCRIPT_LIB"] = lib
    if api:
        modules = str(Path(api) / "Modules")
        parts = os.environ.get("PYTHONPATH", "").split(os.pathsep) if os.environ.get("PYTHONPATH") else []
        if modules not in parts:
            parts.append(modules)
            os.environ["PYTHONPATH"] = os.pathsep.join(p for p in parts if p)
        if modules not in sys.path:
            sys.path.append(modules)
    return {
        "RESOLVE_SCRIPT_API": os.environ.get("RESOLVE_SCRIPT_API", ""),
        "RESOLVE_SCRIPT_LIB": os.environ.get("RESOLVE_SCRIPT_LIB", ""),
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }


def load_module() -> Any:
    """Import and return the ``DaVinciResolveScript`` module, or raise :class:`ResolveUnavailable`."""
    configure_env()
    try:
        import DaVinciResolveScript as dvr  # type: ignore

        return dvr
    except Exception as exc:  # pragma: no cover - depends on local install
        raise ResolveUnavailable(
            "Could not import DaVinciResolveScript. Ensure DaVinci Resolve (Studio) is installed "
            "and RESOLVE_SCRIPT_API / RESOLVE_SCRIPT_LIB are set."
        ) from exc


def require(value: Any, message: str) -> Any:
    """Raise :class:`ResolveOperationFailed` if ``value`` is falsy (None / False / '').

    Resolve returns falsy on failure instead of raising; wrap every load-bearing call so a
    silent failure aborts loudly instead of corrupting an apply transaction.
    """
    if not value:
        raise ResolveOperationFailed(message)
    return value


@dataclass
class ResolveSession:
    """Thin façade over the raw Resolve object graph.

    Holds the connected ``resolve`` handle and exposes only the small surface film-grip needs.
    Constructed by :func:`connect`; never instantiate against a ``None`` handle.
    """

    resolve: Any
    _project_manager: Any = field(default=None, repr=False)

    # -- identity ---------------------------------------------------------------
    def product_name(self) -> str:
        return self.resolve.GetProductName() or "DaVinci Resolve"

    def version(self) -> str:
        return self.resolve.GetVersionString() or "unknown"

    def is_studio(self) -> bool:
        return "Studio" in (self.resolve.GetProductName() or "")

    # -- navigation -------------------------------------------------------------
    def project_manager(self) -> Any:
        if self._project_manager is None:
            self._project_manager = require(
                self.resolve.GetProjectManager(), "GetProjectManager() returned None"
            )
        return self._project_manager

    def current_project(self) -> Optional[Any]:
        return self.project_manager().GetCurrentProject()

    def current_timeline(self) -> Optional[Any]:
        proj = self.current_project()
        if not proj:
            return None
        return proj.GetCurrentTimeline()

    def media_pool(self) -> Optional[Any]:
        proj = self.current_project()
        if not proj:
            return None
        return proj.GetMediaPool()

    def timeline_items(self, track_type: str, index: int, timeline: Any = None) -> list[Any]:
        """Return the items on one track (``track_type`` in {'video','audio','subtitle'})."""
        tl = timeline or self.current_timeline()
        if not tl:
            return []
        return list(tl.GetItemListInTrack(track_type, index) or [])

    def open_page(self, page: str) -> bool:
        return bool(self.resolve.OpenPage(page))


def connect(resolve: Any = None) -> Optional[ResolveSession]:
    """Connect to a running DaVinci Resolve and return a :class:`ResolveSession`, or ``None``.

    Args:
        resolve: an injected resolve-like object (used by tests). When ``None``, the real
            scripting module is loaded and ``scriptapp('Resolve')`` is called.

    Returns ``None`` (never raises) when Resolve is installed but not running, so callers can
    cleanly report "open Resolve first" instead of crashing. Raises :class:`ResolveUnavailable`
    only when the scripting module itself cannot be imported.
    """
    if resolve is None:
        dvr = load_module()
        resolve = dvr.scriptapp("Resolve")
    if not resolve:
        return None
    return ResolveSession(resolve=resolve)


def preflight() -> dict[str, Any]:
    """Diagnose the live Resolve scripting path without raising.

    Returns a dict describing what's reachable — used by the CLI to give the user an honest,
    actionable status (module importable? app running? Studio? project open?).
    """
    report: dict[str, Any] = {
        "env": configure_env(),
        "module_importable": False,
        "app_running": False,
        "is_studio": None,
        "product": None,
        "version": None,
        "project_open": False,
        "timeline_open": False,
    }
    try:
        load_module()
        report["module_importable"] = True
    except ResolveUnavailable:
        return report
    try:
        session = connect()
    except ResolveUnavailable:
        return report
    if session is None:
        return report
    report["app_running"] = True
    report["is_studio"] = session.is_studio()
    report["product"] = session.product_name()
    report["version"] = session.version()
    proj = session.current_project()
    report["project_open"] = bool(proj)
    report["timeline_open"] = bool(session.current_timeline())
    return report
