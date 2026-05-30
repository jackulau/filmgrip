"""In-editor UI for film-grip.

The UI is deliberately thin: a declarative :class:`~filmgrip.ui.panel.PanelSpec` (a widget tree) and
a :class:`~filmgrip.ui.panel.PanelController` (prompt → plan → apply, behind injectable callables).
The Resolve-specific rendering lives in ``scripts/film_grip_resolve_panel.py`` and only *translates*
the spec to Resolve's Fusion UIManager — so the panel's structure and behaviour are unit-testable
with zero Resolve open.
"""
from .panel import PanelController, PanelResult, PanelSpec, Widget, build_panel_spec

__all__ = ["PanelController", "PanelResult", "PanelSpec", "Widget", "build_panel_spec"]
