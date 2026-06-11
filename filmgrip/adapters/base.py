"""The adapter contract every editor integration implements.

An adapter has two halves — *read* (snapshot the timeline into the IR, capture the user's
selection) and *write* (apply a validated EditPlan) — plus an honest :class:`Capabilities`
record. The registry (D15) and CLI surface those capabilities so the tool never promises an
editor automation it cannot deliver (Resolve has no true multi-select; Filmora has no write-back;
interchange is lossy).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from ..core.ir import TimelineIR
from ..protocol.editplan import EditPlan


class NotSupportedError(RuntimeError):
    """Raised by an adapter when an operation is impossible for that editor (e.g. Filmora write)."""


@dataclass
class Capabilities:
    editor: str
    role: str                      # flagship-native | interchange | best-effort | read-only
    mechanism: str
    live_selection: bool
    write_back: bool
    requires_app_running: bool
    lossy_features: list[str] = field(default_factory=list)
    # --- v2 honesty fields (audio / organize / in-app panel / selection confidence) ---
    audio_support: str = "interchange"      # live | interchange | offline | read-only | none
    audio_volume_scriptable: bool = False   # per-clip volume/gain/fades settable? (Resolve: NO)
    organize_support: str = "none"          # live | interchange-warn | none
    editor_panel: str = "none"              # native | uxp-future | read-only | none
    selection_confidence: str = "precise"   # precise | reconstructed | readonly

    def as_dict(self) -> dict:
        return {
            "editor": self.editor,
            "role": self.role,
            "mechanism": self.mechanism,
            "live_selection": self.live_selection,
            "write_back": self.write_back,
            "requires_app_running": self.requires_app_running,
            "lossy_features": list(self.lossy_features),
            "audio_support": self.audio_support,
            "audio_volume_scriptable": self.audio_volume_scriptable,
            "organize_support": self.organize_support,
            "editor_panel": self.editor_panel,
            "selection_confidence": self.selection_confidence,
        }


@dataclass
class Selection:
    """The captured selection: stable clip IDs + how they were derived (for honesty in the UI)."""
    ids: list[str]
    basis: str                      # "current_video_item+media_pool" | "interchange_export" | ...
    note: str = ""
    confidence: str = "precise"     # precise | reconstructed | readonly — how sure the basis is

    def as_header(self, ir: TimelineIR) -> dict:
        from ..serialize.fgx import selection_header

        hdr = selection_header(ir, self.ids)
        hdr["basis"] = self.basis
        hdr["confidence"] = self.confidence
        if self.note:
            hdr["note"] = self.note
        return hdr


@dataclass
class ApplyResult:
    ok: bool
    applied: list[str] = field(default_factory=list)   # human-readable per-op descriptions
    diff: str = ""
    errors: list[str] = field(default_factory=list)
    # Ops the user requested that this adapter/path genuinely cannot apply (e.g. add_transition in
    # Resolve, add_marker in CapCut). These make ok=False — film-grip never reports success for an
    # edit it didn't perform. Distinct from `warnings`, which annotate ops that DID apply (lossy
    # round-trip, a bin that fell back to root). The CLI maps a non-empty `unsupported` to exit 3.
    unsupported: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    # Post-apply verification (filled by the --verify loop, perception.verify): proof lines that
    # the re-snapshotted timeline matches the simulated expected geometry, and any divergences.
    verified: list[str] = field(default_factory=list)
    mismatches: list[str] = field(default_factory=list)


class GrabAdapter:
    """Base class. Subclasses override what they support; the rest raises ``NotSupportedError``."""

    name: str = "base"

    def capabilities(self) -> Capabilities:  # pragma: no cover - abstract
        raise NotImplementedError

    def snapshot(self, source: Any) -> TimelineIR:
        raise NotSupportedError(f"{self.name}: snapshot not supported")

    def get_selection(self, source: Any, ir: Optional[TimelineIR] = None) -> Selection:
        raise NotSupportedError(f"{self.name}: selection capture not supported")

    def apply(self, plan: EditPlan, source: Any, **kw) -> ApplyResult:
        raise NotSupportedError(f"{self.name}: write-back not supported")
