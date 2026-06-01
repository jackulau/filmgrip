"""Planner backend selection — which LLM provider plans the edit.

film-grip's planner is provider-agnostic by construction: the validate→apply pipeline only needs a
:class:`~filmgrip.integration.mcp_host.Transport` — something that turns a prompt + JSON schema into
a normalized ``PlanResponse``. A *backend* names a provider and produces that transport.

* :class:`ClaudeBackend` is the live flagship — Claude via the Agent SDK + the in-process MCP server
  (:mod:`filmgrip.integration.mcp_host`), where the model *pulls* FGX context through read-only
  tools and returns a schema-validated EditPlan.
* A Codex/GPT backend (D5) slots in by registering a name here. Nothing else changes — not the
  repair loop, not the CLI, not the panel — because they all talk to the ``Transport`` seam, not to
  a provider directly.

Selection precedence: an explicit ``--backend`` flag, then ``$FILMGRIP_BACKEND``, then the default
(``claude``). The CLI/panel resolve a backend and hand its transport to ``plan_with_repair``.
"""
from __future__ import annotations

import os
from typing import Callable, Protocol, runtime_checkable

from .mcp_host import Transport

DEFAULT_BACKEND = "claude"
ENV_BACKEND = "FILMGRIP_BACKEND"


@runtime_checkable
class PlannerBackend(Protocol):
    """A planner provider: a stable ``name`` and the ``Transport`` that runs one planning turn."""

    name: str

    def transport(self) -> Transport:
        """Return the transport that executes a planning turn for this provider."""
        ...


class UnknownBackendError(ValueError):
    """Raised when a backend name has no registered factory."""


_REGISTRY: dict[str, Callable[[], PlannerBackend]] = {}


def register(name: str, factory: Callable[[], PlannerBackend]) -> None:
    """Register a backend factory under ``name`` (idempotent — last registration wins)."""
    _REGISTRY[name] = factory


def available_backends() -> list[str]:
    return sorted(_REGISTRY)


def get_backend(name: str | None = None) -> PlannerBackend:
    """Resolve a backend, applying precedence: ``name`` > ``$FILMGRIP_BACKEND`` > default.

    Raises :class:`UnknownBackendError` (with the list of known names) for an unregistered backend,
    so the CLI can print an actionable message rather than crashing.
    """
    chosen = name or os.environ.get(ENV_BACKEND) or DEFAULT_BACKEND
    factory = _REGISTRY.get(chosen)
    if factory is None:
        known = ", ".join(available_backends()) or "(none registered)"
        raise UnknownBackendError(f"unknown planner backend '{chosen}'. Available: {known}.")
    return factory()


class ClaudeBackend:
    """Flagship backend: Claude via the Agent SDK + in-process MCP server (see ``mcp_host``)."""

    name = "claude"

    def transport(self) -> Transport:
        # Lazy import keeps the Agent SDK an optional dependency — constructing the transport
        # needs no SDK; only running a real turn does.
        from .mcp_host import ClaudeAgentTransport

        return ClaudeAgentTransport()


register("claude", ClaudeBackend)
