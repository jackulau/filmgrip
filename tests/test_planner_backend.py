"""D3 — PlannerBackend seam: registry, selection precedence, Claude backend wiring."""
from __future__ import annotations

import pytest

from filmgrip.integration import backend as be
from filmgrip.integration.backend import (
    ClaudeBackend,
    PlannerBackend,
    UnknownBackendError,
    available_backends,
    get_backend,
)
from filmgrip.integration.mcp_host import Transport


@pytest.fixture
def registry_guard():
    """Snapshot/restore the backend registry so tests can register throwaway backends."""
    saved = dict(be._REGISTRY)
    yield
    be._REGISTRY.clear()
    be._REGISTRY.update(saved)


def test_claude_is_registered_by_default():
    assert "claude" in available_backends()


def test_default_backend_is_claude(monkeypatch):
    monkeypatch.delenv("FILMGRIP_BACKEND", raising=False)
    b = get_backend()
    assert b.name == "claude" and isinstance(b, ClaudeBackend)


def test_explicit_name_resolves():
    assert get_backend("claude").name == "claude"


def test_claude_backend_is_a_planner_backend():
    # runtime_checkable Protocol — duck-typed structural check, no inheritance required.
    assert isinstance(ClaudeBackend(), PlannerBackend)


def test_claude_transport_is_a_transport_without_sdk():
    # Constructing the transport must NOT require the optional Agent SDK (only running a turn does).
    t = ClaudeBackend().transport()
    assert isinstance(t, Transport)


def test_unknown_backend_raises_with_available_listed():
    with pytest.raises(UnknownBackendError) as ei:
        get_backend("does-not-exist")
    msg = str(ei.value)
    assert "does-not-exist" in msg and "claude" in msg  # actionable: names the known backends


def test_env_selects_backend(monkeypatch, registry_guard):
    class _Dummy:
        name = "dummy"

        def transport(self):
            return Transport()

    be.register("dummy", _Dummy)
    monkeypatch.setenv("FILMGRIP_BACKEND", "dummy")
    assert get_backend().name == "dummy"


def test_explicit_name_overrides_env(monkeypatch, registry_guard):
    class _Dummy:
        name = "dummy"

        def transport(self):
            return Transport()

    be.register("dummy", _Dummy)
    monkeypatch.setenv("FILMGRIP_BACKEND", "dummy")
    # An explicit arg wins over the env var.
    assert get_backend("claude").name == "claude"
