"""D5 — Codex/GPT backend seam: registered, resolves, fails honestly (no crash, no silent fallback)."""
from __future__ import annotations

from filmgrip.adapters.base import Selection
from filmgrip.cli import main
from filmgrip.core.ir import TimelineIR
from filmgrip.integration.backend import PlannerBackend, available_backends, get_backend
from filmgrip.integration.backend_codex import NOT_IMPLEMENTED, CodexBackend, CodexTransport
from filmgrip.integration.mcp_host import PlannerContext

FIX = "tests/fixtures/cut.otio"


def _ctx():
    ir = TimelineIR.from_otio_file(FIX)
    return PlannerContext(ir=ir, selection=Selection(ids=[ir.real_clips()[0].id], basis="x"))


def test_codex_is_registered():
    assert "codex" in available_backends()


def test_get_backend_codex_resolves():
    b = get_backend("codex")
    assert b.name == "codex" and isinstance(b, CodexBackend)


def test_codex_backend_satisfies_protocol():
    assert isinstance(CodexBackend(), PlannerBackend)


def test_codex_transport_returns_honest_error_not_crash():
    resp = CodexTransport().run(system_prompt="", user_prompt="x", schema={}, ctx=_ctx())
    assert resp.subtype == "error"
    assert resp.structured_output is None
    assert any("not yet implemented" in e for e in resp.errors)


def test_codex_through_repair_loop_fails_cleanly():
    from filmgrip.integration.repair import plan_with_repair

    result = plan_with_repair(_ctx(), "tighten the open", CodexTransport())
    assert result.ok is False and result.plan is None
    assert any("not yet implemented" in e for e in result.errors)


def test_cli_edit_backend_codex_surfaces_not_implemented(capsys):
    # Fixture-mode planning routes through the selected backend with no live editor.
    code = main(["edit", "--backend", "codex", "--fixture", FIX, "make the open punchier"])
    out = capsys.readouterr().out
    assert code == 1
    assert "not yet implemented" in out


def test_not_implemented_message_points_at_the_doc():
    assert "docs/BACKENDS.md" in NOT_IMPLEMENTED
