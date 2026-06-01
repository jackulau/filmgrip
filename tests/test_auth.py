"""D4 — LLM auth detection + subscription billing path."""
from __future__ import annotations

from filmgrip.integration.auth import (
    AuthStatus,
    detect_auth,
    subscription_billing,
    use_subscription_default,
)


# --- detect_auth -------------------------------------------------------------
def test_api_key_billing_when_subscription_disabled():
    env = {"ANTHROPIC_API_KEY": "sk-x", "FILMGRIP_USE_SUBSCRIPTION": "0"}
    a = detect_auth(env, sub_available=True)
    assert a.method == "api-key" and "API" in a.detail


def test_subscription_preferred_ignores_api_key_by_default():
    a = detect_auth({"ANTHROPIC_API_KEY": "sk-x"}, sub_available=True)  # prefer-sub default on
    assert a.method == "subscription" and "ignored" in a.detail


def test_subscription_when_no_key_and_login_present():
    assert detect_auth({}, sub_available=True).method == "subscription"


def test_api_key_when_no_subscription_available():
    assert detect_auth({"ANTHROPIC_API_KEY": "sk-x"}, sub_available=False).method == "api-key"


def test_none_when_nothing_configured():
    a = detect_auth({}, sub_available=False)
    assert a.method == "none" and a.hint


def test_use_subscription_default():
    assert use_subscription_default({}) is True
    assert use_subscription_default({"FILMGRIP_USE_SUBSCRIPTION": "0"}) is False
    assert use_subscription_default({"FILMGRIP_USE_SUBSCRIPTION": "1"}) is True


# --- subscription_billing context manager ------------------------------------
def test_subscription_billing_strips_and_restores_keys():
    env = {"ANTHROPIC_API_KEY": "sk-x", "ANTHROPIC_AUTH_TOKEN": "tok", "OTHER": "keep"}
    with subscription_billing(True, env):
        assert "ANTHROPIC_API_KEY" not in env and "ANTHROPIC_AUTH_TOKEN" not in env
        assert env["OTHER"] == "keep"
    assert env["ANTHROPIC_API_KEY"] == "sk-x" and env["ANTHROPIC_AUTH_TOKEN"] == "tok"


def test_subscription_billing_disabled_is_noop():
    env = {"ANTHROPIC_API_KEY": "sk-x"}
    with subscription_billing(False, env):
        assert env["ANTHROPIC_API_KEY"] == "sk-x"
    assert env["ANTHROPIC_API_KEY"] == "sk-x"


def test_subscription_billing_restores_on_error():
    env = {"ANTHROPIC_API_KEY": "sk-x"}
    try:
        with subscription_billing(True, env):
            assert "ANTHROPIC_API_KEY" not in env
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert env["ANTHROPIC_API_KEY"] == "sk-x"  # restored despite the error


# --- status integration ------------------------------------------------------
def test_status_renders_auth_line():
    from filmgrip.cli_status import render_status

    auth = AuthStatus("subscription", "Claude Code OAuth (claude.ai subscription)", "hint here")
    out = render_status({"module_importable": False, "app_running": False}, "none", 7, auth=auth)
    assert "LLM auth" in out and "subscription" in out and "hint here" in out
