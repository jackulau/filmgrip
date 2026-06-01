"""LLM auth + billing path for the planner.

film-grip's planner runs on Claude via the Agent SDK, which (like Claude Code) picks credentials
from the environment: an ``ANTHROPIC_API_KEY`` means **pay-per-token API billing**, while *no* key
means it uses your logged-in Claude Code OAuth session — i.e. your **Claude subscription**.

Two things live here:

* :func:`detect_auth` — what billing path a planning turn WILL take, surfaced by ``film-grip
  status`` so the cost is never a surprise.
* :func:`subscription_billing` — a context manager that drops the API key for the duration of a
  planning call so it bills to the subscription even when a key is present in the environment (the
  same trick the autorun daemon uses). film-grip prefers the subscription by default
  (``FILMGRIP_USE_SUBSCRIPTION``, default on) because that's what a Max/Pro user already pays for;
  set ``FILMGRIP_USE_SUBSCRIPTION=0`` to bill an API key instead.
"""
from __future__ import annotations

import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ENV_USE_SUBSCRIPTION = "FILMGRIP_USE_SUBSCRIPTION"
_KEY_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN")


@dataclass
class AuthStatus:
    method: str        # "subscription" | "api-key" | "none"
    detail: str
    hint: str = ""


def use_subscription_default(env: Optional[dict] = None) -> bool:
    """Prefer subscription billing? Default yes; toggle off with ``FILMGRIP_USE_SUBSCRIPTION=0``."""
    env = os.environ if env is None else env
    return env.get(ENV_USE_SUBSCRIPTION, "1") != "0"


def _subscription_available() -> bool:
    """Best-effort: is a Claude Code OAuth login present? (``claude`` on PATH, or a creds file.)"""
    if shutil.which("claude"):
        return True
    return (Path.home() / ".claude" / ".credentials.json").exists()


def detect_auth(env: Optional[dict] = None, *, sub_available: Optional[bool] = None) -> AuthStatus:
    """Report the billing path a planning turn will take (used by ``film-grip status``).

    Honest about the preference: with ``FILMGRIP_USE_SUBSCRIPTION`` on (default) and a Claude login
    present, a set ``ANTHROPIC_API_KEY`` is reported as *ignored* — because :func:`subscription_billing`
    will drop it — rather than silently pretending it isn't there.
    """
    env = os.environ if env is None else env
    prefer_sub = use_subscription_default(env)
    has_key = any(env.get(k) for k in _KEY_VARS)
    available = _subscription_available() if sub_available is None else sub_available

    if prefer_sub and available:
        extra = "; ANTHROPIC_API_KEY is set but ignored (FILMGRIP_USE_SUBSCRIPTION=1)" if has_key else ""
        hint = ("set FILMGRIP_USE_SUBSCRIPTION=0 to bill the API key instead" if has_key
                else "run `claude auth status` to confirm your Claude login")
        return AuthStatus("subscription", "Claude Code OAuth (claude.ai subscription)" + extra, hint)
    if has_key:
        return AuthStatus(
            "api-key", "ANTHROPIC_API_KEY (pay-per-token API billing)",
            "unset ANTHROPIC_API_KEY, or set FILMGRIP_USE_SUBSCRIPTION=1, to bill your Claude subscription")
    if available:
        return AuthStatus("subscription", "Claude Code OAuth (claude.ai subscription)",
                          "run `claude auth status` to confirm your Claude login")
    return AuthStatus(
        "none", "no ANTHROPIC_API_KEY and no Claude Code login detected",
        "run `claude` and log in to use your subscription, or set ANTHROPIC_API_KEY for API billing")


@contextmanager
def subscription_billing(enabled: bool = True, env: Optional[dict] = None):
    """Temporarily remove the API key so the Agent SDK bills to the Claude subscription.

    A no-op when ``enabled`` is False. Always restores the environment, even on error.
    """
    env = os.environ if env is None else env
    if not enabled:
        yield
        return
    saved = {k: env[k] for k in _KEY_VARS if k in env}
    for k in saved:
        env.pop(k, None)
    try:
        yield
    finally:
        env.update(saved)
