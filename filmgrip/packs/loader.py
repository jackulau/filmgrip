"""Discover user-authored packs from ``$FILMGRIP_PACK_DIR`` or ``~/.filmgrip/packs``.

User packs are **prompt packs** — small data files (``.toml`` or ``.json``) holding a name,
description, a parameterized prompt, and optional params. Deterministic recipes stay built-in code
(they need real logic); what a user wants to save and reuse is an *instruction*, and that's data.

A malformed file must never break ``film-grip pack list`` — it's skipped, not raised. User packs
override built-ins of the same name (honest precedence: your config wins).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from . import Pack

ENV_PACK_DIR = "FILMGRIP_PACK_DIR"


def user_pack_dir() -> Path:
    """The directory user packs are read from: ``$FILMGRIP_PACK_DIR`` or ``~/.filmgrip/packs``."""
    env = os.environ.get(ENV_PACK_DIR)
    return Path(env).expanduser() if env else Path.home() / ".filmgrip" / "packs"


def _parse_toml(path: Path) -> Optional[dict]:
    try:
        import tomllib  # py3.11+ ; absent on 3.10 → .toml packs are skipped there
    except ModuleNotFoundError:
        return None
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def _load_one(path: Path) -> Optional[Pack]:
    try:
        if path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
        elif path.suffix == ".toml":
            data = _parse_toml(path)
            if data is None:
                return None  # TOML unsupported on this interpreter
        else:
            return None
        if data.get("kind", "prompt") != "prompt":
            return None  # user packs are prompt packs; deterministic recipes are built-in code
        prompt = data.get("prompt", "")
        if not prompt:
            return None
        return Pack(name=data.get("name") or path.stem, description=data.get("description", ""),
                    kind="prompt", prompt=prompt, params=dict(data.get("params", {})),
                    source=str(path))
    except Exception:
        return None  # a broken user file is skipped, never fatal


def load_user_packs(directory=None) -> dict:
    """Load ``{name: Pack}`` from the user pack dir. Empty dict if the dir is absent."""
    d = Path(directory) if directory else user_pack_dir()
    if not d.is_dir():
        return {}
    packs: dict = {}
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix in (".json", ".toml"):
            pack = _load_one(p)
            if pack is not None:
                packs[pack.name] = pack
    return packs
