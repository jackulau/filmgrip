"""D7 — prompt-template packs + user pack dir discovery + planner routing."""
from __future__ import annotations

import json

from filmgrip.cli import main
from filmgrip.core.ir import TimelineIR
from filmgrip.packs import get_pack
from filmgrip.packs.loader import load_user_packs, user_pack_dir

FIX = "tests/fixtures/cut.otio"


# --- built-in prompt packs ---------------------------------------------------
def test_builtin_prompt_pack_exists():
    p = get_pack("podcast-cleanup")
    assert p.kind == "prompt" and p.prompt


def test_pack_show_prints_prompt(capsys):
    code = main(["pack", "show", "podcast-cleanup"])
    out = capsys.readouterr().out
    assert code == 0 and "prompt" in out


def test_prompt_params_fill_placeholders():
    from filmgrip.cli_pack import _format_prompt

    filled = _format_prompt(get_pack("podcast-cleanup"))
    assert "{min_silence}" not in filled and "0.5s" in filled


# --- user pack dir -----------------------------------------------------------
def test_user_pack_dir_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILMGRIP_PACK_DIR", str(tmp_path))
    assert user_pack_dir() == tmp_path


def test_user_pack_loaded_and_listed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FILMGRIP_PACK_DIR", str(tmp_path))
    (tmp_path / "mypack.json").write_text(json.dumps({
        "name": "my-trim", "kind": "prompt", "description": "my custom",
        "prompt": "tighten every cut by 6 frames"}))
    packs = load_user_packs()
    assert "my-trim" in packs and packs["my-trim"].source.endswith("mypack.json")
    code = main(["pack", "list"])
    assert code == 0 and "my-trim" in capsys.readouterr().out


def test_user_pack_overrides_builtin(monkeypatch, tmp_path):
    monkeypatch.setenv("FILMGRIP_PACK_DIR", str(tmp_path))
    (tmp_path / "podcast-cleanup.json").write_text(json.dumps({
        "name": "podcast-cleanup", "kind": "prompt", "prompt": "OVERRIDDEN"}))
    p = get_pack("podcast-cleanup")
    assert p.prompt == "OVERRIDDEN" and p.source.endswith(".json")


def test_malformed_user_pack_is_ignored(monkeypatch, tmp_path):
    monkeypatch.setenv("FILMGRIP_PACK_DIR", str(tmp_path))
    (tmp_path / "broken.json").write_text("{ not valid json")
    assert "broken" not in load_user_packs()  # skipped, not fatal


def test_deterministic_user_file_is_skipped(monkeypatch, tmp_path):
    # User packs are prompt packs; a data file claiming kind=deterministic can't carry logic.
    monkeypatch.setenv("FILMGRIP_PACK_DIR", str(tmp_path))
    (tmp_path / "x.json").write_text(json.dumps({"name": "x", "kind": "deterministic"}))
    assert "x" not in load_user_packs()


# --- prompt pack apply routes through the backend ----------------------------
def test_prompt_pack_apply_routes_through_backend(monkeypatch, capsys):
    from filmgrip.integration import backend as be
    from filmgrip.integration.mcp_host import PlanResponse, Transport

    cid = TimelineIR.from_otio_file(FIX).real_clips()[0].id

    class _FakeTransport(Transport):
        def run(self, *, system_prompt, user_prompt, schema, ctx, session_id=None, model=None):
            return PlanResponse(
                subtype="success",
                structured_output={"ops": [{"op": "add_marker", "clip_id": cid, "frame": 0}]})

    class _FakeBackend:
        name = "fake"

        def transport(self):
            return _FakeTransport()

    saved = dict(be._REGISTRY)
    be.register("fake", _FakeBackend)
    monkeypatch.setenv("FILMGRIP_BACKEND", "fake")
    try:
        code = main(["pack", "apply", "podcast-cleanup", "--fixture", FIX, "--dry-run"])
    finally:
        be._REGISTRY.clear()
        be._REGISTRY.update(saved)
    out = capsys.readouterr().out
    assert code == 0 and "PLAN OK" in out
