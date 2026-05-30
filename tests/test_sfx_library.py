"""D4 — the user's sound-effects library: manifest + deterministic description resolution."""
from __future__ import annotations

from filmgrip.audio.library import SfxLibrary, default_dir
from filmgrip.cli import main


def _lib(fixtures_dir):
    return SfxLibrary.load(fixtures_dir / "sfx")


def test_loads_manifest_entries_and_tags(fixtures_dir):
    lib = _lib(fixtures_dir)
    names = lib.names()
    assert "whoosh" in names and "door_slam" in names and "riser" in names
    whoosh = lib.get("whoosh")
    assert whoosh is not None
    assert whoosh.file == "whoosh_01.wav"
    assert "transition" in whoosh.tags
    assert whoosh.duration_frames == 18


def test_autodiscovers_loose_audio_not_in_manifest(fixtures_dir):
    # loose_clap.wav exists on disk but isn't in manifest.json -> picked up by scan.
    lib = _lib(fixtures_dir)
    assert "loose_clap" in lib.names()
    clap = lib.get("loose_clap")
    assert "clap" in clap.tags  # tags auto-derived from the filename stem


def test_missing_manifest_files_are_surfaced_not_hidden(fixtures_dir):
    lib = _lib(fixtures_dir)
    assert "not_here.wav" in lib.missing
    ghost = lib.get("ghost_missing")
    assert ghost is not None and not ghost.exists(lib.base)


def test_resolve_description_to_best_effect(fixtures_dir):
    lib = _lib(fixtures_dir)
    assert lib.resolve("add a whoosh as the title flies in").name == "whoosh"
    assert lib.resolve("big door slam impact").name == "door_slam"
    assert lib.resolve("a tension riser building up").name == "riser"


def test_resolve_returns_none_when_nothing_matches(fixtures_dir):
    assert _lib(fixtures_dir).resolve("xylophone glissando") is None


def test_manifest_rows_are_token_frugal(fixtures_dir):
    # The choosing surface handed to Claude carries names + tags only, never file paths/bytes.
    rows = _lib(fixtures_dir).manifest_rows()
    assert all(set(r.keys()) == {"name", "tags"} for r in rows)


def test_get_is_case_insensitive(fixtures_dir):
    assert _lib(fixtures_dir).get("WHOOSH").file == "whoosh_01.wav"


def test_default_dir_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FILMGRIP_SFX_DIR", str(tmp_path / "mine"))
    assert default_dir() == (tmp_path / "mine")


def test_cli_sfx_list_exits_zero(fixtures_dir, capsys):
    code = main(["sfx", "list", "--dir", str(fixtures_dir / "sfx")])
    out = capsys.readouterr().out
    assert code == 0
    assert "whoosh" in out and "door_slam" in out


def test_cli_sfx_resolve(fixtures_dir, capsys):
    code = main(["sfx", "resolve", "title whoosh", "--dir", str(fixtures_dir / "sfx")])
    out = capsys.readouterr().out
    assert code == 0
    assert "whoosh" in out and "whoosh_01.wav" in out


def test_cli_sfx_scan_writes_manifest(tmp_path, capsys):
    (tmp_path / "boom.wav").write_bytes(b"RIFF")
    (tmp_path / "click.mp3").write_bytes(b"ID3")
    code = main(["sfx", "scan", "--dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert (tmp_path / "manifest.json").is_file()
    reloaded = SfxLibrary.load(tmp_path)
    assert {"boom", "click"} <= set(reloaded.names())
