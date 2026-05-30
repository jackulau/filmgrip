"""The user's sound-effects library: a folder + optional manifest, resolved by description.

Why this exists: Claude can't reason over raw audio bytes, and shipping a sound pack would be both
heavy and presumptuous. Instead film-grip reads a folder the user already owns and exposes a tiny
**manifest** — just ``name``, ``tags`` and ``file`` per effect — as the choosing surface. "add a
whoosh as the title flies in" becomes a lookup against that manifest, not a scan of megabytes of WAV.

Resolution is deterministic and token-free (keyword overlap), so the same description always maps to
the same file; the LLM's only job is to pick a name from the manifest it's shown (D5's import_audio
op references an entry by name, and the adapter resolves name -> path here).
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Audio containers film-grip recognises when scanning a folder.
AUDIO_EXTS: frozenset[str] = frozenset({
    ".wav", ".aif", ".aiff", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus", ".wma",
})

MANIFEST_NAME = "manifest.json"
_DEFAULT_DIR = "~/.filmgrip/sfx"
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def default_dir() -> Path:
    """The SFX folder: ``$FILMGRIP_SFX_DIR`` if set, else ``~/.filmgrip/sfx``."""
    return Path(os.path.expanduser(os.environ.get("FILMGRIP_SFX_DIR", _DEFAULT_DIR)))


def _tokens(*parts: str) -> set[str]:
    out: set[str] = set()
    for p in parts:
        out.update(_TOKEN_RE.findall(p.lower()))
    return out


@dataclass
class SfxEntry:
    """One sound effect: a stable ``name`` Claude references, plus the backing file and tags."""

    name: str
    file: str                       # path relative to the library dir (or absolute)
    tags: list[str] = field(default_factory=list)
    duration_frames: Optional[int] = None  # optional hint from the manifest; not auto-decoded

    def path(self, base: Path) -> Path:
        p = Path(self.file)
        return p if p.is_absolute() else (base / p)

    def exists(self, base: Path) -> bool:
        return self.path(base).is_file()

    def tokenset(self) -> set[str]:
        return _tokens(self.name, Path(self.file).stem, *self.tags)

    def as_manifest_row(self) -> dict:
        row = {"name": self.name, "file": self.file, "tags": list(self.tags)}
        if self.duration_frames is not None:
            row["duration_frames"] = self.duration_frames
        return row


class SfxLibrary:
    """A resolved view of a sound-effects folder (manifest entries + scanned audio files)."""

    def __init__(self, base: Path, entries: list[SfxEntry], missing: Optional[list[str]] = None):
        self.base = base
        self.entries = entries
        self.missing = missing or []  # manifest entries whose file is absent (surfaced, not hidden)

    # -- construction -----------------------------------------------------------
    @classmethod
    def load(cls, directory: Optional[os.PathLike | str] = None) -> "SfxLibrary":
        """Load a library: manifest entries (if any) merged with auto-discovered audio files.

        Never raises on a missing folder — returns an empty library so the CLI can report it
        cleanly. Manifest entries whose file is gone are recorded in ``missing``.
        """
        base = Path(directory).expanduser() if directory else default_dir()
        entries: list[SfxEntry] = []
        seen_files: set[str] = set()
        missing: list[str] = []

        manifest = base / MANIFEST_NAME
        if manifest.is_file():
            for row in cls._read_manifest(manifest):
                entry = SfxEntry(
                    name=str(row.get("name") or Path(str(row.get("file", ""))).stem),
                    file=str(row.get("file", "")),
                    tags=list(row.get("tags", []) or []),
                    duration_frames=row.get("duration_frames"),
                )
                if not entry.file:
                    continue
                entries.append(entry)
                seen_files.add(os.path.normpath(entry.file))
                if not entry.exists(base):
                    missing.append(entry.file)

        # Auto-discover loose audio files not already named in the manifest.
        if base.is_dir():
            for f in sorted(base.iterdir()):
                if f.is_file() and f.suffix.lower() in AUDIO_EXTS \
                        and os.path.normpath(f.name) not in seen_files:
                    entries.append(SfxEntry(name=f.stem, file=f.name,
                                            tags=sorted(_tokens(f.stem))))
        return cls(base, entries, missing)

    @staticmethod
    def _read_manifest(path: Path) -> list[dict]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []
        if isinstance(data, dict):
            rows = data.get("effects") or data.get("sfx") or []
        elif isinstance(data, list):
            rows = data
        else:
            rows = []
        return [r for r in rows if isinstance(r, dict)]

    # -- queries ----------------------------------------------------------------
    def names(self) -> list[str]:
        return [e.name for e in self.entries]

    def get(self, name: str) -> Optional[SfxEntry]:
        """Exact (case-insensitive) name lookup — the path D5's import_audio op takes."""
        low = name.lower()
        return next((e for e in self.entries if e.name.lower() == low), None)

    def resolve(self, description: str) -> Optional[SfxEntry]:
        """Best-effort description → entry by deterministic keyword overlap (no LLM, no bytes).

        Scores each entry by how many description tokens hit its name/file/tag tokens (exact = 1,
        substring = 0.5), tie-broken by the shorter name so results are stable. Returns ``None`` when
        nothing overlaps, so callers never silently grab an unrelated effect.
        """
        want = _tokens(description)
        if not want:
            return None
        best: Optional[SfxEntry] = None
        best_score = 0.0
        for e in self.entries:
            toks = e.tokenset()
            score = 0.0
            for w in want:
                if w in toks:
                    score += 1.0
                elif any(w in t or t in w for t in toks):
                    score += 0.5
            if score > best_score or (score == best_score and best is not None
                                      and score > 0 and len(e.name) < len(best.name)):
                best, best_score = e, score
        return best if best_score > 0 else None

    def manifest_rows(self) -> list[dict]:
        """The compact list handed to Claude (names + tags only — never file bytes)."""
        return [{"name": e.name, "tags": e.tags} for e in self.entries]

    def write_manifest(self) -> Path:
        """Persist the current entries to ``<dir>/manifest.json`` (the scan output)."""
        self.base.mkdir(parents=True, exist_ok=True)
        path = self.base / MANIFEST_NAME
        payload = {"version": 1, "effects": [e.as_manifest_row() for e in self.entries]}
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return path


# --------------------------------------------------------------------------- CLI
def cmd_sfx(args) -> int:
    """``film-grip sfx <list|scan|resolve>`` — inspect and resolve the user's SFX library."""
    base = Path(args.dir).expanduser() if getattr(args, "dir", None) else default_dir()
    lib = SfxLibrary.load(base)

    action = getattr(args, "sfx_action", None) or "list"
    if action == "scan":
        out = lib.write_manifest()
        print(f"wrote {len(lib.entries)} effect(s) to {out}")
        return 0

    if action == "resolve":
        match = lib.resolve(args.query or "")
        if match is None:
            print(f"no SFX in {base} matches {args.query!r}")
            return 1
        print(f"{match.name}\t{match.path(base)}")
        return 0

    # default: list
    if not lib.entries:
        print(f"no sound effects found in {base}")
        print("  → add audio files there (or set $FILMGRIP_SFX_DIR / pass --dir), then `film-grip sfx scan`")
        return 0
    print(f"{len(lib.entries)} sound effect(s) in {base}:")
    for e in lib.entries:
        tags = (" [" + ", ".join(e.tags) + "]") if e.tags else ""
        flag = "" if e.exists(base) else "  (MISSING FILE)"
        print(f"  {e.name:<24} {e.file}{tags}{flag}")
    if lib.missing:
        print(f"  ⚠ {len(lib.missing)} manifest entr(y/ies) point at missing files")
    return 0
