"""Stable clip identity for film-grip.

Claude references clips by short opaque IDs in every EditPlan op. Those IDs must be:

* **deterministic** — the same clip yields the same ID across reloads/processes, so a plan
  authored against one snapshot resolves against the next. Python's salted ``hash()`` is
  therefore unusable; we hash with blake2b.
* **short** — a few base36 chars, because each ID is repeated many times in the token stream.
* **drift-tolerant** — after a ripple edit shifts frame positions, the original ID must still
  re-resolve to "the same clip". We keep a fingerprint (name, source ref, track, original
  start) per ID and re-match by identity + nearest start.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:  # avoid a runtime import cycle (ir imports idmap)
    from .ir import Clip, TimelineIR

_B36 = "0123456789abcdefghijklmnopqrstuvwxyz"


def to_base36(n: int) -> str:
    if n == 0:
        return "0"
    out = ""
    while n:
        n, r = divmod(n, 36)
        out = _B36[r] + out
    return out


def stable_clip_id(track_kind: str, track_index: int, start: int, src_ref: str) -> str:
    """Deterministic short ID from a clip's identifying coordinates."""
    key = f"{track_kind}:{track_index}:{start}:{src_ref}".encode("utf-8")
    digest = hashlib.blake2b(key, digest_size=4).digest()  # 32 bits
    n = int.from_bytes(digest, "big")
    return "c" + to_base36(n).rjust(5, "0")


@dataclass(frozen=True)
class Fingerprint:
    name: str
    src_ref: str
    track_kind: str
    track_index: int
    start: int


class IdMap:
    """Bidirectional clip-ID registry with drift-tolerant re-resolution."""

    def __init__(self) -> None:
        self._by_id: dict[str, "Clip"] = {}
        self._fp: dict[str, Fingerprint] = {}

    def mint(self, clip: "Clip") -> str:
        """Assign (or reuse) a deterministic ID for ``clip`` and register its fingerprint."""
        base = stable_clip_id(clip.track_kind, clip.track_index, clip.start, clip.src_ref)
        cid = base
        n = 1
        # Disambiguate the rare hash collision deterministically.
        while cid in self._by_id and self._fp[cid] != _fingerprint(clip):
            cid = base + to_base36(n)
            n += 1
        clip.id = cid
        self._by_id[cid] = clip
        self._fp[cid] = _fingerprint(clip)
        return cid

    def get(self, cid: str) -> Optional["Clip"]:
        return self._by_id.get(cid)

    def fingerprint(self, cid: str) -> Optional[Fingerprint]:
        return self._fp.get(cid)

    def ids(self) -> list[str]:
        return list(self._by_id)

    def reresolve(self, cid: str, new_ir: "TimelineIR") -> Optional["Clip"]:
        """Find the clip in ``new_ir`` that corresponds to ``cid`` after positions drifted.

        Matches first on (name, src_ref, track), falling back to src_ref alone, then picks the
        candidate whose start is nearest the fingerprinted start. Returns ``None`` when nothing
        plausibly matches — callers must treat that as a hard validation failure, never a guess.
        """
        fp = self._fp.get(cid)
        if fp is None:
            return new_ir.clip(cid)  # maybe the ID is still exact in the new IR
        candidates = [
            c for c in new_ir.clips
            if c.name == fp.name and c.src_ref == fp.src_ref
            and c.track_kind == fp.track_kind and c.track_index == fp.track_index
        ]
        if not candidates:
            candidates = [c for c in new_ir.clips if c.src_ref == fp.src_ref and c.track_kind == fp.track_kind]
        if not candidates:
            candidates = [c for c in new_ir.clips if c.src_ref == fp.src_ref]
        if not candidates:
            return None
        return min(candidates, key=lambda c: abs(c.start - fp.start))


def _fingerprint(clip: "Clip") -> Fingerprint:
    return Fingerprint(
        name=clip.name,
        src_ref=clip.src_ref,
        track_kind=clip.track_kind,
        track_index=clip.track_index,
        start=clip.start,
    )
