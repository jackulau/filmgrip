"""LUT handling — the *look* primitive that primaries (CDL) can't express.

Where a CDL carries a re-editable primary-grade decision, a LUT is a sampled, pre-baked transform
(input RGB → output RGB by interpolation). film-grip references a LUT by file and validates its
shape so a hallucinated or malformed path can't survive into an apply — the same guarantee the
clip-ID/frame validator gives the cut side.

Supported, in priority order:

* **.cube** (IRIDAS/Adobe; native in Resolve/Premiere/FCP) — fully parsed and sanity-checked.
  1D **or** 3D (never both): exactly one ``LUT_1D_SIZE``/``LUT_3D_SIZE`` keyword, optional
  ``TITLE``/``DOMAIN_MIN``/``DOMAIN_MAX``, then ``N`` rows (1D) or ``N**3`` rows (3D) of ``R G B``
  triplets. Order convention (3D): red varies fastest, blue slowest — recorded, not reordered.
* **.3dl** (Autodesk Lustre/Flame) — best-effort: integer rows whose count is a perfect cube
  (after an optional leading mesh line). The byte layout varies by vendor, so this is validated
  loosely and flagged as such.

Honest interchange note (surfaced by the adapters): a LUT is just a file. There is no robust
embedding in mainstream interchange — a bare path reference breaks across machines. Ship the
``.cube`` alongside the project or bake it; never trust a path to resolve on another box.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional


class LutError(ValueError):
    """A LUT file is missing, unreadable, or malformed."""


@dataclass(frozen=True)
class LutInfo:
    kind: str                       # "1D" | "3D"
    size: int                       # samples per axis
    title: str = ""
    domain_min: tuple = (0.0, 0.0, 0.0)
    domain_max: tuple = (1.0, 1.0, 1.0)
    fmt: str = "cube"               # "cube" | "3dl"
    rows: int = 0                   # data rows actually counted

    @property
    def points(self) -> int:
        return self.size if self.kind == "1D" else self.size ** 3

    def __str__(self) -> str:
        t = f" '{self.title}'" if self.title else ""
        return f"{self.fmt} {self.kind} size {self.size} ({self.points} pts){t}"


# Resolve/most apps cap a 3D cube around here; 1D LUTs can be far larger (per-channel curves).
_MAX_3D_SIZE = 256
_MAX_1D_SIZE = 65536


def parse_cube(text: str) -> LutInfo:
    """Parse + validate an Adobe/IRIDAS ``.cube`` LUT. Raises :class:`LutError` on any malformation."""
    title = ""
    size: Optional[int] = None
    kind: Optional[str] = None
    dmin = (0.0, 0.0, 0.0)
    dmax = (1.0, 1.0, 1.0)
    data_rows = 0

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        head = line.split(None, 1)[0].upper()
        if head == "TITLE":
            title = line.split(None, 1)[1].strip().strip('"') if " " in line else ""
        elif head == "LUT_1D_SIZE":
            if size is not None:
                raise LutError("more than one LUT size keyword (a .cube is 1D OR 3D, not both)")
            kind, size = "1D", _int(line, "LUT_1D_SIZE")
        elif head == "LUT_3D_SIZE":
            if size is not None:
                raise LutError("more than one LUT size keyword (a .cube is 1D OR 3D, not both)")
            kind, size = "3D", _int(line, "LUT_3D_SIZE")
        elif head == "DOMAIN_MIN":
            dmin = _triple(line, "DOMAIN_MIN")
        elif head == "DOMAIN_MAX":
            dmax = _triple(line, "DOMAIN_MAX")
        else:
            # Must be a data row: exactly three floats.
            parts = line.split()
            if len(parts) != 3:
                raise LutError(f"data row is not 3 values: {line!r}")
            try:
                [float(x) for x in parts]
            except ValueError:
                raise LutError(f"non-numeric data row: {line!r}")
            data_rows += 1

    if size is None or kind is None:
        raise LutError("no LUT_1D_SIZE or LUT_3D_SIZE keyword — not a valid .cube")
    if size < 2:
        raise LutError(f"LUT size {size} too small (need >= 2)")
    if kind == "3D" and size > _MAX_3D_SIZE:
        raise LutError(f"3D LUT size {size} exceeds sane maximum {_MAX_3D_SIZE}")
    if kind == "1D" and size > _MAX_1D_SIZE:
        raise LutError(f"1D LUT size {size} exceeds sane maximum {_MAX_1D_SIZE}")
    expected = size if kind == "1D" else size ** 3
    if data_rows != expected:
        raise LutError(f"{kind} LUT declares size {size} → expected {expected} data rows, found {data_rows}")
    return LutInfo(kind=kind, size=size, title=title, domain_min=dmin, domain_max=dmax,
                   fmt="cube", rows=data_rows)


def parse_3dl(text: str) -> LutInfo:
    """Best-effort ``.3dl`` validation: integer triplet rows whose count is a perfect cube
    (an optional leading mesh line of sample coords is skipped). Loose by design — .3dl byte
    layout differs across Lustre/Flame variants."""
    rows = []
    mesh_seen = False
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) == 3 and all(_is_int(p) for p in parts):
            rows.append(parts)
        elif len(parts) > 3 and all(_is_int(p) for p in parts) and not mesh_seen:
            mesh_seen = True   # the leading mesh/sample-point line
        else:
            raise LutError(f".3dl row is not integers: {line!r}")
    n = len(rows)
    if n == 0:
        raise LutError("no data rows in .3dl")
    size = round(n ** (1 / 3))
    if size ** 3 != n:
        raise LutError(f".3dl has {n} rows, not a perfect cube (size**3)")
    return LutInfo(kind="3D", size=size, fmt="3dl", rows=n)


def inspect_lut(path: str) -> LutInfo:
    """Read + validate a LUT file by extension. Raises :class:`LutError` if missing or malformed."""
    if not os.path.isfile(path):
        raise LutError(f"LUT file not found: {path}")
    try:
        text = open(path, "r", encoding="utf-8", errors="replace").read()
    except OSError as exc:
        raise LutError(f"cannot read LUT {path}: {exc}")
    ext = os.path.splitext(path)[1].lower()
    if ext == ".cube":
        return parse_cube(text)
    if ext == ".3dl":
        return parse_3dl(text)
    raise LutError(f"unsupported LUT extension '{ext}' (use .cube or .3dl)")


def looks_like_bare_reference(path: str) -> bool:
    """True for a name with no directory separator AND no known LUT extension — i.e. something the
    editor is expected to resolve against its own LUT folder, not a file film-grip can open here."""
    if os.sep in path or (os.altsep and os.altsep in path):
        return False
    return os.path.splitext(path)[1].lower() not in (".cube", ".3dl")


# ----------------------------------------------------------------------- helpers
def _int(line: str, kw: str) -> int:
    try:
        return int(line.split()[1])
    except (IndexError, ValueError):
        raise LutError(f"{kw} needs an integer size")


def _triple(line: str, kw: str) -> tuple:
    parts = line.split()[1:]
    if len(parts) != 3:
        raise LutError(f"{kw} needs 3 values")
    try:
        return tuple(float(x) for x in parts)
    except ValueError:
        raise LutError(f"{kw} values must be numeric")


def _is_int(s: str) -> bool:
    try:
        int(s)
        return True
    except ValueError:
        return False
