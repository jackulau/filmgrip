"""film-grip color — the portable color-grading layer.

Mirrors the cut side's design: just as every timeline normalizes to one OTIO graph, every
color decision normalizes to one **ASC CDL** value (the vendor-neutral primary-grade pivot)
plus, for looks that primaries can't express, a **LUT** reference. CDL rides losslessly through
OTIO clip metadata, ``.cc``/``.ccc``/``.cdl`` sidecars, EDL ``*ASC_SOP``/``*ASC_SAT`` comments,
and FCPXML ``info-asc-cdl`` — so a grade proposed once is portable across the whole adapter
family, not only DaVinci Resolve.

Public surface:

* :class:`~filmgrip.color.cdl.CDL` — the 10-number ASC CDL value (SOP + saturation) + its math.
* :func:`~filmgrip.color.cdl.lgg_to_cdl` — compile colorist-ergonomic
  lift/gamma/gain/contrast/temp/tint into a canonical CDL, so an agent (or a grade pack) can
  reason in natural color terms while the stored/wire primitive stays portable and validated.
"""
from __future__ import annotations

from .cdl import CDL, lgg_to_cdl

__all__ = ["CDL", "lgg_to_cdl"]
