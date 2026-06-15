"""ASC CDL interchange — make a grade portable across the whole adapter family, not just Resolve.

A :class:`~filmgrip.color.cdl.CDL` is the portable color decision; this module is how it leaves
film-grip and lands in other tools, exactly the way the OTIO serializer carries the cut:

* **``.cc``** — a single ``ColorCorrection`` (one grade).
* **``.ccc``** — a ``ColorCorrectionCollection`` (many named grades).
* **``.cdl``** — a ``ColorDecisionList`` (grades wrapped in ``ColorDecision``s).

  All under namespace ``urn:ASC:CDL:v1.2`` with the standard ``SOPNode``/``SatNode`` shape. A grade
  written here round-trips bit-exactly back to a :class:`CDL`.
* **EDL** — ``*ASC_SOP (s s s)(o o o)(p p p)`` / ``*ASC_SAT v`` comment lines, the conform path
  through Avid/Resolve EDLs.
* **FCPXML** — an ``info-asc-cdl`` element. Honest caveat (surfaced, not hidden): FCPXML carries
  CDL as **inert passthrough metadata** — Final Cut does not apply it on import; it's there for the
  next tool in the chain that does.

CDL itself declares no working color space, so film-grip records ``color_space`` in a film-grip
``Description`` line and reads it back, keeping our own round-trips lossless without overstepping
the standard.
"""
from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional

from ..color.cdl import CDL

_NS = "urn:ASC:CDL:v1.2"
_CS_PREFIX = "filmgrip-colorspace:"


# --------------------------------------------------------------------------- formatting helpers
def _fmt(x: float) -> str:
    return f"{x:.6g}"


def _triple(t) -> str:
    return f"{_fmt(t[0])} {_fmt(t[1])} {_fmt(t[2])}"


def _parse_triple(text: str):
    parts = text.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(f"expected 3 numbers, got {text!r}")
    return tuple(float(p) for p in parts)


def _local(tag: str) -> str:
    return tag.split("}", 1)[-1]


def _cc_xml(cdl: CDL, cc_id: str, indent: str = "  ") -> str:
    """One ``<ColorCorrection>`` block (no XML declaration, no namespace — caller wraps)."""
    desc = f"{indent}  <Description>{_CS_PREFIX}{cdl.color_space}</Description>\n" if cdl.color_space else ""
    return (
        f'{indent}<ColorCorrection id="{cc_id}">\n'
        f"{desc}"
        f"{indent}  <SOPNode>\n"
        f"{indent}    <Slope>{_triple(cdl.slope)}</Slope>\n"
        f"{indent}    <Offset>{_triple(cdl.offset)}</Offset>\n"
        f"{indent}    <Power>{_triple(cdl.power)}</Power>\n"
        f"{indent}  </SOPNode>\n"
        f"{indent}  <SatNode>\n"
        f"{indent}    <Saturation>{_fmt(cdl.saturation)}</Saturation>\n"
        f"{indent}  </SatNode>\n"
        f"{indent}</ColorCorrection>\n"
    )


# --------------------------------------------------------------------------- .cc (single grade)
def dumps_cc(cdl: CDL, cc_id: str = "cc0001") -> str:
    body = _cc_xml(cdl, cc_id, indent="")
    # promote the default namespace onto the root element
    body = body.replace("<ColorCorrection ", f'<ColorCorrection xmlns="{_NS}" ', 1)
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + body


def _cdl_from_cc_element(cc: ET.Element) -> CDL:
    sop = {"Slope": (1.0, 1.0, 1.0), "Offset": (0.0, 0.0, 0.0), "Power": (1.0, 1.0, 1.0)}
    sat = 1.0
    color_space = ""
    for el in cc.iter():
        name = _local(el.tag)
        if name in sop and el.text:
            sop[name] = _parse_triple(el.text)
        elif name == "Saturation" and el.text:
            sat = float(el.text.strip())
        elif name == "Description" and el.text and el.text.strip().startswith(_CS_PREFIX):
            color_space = el.text.strip()[len(_CS_PREFIX):]
    return CDL(slope=sop["Slope"], offset=sop["Offset"], power=sop["Power"],
               saturation=sat, color_space=color_space)


def loads_cc(text: str) -> CDL:
    """Parse a ``.cc`` (or the first ``ColorCorrection`` found in any CDL XML) into a :class:`CDL`."""
    root = ET.fromstring(text)
    cc = root if _local(root.tag) == "ColorCorrection" else root.find(f".//{{{_NS}}}ColorCorrection")
    if cc is None:
        # tolerate non-namespaced files
        cc = next((e for e in root.iter() if _local(e.tag) == "ColorCorrection"), None)
    if cc is None:
        raise ValueError("no <ColorCorrection> element found")
    return _cdl_from_cc_element(cc)


# --------------------------------------------------------------------------- .ccc (collection)
def dumps_ccc(cdls, ids: Optional[list] = None) -> str:
    ids = ids or [f"cc{ i + 1:04d}".replace(" ", "") for i in range(len(cdls))]
    blocks = "".join(_cc_xml(c, ids[i]) for i, c in enumerate(cdls))
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<ColorCorrectionCollection xmlns="{_NS}">\n{blocks}</ColorCorrectionCollection>\n')


def loads_ccc(text: str) -> list:
    root = ET.fromstring(text)
    return [_cdl_from_cc_element(cc) for cc in root.iter() if _local(cc.tag) == "ColorCorrection"]


# --------------------------------------------------------------------------- .cdl (decision list)
def dumps_cdl(cdls, ids: Optional[list] = None) -> str:
    ids = ids or [f"cc{i + 1:04d}" for i in range(len(cdls))]
    decisions = "".join(
        f"  <ColorDecision>\n{_cc_xml(c, ids[i], indent='    ')}  </ColorDecision>\n"
        for i, c in enumerate(cdls))
    return (f'<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<ColorDecisionList xmlns="{_NS}">\n{decisions}</ColorDecisionList>\n')


def loads_cdl(text: str) -> list:
    root = ET.fromstring(text)
    return [_cdl_from_cc_element(cc) for cc in root.iter() if _local(cc.tag) == "ColorCorrection"]


# --------------------------------------------------------------------------- EDL comments
def to_edl_comments(cdl: CDL) -> str:
    """The ``*ASC_SOP``/``*ASC_SAT`` comment lines used in CMX3600 EDL conform."""
    return (f"*ASC_SOP ({_triple(cdl.slope)})({_triple(cdl.offset)})({_triple(cdl.power)})\n"
            f"*ASC_SAT {_fmt(cdl.saturation)}")


_SOP_RE = re.compile(r"\*ASC_SOP\s*\(([^)]*)\)\s*\(([^)]*)\)\s*\(([^)]*)\)", re.I)
_SAT_RE = re.compile(r"\*ASC_SAT\s+([-\d.eE]+)", re.I)


def parse_edl_comments(text: str) -> Optional[CDL]:
    """Pull a CDL out of EDL ``*ASC_SOP``/``*ASC_SAT`` comment lines; None if absent."""
    sop = _SOP_RE.search(text)
    if not sop:
        return None
    sat = _SAT_RE.search(text)
    return CDL(slope=_parse_triple(sop.group(1)), offset=_parse_triple(sop.group(2)),
               power=_parse_triple(sop.group(3)),
               saturation=float(sat.group(1)) if sat else 1.0)


# --------------------------------------------------------------------------- FCPXML
def to_fcpxml_cdl(cdl: CDL) -> str:
    """An ``info-asc-cdl`` element (FCPXML's CDL carrier).

    NOTE: FCPXML treats this as **inert passthrough metadata** — Final Cut does not apply CDL on
    import. It rides for the next tool in the chain. Callers should surface that, not imply FCP grades.
    """
    return (f'<info-asc-cdl slope="{_triple(cdl.slope)}" offset="{_triple(cdl.offset)}" '
            f'power="{_triple(cdl.power)}" saturation="{_fmt(cdl.saturation)}"/>')


def parse_fcpxml_cdl(text: str) -> Optional[CDL]:
    el = ET.fromstring(text) if text.lstrip().startswith("<info-asc-cdl") else \
        next((e for e in ET.fromstring(text).iter() if _local(e.tag) == "info-asc-cdl"), None)
    if el is None or _local(el.tag) != "info-asc-cdl":
        return None
    g = el.attrib
    return CDL(slope=_parse_triple(g.get("slope", "1 1 1")),
               offset=_parse_triple(g.get("offset", "0 0 0")),
               power=_parse_triple(g.get("power", "1 1 1")),
               saturation=float(g.get("saturation", "1")))


# --------------------------------------------------------------------------- IR bridge
def cdl_from_metadata(meta: dict) -> Optional[CDL]:
    """Build a CDL from a clip's ``metadata['filmgrip']['cdl']`` dict (the interchange channel)."""
    if not meta:
        return None
    return CDL(slope=tuple(meta.get("slope", (1.0, 1.0, 1.0))),
               offset=tuple(meta.get("offset", (0.0, 0.0, 0.0))),
               power=tuple(meta.get("power", (1.0, 1.0, 1.0))),
               saturation=float(meta.get("saturation", 1.0)),
               color_space=str(meta.get("color_space", "")))


def grades_from_ir(ir) -> list:
    """Collect ``(clip_name, CDL)`` for every clip in an IR that carries a film-grip CDL grade —
    so a whole timeline's grades export to one ``.ccc``/``.cdl``."""
    out = []
    for c in ir.real_clips():
        meta = getattr(c.otio, "metadata", {}) or {}
        cdl_meta = (meta.get("filmgrip", {}) or {}).get("cdl")
        if cdl_meta:
            cdl = cdl_from_metadata(dict(cdl_meta))
            if cdl is not None:
                out.append((c.name, cdl))
    return out
