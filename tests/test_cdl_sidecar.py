"""Color deliverable 8 — CDL interchange sidecars: .cc/.ccc/.cdl (ASC v1.2), EDL ASC_SOP/ASC_SAT,
FCPXML info-asc-cdl. A grade written out must round-trip back bit-exactly."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from filmgrip.color.cdl import CDL
from filmgrip.serialize import cdl as S

C1 = CDL(slope=(1.1, 1.0, 0.9), offset=(0.0, 0.01, -0.02), power=(1.0, 0.95, 1.05),
         saturation=1.2, color_space="arri_logc4")
C2 = CDL(slope=(0.8, 0.9, 1.3), offset=(0.02, 0.0, 0.0), power=(1.0, 1.0, 1.0), saturation=0.7)


def _eq(a: CDL, b: CDL, tol=1e-6):
    return (all(abs(x - y) < tol for x, y in zip(a.slope, b.slope))
            and all(abs(x - y) < tol for x, y in zip(a.offset, b.offset))
            and all(abs(x - y) < tol for x, y in zip(a.power, b.power))
            and abs(a.saturation - b.saturation) < tol)


# ----------------------------------------------------------------------- .cc
def test_cc_roundtrip():
    back = S.loads_cc(S.dumps_cc(C1))
    assert _eq(back, C1) and back.color_space == "arri_logc4"


def test_cc_is_valid_xml_with_namespace():
    text = S.dumps_cc(C1)
    root = ET.fromstring(text)            # parses → well-formed
    assert root.tag == "{urn:ASC:CDL:v1.2}ColorCorrection"
    assert "<Slope>1.1 1 0.9</Slope>" in text


def test_cc_saturation_roundtrip_exact():
    # the precise contract from the deliverable verify line
    assert S.loads_cc(S.dumps_cc(C1)).saturation == 1.2


def test_loads_cc_tolerates_no_namespace():
    plain = ("<ColorCorrection id='x'><SOPNode><Slope>2 2 2</Slope>"
             "<Offset>0 0 0</Offset><Power>1 1 1</Power></SOPNode>"
             "<SatNode><Saturation>1.5</Saturation></SatNode></ColorCorrection>")
    c = S.loads_cc(plain)
    assert c.slope == (2.0, 2.0, 2.0) and c.saturation == 1.5


# ----------------------------------------------------------------------- .ccc / .cdl
def test_ccc_roundtrip_multiple():
    grades = S.loads_ccc(S.dumps_ccc([C1, C2]))
    assert len(grades) == 2 and _eq(grades[0], C1) and _eq(grades[1], C2)


def test_cdl_decision_list_roundtrip():
    text = S.dumps_cdl([C1, C2])
    assert "ColorDecisionList" in text and "ColorDecision" in text
    grades = S.loads_cdl(text)
    assert len(grades) == 2 and _eq(grades[1], C2)


# ----------------------------------------------------------------------- EDL comments
def test_edl_comment_roundtrip():
    text = S.to_edl_comments(C1)
    assert text.startswith("*ASC_SOP")
    back = S.parse_edl_comments("TITLE: show\n" + text + "\n")
    assert _eq(back, C1)


def test_edl_absent_returns_none():
    assert S.parse_edl_comments("TITLE: show\n001 clip V C\n") is None


# ----------------------------------------------------------------------- FCPXML
def test_fcpxml_cdl_roundtrip():
    el = S.to_fcpxml_cdl(C2)
    assert el.startswith("<info-asc-cdl")
    assert _eq(S.parse_fcpxml_cdl(el), C2)


# ----------------------------------------------------------------------- IR bridge
def test_grades_from_ir_exports_applied_cdls(tmp_path):
    import opentimelineio as otio
    from filmgrip.adapters.interchange import OtioMutator
    from filmgrip.core.ir import TimelineIR
    from filmgrip.protocol.editplan import EditPlan

    rt = lambda f: otio.opentime.RationalTime(f, 24.0)  # noqa: E731
    tl = otio.schema.Timeline(name="t")
    tl.global_start_time = rt(0)
    v = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    v.append(otio.schema.Clip(name="A", media_reference=otio.schema.ExternalReference(target_url="/m/a.mov"),
                              source_range=otio.opentime.TimeRange(rt(0), rt(48))))
    tl.tracks.append(v)
    p = str(tmp_path / "t.otio")
    otio.adapters.write_to_file(tl, p)
    ir = TimelineIR.from_otio_file(p)
    cid = ir.real_clips()[0].id
    OtioMutator(ir).apply(EditPlan.parse({"ops": [
        {"op": "set_cdl", "clip_id": cid, "slope": [1.1, 1.0, 0.9], "saturation": 1.2}]}))

    grades = S.grades_from_ir(ir)
    assert len(grades) == 1 and grades[0][0] == "A"
    # the collected grade exports + re-imports cleanly
    assert S.loads_ccc(S.dumps_ccc([g for _, g in grades]))[0].saturation == 1.2
