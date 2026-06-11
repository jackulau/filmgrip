"""D3 — speech analysis: silences, fillers, cut candidates, the silence-cut pack, MCP payload.

Scenario (24fps): one clip frames [0,480) over media 0..20s with words
Hey(1.0-1.4) um(2.0-2.3) welcome(2.5-3.0) back(6.0-6.4) → head silence [0,24),
interior silence [72,144) (3s), tail silence [154,480), one filler "um" at [48,55).
"""
from __future__ import annotations

import json

import opentimelineio as otio
import pytest

from filmgrip.adapters.interchange import REBUILD_OPS, OtioMutator
from filmgrip.core.ir import TimelineIR
from filmgrip.perception.align import AlignedWord
from filmgrip.perception.speech import (
    Filler,
    Silence,
    analyze_clips,
    filler_cut_ops,
    find_fillers,
    find_silences,
    silence_cut_ops,
)
from filmgrip.perception.transcribe import Transcript, Word

RATE = 24.0
WORDS_MEDIA = [
    Word("Hey", 1.00, 1.40),
    Word("um,", 2.00, 2.30),
    Word("welcome", 2.50, 3.00),
    Word("back", 6.00, 6.40),
]
ALIGNED = [
    AlignedWord("Hey", 24, 34),
    AlignedWord("um,", 48, 55),
    AlignedWord("welcome", 60, 72),
    AlignedWord("back", 144, 154),
]


# --------------------------------------------------------------------------- silences
def test_find_silences_head_interior_tail():
    sil = find_silences("c1", (0, 480), ALIGNED, RATE)
    assert [(s.kind, s.start_frame, s.end_frame) for s in sil] == [
        ("head", 0, 24),
        ("interior", 34, 48),      # Hey→um breath (0.58s)
        ("interior", 72, 144),     # welcome→back pause (3s)
        ("tail", 154, 480),
    ]
    assert sil[2].seconds == pytest.approx(3.0)


def test_find_silences_respects_threshold():
    sil = find_silences("c1", (0, 480), ALIGNED, RATE, min_silence_s=4.0)
    assert [(s.kind,) for s in sil] == [("tail",)]   # only the 13.6s tail survives a 4s bar


def test_speechless_clip_is_one_big_silence():
    sil = find_silences("c1", (0, 480), [], RATE)
    assert [(s.start_frame, s.end_frame) for s in sil] == [(0, 480)]


# --------------------------------------------------------------------------- fillers
def test_find_fillers_normalizes_punctuation_and_case():
    fills = find_fillers("c1", ALIGNED)
    assert [(f.text, f.start_frame, f.end_frame) for f in fills] == [("um,", 48, 55)]


def test_find_fillers_extra_vocab():
    words = ALIGNED + [AlignedWord("Basically", 200, 210)]
    fills = find_fillers("c1", words, extra={"basically"})
    assert {f.text for f in fills} == {"um,", "Basically"}


def test_meaningful_words_never_flagged():
    assert find_fillers("c1", [AlignedWord("umbrella", 10, 20)]) == []


# --------------------------------------------------------------------------- candidates
def test_silence_cut_ops_pad_inward_only_on_speech_side():
    sil = find_silences("c1", (0, 480), ALIGNED, RATE)
    ops = silence_cut_ops(sil, RATE, pad_s=0.1)   # pad = 2 frames @24
    by_start = sorted(ops, key=lambda o: o["start_frame"])
    assert by_start == [
        {"op": "cut_range", "clip_id": "c1", "start_frame": 0, "end_frame": 22, "ripple": True},
        {"op": "cut_range", "clip_id": "c1", "start_frame": 36, "end_frame": 46, "ripple": True},
        {"op": "cut_range", "clip_id": "c1", "start_frame": 74, "end_frame": 142, "ripple": True},
        {"op": "cut_range", "clip_id": "c1", "start_frame": 156, "end_frame": 480, "ripple": True},
    ]
    assert ops == sorted(ops, key=lambda o: o["start_frame"], reverse=True)  # descending contract


def test_tiny_silence_vanishes_under_padding():
    sil = [Silence("c1", 100, 103, 0.125, "interior")]   # 3 frames < 2*pad
    assert silence_cut_ops(sil, RATE, pad_s=0.1) == []


def test_filler_cut_pads_outward_but_never_into_words():
    fills = [Filler("c1", "um,", 48, 55)]
    ops = filler_cut_ops(fills, {"c1": ALIGNED}, RATE, pad_s=0.1)
    assert ops == [{"op": "cut_range", "clip_id": "c1",
                    "start_frame": 46, "end_frame": 57, "ripple": True}]
    # squeeze: neighbor words hard against the filler clamp the pad
    tight = [AlignedWord("a", 40, 47), AlignedWord("um", 48, 55), AlignedWord("b", 56, 60)]
    ops = filler_cut_ops([Filler("c1", "um", 48, 55)], {"c1": tight}, RATE, pad_s=0.1)
    assert ops == [{"op": "cut_range", "clip_id": "c1",
                    "start_frame": 48, "end_frame": 55, "ripple": True}]


# --------------------------------------------------------------------------- full analysis
def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


@pytest.fixture
def media(tmp_path):
    p = tmp_path / "interview.mov"
    p.write_bytes(b"\x00" * 32)
    return p


@pytest.fixture
def ir(media) -> TimelineIR:
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(otio.schema.Clip(
        name="take1",
        media_reference=otio.schema.ExternalReference(target_url=media.as_uri()),
        source_range=otio.opentime.TimeRange(_rt(0), _rt(480)),
    ))
    return TimelineIR(tl)


@pytest.fixture
def transcriber():
    def fake(path, *, backend=None, **kw):
        return Transcript(media_path=path, backend="fake", words=WORDS_MEDIA)
    return fake


def test_analyze_clips_payload(ir, transcriber):
    cid = ir.real_clips()[0].id
    payload = analyze_clips(ir, [cid], transcriber=transcriber)
    assert payload["errors"] == []
    clip = payload["clips"][0]
    assert [s[3] for s in clip["silences"]] == ["head", "interior", "interior", "tail"]
    assert clip["fillers"] == [["um,", 48, 55]]
    starts = [c["start_frame"] for c in payload["candidates"]]
    assert starts == sorted(starts, reverse=True)
    assert {c["op"] for c in payload["candidates"]} == {"cut_range"}


def test_analyze_clips_refuses_retimed(ir, transcriber):
    clip = ir.real_clips()[0]
    clip.otio.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    payload = analyze_clips(ir, [clip.id], transcriber=transcriber)
    assert payload["clips"] == [] and payload["candidates"] == []
    assert any("retimed" in e for e in payload["errors"])


# --------------------------------------------------------------------------- silence-cut pack
@pytest.fixture
def fake_asr_env(monkeypatch, tmp_path):
    payload = {"words": [w.to_dict() for w in WORDS_MEDIA]}
    fake_json = tmp_path / "asr.json"
    fake_json.write_text(json.dumps(payload))
    monkeypatch.setenv("FILMGRIP_ASR_BACKEND", "fake")
    monkeypatch.setenv("FILMGRIP_FAKE_ASR_JSON", str(fake_json))
    monkeypatch.setenv("FILMGRIP_CACHE_DIR", str(tmp_path / "cache"))


def test_silence_cut_pack_compiles_validates_and_applies(ir, fake_asr_env):
    from filmgrip.packs import get_pack
    from filmgrip.packs.engine import compile_pack
    from filmgrip.protocol.validate import validate

    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("silence-cut"), ir, [cid])
    assert plan.ops, "expected cut candidates"
    assert {op.op for op in plan.ops} <= REBUILD_OPS          # applicability honesty
    res = validate(plan, ir)
    assert res.ok, [str(e) for e in res.errors]
    OtioMutator(ir).apply(plan)
    ir.reindex()
    total = sum(c.duration for c in ir.real_clips())
    # removed: head 22 + breath 10 + pause 68 + tail 324 + filler 11 = 435 of 480 frames
    assert total == 45


def test_silence_cut_pack_fillers_off(ir, fake_asr_env):
    from filmgrip.packs import get_pack
    from filmgrip.packs.engine import compile_pack

    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("silence-cut"), ir, [cid], params={"fillers": "false"})
    starts = {op.start_frame for op in plan.ops}
    assert 46 not in starts                                   # the um cut is gone
    assert len(plan.ops) == 4


def test_silence_cut_pack_min_silence_param(ir, fake_asr_env):
    from filmgrip.packs import get_pack
    from filmgrip.packs.engine import compile_pack

    cid = ir.real_clips()[0].id
    plan = compile_pack(get_pack("silence-cut"), ir, [cid],
                        params={"min_silence": "4s", "fillers": "false"})
    assert len(plan.ops) == 1                                  # only the 13.6s tail
    assert plan.ops[0].start_frame == 156


def test_silence_cut_pack_honest_without_backend(ir, monkeypatch):
    from filmgrip.packs import PackError, get_pack
    from filmgrip.packs.engine import compile_pack

    monkeypatch.delenv("FILMGRIP_ASR_BACKEND", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("FILMGRIP_WHISPER_CPP_MODEL", raising=False)
    import filmgrip.perception.transcribe as tr
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    import builtins
    real_import = builtins.__import__

    def block(name, *a, **kw):
        if name == "faster_whisper":
            raise ImportError("blocked")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", block)
    with pytest.raises(PackError, match="transcription backend"):
        compile_pack(get_pack("silence-cut"), ir, [ir.real_clips()[0].id])


def test_silence_cut_pack_declares_requirements():
    from filmgrip.packs import get_pack

    assert get_pack("silence-cut").requires


# --------------------------------------------------------------------------- MCP payload
def test_payload_analyze_speech(ir, fake_asr_env):
    from filmgrip.adapters.base import Selection
    from filmgrip.integration import mcp_host as mh

    cid = ir.real_clips()[0].id
    ctx = mh.PlannerContext(ir=ir, selection=Selection(ids=[cid], basis="fixture"))
    payload = mh.payload_analyze_speech(ctx)
    assert payload["errors"] == []
    assert payload["candidates"], "expected cut candidates"
    assert payload["clips"][0]["fillers"] == [["um,", 48, 55]]
