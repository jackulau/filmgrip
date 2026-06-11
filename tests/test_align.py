"""D2 — transcript↔timeline alignment + the get_transcript MCP payload + CLI.

Timeline under test (24fps): two clips cut from the same interview.mov —
take1 uses media 0–2s placed at frames 0–48, take2 uses media 4–6s placed at frames 48–96.
Words at media 1.0s must land ~frame 24; words at 4.5s must land ~frame 60 (inside take2).
"""
from __future__ import annotations

import json

import opentimelineio as otio
import pytest

from filmgrip.adapters.base import Selection
from filmgrip.core.ir import TimelineIR
from filmgrip.perception.align import (
    AlignedWord,
    align_clip_words,
    aligned_srt,
    is_retimed,
    media_path_of,
    pack_aligned,
    transcript_for_clips,
)
from filmgrip.perception.transcribe import FakeBackend, Transcript, Word

WORDS = [
    Word("Hey", 1.00, 1.40, None),
    Word("everyone", 1.45, 1.90, None),
    Word("dropped", 3.00, 3.20, None),    # media 3s is cut out (between the two takes)
    Word("welcome", 4.50, 4.90, None),
    Word("back", 4.95, 5.40, None),
]


def _rt(frames: int) -> otio.opentime.RationalTime:
    return otio.opentime.RationalTime(frames, 24)


def _clip(name: str, media_uri: str, src_in_f: int, dur_f: int) -> otio.schema.Clip:
    return otio.schema.Clip(
        name=name,
        media_reference=otio.schema.ExternalReference(target_url=media_uri),
        source_range=otio.opentime.TimeRange(_rt(src_in_f), _rt(dur_f)),
    )


@pytest.fixture
def media(tmp_path):
    p = tmp_path / "interview take.mov"     # space on purpose: exercises %20 decoding
    p.write_bytes(b"\x00" * 32)
    return p


@pytest.fixture
def ir(media) -> TimelineIR:
    tl = otio.schema.Timeline(name="seq")
    track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    tl.tracks.append(track)
    track.append(_clip("take1", media.as_uri(), 0, 48))
    track.append(_clip("take2", media.as_uri(), 96, 48))
    return TimelineIR(tl)


def _ids(ir):
    return [c.id for c in ir.real_clips()]


@pytest.fixture
def transcriber():
    """Injectable transcriber: returns WORDS for any path, counts calls."""
    calls = []

    def fake(path, *, backend=None, **kw):
        calls.append(path)
        return Transcript(media_path=path, backend="fake", words=WORDS)

    fake.calls = calls
    return fake


# --------------------------------------------------------------------------- media paths
def test_media_path_of_decodes_file_url(ir, media):
    clip = ir.real_clips()[0]
    assert media_path_of(clip) == str(media)


def _solo_ir(media_reference) -> TimelineIR:
    c = otio.schema.Clip(name="x", media_reference=media_reference,
                         source_range=otio.opentime.TimeRange(_rt(0), _rt(48)))
    tl = otio.schema.Timeline(name="t")
    tr = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    tl.tracks.append(tr)
    tr.append(c)
    return TimelineIR(tl)


def test_media_path_of_plain_path():
    ir = _solo_ir(otio.schema.ExternalReference(target_url="/footage/raw.mov"))
    assert media_path_of(ir.real_clips()[0]) == "/footage/raw.mov"


def test_media_path_of_missing_reference_is_none():
    ir = _solo_ir(otio.schema.MissingReference())
    assert media_path_of(ir.real_clips()[0]) is None


# --------------------------------------------------------------------------- alignment math
def test_words_map_to_timeline_frames(ir):
    t = Transcript(media_path="m", backend="fake", words=WORDS)
    take1, take2 = ir.real_clips()
    w1 = align_clip_words(take1, t, ir.rate)
    assert [w.text for w in w1] == ["Hey", "everyone"]
    assert w1[0].start_frame == 24            # 1.00s * 24
    assert w1[0].end_frame == 34              # 1.40s * 24 = 33.6 → 34
    w2 = align_clip_words(take2, t, ir.rate)
    assert [w.text for w in w2] == ["welcome", "back"]
    assert w2[0].start_frame == 60            # 48 + (4.5-4.0)*24
    assert w2[-1].end_frame <= take2.end      # clamped inside the clip


def test_word_outside_every_clip_is_dropped(ir):
    t = Transcript(media_path="m", backend="fake", words=WORDS)
    texts = [w.text for c in ir.real_clips() for w in align_clip_words(c, t, ir.rate)]
    assert "dropped" not in texts


def test_word_straddling_clip_end_clamps_to_span(ir):
    take1 = ir.real_clips()[0]
    t = Transcript(media_path="m", backend="fake",
                   words=[Word("edge", 1.95, 2.40)])   # midpoint 2.175 > 2.0 → excluded
    assert align_clip_words(take1, t, ir.rate) == []
    t2 = Transcript(media_path="m", backend="fake",
                    words=[Word("edge", 1.80, 2.30)])  # midpoint 2.05 → out as well
    assert align_clip_words(take1, t2, ir.rate) == []
    t3 = Transcript(media_path="m", backend="fake",
                    words=[Word("edge", 1.70, 2.20)])  # midpoint 1.95 → in, end clamps to 48
    aligned = align_clip_words(take1, t3, ir.rate)
    assert aligned and aligned[0].end_frame == take1.end


# --------------------------------------------------------------------------- packing
def test_pack_aligned_breaks_on_gap_and_speaker():
    words = [
        AlignedWord("Hey", 24, 34, "S0"),
        AlignedWord("everyone", 35, 46, "S0"),
        AlignedWord("welcome", 70, 80, "S0"),     # 24-frame gap (1s @24) → break
        AlignedWord("thanks", 82, 90, "S1"),      # speaker change → break
    ]
    lines = pack_aligned(words, 24.0)
    assert lines == [
        "[0024-0046] S0 Hey everyone",
        "[0070-0080] S0 welcome",
        "[0082-0090] S1 thanks",
    ]


def test_pack_aligned_omits_speaker_when_absent():
    lines = pack_aligned([AlignedWord("solo", 10, 20)], 24.0)
    assert lines == ["[0010-0020] solo"]


# --------------------------------------------------------------------------- payload
def test_transcript_for_clips_payload_shape(ir, transcriber):
    payload = transcript_for_clips(ir, _ids(ir), transcriber=transcriber)
    assert payload["r"] == "24"
    assert payload["frames"] == "timeline"
    assert payload["errors"] == []
    assert [c["id"] for c in payload["clips"]] == _ids(ir)
    take1 = payload["clips"][0]
    assert take1["span"] == [0, 48]
    assert take1["phrases"] == ["[0024-0046] Hey everyone"]
    take2 = payload["clips"][1]
    assert take2["phrases"][0].startswith("[0060-")


def test_transcribes_each_media_file_once(ir, transcriber):
    transcript_for_clips(ir, _ids(ir), transcriber=transcriber)
    assert len(transcriber.calls) == 1        # two clips, one source file → one ASR run


def test_retimed_clip_is_refused_honestly(ir, transcriber):
    clip = ir.real_clips()[0]
    clip.otio.effects.append(otio.schema.LinearTimeWarp(time_scalar=2.0))
    assert is_retimed(clip)
    payload = transcript_for_clips(ir, _ids(ir), transcriber=transcriber)
    assert len(payload["clips"]) == 1         # take2 still works
    assert any("retimed" in e for e in payload["errors"])


def test_unknown_media_path_is_an_error_not_a_guess(transcriber):
    c = otio.schema.Clip(name="offline", media_reference=otio.schema.MissingReference(),
                         source_range=otio.opentime.TimeRange(_rt(0), _rt(48)))
    tl = otio.schema.Timeline(name="t")
    tr = otio.schema.Track(kind=otio.schema.TrackKind.Video)
    tl.tracks.append(tr)
    tr.append(c)
    ir = TimelineIR(tl)
    payload = transcript_for_clips(ir, _ids(ir), transcriber=transcriber)
    assert payload["clips"] == []
    assert any("media path unknown" in e for e in payload["errors"])
    assert transcriber.calls == []


def test_bogus_id_is_an_error(ir, transcriber):
    payload = transcript_for_clips(ir, ["nope"], transcriber=transcriber)
    assert any("not a clip" in e for e in payload["errors"])


# --------------------------------------------------------------------------- SRT
def test_aligned_srt_uses_timeline_time(ir, transcriber):
    srt, errors = aligned_srt(ir, _ids(ir), transcriber=transcriber)
    assert errors == []
    assert "00:00:01,000 --> " in srt          # frame 24 @24fps = 1.0s timeline
    assert "Hey everyone" in srt
    assert "welcome back" in srt


# --------------------------------------------------------------------------- MCP payload
@pytest.fixture
def fake_asr_env(monkeypatch, tmp_path):
    payload = {"words": [w.to_dict() for w in WORDS]}
    fake_json = tmp_path / "asr.json"
    fake_json.write_text(json.dumps(payload))
    monkeypatch.setenv("FILMGRIP_ASR_BACKEND", "fake")
    monkeypatch.setenv("FILMGRIP_FAKE_ASR_JSON", str(fake_json))
    monkeypatch.setenv("FILMGRIP_CACHE_DIR", str(tmp_path / "cache"))


def test_payload_get_transcript_end_to_end(ir, fake_asr_env):
    from filmgrip.integration import mcp_host as mh

    ctx = mh.PlannerContext(ir=ir, selection=Selection(ids=_ids(ir), basis="fixture"))
    payload = mh.payload_get_transcript(ctx)
    assert payload["errors"] == []
    assert payload["clips"][0]["phrases"] == ["[0024-0046] Hey everyone"]


def test_payload_get_transcript_no_backend_is_honest(ir, monkeypatch):
    from filmgrip.integration import mcp_host as mh

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
    ctx = mh.PlannerContext(ir=ir, selection=Selection(ids=_ids(ir), basis="fixture"))
    payload = mh.payload_get_transcript(ctx)
    assert payload["clips"] == []
    assert any("no ASR backend available" in e for e in payload["errors"])


# --------------------------------------------------------------------------- CLI
def test_cli_transcribe_fixture(ir, fake_asr_env, tmp_path, capsys):
    fixture = tmp_path / "seq.otio"
    otio.adapters.write_to_file(ir.timeline, str(fixture))
    from filmgrip.cli import main

    rc = main(["transcribe", "--fixture", str(fixture)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Hey everyone" in out
    assert "[0024-0046]" in out


def test_cli_transcribe_fixture_srt(ir, fake_asr_env, tmp_path, capsys):
    fixture = tmp_path / "seq.otio"
    otio.adapters.write_to_file(ir.timeline, str(fixture))
    out_srt = tmp_path / "captions.srt"
    from filmgrip.cli import main

    rc = main(["transcribe", "--fixture", str(fixture), "--srt", str(out_srt)])
    assert rc == 0
    assert "Hey everyone" in out_srt.read_text()


def test_cli_transcribe_media_packed(fake_asr_env, tmp_path, capsys):
    media = tmp_path / "raw.mov"
    media.write_bytes(b"\x00" * 16)
    from filmgrip.cli import main

    rc = main(["transcribe", "--media", str(media)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Hey everyone" in out               # media-time packing
    assert "[0001.00-0001.90]" in out


def test_cli_transcribe_no_backend_errors_cleanly(tmp_path, capsys, monkeypatch):
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
    media = tmp_path / "raw.mov"
    media.write_bytes(b"\x00" * 16)
    from filmgrip.cli import main

    rc = main(["transcribe", "--media", str(media)])
    out = capsys.readouterr().out
    assert rc == 1
    assert "no ASR backend available" in out
