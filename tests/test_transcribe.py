"""D1 — transcription core: backends, detection, cache, packing, SRT.

Hermetic by design: the fake backend supplies words, env/`shutil.which` are monkeypatched so a
machine with (or without) real ASR engines installed sees the same results, and no test ever
downloads a model or calls a network API.
"""
from __future__ import annotations

import json
import os

import pytest

from filmgrip.perception.transcribe import (
    FakeBackend,
    PerceptionUnavailable,
    Transcript,
    Word,
    cache_path_for,
    detect_backend,
    pack_transcript,
    to_srt,
    transcribe_media,
)

WORDS = [
    Word("Hey", 1.02, 1.50, "S0"),
    Word("everyone", 1.55, 2.10, "S0"),
    # ≥0.5s gap → new phrase
    Word("welcome", 2.80, 3.20, "S0"),
    Word("back", 3.25, 3.60, "S0"),
    # speaker change → new phrase
    Word("thanks", 3.70, 4.10, "S1"),
]


@pytest.fixture
def media(tmp_path):
    """A stand-in media file (contents irrelevant — the fake backend never reads it)."""
    p = tmp_path / "interview.mov"
    p.write_bytes(b"\x00" * 64)
    return str(p)


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    monkeypatch.setenv("FILMGRIP_CACHE_DIR", str(tmp_path / "cache"))


# --------------------------------------------------------------------------- data model
def test_word_round_trip_keeps_speaker_and_times():
    w = Word("Hey", 1.02, 1.5, "S0")
    assert Word.from_dict(w.to_dict()) == w


def test_word_round_trip_without_speaker_omits_key():
    w = Word("Hey", 1.02, 1.5)
    assert "spk" not in w.to_dict()
    assert Word.from_dict(w.to_dict()) == w


def test_transcript_round_trip():
    t = Transcript(media_path="/m.mov", backend="fake", words=WORDS, duration_s=4.1,
                   language="en")
    t2 = Transcript.from_dict(t.to_dict())
    assert t2.words == t.words
    assert t2.backend == "fake"
    assert t2.duration_s == pytest.approx(4.1)
    assert t2.language == "en"


# --------------------------------------------------------------------------- packing
def test_pack_breaks_on_silence_and_speaker_change():
    t = Transcript(media_path="m", backend="fake", words=WORDS)
    lines = pack_transcript(t).splitlines()
    assert lines == [
        "[0001.02-0002.10] S0 Hey everyone",
        "[0002.80-0003.60] S0 welcome back",
        "[0003.70-0004.10] S1 thanks",
    ]


def test_pack_omits_speaker_when_not_diarized():
    words = [Word("solo", 0.0, 0.4), Word("take", 0.45, 0.9)]
    t = Transcript(media_path="m", backend="fake", words=words)
    assert pack_transcript(t) == "[0000.00-0000.90] solo take"


def test_pack_empty_transcript_is_empty_string():
    t = Transcript(media_path="m", backend="fake", words=[])
    assert pack_transcript(t) == ""


def test_pack_is_token_frugal_vs_word_json():
    words = [Word(f"word{i}", i * 0.4, i * 0.4 + 0.3, "S0") for i in range(200)]
    t = Transcript(media_path="m", backend="fake", words=words)
    packed = pack_transcript(t)
    raw = json.dumps(t.to_dict())
    assert len(packed) < len(raw) / 4   # at least a 4x size win on continuous speech


# --------------------------------------------------------------------------- SRT
def test_to_srt_emits_valid_blocks():
    t = Transcript(media_path="m", backend="fake", words=WORDS)
    srt = to_srt(t)
    assert srt.startswith("1\n00:00:01,020 --> ")
    assert "Hey everyone" in srt
    # speaker-change phrase still becomes its own readable cue text
    assert "thanks" in srt


def test_to_srt_wraps_long_lines():
    words = [Word("supercalifragilistic", i * 1.0, i * 1.0 + 0.8) for i in range(6)]
    t = Transcript(media_path="m", backend="fake", words=words)
    blocks = to_srt(t, max_chars=25).strip().split("\n\n")
    assert len(blocks) >= 3   # 6 long words can't fit one 25-char cue


# --------------------------------------------------------------------------- cache
def test_transcribe_media_caches_by_content_identity(media):
    backend = FakeBackend(words=WORDS)
    t1 = transcribe_media(media, backend=backend)
    t2 = transcribe_media(media, backend=backend)
    assert backend.calls == 1            # second call served from the sidecar
    assert t2.words == t1.words
    assert os.path.isfile(cache_path_for(media, backend.name))


def test_cache_invalidates_when_media_changes(media):
    backend = FakeBackend(words=WORDS)
    transcribe_media(media, backend=backend)
    os.utime(media, ns=(1, 1))           # touch: new mtime → new cache key
    transcribe_media(media, backend=backend)
    assert backend.calls == 2


def test_corrupt_cache_entry_re_transcribes(media):
    backend = FakeBackend(words=WORDS)
    transcribe_media(media, backend=backend)
    with open(cache_path_for(media, backend.name), "w", encoding="utf-8") as fh:
        fh.write("{not json")
    t = transcribe_media(media, backend=backend)
    assert backend.calls == 2
    assert t.words == WORDS


def test_use_cache_false_bypasses(media):
    backend = FakeBackend(words=WORDS)
    transcribe_media(media, backend=backend, use_cache=False)
    transcribe_media(media, backend=backend, use_cache=False)
    assert backend.calls == 2


def test_missing_media_is_an_honest_error():
    with pytest.raises(PerceptionUnavailable, match="media file not found"):
        transcribe_media("/nope/missing.mov", backend=FakeBackend(words=WORDS))


# --------------------------------------------------------------------------- detection
@pytest.fixture
def no_real_backends(monkeypatch):
    """Make auto-detect deterministic: no python pkg, no CLI, no API key, no fake env."""
    monkeypatch.delenv("FILMGRIP_ASR_BACKEND", raising=False)
    monkeypatch.delenv("FILMGRIP_FAKE_ASR_JSON", raising=False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.delenv("FILMGRIP_WHISPER_CPP_MODEL", raising=False)
    import filmgrip.perception.transcribe as tr
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    import builtins
    real_import = builtins.__import__

    def block_faster_whisper(name, *a, **kw):
        if name == "faster_whisper":
            raise ImportError("blocked for test")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", block_faster_whisper)


def test_detect_with_nothing_available_lists_every_option(no_real_backends):
    with pytest.raises(PerceptionUnavailable) as exc:
        detect_backend()
    msg = str(exc.value)
    assert "faster-whisper" in msg
    assert "whisper-cpp" in msg
    assert "elevenlabs" in msg
    assert "film-grip[transcribe]" in msg   # actionable install hint


def test_detect_unknown_name_raises(no_real_backends):
    with pytest.raises(PerceptionUnavailable, match="unknown ASR backend"):
        detect_backend("magic-ears")


def test_detect_explicit_backend_unavailable_says_why(no_real_backends):
    with pytest.raises(PerceptionUnavailable, match="elevenlabs.*ELEVENLABS_API_KEY"):
        detect_backend("elevenlabs")


def test_detect_env_override_fake(monkeypatch, tmp_path, no_real_backends):
    payload = {"words": [w.to_dict() for w in WORDS], "language": "en"}
    fake_json = tmp_path / "fake.json"
    fake_json.write_text(json.dumps(payload))
    monkeypatch.setenv("FILMGRIP_ASR_BACKEND", "fake")
    monkeypatch.setenv("FILMGRIP_FAKE_ASR_JSON", str(fake_json))
    backend = detect_backend()
    assert backend.name == "fake"
    t = backend.transcribe("whatever.mov")
    assert t.words == WORDS
    assert t.language == "en"
