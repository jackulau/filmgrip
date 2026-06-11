"""Word-level transcription with pluggable ASR backends, a sidecar cache, and token packing.

The design steals the proven shape from the strongest pipeline tools (video-use, the Fable
launch-video workflow) and keeps film-grip's honesty rules:

* **word timestamps are the product** — ASR text may be wrong ("Thariq" → "Sark"); the
  ``start``/``end`` seconds are what edits anchor to.
* **pluggable backends, honest detection** — ``faster-whisper`` (python), ``whisper.cpp``
  (CLI), ElevenLabs Scribe (API, adds diarization + audio events), or an injected fake for
  tests. If nothing is available, :func:`detect_backend` raises with install options instead
  of pretending.
* **cache or it didn't happen** — ASR is the slowest, most expensive step; results are JSON
  sidecars keyed by (path, size, mtime, backend) under ``~/.filmgrip/transcripts``.
* **packed phrases, not raw JSON** — :func:`pack_transcript` emits ``[0002.52-0005.36] S0 text``
  lines broken on silence/speaker change: ~1/10 the tokens of word JSON, still timestamped.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from typing import Optional, Protocol

#: Phrase break threshold for packing: a gap of silence this long (seconds) starts a new line.
PACK_GAP_S = 0.5

#: ASR word timestamps drift 50–100ms in practice; cut candidates must pad by at least this.
ASR_DRIFT_PAD_S = (0.030, 0.200)  # (min, max) padding seconds — used by perception.speech


class PerceptionUnavailable(RuntimeError):
    """A perception dependency (ASR backend, ffmpeg, media file) is missing — with the fix."""


# --------------------------------------------------------------------------- data model
@dataclass(frozen=True)
class Word:
    """One recognized word (or audio event like ``(laughter)``) in MEDIA seconds."""

    text: str
    start: float
    end: float
    speaker: Optional[str] = None   # diarization label ("S0") when the backend provides one

    def to_dict(self) -> dict:
        d: dict = {"t": self.text, "s": round(self.start, 3), "e": round(self.end, 3)}
        if self.speaker is not None:
            d["spk"] = self.speaker
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Word":
        return cls(text=d["t"], start=float(d["s"]), end=float(d["e"]), speaker=d.get("spk"))


@dataclass
class Transcript:
    """All recognized words for one media file, in media time."""

    media_path: str
    backend: str
    words: list[Word] = field(default_factory=list)
    duration_s: float = 0.0          # media duration (ffprobe when available, else last word end)
    language: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "media_path": self.media_path,
            "backend": self.backend,
            "duration_s": round(self.duration_s, 3),
            "language": self.language,
            "words": [w.to_dict() for w in self.words],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Transcript":
        return cls(
            media_path=d["media_path"],
            backend=d["backend"],
            words=[Word.from_dict(w) for w in d.get("words", [])],
            duration_s=float(d.get("duration_s", 0.0)),
            language=d.get("language"),
        )


class Backend(Protocol):
    """One ASR engine. ``available()`` returns (ok, reason-if-not)."""

    name: str

    def available(self) -> tuple[bool, str]: ...

    def transcribe(self, media_path: str) -> Transcript: ...


# --------------------------------------------------------------------------- ffmpeg helpers
def ffmpeg_path() -> Optional[str]:
    return shutil.which("ffmpeg")


def ffprobe_duration(media_path: str) -> float:
    """Media duration in seconds via ffprobe; 0.0 when ffprobe is unavailable or fails."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        out = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", media_path],
            capture_output=True, text=True, timeout=30,
        )
        return float(out.stdout.strip()) if out.returncode == 0 and out.stdout.strip() else 0.0
    except (subprocess.SubprocessError, ValueError):
        return 0.0


def extract_wav(media_path: str, out_path: str, *, rate: int = 16000) -> None:
    """Extract mono 16k WAV (what ASR engines want) from any media file via ffmpeg."""
    ffmpeg = ffmpeg_path()
    if not ffmpeg:
        raise PerceptionUnavailable(
            "ffmpeg is required to extract audio for transcription — install it "
            "(e.g. `brew install ffmpeg`) and re-run."
        )
    proc = subprocess.run(
        [ffmpeg, "-y", "-v", "error", "-i", media_path, "-vn", "-ac", "1",
         "-ar", str(rate), "-f", "wav", out_path],
        capture_output=True, text=True, timeout=600,
    )
    if proc.returncode != 0:
        raise PerceptionUnavailable(
            f"ffmpeg could not extract audio from '{media_path}': {proc.stderr.strip()[:300]}"
        )


def _multipart(*, fields: dict[str, str], file_field: str, filename: str,
               file_bytes: bytes, file_type: str) -> tuple[bytes, str]:
    """Build a multipart/form-data body with the stdlib (keeps `requests` out of core deps)."""
    boundary = f"filmgrip{uuid.uuid4().hex}"
    parts: list[bytes] = []
    for key, value in fields.items():
        parts.append(
            (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{key}\"\r\n\r\n"
             f"{value}\r\n").encode("utf-8"))
    parts.append(
        (f"--{boundary}\r\nContent-Disposition: form-data; name=\"{file_field}\"; "
         f"filename=\"{filename}\"\r\nContent-Type: {file_type}\r\n\r\n").encode("utf-8"))
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


# --------------------------------------------------------------------------- backends
class FasterWhisperBackend:
    """Local ASR via the ``faster-whisper`` python package (no network after model download)."""

    name = "faster-whisper"

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "python package not installed (pip install 'film-grip[transcribe]')"
        return True, ""

    def transcribe(self, media_path: str) -> Transcript:
        from faster_whisper import WhisperModel  # lazy: optional dep

        model_size = os.environ.get("FILMGRIP_WHISPER_MODEL", "base")
        model = WhisperModel(model_size, compute_type="auto")
        segments, info = model.transcribe(media_path, word_timestamps=True)
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                words.append(Word(text=w.word.strip(), start=float(w.start), end=float(w.end)))
        return Transcript(
            media_path=media_path, backend=self.name, words=words,
            duration_s=ffprobe_duration(media_path) or (words[-1].end if words else 0.0),
            language=getattr(info, "language", None),
        )


class WhisperCppBackend:
    """Local ASR via the whisper.cpp CLI (``whisper-cli``, e.g. ``brew install whisper-cpp``).

    Needs a ggml model file: set ``FILMGRIP_WHISPER_CPP_MODEL=/path/to/ggml-base.bin``.
    ``-ml 1`` makes each JSON segment ≈ one word, giving word-level timestamps.
    """

    name = "whisper-cpp"

    def _binary(self) -> Optional[str]:
        return shutil.which("whisper-cli") or shutil.which("whisper-cpp")

    def available(self) -> tuple[bool, str]:
        if not self._binary():
            return False, "whisper-cli not on PATH (brew install whisper-cpp)"
        if not os.environ.get("FILMGRIP_WHISPER_CPP_MODEL"):
            return False, "FILMGRIP_WHISPER_CPP_MODEL not set (path to a ggml model file)"
        if not os.path.isfile(os.environ["FILMGRIP_WHISPER_CPP_MODEL"]):
            return False, f"model file not found: {os.environ['FILMGRIP_WHISPER_CPP_MODEL']}"
        return True, ""

    def transcribe(self, media_path: str) -> Transcript:
        binary = self._binary()
        model = os.environ["FILMGRIP_WHISPER_CPP_MODEL"]
        with tempfile.TemporaryDirectory(prefix="filmgrip-asr-") as tmp:
            wav = os.path.join(tmp, "audio.wav")
            extract_wav(media_path, wav)
            out_base = os.path.join(tmp, "out")
            proc = subprocess.run(
                [binary, "-m", model, "-f", wav, "-ml", "1", "-oj", "-of", out_base],
                capture_output=True, text=True, timeout=3600,
            )
            if proc.returncode != 0:
                raise PerceptionUnavailable(
                    f"whisper-cli failed on '{media_path}': {proc.stderr.strip()[:300]}")
            with open(out_base + ".json", "r", encoding="utf-8") as fh:
                data = json.load(fh)
        words = []
        for seg in data.get("transcription", []):
            text = (seg.get("text") or "").strip()
            offs = seg.get("offsets") or {}
            if not text:
                continue
            words.append(Word(text=text, start=float(offs.get("from", 0)) / 1000.0,
                              end=float(offs.get("to", 0)) / 1000.0))
        return Transcript(
            media_path=media_path, backend=self.name, words=words,
            duration_s=ffprobe_duration(media_path) or (words[-1].end if words else 0.0),
            language=(data.get("result") or {}).get("language"),
        )


class ElevenLabsBackend:
    """ElevenLabs Scribe (API) — word timestamps + speaker diarization + audio events.

    The only backend that labels speakers and events like ``(laughter)``; needs
    ``ELEVENLABS_API_KEY``. Audio is extracted to 16k mono WAV before upload.
    """

    name = "elevenlabs"
    _url = "https://api.elevenlabs.io/v1/speech-to-text"

    def available(self) -> tuple[bool, str]:
        if not os.environ.get("ELEVENLABS_API_KEY"):
            return False, "ELEVENLABS_API_KEY not set"
        return True, ""

    def transcribe(self, media_path: str) -> Transcript:
        api_key = os.environ["ELEVENLABS_API_KEY"]
        with tempfile.TemporaryDirectory(prefix="filmgrip-asr-") as tmp:
            wav = os.path.join(tmp, "audio.wav")
            extract_wav(media_path, wav)
            with open(wav, "rb") as fh:
                audio = fh.read()
        body, content_type = _multipart(
            fields={"model_id": "scribe_v1", "diarize": "true", "timestamps_granularity": "word"},
            file_field="file", filename="audio.wav", file_bytes=audio, file_type="audio/wav",
        )
        req = urllib.request.Request(
            self._url, data=body,
            headers={"xi-api-key": api_key, "Content-Type": content_type}, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300]
            raise PerceptionUnavailable(
                f"ElevenLabs Scribe rejected '{media_path}' (HTTP {exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise PerceptionUnavailable(f"ElevenLabs Scribe unreachable: {exc.reason}") from exc
        words = []
        for w in data.get("words", []):
            kind = w.get("type", "word")
            if kind == "spacing":
                continue
            text = (w.get("text") or "").strip()
            if not text:
                continue
            spk = w.get("speaker_id")
            words.append(Word(text=text, start=float(w.get("start", 0.0)),
                              end=float(w.get("end", 0.0)),
                              speaker=str(spk) if spk is not None else None))
        return Transcript(
            media_path=media_path, backend=self.name, words=words,
            duration_s=ffprobe_duration(media_path) or (words[-1].end if words else 0.0),
            language=data.get("language_code"),
        )


class FakeBackend:
    """Deterministic backend for tests/offline demos: words come from a JSON file.

    Activate with ``FILMGRIP_ASR_BACKEND=fake`` + ``FILMGRIP_FAKE_ASR_JSON=/path/words.json``
    (a ``Transcript.to_dict()`` payload or just ``{"words": [...]}``), or construct directly.
    """

    name = "fake"

    def __init__(self, words: Optional[list[Word]] = None, language: Optional[str] = None):
        self._words = words
        self._language = language
        self.calls = 0   # tests assert the cache prevents repeat calls

    def available(self) -> tuple[bool, str]:
        if self._words is None and not os.environ.get("FILMGRIP_FAKE_ASR_JSON"):
            return False, "FILMGRIP_FAKE_ASR_JSON not set"
        return True, ""

    def transcribe(self, media_path: str) -> Transcript:
        self.calls += 1
        words, language = self._words, self._language
        if words is None:
            with open(os.environ["FILMGRIP_FAKE_ASR_JSON"], "r", encoding="utf-8") as fh:
                data = json.load(fh)
            words = [Word.from_dict(w) for w in data.get("words", [])]
            language = data.get("language")
        return Transcript(media_path=media_path, backend=self.name, words=list(words),
                          duration_s=(words[-1].end if words else 0.0), language=language)


_BACKENDS: dict[str, type] = {
    FasterWhisperBackend.name: FasterWhisperBackend,
    WhisperCppBackend.name: WhisperCppBackend,
    ElevenLabsBackend.name: ElevenLabsBackend,
    FakeBackend.name: FakeBackend,
}

#: Auto-detect preference: local python > local CLI > paid API. ``fake`` is opt-in only.
_DETECT_ORDER = [FasterWhisperBackend, WhisperCppBackend, ElevenLabsBackend]


def detect_backend(name: Optional[str] = None) -> Backend:
    """Resolve the ASR backend: explicit arg > ``FILMGRIP_ASR_BACKEND`` env > auto-detect.

    Raises :class:`PerceptionUnavailable` with every option's status when nothing works —
    film-grip never silently degrades to a worse engine than the one you asked for.
    """
    name = name or os.environ.get("FILMGRIP_ASR_BACKEND")
    if name:
        cls = _BACKENDS.get(name)
        if cls is None:
            raise PerceptionUnavailable(
                f"unknown ASR backend '{name}' (choose from: {', '.join(sorted(_BACKENDS))})")
        backend = cls()
        ok, reason = backend.available()
        if not ok:
            raise PerceptionUnavailable(f"ASR backend '{name}' unavailable: {reason}")
        return backend
    reasons = []
    for cls in _DETECT_ORDER:
        backend = cls()
        ok, reason = backend.available()
        if ok:
            return backend
        reasons.append(f"  - {backend.name}: {reason}")
    raise PerceptionUnavailable(
        "no ASR backend available — transcription needs one of:\n" + "\n".join(reasons)
        + "\nInstall one (fastest: pip install 'film-grip[transcribe]') or set "
          "FILMGRIP_ASR_BACKEND/ELEVENLABS_API_KEY."
    )


# --------------------------------------------------------------------------- cache
def cache_dir() -> str:
    return os.environ.get("FILMGRIP_CACHE_DIR") or os.path.expanduser("~/.filmgrip/transcripts")


def _cache_key(media_path: str, backend_name: str) -> str:
    st = os.stat(media_path)
    raw = f"{os.path.abspath(media_path)}|{st.st_size}|{st.st_mtime_ns}|{backend_name}|v1"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def cache_path_for(media_path: str, backend_name: str) -> str:
    return os.path.join(cache_dir(), f"{_cache_key(media_path, backend_name)}.json")


def transcribe_media(media_path: str, *, backend: Optional[Backend] = None,
                     use_cache: bool = True) -> Transcript:
    """Transcribe one media file, hitting the sidecar cache first.

    The cache key includes size + mtime, so a re-exported file re-transcribes; a re-run on
    untouched footage is free (ASR is the slowest step — video-use's hard rule #1 is "cache
    transcripts", and it is right).
    """
    if not os.path.isfile(media_path):
        raise PerceptionUnavailable(
            f"media file not found: '{media_path}' — perception needs the source media on disk "
            f"(offline/proxy-only media cannot be transcribed)")
    backend = backend or detect_backend()
    cpath = cache_path_for(media_path, backend.name)
    if use_cache and os.path.isfile(cpath):
        try:
            with open(cpath, "r", encoding="utf-8") as fh:
                return Transcript.from_dict(json.load(fh))
        except (json.JSONDecodeError, KeyError, ValueError):
            pass  # corrupt cache entry → re-transcribe over it
    transcript = backend.transcribe(media_path)
    if use_cache:
        os.makedirs(cache_dir(), exist_ok=True)
        tmp = cpath + f".tmp{uuid.uuid4().hex[:8]}"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(transcript.to_dict(), fh, separators=(",", ":"))
        os.replace(tmp, cpath)
    return transcript


# --------------------------------------------------------------------------- packing
def pack_transcript(transcript: Transcript, *, gap_s: float = PACK_GAP_S) -> str:
    """Pack words into the token-frugal phrase format the planner reads.

    One line per phrase: ``[0002.52-0005.36] S0 text…``, broken on silence ≥ ``gap_s`` or a
    speaker change. ~1/10 the tokens of word JSON while keeping the timestamps that matter.
    The speaker label is omitted entirely when the backend did not diarize.
    """
    lines: list[str] = []
    phrase: list[Word] = []

    def flush() -> None:
        if not phrase:
            return
        spk = f" {phrase[0].speaker}" if phrase[0].speaker is not None else ""
        text = " ".join(w.text for w in phrase)
        lines.append(f"[{phrase[0].start:07.2f}-{phrase[-1].end:07.2f}]{spk} {text}")
        phrase.clear()

    for word in transcript.words:
        if phrase and (word.start - phrase[-1].end >= gap_s
                       or word.speaker != phrase[0].speaker):
            flush()
        phrase.append(word)
    flush()
    return "\n".join(lines)


def to_srt(transcript: Transcript, *, max_chars: int = 42, max_dur_s: float = 5.0) -> str:
    """Render the transcript as an SRT caption file (importable into any NLE)."""
    cues: list[tuple[float, float, str]] = []
    cur: list[Word] = []
    for word in transcript.words:
        candidate = " ".join(w.text for w in cur + [word])
        if cur and (len(candidate) > max_chars
                    or word.end - cur[0].start > max_dur_s
                    or word.start - cur[-1].end >= PACK_GAP_S):
            cues.append((cur[0].start, cur[-1].end, " ".join(w.text for w in cur)))
            cur = []
        cur.append(word)
    if cur:
        cues.append((cur[0].start, cur[-1].end, " ".join(w.text for w in cur)))

    def ts(seconds: float) -> str:
        ms = int(round(seconds * 1000))
        h, rem = divmod(ms, 3_600_000)
        m, rem = divmod(rem, 60_000)
        s, ms = divmod(rem, 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    blocks = [f"{i}\n{ts(s)} --> {ts(e)}\n{text}\n" for i, (s, e, text) in enumerate(cues, 1)]
    return "\n".join(blocks)
