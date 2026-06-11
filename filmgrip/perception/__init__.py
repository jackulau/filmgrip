"""Perception — the agent's senses over the actual media, not just the timeline graph.

Everything else in film-grip projects the *timeline structure* (FGX rows: ids, frames,
durations). Perception is the layer that lets the planner see and hear the *content*:

* :mod:`~filmgrip.perception.transcribe` — word-level ASR with timestamps (the load-bearing
  data: "Claude greps the word timestamps, never scrubs"), cached per media file, packed to a
  token-frugal phrase format.
* :mod:`~filmgrip.perception.align` — maps media-time words into TIMELINE frames so an edit can
  be addressed by what was said ("cut after 'right'") with frame accuracy.
* :mod:`~filmgrip.perception.speech` — deterministic silence/filler analysis that turns a
  transcript into EditPlan-ready cut candidates.
* :mod:`~filmgrip.perception.frames` — ffmpeg contact sheets (filmstrip + waveform) the model
  reads multimodally at decision points instead of "watching" video.
* :mod:`~filmgrip.perception.verify` — the post-apply check: structural re-verification plus
  boundary stills around every changed cut.

All of it is pull-on-demand (MCP tools / CLI), never auto-bundled into FGX — perception is
opt-in tokens. Imports stay lazy and every hard dependency (ffmpeg, an ASR backend, media on
disk) degrades to an actionable error instead of a fake result.
"""
