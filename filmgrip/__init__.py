"""film-grip — react-grab for video editors.

Select clips/resources inside a video editor, add them to context with a natural-language
prompt, and Claude produces a typed, validated EditPlan that film-grip applies through the
right per-editor adapter.

Architecture: one universal OpenTimelineIO IR + a typed EditPlan protocol over stable clip
IDs + thin per-editor adapters + a token-frugal context serializer (FGX) + a Claude Agent
SDK / MCP integration. See ``filmgrip.core``, ``filmgrip.protocol``, ``filmgrip.adapters``,
``filmgrip.serialize`` and ``filmgrip.integration``.
"""

__version__ = "0.1.0"
