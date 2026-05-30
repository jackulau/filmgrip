"""Audio + sound-effects support for film-grip.

Users point film-grip at THEIR own folder of sound effects (``~/.filmgrip/sfx`` by default, or
``$FILMGRIP_SFX_DIR``/``--dir``). The :mod:`~filmgrip.audio.library` module turns that folder into a
small, token-frugal manifest of named effects that Claude can choose from by description — "add a
whoosh" resolves to a real file path, which the Resolve adapter imports and places on an audio track.
"""
from .library import SfxEntry, SfxLibrary

__all__ = ["SfxEntry", "SfxLibrary"]
