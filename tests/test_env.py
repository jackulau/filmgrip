"""D1 smoke test — proves the toolchain the rest of the plan depends on imports HERE.

This is intentionally strict: if these assertions fail, no later deliverable can be trusted,
so we fail loudly at the foundation rather than discovering a missing wheel in D9.
"""
from __future__ import annotations

import sys


def test_python_is_modern() -> None:
    assert sys.version_info >= (3, 10), f"need Python >=3.10, have {sys.version_info}"


def test_opentimelineio_imports_and_builds_a_timeline() -> None:
    import opentimelineio as otio

    tl = otio.schema.Timeline(name="smoke")
    tl.tracks.append(otio.schema.Track(name="V1"))
    assert tl.name == "smoke"
    assert len(tl.tracks) == 1


def test_pydantic_is_v2() -> None:
    import pydantic

    assert pydantic.VERSION.startswith("2."), f"need pydantic v2, have {pydantic.VERSION}"


def test_mcp_imports() -> None:
    # The MCP SDK is a base dependency — the Claude integration is the product's point.
    import mcp  # noqa: F401


def test_lxml_imports() -> None:
    from lxml import etree

    root = etree.fromstring("<mlt><tractor/></mlt>")
    assert root.tag == "mlt"


def test_filmgrip_package_version() -> None:
    import filmgrip

    assert filmgrip.__version__ == "0.1.0"
