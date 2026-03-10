from __future__ import annotations

import base64

from conduit.domain.message.message import ImageContent


def test_from_bytes_produces_data_url():
    """AC8: from_bytes encodes bytes as a base64 data URL with correct MIME prefix."""
    data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
    content = ImageContent.from_bytes(data, "image/png")
    assert content.url.startswith("data:image/png;base64,")
    decoded = base64.b64decode(content.url.split(",", 1)[1])
    assert decoded == data


def test_from_bytes_default_mime_is_png():
    """AC8 (default): omitting mime_type defaults to image/png."""
    content = ImageContent.from_bytes(b"\x00\x01\x02")
    assert content.url.startswith("data:image/png;base64,")
