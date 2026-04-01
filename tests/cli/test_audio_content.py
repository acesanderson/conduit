from __future__ import annotations

import base64
import pytest
from conduit.domain.message.message import AudioContent


def test_from_bytes_default_format_is_mp3():
    """AC9: AudioContent.from_bytes(data) defaults format to 'mp3'."""
    data = b"fake audio bytes"
    result = AudioContent.from_bytes(data)
    assert result.format == "mp3"


def test_from_bytes_wav_format():
    """AC10: AudioContent.from_bytes(data, format='wav') sets format to 'wav'."""
    data = b"fake audio bytes"
    result = AudioContent.from_bytes(data, format="wav")
    assert result.format == "wav"


def test_from_bytes_encodes_as_base64():
    """from_bytes stores data as base64-encoded string."""
    data = b"hello audio"
    result = AudioContent.from_bytes(data)
    assert result.data == base64.b64encode(data).decode("utf-8")
