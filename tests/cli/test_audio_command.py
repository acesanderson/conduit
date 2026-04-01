from __future__ import annotations

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch


def _make_cli():
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands

    conduit_cli = ConduitCLI()
    conduit_cli.attach(BaseCommands())
    return conduit_cli.cli


def test_audio_with_chat_raises_usage_error(tmp_path):
    """AC4: --audio + --chat raises UsageError."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 16)

    runner = CliRunner()
    result = runner.invoke(_make_cli(), ["query", "--audio", str(audio), "--chat", "summarize"])
    assert result.exit_code != 0
    assert "--audio cannot be used with --chat" in result.output


def test_audio_with_search_raises_usage_error(tmp_path):
    """AC5: --audio + --search raises UsageError."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 16)

    runner = CliRunner()
    result = runner.invoke(_make_cli(), ["query", "--audio", str(audio), "--search", "summarize"])
    assert result.exit_code != 0
    assert "--audio cannot be used with --search" in result.output


def test_audio_with_image_raises_usage_error(tmp_path):
    """AC6: --audio + --image raises UsageError."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 16)
    img = tmp_path / "img.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    runner = CliRunner()
    result = runner.invoke(
        _make_cli(),
        ["query", "--audio", str(audio), "--image", str(img), "describe"],
    )
    assert result.exit_code != 0
    assert "--audio cannot be used with --image" in result.output


def test_nonexistent_audio_raises_usage_error():
    """AC7: nonexistent audio path raises UsageError."""
    runner = CliRunner()
    result = runner.invoke(_make_cli(), ["query", "--audio", "/nonexistent/clip.mp3", "q"])
    assert result.exit_code != 0
    assert "--audio: file not found:" in result.output


def test_unsupported_audio_format_raises_usage_error(tmp_path):
    """AC8: .m4a extension raises UsageError with 'unsupported audio format' message."""
    audio = tmp_path / "clip.m4a"
    audio.write_bytes(b"\x00" * 16)

    runner = CliRunner()
    result = runner.invoke(_make_cli(), ["query", "--audio", str(audio), "q"])
    assert result.exit_code != 0
    assert "unsupported audio format" in result.output


def test_audio_with_no_query_defaults_to_transcribe_this(tmp_path):
    """AC12: --audio with no positional args uses 'transcribe this' as the query."""
    audio = tmp_path / "clip.mp3"
    audio.write_bytes(b"\xff\xfb" + b"\x00" * 16)

    captured = {}

    def fake_handle_query(**kwargs):
        captured.update(kwargs)

    runner = CliRunner()
    with patch(
        "conduit.apps.cli.commands.base_commands.handlers.handle_query",
        side_effect=fake_handle_query,
    ):
        runner.invoke(_make_cli(), ["query", "--audio", str(audio)])

    assert captured.get("query_input") == "transcribe this"
