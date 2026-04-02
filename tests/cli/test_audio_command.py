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


import base64
from pathlib import Path
from unittest.mock import MagicMock, patch

from conduit.domain.message.message import AudioOutput, AssistantMessage


def _make_conversation_with_audio(fmt: str = "mp3") -> MagicMock:
    """Return a mock Conversation whose last message has AudioOutput."""
    audio_data = base64.b64encode(b"fake audio bytes").decode()
    audio_out = AudioOutput(id="x", data=audio_data, format=fmt)
    assistant = AssistantMessage(audio=audio_out)
    conv = MagicMock()
    conv.last = assistant
    conv.content = None
    return conv


def _make_conversation_with_text(text: str = "hello world") -> MagicMock:
    """Return a mock Conversation whose last message has text content."""
    assistant = AssistantMessage(content=text)
    conv = MagicMock()
    conv.last = assistant
    conv.content = text
    return conv


def test_save_response_writes_audio_bytes(tmp_path):
    """AC14: _save_response writes decoded audio bytes to file when last.audio is set."""
    from conduit.apps.cli.handlers.base_handlers import _save_response

    response = _make_conversation_with_audio()
    out = tmp_path / "out.mp3"
    _save_response(response, str(out))

    assert out.exists()
    assert out.read_bytes() == b"fake audio bytes"


def test_save_response_writes_text(tmp_path):
    """AC15: _save_response writes str(response.last) to file when no audio/images."""
    from conduit.apps.cli.handlers.base_handlers import _save_response

    response = _make_conversation_with_text("the answer is 42")
    out = tmp_path / "out.txt"
    _save_response(response, str(out))

    assert out.exists()
    assert out.read_text() == "the answer is 42"


def test_save_flag_suppresses_printer(tmp_path):
    """AC13: when --save is set, printer.print_markdown and printer.print_raw are not called."""
    from conduit.apps.cli.handlers.base_handlers import BaseHandlers

    save_path = str(tmp_path / "out.txt")

    mock_printer = MagicMock()
    mock_qf = MagicMock(return_value=_make_conversation_with_text("hi"))

    BaseHandlers.handle_query(
        query_input="hello",
        model="gpt-4o",
        local=False,
        raw=False,
        temperature=None,
        chat=False,
        append=None,
        verbosity=MagicMock(),
        printer=mock_printer,
        query_function=mock_qf,
        stdin=None,
        save=save_path,
    )

    mock_printer.print_markdown.assert_not_called()
    mock_printer.print_raw.assert_not_called()


def test_play_flag_autosaves_to_tmp():
    """AC16: when --play is set and --save is not, effective_save is under /tmp/ with correct extension."""
    from conduit.apps.cli.handlers.base_handlers import BaseHandlers

    response = _make_conversation_with_audio(fmt="mp3")
    mock_qf = MagicMock(return_value=response)
    mock_printer = MagicMock()

    saved_paths: list[str] = []

    def fake_save(resp, path):
        saved_paths.append(path)

    with patch("conduit.apps.cli.handlers.base_handlers._save_response", side_effect=fake_save), \
         patch("conduit.apps.cli.handlers.base_handlers._play_audio"):
        BaseHandlers.handle_query(
            query_input="say hello",
            model="tts",
            local=False,
            raw=False,
            temperature=None,
            chat=False,
            append=None,
            verbosity=MagicMock(),
            printer=mock_printer,
            query_function=mock_qf,
            stdin=None,
            play=True,
        )

    assert len(saved_paths) == 1
    assert saved_paths[0].startswith("/tmp/")
    assert saved_paths[0].endswith(".mp3")
