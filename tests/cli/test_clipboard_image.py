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


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

from unittest.mock import MagicMock, patch
from click.testing import CliRunner


def _make_cli():
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands
    conduit_cli = ConduitCLI()
    conduit_cli.attach(BaseCommands())
    return conduit_cli.cli


# ---------------------------------------------------------------------------
# AC2: empty clipboard
# ---------------------------------------------------------------------------

def test_clipboard_empty_raises_usage_error():
    """AC2: grabclipboard() returns None → UsageError with exact message."""
    runner = CliRunner()
    cli = _make_cli()

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = None
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "describe"])

    assert result.exit_code != 0
    assert "--image @clipboard: clipboard is empty or contains no image." in result.output
    assert "Accessibility/Paste permissions" in result.output


# ---------------------------------------------------------------------------
# AC3: non-image clipboard (e.g. text string wrapped in list)
# ---------------------------------------------------------------------------

def test_clipboard_non_image_raises_usage_error():
    """AC3: grabclipboard() returns a non-Image type → UsageError with type name."""
    runner = CliRunner()
    cli = _make_cli()

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = ["some text"]
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "describe"])

    assert result.exit_code != 0
    assert "--image @clipboard: clipboard contains data but not an image" in result.output
    assert "list" in result.output


# ---------------------------------------------------------------------------
# AC4: file list in clipboard
# ---------------------------------------------------------------------------

def test_clipboard_file_list_raises_usage_error():
    """AC4: grabclipboard() returns a list of file paths → UsageError."""
    runner = CliRunner()
    cli = _make_cli()

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = ["/Users/user/photo.png"]
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "describe"])

    assert result.exit_code != 0
    assert "--image @clipboard: clipboard contains data but not an image" in result.output


# ---------------------------------------------------------------------------
# AC5: existing file path behavior unchanged
# ---------------------------------------------------------------------------

def test_file_path_passes_through_unchanged(tmp_path):
    """AC5: a valid file path is passed to the handler as image_path (not image_content)."""
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    runner = CliRunner()
    captured = {}

    def fake_query_fn(inputs):
        captured["inputs"] = inputs
        mock_conv = MagicMock()
        mock_conv.content = "ok"
        mock_conv.last = MagicMock()
        mock_conv.last.metadata = {}
        return mock_conv

    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands
    cli2 = ConduitCLI(query_function=fake_query_fn)
    cli2.attach(BaseCommands())
    result = runner.invoke(cli2.cli, ["query", "--image", str(img), "describe"])

    assert captured.get("inputs") is not None, result.output
    inputs = captured["inputs"]
    assert inputs.image_path == str(img)
    assert inputs.image_content is None


def test_nonexistent_file_raises_usage_error():
    """AC5 (file-not-found): manual validation emits our exact error message."""
    runner = CliRunner()
    cli = _make_cli()
    result = runner.invoke(cli, ["query", "--image", "/no/such/file.png", "describe"])
    assert result.exit_code != 0
    assert "--image: file not found:" in result.output


# ---------------------------------------------------------------------------
# AC6: --image @clipboard --chat rejected before clipboard access
# ---------------------------------------------------------------------------

def test_clipboard_with_chat_raises_usage_error():
    """AC6: --image @clipboard --chat is rejected before clipboard is accessed."""
    runner = CliRunner()
    cli = _make_cli()

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "--chat", "describe"])
        mock_grab.grabclipboard.assert_not_called()

    assert result.exit_code != 0
    assert "--image cannot be used with --chat" in result.output


# ---------------------------------------------------------------------------
# AC9: CMYK image converted to RGB without raising
# ---------------------------------------------------------------------------

def test_clipboard_cmyk_image_converted_to_rgb():
    """AC9: CMYK-mode PIL image is converted to RGB before PNG encoding; no exception raised."""
    from PIL import Image as PILImage

    runner = CliRunner()
    cmyk_image = PILImage.new("CMYK", (10, 10), color=(0, 0, 0, 0))
    captured_image_content = {}

    def fake_query_fn(inputs):
        captured_image_content["image_content"] = inputs.image_content
        mock_conv = MagicMock()
        mock_conv.content = "ok"
        mock_conv.last = MagicMock()
        mock_conv.last.metadata = {}
        return mock_conv

    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands
    cli2 = ConduitCLI(query_function=fake_query_fn)
    cli2.attach(BaseCommands())

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = cmyk_image
        result = runner.invoke(cli2.cli, ["query", "--image", "@clipboard", "describe"])

    assert result.exit_code == 0, result.output
    assert captured_image_content.get("image_content") is not None
    assert captured_image_content["image_content"].url.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# AC1: end-to-end success path
# ---------------------------------------------------------------------------

def test_clipboard_image_success_path():
    """AC1: valid clipboard image → ImageContent passed to query function, query succeeds."""
    from PIL import Image as PILImage
    from conduit.domain.message.message import ImageContent
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands

    runner = CliRunner()
    rgb_image = PILImage.new("RGB", (100, 100), color=(255, 0, 0))
    captured = {}

    def fake_query_fn(inputs):
        captured["inputs"] = inputs
        mock_conv = MagicMock()
        mock_conv.content = "It is a red square."
        mock_conv.last = MagicMock()
        mock_conv.last.metadata = {}
        return mock_conv

    cli = ConduitCLI(query_function=fake_query_fn)
    cli.attach(BaseCommands())

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = rgb_image
        result = runner.invoke(cli.cli, ["query", "--image", "@clipboard", "describe this"])

    assert result.exit_code == 0, result.output
    assert "inputs" in captured
    inputs = captured["inputs"]
    assert inputs.image_content is not None
    assert isinstance(inputs.image_content, ImageContent)
    assert inputs.image_content.url.startswith("data:image/png;base64,")
    assert inputs.image_path is None
