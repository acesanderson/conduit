from __future__ import annotations

from click.testing import CliRunner
from unittest.mock import MagicMock, patch


def _make_cli():
    """Wire up a minimal conduit CLI for testing."""
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands

    conduit_cli = ConduitCLI()
    conduit_cli.attach(BaseCommands())
    return conduit_cli.cli


def test_image_with_chat_raises_usage_error(tmp_path):
    """AC2: --image + --chat raises UsageError."""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    runner = CliRunner()
    cli = _make_cli()

    result = runner.invoke(cli, ["query", "-i", str(img), "--chat", "describe"])
    assert result.exit_code != 0
    assert "--image cannot be used with --chat" in result.output


def test_image_with_search_raises_usage_error(tmp_path):
    """AC3: --image + --search raises UsageError."""
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    runner = CliRunner()
    cli = _make_cli()

    result = runner.invoke(cli, ["query", "-i", str(img), "--search", "describe"])
    assert result.exit_code != 0
    assert "--image cannot be used with --search" in result.output


def test_nonexistent_image_raises_bad_parameter():
    """AC4: nonexistent path rejected by Click before reaching handler."""
    runner = CliRunner()
    cli = _make_cli()

    result = runner.invoke(cli, ["query", "-i", "/nonexistent/path.png", "describe"])
    assert result.exit_code != 0
    assert "does not exist" in result.output or "Invalid value" in result.output
