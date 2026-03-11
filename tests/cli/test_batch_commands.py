from __future__ import annotations
from unittest.mock import patch
import pytest
from click.testing import CliRunner
from conduit.apps.cli.cli_class import ConduitCLI
from conduit.apps.cli.commands.batch_commands import batch_command


def _make_cli():
    cli_app = ConduitCLI()
    cli_app.cli.add_command(batch_command)
    return cli_app.cli


PATCH_HANDLER = "conduit.apps.cli.commands.batch_commands.BatchHandlers.handle_batch"


def test_batch_help_shows_flags():
    cli = _make_cli()
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", "--help"])
    assert result.exit_code == 0
    for flag in ["--model", "--raw", "--json", "--file", "--max-concurrent", "--append"]:
        assert flag in result.output


def test_batch_inline_prompts_passed_to_handler():
    cli = _make_cli()
    runner = CliRunner()
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch", "prompt one", "prompt two"])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert call_kwargs["prompts"] == ["prompt one", "prompt two"]


def test_batch_model_flag():
    cli = _make_cli()
    runner = CliRunner()
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch", "-m", "sonar-pro", "hello"])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert call_kwargs["model"] == "sonar-pro"


def test_batch_raw_and_json_mutually_exclusive():
    cli = _make_cli()
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", "--raw", "--json", "hello"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or "raw" in result.output.lower()


def test_batch_no_prompts_raises_error():
    cli = _make_cli()
    runner = CliRunner()
    result = runner.invoke(cli, ["batch"])
    assert result.exit_code != 0


def test_batch_file_input(tmp_path):
    cli = _make_cli()
    runner = CliRunner()
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("line one\nline two\n\n")
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch", "-f", str(prompt_file)])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert call_kwargs["prompts"] == ["line one", "line two"]


def test_batch_file_and_inline_merged(tmp_path):
    cli = _make_cli()
    runner = CliRunner()
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("from file\n")
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch", "-f", str(prompt_file), "inline one"])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert "from file" in call_kwargs["prompts"]
    assert "inline one" in call_kwargs["prompts"]


def test_batch_append_suffix_applied():
    cli = _make_cli()
    runner = CliRunner()
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch", "--append", "be concise", "question"])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert all("be concise" in p for p in call_kwargs["prompts"])


def test_batch_max_concurrent_passed():
    cli = _make_cli()
    runner = CliRunner()
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch", "-n", "4", "hello"])
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert call_kwargs["max_concurrent"] == 4


def test_batch_stdin_input():
    cli = _make_cli()
    runner = CliRunner()
    with patch(PATCH_HANDLER) as mock_handler:
        result = runner.invoke(cli, ["batch"], input="stdin prompt one\nstdin prompt two\n")
    assert result.exit_code == 0, result.output
    call_kwargs = mock_handler.call_args.kwargs
    assert "stdin prompt one" in call_kwargs["prompts"]
    assert "stdin prompt two" in call_kwargs["prompts"]
