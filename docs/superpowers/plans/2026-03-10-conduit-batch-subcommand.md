# conduit batch Subcommand Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `conduit batch` subcommand that accepts multiple prompts and runs them in parallel via `ConduitBatchSync`.

**Architecture:** A standalone Click command reads prompts from CLI args, `--file`, or stdin (merged), then delegates to a new `BatchHandlers.handle_batch()` static method that calls `ConduitBatchSync.create().run()` directly — bypassing the single-query `query_function` protocol. Output is formatted in one of three modes: pretty (default), raw (`--raw`), or JSON (`--json`).

**Tech Stack:** Click, `ConduitBatchSync` (already exists at `conduit.batch`), `Conversation.content`, `rich` via `Printer`, `click.testing.CliRunner` for tests.

---

## Chunk 1: Handler + command

### Task 1: BatchHandlers

**Files:**
- Create: `src/conduit/apps/cli/handlers/batch_handlers.py`
- Create: `tests/cli/test_batch_handlers.py`

#### Context

`ConduitBatchSync.create()` signature (from `conduit_batch_sync.py`):
```python
ConduitBatchSync.create(
    model: str,
    *,
    verbosity: Verbosity = settings.default_verbosity,
    **param_kwargs,          # temperature, etc. → forwarded to GenerationParams
) -> ConduitBatchSync
```

`batch.run(prompt_strings_list=[...], max_concurrent=N) -> list[Conversation]`

`Conversation.content` → the last message text as `str`.

Output modes:
- **pretty**: numbered header line per result, markdown body
- **raw**: response texts separated by `---\n`
- **json**: `click.echo(json.dumps([{"index": i, "prompt": p, "response": r}]))`

- [ ] **Step 1.1: Write failing tests**

```python
# tests/cli/test_batch_handlers.py
from __future__ import annotations
import json
from unittest.mock import MagicMock, patch
import pytest
from conduit.apps.cli.handlers.batch_handlers import BatchHandlers


def _make_mock_conversation(text: str) -> MagicMock:
    conv = MagicMock()
    conv.content = text
    return conv


def _make_printer() -> MagicMock:
    return MagicMock()


@pytest.fixture()
def mock_batch_run():
    """Patch ConduitBatchSync.create so no real LLM calls happen."""
    conversations = [
        _make_mock_conversation("Response A"),
        _make_mock_conversation("Response B"),
    ]
    mock_instance = MagicMock()
    mock_instance.run.return_value = conversations
    with patch(
        "conduit.apps.cli.handlers.batch_handlers.ConduitBatchSync"
    ) as mock_cls:
        mock_cls.create.return_value = mock_instance
        yield mock_cls, mock_instance


PROMPTS = ["Prompt one", "Prompt two"]


def test_handle_batch_calls_batch_sync(mock_batch_run):
    mock_cls, mock_instance = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=False,
        printer=printer,
    )
    mock_cls.create.assert_called_once()
    call_kwargs = mock_cls.create.call_args.kwargs
    assert call_kwargs["model"] == "gpt-4o"
    mock_instance.run.assert_called_once_with(
        prompt_strings_list=PROMPTS, max_concurrent=None
    )


def test_handle_batch_pretty_mode_prints_headers(mock_batch_run):
    _, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=False,
        printer=printer,
    )
    # Should call print_pretty for headers and print_markdown for bodies
    assert printer.print_pretty.call_count == 2
    assert printer.print_markdown.call_count == 2


def test_handle_batch_raw_mode(mock_batch_run, capsys):
    _, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=True,
        as_json=False,
        printer=printer,
    )
    captured = capsys.readouterr()
    assert "Response A" in captured.out
    assert "Response B" in captured.out
    assert "---" in captured.out
    # Pretty mode should NOT have been called
    printer.print_pretty.assert_not_called()


def test_handle_batch_json_mode(mock_batch_run, capsys):
    _, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=PROMPTS,
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=True,
        printer=printer,
    )
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert len(data) == 2
    assert data[0]["index"] == 0
    assert data[0]["prompt"] == "Prompt one"
    assert data[0]["response"] == "Response A"
    assert data[1]["index"] == 1
    assert data[1]["response"] == "Response B"


def test_handle_batch_temperature_forwarded(mock_batch_run):
    mock_cls, _ = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=["Q"],
        model="gpt-4o",
        temperature=0.3,
        local=False,
        citations=False,
        max_concurrent=None,
        raw=False,
        as_json=False,
        printer=printer,
    )
    call_kwargs = mock_cls.create.call_args.kwargs
    assert call_kwargs.get("temperature") == 0.3


def test_handle_batch_max_concurrent_forwarded(mock_batch_run):
    _, mock_instance = mock_batch_run
    printer = _make_printer()
    BatchHandlers.handle_batch(
        prompts=["Q"],
        model="gpt-4o",
        temperature=None,
        local=False,
        citations=False,
        max_concurrent=3,
        raw=False,
        as_json=False,
        printer=printer,
    )
    mock_instance.run.assert_called_once_with(
        prompt_strings_list=["Q"], max_concurrent=3
    )
```

- [ ] **Step 1.2: Run tests to confirm they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
uv run pytest tests/cli/test_batch_handlers.py -v 2>&1 | tail -20
```

Expected: `ImportError` or `ModuleNotFoundError` — `batch_handlers` doesn't exist yet.

- [ ] **Step 1.3: Implement BatchHandlers**

```python
# src/conduit/apps/cli/handlers/batch_handlers.py
from __future__ import annotations
import json
import click
import logging
from typing import TYPE_CHECKING

from conduit.batch import ConduitBatchSync

if TYPE_CHECKING:
    from conduit.apps.cli.utils.printer import Printer
    from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)

_HEADER_WIDTH = 60


class BatchHandlers:
    @staticmethod
    def handle_batch(
        prompts: list[str],
        model: str,
        temperature: float | None,
        local: bool,
        citations: bool,
        max_concurrent: int | None,
        raw: bool,
        as_json: bool,
        printer: Printer,
    ) -> None:
        """Run prompts in parallel and display results."""
        from conduit.config import settings

        param_kwargs: dict = {}
        if temperature is not None:
            param_kwargs["temperature"] = temperature

        batch = ConduitBatchSync.create(
            model=model,
            verbosity=settings.default_verbosity,
            **param_kwargs,
        )

        conversations = batch.run(
            prompt_strings_list=prompts,
            max_concurrent=max_concurrent,
        )

        results = [
            {"index": i, "prompt": p, "response": conv.content}
            for i, (p, conv) in enumerate(zip(prompts, conversations))
        ]

        if as_json:
            click.echo(json.dumps(results, ensure_ascii=False, indent=2))
            return

        if raw:
            for i, item in enumerate(results):
                click.echo(item["response"])
                if i < len(results) - 1:
                    click.echo("---")
            return

        # Pretty mode
        total = len(results)
        for item in results:
            idx = item["index"] + 1
            truncated = item["prompt"][:50].replace("\n", " ")
            if len(item["prompt"]) > 50:
                truncated += "..."
            header = f"[{idx}/{total}] {truncated}"
            printer.print_pretty(header, style="bold cyan")
            printer.print_markdown(item["response"])
```

- [ ] **Step 1.4: Run tests to confirm they pass**

```bash
uv run pytest tests/cli/test_batch_handlers.py -v 2>&1 | tail -20
```

Expected: all 6 tests PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/conduit/apps/cli/handlers/batch_handlers.py tests/cli/test_batch_handlers.py
git commit -m "feat: add BatchHandlers.handle_batch() with pretty/raw/json output modes"
```

---

### Task 2: batch_commands.py — Click command

**Files:**
- Create: `src/conduit/apps/cli/commands/batch_commands.py`
- Create: `tests/cli/test_batch_commands.py`

#### Input resolution rules

Prompts are collected from three sources and **merged** in this order:
1. `--file PATH` — read file, split on `\n`, strip blank lines
2. Inline `PROMPTS` arguments
3. stdin — read only when no args AND no `--file` AND stdin is not a TTY

If all sources are empty after merging, raise `click.UsageError`.

`--append TEXT` is concatenated to **every** prompt: `f"{prompt}\n{append}"`.

`--raw` and `--json` are mutually exclusive — raise `click.UsageError` if both are set.

- [ ] **Step 2.1: Write failing tests**

```python
# tests/cli/test_batch_commands.py
from __future__ import annotations
from unittest.mock import MagicMock, patch, call
import json
import pytest
from click.testing import CliRunner
from conduit.apps.cli.cli_class import ConduitCLI
from conduit.apps.cli.commands.batch_commands import batch_command


def _make_cli() -> object:
    """Build a minimal CLI group with only the batch command attached."""
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
        runner.invoke(cli, ["batch", "-m", "sonar-pro", "hello"])
    call_kwargs = mock_handler.call_args.kwargs
    assert call_kwargs["model"] == "sonar-pro"


def test_batch_raw_and_json_mutually_exclusive():
    cli = _make_cli()
    runner = CliRunner()
    result = runner.invoke(cli, ["batch", "--raw", "--json", "hello"])
    assert result.exit_code != 0
    assert "mutually exclusive" in result.output.lower() or "cannot" in result.output.lower()


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
        runner.invoke(cli, ["batch", "--append", "be concise", "question"])
    call_kwargs = mock_handler.call_args.kwargs
    assert all("be concise" in p for p in call_kwargs["prompts"])


def test_batch_max_concurrent_passed():
    cli = _make_cli()
    runner = CliRunner()
    with patch(PATCH_HANDLER) as mock_handler:
        runner.invoke(cli, ["batch", "-n", "4", "hello"])
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
```

- [ ] **Step 2.2: Run tests to confirm they fail**

```bash
uv run pytest tests/cli/test_batch_commands.py -v 2>&1 | tail -20
```

Expected: `ImportError` — `batch_commands` doesn't exist yet.

- [ ] **Step 2.3: Implement batch_commands.py**

```python
# src/conduit/apps/cli/commands/batch_commands.py
from __future__ import annotations
import sys
import click
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from conduit.apps.cli.handlers.batch_handlers import BatchHandlers

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@click.command("batch")
@click.option("-m", "--model", type=str, default=None, help="Model for all prompts.")
@click.option("-L", "--local", is_flag=True, help="Use local Ollama.")
@click.option("-t", "--temperature", type=float, default=None, help="Temperature (0.0-1.0).")
@click.option("-C", "--citations", is_flag=True, default=False, help="Print citations (Perplexity only).")
@click.option("-f", "--file", "prompt_file", type=click.Path(exists=True, dir_okay=False), default=None, help="File of prompts, one per line.")
@click.option("-n", "--max-concurrent", type=int, default=None, help="Max parallel requests.")
@click.option("-a", "--append", type=str, default=None, help="Suffix appended to every prompt.")
@click.option("-r", "--raw", is_flag=True, default=False, help="Plain text output, separated by ---.")
@click.option("--json", "as_json", is_flag=True, default=False, help="Output as JSON array.")
@click.argument("prompts", nargs=-1)
@click.pass_context
def batch_command(
    ctx: click.Context,
    model: str | None,
    local: bool,
    temperature: float | None,
    citations: bool,
    prompt_file: str | None,
    max_concurrent: int | None,
    append: str | None,
    raw: bool,
    as_json: bool,
    prompts: tuple[str, ...],
) -> None:
    """
    Run multiple prompts in parallel against an LLM.

    Prompts can be passed as arguments, read from --file (one per line),
    or piped via stdin. All sources are merged.

    Examples:
        conduit batch "What is X?" "What is Y?" -m sonar-pro
        conduit batch -f prompts.txt -m gpt-4o -n 5
        cat prompts.txt | conduit batch -m claude
    """
    if raw and as_json:
        raise click.UsageError("--raw and --json are mutually exclusive.")

    printer = ctx.obj["printer"]
    preferred_model = ctx.obj.get("preferred_model")
    resolved_model = model or preferred_model or "gpt-4o"

    # Collect prompts from all sources
    collected: list[str] = []

    # 1. File
    if prompt_file:
        text = Path(prompt_file).read_text(encoding="utf-8")
        collected.extend(line for line in text.splitlines() if line.strip())

    # 2. Inline args
    collected.extend(prompts)

    # 3. Stdin (only when no args and no file, and stdin is not a TTY)
    if not collected and not sys.stdin.isatty():
        stdin_text = sys.stdin.read()
        collected.extend(line for line in stdin_text.splitlines() if line.strip())

    if not collected:
        raise click.UsageError(
            "No prompts provided. Pass prompts as arguments, use --file, or pipe via stdin."
        )

    # Apply --append suffix to every prompt
    if append:
        collected = [f"{p}\n{append}" for p in collected]

    BatchHandlers.handle_batch(
        prompts=collected,
        model=resolved_model,
        temperature=temperature,
        local=local,
        citations=citations,
        max_concurrent=max_concurrent,
        raw=raw,
        as_json=as_json,
        printer=printer,
    )
```

- [ ] **Step 2.4: Run tests to confirm they pass**

```bash
uv run pytest tests/cli/test_batch_commands.py -v 2>&1 | tail -25
```

Expected: all 10 tests PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/conduit/apps/cli/commands/batch_commands.py tests/cli/test_batch_commands.py
git commit -m "feat: add batch_command Click command with arg/file/stdin input merging"
```

---

### Task 3: Wire batch_command into conduit_cli.py

**Files:**
- Modify: `src/conduit/apps/scripts/conduit_cli.py`

- [ ] **Step 3.1: Write a wiring test**

```python
# tests/cli/test_batch_wiring.py
from __future__ import annotations
from click.testing import CliRunner
from conduit.apps.scripts.conduit_cli import main
from unittest.mock import patch


def test_batch_subcommand_appears_in_help():
    """conduit --help should list 'batch' as a subcommand."""
    runner = CliRunner()
    # patch sys.argv so main() sees no extra args
    with patch("sys.argv", ["conduit"]):
        result = runner.invoke(main.__wrapped__ if hasattr(main, "__wrapped__") else _get_cli())
    # We invoke the CLI group directly
    from conduit.apps.scripts.conduit_cli import _build_cli
    cli = _build_cli()
    result = runner.invoke(cli, ["--help"])
    assert "batch" in result.output


def _build_cli():
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands
    from conduit.apps.cli.commands.cache_commands import cache
    from conduit.apps.cli.commands.models_commands import models_command
    from conduit.apps.cli.commands.batch_commands import batch_command
    cli_app = ConduitCLI()
    cli_app.attach(BaseCommands())
    cli_app.cli.add_command(cache)
    cli_app.cli.add_command(models_command)
    cli_app.cli.add_command(batch_command)
    return cli_app.cli
```

- [ ] **Step 3.2: Run wiring test to confirm it fails**

```bash
uv run pytest tests/cli/test_batch_wiring.py -v 2>&1 | tail -15
```

Expected: FAIL — `batch` not in help output.

- [ ] **Step 3.3: Edit conduit_cli.py to wire in batch_command**

Modify `src/conduit/apps/scripts/conduit_cli.py`:

```python
from conduit.apps.cli.commands.base_commands import BaseCommands
from conduit.apps.cli.commands.cache_commands import cache
from conduit.apps.cli.commands.models_commands import models_command
from conduit.apps.cli.commands.batch_commands import batch_command   # add this
from conduit.apps.cli.cli_class import ConduitCLI
import sys


def query_entrypoint():
    if len(sys.argv) > 1 and sys.argv[1] == "query":
        pass
    else:
        sys.argv.insert(1, "query")
    main()


def models_entrypoint():
    if len(sys.argv) > 1 and sys.argv[1] == "models":
        pass
    else:
        sys.argv.insert(1, "models")
    main()


def main():
    conduit_cli = ConduitCLI()
    commands = BaseCommands()
    conduit_cli.attach(commands)
    conduit_cli.cli.add_command(cache)
    conduit_cli.cli.add_command(models_command)
    conduit_cli.cli.add_command(batch_command)   # add this
    conduit_cli.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3.4: Run wiring test to confirm it passes**

```bash
uv run pytest tests/cli/test_batch_wiring.py -v 2>&1 | tail -15
```

Expected: PASS.

- [ ] **Step 3.5: Smoke-test the CLI end-to-end (no real LLM call)**

```bash
conduit batch --help
```

Expected: shows usage, flags, and examples.

- [ ] **Step 3.6: Run all CLI tests to check for regressions**

```bash
uv run pytest tests/cli/ -v 2>&1 | tail -30
```

Expected: all tests pass, no regressions.

- [ ] **Step 3.7: Commit**

```bash
git add src/conduit/apps/scripts/conduit_cli.py tests/cli/test_batch_wiring.py
git commit -m "feat: wire batch_command into conduit_cli, add wiring test"
```

---

## Done

All three commits together deliver:
- `conduit batch "p1" "p2" ... -m <model>` — parallel LLM calls
- `--file`, stdin, inline args, all merged
- `--append` suffix on every prompt
- `--raw` and `--json` output modes
- `-n/--max-concurrent` concurrency cap
- Full unit test coverage of handler + command + wiring
