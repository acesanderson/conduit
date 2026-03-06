# Citations Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a `--citations` / `-C` flag to `conduit query` that passes `return_citations: true` to the Perplexity API and prints citations after the response, with a non-fatal stderr warning on non-Perplexity providers.

**Architecture:** The change is purely in the CLI layer — no client, domain, or storage code is touched. `client_params` is threaded from the new `CLIQueryFunctionInputs.client_params` field down to `ConduitSync.create()` via its existing `**param_kwargs` → `GenerationParams.client_params` path. Citation display and provider-mismatch warnings live in a new `BaseHandlers.handle_citations()` static method; all warning output goes to stderr, all data output goes to stdout.

**Tech Stack:** Python 3.12, Click, Rich, Pydantic v2, pytest, unittest.mock

---

## Pre-flight reading (do this before Task 1)

Before writing any code, read these files to orient yourself. The plan references exact line numbers and signatures.

| File | Why |
|---|---|
| `src/conduit/apps/cli/commands/base_commands.py` | Where the `query` Click command lives |
| `src/conduit/apps/cli/handlers/base_handlers.py` | Where `handle_query` and new `handle_citations` live |
| `src/conduit/apps/cli/query/query_function.py` | `CLIQueryFunctionInputs` dataclass + `default_query_function` |
| `src/conduit/apps/cli/utils/printer.py` | `Printer` class — full implementation |
| `src/conduit/domain/request/generation_params.py` | Confirms `client_params: dict | None = None` already exists |
| `src/conduit/core/conduit/conduit_sync.py` | Confirms `create(**param_kwargs)` passes kwargs to `GenerationParams` |
| `src/conduit/domain/conversation/conversation.py` | **IMPORTANT**: verify the actual runtime return type of `default_query_function`. The function is annotated as returning `GenerationResponse` but `conduit()` returns `Conversation`. Determine whether `Conversation` has a `.content` property and how to access message metadata (citations) from it. |
| `tests/conftest.py` | Autouse fixtures available in all tests |
| `tests/factories.py` | Factory classes for test data |

**Critical invariant before proceeding:** confirm that `response.message.metadata` (where `response` is the object returned by `query_function(inputs)`) gives you the dict containing `"citations"` and `"provider"` keys. If `response` is a `Conversation`, you may need `response.messages[-1].metadata` or similar. Do not guess — read the code.

---

### Task 1: Add `client_params` to `CLIQueryFunctionInputs` and wire through `default_query_function`

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Test: `tests/cli/test_query_function.py` (create)

The goal is that when `citations=True` is set on the CLI, `{"return_citations": True}` reaches `GenerationParams.client_params`. The path is:

`CLIQueryFunctionInputs.client_params` → `default_query_function` → `ConduitSync.create(client_params=...)` → `GenerationParams(client_params=...)`

This works because `ConduitSync.create()` accepts `**param_kwargs` and passes them directly to `GenerationParams(model=model, **param_kwargs, system=system)`, and `GenerationParams.client_params: dict | None = None` already exists.

**Step 1: Create the test file and write failing tests**

Create `tests/cli/__init__.py` (empty file) and `tests/cli/test_query_function.py`:

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch
from conduit.apps.cli.query.query_function import CLIQueryFunctionInputs, default_query_function


def make_inputs(**overrides) -> CLIQueryFunctionInputs:
    defaults = dict(
        project_name="test",
        query_input="hello",
        printer=MagicMock(),
        client_params={},
    )
    defaults.update(overrides)
    return CLIQueryFunctionInputs(**defaults)


def test_inputs_has_client_params_field():
    """CLIQueryFunctionInputs accepts client_params kwarg."""
    inputs = make_inputs(client_params={"return_citations": True})
    assert inputs.client_params == {"return_citations": True}


def test_inputs_client_params_defaults_to_empty_dict():
    """client_params defaults to empty dict, not None."""
    inputs = make_inputs()
    assert inputs.client_params == {}


def test_default_query_function_passes_client_params_to_conduit():
    """default_query_function passes client_params to ConduitSync.create()."""
    inputs = make_inputs(client_params={"return_citations": True})

    mock_response = MagicMock()
    mock_conduit = MagicMock(return_value=mock_response)

    with patch(
        "conduit.apps.cli.query.query_function.ConduitSync.create",
        return_value=mock_conduit,
    ) as mock_create:
        default_query_function(inputs)

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs.get("client_params") == {"return_citations": True}


def test_default_query_function_omits_client_params_when_empty():
    """When client_params is empty, None is passed to avoid polluting GenerationParams."""
    inputs = make_inputs(client_params={})

    mock_response = MagicMock()
    mock_conduit = MagicMock(return_value=mock_response)

    with patch(
        "conduit.apps.cli.query.query_function.ConduitSync.create",
        return_value=mock_conduit,
    ) as mock_create:
        default_query_function(inputs)

    call_kwargs = mock_create.call_args.kwargs
    # Empty dict should be normalized to None (no-op for GenerationParams)
    assert call_kwargs.get("client_params") is None
```

**Step 2: Run tests to confirm they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_query_function.py -v
```

Expected: `AttributeError` or `TypeError` — `CLIQueryFunctionInputs` doesn't have `client_params` yet.

**Step 3: Add `client_params` to `CLIQueryFunctionInputs`**

In `src/conduit/apps/cli/query/query_function.py`, `CLIQueryFunctionInputs` is a `@dataclass`. Add the field after the other optional fields with defaults:

```python
from dataclasses import dataclass, field

@dataclass
class CLIQueryFunctionInputs:
    # ... existing fields ...
    client_params: dict = field(default_factory=dict)
```

Note: `@dataclass` with `field(default_factory=dict)` must come after any fields without defaults. Check the existing field ordering before inserting.

**Step 4: Pass `client_params` through in `default_query_function`**

In `default_query_function`, the `ConduitSync.create(...)` call currently does not pass `client_params`. Add it:

```python
client_params = inputs.client_params or None  # normalize empty dict → None

conduit = ConduitSync.create(
    project_name=project_name,
    model=preferred_model,
    prompt=prompt,
    system=system,
    cache=cache,
    persist=not ephemeral,
    verbose=verbose,
    debug_payload=False,
    include_history=include_history,
    use_remote=local,
    client_params=client_params,   # <-- add this line
)
```

**Step 5: Run tests to confirm they pass**

```bash
pytest tests/cli/test_query_function.py -v
```

Expected: all 4 tests PASS.

**Step 6: Commit**

```bash
git add tests/cli/__init__.py tests/cli/test_query_function.py src/conduit/apps/cli/query/query_function.py
git commit -m "feat: add client_params to CLIQueryFunctionInputs and wire through default_query_function"
```

---

### Task 2: Add `print_err` and `print_citations` to `Printer`

**Files:**
- Modify: `src/conduit/apps/cli/utils/printer.py`
- Test: `tests/cli/test_printer.py` (create)

**Background on `Printer`'s IO model:**

The existing `Printer` routes output based on TTY status:
- `emit_ui = IS_TTY and (not raw)` — rich stderr console, used by `print_pretty`
- `emit_data = (not IS_TTY) or raw` — stdout writes, used by `print_raw` and `print_markdown`

`print_pretty` already goes to stderr, but only when `emit_ui=True` (TTY, non-raw). In a pipe or `--raw` context, `print_pretty` is silent. We need:

- `print_err(message)`: **always** prints to stderr regardless of TTY/raw. Used for warnings (provider mismatch, empty citations) that must reach the user in all contexts.
- `print_citations(citations)`: prints formatted citation list, following the same routing as `print_markdown` (i.e., rich stderr in TTY mode, nothing in data mode — the raw JSON path is handled by `handle_citations` directly via `click.echo`, not by this method).

**Step 1: Write failing tests**

Create `tests/cli/test_printer.py`:

```python
from __future__ import annotations
import sys
import json
from io import StringIO
from unittest.mock import patch, MagicMock
from conduit.apps.cli.utils.printer import Printer


def make_printer(raw: bool = False, is_tty: bool = True) -> Printer:
    with patch("conduit.apps.cli.utils.printer.IS_TTY", is_tty):
        return Printer(raw=raw)


def test_print_err_goes_to_stderr_in_tty():
    """print_err writes to stderr in TTY mode."""
    printer = make_printer(is_tty=True)
    stderr_capture = StringIO()
    with patch.object(printer, "_err_console") as mock_console:
        printer.print_err("[red]warning[/red]")
        mock_console.print.assert_called_once_with("[red]warning[/red]")


def test_print_err_goes_to_stderr_in_pipe():
    """print_err writes to stderr even when piped (not TTY)."""
    printer = make_printer(is_tty=False)
    with patch.object(printer, "_err_console") as mock_console:
        printer.print_err("[red]warning[/red]")
        mock_console.print.assert_called_once_with("[red]warning[/red]")


def test_print_citations_renders_numbered_list_in_tty():
    """print_citations renders a numbered list via print_pretty in TTY mode."""
    printer = make_printer(is_tty=True, raw=False)
    citations = [
        {"title": "Article One", "url": "https://one.com", "source": "", "date": ""},
        {"title": "Article Two", "url": "https://two.com", "source": "", "date": ""},
    ]
    with patch.object(printer, "print_pretty") as mock_pp:
        printer.print_citations(citations)
    calls = [str(c) for c in mock_pp.call_args_list]
    # Verify a numbered entry appears in the calls
    assert any("1." in c and "Article One" in c and "https://one.com" in c for c in calls)
    assert any("2." in c and "Article Two" in c and "https://two.com" in c for c in calls)


def test_print_citations_handles_missing_keys_gracefully():
    """print_citations doesn't crash on citations with missing title or url."""
    printer = make_printer(is_tty=True, raw=False)
    citations = [
        {"title": "", "url": "", "source": "", "date": ""},   # both empty — should be skipped
        {"title": "Only Title", "url": "", "source": "", "date": ""},
        {"url": "https://only-url.com"},                       # no title key at all
    ]
    # Should not raise
    with patch.object(printer, "print_pretty"):
        printer.print_citations(citations)


def test_print_citations_silent_in_pipe_mode():
    """print_citations does nothing in pipe mode (data mode handles JSON separately)."""
    printer = make_printer(is_tty=False, raw=False)
    with patch.object(printer, "print_pretty") as mock_pp:
        printer.print_citations([{"title": "X", "url": "y"}])
    mock_pp.assert_not_called()
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/cli/test_printer.py -v
```

Expected: `AttributeError` — `print_err` and `print_citations` don't exist yet.

**Step 3: Implement `print_err` and `print_citations`**

In `src/conduit/apps/cli/utils/printer.py`, add an `_err_console` attribute and two methods:

```python
from rich.console import Console
import sys

class Printer:
    def __init__(self, raw: bool = False):
        # ... existing init ...
        # Always-on stderr console for warnings (independent of TTY/raw state)
        self._err_console: Console = Console(file=sys.stderr)

    def print_err(self, message: str) -> None:
        """
        Print rich-formatted message to stderr unconditionally.
        Used for warnings that must reach the user regardless of TTY or --raw state.
        """
        self._err_console.print(message)

    def print_citations(self, citations: list[dict]) -> None:
        """
        Print formatted citations list via print_pretty (stderr in TTY, silent in pipe).
        The raw-JSON path is handled by handle_citations directly, not here.
        """
        if not self.emit_ui:
            return
        self.print_pretty("")
        self.print_pretty("[bold]Sources[/bold]")
        index = 1
        for c in citations:
            title = c.get("title") or ""
            url = c.get("url") or ""
            if not title and not url:
                continue
            display = title if title else url
            line = f"  {index}. {display}"
            if title and url:
                line += f" — {url}"
            self.print_pretty(line)
            index += 1
```

**Step 4: Run tests to confirm they pass**

```bash
pytest tests/cli/test_printer.py -v
```

Expected: all tests PASS.

**Step 5: Commit**

```bash
git add src/conduit/apps/cli/utils/printer.py tests/cli/test_printer.py
git commit -m "feat: add print_err and print_citations to Printer"
```

---

### Task 3: Add `handle_citations` to `BaseHandlers`

**Files:**
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py`
- Test: `tests/cli/test_handle_citations.py` (create)

**Step 1: Understand the return type of `query_function`**

Before writing this task, re-read `src/conduit/domain/conversation/conversation.py` to determine how to access the last assistant message's metadata from the object returned by `query_function(inputs)`. The function is annotated as `-> GenerationResponse` but the actual runtime object may be a `Conversation`. Confirm the attribute path to `metadata["citations"]` and `metadata["provider"]`.

The tests below assume the return object has `response.message.metadata` (i.e., `GenerationResponse`). If it's actually a `Conversation`, adjust the attribute access in both the tests and implementation accordingly — check `conversation.messages[-1].metadata` or similar.

**Step 2: Write failing tests**

Create `tests/cli/test_handle_citations.py`:

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch
import json
import pytest
from conduit.apps.cli.handlers.base_handlers import BaseHandlers


def make_response(provider: str | None, citations: list[dict]) -> MagicMock:
    """Build a mock response object matching GenerationResponse shape."""
    metadata = {}
    if provider is not None:
        metadata["provider"] = provider
    metadata["citations"] = citations

    msg = MagicMock()
    msg.metadata = metadata

    response = MagicMock()
    response.message = msg
    return response


def make_printer() -> MagicMock:
    return MagicMock()


def test_handle_citations_noop_when_flag_false():
    """handle_citations does nothing when citations=False."""
    printer = make_printer()
    response = make_response("perplexity", [{"title": "T", "url": "U"}])
    BaseHandlers.handle_citations(response, citations=False, raw=False, printer=printer)
    printer.print_err.assert_not_called()
    printer.print_citations.assert_not_called()


def test_handle_citations_warns_when_non_perplexity_provider(caplog):
    """Prints red stderr warning when provider != 'perplexity'."""
    import logging
    printer = make_printer()
    response = make_response("openai", [])

    with caplog.at_level(logging.WARNING, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[red]" in call_arg
    printer.print_citations.assert_not_called()
    assert "openai" in caplog.text.lower() or "perplexity" in caplog.text.lower()


def test_handle_citations_warns_when_provider_missing(caplog):
    """Prints red stderr warning when provider key is absent."""
    import logging
    printer = make_printer()
    response = make_response(None, [])  # no provider key set

    with caplog.at_level(logging.WARNING, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[red]" in call_arg


def test_handle_citations_warns_yellow_when_citations_empty(caplog):
    """Prints yellow stderr warning when perplexity returns empty citations."""
    import logging
    printer = make_printer()
    response = make_response("perplexity", [])

    with caplog.at_level(logging.DEBUG, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_err.assert_called_once()
    call_arg = printer.print_err.call_args[0][0]
    assert "[yellow]" in call_arg
    printer.print_citations.assert_not_called()
    assert "citation" in caplog.text.lower()


def test_handle_citations_prints_formatted_when_raw_false(caplog):
    """Calls printer.print_citations in non-raw mode."""
    import logging
    printer = make_printer()
    citations = [{"title": "X", "url": "https://x.com", "source": "", "date": ""}]
    response = make_response("perplexity", citations)

    with caplog.at_level(logging.DEBUG, logger="conduit.apps.cli.handlers.base_handlers"):
        BaseHandlers.handle_citations(response, citations=True, raw=False, printer=printer)

    printer.print_citations.assert_called_once_with(citations)
    printer.print_err.assert_not_called()
    assert "1" in caplog.text  # count logged


def test_handle_citations_prints_json_when_raw_true(caplog):
    """click.echo(json.dumps(citations)) called in raw mode, not print_citations."""
    import logging
    printer = make_printer()
    citations = [{"title": "X", "url": "https://x.com", "source": "", "date": ""}]
    response = make_response("perplexity", citations)

    with caplog.at_level(logging.DEBUG, logger="conduit.apps.cli.handlers.base_handlers"):
        with patch("conduit.apps.cli.handlers.base_handlers.click.echo") as mock_echo:
            BaseHandlers.handle_citations(response, citations=True, raw=True, printer=printer)

    mock_echo.assert_called_once()
    echoed = mock_echo.call_args[0][0]
    parsed = json.loads(echoed)
    assert parsed == citations
    printer.print_citations.assert_not_called()
```

**Step 3: Run tests to confirm they fail**

```bash
pytest tests/cli/test_handle_citations.py -v
```

Expected: `AttributeError` — `handle_citations` doesn't exist yet.

**Step 4: Implement `handle_citations`**

In `src/conduit/apps/cli/handlers/base_handlers.py`, add `import click` and `import json` at the top (check if already present), then add the static method to `BaseHandlers`:

```python
@staticmethod
def handle_citations(
    response: object,
    citations: bool,
    raw: bool,
    printer: Printer,
) -> None:
    """
    Print citations after a query response.
    All warnings go to stderr. Raw JSON goes to stdout. Formatted list goes to printer.
    """
    if not citations:
        return

    # Provider check — only Perplexity populates citations
    metadata: dict = getattr(getattr(response, "message", None), "metadata", {}) or {}
    provider: str | None = metadata.get("provider")

    if provider != "perplexity":
        logger.warning(
            "--citations requested but provider is %r, not 'perplexity'", provider
        )
        printer.print_err(
            "[red]--citations is only supported for Perplexity models "
            "(sonar, sonar-pro). No citations available.[/red]"
        )
        return

    citations_list: list[dict] = metadata.get("citations", [])

    if not citations_list:
        logger.debug("--citations: no citations returned by model")
        printer.print_err(
            "[yellow]No citations were returned by the model.[/yellow]"
        )
        return

    logger.debug("--citations: printing %d citations", len(citations_list))

    if raw:
        import json
        import click
        click.echo(json.dumps(citations_list))
    else:
        printer.print_citations(citations_list)
```

**Step 5: Run tests to confirm they pass**

```bash
pytest tests/cli/test_handle_citations.py -v
```

Expected: all tests PASS.

**Step 6: Commit**

```bash
git add src/conduit/apps/cli/handlers/base_handlers.py tests/cli/test_handle_citations.py
git commit -m "feat: add handle_citations static method to BaseHandlers"
```

---

### Task 4: Wire `--citations` flag through command layer and `handle_query`

**Files:**
- Modify: `src/conduit/apps/cli/commands/base_commands.py`
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py` (edit `handle_query` signature and body)
- Test: `tests/cli/test_citations_e2e.py` (create)

This task connects the Click flag to the full pipeline: flag → `query` command → `handle_query` → `handle_citations`.

**Step 1: Write failing tests**

Create `tests/cli/test_citations_e2e.py`:

```python
from __future__ import annotations
from unittest.mock import MagicMock, patch, call
from click.testing import CliRunner
from conduit.apps.cli.cli_class import ConduitCLI
from conduit.apps.cli.commands.base_commands import BaseCommands


def make_cli():
    """Build a test CLI instance with a mock query function."""
    mock_qf = MagicMock()
    # Simulate a Perplexity response with citations
    mock_response = MagicMock()
    mock_response.content = "The answer."
    mock_response.message.metadata = {
        "provider": "perplexity",
        "citations": [{"title": "Src", "url": "https://src.com", "source": "", "date": ""}],
    }
    mock_qf.return_value = mock_response
    cli_app = ConduitCLI(query_function=mock_qf)
    commands = BaseCommands()
    cli_app.attach(commands)
    return cli_app.cli, mock_qf


def test_citations_flag_appears_in_help():
    """--citations and -C appear in `conduit query --help`."""
    cli, _ = make_cli()
    runner = CliRunner()
    result = runner.invoke(cli, ["query", "--help"])
    assert "-C" in result.output or "--citations" in result.output


def test_citations_flag_sets_client_params():
    """-C flag causes client_params={"return_citations": True} in query function inputs."""
    cli, mock_qf = make_cli()
    runner = CliRunner()
    runner.invoke(cli, ["query", "--citations", "what is rust"])
    assert mock_qf.called
    inputs = mock_qf.call_args[0][0]
    assert inputs.client_params == {"return_citations": True}


def test_no_citations_flag_leaves_client_params_empty():
    """Without --citations, client_params is empty."""
    cli, mock_qf = make_cli()
    runner = CliRunner()
    runner.invoke(cli, ["query", "what is rust"])
    inputs = mock_qf.call_args[0][0]
    assert inputs.client_params == {}


def test_citations_flag_short_form():
    """-C short form works identically to --citations."""
    cli, mock_qf = make_cli()
    runner = CliRunner()
    runner.invoke(cli, ["query", "-C", "what is rust"])
    inputs = mock_qf.call_args[0][0]
    assert inputs.client_params == {"return_citations": True}


def test_citations_non_perplexity_prints_red_warning():
    """Non-Perplexity response triggers red warning, no exception, exit code 0."""
    mock_qf = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "answer"
    mock_response.message.metadata = {"provider": "openai", "citations": []}
    mock_qf.return_value = mock_response

    cli_app = ConduitCLI(query_function=mock_qf)
    cli_app.attach(BaseCommands())
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli_app.cli, ["query", "--citations", "hello"])

    assert result.exit_code == 0
    # Warning should be in stderr
    assert "perplexity" in result.stderr.lower() or "citations" in result.stderr.lower()


def test_no_citations_flag_means_no_citations_output():
    """Without --citations, no citations are printed even if metadata has them."""
    cli, mock_qf = make_cli()
    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli, ["query", "what is rust"])
    # Citations block should not appear
    assert "Sources" not in result.output
    assert "Sources" not in result.stderr
```

**Step 2: Run tests to confirm they fail**

```bash
pytest tests/cli/test_citations_e2e.py -v
```

Expected: failures on `--citations` flag not existing yet.

**Step 3: Add `--citations` to the `query` Click command**

In `src/conduit/apps/cli/commands/base_commands.py`, in the `query` command's decorator chain (after the existing `@click.option` decorators), add:

```python
@click.option("-C", "--citations", is_flag=True, default=False, help="Print citations (Perplexity models only).")
```

Add `citations: bool` to the `query` function signature and pass it to `handlers.handle_query(...)`:

```python
def query(ctx, model, local, raw, temperature, chat, append, query_input, citations):
    ...
    handlers.handle_query(
        ...
        citations=citations,
    )
```

**Step 4: Update `handle_query` in `BaseHandlers`**

In `src/conduit/apps/cli/handlers/base_handlers.py`, update `handle_query`:

1. Add `citations: bool = False` to the signature.
2. When building `CLIQueryFunctionInputs`, set `client_params={"return_citations": True}` if `citations=True`, else `{}`:

```python
client_params = {"return_citations": True} if citations else {}
```

3. Pass `client_params` when constructing `CLIQueryFunctionInputs`:

```python
inputs = CLIQueryFunctionInputs(
    ...
    client_params=client_params,
)
```

4. After the display block (after `printer.print_raw` / `printer.print_markdown`), call:

```python
BaseHandlers.handle_citations(response, citations=citations, raw=raw, printer=printer)
```

**Step 5: Run the full test suite**

```bash
pytest tests/cli/ -v
```

Expected: all tests PASS.

**Step 6: Run full project test suite to check for regressions**

```bash
pytest tests/ -v --ignore=tests/old --ignore=tests/regression -x
```

Expected: no new failures. Fix any that appear before committing.

**Step 7: Smoke test acceptance criteria manually**

```bash
# AC1: --help shows flag
conduit query --help | grep -E "\-C|citations"

# AC4: Non-Perplexity model — response then red warning on stderr, exit 0
conduit query "what is rust" -m gpt-4o --citations; echo "Exit: $?"

# AC5: No --citations — no sources block
conduit query "what is rust" -m gpt-4o
```

For AC2 and AC3 (live Perplexity), these require a real `PERPLEXITY_API_KEY` and are optional at this stage.

**Step 8: Commit**

```bash
git add src/conduit/apps/cli/commands/base_commands.py src/conduit/apps/cli/handlers/base_handlers.py tests/cli/test_citations_e2e.py
git commit -m "feat: add --citations / -C flag to conduit query"
```

---

## Acceptance Criteria Checklist

Before declaring done, verify each item:

- [ ] `conduit query --help` shows `-C, --citations` with description
- [ ] `-C` and `--citations` are equivalent
- [ ] With Perplexity model + `--citations`: formatted Sources block appears after response
- [ ] With Perplexity model + `--citations --raw`: JSON array on stdout, nothing extra on stderr
- [ ] With non-Perplexity model + `--citations`: response printed, red warning on stderr, exit code 0, no exception
- [ ] Without `--citations`: no citations output regardless of `message.metadata` contents
- [ ] `conduit query "..." --citations 2>/dev/null` produces clean stdout with no warning interleaved

## Non-goals (do not implement)

- Do not modify `PerplexityClient`, `GenerationResponse`, `AssistantMessage`, or `ResponseMetadata`
- Do not store citations in session persistence or the database
- Do not modify `GenerationResponse.__str__()`
- Do not add `--citations` to `chat`, `cache`, or any other subcommand
- Do not make `--citations` a persistent config setting
- Do not add citation support for the Headwater/local routing path — treat like non-Perplexity (warn)
