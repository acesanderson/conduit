# Image Flag Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add `-i/--image` to `conduit query` so users can send a local image file alongside a text prompt to any vision-capable model.

**Architecture:** Hand-roll a `Conversation` containing a `UserMessage` with `[TextContent, ImageContent]`, then route it through `ConduitSync.pipe_sync()` — a new method that wraps `ConduitAsync.pipe()` synchronously. This bypasses `_prepare_conversation` intentionally (no history loading for image queries) and avoids leaking an event loop into the query function layer.

**Tech Stack:** Click (CLI), Pydantic (`UserMessage`, `ImageContent`, `Conversation`), `ConduitSync`/`ConduitAsync`, pytest + `unittest.mock`, Click's test runner (`CliRunner`)

**Design doc:** `docs/plans/image-flag.md`

---

## Task 1: Add `pipe_sync()` to `ConduitSync`

**Fulfills:** AC6 — `ConduitSync.pipe_sync(conversation)` delegates to `_impl.pipe` with `self.params` and `self.options`.

**Files:**
- Modify: `src/conduit/core/conduit/conduit_sync.py`
- Create: `tests/unit/test_conduit_sync.py`

---

**Step 1: Write the failing test**

Add to `tests/unit/test_conduit_sync.py`:

```python
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from conduit.core.conduit.conduit_sync import ConduitSync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.conversation.conversation import Conversation


def make_conduit_sync() -> ConduitSync:
    """Construct a minimal ConduitSync with mocked params and options."""
    prompt = Prompt("hello")
    conduit = ConduitSync(prompt=prompt)
    conduit.params = MagicMock()
    conduit.options = MagicMock()
    return conduit


def test_pipe_sync_calls_impl_pipe_with_self_params_and_options():
    """AC6: pipe_sync delegates to _impl.pipe with self.params and self.options."""
    conduit = make_conduit_sync()
    conversation = Conversation()

    mock_result = MagicMock(spec=Conversation)
    conduit._impl.pipe = AsyncMock(return_value=mock_result)

    with patch.object(conduit, "_run_sync", side_effect=lambda coro: mock_result):
        result = conduit.pipe_sync(conversation)

    conduit._impl.pipe.assert_called_once_with(
        conversation, conduit.params, conduit.options
    )
    assert result is mock_result
```

**Step 2: Run to verify it fails**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/unit/test_conduit_sync.py::test_pipe_sync_calls_impl_pipe_with_self_params_and_options -v
```

Expected: `FAILED` — `AttributeError: 'ConduitSync' object has no attribute 'pipe_sync'`

**Step 3: Implement**

In `src/conduit/core/conduit/conduit_sync.py`, add after the `run()` method (around line 105), before the `create` classmethod:

```python
def pipe_sync(self, conversation: Conversation) -> Conversation:
    """Run pipe() synchronously using self.params and self.options."""
    return self._run_sync(self._impl.pipe(conversation, self.params, self.options))
```

Also add `Conversation` to the imports at the top of the file:

```python
from conduit.domain.conversation.conversation import Conversation
```

(Place it under `TYPE_CHECKING` if it causes circular import issues — check existing import style in the file.)

**Step 4: Run to verify it passes**

```bash
pytest tests/unit/test_conduit_sync.py::test_pipe_sync_calls_impl_pipe_with_self_params_and_options -v
```

Expected: `PASSED`

**Step 5: Commit**

```bash
git add src/conduit/core/conduit/conduit_sync.py tests/unit/test_conduit_sync.py
git commit -m "feat: add pipe_sync() to ConduitSync"
```

---

## Task 2: Add `image_path` to `CLIQueryFunctionInputs`

**Fulfills:** Prerequisite for AC1, AC5, AC7 — `image_path` field must exist before the query function or command can use it.

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Modify: `tests/cli/test_query_function.py`

---

**Step 1: Write the failing test**

Add to `tests/cli/test_query_function.py`:

```python
def test_inputs_image_path_defaults_to_none():
    """image_path field exists and defaults to None."""
    inputs = make_inputs()
    assert inputs.image_path is None


def test_inputs_accepts_image_path():
    """image_path accepts a string path."""
    inputs = make_inputs(image_path="/tmp/test.png")
    assert inputs.image_path == "/tmp/test.png"
```

**Step 2: Run to verify they fail**

```bash
pytest tests/cli/test_query_function.py::test_inputs_image_path_defaults_to_none tests/cli/test_query_function.py::test_inputs_accepts_image_path -v
```

Expected: `FAILED` — `TypeError: CLIQueryFunctionInputs.__init__() got an unexpected keyword argument 'image_path'`

**Step 3: Implement**

In `src/conduit/apps/cli/query/query_function.py`, add to `CLIQueryFunctionInputs` after the `client_params` field:

```python
image_path: str | None = None
```

**Step 4: Run to verify they pass**

```bash
pytest tests/cli/test_query_function.py::test_inputs_image_path_defaults_to_none tests/cli/test_query_function.py::test_inputs_accepts_image_path -v
```

Expected: `PASSED`

**Step 5: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py tests/cli/test_query_function.py
git commit -m "feat: add image_path field to CLIQueryFunctionInputs"
```

---

## Task 3: Implement image branch in `default_query_function`

**Fulfills:** AC1 — image query builds a `UserMessage` with `[TextContent, ImageContent]` (text first) and routes through `pipe_sync`.

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Modify: `tests/cli/test_query_function.py`

---

**Step 1: Write the failing test**

Add to `tests/cli/test_query_function.py`:

```python
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from conduit.domain.message.message import TextContent, ImageContent, UserMessage
from conduit.domain.conversation.conversation import Conversation


def test_image_query_builds_multimodal_usermessage(tmp_path):
    """AC1: image branch builds UserMessage with [TextContent, ImageContent] in that order."""
    # Create a real (tiny) PNG file so ImageContent.from_file() does not raise
    img = tmp_path / "test.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)  # minimal PNG header

    inputs = make_inputs(query_input="describe this", image_path=str(img))

    captured = {}

    def fake_pipe_sync(conversation):
        captured["conversation"] = conversation
        return MagicMock()

    mock_conduit = MagicMock()
    mock_conduit.pipe_sync.side_effect = fake_pipe_sync

    with patch(
        "conduit.apps.cli.query.query_function.ConduitSync",
        return_value=mock_conduit,
    ):
        default_query_function(inputs)

    conversation = captured["conversation"]
    # Find the UserMessage (skip SystemMessage if present)
    user_msgs = [m for m in conversation.messages if hasattr(m, "role") and str(m.role) == "Role.USER"]
    assert len(user_msgs) == 1
    content = user_msgs[0].content
    assert isinstance(content, list)
    assert isinstance(content[0], TextContent), "TextContent must be first"
    assert isinstance(content[1], ImageContent), "ImageContent must be second"
    assert content[0].text == "describe this"
```

**Step 2: Run to verify it fails**

```bash
pytest tests/cli/test_query_function.py::test_image_query_builds_multimodal_usermessage -v
```

Expected: `FAILED` — test captures no conversation because the image branch doesn't exist yet.

**Step 3: Implement**

In `src/conduit/apps/cli/query/query_function.py`, add the image branch inside `default_query_function`, before the existing `if inputs.search:` check:

```python
import mimetypes

def default_query_function(inputs: CLIQueryFunctionInputs) -> Conversation:
    if inputs.search:
        return _search_query_function(inputs)

    # --- Image branch ---
    if inputs.image_path:
        return _image_query_function(inputs)

    # ... existing code unchanged ...
```

Add `_image_query_function` as a new top-level function (place it near `_search_query_function`):

```python
def _image_query_function(inputs: CLIQueryFunctionInputs) -> Conversation:
    """
    Query function variant for multimodal (image + text) queries.
    Hand-rolls a Conversation with a UserMessage containing [TextContent, ImageContent],
    then routes through ConduitSync.pipe_sync() to bypass _prepare_conversation.
    No history loading — --chat is blocked upstream.
    """
    import mimetypes
    from conduit.core.conduit.conduit_sync import ConduitSync
    from conduit.core.prompt.prompt import Prompt
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.message.message import UserMessage, TextContent, ImageContent
    from conduit.domain.request.generation_params import GenerationParams

    image_path = inputs.image_path
    combined_query = "\n\n".join(
        [inputs.query_input, inputs.context, inputs.append]
    ).strip()

    # Log before loading — surface MIME and path for debugging
    mime_type, _ = mimetypes.guess_type(image_path)
    mime_type = mime_type or "application/octet-stream"
    logger.info("Image query: loading %s (MIME: %s)", image_path, mime_type)

    # Build multimodal UserMessage — TextContent always first
    user_message = UserMessage(
        content=[
            TextContent(text=combined_query),
            ImageContent.from_file(image_path),
        ]
    )

    # Hand-roll the Conversation
    conversation = Conversation()
    if inputs.system_message:
        conversation.ensure_system_message(inputs.system_message)
    conversation.add(user_message)

    logger.debug(
        "pipe_sync: entering with conversation length %d", len(conversation.messages)
    )

    # Build params and options — mirror what default_query_function does
    params = GenerationParams(
        model=inputs.preferred_model,
        system=inputs.system_message or None,
        temperature=inputs.temperature,
    )
    options = settings.default_conduit_options()
    opt_updates: dict = {
        "verbosity": inputs.verbose,
        "include_history": False,  # no history for image queries
    }
    if inputs.cache:
        cache_name = inputs.project_name or settings.default_project_name
        opt_updates["cache"] = settings.default_cache(project_name=cache_name)
    options = options.model_copy(update=opt_updates)

    conduit = ConduitSync(
        prompt=Prompt(combined_query or " "),  # Prompt requires non-empty string
        params=params,
        options=options,
    )
    return conduit.pipe_sync(conversation)
```

**Step 4: Run to verify it passes**

```bash
pytest tests/cli/test_query_function.py::test_image_query_builds_multimodal_usermessage -v
```

Expected: `PASSED`

**Step 5: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py tests/cli/test_query_function.py
git commit -m "feat: implement image branch in default_query_function"
```

---

## Task 4: Stdin + `--append` compose into `TextContent.text`

**Fulfills:** AC5 — when stdin context and query_input are both present, `TextContent.text` contains them joined by `\n\n`.

**Files:**
- Modify: `tests/cli/test_query_function.py`

(No implementation change needed — the existing `"\n\n".join([query_input, context, append]).strip()` already handles this. The test confirms it.)

---

**Step 1: Write the test**

Add to `tests/cli/test_query_function.py`:

```python
def test_image_query_composes_stdin_and_query_into_text_content(tmp_path):
    """AC5: TextContent.text = query_input + '\\n\\n' + stdin context, joined and stripped."""
    img = tmp_path / "slide.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    inputs = make_inputs(
        query_input="convert this",
        context="some stdin context",
        image_path=str(img),
    )

    captured = {}

    def fake_pipe_sync(conversation):
        captured["conversation"] = conversation
        return MagicMock()

    mock_conduit = MagicMock()
    mock_conduit.pipe_sync.side_effect = fake_pipe_sync

    with patch(
        "conduit.apps.cli.query.query_function.ConduitSync",
        return_value=mock_conduit,
    ):
        default_query_function(inputs)

    user_msgs = [
        m for m in captured["conversation"].messages
        if hasattr(m, "role") and str(m.role) == "Role.USER"
    ]
    text = user_msgs[0].content[0].text
    assert text == "convert this\n\nsome stdin context"
```

**Step 2: Run to verify it passes immediately**

```bash
pytest tests/cli/test_query_function.py::test_image_query_composes_stdin_and_query_into_text_content -v
```

Expected: `PASSED` (implementation from Task 3 already handles this)

**Step 3: Commit**

```bash
git add tests/cli/test_query_function.py
git commit -m "test: verify stdin composes into TextContent for image queries (AC5)"
```

---

## Task 5: Empty text query is valid

**Fulfills:** AC7 — `conduit query -i image.png` with no text and no stdin produces `TextContent(text="")`.

**Files:**
- Modify: `tests/cli/test_query_function.py`

---

**Step 1: Write the test**

```python
def test_image_query_with_no_text_produces_empty_text_content(tmp_path):
    """AC7: No query text and no stdin → TextContent(text='') — provider decides if valid."""
    img = tmp_path / "diagram.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    inputs = make_inputs(query_input="", context="", append="", image_path=str(img))

    captured = {}

    def fake_pipe_sync(conversation):
        captured["conversation"] = conversation
        return MagicMock()

    mock_conduit = MagicMock()
    mock_conduit.pipe_sync.side_effect = fake_pipe_sync

    with patch(
        "conduit.apps.cli.query.query_function.ConduitSync",
        return_value=mock_conduit,
    ):
        default_query_function(inputs)

    user_msgs = [
        m for m in captured["conversation"].messages
        if hasattr(m, "role") and str(m.role) == "Role.USER"
    ]
    text_content = user_msgs[0].content[0]
    assert isinstance(text_content, TextContent)
    assert text_content.text == ""
```

**Step 2: Run to verify it passes**

```bash
pytest tests/cli/test_query_function.py::test_image_query_with_no_text_produces_empty_text_content -v
```

Expected: `PASSED`

**Step 3: Commit**

```bash
git add tests/cli/test_query_function.py
git commit -m "test: empty text query with image is valid (AC7)"
```

---

## Task 6: Add `-i/--image` flag and guards to `base_commands.py`

**Fulfills:** AC2, AC3, AC4 — `--chat` and `--search` raise `UsageError`; nonexistent path raises `BadParameter` via Click.

**Files:**
- Modify: `src/conduit/apps/cli/commands/base_commands.py`
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py`
- Create: `tests/cli/test_image_command.py`

---

**Step 1: Write the failing tests**

Create `tests/cli/test_image_command.py`:

```python
from __future__ import annotations

import pytest
from click.testing import CliRunner
from unittest.mock import MagicMock, patch


def _make_cli():
    """Wire up a minimal conduit CLI for testing."""
    import asyncio
    import click
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands

    cli = ConduitCLI()
    return cli.build()


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
    # Click's Path(exists=True) produces "Path '/...' does not exist."
    assert "does not exist" in result.output or "Invalid value" in result.output
```

**Step 2: Run to verify they fail**

```bash
pytest tests/cli/test_image_command.py -v
```

Expected: `FAILED` — `query` command has no `-i` option.

**Step 3: Implement — `base_commands.py`**

In `src/conduit/apps/cli/commands/base_commands.py`, inside `_register_commands()`:

Add the new option decorator to the `query` command (after the existing `--search` option, before `@click.argument`):

```python
@click.option(
    "-i",
    "--image",
    type=click.Path(exists=True, readable=True),
    default=None,
    help="Path to a local image file to include in the query.",
)
```

Add `image: str | None` to the `query` function signature.

Add the guards immediately inside the `query` function body, before unpacking `ctx.obj`:

```python
if image and chat:
    raise click.UsageError("--image cannot be used with --chat")
if image and search:
    raise click.UsageError("--image cannot be used with --search")
```

Pass `image_path=image` to `handlers.handle_query(...)`.

**Step 4: Implement — `base_handlers.py`**

In `handle_query`, add `image_path: str | None = None` to the signature and pass it through to `CLIQueryFunctionInputs`:

```python
inputs = CLIQueryFunctionInputs(
    ...
    image_path=image_path,
)
```

**Step 5: Run to verify tests pass**

```bash
pytest tests/cli/test_image_command.py -v
```

Expected: `PASSED`

**Step 6: Run full test suite to check for regressions**

```bash
pytest tests/cli/ tests/unit/ -v
```

Expected: all previously passing tests still pass.

**Step 7: Commit**

```bash
git add src/conduit/apps/cli/commands/base_commands.py \
        src/conduit/apps/cli/handlers/base_handlers.py \
        tests/cli/test_image_command.py
git commit -m "feat: add -i/--image flag to conduit query with --chat/--search guards"
```

---

## Task 7: Final regression sweep

**Files:**
- No changes — verification only.

**Step 1: Run full test suite**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/ -v --ignore=tests/regression
```

Expected: all tests pass with no errors.

**Step 2: Smoke-test the CLI help output**

```bash
conduit query --help
```

Expected: `-i, --image PATH` appears in the option list.

**Step 3: Commit if any fixups were needed; otherwise tag**

```bash
git log --oneline -6
```

Confirm the five feature commits are present and clean.
