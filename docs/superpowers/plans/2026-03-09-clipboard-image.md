# Clipboard Image Support (`--image @clipboard`) Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `conduit query --image @clipboard` to read an image from the system clipboard and use it as multimodal input, with precise error messages for empty or non-image clipboard contents.

**Architecture:** The command layer resolves `@clipboard` into an `ImageContent` object (via Pillow) before any handler or query function sees it. A new `ImageContent.from_bytes()` classmethod handles encoding. `CLIQueryFunctionInputs` gains an `image_content` field (mutually exclusive with `image_path`) so `_image_query_function` can use a pre-resolved content object without touching the filesystem.

**Tech Stack:** Python 3.12, Click, Pillow (`PIL.ImageGrab`), pytest, `click.testing.CliRunner`

---

## Background: Existing Dead Code

`BaseHandlers` already has two orphaned methods — `grab_image_from_clipboard` and `create_image_message` — that are **never called** from any production path. They use `sys.exit()` for errors and return a raw base64 tuple rather than an `ImageContent`. They must **not** be used or wired up as-is. The new design bypasses them entirely. Do not delete them (out of scope); do not call them.

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `src/conduit/domain/message/message.py` | Add `ImageContent.from_bytes()` |
| Modify | `src/conduit/apps/cli/query/query_function.py` | Add `image_content` field + mutual-exclusivity validator to `CLIQueryFunctionInputs`; update `_image_query_function` to use it |
| Modify | `src/conduit/apps/cli/handlers/base_handlers.py` | Add `image_content` param to `handle_query`; pass it into `CLIQueryFunctionInputs` |
| Modify | `src/conduit/apps/cli/commands/base_commands.py` | Change `--image` to `type=str`; add `@clipboard` sentinel resolution; add manual file validation; wire `image_content` through to handler |
| Modify | `tests/cli/test_image_command.py` | Update `test_nonexistent_image_raises_bad_parameter` to match new manual-validation error message |
| Create | `tests/cli/test_clipboard_image.py` | AC1–AC6, AC8, AC9 |
| Modify | `tests/cli/test_query_function.py` | AC7: mutual-exclusivity validator |

---

## Exact Error Messages (copy these verbatim — tests assert against them)

```
EMPTY_CLIPBOARD = (
    "--image @clipboard: clipboard is empty or contains no image. "
    "On macOS, check that your terminal has Accessibility/Paste permissions in System Settings."
)
NON_IMAGE_CLIPBOARD = "--image @clipboard: clipboard contains data but not an image (found: {type_name})."
FILE_NOT_FOUND    = "--image: file not found: {path}"
```

---

## Chunk 1: Domain and Data Model

### Task 1: `ImageContent.from_bytes()` classmethod

**Fulfills:** AC8 — `ImageContent.from_bytes(b"...", "image/png").url` starts with `"data:image/png;base64,"`

**Files:**
- Modify: `src/conduit/domain/message/message.py` (after `from_file`, ~line 56)
- Modify: `tests/cli/test_clipboard_image.py` (create file)

- [ ] **Step 1: Write the failing test (AC8)**

Create `tests/cli/test_clipboard_image.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

```
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_clipboard_image.py::test_from_bytes_produces_data_url tests/cli/test_clipboard_image.py::test_from_bytes_default_mime_is_png -v
```

Expected: `AttributeError: type object 'ImageContent' has no attribute 'from_bytes'`

- [ ] **Step 3: Implement `ImageContent.from_bytes`**

In `src/conduit/domain/message/message.py`, add after the `from_file` classmethod (~line 56):

```python
    @classmethod
    def from_bytes(cls, data: bytes, mime_type: str = "image/png") -> ImageContent:
        """
        Encode raw bytes as a base64 data URL.
        Use this when image data is already in memory (e.g., from clipboard).
        """
        image_b64 = base64.b64encode(data).decode("utf-8")
        return cls(url=f"data:{mime_type};base64,{image_b64}")
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/cli/test_clipboard_image.py::test_from_bytes_produces_data_url tests/cli/test_clipboard_image.py::test_from_bytes_default_mime_is_png -v
```

Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/domain/message/message.py tests/cli/test_clipboard_image.py
git commit -m "feat: add ImageContent.from_bytes() classmethod (AC8)"
```

---

### Task 2: `CLIQueryFunctionInputs` — `image_content` field with mutual-exclusivity validator

**Fulfills:** AC7 — `CLIQueryFunctionInputs(image_path="x", image_content=ImageContent(...))` raises `ValueError`

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Modify: `tests/cli/test_query_function.py`

- [ ] **Step 1: Write the failing test (AC7)**

Append to `tests/cli/test_query_function.py`:

```python
def test_inputs_rejects_both_image_path_and_image_content():
    """AC7: Setting both image_path and image_content raises ValueError."""
    import pytest
    from conduit.domain.message.message import ImageContent

    content = ImageContent(url="data:image/png;base64,abc")
    with pytest.raises(ValueError, match="Only one of image_path or image_content"):
        make_inputs(image_path="/tmp/test.png", image_content=content)


def test_inputs_image_content_defaults_to_none():
    """AC7 (field existence): image_content field exists and defaults to None."""
    inputs = make_inputs()
    assert inputs.image_content is None


def test_inputs_accepts_image_content_alone():
    """AC7 (valid): image_content can be set without image_path."""
    from conduit.domain.message.message import ImageContent
    content = ImageContent(url="data:image/png;base64,abc")
    inputs = make_inputs(image_content=content)
    assert inputs.image_content is content
```

- [ ] **Step 2: Run tests to verify they fail**

```
pytest tests/cli/test_query_function.py::test_inputs_rejects_both_image_path_and_image_content tests/cli/test_query_function.py::test_inputs_image_content_defaults_to_none tests/cli/test_query_function.py::test_inputs_accepts_image_content_alone -v
```

Expected: `TypeError` or `AttributeError` — `image_content` field does not exist yet.

- [ ] **Step 3: Add `image_content` field and validator to `CLIQueryFunctionInputs`**

In `src/conduit/apps/cli/query/query_function.py`:

1. Add to `TYPE_CHECKING` block:
```python
if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.message.message import ImageContent  # add this line
```

2. Add `image_content` field to the dataclass (after `image_path`):
```python
    image_path: str | None = None
    image_content: ImageContent | None = None
```

3. Add `__post_init__` method to the dataclass:
```python
    def __post_init__(self):
        if self.image_path is not None and self.image_content is not None:
            raise ValueError(
                "Only one of image_path or image_content may be set, not both."
            )
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/cli/test_query_function.py::test_inputs_rejects_both_image_path_and_image_content tests/cli/test_query_function.py::test_inputs_image_content_defaults_to_none tests/cli/test_query_function.py::test_inputs_accepts_image_content_alone -v
```

Expected: `3 passed`

- [ ] **Step 5: Verify existing query_function tests still pass**

```
pytest tests/cli/test_query_function.py -v
```

Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py tests/cli/test_query_function.py
git commit -m "feat: add image_content field with mutual-exclusivity validator to CLIQueryFunctionInputs (AC7)"
```

---

### Task 3: `_image_query_function` uses `image_content` when set

**Fulfills:** Prerequisite for AC1 — clipboard image content reaches the LLM without a file path.

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Modify: `tests/cli/test_query_function.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/cli/test_query_function.py`:

```python
def test_image_query_uses_image_content_when_set():
    """image_content takes precedence over image_path when set in _image_query_function."""
    from conduit.domain.message.message import ImageContent, TextContent

    pre_resolved = ImageContent(url="data:image/png;base64,FAKEDATA")
    inputs = make_inputs(query_input="describe", image_content=pre_resolved)

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
    content = user_msgs[0].content
    assert isinstance(content[1], ImageContent)
    assert content[1].url == "data:image/png;base64,FAKEDATA"
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/cli/test_query_function.py::test_image_query_uses_image_content_when_set -v
```

Expected: `FAIL` — `_image_query_function` does not check `image_content` yet.

- [ ] **Step 3: Update `_image_query_function` to use `image_content`**

In `src/conduit/apps/cli/query/query_function.py`, replace the image loading block in `_image_query_function`. Current code (~line 130):

```python
    image_path = inputs.image_path
    # ... later ...
    user_message = UserMessage(
        content=[
            TextContent(text=combined_query),
            ImageContent.from_file(image_path),
        ]
    )
```

Replace with:

```python
    # Resolve image: pre-built ImageContent takes priority over a file path
    if inputs.image_content is not None:
        resolved_image = inputs.image_content
    else:
        resolved_image = ImageContent.from_file(inputs.image_path)

    user_message = UserMessage(
        content=[
            TextContent(text=combined_query),
            resolved_image,
        ]
    )
```

Also remove the now-unused `image_path = inputs.image_path` line and the `mime_type` logging block that references it (since that path is only relevant when loading from a file).

The updated logging should be:

```python
    if inputs.image_content is not None:
        logger.info("Image query: using pre-resolved ImageContent from caller")
    else:
        mime_type, _ = mimetypes.guess_type(inputs.image_path)
        mime_type = mime_type or "application/octet-stream"
        logger.info("Image query: loading %s (MIME: %s)", inputs.image_path, mime_type)
```

- [ ] **Step 4: Run tests to verify they pass**

```
pytest tests/cli/test_query_function.py -v
```

Expected: all pass (including the new test and all pre-existing ones)

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/query/query_function.py tests/cli/test_query_function.py
git commit -m "feat: _image_query_function uses image_content directly when pre-resolved"
```

---

## Chunk 2: Handler and Command Layer

### Task 4: `BaseHandlers.handle_query` accepts `image_content`

**Fulfills:** Plumbing prerequisite for AC1 — `image_content` must flow from command → handler → `CLIQueryFunctionInputs`.

**Files:**
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py`

No new tests needed — `handle_query` is an integration seam; it is tested transitively by the CLI-level tests in Task 6.

- [ ] **Step 1: Add `image_content` parameter to `handle_query`**

In `src/conduit/apps/cli/handlers/base_handlers.py`:

1. Add to the `TYPE_CHECKING` block:
```python
if TYPE_CHECKING:
    # ... existing imports ...
    from conduit.domain.message.message import ImageContent  # add this
```

2. Add `image_content` to `handle_query` signature (after `image_path`):
```python
    @staticmethod
    def handle_query(
        # ... existing params ...
        image_path: str | None = None,
        image_content: ImageContent | None = None,  # add this
    ) -> None:
```

3. Pass `image_content` into `CLIQueryFunctionInputs` construction:
```python
        inputs = CLIQueryFunctionInputs(
            # ... existing fields ...
            image_path=image_path,
            image_content=image_content,  # add this
        )
```

- [ ] **Step 2: Run the full CLI test suite to verify nothing broke**

```
pytest tests/cli/ -v
```

Expected: all existing tests pass

- [ ] **Step 3: Commit**

```bash
git add src/conduit/apps/cli/handlers/base_handlers.py
git commit -m "feat: thread image_content through BaseHandlers.handle_query"
```

---

### Task 5: Command layer — `@clipboard` sentinel, manual file validation, CMYK handling

**Fulfills:** AC2, AC3, AC4, AC5, AC6, AC9

**Files:**
- Modify: `src/conduit/apps/cli/commands/base_commands.py`
- Modify: `tests/cli/test_image_command.py` (update one existing test)
- Modify: `tests/cli/test_clipboard_image.py` (add AC2–AC6, AC9 tests)

#### 5a — AC2: Empty clipboard raises UsageError

- [ ] **Step 1: Write the failing test (AC2)**

Append to `tests/cli/test_clipboard_image.py`:

```python
from unittest.mock import MagicMock, patch
from click.testing import CliRunner


def _make_cli():
    from conduit.apps.cli.cli_class import ConduitCLI
    from conduit.apps.cli.commands.base_commands import BaseCommands
    conduit_cli = ConduitCLI()
    conduit_cli.attach(BaseCommands())
    return conduit_cli.cli


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
```

- [ ] **Step 2: Run test to verify it fails**

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_empty_raises_usage_error -v
```

Expected: `FAIL` — `@clipboard` sentinel not yet handled; Click may error on unknown path.

- [ ] **Step 3: Implement sentinel handling in `base_commands.py`**

In `src/conduit/apps/cli/commands/base_commands.py`:

1. Add imports at module level (after existing imports):
```python
import io
import logging
import warnings
from pathlib import Path

logger = logging.getLogger(__name__)
```

2. Add module-level sentinel constant:
```python
_CLIPBOARD_SENTINEL = "@clipboard"
```

3. Add module-level helper function (before the `BaseCommands` class):
```python
def _resolve_clipboard_image() -> ImageContent:
    """
    Grab an image from the system clipboard and return it as an ImageContent object.
    Raises click.UsageError for empty clipboard or non-image clipboard contents.
    """
    from PIL import ImageGrab, Image as PILImage

    logger.info("--image @clipboard: grabbing image from clipboard")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clip = ImageGrab.grabclipboard()

    if not isinstance(clip, PILImage.Image):
        logger.warning("grabclipboard() returned %s", type(clip))
        if clip is None:
            raise click.UsageError(
                "--image @clipboard: clipboard is empty or contains no image. "
                "On macOS, check that your terminal has Accessibility/Paste "
                "permissions in System Settings."
            )
        raise click.UsageError(
            f"--image @clipboard: clipboard contains data but not an image "
            f"(found: {type(clip).__name__})."
        )

    mode = "RGBA" if clip.mode in ("RGBA", "LA", "PA") else "RGB"
    img = clip.convert(mode)
    w, h = img.size
    buf = io.BytesIO()
    img.save(buf, "PNG")
    buf.seek(0)
    data = buf.read()
    logger.info(
        "Clipboard image: mode=%s, size=%dx%d, encoded_bytes=%d", mode, w, h, len(data)
    )

    from conduit.domain.message.message import ImageContent
    return ImageContent.from_bytes(data, "image/png")
```

4. Change the `--image` Click option from:
```python
        @click.option(
            "-i",
            "--image",
            type=click.Path(exists=True, readable=True),
            default=None,
            help="Path to a local image file to include in the query.",
        )
```
to:
```python
        @click.option(
            "-i",
            "--image",
            type=str,
            default=None,
            help='Path to a local image file, or "@clipboard" to read from clipboard.',
        )
```

5. In the `query` function body, add image resolution logic **before** the handler call (after the existing guard checks for `--chat` and `--search`):
```python
            # Resolve --image flag into image_path or image_content
            image_path: str | None = None
            image_content: ImageContent | None = None

            if image is not None:
                if image == _CLIPBOARD_SENTINEL:
                    image_content = _resolve_clipboard_image()
                else:
                    p = Path(image)
                    if not p.exists():
                        raise click.UsageError(f"--image: file not found: {image}")
                    if not p.is_file():
                        raise click.UsageError(f"--image: not a file: {image}")
                    image_path = image
```

6. Update the `handlers.handle_query(...)` call to pass `image_content` and use the local `image_path` variable (not the old `image` parameter):
```python
            handlers.handle_query(
                # ... existing args ...
                image_path=image_path,
                image_content=image_content,
            )
```

Also add `ImageContent` to the `TYPE_CHECKING` block in `base_commands.py` if needed for type hints (it's only used at runtime inside the helper, so the import is inside the function — no changes to `TYPE_CHECKING` needed).

- [ ] **Step 4: Run AC2 test to verify it passes**

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_empty_raises_usage_error -v
```

Expected: `1 passed`

- [ ] **Step 5: Commit**

```bash
git add src/conduit/apps/cli/commands/base_commands.py tests/cli/test_clipboard_image.py
git commit -m "feat: add @clipboard sentinel resolution to --image flag (AC2 passing)"
```

#### 5b — AC3: Non-image clipboard data raises UsageError

- [ ] **Step 1: Write the failing test (AC3)**

Append to `tests/cli/test_clipboard_image.py`:

```python
def test_clipboard_non_image_raises_usage_error():
    """AC3: grabclipboard() returns a non-Image type → UsageError with type name."""
    runner = CliRunner()
    cli = _make_cli()

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = ["some text"]  # list, not Image
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "describe"])

    assert result.exit_code != 0
    assert "--image @clipboard: clipboard contains data but not an image" in result.output
    assert "list" in result.output
```

- [ ] **Step 2: Run test to verify it passes** (should already pass from Task 5a implementation)

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_non_image_raises_usage_error -v
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/cli/test_clipboard_image.py
git commit -m "test: non-image clipboard raises UsageError with type name (AC3)"
```

#### 5c — AC4: File list in clipboard raises UsageError

- [ ] **Step 1: Write the failing test (AC4)**

Append to `tests/cli/test_clipboard_image.py`:

```python
def test_clipboard_file_list_raises_usage_error():
    """AC4: grabclipboard() returns a list of file paths → UsageError."""
    runner = CliRunner()
    cli = _make_cli()

    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        mock_grab.grabclipboard.return_value = ["/Users/user/photo.png"]
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "describe"])

    assert result.exit_code != 0
    assert "--image @clipboard: clipboard contains data but not an image" in result.output
```

- [ ] **Step 2: Run test to verify it passes**

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_file_list_raises_usage_error -v
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/cli/test_clipboard_image.py
git commit -m "test: clipboard file list raises UsageError (AC4)"
```

#### 5d — AC5: Existing file path behavior unchanged; update broken test

- [ ] **Step 1: Update the broken existing test**

In `tests/cli/test_image_command.py`, update `test_nonexistent_image_raises_bad_parameter`:

Old assertion:
```python
    assert "does not exist" in result.output or "Invalid value" in result.output
```

New assertion (matches our manual-validation message):
```python
    assert "--image: file not found:" in result.output
```

- [ ] **Step 2: Write a new AC5 test**

Append to `tests/cli/test_clipboard_image.py`:

```python
def test_file_path_passes_through_unchanged(tmp_path):
    """AC5: a valid file path is passed to the handler as image_path (not image_content)."""
    img = tmp_path / "photo.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 8)

    runner = CliRunner()
    cli = _make_cli()

    captured = {}

    def fake_query_fn(inputs):
        captured["inputs"] = inputs
        return MagicMock()

    with patch("conduit.apps.cli.scripts.conduit_cli") as _:
        # Patch at the ConduitCLI level: inject our fake query function
        from conduit.apps.cli.cli_class import ConduitCLI
        from conduit.apps.cli.commands.base_commands import BaseCommands

        cli2 = ConduitCLI(query_function=fake_query_fn)
        cli2.attach(BaseCommands())
        result = runner.invoke(cli2.cli, ["query", "--image", str(img), "describe"])

    assert captured.get("inputs") is not None
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
```

- [ ] **Step 3: Run AC5 tests**

```
pytest tests/cli/test_clipboard_image.py::test_file_path_passes_through_unchanged tests/cli/test_clipboard_image.py::test_nonexistent_file_raises_usage_error tests/cli/test_image_command.py::test_nonexistent_image_raises_bad_parameter -v
```

Expected: `3 passed`

- [ ] **Step 4: Commit**

```bash
git add tests/cli/test_clipboard_image.py tests/cli/test_image_command.py
git commit -m "test: AC5 file path unchanged; update nonexistent-file assertion for manual validation"
```

#### 5e — AC6: `--image @clipboard --chat` raises UsageError

- [ ] **Step 1: Write the failing test (AC6)**

Append to `tests/cli/test_clipboard_image.py`:

```python
def test_clipboard_with_chat_raises_usage_error():
    """AC6: --image @clipboard --chat is rejected before clipboard is accessed."""
    runner = CliRunner()
    cli = _make_cli()

    # ImageGrab must NOT be called — the guard fires first
    with patch("conduit.apps.cli.commands.base_commands.ImageGrab") as mock_grab:
        result = runner.invoke(cli, ["query", "--image", "@clipboard", "--chat", "describe"])
        mock_grab.grabclipboard.assert_not_called()

    assert result.exit_code != 0
    assert "--image cannot be used with --chat" in result.output
```

- [ ] **Step 2: Run test to verify it passes** (guard already exists in command body)

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_with_chat_raises_usage_error -v
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/cli/test_clipboard_image.py
git commit -m "test: --image @clipboard --chat rejected before clipboard access (AC6)"
```

#### 5f — AC9: CMYK image converted to RGB without raising

- [ ] **Step 1: Write the failing test (AC9)**

Append to `tests/cli/test_clipboard_image.py`:

```python
def test_clipboard_cmyk_image_converted_to_rgb():
    """AC9: CMYK-mode PIL image is converted to RGB before PNG encoding; no exception raised."""
    from PIL import Image as PILImage

    runner = CliRunner()
    cli = _make_cli()

    cmyk_image = PILImage.new("CMYK", (10, 10), color=(0, 0, 0, 0))

    captured_image_content = {}

    def fake_query_fn(inputs):
        captured_image_content["image_content"] = inputs.image_content
        return MagicMock()

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
```

- [ ] **Step 2: Run test to verify it passes** (CMYK handling is already in `_resolve_clipboard_image`)

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_cmyk_image_converted_to_rgb -v
```

Expected: `1 passed`

- [ ] **Step 3: Commit**

```bash
git add tests/cli/test_clipboard_image.py
git commit -m "test: CMYK clipboard image converted to RGB without error (AC9)"
```

---

## Chunk 3: Integration — AC1 (success path)

### Task 6: End-to-end clipboard success path

**Fulfills:** AC1 — `conduit query "describe" --image @clipboard` with a PNG in clipboard → query succeeds.

**Files:**
- Modify: `tests/cli/test_clipboard_image.py`

- [ ] **Step 1: Write the failing test (AC1)**

Append to `tests/cli/test_clipboard_image.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails** (check the `ConduitCLI` constructor signature — it may not accept `query_function` as a kwarg; adjust if needed)

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_image_success_path -v
```

If `ConduitCLI` doesn't accept `query_function` in its constructor, inspect `src/conduit/apps/cli/cli_class.py` to find the right injection point and adjust the test accordingly.

- [ ] **Step 3: Implement any missing wiring, then run to verify pass**

```
pytest tests/cli/test_clipboard_image.py::test_clipboard_image_success_path -v
```

Expected: `1 passed`

- [ ] **Step 4: Run the full test suite**

```
pytest tests/cli/ -v
```

Expected: all pass, no regressions

- [ ] **Step 5: Commit**

```bash
git add tests/cli/test_clipboard_image.py
git commit -m "test: end-to-end clipboard image success path (AC1)"
```

---

## Final Verification

- [ ] Run all tests once more:

```
pytest tests/cli/ tests/unit/ -v
```

- [ ] Confirm all 9 ACs are covered:

| AC | Test | File |
|----|------|------|
| AC1 | `test_clipboard_image_success_path` | `test_clipboard_image.py` |
| AC2 | `test_clipboard_empty_raises_usage_error` | `test_clipboard_image.py` |
| AC3 | `test_clipboard_non_image_raises_usage_error` | `test_clipboard_image.py` |
| AC4 | `test_clipboard_file_list_raises_usage_error` | `test_clipboard_image.py` |
| AC5 | `test_file_path_passes_through_unchanged`, `test_nonexistent_file_raises_usage_error` | `test_clipboard_image.py` |
| AC6 | `test_clipboard_with_chat_raises_usage_error` | `test_clipboard_image.py` |
| AC7 | `test_inputs_rejects_both_image_path_and_image_content` | `test_query_function.py` |
| AC8 | `test_from_bytes_produces_data_url`, `test_from_bytes_default_mime_is_png` | `test_clipboard_image.py` |
| AC9 | `test_clipboard_cmyk_image_converted_to_rgb` | `test_clipboard_image.py` |
