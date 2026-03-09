# Spec: `--image` flag for `conduit query`

## Goal
Allow `conduit query -i path/to/image.png "describe this"` to send a multimodal (`TextContent` + `ImageContent`) message to a vision-capable model via `ConduitSync.pipe_sync()`.

---

## Interface

```
conduit query -i <path> "your text prompt"
cat notes.txt | conduit query -i slide.png "convert to markdown"
conduit query -i diagram.png             # text query may be empty if stdin provides context
```

- `-i` / `--image` — path to a local image file. Validated as an existing, readable file by Click (`click.Path(exists=True, readable=True)`). Format is not validated beyond MIME inference from extension — malformed files will fail at the provider.
- Stdin and `--append` compose into `TextContent` as usual. Combined query may be empty string (valid — provider accepts image-only messages).
- Incompatible with `--chat`: raise `click.UsageError("--image cannot be used with --chat (no multimodal history support)")`.
- Incompatible with `--search`: raise `click.UsageError("--image cannot be used with --search")`.
- `-r/--raw` behavior is unchanged — applied to the response after `query_function` returns.
- URL inputs, base64 stdin, and clipboard (`--paste`) are **out of scope**.

---

## Non-goals (explicit)
- URL image inputs (`http://`, `https://`) — not in scope; `ImageContent.url` supports it but the CLI will not expose it here
- Clipboard image capture — `grab_image_from_clipboard` exists but is not wired by this change
- Multiple images (`-i a.png -i b.png`) — YAGNI; single image only
- MIME type validation beyond `mimetypes.guess_type` — provider error is the contract
- History loading for image queries — bypassing `_prepare_conversation` means no repository load; this is intentional and must not be worked around

---

## Design decisions
1. **We chose `pipe()` directly over extending `_prepare_conversation`** because adding a `UserMessage | str` branch there modifies core framework internals for a CLI-level concern. The framework already supports hand-rolled `Conversation` objects at the `pipe()` boundary — that's the right seam.
2. **We chose to hard-block `--chat` with `--image`** over loading history and appending a multimodal message because multimodal messages in a persisted session create serialization complexity we haven't solved. Fail loudly now, enable later.
3. **We chose to validate the model at the provider level** over maintaining a vision-capable model list in the framework because that list would rot and the provider error is already informative enough.
4. **We chose YAGNI on multiple images** over `nargs=-1` from the start. One image covers the use case; generalize when there's a real need.
5. **We chose to add `pipe_sync()` to `ConduitSync`** over leaking an event loop into `CLIQueryFunctionInputs`, because `ConduitSync`'s entire purpose is wrapping `ConduitAsync` for synchronous callers. This is exactly that use case.
6. **We chose file path over clipboard** for the primary interface. The clipboard helpers already exist in `BaseHandlers` but are wired to nothing. Clipboard as a separate flag is a natural follow-on, not part of this change.

---

## Changes

### `ConduitSync` (`conduit_sync.py`)
```python
def pipe_sync(self, conversation: Conversation) -> Conversation:
    """Run pipe() synchronously using self.params and self.options."""
    return self._run_sync(self._impl.pipe(conversation, self.params, self.options))
```

### `CLIQueryFunctionInputs` (`query_function.py`)
- Add `image_path: str | None = None`

### `base_commands.py`
- Add option: `@click.option("-i", "--image", type=click.Path(exists=True, readable=True), default=None)`
- Guard (before handler call):
  ```python
  if image and chat:
      raise click.UsageError("--image cannot be used with --chat")
  if image and search:
      raise click.UsageError("--image cannot be used with --search")
  ```
- Pass `image_path=image` to `handle_query`

### `base_handlers.py` — `handle_query`
- Accept and pass through `image_path: str | None = None` to `CLIQueryFunctionInputs`

### `default_query_function` (`query_function.py`)

Image branch (when `inputs.image_path` is set):
```
1. Build combined_query string (query_input + context + append) as usual
2. Build UserMessage:
     content = [TextContent(text=combined_query), ImageContent.from_file(image_path)]
     # TextContent always first; some providers are ordering-sensitive
3. Build Conversation:
     conversation = Conversation()
     if params.system:
         conversation.ensure_system_message(params.system)
     conversation.add(user_message)
4. Construct ConduitSync(prompt=Prompt(combined_query), params=params, options=options)
5. Return conduit.pipe_sync(conversation)
```

`from_file` is not wrapped — `OSError` and `IOError` propagate up and surface as unhandled exceptions (acceptable; Click will print the traceback). If we want a friendlier message, that's a follow-on.

Standard branch (no image): unchanged.

---

## Logging

Add to the image branch in `default_query_function`:
```python
logger.info("Image query: loading %s (MIME: %s)", image_path, detected_mime_type)
logger.debug("pipe_sync: entering with conversation length %d", len(conversation.messages))
```

`detected_mime_type` is recoverable from `ImageContent.from_file` internals or via `mimetypes.guess_type` before constructing the object.

---

## Acceptance criteria

1. `conduit query -i valid.png "describe"` — produces a response without error. Verifiable by mocking `ConduitSync.pipe_sync` and asserting the `Conversation` passed to it contains a `UserMessage` with `[TextContent, ImageContent]` in that order.
2. `conduit query -i valid.png "describe" --chat` — raises `UsageError` with the specified message. Verifiable via Click test runner.
3. `conduit query -i valid.png "describe" --search` — same.
4. `conduit query -i nonexistent.png "describe"` — Click raises `BadParameter` (handled by `click.Path(exists=True)`). No custom code needed.
5. `cat context.txt | conduit query -i slide.png "convert"` — `TextContent.text` contains both stdin and query argument joined by `\n\n`. Verifiable by inspecting the `UserMessage` passed to `pipe_sync`.
6. `ConduitSync.pipe_sync(conversation)` — unit test: mock `_impl.pipe`, assert it's called with `self.params` and `self.options`.
7. `conduit query -i valid.png` (no text, no stdin) — produces a response with `TextContent(text="")`. Provider behavior on empty text is not our contract to enforce.
