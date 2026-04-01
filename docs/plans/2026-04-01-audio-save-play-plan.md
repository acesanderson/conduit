# Audio Input, `--save`, and `--play` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--audio` (file path or `@mic` live recording), `--save` (write any response to file), and `--play` (play audio response) flags to `conduit query` / `ask`.

**Architecture:** Mirror the existing `--image` / `@clipboard` pattern exactly. `--audio` resolves to `AudioContent` in the command layer, dispatches to a new `_audio_query_function` that builds `UserMessage([TextContent, AudioContent])` and routes via `pipe_sync`. `--save` is handled post-response in the handler; `--play` triggers `pydub` playback from the saved file. `@mic` recording lifted from `tap/scripts/record_cli.py`.

**Tech Stack:** Click (CLI), Pydantic (`AudioContent`, `AudioOutput`, `UserMessage`), `pyaudio` + `pydub` (mic recording and playback), `ConduitSync.pipe_sync`, pytest + `unittest.mock`, Click's `CliRunner`

**Design doc:** `docs/plans/2026-04-01-audio-save-play-design.md`

---

## File Map

| File | Change |
|------|--------|
| `src/conduit/domain/message/message.py` | Add `AudioContent.from_bytes()` classmethod |
| `src/conduit/apps/cli/query/query_function.py` | Add `audio_path`/`audio_content` to `CLIQueryFunctionInputs`; add `_audio_query_function()`; update dispatch in `default_query_function` |
| `src/conduit/apps/cli/commands/base_commands.py` | Add `--audio`, `--save`, `--play` flags; `_AUDIO_SENTINEL`; `AudioRecorder`; `_resolve_mic_audio()`; "transcribe this" default logic |
| `src/conduit/apps/cli/handlers/base_handlers.py` | Update `handle_query()` signature; add `_save_response()`, `_play_audio()` |
| `tests/cli/test_audio_content.py` | New — unit tests for `AudioContent.from_bytes()` |
| `tests/cli/test_query_function.py` | Extend — audio fields on `CLIQueryFunctionInputs`, `_audio_query_function` shape |
| `tests/cli/test_audio_command.py` | New — CLI flag guards, `_save_response`, `--play` / `--save` wiring |

---

## Task 1: Add `AudioContent.from_bytes()`

**Fulfills:** AC9, AC10

**Files:**
- Modify: `src/conduit/domain/message/message.py`
- Create: `tests/cli/test_audio_content.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_audio_content.py`:

```python
from __future__ import annotations

import base64
import pytest
from conduit.domain.message.message import AudioContent


def test_from_bytes_default_format_is_mp3():
    """AC9: AudioContent.from_bytes(data) defaults format to 'mp3'."""
    data = b"fake audio bytes"
    result = AudioContent.from_bytes(data)
    assert result.format == "mp3"


def test_from_bytes_wav_format():
    """AC10: AudioContent.from_bytes(data, format='wav') sets format to 'wav'."""
    data = b"fake audio bytes"
    result = AudioContent.from_bytes(data, format="wav")
    assert result.format == "wav"


def test_from_bytes_encodes_as_base64():
    """from_bytes stores data as base64-encoded string."""
    data = b"hello audio"
    result = AudioContent.from_bytes(data)
    assert result.data == base64.b64encode(data).decode("utf-8")
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_audio_content.py -v
```

Expected: `FAILED` — `TypeError: AudioContent.from_bytes() is not defined` (or similar AttributeError)

- [ ] **Step 3: Implement**

In `src/conduit/domain/message/message.py`, add inside the `AudioContent` class after the existing `from_file` classmethod:

```python
@classmethod
def from_bytes(cls, data: bytes, format: Literal["wav", "mp3"] = "mp3") -> AudioContent:
    """
    Encode raw bytes as base64. Used for @mic recordings.
    Mirrors ImageContent.from_bytes().
    """
    audio_b64 = base64.b64encode(data).decode("utf-8")
    return cls(data=audio_b64, format=format)
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/cli/test_audio_content.py -v
```

Expected: all 3 tests `PASSED`

- [ ] **Step 5: Commit**

```bash
git -C /Users/bianders/Brian_Code/conduit-project add \
    src/conduit/domain/message/message.py \
    tests/cli/test_audio_content.py
git -C /Users/bianders/Brian_Code/conduit-project commit -m "feat: add AudioContent.from_bytes()"
```

---

## Task 2: Add `audio_path` / `audio_content` to `CLIQueryFunctionInputs`

**Fulfills:** AC1, AC2, AC3

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Modify: `tests/cli/test_query_function.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli/test_query_function.py`:

```python
import pytest
from conduit.domain.message.message import AudioContent, ImageContent


def test_inputs_audio_path_defaults_to_none():
    """audio_path field exists and defaults to None."""
    inputs = make_inputs()
    assert inputs.audio_path is None


def test_inputs_audio_content_defaults_to_none():
    """audio_content field exists and defaults to None."""
    inputs = make_inputs()
    assert inputs.audio_content is None


def test_inputs_rejects_both_audio_path_and_audio_content():
    """AC1: both audio_path and audio_content raises ValueError."""
    content = AudioContent.from_bytes(b"data")
    with pytest.raises(ValueError, match="Only one of audio_path or audio_content"):
        make_inputs(audio_path="/tmp/clip.mp3", audio_content=content)


def test_inputs_rejects_audio_path_with_image_path():
    """AC2: audio_path + image_path raises ValueError."""
    with pytest.raises(ValueError, match="--audio and --image cannot be used together"):
        make_inputs(audio_path="/tmp/clip.mp3", image_path="/tmp/img.png")


def test_inputs_rejects_audio_content_with_image_content():
    """AC3: audio_content + image_content raises ValueError."""
    audio = AudioContent.from_bytes(b"data")
    image = ImageContent(url="data:image/png;base64,abc")
    with pytest.raises(ValueError, match="--audio and --image cannot be used together"):
        make_inputs(audio_content=audio, image_content=image)
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_query_function.py::test_inputs_audio_path_defaults_to_none \
       tests/cli/test_query_function.py::test_inputs_audio_content_defaults_to_none \
       tests/cli/test_query_function.py::test_inputs_rejects_both_audio_path_and_audio_content \
       tests/cli/test_query_function.py::test_inputs_rejects_audio_path_with_image_path \
       tests/cli/test_query_function.py::test_inputs_rejects_audio_content_with_image_content \
       -v
```

Expected: all `FAILED` — `TypeError: CLIQueryFunctionInputs.__init__() got an unexpected keyword argument 'audio_path'`

- [ ] **Step 3: Implement**

In `src/conduit/apps/cli/query/query_function.py`:

Add to `CLIQueryFunctionInputs` after the existing `image_content` field:

```python
audio_path: str | None = None
audio_content: AudioContent | None = None
```

Add the import in the `TYPE_CHECKING` block at the top of the file (alongside the existing `ImageContent`):

```python
if TYPE_CHECKING:
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.message.message import ImageContent, AudioContent
```

Update `__post_init__` to add the two new mutual-exclusion checks after the existing image check:

```python
def __post_init__(self):
    if self.image_path is not None and self.image_content is not None:
        raise ValueError(
            "Only one of image_path or image_content may be set, not both."
        )
    if self.audio_path is not None and self.audio_content is not None:
        raise ValueError(
            "Only one of audio_path or audio_content may be set, not both."
        )
    if (self.audio_path is not None or self.audio_content is not None) and (
        self.image_path is not None or self.image_content is not None
    ):
        raise ValueError(
            "--audio and --image cannot be used together."
        )
```

- [ ] **Step 4: Run to verify they pass**

```bash
pytest tests/cli/test_query_function.py::test_inputs_audio_path_defaults_to_none \
       tests/cli/test_query_function.py::test_inputs_audio_content_defaults_to_none \
       tests/cli/test_query_function.py::test_inputs_rejects_both_audio_path_and_audio_content \
       tests/cli/test_query_function.py::test_inputs_rejects_audio_path_with_image_path \
       tests/cli/test_query_function.py::test_inputs_rejects_audio_content_with_image_content \
       -v
```

Expected: all `PASSED`

- [ ] **Step 5: Run full existing test suite to confirm no regressions**

```bash
pytest tests/cli/test_query_function.py tests/cli/test_image_command.py -v
```

Expected: all previously passing tests still `PASSED`

- [ ] **Step 6: Commit**

```bash
git -C /Users/bianders/Brian_Code/conduit-project add \
    src/conduit/apps/cli/query/query_function.py \
    tests/cli/test_query_function.py
git -C /Users/bianders/Brian_Code/conduit-project commit -m "feat: add audio_path/audio_content to CLIQueryFunctionInputs with mutual-exclusion guards"
```

---

## Task 3: Implement `_audio_query_function`

**Fulfills:** AC11

**Files:**
- Modify: `src/conduit/apps/cli/query/query_function.py`
- Modify: `tests/cli/test_query_function.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/cli/test_query_function.py`:

```python
from conduit.domain.message.message import AudioContent, TextContent


def test_audio_query_builds_multimodal_usermessage(tmp_path):
    """AC11: _audio_query_function builds UserMessage([TextContent, AudioContent]) — text first."""
    import base64
    from conduit.apps.cli.query.query_function import default_query_function

    # Create a minimal valid MP3 file (just non-empty bytes; no real decoding happens here)
    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"\xff\xfb" + b"\x00" * 16)  # minimal MP3-ish bytes

    inputs = make_inputs(query_input="summarize this", audio_path=str(audio_file))

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

    assert "conversation" in captured, "_audio_query_function did not call pipe_sync"
    user_msgs = [
        m for m in captured["conversation"].messages
        if hasattr(m, "role") and str(m.role) == "Role.USER"
    ]
    assert len(user_msgs) == 1
    content = user_msgs[0].content
    assert isinstance(content, list), "content must be a list"
    assert isinstance(content[0], TextContent), "content[0] must be TextContent"
    assert isinstance(content[1], AudioContent), "content[1] must be AudioContent"
    assert content[0].text == "summarize this"
```

- [ ] **Step 2: Run to verify it fails**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_query_function.py::test_audio_query_builds_multimodal_usermessage -v
```

Expected: `FAILED` — `_audio_query_function` does not exist yet; `default_query_function` falls through to the text path and `pipe_sync` is never called.

- [ ] **Step 3: Implement `_audio_query_function`**

In `src/conduit/apps/cli/query/query_function.py`, add the following new function. Place it directly after `_image_query_function`:

```python
def _audio_query_function(inputs: CLIQueryFunctionInputs) -> Conversation:
    """
    Query function variant for audio (audio + text) queries.
    Builds UserMessage([TextContent, AudioContent]) and routes through pipe_sync.
    History is disabled — --chat is blocked upstream.
    combined_query already contains the "transcribe this" default if no query was given.
    """
    from conduit.domain.conversation.conversation import Conversation
    from conduit.domain.message.message import UserMessage, TextContent, AudioContent
    from conduit.domain.request.generation_params import GenerationParams

    combined_query = "\n\n".join(
        [inputs.query_input, inputs.context, inputs.append]
    ).strip()

    if inputs.audio_content is not None:
        logger.info("Audio query: using pre-resolved AudioContent from caller")
        resolved_audio = inputs.audio_content
    else:
        fmt = inputs.audio_path.rsplit(".", 1)[-1].lower()
        logger.info("Audio query: loading %s (format: %s)", inputs.audio_path, fmt)
        resolved_audio = AudioContent.from_file(inputs.audio_path)

    user_message = UserMessage(
        content=[
            TextContent(text=combined_query),
            resolved_audio,
        ]
    )

    conversation = Conversation()
    if inputs.system_message:
        conversation.ensure_system_message(inputs.system_message)
    conversation.add(user_message)

    logger.debug(
        "pipe_sync: entering with conversation length %d", len(conversation.messages)
    )

    params = GenerationParams(
        model=inputs.preferred_model,
        system=inputs.system_message or None,
        temperature=inputs.temperature,
    )
    options = settings.default_conduit_options()
    opt_updates: dict = {
        "verbosity": inputs.verbose,
        "include_history": False,
    }
    if inputs.cache:
        cache_name = inputs.project_name or settings.default_project_name
        opt_updates["cache"] = settings.default_cache(project_name=cache_name)
    options = options.model_copy(update=opt_updates)

    conduit = ConduitSync(
        prompt=Prompt(combined_query or " "),
        params=params,
        options=options,
    )
    return conduit.pipe_sync(conversation)
```

Then update the dispatch block in `default_query_function`. Find the existing:

```python
if inputs.search:
    return _search_query_function(inputs)

if inputs.image_path or inputs.image_content:
    return _image_query_function(inputs)
```

And add the audio branch immediately after the image branch:

```python
if inputs.audio_path or inputs.audio_content:
    return _audio_query_function(inputs)
```

- [ ] **Step 4: Run to verify it passes**

```bash
pytest tests/cli/test_query_function.py::test_audio_query_builds_multimodal_usermessage -v
```

Expected: `PASSED`

- [ ] **Step 5: Run full query function test suite**

```bash
pytest tests/cli/test_query_function.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git -C /Users/bianders/Brian_Code/conduit-project add \
    src/conduit/apps/cli/query/query_function.py \
    tests/cli/test_query_function.py
git -C /Users/bianders/Brian_Code/conduit-project commit -m "feat: add _audio_query_function and dispatch in default_query_function"
```

---

## Task 4: Add `--audio` flag, guards, and "transcribe this" default

**Fulfills:** AC4, AC5, AC6, AC7, AC8, AC12

**Files:**
- Modify: `src/conduit/apps/cli/commands/base_commands.py`
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py`
- Create: `tests/cli/test_audio_command.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/cli/test_audio_command.py`:

```python
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_audio_command.py -v
```

Expected: all `FAILED` — `--audio` option does not exist on the `query` command yet.

- [ ] **Step 3: Implement the flag in `base_commands.py`**

In `src/conduit/apps/cli/commands/base_commands.py`, add the following imports at the top of the file (alongside the existing `_CLIPBOARD_SENTINEL` block):

```python
_AUDIO_SENTINEL = "@mic"
```

Add these option decorators to the `query` command inside `_register_commands()`, after the existing `--image` option and before `@click.argument`:

```python
@click.option(
    "-A",
    "--audio",
    type=str,
    default=None,
    help='Path to a .wav or .mp3 audio file, or "@mic" to record from microphone.',
)
@click.option(
    "-s",
    "--save",
    type=str,
    default=None,
    help="Save response to file. Suppresses stdout output.",
)
@click.option(
    "--play",
    is_flag=True,
    default=False,
    help="Play audio response after generation. Requires audio output from model.",
)
```

Add `audio`, `save`, `play` to the `query` function signature:

```python
def query(
    ctx: click.Context,
    model: str | None,
    local: bool,
    raw: bool,
    temperature: float | None,
    chat: bool,
    append: str | None,
    citations: bool,
    search: bool,
    image: str | None,
    audio: str | None,
    save: str | None,
    play: bool,
    query_input: tuple[str, ...],
):
```

Add guards immediately after the existing `--image` guards, before unpacking `ctx.obj`:

```python
if audio and chat:
    raise click.UsageError("--audio cannot be used with --chat")
if audio and search:
    raise click.UsageError("--audio cannot be used with --search")
if audio and image:
    raise click.UsageError("--audio cannot be used with --image")
```

Add the audio resolution block after the existing image resolution block (after the `image_path`/`image_content` logic):

```python
# Resolve --audio into audio_path (file) or audio_content (@mic)
audio_path: str | None = None
audio_content_obj: AudioContent | None = None
if audio is not None:
    if audio == _AUDIO_SENTINEL:
        audio_content_obj = _resolve_mic_audio()
    else:
        p = Path(audio)
        if not p.exists():
            raise click.UsageError(f"--audio: file not found: {audio}")
        if not p.is_file():
            raise click.UsageError(f"--audio: not a file: {audio}")
        ext = p.suffix.lower()
        if ext not in {".wav", ".mp3"}:
            raise click.UsageError(
                f"--audio: unsupported audio format \"{ext}\" — use .wav or .mp3"
            )
        audio_path = audio
```

Add the "transcribe this" default after `query_input_str` is computed:

```python
# Default query for audio with no explicit prompt
if audio and not query_input_str:
    query_input_str = "transcribe this"
```

Pass the new args to `handlers.handle_query(...)`:

```python
handlers.handle_query(
    ...,
    audio_path=audio_path,
    audio_content=audio_content_obj,
    save=save,
    play=play,
)
```

Also add `AudioContent` to the `TYPE_CHECKING` imports at the top of the file:

```python
if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    from conduit.storage.repository.protocol import ConversationRepository
    from conduit.apps.cli.utils.printer import Printer
    from conduit.domain.message.message import ImageContent, AudioContent
```

- [ ] **Step 4: Add `_resolve_mic_audio()` and `AudioRecorder` to `base_commands.py`**

Add these before the `handlers = BaseHandlers()` line:

```python
class AudioRecorder:
    """
    Minimal mic recorder lifted from tap/scripts/record_cli.py.
    Uses pyaudio for capture, pydub for WAV→MP3 conversion.
    """

    def __init__(self):
        import pyaudio
        self.chunk = 1024
        self.format = pyaudio.paInt16
        self.channels = 1
        self.rate = 44100
        self.recording = False
        self.frames: list[bytes] = []
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def start_recording(self):
        import threading
        self.frames = []
        self.recording = True
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer=self.chunk,
        )

        def _record():
            while self.recording:
                try:
                    data = self.stream.read(self.chunk, exception_on_overflow=False)
                    self.frames.append(data)
                except Exception:
                    break

        t = threading.Thread(target=_record, daemon=True)
        t.start()

    def stop_recording(self):
        self.recording = False
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.audio.terminate()

    def get_mp3_bytes(self) -> bytes:
        """Convert captured frames to MP3 bytes via pydub (in-memory)."""
        import io
        import wave
        from pydub import AudioSegment

        wav_buf = io.BytesIO()
        wf = wave.open(wav_buf, "wb")
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.audio.get_sample_size(self.format))
        wf.setframerate(self.rate)
        wf.writeframes(b"".join(self.frames))
        wf.close()
        wav_buf.seek(0)

        mp3_buf = io.BytesIO()
        AudioSegment.from_wav(wav_buf).export(mp3_buf, format="mp3", bitrate="192k")
        return mp3_buf.getvalue()


def _resolve_mic_audio() -> AudioContent:
    """
    Records from microphone, stops on Enter, returns AudioContent (mp3).
    Raises click.UsageError for: SSH, missing deps, portaudio failure, empty recording.
    """
    import os

    if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
        raise click.UsageError("@mic not available over SSH")

    try:
        import pyaudio  # noqa: F401
    except ImportError:
        raise click.UsageError(
            "@mic requires pyaudio — brew install portaudio && pip install pyaudio"
        )
    try:
        from pydub import AudioSegment  # noqa: F401
    except ImportError:
        raise click.UsageError("@mic requires pydub — pip install pydub")

    from conduit.domain.message.message import AudioContent

    logger.info("@mic recording started")
    try:
        recorder = AudioRecorder()
        recorder.start_recording()
        click.echo("Recording... Press Enter to stop.")
        input()
        recorder.stop_recording()
    except OSError as e:
        raise click.UsageError(
            f"could not open microphone — is portaudio installed? ({e})"
        )

    data = recorder.get_mp3_bytes()
    if not data:
        raise click.UsageError("no audio recorded")

    logger.info("@mic recording stopped, %d bytes captured (mp3)", len(data))
    return AudioContent.from_bytes(data, format="mp3")
```

- [ ] **Step 5: Update `handle_query` signature in `base_handlers.py`**

In `src/conduit/apps/cli/handlers/base_handlers.py`, update the `handle_query` signature to add the three new parameters after `image_content`:

```python
@staticmethod
def handle_query(
    query_input: str,
    model: str,
    local: bool,
    raw: bool,
    temperature: float | None,
    chat: bool,
    append: str | None,
    verbosity: Verbosity,
    printer: Printer,
    query_function: CLIQueryFunctionProtocol,
    stdin: str | None,
    system_message: str = "",
    project_name: str = "",
    search: bool = False,
    citations: bool = False,
    image_path: str | None = None,
    image_content: ImageContent | None = None,
    audio_path: str | None = None,
    audio_content: AudioContent | None = None,
    save: str | None = None,
    play: bool = False,
) -> None:
```

Also add `AudioContent` to the `TYPE_CHECKING` block in `base_handlers.py`:

```python
if TYPE_CHECKING:
    from conduit.utils.progress.verbosity import Verbosity
    from conduit.domain.conversation.conversation import Conversation
    from conduit.apps.cli.utils.printer import Printer
    from conduit.domain.message.message import UserMessage, Message, ImageContent, AudioContent
    from conduit.storage.repository.protocol import AsyncSessionRepository
    from uuid import UUID
```

Pass the new fields through to `CLIQueryFunctionInputs` inside `handle_query`:

```python
inputs = CLIQueryFunctionInputs(
    query_input=query_input,
    printer=printer,
    context=context_text,
    append=append or "",
    system_message=system_message,
    project_name=project_name,
    search=search,
    cache=not local,
    local=local,
    preferred_model=model,
    verbose=verbosity,
    include_history=chat,
    temperature=temperature,
    client_params=client_params,
    image_path=image_path,
    image_content=image_content,
    audio_path=audio_path,
    audio_content=audio_content,
)
```

- [ ] **Step 6: Run to verify the tests pass**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_audio_command.py -v
```

Expected: all 6 tests `PASSED`

- [ ] **Step 7: Run full CLI test suite for regressions**

```bash
pytest tests/cli/ -v
```

Expected: all tests `PASSED`

- [ ] **Step 8: Commit**

```bash
git -C /Users/bianders/Brian_Code/conduit-project add \
    src/conduit/apps/cli/commands/base_commands.py \
    src/conduit/apps/cli/handlers/base_handlers.py \
    tests/cli/test_audio_command.py
git -C /Users/bianders/Brian_Code/conduit-project commit -m "feat: add --audio flag with @mic sentinel, guards, and transcribe-this default"
```

---

## Task 5: Implement `_save_response`, `_play_audio`, and post-response wiring

**Fulfills:** AC13, AC14, AC15, AC16

**Files:**
- Modify: `src/conduit/apps/cli/handlers/base_handlers.py`
- Modify: `tests/cli/test_audio_command.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/cli/test_audio_command.py`:

```python
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

    audio_file = tmp_path / "clip.mp3"
    audio_file.write_bytes(b"\xff\xfb" + b"\x00" * 16)
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
```

- [ ] **Step 2: Run to verify they fail**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_audio_command.py::test_save_response_writes_audio_bytes \
       tests/cli/test_audio_command.py::test_save_response_writes_text \
       tests/cli/test_audio_command.py::test_save_flag_suppresses_printer \
       tests/cli/test_audio_command.py::test_play_flag_autosaves_to_tmp \
       -v
```

Expected: all `FAILED` — `_save_response` and `_play_audio` do not exist; `handle_query` ignores `save` and `play`.

- [ ] **Step 3: Add `_save_response` and `_play_audio` to `base_handlers.py`**

Add these as module-level functions at the bottom of `base_handlers.py`, before the `class BaseHandlers` line (or after the class — either works, but before use):

```python
def _save_response(response: object, path: str) -> None:
    """
    Write response output to a file. Priority: audio > images > text.
    response is a Conversation; response.last is the AssistantMessage.
    """
    import base64
    from pathlib import Path

    last = response.last
    if getattr(last, "audio", None):
        data = base64.b64decode(last.audio.data)
        Path(path).write_bytes(data)
        logger.info("Saved audio response to %s", path)
    elif getattr(last, "images", None):
        data = base64.b64decode(last.images[0].b64_json)
        Path(path).write_bytes(data)
        logger.info("Saved image response to %s", path)
    else:
        Path(path).write_text(str(last))
        logger.info("Saved text response to %s", path)


def _play_audio(path: str) -> None:
    """Play an audio file via pydub. Requires simpleaudio or pyaudio as pydub backend."""
    from pydub import AudioSegment
    from pydub.playback import play

    logger.info("Playing audio from %s", path)
    audio = AudioSegment.from_file(path)
    play(audio)
```

- [ ] **Step 4: Wire `save`, `play`, and auto-save into `handle_query`**

In `src/conduit/apps/cli/handlers/base_handlers.py`, replace the display block at the end of `handle_query` (the part that currently reads):

```python
        # 4. Display
        if raw:
            printer.print_raw(response.content)
        else:
            printer.print_markdown(response.content)

        # 5. Citations
        BaseHandlers.handle_citations(response, citations=citations, raw=raw, printer=printer)
```

With:

```python
        # 4. Resolve effective save path
        from datetime import datetime
        last = response.last
        effective_save = save

        if getattr(last, "audio", None) and not save:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            effective_save = f"/tmp/conduit_audio_{ts}.{last.audio.format}"
            logger.info("Auto-saving audio to %s", effective_save)

        # 5. Save, play, or display
        if effective_save:
            _save_response(response, effective_save)
        else:
            if raw:
                printer.print_raw(response.content)
            else:
                printer.print_markdown(response.content)

        if play:
            _play_audio(effective_save)

        # 6. Citations
        BaseHandlers.handle_citations(response, citations=citations, raw=raw, printer=printer)
```

- [ ] **Step 5: Run to verify tests pass**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/test_audio_command.py -v
```

Expected: all tests `PASSED`

- [ ] **Step 6: Run full test suite**

```bash
pytest tests/cli/ -v
```

Expected: all tests `PASSED`

- [ ] **Step 7: Commit**

```bash
git -C /Users/bianders/Brian_Code/conduit-project add \
    src/conduit/apps/cli/handlers/base_handlers.py \
    tests/cli/test_audio_command.py
git -C /Users/bianders/Brian_Code/conduit-project commit -m "feat: add _save_response, _play_audio, and --save/--play wiring in handle_query"
```

---

## Task 6: Final regression sweep and smoke test

**Files:** No changes — verification only.

- [ ] **Step 1: Run full test suite (excluding regression tests that require live API)**

```bash
cd /Users/bianders/Brian_Code/conduit-project
pytest tests/cli/ tests/unit/ tests/storage/ -v
```

Expected: all tests `PASSED`

- [ ] **Step 2: Confirm `--help` shows new flags**

```bash
ask --help
```

Expected output includes:
```
  -A, --audio TEXT    Path to a .wav or .mp3 audio file, or "@mic" to record
  -s, --save TEXT     Save response to file. Suppresses stdout output.
  --play              Play audio response after generation.
```

- [ ] **Step 3: Smoke test the primary use case**

```bash
ask "summarize this podcast" --audio ~/recordings/people_inc.mp3 --model gemini
```

Expected: model response printed to terminal (no crash, no empty output).

- [ ] **Step 4: Confirm git log**

```bash
git -C /Users/bianders/Brian_Code/conduit-project log --oneline -6
```

Expected — five feature commits in order:
```
feat: add _save_response, _play_audio, and --save/--play wiring in handle_query
feat: add --audio flag with @mic sentinel, guards, and transcribe-this default
feat: add _audio_query_function and dispatch in default_query_function
feat: add audio_path/audio_content to CLIQueryFunctionInputs with mutual-exclusion guards
feat: add AudioContent.from_bytes()
```
