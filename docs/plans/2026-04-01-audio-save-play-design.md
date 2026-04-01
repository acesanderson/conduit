# Audio Input, `--save`, and `--play` Design

## 1. Goal

Add `--audio` to `conduit query` / `ask` for sending audio files or live mic recordings to the LLM. Add `--save` to write any response (text, audio, image) to a file instead of stdout. Add `--play` to play generated audio immediately after the model responds.

## 2. Constraints and Non-Goals

**In scope:**
- `--audio <path>` accepting `.wav` and `.mp3` files only
- `--audio @mic` for live microphone recording (pyaudio + pydub, lifted from tap)
- `--save <path>` on `conduit query` / `ask` writing text, audio, or image output to a file (suppresses stdout)
- `--play` flag for playing audio responses via pydub
- Auto-save to `/tmp/conduit_audio_{timestamp}.{format}` when `--play` is set and `--save` is not
- `AudioContent.from_bytes()` added to `message.py` (mirrors `ImageContent`)

**Explicitly out of scope:**
- Audio transcoding (`.m4a`, `.ogg`, `.flac` → `.mp3` / `.wav`) — unsupported extensions raise `UsageError`
- Combining `--audio` and `--image` in a single query
- `--save -` (stdout binary pipe) for audio output
- Audio chunking for long files (e.g., hour-long podcasts that exceed model context)
- Multiple audio files per query
- Preview playback of `@mic` recording before sending
- Audio output format conversion (model returns WAV → user wants MP3)
- `--play` for text-to-speech streaming during generation
- Model capability checking — user is responsible for choosing an audio-capable model
- Adding `--save` or `--play` to `imagegen` (already has `--save`; out of scope here)

## 3. Interface Contracts

### `message.py` — new method

```python
class AudioContent(BaseModel):
    @classmethod
    def from_bytes(cls, data: bytes, format: Literal["wav", "mp3"] = "mp3") -> AudioContent:
        """Encode raw bytes as base64. Used for @mic recordings."""
        audio_b64 = base64.b64encode(data).decode("utf-8")
        return cls(data=audio_b64, format=format)
```

### `query_function.py` — updated dataclass and new query function

```python
@dataclass
class CLIQueryFunctionInputs:
    # existing fields unchanged ...
    audio_path: str | None = None        # mutually exclusive with audio_content
    audio_content: AudioContent | None = None  # mutually exclusive with audio_path

    def __post_init__(self):
        # existing image check unchanged
        if self.audio_path is not None and self.audio_content is not None:
            raise ValueError("Only one of audio_path or audio_content may be set, not both.")
        if (self.audio_path or self.audio_content) and (self.image_path or self.image_content):
            raise ValueError("--audio and --image cannot be used together.")


def _audio_query_function(inputs: CLIQueryFunctionInputs) -> Conversation:
    """
    Builds UserMessage([TextContent, AudioContent]) and routes through pipe_sync.
    History loading is disabled — --chat is blocked upstream.
    combined_query will already contain the "transcribe this" default if no query was given.
    """
    ...
```

`default_query_function` dispatch order:
1. `inputs.search` → `_search_query_function`
2. `inputs.image_path or inputs.image_content` → `_image_query_function`
3. `inputs.audio_path or inputs.audio_content` → `_audio_query_function`
4. else → default text path

### `base_commands.py` — new sentinel, resolver, flags

```python
_AUDIO_SENTINEL = "@mic"

def _resolve_mic_audio() -> AudioContent:
    """
    Records from microphone (pyaudio), stops on Enter, converts to MP3 (pydub),
    returns AudioContent. Raises click.UsageError on: SSH session, missing pyaudio/pydub,
    portaudio unavailable, or empty recording.
    """
    ...

# New flags on the query command:
@click.option("-A", "--audio", type=str, default=None,
    help='Path to a .wav or .mp3 audio file, or "@mic" to record from microphone.')
@click.option("-s", "--save", type=str, default=None,
    help="Save response to file. Suppresses stdout. Auto-saves audio to /tmp/ if --play is set.")
@click.option("--play", is_flag=True, default=False,
    help="Play audio response immediately. Requires audio output from model.")
```

**"transcribe this" default** — applied in the command layer, before `handle_query`:
```python
# In base_commands.query():
if audio and not query_input_str:
    query_input_str = "transcribe this"
```

### `base_handlers.py` — updated `handle_query` signature

```python
@staticmethod
def handle_query(
    ...,
    audio_path: str | None = None,
    audio_content: AudioContent | None = None,
    save: str | None = None,
    play: bool = False,
) -> None: ...
```

**Save dispatch** (inside `handle_query`, after `query_function` returns):

```python
def _save_response(response: Conversation, path: str) -> None:
    """Priority: audio > images > text content."""
    last = response.last
    if last.audio:
        data = base64.b64decode(last.audio.data)
        Path(path).write_bytes(data)
        logger.info("Saved audio response to %s", path)
    elif last.images:
        data = base64.b64decode(last.images[0].b64_json)
        Path(path).write_bytes(data)
        logger.info("Saved image response to %s", path)
    else:
        Path(path).write_text(str(last))
        logger.info("Saved text response to %s", path)
```

**Auto-save + play logic** (inside `handle_query`):

```python
# After query_function returns `response`:
last = response.last
effective_save = save

if last.audio and not save:
    # Auto-save to /tmp/ so --play has a file to play
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    effective_save = f"/tmp/conduit_audio_{ts}.{last.audio.format}"
    logger.info("Auto-saving audio to %s", effective_save)

if effective_save:
    _save_response(response, effective_save)
elif not play:
    # Only print if not saving and not playing
    if raw:
        printer.print_raw(response.content)
    else:
        printer.print_markdown(response.content)

if play:
    _play_audio(effective_save)
```

**Play helper:**

```python
def _play_audio(path: str) -> None:
    from pydub import AudioSegment
    from pydub.playback import play
    logger.info("Playing audio from %s", path)
    audio = AudioSegment.from_file(path)
    play(audio)
```

## 4. Acceptance Criteria

All assertions are structural (no live API required):

1. `CLIQueryFunctionInputs(audio_path="x", audio_content=AudioContent(...))` raises `ValueError`.
2. `CLIQueryFunctionInputs(audio_path="x", image_path="y")` raises `ValueError`.
3. `CLIQueryFunctionInputs(audio_content=..., image_content=...)` raises `ValueError`.
4. `runner.invoke(cli, ["query", "--audio", "f.mp3", "--chat", "q"])` exits non-zero with `"--audio cannot be used with --chat"` in output.
5. `runner.invoke(cli, ["query", "--audio", "f.mp3", "--search", "q"])` exits non-zero with `"--audio cannot be used with --search"` in output.
6. `runner.invoke(cli, ["query", "--audio", "f.mp3", "--image", "f.png", "q"])` exits non-zero with `"--audio cannot be used with --image"` in output.
7. `runner.invoke(cli, ["query", "--audio", "/nonexistent.mp3", "q"])` exits non-zero with `"not found"` or `"does not exist"` in output.
8. `runner.invoke(cli, ["query", "--audio", "f.m4a", "q"])` exits non-zero with `"unsupported audio format"` in output.
9. `AudioContent.from_bytes(b"data")` produces `AudioContent(format="mp3")` (default format).
10. `AudioContent.from_bytes(b"data", format="wav")` produces `AudioContent(format="wav")`.
11. `_audio_query_function` with non-empty `combined_query` builds `UserMessage([TextContent(text=combined_query), AudioContent(...)])` — TextContent is index 0, AudioContent is index 1.
12. When `--audio f.mp3` with no positional args and no stdin, the `query_input_str` passed to `handle_query` equals `"transcribe this"`.
13. When `--save out.txt` is set, `printer.print_markdown` and `printer.print_raw` are not called.
14. `_save_response(response_with_audio, "out.mp3")` writes bytes to `out.mp3` (file exists and is non-empty).
15. `_save_response(response_with_text, "out.txt")` writes the text of `response.last` to `out.txt`.
16. When `--play` is set and `--save` is not, `effective_save` resolves to a path under `/tmp/` with the correct extension matching `last.audio.format`.

## 5. Error Handling / Failure Modes

| Condition | Layer | Behavior |
|-----------|-------|----------|
| `--audio /nonexistent.mp3` | Command | `click.UsageError: --audio: file not found: /nonexistent.mp3` |
| `--audio file.m4a` (unsupported format) | Command | `click.UsageError: --audio: unsupported format ".m4a" — use .wav or .mp3` |
| `--audio` + `--chat` | Command | `click.UsageError: --audio cannot be used with --chat` |
| `--audio` + `--search` | Command | `click.UsageError: --audio cannot be used with --search` |
| `--audio` + `--image` | Command | `click.UsageError: --audio cannot be used with --image` |
| `@mic` over SSH | `_resolve_mic_audio` | `click.UsageError: @mic not available over SSH` |
| `pyaudio` not installed | `_resolve_mic_audio` | `click.UsageError: @mic requires pyaudio — install with: brew install portaudio && pip install pyaudio` |
| `pydub` not installed | `_resolve_mic_audio` | `click.UsageError: @mic requires pydub — install with: pip install pydub` |
| `portaudio` not available (pyaudio OSError) | `_resolve_mic_audio` | `click.UsageError: could not open microphone — is portaudio installed? (brew install portaudio)` |
| Empty recording (Enter pressed immediately) | `_resolve_mic_audio` | `click.UsageError: no audio recorded` |
| `audio_path` + `audio_content` both set | `CLIQueryFunctionInputs.__post_init__` | `ValueError` |
| `--audio` + `--image` both set | `CLIQueryFunctionInputs.__post_init__` | `ValueError` (defense in depth; also caught at command layer) |
| `--play` with no audio in response | `_play_audio` | Surfaces as `AttributeError` or `pydub` error — user's responsibility |
| `--save` path directory does not exist | `_save_response` | `FileNotFoundError` surfaces naturally |
| `/tmp/` not writable | Auto-save | `PermissionError` surfaces naturally |

## 6. Code Example

The `@mic` resolver follows the exact pattern of `_resolve_clipboard_image`:

```python
_AUDIO_SENTINEL = "@mic"

def _resolve_mic_audio() -> AudioContent:
    import os
    if "SSH_CLIENT" in os.environ or "SSH_TTY" in os.environ:
        raise click.UsageError("@mic not available over SSH")

    try:
        import pyaudio
    except ImportError:
        raise click.UsageError(
            "@mic requires pyaudio — brew install portaudio && pip install pyaudio"
        )
    try:
        from pydub import AudioSegment
    except ImportError:
        raise click.UsageError("@mic requires pydub — pip install pydub")

    try:
        recorder = AudioRecorder()
        recorder.start_recording()   # prints "Recording... Press Enter to stop."
        input()
        recorder.stop_recording()
    except OSError as e:
        raise click.UsageError(f"could not open microphone — is portaudio installed? ({e})")

    data = recorder.get_mp3_bytes()  # in-memory pydub conversion
    if not data:
        raise click.UsageError("no audio recorded")

    logger.info("@mic captured %d bytes (mp3)", len(data))
    return AudioContent.from_bytes(data, format="mp3")
```

## 7. Domain Language

| Term | Definition |
|------|------------|
| `AudioContent` | Pydantic model for LLM-bound audio input: base64-encoded bytes + format (`"wav"` or `"mp3"`) |
| `AudioOutput` | Pydantic model for LLM-generated audio response — already exists in `message.py` |
| `AudioRecorder` | Internal class (lifted from tap's `record_cli.py`) that wraps pyaudio recording + pydub MP3 conversion |
| `@mic` | Sentinel string for live microphone recording; resolved in the command layer before reaching the handler |
| `_resolve_mic_audio()` | Command-layer function that runs the full record flow and returns `AudioContent` |
| `_audio_query_function` | Query function variant that builds `UserMessage([TextContent, AudioContent])` and routes via `pipe_sync` |
| `_save_response` | Handler-layer function that dispatches file writes by output type (audio > images > text) |
| `effective_save` | The resolved save path — either user-supplied via `--save` or auto-generated under `/tmp/` for `--play` |
| `auto-save` | Writing audio output to `/tmp/conduit_audio_{timestamp}.{format}` when `--play` is set and `--save` is not |

## 8. Invalid State Transitions

These must raise errors — not silently degrade:

1. `CLIQueryFunctionInputs` with `audio_path != None` AND `audio_content != None` → `ValueError` in `__post_init__`
2. `CLIQueryFunctionInputs` with any audio field set AND any image field set → `ValueError` in `__post_init__`
3. `AudioContent.from_file()` called on a path with extension not in `{"wav", "mp3"}` → must raise before Pydantic validation failure surfaces as a cryptic error; caller (`base_commands.py`) is responsible for this guard
4. `_resolve_mic_audio()` called in an SSH session → `click.UsageError` (not a silent no-op)
5. `_resolve_mic_audio()` producing zero bytes → `click.UsageError` (not silently sending empty audio to the model)
6. `--play` flag set while `effective_save` is `None` → must not reach `_play_audio(None)` — auto-save must always produce a path before play is invoked

## Observability

Log statements required (mirrors existing image logging in `query_function.py`):

- `logger.info("Audio query: loading %s (format: %s)", audio_path, fmt)` — in `_audio_query_function` before `from_file()`
- `logger.info("@mic recording started")` — in `_resolve_mic_audio`
- `logger.info("@mic recording stopped, %d bytes captured", len(data))` — in `_resolve_mic_audio`
- `logger.info("Auto-saving audio to %s", effective_save)` — in `handle_query` auto-save branch
- `logger.info("Saving response to %s", path)` — in `_save_response`
- `logger.info("Playing audio from %s", path)` — in `_play_audio`
