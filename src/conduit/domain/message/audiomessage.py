"""
Factory methods for creating Messages from audio files or base64 data.
"""

import base64
import re
from pathlib import Path


def is_base64_simple(s):
    """
    Simple validation for base64 strings.
    """
    return bool(re.match(r"^[A-Za-z0-9+/]*={0,2}$", s)) and len(s) % 4 == 0


def from_audio_file(audio_file: str | Path, text_content: str, role: str = "user"):
    raise NotImplementedError("Not implemented yet")
    # Do all the file processing here
    audio_file = Path(audio_file)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file {audio_file} does not exist.")

    audio_content = _convert_audio_to_base64(audio_file)
    format = audio_file.suffix.lower()[1:]  # "mp3" or "wav"

    return Message(
        role=role,
        content=[audio_content, text_content],
        text_content=text_content,
        audio_content=audio_content,
        format=format,
    )


def from_base64(
    audio_content: str,
    text_content: str,
    format: str = "mp3",
    role: str = "user",
):
    raise NotImplementedError("Not implemented yet")
    # Validate base64
    if not is_base64_simple(audio_content):
        raise ValueError("Invalid base64 image data")
    # TBD: optional conversion logic
    # Construct the AudioMessage object
    return cls(
        text_content=text_content,
        audio_content=audio_content,
        format=format,
        role=role,
        content=[audio_content, text_content],
    )


def _convert_audio_to_base64(file_path: Path) -> str:
    """
    Convert the audio file to base64 string.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
