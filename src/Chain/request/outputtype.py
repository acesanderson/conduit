"""
Primitives for various types of llm outputs.
- completion: get a TextMessage (default)
- imagegen: get an ImageMessage
- tts: get an AudioMessage
"""

from typing import Literal

OutputType = Literal["text", "image", "audio"]
