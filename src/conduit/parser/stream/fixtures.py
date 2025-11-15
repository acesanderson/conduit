"""
Test data and mock stream objects for stream parser testing.

Provides example streaming responses that mimic behavior of real LLM providers,
with various edge cases for robust parser testing.
"""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class MockUsage:
    """Mock usage data matching provider usage objects."""

    prompt_tokens: int
    completion_tokens: int


@dataclass
class MockChunk:
    """Mock chunk matching StreamChunk protocol."""

    content: str | None = None
    usage: MockUsage | None = None


class MockStream:
    """
    Mock stream object that matches SyncStream protocol.

    Wraps a list of chunks and provides iterator interface for testing.
    """

    def __init__(self, chunks: list[MockChunk]):
        self.chunks = chunks
        self._closed = False

    def __iter__(self) -> Iterator[MockChunk]:
        """Iterate over chunks."""
        for chunk in self.chunks:
            if self._closed:
                break
            yield chunk

    def close(self) -> None:
        """Close the stream."""
        self._closed = True


# === Simple Cases ===

SIMPLE_TEXT_CHUNKS = [
    MockChunk(content="Hello, "),
    MockChunk(content="how "),
    MockChunk(content="can "),
    MockChunk(content="I "),
    MockChunk(content="help?"),
    MockChunk(content=None, usage=MockUsage(10, 5)),
]

SIMPLE_XML_CHUNKS = [
    MockChunk(content="Let me help. "),
    MockChunk(content="<function_calls>"),
    MockChunk(content="<invoke name='file_read'>"),
    MockChunk(content="<parameters>"),
    MockChunk(content="<parameter name='path'>"),
    MockChunk(content="/home/test.py"),
    MockChunk(content="</parameter>"),
    MockChunk(content="</parameters>"),
    MockChunk(content="</invoke>"),
    MockChunk(content="</function_calls>"),
    MockChunk(content=" Done."),
    MockChunk(content=None, usage=MockUsage(20, 15)),
]

SIMPLE_JSON_CHUNKS = [
    MockChunk(content="Here's the data: "),
    MockChunk(content='{"name": '),
    MockChunk(content='"test", '),
    MockChunk(content='"value": '),
    MockChunk(content="42}"),
    MockChunk(content=" Hope that helps!"),
    MockChunk(content=None, usage=MockUsage(15, 10)),
]


# === Edge Cases ===

# XML split mid-tag
SPLIT_XML_TAG_CHUNKS = [
    MockChunk(content="Text <func"),
    MockChunk(content="tion_calls><inv"),
    MockChunk(content="oke name='test'/>"),
    MockChunk(content="</function_calls>"),
    MockChunk(content=None, usage=MockUsage(10, 8)),
]

# Multiple XML objects
MULTI_XML_CHUNKS = [
    MockChunk(
        content="First: <function_calls><invoke name='first'/></function_calls>\n"
    ),
    MockChunk(
        content="Second: <function_calls><invoke name='second'/></function_calls>\n"
    ),
    MockChunk(content="Done."),
    MockChunk(content=None, usage=MockUsage(25, 20)),
]

# Nested XML
NESTED_XML_CHUNKS = [
    MockChunk(content="<function_calls>"),
    MockChunk(content="<invoke name='outer'>"),
    MockChunk(content="<parameters>"),
    MockChunk(content="<parameter name='nested'>"),
    MockChunk(content="<inner>value</inner>"),
    MockChunk(content="</parameter>"),
    MockChunk(content="</parameters>"),
    MockChunk(content="</invoke>"),
    MockChunk(content="</function_calls>"),
    MockChunk(content=None, usage=MockUsage(15, 12)),
]

# False positive - XML in text
TRICKY_XML_CHUNKS = [
    MockChunk(content='Someone said "<function_calls>" but it\'s fake.\n'),
    MockChunk(
        content="Real one: <function_calls><invoke name='real'/></function_calls>"
    ),
    MockChunk(content=None, usage=MockUsage(20, 15)),
]

# JSON with nested objects
NESTED_JSON_CHUNKS = [
    MockChunk(content='{"outer": {'),
    MockChunk(content='"inner": {'),
    MockChunk(content='"deep": "value"'),
    MockChunk(content="}}}"),
    MockChunk(content=None, usage=MockUsage(10, 8)),
]

# JSON with tricky strings
TRICKY_JSON_CHUNKS = [
    MockChunk(content='{"string": "has } brace", '),
    MockChunk(content='"escaped": "has \\" quote"}'),
    MockChunk(content=None, usage=MockUsage(12, 10)),
]

# No structured content
PLAIN_TEXT_CHUNKS = [
    MockChunk(content="Just "),
    MockChunk(content="some "),
    MockChunk(content="plain "),
    MockChunk(content="text."),
    MockChunk(content=None, usage=MockUsage(8, 4)),
]

# Empty stream
EMPTY_CHUNKS = [
    MockChunk(content=None, usage=MockUsage(0, 0)),
]


# === Test Stream Factory ===


def create_test_stream(chunks: list[MockChunk]) -> MockStream:
    """Factory function to create test streams."""
    return MockStream(chunks)


# === Pre-built Test Streams ===

SIMPLE_TEXT_STREAM = create_test_stream(SIMPLE_TEXT_CHUNKS)
SIMPLE_XML_STREAM = create_test_stream(SIMPLE_XML_CHUNKS)
SIMPLE_JSON_STREAM = create_test_stream(SIMPLE_JSON_CHUNKS)
SPLIT_XML_TAG_STREAM = create_test_stream(SPLIT_XML_TAG_CHUNKS)
MULTI_XML_STREAM = create_test_stream(MULTI_XML_CHUNKS)
NESTED_XML_STREAM = create_test_stream(NESTED_XML_CHUNKS)
TRICKY_XML_STREAM = create_test_stream(TRICKY_XML_CHUNKS)
NESTED_JSON_STREAM = create_test_stream(NESTED_JSON_CHUNKS)
TRICKY_JSON_STREAM = create_test_stream(TRICKY_JSON_CHUNKS)
PLAIN_TEXT_STREAM = create_test_stream(PLAIN_TEXT_CHUNKS)
EMPTY_STREAM = create_test_stream(EMPTY_CHUNKS)
