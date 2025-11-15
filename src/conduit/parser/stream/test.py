# test_stream.py
"""
Tests for sync and async stream parsers.

This test script will fail until the implementation logic in parsers.py
is completed according to the agreed-upon architecture.

To run:
pip install pytest pytest-asyncio
pytest
"""

import asyncio
import pytest
from collections.abc import AsyncIterator

# DUT (Device Under Test)
from conduit.parser.stream.parsers import XMLStreamParser, JSONStreamParser

# Dependencies
from conduit.parser.stream.protocol import StreamChunk
from conduit.parser.stream.fixtures import MockChunk, TestStream, MockUsage

# Import raw chunk lists
from conduit.parser.stream.fixtures import (
    SIMPLE_XML_CHUNKS,
    SIMPLE_JSON_CHUNKS,
    SPLIT_XML_TAG_CHUNKS,
    TRICKY_JSON_CHUNKS,
    PLAIN_TEXT_CHUNKS,
    MULTI_XML_CHUNKS,
    NESTED_XML_CHUNKS,
    TRICKY_XML_CHUNKS,  # Used for false positive
)


# === Mock Async Stream ===


class MockAsyncStream:
    """Minimal mock stream object that matches AsyncStream protocol."""

    def __init__(self, chunks: list[MockChunk]):
        self.chunks = chunks
        self._closed = False

    async def __aiter__(self) -> AsyncIterator[StreamChunk]:
        """Async iterate over chunks."""
        for chunk in self.chunks:
            if self._closed:
                break
            await asyncio.sleep(0)
            yield chunk

    async def close(self) -> None:
        """Close the stream."""
        self._closed = True


# === Test Suites ===


class TestXMLParserSync:
    """Synchronous XML parser tests."""

    def test_simple_xml(self):
        """Test parsing a simple XML object from a sync stream."""
        stream = TestStream(SIMPLE_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        assert not stream._closed
        text, xml_obj, full_content = parser.parse()

        expected_text = "Let me help.  Done."
        expected_xml = (
            "<function_calls>"
            "<invoke name='file_read'>"
            "<parameters>"
            "<parameter name='path'>"
            "/home/test.py"
            "</parameter>"
            "</parameters>"
            "</invoke>"
            "</function_calls>"
        )

        assert text == expected_text
        assert xml_obj == expected_xml
        assert "Let me help. <function_calls>" in full_content
        assert stream._closed is True

    def test_split_tag(self):
        """Test XML parser with a tag split across chunks."""
        stream = TestStream(SPLIT_XML_TAG_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = parser.parse()

        expected_xml = "<function_calls><invoke name='test'/></function_calls>"
        assert text == "Text "
        assert xml_obj == expected_xml
        assert stream._closed is True

    def test_no_match(self):
        """Test XML parser on plain text, expecting no match."""
        stream = TestStream(PLAIN_TEXT_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = parser.parse()

        expected_content = "Just some plain text."
        assert text == expected_content
        assert xml_obj is None
        assert full_content == expected_content
        assert stream._closed is True

    def test_multiple_xml_objects(self):
        """Test that parser finds FIRST XML object only."""
        stream = TestStream(MULTI_XML_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = parser.parse()

        expected_xml = "<function_calls><invoke name='first'/></function_calls>"
        expected_text = (
            "First: \n"
            "Second: <function_calls><invoke name='second'/></function_calls>\n"
            "Done."
        )

        assert xml_obj == expected_xml
        assert text == expected_text
        assert stream._closed

    def test_nested_xml(self):
        """Test nested XML with proper nesting depth tracking."""
        stream = TestStream(NESTED_XML_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = parser.parse()

        assert xml_obj is not None
        assert "<invoke name='outer'>" in xml_obj
        assert "<inner>value</inner>" in xml_obj
        assert "<parameter name='nested'>" in xml_obj
        assert xml_obj.count("<invoke") == xml_obj.count("</invoke>")
        assert xml_obj.count("<parameter") == xml_obj.count("</parameter>")
        assert stream._closed

    def test_incomplete_xml(self):
        """Test XML without closing tag."""
        incomplete_chunks = [
            MockChunk(content="<function_calls><invoke name='test'>"),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = TestStream(incomplete_chunks)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = parser.parse()

        assert xml_obj is None
        assert text == full_content
        assert text == "<function_calls><invoke name='test'>"
        assert stream._closed

    def test_false_positive_xml(self):
        """Test text mentioning XML tags but not valid/balanced XML."""
        # TRICKY_XML_CHUNKS contains a fake tag before a real one
        stream = TestStream(TRICKY_XML_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = parser.parse()

        # Should skip the fake one and find the real one
        assert xml_obj is not None
        assert "<invoke name='real'/>" in xml_obj
        assert 'Someone said "<function_calls>" but it\'s fake.\nReal one: ' in text
        assert stream._closed


class TestJSONParserSync:
    """Synchronous JSON parser tests."""

    def test_simple_json(self):
        """Test parsing a simple JSON object from a sync stream."""
        stream = TestStream(SIMPLE_JSON_CHUNKS)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = parser.parse()

        expected_text = "Here's the data:  Hope that helps!"
        expected_json = {"name": "test", "value": 42}

        assert text == expected_text
        assert json_obj == expected_json
        assert stream._closed is True

    def test_tricky_json(self):
        """Test JSON parser with braces and quotes in strings."""
        stream = TestStream(TRICKY_JSON_CHUNKS)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = parser.parse()

        expected_json = {"string": "has } brace", "escaped": 'has " quote'}
        assert text == ""
        assert json_obj == expected_json
        assert stream._closed is True

    def test_no_match_json(self):
        """Test JSON parser on plain text, expecting no match."""
        stream = TestStream(PLAIN_TEXT_CHUNKS)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = parser.parse()

        expected_content = "Just some plain text."
        assert text == expected_content
        assert json_obj is None
        assert stream._closed is True

    def test_incomplete_json(self):
        """Test JSON with unbalanced braces."""
        malformed_chunks = [
            MockChunk(content='{"key": "value"'),  # Missing closing }
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = TestStream(malformed_chunks)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = parser.parse()

        assert json_obj is None
        assert text == full_content
        assert stream._closed

    def test_false_positive_json(self):
        """Test text with braces that isn't valid JSON."""
        false_positive_chunks = [
            MockChunk(content="Someone wrote { and } in text"),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = TestStream(false_positive_chunks)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = parser.parse()

        assert json_obj is None
        assert text == full_content
        assert stream._closed


class TestXMLParserAsync:
    """Asynchronous XML parser tests."""

    @pytest.mark.asyncio
    async def test_async_simple_xml(self):
        stream = MockAsyncStream(SIMPLE_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        assert not stream._closed
        text, xml_obj, full_content = await parser.parse_async()

        expected_text = "Let me help.  Done."
        assert text == expected_text
        assert xml_obj is not None
        assert "<invoke name='file_read'>" in xml_obj
        assert stream._closed is True

    @pytest.mark.asyncio
    async def test_async_split_xml(self):
        stream = MockAsyncStream(SPLIT_XML_TAG_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = await parser.parse_async()

        expected_xml = "<function_calls><invoke name='test'/></function_calls>"
        assert text == "Text "
        assert xml_obj == expected_xml
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_no_match_xml(self):
        stream = MockAsyncStream(PLAIN_TEXT_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = await parser.parse_async()

        assert xml_obj is None
        assert text == "Just some plain text."
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_multiple_xml_objects(self):
        stream = MockAsyncStream(MULTI_XML_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = await parser.parse_async()

        assert xml_obj is not None
        assert "<invoke name='first'/>" in xml_obj
        assert "<invoke name='second'/>" in text
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_nested_xml(self):
        stream = MockAsyncStream(NESTED_XML_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = await parser.parse_async()

        assert xml_obj is not None
        assert "<invoke name='outer'>" in xml_obj
        assert "<inner>value</inner>" in xml_obj
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_incomplete_xml(self):
        incomplete_chunks = [
            MockChunk(content="<function_calls>"),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockAsyncStream(incomplete_chunks)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = await parser.parse_async()

        assert xml_obj is None
        assert text == full_content
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_false_positive_xml(self):
        stream = MockAsyncStream(TRICKY_XML_CHUNKS)
        parser = XMLStreamParser(stream)
        text, xml_obj, full_content = await parser.parse_async()

        assert xml_obj is not None
        assert "<invoke name='real'/>" in xml_obj
        assert 'Someone said "<function_calls>"' in text
        assert stream._closed


class TestJSONParserAsync:
    """Asynchronous JSON parser tests."""

    @pytest.mark.asyncio
    async def test_async_simple_json(self):
        stream = MockAsyncStream(SIMPLE_JSON_CHUNKS)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = await parser.parse_async()

        expected_text = "Here's the data:  Hope that helps!"
        expected_json = {"name": "test", "value": 42}

        assert text == expected_text
        assert json_obj == expected_json
        assert stream._closed is True

    @pytest.mark.asyncio
    async def test_async_tricky_json(self):
        stream = MockAsyncStream(TRICKY_JSON_CHUNKS)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = await parser.parse_async()

        expected_json = {"string": "has } brace", "escaped": 'has " quote'}
        assert text == ""
        assert json_obj == expected_json
        assert stream._closed is True

    @pytest.mark.asyncio
    async def test_async_no_match_json(self):
        stream = MockAsyncStream(PLAIN_TEXT_CHUNKS)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = await parser.parse_async()

        assert json_obj is None
        assert text == "Just some plain text."
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_incomplete_json(self):
        incomplete_chunks = [
            MockChunk(content='{"key": "value"'),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockAsyncStream(incomplete_chunks)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = await parser.parse_async()

        assert json_obj is None
        assert text == full_content
        assert stream._closed

    @pytest.mark.asyncio
    async def test_async_false_positive_json(self):
        false_positive_chunks = [
            MockChunk(content="Someone wrote { and } in text"),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockAsyncStream(false_positive_chunks)
        parser = JSONStreamParser(stream)
        text, json_obj, full_content = await parser.parse_async()

        assert json_obj is None
        assert text == full_content
        assert stream._closed
