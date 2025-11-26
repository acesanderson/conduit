"""
Tests for sync and async stream parsers.

Tests both modes of operation:
- Token-efficient mode (close_on_match=True): Closes stream after finding object
- Complete mode (close_on_match=False): Consumes entire stream

To run:
    pip install pytest pytest-asyncio
    pytest test_stream.py
"""

import asyncio
import pytest
from collections.abc import AsyncIterator

# DUT (Device Under Test)
from conduit.parser.stream.parsers import XMLStreamParser
from conduit.parser.stream.parsers import JSONStreamParser

# Dependencies
from conduit.parser.stream.protocol import StreamChunk
from fixtures import MockChunk, MockStream, MockUsage

# Import raw chunk lists
from fixtures import (
    SIMPLE_XML_CHUNKS,
    SIMPLE_JSON_CHUNKS,
    SPLIT_XML_TAG_CHUNKS,
    TRICKY_JSON_CHUNKS,
    PLAIN_TEXT_CHUNKS,
    MULTI_XML_CHUNKS,
    NESTED_XML_CHUNKS,
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


# === Sync Tests: XML (Token-Efficient Mode) ===


class TestXMLParserSync:
    """Synchronous XML parser tests with early termination (token-efficient mode)."""

    def test_simple_xml(self):
        """Test parsing simple XML with early termination."""
        stream = MockStream(SIMPLE_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        assert not stream._closed
        text, xml_obj, full_content = parser.parse(close_on_match=True)

        # Stream closed after XML found, so " Done." not received
        expected_text = "Let me help. "
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
        assert " Done." not in full_content  # Stream closed before this arrived
        assert stream._closed is True

    def test_split_tag(self):
        """Test XML parser with tag split across chunks."""
        stream = MockStream(SPLIT_XML_TAG_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=True)

        expected_xml = "<function_calls><invoke name='test'/></function_calls>"
        assert text == "Text "
        assert xml_obj == expected_xml
        assert (
            full_content
            == "Text <function_calls><invoke name='test'/></function_calls>"
        )
        assert stream._closed is True

    def test_no_match(self):
        """Test XML parser on plain text (no match, full stream consumed)."""
        stream = MockStream(PLAIN_TEXT_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=True)

        expected_content = "Just some plain text."
        assert text == expected_content
        assert xml_obj is None
        assert full_content == expected_content
        assert stream._closed is True

    def test_multiple_objects(self):
        """Test that parser finds FIRST object only and closes early."""
        stream = MockStream(MULTI_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=True)

        expected_xml = "<function_calls><invoke name='first'/></function_calls>"

        assert xml_obj == expected_xml
        assert "<invoke name='first'/>" in xml_obj
        # Stream closed after first object, so second object not received
        assert "<invoke name='second'/>" not in full_content
        assert stream._closed

    def test_nested_xml(self):
        """Test nested XML with proper depth tracking."""
        stream = MockStream(NESTED_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=True)

        # Should extract complete outer XML including nested content
        assert xml_obj is not None
        assert xml_obj.startswith("<function_calls>")
        assert xml_obj.endswith("</function_calls>")
        assert "<invoke name='outer'>" in xml_obj
        assert "<parameters>" in xml_obj
        assert "<parameter name='nested'>" in xml_obj
        assert "<inner>value</inner>" in xml_obj

        # Verify it's valid XML by attempting to parse it
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(xml_obj)
            assert root.tag == "function_calls"
        except ET.ParseError:
            pytest.fail("Extracted XML is not valid/balanced")

        assert stream._closed

    def test_incomplete_xml(self):
        """Test XML without closing tag."""
        incomplete_chunks = [
            MockChunk(content="<function_calls><invoke name='test'>"),
            # Missing </invoke></function_calls>
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockStream(incomplete_chunks)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=True)

        # Parser should handle gracefully
        assert xml_obj is None
        assert text == full_content
        assert stream._closed

    def test_false_positive_xml(self):
        """Test text mentioning XML tags but not actual XML."""
        false_chunks = [
            MockChunk(content='Someone mentioned "<function_calls>" in chat.'),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockStream(false_chunks)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=True)

        # Should not match unbalanced/quoted XML
        assert xml_obj is None
        assert stream._closed


# === Sync Tests: XML (Complete Mode) ===


class TestXMLParserSyncFullStream:
    """Synchronous XML parser tests with full stream consumption."""

    def test_simple_xml_full_stream(self):
        """Test parsing with full stream consumption."""
        stream = MockStream(SIMPLE_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=False)

        # With close_on_match=False, we get everything
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
        assert "</function_calls> Done." in full_content
        assert stream._closed is True

    def test_multiple_objects_full_stream(self):
        """Test multiple objects with full stream consumption."""
        stream = MockStream(MULTI_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = parser.parse(close_on_match=False)

        # Parser extracts FIRST object only
        expected_xml = "<function_calls><invoke name='first'/></function_calls>"
        expected_text = (
            "First: \n"
            "Second: <function_calls><invoke name='second'/></function_calls>\n"
            "Done."
        )

        assert xml_obj == expected_xml
        assert text == expected_text
        # Second object is in text (not parsed, but received)
        assert "<invoke name='second'/>" in text
        # Full content has everything
        assert "<invoke name='second'/>" in full_content
        assert "Done." in full_content
        assert stream._closed


# === Sync Tests: JSON ===


class TestJSONParserSync:
    """Synchronous JSON parser tests (now enabled)."""

    def test_simple_json(self):
        """Test parsing a simple JSON object from a sync stream."""
        stream = MockStream(SIMPLE_JSON_CHUNKS)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        expected_json = {"name": "test", "value": 42}
        assert json_obj == expected_json
        assert text == "Here's the data: "
        # Stream closed early
        assert "Hope that helps!" not in full_content
        assert stream._closed

    def test_tricky_json(self):
        """Test JSON parser with braces and quotes in strings."""
        stream = MockStream(TRICKY_JSON_CHUNKS)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        expected_json = {"string": "has } brace", "escaped": 'has " quote'}

        assert json_obj == expected_json
        assert text == ""
        assert stream._closed

    def test_no_match_json(self):
        """Test JSON parser on plain text, expecting no match."""
        stream = MockStream(PLAIN_TEXT_CHUNKS)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        expected_content = "Just some plain text."
        assert text == expected_content
        assert json_obj is None
        assert full_content == expected_content
        assert stream._closed

    def test_incomplete_json(self):
        """Test JSON without closing brace."""
        incomplete_chunks = [
            MockChunk(content='{"key": "value"'),  # Missing }
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockStream(incomplete_chunks)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        assert json_obj is None
        assert text == full_content
        assert text == '{"key": "value"'
        assert stream._closed

    def test_false_positive_json(self):
        """Test text with braces that isn't valid JSON."""
        false_chunks = [
            MockChunk(content="Someone wrote { and } in text"),
            MockChunk(content=None, usage=MockUsage(5, 3)),
        ]
        stream = MockStream(false_chunks)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        assert json_obj is None
        assert text == full_content
        assert stream._closed

    def test_array_root_json(self):
        """Test JSON with an array as the root element."""
        array_chunks = [
            MockChunk(content="Here is the list: "),
            MockChunk(content='[1, {"key": "value"}, 3]'),
            MockChunk(content="...done."),
        ]
        stream = MockStream(array_chunks)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        assert json_obj == [1, {"key": "value"}, 3]
        assert text == "Here is the list: "
        assert "...done." not in full_content
        assert stream._closed

    def test_invalid_then_valid_json(self):
        """Test that parser skips invalid JSON and finds the next valid one."""
        chunks = [
            MockChunk(content="Invalid: {key: 'no quotes'}. "),
            MockChunk(content='Valid: {"id": 123}'),
            MockChunk(content="The end."),
        ]
        stream = MockStream(chunks)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = parser.parse(close_on_match=True)

        # The parser should find the *valid* JSON
        assert json_obj == {"id": 123}
        assert text == "Invalid: {key: 'no quotes'}. Valid: "
        assert "The end." not in full_content
        assert stream._closed


# === Async Tests: XML ===


class TestXMLParserAsync:
    """Asynchronous XML parser tests."""

    @pytest.mark.asyncio
    async def test_simple_xml(self):
        """Test parsing a simple XML object from an async stream."""
        stream = MockAsyncStream(SIMPLE_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        assert not stream._closed
        text, xml_obj, full_content = await parser.parse_async(close_on_match=True)

        # Stream closed after XML found
        expected_text = "Let me help. "
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
        assert " Done." not in full_content
        assert stream._closed

    @pytest.mark.asyncio
    async def test_split_xml(self):
        """Test async XML parser with tag split across chunks."""
        stream = MockAsyncStream(SPLIT_XML_TAG_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = await parser.parse_async(close_on_match=True)

        expected_xml = "<function_calls><invoke name='test'/></function_calls>"
        assert text == "Text "
        assert xml_obj == expected_xml
        assert stream._closed

    @pytest.mark.asyncio
    async def test_no_match(self):
        """Test async XML parser on plain text."""
        stream = MockAsyncStream(PLAIN_TEXT_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = await parser.parse_async(close_on_match=True)

        expected_content = "Just some plain text."
        assert text == expected_content
        assert xml_obj is None
        assert full_content == expected_content
        assert stream._closed

    @pytest.mark.asyncio
    async def test_nested_xml(self):
        """Test async nested XML parsing."""
        stream = MockAsyncStream(NESTED_XML_CHUNKS)
        parser = XMLStreamParser(stream)

        text, xml_obj, full_content = await parser.parse_async(close_on_match=True)

        assert xml_obj is not None
        assert "<invoke name='outer'>" in xml_obj
        assert "<inner>value</inner>" in xml_obj
        assert stream._closed


# === Async Tests: JSON ===


class TestJSONParserAsync:
    """Asynchronous JSON parser tests (now enabled)."""

    @pytest.mark.asyncio
    async def test_simple_json(self):
        """Test parsing a simple JSON object from an async stream."""
        stream = MockAsyncStream(SIMPLE_JSON_CHUNKS)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = await parser.parse_async(close_on_match=True)

        expected_json = {"name": "test", "value": 42}
        assert json_obj == expected_json
        assert text == "Here's the data: "
        assert "Hope that helps!" not in full_content
        assert stream._closed

    @pytest.mark.asyncio
    async def test_tricky_json(self):
        """Test async JSON parser with braces and quotes in strings."""
        stream = MockAsyncStream(TRICKY_JSON_CHUNKS)
        parser = JSONStreamParser(stream)

        text, json_obj, full_content = await parser.parse_async(close_on_match=True)

        expected_json = {"string": "has } brace", "escaped": 'has " quote'}

        assert json_obj == expected_json
        assert text == ""
        assert stream._closed
