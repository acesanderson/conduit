import logging
from collections.abc import Iterable

logger = logging.getLogger(__name__)


class StreamToolParser:
    """
    A minimal class, modeled on the Host class patterns, to process a
    streaming query, parse it for a well-formed XML tool call,
    and stop the stream once a tool call is found.
    """

    # Define the tags we are looking for
    TAG_START = "<function_calls>"
    TAG_END = "</function_calls>"

    def __init__(self, stream: Iterable):
        """
        Initializes the parser with a streaming query request.
        """
        self.stream = stream
        self.pre_tool_text: str = ""
        self.tool_call_xml: Optional[str] = None
        self._buffer: str = ""

    def _find_xml_tool_call(self, text: str) -> tuple[int, int | None]:
        """
        Finds the start and end indices of the *first* complete
        <function_calls>...</function_calls> block.

        This is analogous to your _find_json_objects, but for XML.
        """
        start_idx = text.find(self.TAG_START)
        if start_idx == -1:
            # No start tag found yet
            return None

        # Look for the end tag *after* the start tag
        end_idx_start = text.find(self.TAG_END, start_idx)
        if end_idx_start == -1:
            # Start tag found, but no closing tag *yet*
            return None

        # We found a complete block.
        # Calculate the full end index
        end_idx = end_idx_start + len(self.TAG_END)
        return (start_idx, end_idx)

    def parse(self) -> tuple[str, str | None]:
        """
        Processes the stream, fulfilling the requirements.
        This is analogous to your _process_stream method.

        (1) Grabs the streaming query
        (2) Parses it for well-formed XML
        (3) Stops the stream if there is a tool request
        (4) Saves the response: everything before and the tool call.

        Returns:
            tuple[str, str | None]:
            - (pre_tool_text, tool_call_xml) if a tool call is found.
            - (full_response_text, None) if no tool call is found.
        """
        try:
            for chunk in self.stream:
                # Assuming chunk is a string-like object,
                # adapt this line if your stream yields objects
                content = str(chunk)
                if content:
                    self._buffer += content

                    # Check for a complete tool call
                    indices = self._find_xml_tool_call(self._buffer)

                    if indices:
                        # (3) Stop the stream
                        # (We stop by simply breaking the loop)
                        start_idx, end_idx = indices

                        # (4) Save the response
                        self.pre_tool_text = self._buffer[:start_idx]
                        self.tool_call_xml = self._buffer[start_idx:end_idx]

                        # And NOTHING afterwards
                        logger.info("Tool call found. Stopping stream.")
                        return (self.pre_tool_text, self.tool_call_xml)

        except KeyboardInterrupt:
            logger.info("Stream parsing cancelled by user")
            # Return whatever we had before the interrupt
            return (self._buffer, None)
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            # Return buffer to avoid data loss
            return (self._buffer, None)
        finally:
            # Ensure the stream is closed if it has a close() method
            if hasattr(self.stream, "close"):
                self.stream.close()

        # Stream finished without finding a tool call.
        # The "pre_tool_text" is the entire buffer.
        self.pre_tool_text = self._buffer
        return (self.pre_tool_text, None)
