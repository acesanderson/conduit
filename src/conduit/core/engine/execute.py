"""
Execute a Tool Call.
"""

from __future__ import annotations
from conduit.domain.exceptions.exceptions import EngineError
from conduit.domain.message.message import ToolMessage
from typing import TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.domain.conversation.conversation import Conversation


async def execute(
    conversation: Conversation, params: GenerationParams, options: ConduitOptions
) -> Conversation:
    # Unpack params
    _ = params  # Unused parameter
    if not options.tool_registry:
        raise EngineError("Tool registry is not configured in options.")
    tool_registry = options.tool_registry
    tool_calls = conversation.tool_calls
    if len(tool_calls) == 0:
        raise EngineError("No tool calls found in the conversation.")
    # Execute each tool call and add the result to the conversation
    logger.debug(f"Executing {len(tool_calls)} tool calls.")
    for index, tool_call in enumerate(tool_calls):
        # Execute the tool call
        logger.debug(f"Executing tool call {index + 1}/{len(tool_calls)}: {tool_call}")
        content = await tool_registry.call_tool(tool_call)
        tool_call_id = tool_call.id
        name = tool_call.function_name
        # Create a ToolMessage from the result and add it to the conversation
        tool_message = ToolMessage.from_result(
            result=content,
            tool_call_id=tool_call_id,
            name=name,
        )
        conversation.add(tool_message)
    logger.debug("Finished executing tool calls.")
    return conversation
