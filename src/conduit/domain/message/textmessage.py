"""
Factory function for creating system messages.
"""

from conduit.core.prompt.prompt import Prompt
from conduit.domain.message.message import SystemMessage
import logging

logger = logging.getLogger(__name__)


# Some helpful functions
def create_system_message(
    system_prompt: str | Prompt, input_variables: dict[str, str] | None = None
) -> SystemMessage:
    raise NotImplementedError("Not implemented yet")
    if isinstance(system_prompt, str):
        system_prompt = Prompt(system_prompt)
    if input_variables:
        system_message = [
            SystemMessage(
                content=system_prompt.render(input_variables=input_variables),
            )
        ]
    else:
        system_message = [SystemMessage(content=system_prompt.prompt_string)]
    return system_message
