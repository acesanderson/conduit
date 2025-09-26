# Imports
from Chain.chain.chain import Chain
from Chain.chain.asyncchain import AsyncChain
from Chain.prompt.prompt import Prompt
from Chain.model.model import Model
from Chain.model.model_client import ModelClient
from Chain.model.model_async import ModelAsync
from Chain.model.models.modelstore import ModelStore
from Chain.message.message import Message
from Chain.message.messagestore import MessageStore
from Chain.message.textmessage import create_system_message
from Chain.cache.cache import ChainCache
from Chain.result.response import Response
from Chain.result.error import ChainError
from Chain.result.result import ChainResult
from Chain.parser.parser import Parser
from Chain.progress.verbosity import Verbosity
from Chain.chat.chat import Chat

# Set global logging settings
import logging


def set_log_level(level):
    """Set logging level for entire Chain package"""
    logging.getLogger("Chain").setLevel(level)

    # Also set for common third-party libraries
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


# Convenience functions
def disable_logging():
    set_log_level(logging.CRITICAL)


def enable_debug_logging():
    set_log_level(logging.DEBUG)


__all__ = [
    "Chain",
    "AsyncChain",
    "Prompt",
    "Model",
    "ModelClient",
    "ModelAsync",
    "ModelStore",
    "Parser",
    "Response",
    "ChainError",
    "ChainResult",
    "Verbosity",
    "set_log_level",
    "disable_logging",
    "enable_debug_logging",
    "Message",
    "MessageStore",
    "ChainCache",
    "create_system_message",
    "Chat",
]
