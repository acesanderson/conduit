from conduit.domain.message.message import Message, UserMessage, AssistantMessage
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.conversation.session import Session

um = UserMessage(content="Write a python function that adds two numbers.")
messages = [um]

conv = Conversation(messages=messages)

session = conv.session

am = AssistantMessage(
    content="Here is a function that adds two numbers.",
    predecessor_id=um.message_id,
    session_id=session.session_id,
)
