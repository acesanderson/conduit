"""
Note: the mix of Message / MessageBase is intentional, as python has different requirements for
- type checking (mypy, ruff)
- runtime checks (isinstance)
"""

from conduit.domain.message.message import Message, MessageBase, UserMessage


QueryInput = str | list[Message] | Message


def constrain_query_input(query_input: QueryInput) -> list[Message]:
    """
    Constrains the input to a list of Message objects.
    """
    if isinstance(query_input, str):
        return [UserMessage(content=query_input)]
    elif isinstance(query_input, MessageBase):
        return [query_input]
    elif isinstance(query_input, list) and all(
        isinstance(msg, MessageBase) for msg in query_input
    ):
        return query_input
    else:
        raise TypeError("Input must be a string, a Message, or a list of Messages.")
