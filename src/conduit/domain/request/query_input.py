from conduit.domain.message.message import Message, UserMessage


QueryInput = str | list[Message] | Message


def constrain_query_input(query_input: QueryInput) -> list[Message]:
    """
    Constrains the input to a list of Message objects.
    """
    if isinstance(query_input, str):
        return [UserMessage(content=query_input)]
    elif isinstance(query_input, Message):
        return [query_input]
    elif isinstance(query_input, list) and all(
        isinstance(msg, Message) for msg in query_input
    ):
        return query_input
    else:
        raise TypeError("Input must be a string, a Message, or a list of Messages.")
