from conduit.message.messages import MessageUnion


QueryInput = str | list[MessageUnion] | MessageUnion
