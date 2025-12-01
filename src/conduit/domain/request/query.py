from conduit.domain.message.messages import MessageUnion


QueryInput = str | list[MessageUnion] | MessageUnion
