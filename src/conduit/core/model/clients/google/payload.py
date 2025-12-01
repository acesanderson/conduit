from conduit.model.clients.openai.payload import OpenAIPayload


class GooglePayload(OpenAIPayload):
    frequency_penalty: None = None
    presence_penalty: None = None
