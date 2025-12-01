from conduit.model.clients.openai.payload import OpenAIPayload
from conduit.model.clients.ollama.payload import OllamaPayload
from conduit.model.clients.google.payload import GooglePayload
from conduit.model.clients.perplexity.payload import PerplexityPayload
from conduit.model.clients.anthropic.payload import AnthropicPayload

Payload = (
    AnthropicPayload | OllamaPayload | GooglePayload | PerplexityPayload | OpenAIPayload
)
