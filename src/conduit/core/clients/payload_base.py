from conduit.core.model.clients.openai.payload import OpenAIPayload
from conduit.core.model.clients.ollama.payload import OllamaPayload
from conduit.core.model.clients.google.payload import GooglePayload
from conduit.core.model.clients.perplexity.payload import PerplexityPayload
from conduit.core.model.clients.anthropic.payload import AnthropicPayload

Payload = (
    AnthropicPayload | OllamaPayload | GooglePayload | PerplexityPayload | OpenAIPayload
)
