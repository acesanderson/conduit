from conduit.core.clients.openai.payload import OpenAIPayload
from conduit.core.clients.ollama.payload import OllamaPayload
from conduit.core.clients.google.payload import GooglePayload
from conduit.core.clients.perplexity.payload import PerplexityPayload
from conduit.core.clients.anthropic.payload import AnthropicPayload
from conduit.core.clients.mistral.payload import MistralPayload

Payload = (
    AnthropicPayload | OllamaPayload | GooglePayload | PerplexityPayload | OpenAIPayload | MistralPayload
)
