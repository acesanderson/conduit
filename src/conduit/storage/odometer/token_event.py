from pydantic import BaseModel, Field, model_validator
from conduit.core.model.models.provider import Provider
from typing import Any


class TokenEvent(BaseModel):
    # Required input fields
    model: str = Field(
        ..., description="Specific model by name (gpt-4o, claude-3.5-sonnet, etc.)."
    )
    input_tokens: int = Field(
        ..., description="Prompt tokens as defined and provided in API response."
    )
    output_tokens: int = Field(
        ..., description="Output tokens as defined and provided in API response."
    )

    # Generated fields (optional on input, filled automatically if missing/None)
    timestamp: int | None = Field(
        default=None, description="Unix epoch time in seconds."
    )
    host: str | None = Field(
        default=None, description="Simple host detection for multi-machine tracking."
    )
    provider: Provider | None = Field(
        default=None,
        description="Model provider (Anthropic, OpenAI, Google, etc.).",
    )

    @model_validator(mode="before")
    def fill_derived_fields(cls, data: dict[str, Any]):
        # timestamp: if missing or None, generate epoch seconds
        ts = data.get("timestamp")
        if ts is None:
            data["timestamp"] = cls._get_current_timestamp_s()

        # host: if missing or None, detect hostname
        host = data.get("host")
        if host is None:
            data["host"] = cls._get_hostname()

        # provider: if missing or None, infer from model
        prov = data.get("provider")
        if prov is None:
            model = data["model"]
            data["provider"] = cls._identify_provider(model)

        return data

    @staticmethod
    def _get_current_timestamp_s() -> int:
        import time

        # seconds to match PostgresBackend usage and existing data
        return int(time.time())

    @staticmethod
    def _get_hostname() -> str:
        import socket

        try:
            return socket.gethostname()
        except Exception:
            return "unknown"

    @staticmethod
    def _identify_provider(model: str) -> Provider:
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.identify_provider(model)


if __name__ == "__main__":
    # Example usage
    event = TokenEvent(model="gpt-4o", input_tokens=150, output_tokens=50)
    print(event.model_dump_json(indent=2))
