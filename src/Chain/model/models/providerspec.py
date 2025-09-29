from pydantic import BaseModel, Field
from typing import Literal
from pathlib import Path

dir_path = Path(__file__).parent
providers_file = dir_path / "providers.jsonl"

class ProviderSpec(BaseModel):
    """
    A specification for a provider's capabilities.
    This is entirely for requests (to Perplexity).
    """
    provider: str = Field(..., description="The name of the provider")
    temperature_range: list[float] | Literal["Not supported"] = Field(
        ..., description="The range of temperature settings available for the provider"
    )

def populate_providers_file():
    """
    Populate the providers.jsonl file with provider specifications.
    """
    if providers_file.exists():
        providers_file.unlink()  # Remove existing file if it exists

    providers_file.touch()  # Create a new empty file

    # Define provider specifications
    openai = ProviderSpec(
        provider="openai",
        temperature_range=[0.0, 2.0]
    )

    anthropic = ProviderSpec(
        provider="anthropic",
        temperature_range=[0.0, 1.0]
        )

    google = ProviderSpec(
        provider="google",
        temperature_range=[0.0, 1.0]
    )

    groq = ProviderSpec(
        provider="groq",
        temperature_range=[0.0, 1.0]
    )

    perplexity = ProviderSpec(
        provider="perplexity",
        temperature_range="Not supported"
    )

    ollama = ProviderSpec(
        provider="ollama",
        temperature_range=[0.0, 1.0]
    )

    providers = [openai, anthropic, google, groq, perplexity, ollama]

    providers_file.write_text(
        "\n".join(provider.model_dump_json() for provider in providers)
    )
    print(f"Populated {providers_file} with provider specifications.")

if __name__ == "__main__":
    populate_providers_file()
