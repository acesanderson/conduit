from pydantic import BaseModel, Field
from Chain.model.models.provider import Provider
from typing import Optional


class ModelSpec(BaseModel):
    model: str = Field(
        ...,
        description="The name of the model, e.g. gpt-4o, claude-2, gemini-1.5-flash, etc.",
    )
    description: Optional[str] = Field(
        ..., description="A brief description of the model's capabilities and features."
    )

    # Provider-specific capabilities
    provider: Provider = Field(
        ...,
        description="The provider of the model, e.g. openai, ollama, perplexity, google, anthropic, etc.",
    )
    temperature_range: list[float] = Field(
        ...,
        description="The supported temperature range for controlling randomness in model outputs, where lower values produce more deterministic responses and higher values increase creativity. OpenAI's standard range is 0.0 to 2.0, e.g. [0.0, 2.0]",
    )
    context_window: int = Field(
        ...,
        description="The maximum number of tokens the model can process in a single request, including both input and output tokens. OpenAI refers to this as 'max_tokens' in their API documentation. This represents Ollama's 'num_ctx' parameter, e.g. 128000",
    )

    # Model characteristics
    parameter_count: Optional[str] = Field(
        None, description="Parameter count (e.g., '7b', '70b', '405b')"
    )
    context_window: int = Field(..., description="Maximum context window in tokens")
    knowledge_cutoff: Optional[str] = Field(
        default=None,
        description="Date when the model's knowledge was last updated; None if unknown or continuously updated",
    )
    text_completion: bool = Field(
        ...,
        description="The ability to generate, continue, or complete text based on prompts and context, e.g. gpt-4o",
    )
    image_analysis: bool = Field(
        ...,
        description="The capability to analyze, understand, and describe visual content in images, e.g. claude-sonnet-4-20250514",
    )
    image_gen: bool = Field(
        ...,
        description="The ability to create or generate new images from text descriptions or prompts, e.g. gemini-2.0-flash-001",
    )
    audio_analysis: bool = Field(
        ...,
        description="The capability to process, transcribe, or analyze audio content and speech, e.g. gpt-4o-audio-preview",
    )
    audio_gen: bool = Field(
        ...,
        description="The ability to synthesize speech or generate audio content from text or other inputs, e.g. gpt-4o-audio-preview",
    )
    video_analysis: bool = Field(
        ...,
        description="The capability to analyze, understand, and extract information from video content, e.g. llama3.2-vision:11b",
    )
    video_gen: bool = Field(
        ...,
        description="The ability to create or generate video content from text descriptions or other inputs, e.g. gemini-2.5-pro-preview-05-06",
    )
    reasoning: bool = Field(
        ...,
        description="The capability to perform logical reasoning, problem-solving, and complex analytical tasks, e.g. o1-preview",
    )

    def __str__(self):
        return f"{self.model} ({self.provider})"

    def __repr__(self):
        return f"ModelSpec(model={self.model}, provider={self.provider})"

    @property
    def card(self):
        """
        Pretty print the model capabilities.
        """
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"[bold bright_yellow]Model Capabilities for {self.provider + '/' + self.model}[/bold bright_yellow]"
        )
        table.add_column("Capability", justify="left", style="cyan")
        table.add_column("Supported", justify="center", style="green")
        table.add_column("Description", justify="left", style="white")
        if self.temperature_range:
            table.add_row(
                "Temperature Range",
                f"{self.temperature_range[0]} to {self.temperature_range[1]}",
                "The range of temperature values supported by the model",
            )
        else:
            table.add_row(
                "Temperature Range",
                "Unknown",
                "The range of temperature values supported by the model",
            )
        table.add_row(
            "Context Window",
            str(self.context_window),
            "The maximum number of tokens the model can process in a single request",
        )
        table.add_row(
            "Parameter Count",
            self.parameter_count or "Unknown",
            "The number of parameters in the model",
        )
        table.add_row(
            "Knowledge Cutoff",
            self.knowledge_cutoff or "Unknown",
            "The date when the model's knowledge was last updated",
        )
        table.add_row(
            "Text Completion",
            str(self.text_completion),
            "The ability to generate, continue, or complete text based on prompts and context",
        )
        table.add_row(
            "Image Analysis",
            str(self.image_analysis),
            "The capability to analyze, understand, and describe visual content in images",
        )
        table.add_row(
            "Image Generation",
            str(self.image_gen),
            "The ability to create or generate new images from text descriptions or prompts",
        )
        table.add_row(
            "Audio Analysis",
            str(self.audio_analysis),
            "The capability to process, transcribe, or analyze audio content and speech",
        )
        table.add_row(
            "Audio Generation",
            str(self.audio_gen),
            "The ability to synthesize speech or generate audio content from text or other inputs",
        )
        table.add_row(
            "Video Analysis",
            str(self.video_analysis),
            "The capability to analyze, understand, and extract information from video content",
        )
        table.add_row(
            "Video Generation",
            str(self.video_gen),
            "The ability to create or generate video content from text descriptions or other inputs",
        )
        table.add_row(
            "Reasoning",
            str(self.reasoning),
            "The capability to perform logical reasoning, problem-solving, and complex analytical tasks",
        )
        description = f"\n[bold green]Description:[/bold green] [yellow]{self.description or 'No description available'}[/yellow]\n"

        # Add a bar for each capability
        console.print("\n")
        console.print(table)
        console.print(description)

    # Validation methods
    def is_image_analysis_supported(self) -> bool:
        """Check if image analysis is supported."""
        return self.image_analysis

    def is_image_generation_supported(self) -> bool:
        """Check if image generation is supported."""
        return self.image_gen

    def is_audio_analysis_supported(self) -> bool:
        """Check if audio analysis is supported."""
        return self.audio_analysis

    def is_audio_generation_supported(self) -> bool:
        """Check if audio generation is supported."""
        return self.audio_gen

    def is_video_analysis_supported(self) -> bool:
        """Check if video analysis is supported."""
        return self.video_analysis

    def is_video_generation_supported(self) -> bool:
        """Check if video generation is supported."""
        return self.video_gen

    def is_text_completion_supported(self) -> bool:
        """Check if text completion is supported."""
        return self.text_completion

    def is_reasoning_supported(self) -> bool:
        """Check if reasoning capabilities are supported."""
        return self.reasoning


class ModelSpecList(BaseModel):
    """
    A list of model capabilities.
    This is entirely for requests (to Perplexity).
    """

    specs: list[ModelSpec]


if __name__ == "__main__":
    example = ModelSpec(
        model="example-model",
        description="An example model for demonstration purposes.",
        provider="openai",
        temperature_range=[0.0, 2.0],
        context_window=4096,
        parameter_count="7b",
        knowledge_cutoff="2023-10-01",
        text_completion=True,
        image_analysis=False,
        image_gen=True,
        audio_analysis=False,
        audio_gen=True,
        video_analysis=False,
        video_gen=False,
        reasoning=True,
    )
    example.card
