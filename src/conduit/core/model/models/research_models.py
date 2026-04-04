from __future__ import annotations

from conduit.core.model.models.modelspec import ModelSpecList, ModelSpec
from conduit.core.model.model_sync import ModelSync as Model
from conduit.sync import Conduit, GenerationParams, ConduitOptions, Verbosity
from conduit.core.prompt.prompt import Prompt
from rich.console import Console


list_prompt_str = """
You are an assistant who will help me identify the capabilities of a list of LLMs.

<provider>
{{provider}}
</provider>

<model_list>
{{model_list}}
</model_list>

You will be identifying, for each model, whether it supports the following functionalities:

- **text_completion**: The ability to generate, continue, or complete text based on prompts and context
- **image_analysis**: The capability to analyze, understand, and describe visual content in images
- **image_gen**: The ability to create or generate new images from text descriptions or prompts
- **audio_analysis**: The capability to process, transcribe, or analyze audio content and speech
- **audio_gen**: The ability to synthesize speech or generate audio content from text or other inputs
- **video_analysis**: The capability to analyze, understand, and extract information from video content
- **video_gen**: The ability to create or generate video content from text descriptions or other inputs
- **reasoning**: The capability to perform logical reasoning, problem-solving, and complex analytical tasks
- **heavy**: True if the model has >30B parameters OR requires >24GB VRAM to run locally. Cloud-hosted models (OpenAI, Anthropic, Google, Perplexity, Mistral) are always False. For Ollama models with unknown parameter counts, err on the side of False.

For each model, please return a ModelCapabilities object with boolean values for each capability, indicating whether the model supports (True) or does not support (False) each functionality.

You should also generate factual description of each model (50-80 words), including:
- The model's architecture type and parameter size (if known)
- Specific capabilities it supports (multimodal inputs, function calling, etc.)
- Context window size and key technical specifications
- Release date or version information
- Primary intended use cases or design focus
- Any notable technical features or limitations

Avoid promotional language or subjective quality assessments. Focus on objective, verifiable information about what the model can do and its technical characteristics.

Return a different object for each model -- since there are {{length}} models in our list, return {{length}} ModelCapabilities objects.
""".strip()

individual_prompt_str = """
You are an assistant who will help me identify the capabilities of a specific LLM.

Here's the provider:

<provider>
{{provider}}
</provider>

And here's the specific model:

<model>
{{model}}
</model>

You will be identifying, for the above model, whether it supports the following functionalities:
- **text_completion**: The ability to generate, continue, or complete text based on prompts and context
- **image_analysis**: The capability to analyze, understand, and describe visual content in images
- **image_gen**: The ability to create or generate new images from text descriptions or prompts
- **audio_analysis**: The capability to process, transcribe, or analyze audio content and speech
- **audio_gen**: The ability to synthesize speech or generate audio content from text or other inputs
- **video_analysis**: The capability to analyze, understand, and extract information from video content
- **video_gen**: The ability to create or generate video content from text descriptions or other inputs
- **reasoning**: The capability to perform logical reasoning, problem-solving, and complex analytical tasks
- **heavy**: True if the model has >30B parameters OR requires >24GB VRAM to run locally. Cloud-hosted models (OpenAI, Anthropic, Google, Perplexity, Mistral) are always False. For Ollama models with unknown parameter counts, err on the side of False.

Return a ModelCapabilities object with boolean values for each capability, indicating whether the model supports (True) or does not support (False) each functionality.

You should also generate a factual description of the model (50-80 words), including:
- The model's architecture type and parameter size (if known)
- Specific capabilities it supports (multimodal inputs, function calling, etc.)
- Context window size and key technical specifications
- Release date or version information
- Primary intended use cases or design focus
- Any notable technical features or limitations

Avoid promotional language or subjective quality assessments. Focus on objective, verifiable information about what the model can do and its technical characteristics.
""".strip()


def get_capabilities_by_provider(
    provider: str, model_list: list[str]
) -> list[ModelSpec]:
    length = len(model_list)
    params = GenerationParams(
        model="sonar-pro",
        response_model=ModelSpecList,
        output_type="structured_response",
    )
    prompt = Prompt(list_prompt_str)
    options = ConduitOptions(project_name="conduit", verbosity=Verbosity.PROGRESS)
    sync_conduit = Conduit(prompt=prompt, params=params, options=options)
    response = sync_conduit.run(
        input_variables={
            "provider": provider,
            "model_list": model_list,
            "length": length,
        }
    )

    return response.content.parsed.specs


def get_all_capabilities() -> list[ModelSpec]:
    """
    Get capabilities for all models across all providers. This shouldn't need to be run often, since it replaces the entire database of model specs.
    """
    all_models = Model.models()
    all_specs = []
    for index, (provider, models) in enumerate(all_models.items()):
        print(
            f"Processing {index + 1}/{len(all_models)}: {provider} with {len(models)} models"
        )
        specs = get_capabilities_by_provider(provider=provider, model_list=models)
        all_specs.extend(specs)
    return all_specs


def create_from_scratch() -> None:
    """
    Rebuild the entire ModelSpec table in Postgres from scratch.
    Deletes all existing rows, then regenerates via Perplexity.
    """
    from conduit.storage.modelspec_repository import ModelSpecRepository

    repo = ModelSpecRepository()
    all_specs = get_all_capabilities()
    for name in repo.get_all_names():
        repo.delete(name)
    for spec in all_specs:
        repo.upsert(spec)
    print(f"Populated ModelSpecs table with {len(all_specs)} entries.")
    retrieved = repo.get_all()
    assert len(retrieved) == len(all_specs), (
        "Retrieved specs do not match created specs."
    )


def get_capabilities_by_model(provider: str, model: str) -> ModelSpec:
    """
    Get capabilities for a specific model.
    """
    params = GenerationParams(
        model="sonar-pro",
        response_model=ModelSpec,
        output_type="structured_response",
    )
    prompt = Prompt(individual_prompt_str)
    options = ConduitOptions(project_name="conduit", verbosity=Verbosity.PROGRESS)
    conduit = Conduit(prompt=prompt, params=params, options=options)
    response = conduit.run(input_variables={"provider": provider, "model": model})
    return response.last.parsed


def create_modelspec(model: str) -> None:
    """
    Generate and persist a ModelSpec for a single model via Perplexity.
    Uses upsert — safe to call more than once for the same model.
    """
    from conduit.core.model.models.modelstore import ModelStore
    from conduit.storage.modelspec_repository import ModelSpecRepository

    provider = ModelStore.identify_provider(model)
    model_spec = get_capabilities_by_model(provider, model)
    if not isinstance(model_spec, ModelSpec):
        raise ValueError(
            f"Expected ModelSpec, got {type(model_spec)} for model {model}"
        )
    model_spec.model = model
    repo = ModelSpecRepository()
    repo.upsert(model_spec)
    print(f"Upserted ModelSpec for {model_spec.model} to Postgres.")


if __name__ == "__main__":
    modelspec = get_capabilities_by_model("ollama", "qwq:latest")
