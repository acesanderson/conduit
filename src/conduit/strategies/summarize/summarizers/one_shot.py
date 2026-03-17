from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.domain.result.response import GenerationResponse
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from typing import override, Any


class OneShotSummarizer(SummarizationStrategy):
    """
    Config schema:
    - model: str (default: "gpt3")
    - prompt: str (default: "Summarize the following text:\n\n{{text}}")
    - max_tokens: int (optional) - max tokens for the summary
    - temperature: float (optional) - temperature for generation
    - top_p: float (optional) - top_p for generation
    """

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            # Grab params
            model = get_param("model", default="gpt3")
            prompt = get_param(
                "prompt", default="Summarize the following text:\n\n{{text}}"
            )

            # Determine target tokens based on compression ratio and local tokenizer estimate
            from conduit.core.model.model_async import ModelAsync
            from conduit.strategies.summarize.compression import (
                get_target_summary_length,
            )

            tokenizer = ModelAsync(model=model).tokenize
            text_token_size: int = await tokenizer(text)
            target_tokens = get_target_summary_length(text_token_size)

            # Run the conduit
            from conduit.core.prompt.prompt import Prompt
            from conduit.domain.request.generation_params import GenerationParams
            from conduit.domain.config.conduit_options import ConduitOptions

            generation_params = GenerationParams(
                model=model,
                max_tokens=get_param("max_tokens", default=None),
                temperature=get_param("temperature", default=None),
                top_p=get_param("top_p", default=None),
            )
            options = ConduitOptions(
                project_name=get_param("project_name", default="conduit"),
                debug_payload=True,
            )
            model = ModelAsync(model=model)
            rendered = Prompt(prompt).render(
                {"text": text, "target_tokens": str(target_tokens)}
            )
            response = await model.query(
                query_input=rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(response, GenerationResponse)
            # Collect trace metadata
            add_metadata("text_token_size", text_token_size)
            add_metadata("target_tokens", target_tokens)
            add_metadata("input_tokens", response.metadata.input_tokens)
            add_metadata("output_tokens", response.metadata.output_tokens)
            return str(response.content)
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
