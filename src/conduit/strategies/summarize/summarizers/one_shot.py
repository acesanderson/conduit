from __future__ import annotations

from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.domain.result.response import GenerationResponse
from conduit.core.workflow.step import step, add_metadata


class OneShotSummarizer(SummarizationStrategy):
    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        prompt: str = "Summarize the following text:\n\n{{text}}"
        max_tokens: int | None = None
        temperature: float | None = None
        top_p: float | None = None
        project_name: str = "conduit"

    config_model = Config

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data

        from conduit.core.model.model_async import ModelAsync
        from conduit.strategies.summarize.compression import get_target_summary_length

        tokenizer = ModelAsync(model=cfg.model).tokenize
        text_token_size: int = await tokenizer(text)
        target_tokens = get_target_summary_length(text_token_size)

        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions

        generation_params = GenerationParams(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        options = ConduitOptions(
            project_name=cfg.project_name,
            debug_payload=True,
        )
        model = ModelAsync(model=cfg.model)
        rendered = Prompt(cfg.prompt).render(
            {"text": text, "target_tokens": str(target_tokens)}
        )
        response = await model.query(
            query_input=rendered,
            params=generation_params,
            options=options,
        )
        assert isinstance(response, GenerationResponse)
        add_metadata("text_token_size", text_token_size)
        add_metadata("target_tokens", target_tokens)
        add_metadata("input_tokens", response.metadata.input_tokens)
        add_metadata("output_tokens", response.metadata.output_tokens)
        return str(response.content)
