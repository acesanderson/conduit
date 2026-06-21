from __future__ import annotations

import tiktoken
from typing import override, Any
from pydantic import BaseModel, ConfigDict
from conduit.strategies.summarize.strategy import SummarizationStrategy
from conduit.domain.result.response import GenerationResponse
from conduit.core.workflow.step import step, add_metadata

_tokenizer = tiktoken.get_encoding("cl100k_base")


class OneShotSummarizer(SummarizationStrategy):
    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore")
        model: str = "gpt-oss:latest"
        prompt: str = "Summarize the following text:\n\n{{text}}"
        max_tokens: int | None = None
        temperature: float | None = None
        top_p: float | None = None
        project_name: str = "conduit"
        use_remote: bool = False
        host_alias: str = "headwater"
        use_cache: bool = True

    config_model = Config

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        cfg = self.Config(**config)
        text = input.data

        from conduit.core.model.model_async import ModelAsync
        from conduit.strategies.summarize.compression import get_target_summary_length

        text_token_size: int = len(_tokenizer.encode(text))
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
            use_remote=cfg.use_remote,
            use_cache=cfg.use_cache,
        )
        if cfg.use_remote:
            from conduit.core.model.model_remote import RemoteModelAsync
            model = RemoteModelAsync(model=cfg.model, host_alias=cfg.host_alias)
        else:
            model = ModelAsync(model=cfg.model)
        rendered = Prompt(cfg.prompt).render(
            {"text": text, "target_tokens": str(target_tokens)}
        )
        guideline = getattr(input, "guideline", None)
        if guideline:
            rendered = f"{guideline}\n\n{rendered}"
            add_metadata("guideline_applied", True)
        add_metadata("rendered_prompt", rendered)
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
