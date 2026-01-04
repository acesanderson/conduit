from conduit.extensions.summarize.strategy import SummarizationStrategy
from conduit.domain.result.response import GenerationResponse
from conduit.core.workflow.workflow import step, get_param, add_metadata
from typing import override


class OneShotSummarizer(SummarizationStrategy):
    @step
    @override
    async def __call__(self, text: str, **kwargs) -> str:
        # Grab params
        model = get_param("model", default="gpt-3.5")
        prompt = get_param(
            "prompt", default="Summarize the following text:\n\n{{text}}"
        )

        # Run the conduit
        from conduit.core.model.model_async import ModelAsync
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
            project_name=get_param("project_name", default="conduit")
        )
        model = ModelAsync(model=model)
        rendered = Prompt(prompt).render({"text": text})
        response = await model.query(
            query_input=rendered,
            params=generation_params,
            options=options,
        )
        assert isinstance(response, GenerationResponse)
        # Collect trace metadata
        add_metadata("input_tokens", response.metadata.input_tokens)
        add_metadata("output_tokens", response.metadata.output_tokens)
        return str(response.content)
