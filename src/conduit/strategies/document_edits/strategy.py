from __future__ import annotations
from conduit.core.workflow.protocols import Strategy
from conduit.core.workflow.step import step, get_param
import logging

logger = logging.getLogger(__name__)


class DocumentEditStrategy(Strategy):
    @step
    async def __call__(self, document: str, user_prompt: str, **kwargs) -> str:
        from conduit.core.conduit.conduit_async import ConduitAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.strategies.document_edits.models import DocumentEdits
        from conduit.strategies.document_edits.prompt import PROMPT_TEMPLATE
        from conduit.strategies.document_edits.apply import apply_edits

        model = get_param("model", default="gpt3")
        temperature = get_param("temperature", default=None)
        max_tokens = get_param("max_tokens", default=None)
        project_name = get_param("project_name", default="conduit")

        params = GenerationParams(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            output_type="structured_response",
            response_model=DocumentEdits,
        )
        options = ConduitOptions(
            project_name=project_name,
            include_history=False,
        )

        prompt = Prompt(PROMPT_TEMPLATE)
        conduit = ConduitAsync(prompt=prompt)
        conversation = await conduit.run(
            input_variables={"user_prompt": user_prompt, "document": document},
            params=params,
            options=options,
        )

        last = conversation.last
        if not isinstance(getattr(last, "parsed", None), DocumentEdits):
            raise TypeError(
                f"Expected DocumentEdits from LLM, got {type(getattr(last, 'parsed', None))}"
            )

        return apply_edits(document, last.parsed.edits)
