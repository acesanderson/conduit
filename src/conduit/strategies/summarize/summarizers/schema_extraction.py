"""
```python
from conduit.core.workflow.harness import ConduitHarness
from conduit.strategies.summarize.summarizers.schema_extraction import SchemaExtractionSummarizer

config = {
    "model": "gpt-oss:latest",
    "schema": {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "key_points": {"type": "array", "items": {"type": "string"}},
            "entities": {"type": "array", "items": {"type": "string"}},
            "dates": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["title", "key_points"],
    },
}

harness = ConduitHarness(config=config)
result = await harness.run(SchemaExtractionSummarizer(), text=document)
```
"""

from __future__ import annotations

import json
import logging
from typing import override, Any
from conduit.core.workflow.step import step, get_param, add_metadata
from conduit.core.workflow.context import context
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.core.model.model_async import ModelAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions
from conduit.domain.result.response import GenerationResponse
from conduit.utils.progress.verbosity import Verbosity

logger = logging.getLogger(__name__)


schema_extraction_prompt = """
Extract structured information from the following document according to the provided JSON schema.

Schema:
<schema>
{{ schema }}
</schema>

Document:
<document>
{{ document }}
</document>

Instructions:
1. Parse the document and extract information matching the schema structure
2. Return ONLY valid JSON that conforms to the schema
3. For arrays, include all relevant items found in the document
4. If a field is not present in the document, use an empty array or appropriate default
5. Do not include any text outside the JSON response

Return JSON only.
""".strip()


class SchemaExtractionSummarizer(SummarizationStrategy):
    """
    Schema-based extraction summarizer.

    Workflow:
    1. Accept a JSON schema defining the desired output structure.
    2. Ask the model to extract information from the document matching the schema.
    3. Return the structured JSON output as the summary.

    This approach produces summaries in a predictable, parseable format that
    can be easily integrated into downstream systems or used for data extraction.
    """

    @step
    @override
    async def __call__(self, input: Any, config: dict) -> str:
        token_conf = context.config.set(config)
        token_defaults = context.use_defaults.set(True)
        try:
            text = input.data
            model = get_param("model", default="gpt3")
            schema = get_param("schema", default=None)

            if schema is None:
                # Provide a default schema for common use cases
                schema = {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "key_points": {"type": "array", "items": {"type": "string"}},
                        "entities": {"type": "array", "items": {"type": "string"}},
                        "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
                    },
                    "required": ["title", "key_points"],
                }

            # Generate the prompt
            rendered = Prompt(schema_extraction_prompt).render({
                "schema": json.dumps(schema, indent=2),
                "document": text,
            })

            # Configure model call
            generation_params = GenerationParams(
                model=model,
                max_tokens=get_param("max_tokens", default=None),
                temperature=get_param("temperature", default=None),
                top_p=get_param("top_p", default=None),
            )
            options = ConduitOptions(
                project_name=get_param("project_name", default="conduit"),
                verbosity=Verbosity.SILENT,
                debug_payload=True,
            )
            model_instance = ModelAsync(model=model)

            # Call the model
            response = await model_instance.query(
                query_input=rendered,
                params=generation_params,
                options=options,
            )
            assert isinstance(response, GenerationResponse)

            # Parse and validate JSON response
            try:
                content = str(response.content)
                # Try to extract JSON from response (in case of extra text)
                json_start = content.find("{")
                json_end = content.rfind("}") + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = content[json_start:json_end]
                    parsed = json.loads(json_str)
                else:
                    parsed = json.loads(content)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse JSON response: {response.content}")
                # Return the raw content if parsing fails
                parsed = {"raw_content": response.content}

            # Convert back to readable string format
            result = json.dumps(parsed, indent=2)

            # Collect metadata
            add_metadata("input_tokens", response.metadata.input_tokens)
            add_metadata("output_tokens", response.metadata.output_tokens)
            add_metadata("schema_valid", True)

            return result
        finally:
            context.config.reset(token_conf)
            context.use_defaults.reset(token_defaults)
