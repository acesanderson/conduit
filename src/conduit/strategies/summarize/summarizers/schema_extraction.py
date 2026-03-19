from __future__ import annotations

import asyncio
import json
import logging
from typing import override, Any, get_origin
from pydantic import BaseModel, ConfigDict
from conduit.core.workflow.step import step, add_metadata
from conduit.strategies.summarize.strategy import SummarizationStrategy, _TextInput
from conduit.strategies.summarize.summarizers.chunker import Chunker

logger = logging.getLogger(__name__)

extraction_prompt = """
Extract structured information from the following text according to the JSON schema below.

Schema:
<schema>
{{ schema }}
</schema>

Text:
<text>
{{ text }}
</text>

Return ONLY valid JSON that conforms to the schema. No explanation, no markdown fences.
""".strip()


def _merge_instances(instances: list[BaseModel], schema_cls: type[BaseModel]) -> BaseModel:
    """Merge list fields by concatenation; scalar fields take first non-None value."""
    merged_data: dict[str, Any] = {}
    for field_name, field_info in schema_cls.model_fields.items():
        values = [getattr(inst, field_name) for inst in instances]
        origin = get_origin(field_info.annotation)
        if origin is list:
            merged_data[field_name] = [item for v in values if v is not None for item in v]
        else:
            merged_data[field_name] = next((v for v in values if v is not None), None)
    return schema_cls(**merged_data)


class SchemaExtractionSummarizer(SummarizationStrategy):
    """
    Structured extraction using a caller-supplied Pydantic schema.

    Workflow:
    1. Accept a `schema` config param — a Pydantic BaseModel class (not an instance).
       Raises ValueError if absent.
    2. Chunk the text.
    3. For each chunk in parallel, prompt the LLM to extract data matching the schema
       as JSON. Parse and validate each response via the schema.
       On parse failure, log a warning and skip that chunk.
    4. Merge all valid extracted instances: list fields concatenated, scalar fields
       take first non-None value.
    5. Return json.dumps(merged.model_dump(), indent=2).

    Config params:
        schema:            Pydantic BaseModel class (required)
        model:             LLM (default: gpt3)
        concurrency_limit: max parallel chunk calls (default: 5)
        max_tokens:        max tokens per call
        temperature:       sampling temperature
    """

    class Config(BaseModel):
        model_config = ConfigDict(extra="ignore", arbitrary_types_allowed=True)
        model: str = "gpt3"
        schema: Any = None  # expected: type[BaseModel]
        concurrency_limit: int = 5
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

        if cfg.schema is None:
            raise ValueError(
                "SchemaExtractionSummarizer requires a 'schema' config param: "
                "a Pydantic BaseModel class (not an instance)."
            )
        schema_cls: type[BaseModel] = cfg.schema

        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.domain.result.response import GenerationResponse
        from conduit.utils.progress.verbosity import Verbosity

        chunker = Chunker()
        chunks = await chunker(text, config)
        total_chunks = len(chunks)
        logger.info(f"SchemaExtractionSummarizer: {total_chunks} chunks")

        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2)
        generation_params = GenerationParams(
            model=cfg.model,
            max_tokens=cfg.max_tokens,
            temperature=cfg.temperature,
            top_p=cfg.top_p,
        )
        options = ConduitOptions(
            project_name=cfg.project_name,
            verbosity=Verbosity.SILENT,
            debug_payload=True,
        )
        model_instance = ModelAsync(model=cfg.model)
        semaphore = asyncio.Semaphore(cfg.concurrency_limit)

        async def extract_chunk(chunk: str) -> BaseModel | None:
            rendered = Prompt(extraction_prompt).render({
                "schema": schema_json,
                "text": chunk,
            })
            async with semaphore:
                response = await model_instance.query(
                    query_input=rendered,
                    params=generation_params,
                    options=options,
                )
            assert isinstance(response, GenerationResponse)
            content = str(response.content).strip()
            # Strip markdown fences if present
            if content.startswith("```"):
                lines = content.splitlines()
                end = -1 if lines[-1].startswith("```") else len(lines)
                content = "\n".join(lines[1:end])
            try:
                data = json.loads(content)
                return schema_cls(**data)
            except Exception as exc:
                logger.warning(f"Failed to parse chunk response: {exc}")
                return None

        results: list[BaseModel | None] = await asyncio.gather(
            *[extract_chunk(chunk) for chunk in chunks]
        )

        valid_instances = [r for r in results if r is not None]
        chunks_failed = total_chunks - len(valid_instances)

        add_metadata("num_chunks", total_chunks)
        add_metadata("chunks_parsed_successfully", len(valid_instances))
        add_metadata("chunks_failed", chunks_failed)

        if not valid_instances:
            logger.warning("No chunks parsed successfully — returning empty extraction")
            return json.dumps({}, indent=2)

        merged = _merge_instances(valid_instances, schema_cls)
        return json.dumps(merged.model_dump(), indent=2)


if __name__ == "__main__":
    import asyncio

    class _MissionSchema(BaseModel):
        mission_name: str | None = None
        launch_date: str | None = None
        crew_members: list[str] = []
        key_events: list[str] = []

    _sample = (
        "The Apollo 11 mission launched on July 16, 1969, carrying astronauts Neil Armstrong, "
        "Buzz Aldrin, and Michael Collins. On July 20, Armstrong and Aldrin landed on the Moon "
        "in the Sea of Tranquility while Collins orbited above. Armstrong became the first human "
        "to walk on the Moon at 02:56 UTC, followed by Aldrin. They collected 21.5 kg of lunar "
        "material and deployed several scientific instruments. The mission returned to Earth on "
        "July 24, splashing down in the Pacific Ocean."
    )

    async def _main() -> None:
        result = await SchemaExtractionSummarizer()(
            _TextInput(_sample),
            {"model": "gpt3", "schema": _MissionSchema},
        )
        print(result)

    asyncio.run(_main())
