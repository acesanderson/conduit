"""
LLM-as-judge scorer using Gemini3 as the reference model.

Usage:
    references = {doc.source_id: doc.reference for doc in docs}
    judge = make_gemini_judge(references)
    eval_results = await evaluate(run_results, eval_function=judge)
"""
from __future__ import annotations

import asyncio
import re
import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)

_RETRY_DELAYS = [2.0, 8.0, 32.0]
_JUDGE_TIMEOUT = 45.0


async def _call_with_retry(coro_fn):
    last_exc: BaseException | None = None
    for attempt, pre_delay in enumerate([0.0] + _RETRY_DELAYS):
        if pre_delay:
            await asyncio.sleep(pre_delay)
        try:
            return await asyncio.wait_for(coro_fn(), timeout=_JUDGE_TIMEOUT)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "judge attempt %d/%d failed: %s: %s",
                attempt + 1,
                len(_RETRY_DELAYS) + 1,
                type(exc).__name__,
                exc,
            )
    raise last_exc


_JUDGE_PROMPT = """\
You are evaluating a generated summary against a gold-standard reference summary.

<reference_summary>
{{ reference }}
</reference_summary>

<generated_summary>
{{ generated }}
</generated_summary>

Score how well the generated summary captures the key information from the reference.

Rubric:
- 1.0: All key facts, entities, relationships, and conclusions present
- 0.8: Most key information present; only minor omissions
- 0.6: Core content captured but notable gaps or inaccuracies
- 0.4: Roughly half the key information; significant gaps
- 0.2: Major gaps or distortions; misses central points
- 0.0: Wrong, empty, or fundamentally off-topic

Respond with a single decimal number between 0.0 and 1.0. Nothing else."""


def _parse_score(text: str) -> float:
    match = re.search(r"\b(0(?:\.\d+)?|1(?:\.0*)?)\b", text.strip())
    if match:
        return round(float(match.group()), 4)
    logger.warning("Could not parse score from judge response: %r", text[:100])
    return 0.0


def make_gemini_judge(references: dict[str, str]) -> Callable:
    """
    Returns an async eval function that scores a RunResult against its reference
    using Gemini3 as judge. Pass the result of this to evals.evaluate().
    """
    async def gemini_judge(run_result) -> float:
        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.utils.progress.verbosity import Verbosity

        reference = references.get(run_result.source_id)
        if not reference:
            logger.warning("No reference found for source_id=%s", run_result.source_id)
            return 0.0

        rendered = Prompt(_JUDGE_PROMPT).render({
            "reference": reference,
            "generated": run_result.output.output,
        })
        model = ModelAsync(model="gemini3")
        params = GenerationParams(model="gemini3", temperature=0.0)
        options = ConduitOptions(
            project_name="summarization_eval",
            verbosity=Verbosity.SILENT,
        )
        async def _query():
            response = await model.query(query_input=rendered, params=params, options=options)
            return _parse_score(str(response.content))

        return await _call_with_retry(_query)

    gemini_judge.__name__ = "gemini_judge"
    return gemini_judge
