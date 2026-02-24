from __future__ import annotations
import asyncio
import logging
from typing import Any, override

from conduit.core.conduit.conduit_async import ConduitAsync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions

logger = logging.getLogger(__name__)


class ConduitBatchAsync:
    """
    Async implementation of Batch Conduit - a stateless execution engine.
    Handles concurrency control (semaphores) and batch strategy (Template vs String).
    """

    def __init__(self, prompt: Prompt | None = None):
        """
        Initialize with an optional prompt (required only for Template mode).
        """
        self.prompt: Prompt | None = prompt

    async def run(
        self,
        input_variables_list: list[dict[str, Any]] | None,
        prompt_strings_list: list[str] | None,
        params: GenerationParams,
        options: ConduitOptions,
        max_concurrent: int | None = None,
    ) -> list[Conversation]:
        """
        Execute the batch asynchronously.

        Args:
            input_variables_list: List of inputs for template rendering (Mode 1).
            prompt_strings_list: List of pre-rendered strings (Mode 2).
            params: Fully resolved generation parameters.
            options: Fully resolved conduit options.
            max_concurrent: Limit on concurrent tasks.

        Returns:
            list[Conversation]: Results in the same order as inputs.
        """
        # 1. Validate Mode
        if input_variables_list and prompt_strings_list:
            raise ValueError(
                "Provide exactly one of: input_variables_list OR prompt_strings_list"
            )
        if not input_variables_list and not prompt_strings_list:
            raise ValueError(
                "Must provide either input_variables_list or prompt_strings_list"
            )
        if input_variables_list and not self.prompt:
            raise ValueError(
                "input_variables_list mode requires a Prompt to be set on the instance"
            )

        logger.info("Running batch asynchronously.")

        # Warm up shared pool before spawning concurrent tasks
        from conduit.storage.db_manager import db_manager
        await db_manager.get_pool()

        # 2. Setup Concurrency
        semaphore = asyncio.Semaphore(max_concurrent) if max_concurrent else None

        # 3. Create Tasks
        tasks = []

        if input_variables_list:
            # Mode 1: Template mode - reuse one ConduitAsync with stored prompt
            # This is efficient as we don't re-parse the template every time
            conduit = ConduitAsync(self.prompt)
            tasks = [
                self._maybe_with_semaphore(
                    conduit.run(input_vars, params, options),
                    semaphore,
                )
                for input_vars in input_variables_list
            ]
            logger.info(
                f"Executing {len(tasks)} conversations in template mode with max_concurrent={max_concurrent or 'unlimited'}"
            )

        elif prompt_strings_list:
            # Mode 2: String mode - create temporary ConduitAsync for each string
            for prompt_str in prompt_strings_list:
                # We create a lightweight Prompt object for each string
                temp_conduit = ConduitAsync(Prompt(prompt_str))
                tasks.append(
                    self._maybe_with_semaphore(
                        # run with None for variables since prompt is pre-rendered
                        temp_conduit.run(None, params, options),
                        semaphore,
                    )
                )
            logger.info(
                f"Executing {len(tasks)} conversations in string mode with max_concurrent={max_concurrent or 'unlimited'}"
            )

        # 4. Execute
        conversations = await asyncio.gather(*tasks, return_exceptions=False)

        # Flush telemetry once for the whole batch
        from conduit.config import settings
        await settings.odometer_registry().flush()

        return list(conversations)

    async def _maybe_with_semaphore(
        self,
        coroutine: Any,
        semaphore: asyncio.Semaphore | None,
    ) -> Conversation:
        """
        Optionally wrap a coroutine with semaphore-based rate limiting.
        """
        if semaphore:
            async with semaphore:
                return await coroutine
        else:
            return await coroutine

    @override
    def __repr__(self) -> str:
        return f"ConduitBatchAsync(prompt={self.prompt!r})"
