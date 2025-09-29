from conduit.conduit.sync_conduit import SyncConduit, Prompt
from conduit.model.model_async import ModelAsync
from conduit.result.response import Response
from conduit.result.error import ConduitError
from conduit.logs.logging_config import configure_logging, logging
from conduit.parser.parser import Parser
from conduit.progress.verbosity import Verbosity
from conduit.message.messagestore import MessageStore
from typing import TYPE_CHECKING, Optional
import asyncio

# Our TYPE_CHECKING imports, these ONLY load for IDEs, so you can still lazy load in production.
if TYPE_CHECKING:
    from rich.console import Console

logger = configure_logging(
    level=logging.INFO,
)


class AsyncConduit(SyncConduit):
    _message_store: Optional[MessageStore] = None
    # If you want rich progress reporting, add a rich.console.Console object to Conduit. (also can be added at Model level)
    _console: Optional["Console"] = None

    def __init__(
        self,
        model: ModelAsync,
        prompt: Prompt | None = None,
        parser: Parser | None = None,
    ):
        if not isinstance(model, ModelAsync):
            raise TypeError("Model must be of type ModelAsync")
        if prompt and not isinstance(prompt, Prompt):
            raise TypeError("Prompt must be of type Prompt")
        if parser and not isinstance(parser, Parser):
            raise TypeError("Parser must be of type Parser")
        """Override to use ModelAsync"""
        self.prompt = prompt
        self.model = model
        self.parser = parser
        if self.prompt:
            self.input_schema = self.prompt.input_schema()  # this is a set
        else:
            self.input_schema = set()

    def run(
        self,
        input_variables_list: list[dict] | None = None,
        prompt_strings: list[str] | None = None,
        semaphore: Optional[asyncio.Semaphore] = None,
        cache=True,
        verbose: Verbosity = Verbosity.PROGRESS,
        print_response=False,
    ) -> list[Response]:
        """
        Asynchronously runs multiple prompt strings or input variable sets in parallel.

        This method facilitates concurrent LLM calls, making it ideal for batch
        processing and high-throughput applications. It manages an internal
        asynchronous event loop to execute queries concurrently. Progress
        display is automatically handled, showing overall batch progress.

        Args:
            input_variables_list (list[dict] | None): A list of dictionaries,
                where each dictionary contains input variables for a prompt
                template. Used when the Conduits's prompt is a template. Defaults
                to None.
            prompt_strings (list[str] | None): A list of pre-rendered prompt
                strings to be sent to the model. Used when no prompt template
                is involved or prompts are pre-generated. Defaults to None.
            semaphore (Optional[asyncio.Semaphore]): An optional asyncio.Semaphore
                to control the maximum number of concurrent requests. If None,
                requests will run as fast as possible. Defaults to None.
            cache (bool): If True, responses will be looked up in the cache
                and saved to it if not found. Defaults to True.
            verbose (Verbosity): If True, displays real-time progress information
                for the entire batch of operations. Individual operation progress
                is suppressed during concurrent execution. Defaults to True.
            print_response (bool): If True, prints the content of each individual
                response as it is received. Useful for debugging but can be verbose.
                Defaults to False.

        Returns:
            list[Response]: A list of `Response` objects, each corresponding
            to an individual query in the batch. Exceptions during individual
            queries are caught and returned as part of the result list.

        Raises:
            ValueError: If neither `prompt_strings` nor `input_variables_list`
                are provided.
            ValueError: If `input_variables_list` is provided but no `Prompt`
                is assigned to the `AsyncConduit` object.

        Examples:
            >>> # Process multiple simple prompts concurrently
            >>> prompts = ["What is 1+1?", "What is 2+2?"]
            >>> model_async = ModelAsync("gpt-4o-mini")
            >>> async_conduit = AsyncConduit(model=model_async)
            >>> responses = async_conduit.run(prompt_strings=prompts)
            >>> for resp in responses:
            >>>     print(resp.content)

            >>> # Process prompts from a template with input variables
            >>> prompt_template = Prompt("Describe a {{color}} {{animal}}.")
            >>> input_vars = [{"color": "red", "animal": "fox"}, {"color": "blue", "animal": "whale"}]
            >>> model_async = ModelAsync("claude-3-haiku")
            >>> async_conduit = AsyncConduit(prompt=prompt_template, model=model_async)
            >>> responses = async_conduit.run(input_variables_list=input_vars, semaphore=asyncio.Semaphore(2))
            >>> for resp in responses:
            >>>     print(resp.content)
        """

        async def _run_async():
            if prompt_strings:
                return await self._run_prompt_strings(
                    prompt_strings, semaphore, cache, verbose, print_response
                )
            if input_variables_list:
                return await self._run_input_variables(
                    input_variables_list, semaphore, cache, verbose, print_response
                )

        responses = asyncio.run(_run_async())

        assert isinstance(responses, list), "Responses should be a list"
        assert all([isinstance(r, Response) for r in responses]), (
            "All results should be Response objects"
        )

        return responses

    async def _run_prompt_strings(
        self,
        prompt_strings: list[str],
        semaphore: Optional[asyncio.Semaphore] = None,
        cache=True,
        verbose: Verbosity = Verbosity.PROGRESS,
        print_response=False,
    ) -> list:
        """Run multiple prompt strings concurrently with enhanced progress display"""

        # Create concurrent progress tracker if verbose
        tracker = None
        if verbose:
            console = self.model.console or self.__class__._console
            from conduit.progress.tracker import ConcurrentTracker
            from conduit.progress.wrappers import create_concurrent_progress_tracker

            tracker = create_concurrent_progress_tracker(console, len(prompt_strings))
            tracker.emit_concurrent_start()

        async def process_with_semaphore_and_tracking(
            prompt_string: str,
            parser: Parser | None,
            semaphore: Optional[asyncio.Semaphore],
            tracker: Optional[ConcurrentTracker],
            cache=True,
            verbose: Verbosity = Verbosity.SILENT,  # Always suppress individual progress during concurrent
            print_response=False,
        ):
            async def do_work():
                if semaphore:
                    async with semaphore:
                        return await self.model.query_async(
                            query_input=prompt_string,
                            response_model=self.parser.pydantic_model
                            if self.parser
                            else None,
                            cache=cache,
                            verbose=verbose,
                            print_response=print_response,
                        )
                else:
                    return await self.model.query_async(
                        query_input=prompt_string,
                        response_model=self.parser.pydantic_model
                        if self.parser
                        else None,
                        cache=cache,
                        verbose=verbose,
                        print_response=print_response,
                    )

            # Wrap with concurrent tracking if available
            if tracker:
                from conduit.progress.wrappers import concurrent_wrapper

                return await concurrent_wrapper(do_work(), tracker)
            else:
                return await do_work()

        # Create coroutines with tracking
        coroutines = [
            process_with_semaphore_and_tracking(
                prompt_string,
                self.parser,
                semaphore,
                tracker,
                cache,
                verbose=False,  # Always suppress individual progress
                print_response=print_response,
            )
            for prompt_string in prompt_strings
        ]

        # Run all operations concurrently
        responses = await asyncio.gather(*coroutines, return_exceptions=True)

        assert isinstance(responses, list), "Responses should be a list"
        assert all([isinstance(r, Response) for r in responses]), (
            "All results should be Response objects"
        )

        # Complete concurrent tracking
        if tracker:
            tracker.emit_concurrent_complete()

        return responses

    async def _run_input_variables(
        self,
        input_variables_list: list[dict],
        semaphore: Optional[asyncio.Semaphore] = None,
        cache=True,
        verbose: Verbosity = Verbosity.PROGRESS,
        print_response=False,
    ) -> list:
        """Run multiple input variable sets concurrently with enhanced progress display"""

        if not self.prompt:
            raise ValueError("No prompt assigned to AsyncConduit object")

        # Create concurrent progress tracker if verbose
        tracker = None
        if verbose:
            console = self.model.console or self.__class__._console
            from conduit.progress.tracker import ConcurrentTracker
            from conduit.progress.wrappers import create_concurrent_progress_tracker

            tracker = create_concurrent_progress_tracker(
                console, len(input_variables_list)
            )
            tracker.emit_concurrent_start()

        async def process_with_semaphore_and_tracking(
            input_variables: dict,
            parser: Parser | None,
            semaphore: Optional[asyncio.Semaphore],
            tracker: Optional[ConcurrentTracker],
            cache=True,
            verbose: Verbosity = Verbosity.SILENT,  # Always suppress individual progress during concurrent
            print_response=False,
        ):
            async def do_work():
                if semaphore:
                    async with semaphore:
                        return await self.model.query_async(
                            query_input=self.prompt.render(
                                input_variables=input_variables
                            ),
                            response_model=parser.pydantic_model if parser else None,
                            cache=cache,
                            verbose=verbose,
                            print_response=print_response,
                        )
                else:
                    return await self.model.query_async(
                        query_input=self.prompt.render(input_variables=input_variables),
                        response_model=parser.pydantic_model if parser else None,
                        cache=cache,
                        verbose=verbose,
                        print_response=print_response,
                    )

            # Wrap with concurrent tracking if available
            if tracker:
                from conduit.progress.wrappers import concurrent_wrapper

                return await concurrent_wrapper(do_work(), tracker)
            else:
                return await do_work()

        # Create coroutines with tracking
        coroutines = [
            process_with_semaphore_and_tracking(
                input_variables,
                self.parser,
                semaphore,
                tracker,
                cache=cache,
                verbose=False,  # Always suppress individual progress
                print_response=print_response,
            )
            for input_variables in input_variables_list
        ]

        # Run all operations concurrently
        responses = await asyncio.gather(*coroutines, return_exceptions=True)

        # Complete concurrent tracking
        if tracker:
            tracker.emit_concurrent_complete()

        if isinstance(responses[0], ConduitError):
            print(responses[0].info)
            print(responses[0].detail)

        assert all([isinstance(r, Response) for r in responses]), (
            "All results should be Response objects"
        )
        assert isinstance(responses, list) and all(
            [isinstance(r, Response) for r in responses]
        ), "All results should be Response objects"
        return responses
