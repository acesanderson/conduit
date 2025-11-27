from __future__ import annotations
from conduit.model.model_base import ModelBase
from conduit.progress.wrappers import progress_display
from conduit.result.result import ConduitResult
from conduit.result.error import ConduitError
from typing import override
from pathlib import Path
from time import time
import logging

logger = logging.getLogger(__name__)
dir_path = Path(__file__).resolve().parent


"""
def _prepare_request(self, query_input, kwargs) -> Request:
def _process_response(
def _check_cache(self, request: Request) -> Response | None:
def _save_cache(self, request: Request, response: Response):
"""


class ModelSync(ModelBase):
    @progress_display
    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        try:
            breakpoint()
            # 1. CPU: Prepare
            request = self._prepare_request(query_input, **kwargs)

            # 2. I/O: Cache Read (Blocking)
            if kwargs.get("cache", False):
                cached = self._check_cache(request)
                if cached:
                    return cached

            # 3. I/O: Network Call (Blocking)
            start = time()
            raw_result, usage = self._client.query(request)
            stop = time()

            # 4. CPU: Process
            response = self._process_response(raw_result, usage, request, start, stop)

            # 5. I/O: Cache Write (Blocking)
            if kwargs.get("cache", False):
                self._save_cache(request, response)

            return response
        except ValidationError as e:
            conduit_error = ConduitError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Validation error: {conduit_error}")
            return conduit_error
        except Exception as e:
            conduit_error = ConduitError.from_exception(
                e,
                code="query_error",
                category="client",
                request_request=request.model_dump() if request else {},
            )
            logger.error(f"Error during query: {conduit_error}")
            return conduit_error

    @override
    def tokenize(self, text: str) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return self._client.tokenize(model=self.model, text=text)


if __name__ == "__main__":
    # Instantiate model (requires valid configuration/API key for gpt-4o)
    model = ModelSync("gpt3")

    # 1. Verify Request object construction (bypasses API call)
    response = model.query(query_input="Explain quantum physics")
    print(f"Response: {response}")

    # 2. Check tokenization interface
    print(f"Tokens for input: {model.tokenize('Explain quantum physics')}")
