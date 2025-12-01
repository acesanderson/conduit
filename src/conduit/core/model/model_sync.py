from __future__ import annotations
from conduit.model.model_base import ModelBase
from conduit.model.clients.client import Client
from conduit.progress.wrappers import progress_display
from conduit.result.result import ConduitResult
from conduit.result.error import ConduitError
from pydantic import ValidationError
from typing import override
from time import time
import logging

logger = logging.getLogger(__name__)


class ModelSync(ModelBase):
    @override
    def get_client(self, model_name: str) -> Client:
        from conduit.model.models.modelstore import ModelStore

        return ModelStore.get_client(model_name, "sync")

    @progress_display
    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        try:
            # 1. CPU: Prepare
            request = self._prepare_request(query_input, **kwargs)

            # 2. I/O: Cache Read (Blocking)
            if kwargs.get("cache", False):
                cached = self._check_cache(request)
                if cached:
                    return cached

            # 3. I/O: Network Call (Blocking)
            start = time()
            raw_result, usage = self.client.query(request)
            stop = time()

            # 4. CPU: Process
            response = self._process_response(raw_result, usage, request, start, stop)

            # 5. I/O: Cache Write (Blocking)
            if kwargs.get("cache", False):
                self._save_cache(request, response)

            return response
        except ValidationError as e:
            try:
                request_request = request.model_dump()
            except Exception:
                request_request = {}
            conduit_error = ConduitError.from_exception(
                e,
                code="validation_error",
                category="client",
                request_request=request_request,
            )
            logger.error(f"Validation error: {conduit_error}")
            return conduit_error
        except Exception as e:
            try:
                request_request = request.model_dump()
            except Exception:
                request_request = {}
            conduit_error = ConduitError.from_exception(
                e,
                code="query_error",
                category="client",
                request_request=request_request,
            )
            logger.error(f"Error during query: {conduit_error}")
            return conduit_error

    @override
    def tokenize(self, text: str) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return self.client.tokenize(model=self.name, text=text)
