from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.clients.client_base import Client
from conduit.domain.result.result import ConduitResult
from conduit.domain.result.error import ConduitError
from pydantic import ValidationError
from typing import override
import logging

logger = logging.getLogger(__name__)


class ModelSync(ModelBase):
    @override
    def get_client(self, model_name: str) -> Client:
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.get_client(model_name, "sync")

    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        try:
            request = self._prepare_request(query_input, **kwargs)
            conduit_result = self._execute(request)
            return conduit_result
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
        return self.client.tokenize(model=self.model_name, text=text)


if __name__ == "__main__":
    model = ModelSync(model_name="gpt3")
    result = model.query("Hello, world!")
