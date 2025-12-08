from __future__ import annotations
from conduit.core.model.model_base import ModelBase
from conduit.core.clients.client_base import Client
from conduit.domain.result.result import ConduitResult
from typing import override, TYPE_CHECKING
import logging

if TYPE_CHECKING:
    from conduit.domain.message.message import Message

logger = logging.getLogger(__name__)


class ModelSync(ModelBase):
    @override
    def get_client(self, model_name: str) -> Client:
        from conduit.core.model.models.modelstore import ModelStore

        return ModelStore.get_client(model_name, "sync")

    @override
    def query(self, query_input=None, **kwargs) -> ConduitResult:
        request = self._prepare_request(query_input, **kwargs)
        conduit_result = self._execute(request)
        return conduit_result

    @override
    def tokenize(self, payload: str | list[Message]) -> int:
        """
        Get the token length for the given model.
        Implementation at the client level.
        """
        return self.client.tokenize(model=self.model_name, payload=payload)


if __name__ == "__main__":
    model = ModelSync(model_name="gpt3", cache="testing")
    result = model.query("Hello, world!")
    print(result)
    ts = model.tokenize("i am the very model of a modern major general")
    print(ts)
