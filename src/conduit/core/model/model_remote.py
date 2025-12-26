"""
Remote Model implementations that use RemoteClient for server-based execution.

Provides both sync and async interfaces with remote server capabilities like
ping, status checks, and batch operations.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, override
import logging

from conduit.core.model.model_sync import ModelSync
from conduit.core.model.model_async import ModelAsync
from conduit.core.clients.remote.client import RemoteClient
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions

if TYPE_CHECKING:
    from headwater_api.classes.server_classes.status import StatusResponse
    from collections.abc import Sequence
    from conduit.domain.request.query_input import QueryInput
    from conduit.domain.result.result import GenerationResult

logger = logging.getLogger(__name__)


class RemoteModelSync(ModelSync):
    """
    Synchronous remote model that uses RemoteClient for server-based execution.

    Inherits all ModelSync functionality while adding remote-specific capabilities
    like ping(), get_status(), and batch() operations.
    """

    def __init__(
        self,
        model: str,
        params: GenerationParams | None = None,
        options: ConduitOptions | None = None,
        **kwargs: Any,
    ):
        """
        Initialize the synchronous remote model.

        Args:
            model: Model name/alias (e.g., "claude-3-sonnet", "gpt-4o")
            params: LLM parameters (temperature, max_tokens, etc.)
            options: Runtime configuration (caching, console, etc.)
            **kwargs: Additional parameters merged into GenerationParams
        """
        # Create RemoteClient and inject it
        remote_client = RemoteClient()
        super().__init__(
            model=model, params=params, options=options, client=remote_client, **kwargs
        )

    def ping(self) -> bool:
        """
        Ping the remote server to check connectivity.

        Returns:
            Server ping response with timing and status information
        """
        if not isinstance(self._impl.client, RemoteClient):
            raise TypeError("ping() requires RemoteClient")
        return self._run_sync(self._impl.client.ping())

    def get_status(self) -> StatusResponse:
        """
        Get the current status of the remote server.

        Returns:
            Server status information including health and model availability
        """
        if not isinstance(self._impl.client, RemoteClient):
            raise TypeError("get_status() requires RemoteClient")
        return self._run_sync(self._impl.client.get_status())

    def batch(self, query_inputs: Sequence[QueryInput]) -> list[GenerationResult]:
        """
        Process multiple queries in a batch on the remote server.

        Args:
            query_inputs: Sequence of input queries to process

        Returns:
            List of GenerationResults corresponding to each input
        """
        if not isinstance(self._impl.client, RemoteClient):
            raise TypeError("batch() requires RemoteClient")
        # Note: This will need to be implemented in RemoteClient
        return self._run_sync(self._impl.client.batch(query_inputs))

    @property
    def is_remote(self) -> bool:
        """Check if this model uses remote execution."""
        return True


class RemoteModelAsync(ModelAsync):
    """
    Asynchronous remote model that uses RemoteClient for server-based execution.

    Inherits all ModelAsync functionality while adding remote-specific capabilities
    like ping(), get_status(), and batch() operations.
    """

    def __init__(self, model: str):
        """
        Initialize the asynchronous remote model.

        Args:
            model: Model name/alias (e.g., "claude-3-sonnet", "gpt-4o")
        """
        # Create RemoteClient and inject it
        remote_client = RemoteClient()
        super().__init__(model=model, client=remote_client)

    async def ping(self) -> bool:
        """
        Ping the remote server to check connectivity.

        Returns:
            Server ping response with timing and status information
        """
        if not isinstance(self.client, RemoteClient):
            raise TypeError("ping() requires RemoteClient")
        return await self.client.ping()

    async def get_status(self) -> StatusResponse:
        """
        Get the current status of the remote server.

        Returns:
            Server status information including health and model availability
        """
        if not isinstance(self.client, RemoteClient):
            raise TypeError("get_status() requires RemoteClient")
        return await self.client.get_status()

    async def batch(self, query_inputs: Sequence[QueryInput]) -> list[GenerationResult]:
        """
        Process multiple queries in a batch on the remote server.

        Args:
            query_inputs: Sequence of input queries to process

        Returns:
            List of GenerationResults corresponding to each input
        """
        if not isinstance(self.client, RemoteClient):
            raise TypeError("batch() requires RemoteClient")
        # Note: This will need to be implemented in RemoteClient
        return await self.client.batch(query_inputs)

    @property
    def is_remote(self) -> bool:
        """Check if this model uses remote execution."""
        return True


# Factory functions for convenient instantiation
def remote_model_sync(
    model: str,
    params: GenerationParams | None = None,
    options: ConduitOptions | None = None,
    **kwargs: Any,
) -> RemoteModelSync:
    """
    Factory function to create a synchronous remote model.

    Args:
        model: Model name/alias (e.g., "claude-3-sonnet", "gpt-4o")
        params: LLM parameters (temperature, max_tokens, etc.)
        options: Runtime configuration (caching, console, etc.)
        **kwargs: Additional parameters merged into GenerationParams

    Returns:
        Configured RemoteModelSync instance
    """
    return RemoteModelSync(model=model, params=params, options=options, **kwargs)


def remote_model_async(model: str) -> RemoteModelAsync:
    """
    Factory function to create an asynchronous remote model.

    Args:
        model: Model name/alias (e.g., "claude-3-sonnet", "gpt-4o")

    Returns:
        Configured RemoteModelAsync instance
    """
    return RemoteModelAsync(model=model)


if __name__ == "__main__":
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.utils.progress.verbosity import Verbosity

    # Instantiate RemoteModelSync
    model = RemoteModelSync(
        model="gpt-4o",
        params=GenerationParams(model="gpt-4o", temperature=0.7),
        options=ConduitOptions(project_name="test", verbosity=Verbosity.PROGRESS),
    )

    # Check if remote
    print(f"Is remote: {model.is_remote}")

    # Try ping (will fail if no server, but tests instantiation)
    try:
        result = model.ping()
        print(f"Ping result: {result}")
    except Exception as e:
        print(f"Ping failed (expected if no server): {type(e).__name__}")

    # Query
    result = model.query("Hello, remote model!")
    print(f"Query result: {result}")
