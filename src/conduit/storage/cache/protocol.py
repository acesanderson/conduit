from __future__ import annotations
from typing import runtime_checkable, TYPE_CHECKING, Protocol
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.result.response import GenerationResponse


@runtime_checkable
class ConduitCache(Protocol):
    """
    Async interface for caching LLM responses.
    """

    async def get(self, request: GenerationRequest) -> GenerationResponse | None:
        """
        Return a cached Response for this Request, or None if not present.
        """
        ...

    async def get_all(self) -> list[GenerationResponse]:
        """
        Return all cached Responses for this cache instance.
        """
        ...

    async def set(
        self, request: GenerationRequest, response: GenerationResponse
    ) -> None:
        """
        Store or update the cached Response for this Request.
        """
        ...

    async def wipe(self) -> None:
        """
        Remove all entries for this cache instance.
        """
        ...

    async def cache_stats(self) -> dict[str, object]:
        """
        Return stats like total_entries, size, etc.
        """
        ...
