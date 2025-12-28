from __future__ import annotations
from typing import runtime_checkable, TYPE_CHECKING, Protocol
from collections.abc import Callable
from contextlib import AbstractContextManager
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from conduit.domain.request.request import GenerationRequest
    from conduit.domain.result.response import GenerationResponse
    from psycopg2.extensions import connection


@runtime_checkable
class ConduitCache(Protocol):
    def __init__(
        self,
        project_name: str,
        conn_factory: Callable[[], AbstractContextManager[connection]],
    ) -> None:
        """
        Initialize a ConduitCache instance.

        Args:
            name (str): The name of the cache backend to use.
        """

        ...

    def get(self, request: GenerationRequest) -> GenerationResponse | None:
        """
        Return a cached Response for this Request, or None if not present.
        """

        ...

    def set(self, request: GenerationRequest, response: GenerationResponse) -> None:
        """
        Store or update the cached Response for this Request.
        """

        ...

    def wipe(self) -> None:
        """
        Remove all entries for this cache instance.
        """

        ...

    @property
    def cache_stats(self) -> dict[str, object]:
        """
        Return a JSON-serializable dict with basic stats, e.g.:

        {
            "cache_name": str,
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "database_path": str(self.db_path),
            "uptime_seconds": float,
            "hits": int,
            "misses": int,
        }
        """
        ...
