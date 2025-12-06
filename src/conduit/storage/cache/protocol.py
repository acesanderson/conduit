from __future__ import annotations
from typing import runtime_checkable, TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from conduit.domain.request.request import Request
    from conduit.domain.result.response import Response


@runtime_checkable
class ConduitCache(Protocol):
    def get(self, request: Request) -> Response | None:
        """
        Return a cached Response for this Request, or None if not present.
        """

        ...

    def set(self, request: Request, response: Response) -> None:
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
