from conduit.result.response import Response
from typing import Optional, Any
from pathlib import Path
from xdg_base_dirs import xdg_cache_home
import sqlite3
import json

DEFAULT_CACHE = Path(xdg_cache_home()) / "cache.db"


class ConduitCache:
    """
    SQLite-based cache for Conduit responses using JSON serialization.
    Automatically handles serialization/deserialization of Response and Request objects.
    """

    def __init__(self, db_path: str | Path = DEFAULT_CACHE):
        """
        Initialize cache with SQLite database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = str(db_path)
        self._connection = None
        self._create_table()

    @property
    def connection(self):
        """Get database connection with automatic reconnection"""
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path, check_same_thread=False)
        else:
            # Test if connection is still alive
            try:
                self._connection.execute("SELECT 1")
            except (sqlite3.ProgrammingError, sqlite3.OperationalError):
                # Connection is dead, create a new one
                try:
                    self._connection.close()
                except:
                    pass
                self._connection = sqlite3.connect(
                    self.db_path, check_same_thread=False
                )

        return self._connection

    def close(self):
        """Properly close the database connection"""
        if hasattr(self, "_connection") and self._connection:
            try:
                self._connection.close()
            except:
                pass
            finally:
                self._connection = None

    def _create_table(self):
        """Create cache table if it doesn't exist."""
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                cache_key TEXT PRIMARY KEY,
                response_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        self.connection.commit()

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve cached response by key.

        Args:
            cache_key: Unique cache key

        Returns:
            Deserialized response object or None if not found
        """
        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT response_data FROM cache WHERE cache_key = ?", (cache_key,)
            )
            result = cursor.fetchone()  # Use SAME cursor
            if result:
                response_data = result[0]
                return self._deserialize_response(response_data)
            return None
        except sqlite3.ProgrammingError as e:
            if "closed database" in str(e).lower():
                self._connection = None  # Reset connection
                return self.get(cache_key)  # Retry once
            raise

    def set(self, cache_key: str, response: Any):
        """
        Store response in cache.

        Args:
            cache_key: Unique cache key
            response: Response object to cache
        """
        serialized_data = self._serialize_response(response)
        cursor = self.connection.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO cache (cache_key, response_data)
            VALUES (?, ?)
        """,
            (cache_key, serialized_data),
        )
        self.connection.commit()

    # INTEGRATED CACHE METHODS - no need for external functions
    def check_for_model(self, Request):
        """
        Check if response exists in cache for the given Request.

        Args:
            Request: Request object containing request parameters

        Returns:
            Cached Response object or None if not found
        """
        cache_key = Request.generate_cache_key()
        return self.get(cache_key)

    def store_for_model(self, Request, response):
        """
        Store response in cache for the given Request.

        Args:
            Request: Request object containing request parameters
            response: Response object to cache
        """
        cache_key = Request.generate_cache_key()
        self.set(cache_key, response)

    def _serialize_response(self, response: Response) -> str:
        return json.dumps(response.to_cache_dict())

    def _deserialize_response(self, cached_response: str) -> Response:
        cache_dict = json.loads(cached_response)
        response = Response.from_cache_dict(cache_dict)
        return response

    def clear_cache(self):
        """Clear all cached responses."""
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM cache")
        self.connection.commit()

    def retrieve_cached_requests(self):
        """
        Get all cached requests for debugging.

        Returns:
            List of tuples: (cache_key, created_at)
        """
        cursor = self.connection.cursor()
        cursor.execute(
            "SELECT cache_key, created_at FROM cache ORDER BY created_at DESC"
        )
        return cursor.fetchall()

    def cache_stats(self):
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        cursor = self.connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache")
        total_entries = cursor.fetchone()[0]

        cursor.execute("SELECT SUM(LENGTH(response_data)) FROM cache")
        total_size = cursor.fetchone()[0] or 0

        return {
            "total_entries": total_entries,
            "total_size_bytes": total_size,
            "database_path": str(self.db_path),
        }

    def delete_cache_entry(self, cache_key: str):
        """
        Delete a specific cache entry.

        Args:
            cache_key: Key of the entry to delete
        """
        cursor = self.connection.cursor()
        cursor.execute("DELETE FROM cache WHERE cache_key = ?", (cache_key,))
        self.connection.commit()

    def cleanup_old_entries(self, days: int = 30):
        """
        Remove cache entries older than specified days.

        Args:
            days: Number of days to keep entries
        """
        cursor = self.connection.cursor()
        cursor.execute(
            """
            DELETE FROM cache 
            WHERE created_at < datetime('now', '-{} days')
        """.format(days)
        )
        self.connection.commit()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __repr__(self):
        """String representation."""
        stats = self.cache_stats()
        return f"ConduitCache(entries={stats['total_entries']}, size={stats['total_size_bytes']} bytes, db='{self.db_path}')"
