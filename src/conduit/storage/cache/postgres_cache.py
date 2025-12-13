from __future__ import annotations
import time
from contextlib import AbstractContextManager
from collections.abc import Callable
from typing import TYPE_CHECKING
from conduit.domain.result.response import GenerationResponse
from conduit.domain.request.request import GenerationRequest
import logging

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from psycopg2.extensions import connection


class PostgresCache:
    """
    Postgres-backed implementation of ConduitCache.

    Schema expectation (one shared table):

        CREATE TABLE IF NOT EXISTS conduit_cache_entries (
            cache_name  text      NOT NULL,
            cache_key   text      NOT NULL,
            payload     jsonb     NOT NULL,
            created_at  timestamptz NOT NULL DEFAULT now(),
            updated_at  timestamptz NOT NULL DEFAULT now(),
            PRIMARY KEY (cache_name, cache_key)
        );

    Rows are partitioned by `cache_name`, so multiple logical caches
    (e.g. different projects or workflows) coexist in the same table.
    """

    def __init__(
        self,
        name: str,
        conn_factory: Callable[[], AbstractContextManager[connection]],
    ):
        """
        Usage:
                    from dbclients import get_postgres_client

                    cache = PostgresCache(
                        name="conduit_llm_cache",
                        conn_factory=get_postgres_client(
                            "context_db",
                            dbname="conduit",
                        ),
                    )
        """
        self.name = name
        self._conn_factory = conn_factory

        self._hits = 0
        self._misses = 0
        self._start_time = time.time()

        self._ensure_schema()

    # API per ConduitCache interface
    def get(self, request: GenerationRequest) -> GenerationResponse | None:
        key = self._request_to_key(request)

        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT payload
                FROM conduit_cache_entries
                WHERE cache_name = %s AND cache_key = %s
                """,
                (self.name, key),
            )
            row = cursor.fetchone()
            cursor.close()

        if row is None:
            self._misses += 1
            return None

        self._hits += 1
        payload = row[0]

        return GenerationResponse.model_validate(payload)

    def set(self, request: GenerationRequest, response: GenerationResponse) -> None:
        key = self._request_to_key(request)

        payload = response.model_dump_json()

        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO conduit_cache_entries (cache_name, cache_key, payload)
                VALUES (%s, %s, %s)
                ON CONFLICT (cache_name, cache_key)
                DO UPDATE SET
                    payload = EXCLUDED.payload,
                    updated_at = now()
                """,
                (self.name, key, payload),
            )
            conn.commit()
            cursor.close()

    def wipe(self) -> None:
        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM conduit_cache_entries
                WHERE cache_name = %s
                """,
                (self.name,),
            )
            conn.commit()
            cursor.close()

        self._hits = 0
        self._misses = 0
        self._start_time = time.time()

    @property
    def cache_stats(self) -> dict[str, object]:
        total_entries = self._count_entries()
        total_size_bytes = self._estimate_size_bytes()
        uptime_seconds = time.time() - self._start_time
        database_path = self._describe_database()
        created_at, updated_at = self._timestamp_bounds()

        return {
            "cache_name": self.name,
            "database_path": database_path,
            "total_entries": total_entries,
            "total_size_bytes": total_size_bytes,
            "uptime_seconds": float(uptime_seconds),
            "hits": self._hits,
            "misses": self._misses,
            "oldest_record": created_at,
            "latest_record": updated_at,
        }

    # Internal methods
    def heartbeat(self) -> bool:
        try:
            with self._conn_factory() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.fetchone()
                cursor.close()
            return True
        except Exception:
            return False

    @property
    def global_stats(self) -> dict[str, object]:
        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_entries,
                    COALESCE(SUM(pg_column_size(payload)), 0) AS total_size_bytes,
                    to_char(MIN(created_at), 'YYYY-MM-DD') AS oldest,
                    to_char(MAX(updated_at), 'YYYY-MM-DD') AS latest,
                    COUNT(DISTINCT cache_name) AS distinct_caches
                FROM conduit_cache_entries
                """
            )
            row = cursor.fetchone()
            cursor.close()

        return {
            "total_entries": int(row[0]),
            "total_size_bytes": int(row[1]),
            "oldest": row[2],
            "latest": row[3],
            "distinct_caches": int(row[4]),
        }

    def _ensure_schema(self) -> None:
        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS conduit_cache_entries (
                    cache_name  text      NOT NULL,
                    cache_key   text      NOT NULL,
                    payload     jsonb     NOT NULL,
                    created_at  timestamptz NOT NULL DEFAULT now(),
                    updated_at  timestamptz NOT NULL DEFAULT now(),
                    PRIMARY KEY (cache_name, cache_key)
                )
                """
            )
            conn.commit()
            cursor.close()

    def _count_entries(self) -> int:
        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*)
                FROM conduit_cache_entries
                WHERE cache_name = %s
                """,
                (self.name,),
            )
            row = cursor.fetchone()
            cursor.close()

        return int(row[0]) if row is not None else 0

    def _estimate_size_bytes(self) -> int:
        """
        Rough per-cache size estimate using pg_column_size(payload).
        This is an approximation, but good enough for stats.
        """
        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COALESCE(SUM(pg_column_size(payload)), 0)
                FROM conduit_cache_entries
                WHERE cache_name = %s
                """,
                (self.name,),
            )
            row = cursor.fetchone()
            cursor.close()

        return int(row[0]) if row is not None else 0

    def _timestamp_bounds(self) -> tuple[str | None, str | None]:
        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT
                    to_char(MIN(created_at), 'YYYY-MM-DD'),
                    to_char(MAX(updated_at), 'YYYY-MM-DD')
                FROM conduit_cache_entries
                WHERE cache_name = %s
                """,
                (self.name,),
            )
            row = cursor.fetchone()
            cursor.close()

        return row[0], row[1]

    def _describe_database(self) -> str:
        """
        Return a human-readable identifier for where this cache lives,
        e.g. 'siphon@hostname:5432'.
        """
        with self._conn_factory() as conn:
            params = conn.get_dsn_parameters()
            # keys: dbname, host, port, user, etc.
        dbname = params.get("dbname", "")
        host = params.get("host", "")
        port = params.get("port", "")
        return f"{dbname}@{host}:{port}"

    def _request_to_key(self, request: GenerationRequest) -> str:
        """
        Stable, deterministic key for a given GenerationRequest.

        You should ensure that GenerationRequest's fields used for LLM behavior
        (model, prompt, params, etc.) are fully represented here.
        """
        return request.generate_cache_key()

    def save_to_csv(self) -> None:
        """
        Generate csv with two columns: key, response.content
        """
        import csv

        with self._conn_factory() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT cache_key, payload
                FROM conduit_cache_entries
                WHERE cache_name = %s
                """,
                (self.name,),
            )
            rows = cursor.fetchall()
            cursor.close()
        with open(f"{self.name}_cache_export.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["key", "response_content"])
            for row in rows:
                cache_key = row[0]
                payload = row[1]
                response = GenerationResponse.model_validate(payload)
                writer.writerow([cache_key, response.content])

        print(f"Cache exported to {self.name}_cache_export.csv")
