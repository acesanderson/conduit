from __future__ import annotations
import logging
import asyncio
from datetime import datetime, date
from typing import TYPE_CHECKING
from conduit.storage.odometer.token_event import TokenEvent

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from asyncpg import Pool


class AsyncPostgresOdometer:
    """
    Async Postgres backend for Odometer.
    Manages its own lazy connection pool to support restarts/different loops.
    """

    def __init__(self, db_name: str = "conduit"):
        self.db_name = db_name
        self._pool: Pool | None = None
        self._pool_loop: asyncio.AbstractEventLoop | None = None

    async def _get_pool(self) -> Pool:
        current_loop = asyncio.get_running_loop()

        if (
            self._pool
            and self._pool_loop is current_loop
            and not current_loop.is_closed()
        ):
            return self._pool

        logger.debug("Initializing asyncpg pool for Odometer")
        from dbclients.clients.postgres import get_postgres_client

        # get_postgres_client("async") returns the pool
        pool_factory = await get_postgres_client(
            client_type="async", dbname=self.db_name
        )
        self._pool = pool_factory
        self._pool_loop = current_loop

        # Ensure schema exists (lightweight check)
        await self._ensure_schema()
        return self._pool

    async def _ensure_schema(self):
        if not self._pool:
            return
        create_sql = """
        CREATE TABLE IF NOT EXISTS token_events (
            id SERIAL PRIMARY KEY,
            provider VARCHAR(50) NOT NULL,
            model VARCHAR(200) NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            timestamp BIGINT NOT NULL,
            host VARCHAR(100) NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
        async with self._pool.acquire() as conn:
            await conn.execute(create_sql)

    async def store_events(self, events: list[TokenEvent]) -> None:
        if not events:
            return

        pool = await self._get_pool()
        insert_sql = """
        INSERT INTO token_events (provider, model, input_tokens, output_tokens, timestamp, host)
        VALUES ($1, $2, $3, $4, $5, $6)
        """
        records = [
            (e.provider, e.model, e.input_tokens, e.output_tokens, e.timestamp, e.host)
            for e in events
        ]
        async with pool.acquire() as conn:
            await conn.executemany(insert_sql, records)
            logger.debug(f"Async Odometer flush: stored {len(events)} events")

    # --- Read Methods for Reporting ---

    async def get_overall_stats(self) -> dict:
        """Get overall statistics."""
        pool = await self._get_pool()
        query = """
        SELECT 
            COUNT(*) as requests,
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            COUNT(DISTINCT provider) as unique_providers,
            COUNT(DISTINCT model) as unique_models
        FROM token_events
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query)
            if not row:
                return {}

            # Handle potential None returns from SUM on empty table
            t_input = row["total_input"] or 0
            t_output = row["total_output"] or 0

            return {
                "requests": row["requests"] or 0,
                "input": t_input,
                "output": t_output,
                "total_tokens": t_input + t_output,
                "providers": row["unique_providers"] or 0,
                "models": row["unique_models"] or 0,
            }

    async def get_aggregates(
        self,
        group_by: str,  # "provider", "model", "host", "date"
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get aggregated statistics."""
        pool = await self._get_pool()

        valid_groups = ["provider", "model", "host", "date"]
        if group_by not in valid_groups:
            raise ValueError(f"group_by must be one of {valid_groups}")

        where_conditions = []
        params = []
        param_idx = 1

        if start_date:
            start_ts = int(
                datetime.combine(start_date, datetime.min.time()).timestamp()
            )
            where_conditions.append(f"timestamp >= ${param_idx}")
            params.append(start_ts)
            param_idx += 1

        if end_date:
            end_ts = int(datetime.combine(end_date, datetime.max.time()).timestamp())
            where_conditions.append(f"timestamp <= ${param_idx}")
            params.append(end_ts)
            param_idx += 1

        if group_by == "date":
            # Postgres specific: convert epoch to date
            group_clause = "DATE(to_timestamp(timestamp))"
            select_clause = f"{group_clause} as group_key"
        else:
            group_clause = group_by
            select_clause = f"{group_by} as group_key"

        base_query = f"""
        SELECT 
            {select_clause},
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            SUM(input_tokens + output_tokens) as total_tokens,
            COUNT(*) as event_count
        FROM token_events
        """

        if where_conditions:
            base_query += f" WHERE {' AND '.join(where_conditions)}"

        base_query += f" GROUP BY {group_clause} ORDER BY total_tokens DESC"

        async with pool.acquire() as conn:
            rows = await conn.fetch(base_query, *params)

            result = {}
            for row in rows:
                key = str(row["group_key"])
                result[key] = {
                    "input": row["total_input"],
                    "output": row["total_output"],
                    "total": row["total_tokens"],
                    "events": row["event_count"],
                }
            return result
