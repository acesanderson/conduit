from datetime import date, datetime
import logging
from typing import override

from conduit.storage.odometer.pgres.persistence_backend import PersistenceBackend
from conduit.storage.odometer.token_event import TokenEvent
from dbclients.clients.postgres import get_postgres_client

logger = logging.getLogger(__name__)

get_db_connection = get_postgres_client("context_db", dbname="conduit")


class PostgresBackend(PersistenceBackend):
    def __init__(self, connection_string: str | None = None):
        self.conn_string: str | None = connection_string
        self._initialized: bool = False
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return

        create_table_sql = """
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
        
        CREATE INDEX IF NOT EXISTS idx_token_events_timestamp ON token_events(timestamp);
        CREATE INDEX IF NOT EXISTS idx_token_events_provider ON token_events(provider);
        CREATE INDEX IF NOT EXISTS idx_token_events_model ON token_events(model);
        CREATE INDEX IF NOT EXISTS idx_token_events_host ON token_events(host);
        CREATE INDEX IF NOT EXISTS idx_token_events_provider_model ON token_events(provider, model);
        """

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(create_table_sql)
                    conn.commit()
                    logger.info("Database schema initialized successfully")
                    self._initialized = True
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")

    @override
    def store_events(self, events: list[TokenEvent]) -> None:
        """Store raw token events using bulk insert."""
        if not events:
            return

        insert_sql = """
        INSERT INTO token_events (provider, model, input_tokens, output_tokens, timestamp, host)
        VALUES (%s, %s, %s, %s, %s, %s)
        """

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    event_data = [
                        (
                            event.provider,
                            event.model,
                            event.input_tokens,
                            event.output_tokens,
                            event.timestamp,
                            event.host,
                        )
                        for event in events
                    ]
                    cursor.executemany(insert_sql, event_data)
                    conn.commit()
                    logger.info("Stored %d token events", len(events))
        except Exception as e:
            logger.error(f"Failed to store events: {e}")
            raise

    @override
    def get_events(
        self,
        start_date: date | None = None,
        end_date: date | None = None,
        provider: str | None = None,
        model: str | None = None,
        host: str | None = None,
    ) -> list[TokenEvent]:
        """Query token events with filters."""
        where_conditions: list[str] = []
        params: list[object] = []

        if start_date:
            start_timestamp = int(
                datetime.combine(start_date, datetime.min.time()).timestamp()
            )
            where_conditions.append("timestamp >= %s")
            params.append(start_timestamp)

        if end_date:
            end_timestamp = int(
                datetime.combine(end_date, datetime.max.time()).timestamp()
            )
            where_conditions.append("timestamp <= %s")
            params.append(end_timestamp)

        if provider:
            where_conditions.append("provider = %s")
            params.append(provider)

        if model:
            where_conditions.append("model = %s")
            params.append(model)

        if host:
            where_conditions.append("host = %s")
            params.append(host)

        base_query = (
            "SELECT provider, model, input_tokens, output_tokens, timestamp, host "
            "FROM token_events"
        )
        if where_conditions:
            query = f"{base_query} WHERE {' AND '.join(where_conditions)}"
        else:
            query = base_query

        query += " ORDER BY timestamp DESC"

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()

            events: list[TokenEvent] = []
            for row in rows:
                events.append(
                    TokenEvent(
                        provider=row[0],
                        model=row[1],
                        input_tokens=row[2],
                        output_tokens=row[3],
                        timestamp=row[4],
                        host=row[5],
                    )
                )

            logger.info("Retrieved %d token events", len(events))
            return events

        except Exception as e:
            logger.error(f"Failed to retrieve events: {e}")
            raise

    @override
    def get_aggregates(
        self,
        group_by: str,  # "provider", "model", "host", "date"
        start_date: date | None = None,
        end_date: date | None = None,
    ) -> dict[str, dict[str, int]]:
        """Get aggregated statistics."""
        valid_groups = ["provider", "model", "host", "date"]
        if group_by not in valid_groups:
            raise ValueError(f"group_by must be one of {valid_groups}")

        where_conditions: list[str] = []
        params: list[object] = []

        if start_date:
            start_timestamp = int(
                datetime.combine(start_date, datetime.min.time()).timestamp()
            )
            where_conditions.append("timestamp >= %s")
            params.append(start_timestamp)

        if end_date:
            end_timestamp = int(
                datetime.combine(end_date, datetime.max.time()).timestamp()
            )
            where_conditions.append("timestamp <= %s")
            params.append(end_timestamp)

        if group_by == "date":
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
            query = f"{base_query} WHERE {' AND '.join(where_conditions)}"
        else:
            query = base_query

        query += f" GROUP BY {group_clause} ORDER BY total_tokens DESC"

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query, params)
                    rows = cursor.fetchall()

            result: dict[str, dict[str, int]] = {}
            for row in rows:
                group_key = str(row[0])
                result[group_key] = {
                    "input": row[1],
                    "output": row[2],
                    "total": row[3],
                    "events": row[4],
                }

            logger.info(
                "Retrieved aggregates for %d groups by %s", len(result), group_by
            )
            return result

        except Exception as e:
            logger.error(f"Failed to retrieve aggregates: {e}")
            raise

    def health_check(self) -> bool:
        """Test if backend is accessible and healthy."""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT 1")
                    return cursor.fetchone()[0] == 1
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False

    def get_total_events(self) -> int:
        """Get total number of events stored."""
        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM token_events")
                    return cursor.fetchone()[0]
        except Exception as e:
            logger.error(f"Failed to get total events: {e}")
            return 0

    def get_overall_stats(self) -> dict:
        """Get overall statistics without fetching individual events."""
        query = """
        SELECT 
            COUNT(*) as requests,
            SUM(input_tokens) as total_input,
            SUM(output_tokens) as total_output,
            COUNT(DISTINCT provider) as unique_providers,
            COUNT(DISTINCT model) as unique_models
        FROM token_events
        """

        try:
            with get_db_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(query)
                    row = cursor.fetchone()
                    return {
                        "requests": row[0],
                        "input": row[1],
                        "output": row[2],
                        "total_tokens": row[1] + row[2],
                        "providers": row[3],
                        "models": row[4],
                    }
        except Exception as e:
            logger.error(f"Failed to get overall stats: {e}")
            raise
