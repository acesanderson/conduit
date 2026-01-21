from __future__ import annotations
import json
import logging
import asyncio
from typing import Any
from pydantic import TypeAdapter

from conduit.domain.conversation.session import Session
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.message.message import MessageUnion, Message

# Type checking import
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from asyncpg import Pool

logger = logging.getLogger(__name__)


class AsyncPostgresSessionRepository:
    """
    Async Project-Scoped Session Repository (DAG Architecture).
    Manages its own lazy connection pool to support restarts/different loops.
    """

    def __init__(self, project_name: str, db_name: str = "conduit"):
        self.project_name = project_name
        self.db_name = db_name
        self._message_adapter = TypeAdapter(MessageUnion)

        # Lazy pool management
        self._pool: Pool | None = None
        self._pool_loop: asyncio.AbstractEventLoop | None = None

    async def _get_pool(self) -> Pool:
        """
        Get or create a connection pool attached to the current event loop.
        """
        current_loop = asyncio.get_running_loop()

        # If we have a pool and the loop hasn't changed, reuse it
        if (
            self._pool
            and self._pool_loop is current_loop
            and not current_loop.is_closed()
        ):
            return self._pool

        # Otherwise (re)initialize
        logger.debug(f"Initializing asyncpg pool for repository '{self.project_name}'")
        from dbclients.clients.postgres import get_postgres_client

        # get_postgres_client("async", ...) returns a coroutine factory for the pool
        pool_factory = await get_postgres_client(
            client_type="async", dbname=self.db_name
        )

        self._pool = pool_factory
        self._pool_loop = current_loop

        # Ensure schema exists on new connection
        await self.initialize()
        return self._pool

    async def initialize(self) -> None:
        """Idempotent schema creation."""
        # Use the pool directly if it exists, or just return (it's called by _get_pool anyway)
        if not self._pool:
            return

        async with self._pool.acquire() as conn:
            await conn.execute("""
                -- 1. Scoped Sessions
                CREATE TABLE IF NOT EXISTS conduit_sessions (
                    session_id TEXT PRIMARY KEY,
                    project_name TEXT NOT NULL,
                    leaf_message_id TEXT,
                    title TEXT,
                    metadata JSONB DEFAULT '{}',
                    created_at BIGINT,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_project ON conduit_sessions(project_name);
                CREATE INDEX IF NOT EXISTS idx_sessions_updated ON conduit_sessions(last_updated);

                -- 2. DAG Messages (Graph Storage)
                CREATE TABLE IF NOT EXISTS conduit_messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES conduit_sessions(session_id) ON DELETE CASCADE,
                    predecessor_id TEXT REFERENCES conduit_messages(message_id),
                    role TEXT NOT NULL,
                    content JSONB,
                    created_at BIGINT,
                    
                    -- Specialized columns for query ease/indexing
                    metadata JSONB DEFAULT '{}',
                    tool_calls JSONB,
                    images JSONB,
                    audio JSONB,
                    parsed JSONB
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session ON conduit_messages(session_id);
                CREATE INDEX IF NOT EXISTS idx_messages_predecessor ON conduit_messages(predecessor_id);
            """)

    @property
    async def last(self) -> Conversation | None:
        """
        Fetches the most recently updated session.
        """
        pool = await self._get_pool()
        query = """
            SELECT session_id 
            FROM conduit_sessions 
            WHERE project_name = $1 
            ORDER BY last_updated DESC 
            LIMIT 1
        """
        async with pool.acquire() as conn:
            session_id = await conn.fetchval(query, self.project_name)

        if not session_id:
            return None

        session = await self.get_session(session_id)
        return session.conversation if session else None

    # --- Read Operations ---

    async def get_session(self, session_id: str) -> Session | None:
        pool = await self._get_pool()
        q_session = """
            SELECT session_id, leaf_message_id, title, metadata, created_at
            FROM conduit_sessions 
            WHERE session_id = $1 AND project_name = $2
        """
        q_messages = """
            SELECT * FROM conduit_messages WHERE session_id = $1
        """

        async with pool.acquire() as conn:
            row = await conn.fetchrow(q_session, session_id, self.project_name)
            if not row:
                return None
            msg_rows = await conn.fetch(q_messages, session_id)

        message_dict = {}
        for r in msg_rows:
            try:
                msg_obj = self._row_to_message(r)
                message_dict[msg_obj.message_id] = msg_obj
            except Exception as e:
                logger.error(f"Failed to hydrate message {r['message_id']}: {e}")
                continue

        session = Session(
            session_id=row["session_id"],
            leaf=row["leaf_message_id"],
            created_at=row["created_at"],
            message_dict=message_dict,
        )
        return session

    async def get_conversation(self, leaf_message_id: str) -> Conversation | None:
        pool = await self._get_pool()
        query = """
        WITH RECURSIVE conversation_tree AS (
            SELECT m.*, 1 as depth
            FROM conduit_messages m
            JOIN conduit_sessions s ON m.session_id = s.session_id
            WHERE m.message_id = $1 AND s.project_name = $2
            
            UNION ALL
            
            SELECT p.*, ct.depth + 1
            FROM conduit_messages p
            INNER JOIN conversation_tree ct ON p.message_id = ct.predecessor_id
        )
        SELECT * FROM conversation_tree ORDER BY depth DESC;
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, leaf_message_id, self.project_name)

        if not rows:
            return None

        messages = [self._row_to_message(r) for r in rows]
        return Conversation(messages=messages)

    async def get_message(self, message_id: str) -> Message | None:
        pool = await self._get_pool()
        query = """
            SELECT m.*
            FROM conduit_messages m
            JOIN conduit_sessions s ON m.session_id = s.session_id
            WHERE m.message_id = $1 AND s.project_name = $2
        """
        async with pool.acquire() as conn:
            row = await conn.fetchrow(query, message_id, self.project_name)

        if not row:
            return None

        return self._row_to_message(row)

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        pool = await self._get_pool()
        query = """
            SELECT session_id, title, created_at, leaf_message_id, last_updated
            FROM conduit_sessions
            WHERE project_name = $1
            ORDER BY last_updated DESC
            LIMIT $2
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(query, self.project_name, limit)

            return [
                {
                    "session_id": r["session_id"],
                    "title": r["title"] or "Untitled Session",
                    "created_at": r["created_at"],
                    "last_updated": r["last_updated"],
                    "leaf_id": r["leaf_message_id"],
                }
                for r in rows
            ]

    # --- Write Operations ---

    async def save_session(self, session: Session, name: str | None = None) -> None:
        pool = await self._get_pool()

        # 1. Topological Sort
        sorted_msgs = self._topological_sort(session.message_dict)

        # 2. Serialize
        msg_records = []
        for msg in sorted_msgs:
            d = msg.model_dump(mode="json")
            msg_records.append(
                (
                    d["message_id"],
                    d["session_id"],
                    d.get("predecessor_id"),
                    d["role"],
                    json.dumps(d.get("content")),
                    d["created_at"],
                    json.dumps(d.get("metadata") or {}),
                    json.dumps(d.get("tool_calls")),
                    json.dumps(d.get("images")),
                    json.dumps(d.get("audio")),
                    json.dumps(d.get("parsed")),
                )
            )

        upsert_session_sql = """
            INSERT INTO conduit_sessions (session_id, project_name, leaf_message_id, title, created_at, last_updated)
            VALUES ($1, $2, $3, $4, $5, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                leaf_message_id = EXCLUDED.leaf_message_id,
                title = COALESCE($4, conduit_sessions.title),
                last_updated = NOW(),
                project_name = EXCLUDED.project_name;
        """

        upsert_message_sql = """
            INSERT INTO conduit_messages (
                message_id, session_id, predecessor_id, role, content, 
                created_at, metadata, tool_calls, images, audio, parsed
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
            ON CONFLICT (message_id) DO NOTHING;
        """

        async with pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute(
                    upsert_session_sql,
                    session.session_id,
                    self.project_name,
                    session.leaf,
                    name,
                    session.created_at,
                )

                if msg_records:
                    await conn.executemany(upsert_message_sql, msg_records)

    # --- Maintenance Operations ---

    async def delete_session(self, session_id: str) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conduit_sessions WHERE session_id = $1 AND project_name = $2",
                session_id,
                self.project_name,
            )

    async def wipe(self) -> None:
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conduit_sessions WHERE project_name = $1",
                self.project_name,
            )

    # --- Helpers ---

    def _row_to_message(self, row: Any) -> Message:
        msg_data = dict(row)
        for field in ["content", "tool_calls", "images", "audio", "parsed", "metadata"]:
            if isinstance(msg_data.get(field), str):
                try:
                    msg_data[field] = json.loads(msg_data[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        if "role" in msg_data:
            msg_data["role_str"] = msg_data["role"]
        return self._message_adapter.validate_python(msg_data)

    def _topological_sort(self, message_dict: dict[str, Message]) -> list[Message]:
        # (Same as before)
        present_ids = set(message_dict.keys())
        children_map = {mid: [] for mid in present_ids}
        roots = []

        for mid, msg in message_dict.items():
            pid = msg.predecessor_id
            if pid and pid in present_ids:
                children_map[pid].append(mid)
            else:
                roots.append(mid)

        sorted_list = []
        queue = roots[:]

        while queue:
            current_id = queue.pop(0)
            if current_id in message_dict:
                sorted_list.append(message_dict[current_id])
                queue.extend(children_map.get(current_id, []))

        if len(sorted_list) < len(message_dict):
            visited = set(m.message_id for m in sorted_list)
            for m in message_dict.values():
                if m.message_id not in visited:
                    sorted_list.append(m)

        return sorted_list

    # --- Async Context Management ---
    async def aclose(self) -> None:
        """Close the connection pool."""
        if self._pool:
            try:
                await self._pool.close()
                logger.info(f"Closed pool for repository '{self.project_name}'")
            except Exception as e:
                logger.warning(f"Error closing repository pool: {e}")
            finally:
                self._pool = None

    async def __aenter__(self) -> AsyncPostgresSessionRepository:
        """Initialize the pool and schema on entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Ensure the pool is closed on exit."""
        await self.aclose()


# Factory function becomes synchronous (just instantiates the lazy object)
def get_async_repository(project_name: str) -> AsyncPostgresSessionRepository:
    """
    Factory function to get an initialized AsyncPostgresSessionRepository.
    Requires 'asyncpg' installed and configured in dbclients.
    """
    # Simply return the object. It will connect on first use.
    return AsyncPostgresSessionRepository(project_name=project_name)
