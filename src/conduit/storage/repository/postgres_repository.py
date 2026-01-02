import json
import logging
from typing import Any
from pydantic import TypeAdapter

from conduit.domain.conversation.session import Session
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.message.message import MessageUnion, Message

logger = logging.getLogger(__name__)


class AsyncPostgresSessionRepository:
    """
    Async Project-Scoped Session Repository (DAG Architecture).

    Architecture:
    - conduit_sessions: Partitioned by 'project_name'.
    - conduit_messages: Stored as a Graph (DAG) linked to sessions.
    """

    def __init__(self, project_name: str, pool: Any):
        """
        :param project_name: The scoping key (e.g., "summarizer", "chatbot-v1").
        :param pool: An initialized asyncpg.Pool instance.
        """
        self.project_name = project_name
        self.pool = pool
        self._message_adapter = TypeAdapter(MessageUnion)

    async def initialize(self) -> None:
        """Idempotent schema creation."""
        async with self.pool.acquire() as conn:
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

    # --- Read Operations ---

    async def get_session(self, session_id: str) -> Session | None:
        """
        Loads a Session and its full message graph.
        Enforces project ownership.
        """
        q_session = """
            SELECT session_id, leaf_message_id, title, metadata, created_at
            FROM conduit_sessions 
            WHERE session_id = $1 AND project_name = $2
        """
        q_messages = """
            SELECT * FROM conduit_messages WHERE session_id = $1
        """

        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(q_session, session_id, self.project_name)
            if not row:
                return None

            # Fetch all messages for this session
            msg_rows = await conn.fetch(q_messages, session_id)

        # Rehydrate Messages
        message_dict = {}
        for r in msg_rows:
            try:
                msg_obj = self._row_to_message(r)
                message_dict[msg_obj.message_id] = msg_obj
            except Exception as e:
                logger.error(f"Failed to hydrate message {r['message_id']}: {e}")
                continue

        # We construct the session using the data from DB
        session = Session(
            session_id=row["session_id"],
            leaf=row["leaf_message_id"],
            created_at=row["created_at"],
            message_dict=message_dict,
        )
        return session

    async def get_conversation(self, leaf_message_id: str) -> Conversation | None:
        """
        Rehydrates a linear Conversation view by walking backwards from a leaf.
        """
        # Recursive CTE to walk up the tree
        # We join on sessions to ensure project scoping logic applies
        query = """
        WITH RECURSIVE conversation_tree AS (
            -- Base Case: The Leaf
            SELECT m.*, 1 as depth
            FROM conduit_messages m
            JOIN conduit_sessions s ON m.session_id = s.session_id
            WHERE m.message_id = $1 AND s.project_name = $2
            
            UNION ALL
            
            -- Recursive Step: The Predecessor
            SELECT p.*, ct.depth + 1
            FROM conduit_messages p
            INNER JOIN conversation_tree ct ON p.message_id = ct.predecessor_id
        )
        SELECT * FROM conversation_tree ORDER BY depth DESC;
        """

        async with self.pool.acquire() as conn:
            rows = await conn.fetch(query, leaf_message_id, self.project_name)

        if not rows:
            return None

        messages = [self._row_to_message(r) for r in rows]
        return Conversation(messages=messages)

    async def get_message(self, message_id: str) -> Message | None:
        """
        Fetch a single specific message by ID.
        Enforces project ownership via JOIN.
        """
        query = """
            SELECT m.*
            FROM conduit_messages m
            JOIN conduit_sessions s ON m.session_id = s.session_id
            WHERE m.message_id = $1 AND s.project_name = $2
        """
        async with self.pool.acquire() as conn:
            row = await conn.fetchrow(query, message_id, self.project_name)

        if not row:
            return None

        return self._row_to_message(row)

    async def list_sessions(self, limit: int = 20) -> list[dict[str, Any]]:
        """Lists sessions for the current project."""
        query = """
            SELECT session_id, title, created_at, leaf_message_id, last_updated
            FROM conduit_sessions
            WHERE project_name = $1
            ORDER BY last_updated DESC
            LIMIT $2
        """
        async with self.pool.acquire() as conn:
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
        """
        Upserts Session (scoped to Project) and its Messages.
        """
        # 1. Topological Sort for FK validity (Parent before Child)
        sorted_msgs = self._topological_sort(session.message_dict)

        # 2. Serialize Messages for Batch Insert
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

        async with self.pool.acquire() as conn:
            async with conn.transaction():
                # A. Upsert Session first (Must exist for Message FKs)
                await conn.execute(
                    upsert_session_sql,
                    session.session_id,
                    self.project_name,
                    session.leaf,
                    name,
                    session.created_at,
                )

                # B. Batch Upsert Messages
                if msg_records:
                    await conn.executemany(upsert_message_sql, msg_records)

    # --- Maintenance Operations ---

    async def delete_session(self, session_id: str) -> None:
        """Delete specific session."""
        async with self.pool.acquire() as conn:
            # Enforce project ownership
            await conn.execute(
                "DELETE FROM conduit_sessions WHERE session_id = $1 AND project_name = $2",
                session_id,
                self.project_name,
            )

    async def wipe(self) -> None:
        """Delete ALL sessions for this project."""
        async with self.pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM conduit_sessions WHERE project_name = $1",
                self.project_name,
            )

    # --- Helpers ---

    def _row_to_message(self, row: Any) -> Message:
        """Helper to rehydrate a Message object from a DB row."""
        msg_data = dict(row)

        # Deserialize JSONB fields
        for field in ["content", "tool_calls", "images", "audio", "parsed", "metadata"]:
            if isinstance(msg_data.get(field), str):
                try:
                    msg_data[field] = json.loads(msg_data[field])
                except (json.JSONDecodeError, TypeError):
                    pass  # Keep as is if decode fails

        # Inject discriminator if needed by Pydantic
        if "role" in msg_data:
            msg_data["role_str"] = msg_data["role"]

        return self._message_adapter.validate_python(msg_data)

    def _topological_sort(self, message_dict: dict[str, Message]) -> list[Message]:
        """
        Sorts messages Root -> Leaf to satisfy Foreign Key constraints.
        Simple BFS approach.
        """
        present_ids = set(message_dict.keys())
        children_map = {mid: [] for mid in present_ids}
        roots = []

        # Build graph
        for mid, msg in message_dict.items():
            pid = msg.predecessor_id
            if pid and pid in present_ids:
                children_map[pid].append(mid)
            else:
                # No predecessor in this batch -> Root
                roots.append(mid)

        # Traverse
        sorted_list = []
        queue = roots[:]

        while queue:
            current_id = queue.pop(0)
            if current_id in message_dict:
                sorted_list.append(message_dict[current_id])
                queue.extend(children_map.get(current_id, []))

        # Fallback: append disconnected nodes (shouldn't happen in valid DAG)
        if len(sorted_list) < len(message_dict):
            visited = set(m.message_id for m in sorted_list)
            for m in message_dict.values():
                if m.message_id not in visited:
                    sorted_list.append(m)

        return sorted_list


async def get_async_repository(project_name: str) -> AsyncPostgresSessionRepository:
    """
    Factory function to get an initialized AsyncPostgresSessionRepository.
    Requires 'asyncpg' installed and configured in dbclients.
    """
    from dbclients.clients.postgres import get_postgres_client

    # Must await the async factory to get the pool
    pool = await get_postgres_client(client_type="async", dbname="conduit")

    repo = AsyncPostgresSessionRepository(project_name=project_name, pool=pool)
    await repo.initialize()
    return repo
