from __future__ import annotations
from typing import TYPE_CHECKING
import json
import logging
import hashlib
from uuid import UUID
from pydantic import TypeAdapter
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.message.message import Message, MessageUnion

if TYPE_CHECKING:
    from psycopg2.extensions import connection
    from contextlib import AbstractContextManager
    from collections.abc import Callable

logger = logging.getLogger(__name__)
adapter = TypeAdapter(MessageUnion)  # For rehydration from JSONB


class PostgresConversationRepository:
    """
    Project-Scoped Conversation Repository with Message Deduplication.

    Architecture:
    - conversations: Partitioned by 'project_name'.
    - messages: Global, deduplicated by content hash (CAS).
    - playlist: Maps conversation order to message IDs.
    """

    def __init__(
        self,
        project_name: str,
        conn_factory: Callable[[], AbstractContextManager[connection]],
    ):
        """
        Args:
            name: The project name (e.g., "summarizer", "chatbot-v1").
            conn_factory: DB connection factory.
        """
        self.project_name: str = project_name
        self._conn_factory: Callable[[], AbstractContextManager[connection]] = (
            conn_factory
        )
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute("""
                -- 1. Global Messages (Content Addressable)
                CREATE TABLE IF NOT EXISTS conduit_messages (
                    id UUID PRIMARY KEY,
                    content_hash TEXT UNIQUE NOT NULL, -- The De-Dupe Key
                    payload JSONB NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );

                -- 2. Scoped Conversations
                CREATE TABLE IF NOT EXISTS conduit_conversations (
                    id UUID PRIMARY KEY,
                    project_name TEXT NOT NULL, -- Scoping field
                    name TEXT,
                    metadata JSONB,
                    last_updated TIMESTAMPTZ DEFAULT NOW()
                );
                CREATE INDEX IF NOT EXISTS idx_conv_project ON conduit_conversations(project_name);

                -- 3. The Playlist (Edges)
                CREATE TABLE IF NOT EXISTS conduit_playlist (
                    conversation_id UUID NOT NULL REFERENCES conduit_conversations(id) ON DELETE CASCADE,
                    message_id UUID NOT NULL REFERENCES conduit_messages(id),
                    seq_index INTEGER NOT NULL,
                    PRIMARY KEY (conversation_id, seq_index)
                );
                """)
            conn.commit()

    def _generate_message_hash(self, message: Message) -> str:
        """
        Generate a deterministic hash based on semantic content.
        Excludes volatile fields like UUIDs and timestamps.
        """
        # We assume standard Pydantic serialization
        # We explicitly exclude 'id', 'timestamp' to match on CONTENT ONLY.
        data = message.model_dump(
            mode="json", exclude={"id", "timestamp", "tool_call_id"}
        )
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode("utf-8")).hexdigest()

    def load_by_conversation_id(
        self,
        conversation_id: str | UUID,
    ) -> Conversation | None:
        """
        Need to add name logic.
        """
        with self._conn_factory() as conn, conn.cursor() as cursor:
            # ... (check conversation existence logic) ...

            # Fetch messages
            cursor.execute(
                """
                    SELECT m.payload
                    FROM conduit_playlist p
                    JOIN conduit_messages m ON p.message_id = m.id
                    WHERE p.conversation_id = %s
                    ORDER BY p.seq_index ASC
                """,
                (str(conversation_id),),
            )

            rows = cursor.fetchall()

        # Rehydrate using the adapter
        # validate_python takes a dict (which psycopg2 returns for JSONB columns)
        messages = [adapter.validate_python(row[0]) for row in rows]

        return Conversation(conversation_id=str(conversation_id), messages=messages)

    def remove_by_conversation_id(self, conversation_id: str | UUID) -> None:
        """Removes conversation by ID within the current project."""
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    DELETE FROM conduit_conversations
                    WHERE project_name = %s AND id = %s
                """,
                (self.project_name, str(conversation_id)),
            )
            conn.commit()

    def load_by_name(self, name: str) -> Conversation | None:
        """Load conversation by name within the current project."""
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT id
                    FROM conduit_conversations
                    WHERE project_name = %s AND name = %s
                    LIMIT 1
                """,
                (self.project_name, name),
            )
            row = cursor.fetchone()
            if not row:
                return None
            conv_id = row[0]
        return self.load_by_conversation_id(conv_id)

    def list_conversations(self, limit: int = 10) -> list[dict[str, str]]:
        """List conversations ONLY for the current project."""
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT id, name, last_updated, 
                           (SELECT COUNT(*) FROM conduit_playlist WHERE conversation_id = c.id) as msg_count
                    FROM conduit_conversations c
                    WHERE project_name = %s
                    ORDER BY last_updated DESC 
                    LIMIT %s
                """,
                (self.project_name, limit),
            )
            # Rehydrate rows
            rows = cursor.fetchall()
        conversations = [
            {
                "conversation_id": row[0],
                "name": row[1],
                "last_updated": row[2],
                "message_count": row[3],
            }
            for row in rows
        ]
        return conversations

    def load_all(self) -> list[Conversation]:
        """Load all conversations for the current project."""
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT id
                    FROM conduit_conversations
                    WHERE project_name = %s
                """,
                (self.project_name,),
            )
            rows = cursor.fetchall()
            conv_ids = [row[0] for row in rows]

        conversations = []
        for conv_id in conv_ids:
            conv = self.load_by_conversation_id(conv_id)
            if conv:
                conversations.append(conv)
        return conversations

    @property
    def last(self) -> Conversation | None:
        """Returns the most recently updated conversation for the project."""
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT id
                    FROM conduit_conversations
                    WHERE project_name = %s
                    ORDER BY last_updated DESC
                    LIMIT 1
                """,
                (self.project_name,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            conv_id = row[0]
        return self.load_by_conversation_id(conv_id)

    def save(self, conversation: Conversation, name: str | None = None) -> None:
        """
        Saves conversation and performs Message Deduplication.
        """
        conv_id = conversation.conversation_id

        with self._conn_factory() as conn, conn.cursor() as cursor:
            # 1. Upsert Conversation Metadata
            cursor.execute(
                """
                    INSERT INTO conduit_conversations (id, project_name, name, last_updated)
                    VALUES (%s, %s, %s, NOW())
                    ON CONFLICT (id) DO UPDATE 
                    SET last_updated = NOW(),
                        name = COALESCE(EXCLUDED.name, conduit_conversations.name),
                        project_name = EXCLUDED.project_name
                """,
                # FIX: Cast conv_id to str()
                (str(conv_id), self.project_name, name),
            )

            # 2. Process Messages
            final_message_ids = []

            for msg in conversation.messages:
                msg_hash = self._generate_message_hash(msg)
                msg_payload = msg.model_dump(mode="json")

                cursor.execute(
                    """
                        INSERT INTO conduit_messages (id, content_hash, payload)
                        VALUES (%s, %s, %s::jsonb)
                        ON CONFLICT (content_hash) 
                        DO UPDATE SET content_hash = EXCLUDED.content_hash
                        RETURNING id
                    """,
                    # FIX: Cast msg.id to str()
                    (str(msg.message_id), msg_hash, json.dumps(msg_payload)),
                )

                authoritative_id = cursor.fetchone()[0]
                final_message_ids.append(authoritative_id)

            # 3. Rewrite Playlist
            cursor.execute(
                "DELETE FROM conduit_playlist WHERE conversation_id = %s",
                (str(conv_id),),  # FIX: Cast to str()
            )

            if final_message_ids:
                playlist_args = [
                    (str(conv_id), str(msg_id), idx)  # FIX: Cast both to str()
                    for idx, msg_id in enumerate(final_message_ids)
                ]
                cursor.executemany(
                    """
                        INSERT INTO conduit_playlist (conversation_id, message_id, seq_index)
                        VALUES (%s, %s, %s)
                    """,
                    playlist_args,
                )

            conn.commit()

    def wipe(self) -> None:
        """Wipes all conversations and messages for the current project."""
        with self._conn_factory() as conn, conn.cursor() as cursor:
            # Delete conversations for the project
            cursor.execute(
                """
                    DELETE FROM conduit_conversations
                    WHERE project_name = %s
                """,
                (self.project_name,),
            )
            conn.commit()

    # Global methods
    def list_all_projects(self) -> list[str]:
        """
        Lists all distinct project names in the conversations table.
        """
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    SELECT DISTINCT project_name
                    FROM conduit_conversations
                """
            )
            rows = cursor.fetchall()
        return [row[0] for row in rows]

    def prune(self, keep: int = 5) -> None:
        """
        Prunes old conversations, keeping only the most recent 'keep' conversations.
        """
        with self._conn_factory() as conn, conn.cursor() as cursor:
            # Find conversations to delete
            cursor.execute(
                """
                    SELECT id
                    FROM conduit_conversations
                    WHERE project_name = %s
                    ORDER BY last_updated DESC
                    OFFSET %s
                """,
                (self.project_name, keep),
            )
            rows = cursor.fetchall()
            conv_ids_to_delete = [row[0] for row in rows]

            if conv_ids_to_delete:
                cursor.execute(
                    """
                        DELETE FROM conduit_conversations
                        WHERE id = ANY(%s)
                    """,
                    (conv_ids_to_delete,),
                )
                conn.commit()

    def delete_orphaned_messages(self) -> None:
        """
        Deletes messages that are not referenced by any conversation.
        """
        with self._conn_factory() as conn, conn.cursor() as cursor:
            cursor.execute(
                """
                    DELETE FROM conduit_messages
                    WHERE id NOT IN (
                        SELECT DISTINCT message_id FROM conduit_playlist
                    )
                """
            )
            conn.commit()


def get_postgres_repository(project_name: str) -> PostgresConversationRepository:
    """
    Factory function to get a PostgresConversationRepository for a given project.
    """
    from dbclients.clients.postgres import get_postgres_client

    conn_factory = get_postgres_client(
        client_type="context_db",
        dbname="conduit",
    )

    return PostgresConversationRepository(
        project_name=project_name,
        conn_factory=conn_factory,
    )
