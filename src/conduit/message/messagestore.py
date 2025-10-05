"""
A MessageStore object inherits from Messages and adds persistence and logging capabilities.
With MessageStore, you can:
- Use all Messages methods (append, extend, indexing, etc.) directly
- Automatically persist changes to TinyDB
- Log conversations in human-readable format
- All while being a drop-in replacement for Messages objects

The MessageStore IS a Messages object with superpowers.
"""

from conduit.message.message import Message, Role
from conduit.message.textmessage import TextMessage
from conduit.message.imagemessage import ImageMessage
from conduit.message.audiomessage import AudioMessage
from conduit.message.messages import Messages
from rich.console import Console
from rich.rule import Rule
from pydantic import BaseModel, Field
from pathlib import Path
from tinydb import TinyDB
from typing import ClassVar, override
from datetime import datetime
import os


class MessageStore(Messages):
    """
    A Messages object with automatic persistence.

    ⚠️  MUTATION WARNING: All list operations (append, extend, etc.)
        will automatically persist to database if history_file was provided.

    Side Effects:
        - append() → database write (if persistent=True)
        - extend() → multiple database writes
        - clear() → database truncation
    """

    # Add Pydantic fields (Messages inherits from BaseModel)
    console: Console | None = Field(default=None, exclude=True, repr=False)
    auto_save: bool = Field(default=True, exclude=True)
    persistent: bool = Field(default=False, exclude=True)
    logging: bool = Field(default=False, exclude=True)
    pruning: bool = Field(default=False, exclude=True)
    history_file: Path | None = Field(default=None, exclude=True, repr=False)
    log_file: Path | None = Field(default=None, exclude=True, repr=False)
    db: TinyDB | None = Field(default=None, exclude=True, repr=False)

    model_config: ClassVar = {
        "arbitrary_types_allowed": True
    }  # Allow Console and TinyDB types

    def __init__(
        self,
        messages: list[Message] | Messages | None = None,
        console: Console | None = None,
        history_file: str | Path = "",
        log_file: str | Path = "",
        pruning: bool = False,
        auto_save: bool = True,
    ):
        """
        Initialize MessageStore with optional persistence and logging.

        Args:
            messages: Initial list of messages (same as Messages class)
            console: Rich console for formatting output
            history_file: Path to TinyDB database file (.json extension recommended)
            log_file: Path to human-readable log file
            pruning: Whether to automatically prune old messages
            auto_save: Whether to automatically save changes to database
        """
        # Coerce Messages object to list for parent class
        if isinstance(messages, Messages):
            messages = messages.messages  # Extract the list
        # Initialize parent Messages class
        super().__init__(messages)

        # Use existing console or create a new one
        if not console:
            self.console = Console(width=100)
        else:
            self.console = console

        self.auto_save = auto_save

        # Config history database if requested
        if history_file:
            # Ensure .json extension for TinyDB
            if not str(history_file).endswith(".json"):
                history_file = str(history_file) + ".json"
            self.history_file = Path(history_file)
            self.persistent = True
            # Initialize TinyDB
            self.db = TinyDB(self.history_file)
            # Load existing messages from database
            self.load()
        else:
            self.history_file = Path()
            self.persistent = False
            self.db = None

        # Config log file if requested
        if log_file:
            self.log_file = Path(log_file)
            with open(self.log_file, "a"):
                os.utime(self.log_file, None)
            self.logging = True
        else:
            self.log_file = Path()
            self.logging = False

        # Set the prune flag
        self.pruning = pruning

    def _auto_save_if_enabled(self):
        """Save to database if auto_save is enabled."""
        if self.auto_save and self.persistent:
            self.save()

    def _log_message_if_enabled(self, message: Message):
        """Log message if logging is enabled."""
        if self.logging:
            self.write_to_log(message)

    # Override Messages methods to add persistence and logging
    @override
    def append(self, message: Message) -> None:
        """Add a message to the end of the list with persistence."""
        super().append(message)
        self._log_message_if_enabled(message)
        self._auto_save_if_enabled()

    @override
    def extend(self, messages: list[Message]) -> None:
        """Extend the list with multiple messages with persistence."""
        super().extend(messages)
        for message in messages:
            self._log_message_if_enabled(message)
        self._auto_save_if_enabled()

    @override
    def insert(self, index: int, message: Message) -> None:
        """Insert a message at the specified index with persistence."""
        super().insert(index, message)
        self._log_message_if_enabled(message)
        self._auto_save_if_enabled()

    @override
    def remove(self, message: Message) -> None:
        """Remove the first occurrence of a message with persistence."""
        super().remove(message)
        self._auto_save_if_enabled()

    @override
    def pop(self, index: int = -1) -> Message:
        """Remove and return message at index with persistence."""
        message = super().pop(index)
        self._auto_save_if_enabled()
        return message

    @override
    def clear(self) -> None:
        """Remove all messages with persistence."""
        super().clear()
        self._auto_save_if_enabled()

    @override
    def __setitem__(self, key, value):
        """Set message(s) by index or slice with persistence."""
        super().__setitem__(key, value)
        if isinstance(value, Message):
            self._log_message_if_enabled(value)
        elif isinstance(value, list):
            for msg in value:
                if isinstance(msg, Message):
                    self._log_message_if_enabled(msg)
        self._auto_save_if_enabled()

    @override
    def __delitem__(self, key):
        """Delete message(s) by index or slice with persistence."""
        super().__delitem__(key)
        self._auto_save_if_enabled()

    @override
    def __iadd__(self, other) -> "MessageStore":
        """In-place concatenation with persistence."""
        result = super().__iadd__(other)
        if isinstance(other, (list, Messages)):
            for message in other:
                if isinstance(message, Message):
                    self._log_message_if_enabled(message)
        self._auto_save_if_enabled()
        return self

    # MessageStore-specific methods
    def add_new(self, role: Role, content: str) -> None:
        """
        Create and add a new message (convenience method).
        """
        message = TextMessage(role=role, content=content)
        self.append(message)  # This will handle persistence and logging

    def add_response(self, response: "Response") -> None:
        """
        Update the store for a successful Response.
        This adds two messages: one for the user and one for the assistant.
        """
        if not isinstance(response.params.messages[-1], Message):
            raise ValueError(
                "Last message in params.messages must be a Message object."
            )
        last_user_message = response.params.messages[-1]  # Last user message
        if last_user_message.role != "user":
            raise ValueError("Last message in params.message must be a user message.")
        last_assistant_message = response.message  # Last assistant message
        if last_assistant_message.role != "assistant":
            raise ValueError("Last message in response must be an assistant message.")
        # Add the messages
        self.append(last_user_message)
        self.append(last_assistant_message)

    def write_to_log(self, item: str | BaseModel) -> None:
        """
        Writes a log to the log file.
        """
        if not self.logging:
            return

        if isinstance(item, str):
            with open(self.log_file, "a", encoding="utf-8") as file:
                file_console = Console(file=file, force_terminal=True)
                file_console.print(f"[bold magenta]{item}[/bold magenta]\n")

        elif isinstance(item, Message):
            with open(self.log_file, "a", encoding="utf-8") as file:
                file_console = Console(file=file, force_terminal=True)
                file_console.print(Rule(title="Message", style="bold green"))
                file_console.print(f"[bold cyan]{item.role}:[/bold cyan]")
                try:
                    if item.role == "user":
                        file_console.print(f"[yellow]{item.content}[/yellow]\n")
                    elif item.role == "assistant":
                        file_console.print(f"[blue]{item.content}[/blue]\n")
                    elif item.role == "system":
                        file_console.print(f"[green]{item.content}[/green]\n")
                    else:
                        file_console.print(f"[white]{item.content}[/white]\n")
                except Exception as e:
                    print(f"MessageStore error: {e}")

    def save(self):
        """
        Saves the current messages to TinyDB.
        """
        if not self.persistent or not isinstance(self.db, TinyDB):
            return

        try:
            # Clear existing messages and save all current messages
            self.db.truncate()

            # Save all current messages
            for i, message in enumerate(self.messages):
                if message:  # Handle None messages
                    doc = {
                        "timestamp": datetime.now().isoformat(),
                        "message_index": i,
                        "message_data": message.to_cache_dict(),
                    }
                    self.db.insert(doc)

        except Exception as e:
            print(f"Error saving history: {e}")

    def load(self):
        """
        Loads messages from TinyDB, replacing current messages.
        """
        if not self.persistent or self.db == None:
            print("This message store is not persistent.")
            return

        try:
            # Get all messages from database
            message_docs = self.db.all()

            # Sort by message_index to maintain order
            message_docs.sort(key=lambda x: x.get("message_index", 0))

            # Deserialize messages
            messages_list = []
            for doc in message_docs:
                message_data = doc["message_data"]
                message_type = message_data.get("message_type", "Message")

                if message_type == "ImageMessage":
                    messages_list.append(ImageMessage.from_cache_dict(message_data))
                elif message_type == "AudioMessage":
                    messages_list.append(AudioMessage.from_cache_dict(message_data))
                else:
                    messages_list.append(Message.from_cache_dict(message_data))

            # Replace current messages (disable auto_save temporarily)
            old_auto_save = self.auto_save
            self.auto_save = False

            self.clear()
            self.extend(messages_list)

            self.auto_save = old_auto_save

            if self.pruning:
                self.prune()

        except Exception as e:
            print(f"Error loading history: {e}. Starting with empty history.")
            super().clear()  # Don't trigger auto_save for error case

    def prune(self):
        """
        Prunes the history to the last 20 messages.
        """
        if len(self) > 20:
            # Keep last 20 messages
            pruned_messages = list(self)[-20:]

            # Disable auto_save temporarily to avoid multiple saves
            old_auto_save = self.auto_save
            self.auto_save = False

            self.clear()
            self.extend(pruned_messages)

            self.auto_save = old_auto_save

            # Save the pruned version
            if self.persistent:
                self.save()

    def query_failed(self):
        """
        Removes the last message if it's a user message (handles failed queries).
        """
        if self and self.last() and self.last().role == "user":
            self.pop()  # This will auto-save
            if self.logging:
                self.write_to_log("Query failed, removing last user message.")

    def view_history(self):
        """
        Pretty prints the history.
        """
        if len(self) == 0:
            self.console.print("No history (yet).", style="bold red")
            return

        for index, message in enumerate(self):
            if message:
                content = str(message.content)[:50].replace("\n", " ")
                self.console.print(
                    f"[green]{index + 1}.[/green] [bold white]{message.role}:[/bold white] [yellow]{content}[/yellow]"
                )

    def clear_logs(self):
        """Clears the log file."""
        if self.logging:
            with open(self.log_file, "w") as file:
                file.write("")

    def delete_database(self):
        """Deletes the TinyDB database file."""
        if self.persistent and self.history_file.exists():
            try:
                if self.db:
                    self.db.close()
                self.history_file.unlink()
                self.db = None
            except Exception as e:
                print(f"Error deleting database: {e}")

    # Enhanced Methods
    def get(self, index: int) -> Message | None:
        """Gets a message by 1-based index (convenience method)."""
        if 1 <= index <= len(self):
            return self[index - 1]
        return None

    def copy(self) -> "MessageStore":
        """Return a copy of the MessageStore (without persistence)."""
        return MessageStore(
            messages=list(self.messages),
            console=self.console,
            # Don't copy persistence settings - new object manages its own state
        )

    def __bool__(self) -> bool:
        """Override; while Messages should be false if empty, MessageStore should be true if it exists."""
        return True

    def __repr__(self) -> str:
        """Enhanced representation showing persistence status."""
        persistent_info = f"persistent={self.persistent}"
        return f"MessageStore({len(self)} messages, {persistent_info})"
