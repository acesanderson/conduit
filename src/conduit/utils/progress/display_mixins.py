"""
Display mixins for Conduit verbosity system.

These mixins provide rich and plain text formatting capabilities to data objects
like Params, Response, and ConduitError. Each mixin handles the specific display
logic for its object type across different verbosity levels.
"""

from conduit.utils.progress.verbosity import Verbosity
from typing import TYPE_CHECKING, Any
import json

# TYPE_CHECKING imports to avoid circular dependencies
if TYPE_CHECKING:
    from rich.console import RenderableType
    from rich.panel import Panel
    from rich.syntax import Syntax


def extract_user_prompt(request) -> str:
    """Extract the user prompt from request messages for display"""
    if not hasattr(request, "messages") or not request.messages:
        return ""

    # Get the last user message (most recent prompt)
    for message in reversed(request.messages):
        if hasattr(message, "role") and message.role == "user":
            content = str(message.content)
            # Handle different message types
            if hasattr(message, "message_type"):
                if message.message_type == "image":
                    return f"{getattr(message, 'text_content', '')} [Image attached]"
                elif message.message_type == "audio":
                    return f"{getattr(message, 'text_content', '')} [Audio attached]"
            return content
    return ""


def safe_json_serialize(obj: Any) -> Any:
    """
    Recursively handle JSON serialization of objects that may contain Verbosity enums.

    Args:
        obj: Object to serialize (dict, list, or primitive)

    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, Verbosity):
        return obj.to_json_serializable()
    elif hasattr(obj, "to_json_serializable"):
        return obj.to_json_serializable()
    elif hasattr(obj, "model_dump"):
        # Pydantic model - recursively serialize
        try:
            dumped = obj.model_dump()
            return safe_json_serialize(dumped)
        except:
            return str(obj)
    elif hasattr(obj, "__dict__"):
        # Object with attributes - recursively serialize
        try:
            return safe_json_serialize(obj.__dict__)
        except:
            return str(obj)
    elif isinstance(obj, dict):
        # Dictionary - recursively serialize values
        return {key: safe_json_serialize(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        # List/tuple - recursively serialize items
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, "name") and hasattr(obj, "value"):
        # Enum-like object
        return f"{obj.name} ({obj.value})"
    else:
        # Primitive or unknown type
        try:
            json.dumps(obj)  # Test if it's JSON serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)


class RichDisplayMixin:
    """Base mixin for Rich console display functionality."""

    def to_rich(self, verbosity: Verbosity) -> "RenderableType":
        """
        Convert object to Rich renderable based on verbosity level.

        Args:
            verbosity: The verbosity level for display

        Returns:
            Rich renderable object (Panel, Syntax, Text, etc.)
        """
        if verbosity == Verbosity.SILENT:
            return ""
        elif verbosity == Verbosity.PROGRESS:
            # PROGRESS level handled by existing progress system
            return ""
        else:
            # Delegate to specific implementation
            return self._to_rich_impl(verbosity)

    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Override in subclasses for specific Rich formatting logic."""
        raise NotImplementedError("Subclasses must implement _to_rich_impl")


class PlainDisplayMixin:
    """Base mixin for plain text display functionality."""

    def to_plain(self, verbosity: Verbosity) -> str:
        """
        Convert object to plain text based on verbosity level.

        Args:
            verbosity: The verbosity level for display

        Returns:
            Plain text string representation
        """
        if verbosity == Verbosity.SILENT:
            return ""
        elif verbosity == Verbosity.PROGRESS:
            # PROGRESS level handled by existing progress system
            return ""
        else:
            # Delegate to specific implementation
            return self._to_plain_impl(verbosity)

    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Override in subclasses for specific plain text formatting logic."""
        raise NotImplementedError("Subclasses must implement _to_plain_impl")


class RichDisplayParamsMixin(RichDisplayMixin):
    """Rich display mixin for Params objects."""

    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Format Params object for Rich console display."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text
        from rich.table import Table

        if verbosity == Verbosity.SUMMARY:
            return self._format_params_summary_rich()
        elif verbosity == Verbosity.DETAILED:
            return self._format_params_detailed_rich()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_params_complete_rich()
        elif verbosity == Verbosity.DEBUG:
            return self._format_params_debug_rich()
        else:
            return Text("")

    def _format_params_summary_rich(self) -> "Panel":
        """Format basic request info for SUMMARY level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        # Build content text
        content = Text()

        # Show user message(s)
        if hasattr(self, "messages") and self.messages:
            for msg in self.messages:
                if hasattr(msg, "role") and msg.role == "user":
                    content.append(f"user: {str(msg.content)}\n", style="yellow")
        elif hasattr(self, "query_input") and self.query_input:
            content.append(f"user: {str(self.query_input)}\n", style="yellow")

        # Show parameters
        params_line = []
        if hasattr(self, "temperature") and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                params_line.append(f"Parser: {parser_name.__name__}")

        if params_line:
            content.append(" • ".join(params_line), style="dim")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title=f"► REQUEST {getattr(self, 'model', 'unknown')}",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
        )

    def _format_params_detailed_rich(self) -> "Panel":
        """Format truncated messages for DETAILED level."""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from datetime import datetime

        # Create messages table
        table = Table.grid(padding=(0, 1))
        table.add_column("Role", style="bold", width=8)
        table.add_column("Content")

        if hasattr(self, "messages") and self.messages:
            for msg in self.messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    content = str(msg.content)
                    # Truncate to ~100 characters
                    if len(content) > 100:
                        content = content[:100] + "..."

                    role_color = {
                        "system": "green",
                        "user": "yellow",
                        "assistant": "blue",
                    }.get(str(msg.role), "white")

                    table.add_row(f"[{role_color}]{msg.role}[/{role_color}]", content)
        elif hasattr(self, "query_input") and self.query_input:
            content = str(self.query_input)
            if len(content) > 100:
                content = content[:100] + "..."
            table.add_row("[yellow]user[/yellow]", content)

        # Add parameters row
        params_line = []
        if hasattr(self, "temperature") and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                params_line.append(f"Parser: {parser_name.__name__}")

        if params_line:
            table.add_row("", f"[dim]{' • '.join(params_line)}[/dim]")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            table,
            title=f"► REQUEST {getattr(self, 'model', 'unknown')} (Detailed)",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
        )

    def _format_params_complete_rich(self) -> "Panel":
        """Format complete messages for COMPLETE level."""
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
        from datetime import datetime

        # Create messages table with full content
        table = Table.grid(padding=(0, 1))
        table.add_column("Role", style="bold", width=8)
        table.add_column("Content")

        if hasattr(self, "messages") and self.messages:
            for msg in self.messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    content = str(msg.content)
                    # No truncation - show full content with word wrapping

                    role_color = {
                        "system": "green",
                        "user": "yellow",
                        "assistant": "blue",
                    }.get(str(msg.role), "white")

                    table.add_row(f"[{role_color}]{msg.role}[/{role_color}]", content)
        elif hasattr(self, "query_input") and self.query_input:
            content = str(self.query_input)
            table.add_row("[yellow]user[/yellow]", content)

        # Add parameters row
        params_line = []
        if hasattr(self, "temperature") and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                params_line.append(f"Parser: {parser_name.__name__}")
        if hasattr(self, "stream"):
            params_line.append(f"Stream: {self.stream}")

        if params_line:
            table.add_row("", f"[dim]{' • '.join(params_line)}[/dim]")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            table,
            title=f"► REQUEST {getattr(self, 'model', 'unknown')} (Complete)",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
        )

    def _format_params_debug_rich(self) -> "Panel":
        """Format full JSON debug for DEBUG level."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from datetime import datetime
        import json

        # Create debug dictionary with all available information
        debug_data = {
            "model": getattr(self, "model", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

        # Add messages
        if hasattr(self, "messages") and self.messages:
            debug_data["messages"] = []
            for msg in self.messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    debug_data["messages"].append(
                        {"role": str(msg.role), "content": str(msg.content)}
                    )
        elif hasattr(self, "query_input") and self.query_input:
            debug_data["messages"] = [
                {"role": "user", "content": str(self.query_input)}
            ]

        # Add all other parameters
        for attr in ["temperature", "stream", "verbose"]:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    # Handle Verbosity enum serialization
                    if hasattr(value, "to_json_serializable"):
                        debug_data[attr] = value.to_json_serializable()
                    elif hasattr(value, "name"):  # Other enums
                        debug_data[attr] = f"{value.name} ({value.value})"
                    else:
                        debug_data[attr] = (
                            str(value)
                            if not isinstance(value, (int, float, bool))
                            else value
                        )

        # Add parser info
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                debug_data["parser"] = parser_name.__name__

        # Add client params if available
        if hasattr(self, "client_params") and self.client_params:
            debug_data["client_params"] = safe_json_serialize(self.client_params)

        # Make the entire debug_data object safe for JSON serialization
        safe_debug_data = safe_json_serialize(debug_data)

        json_content = json.dumps(safe_debug_data, indent=2, ensure_ascii=False)
        syntax = Syntax(json_content, "json", line_numbers=True, theme="monokai")

        return Panel(
            syntax,
            title="🐛 FULL DEBUG REQUEST",
            title_align="left",
            subtitle=f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
            subtitle_align="right",
        )


class PlainDisplayParamsMixin(PlainDisplayMixin):
    """Plain text display mixin for Params objects."""

    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Format Params object for plain text display."""
        if verbosity == Verbosity.SUMMARY:
            return self._format_params_summary_plain()
        elif verbosity == Verbosity.DETAILED:
            return self._format_params_detailed_plain()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_params_complete_plain()
        elif verbosity == Verbosity.DEBUG:
            return self._format_params_debug_plain()
        else:
            return ""

    def _format_params_summary_plain(self) -> str:
        """Format basic request info for SUMMARY level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(f"[{timestamp}] REQUEST {getattr(self, 'model', 'unknown')}")

        # Show user message
        if hasattr(self, "messages") and self.messages:
            for msg in self.messages:
                if hasattr(msg, "role") and msg.role == "user":
                    lines.append(f"user: {str(msg.content)}")
        elif hasattr(self, "query_input") and self.query_input:
            lines.append(f"user: {str(self.query_input)}")

        # Show parameters
        params_line = []
        if hasattr(self, "temperature") and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                params_line.append(f"Parser: {parser_name.__name__}")

        if params_line:
            lines.append(" • ".join(params_line))

        return "\n".join(lines)

    def _format_params_detailed_plain(self) -> str:
        """Format truncated messages for DETAILED level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(
            f"[{timestamp}] REQUEST {getattr(self, 'model', 'unknown')} (Detailed)"
        )
        lines.append("Messages:")

        # Show messages with truncation
        if hasattr(self, "messages") and self.messages:
            for msg in self.messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    content = str(msg.content)
                    if len(content) > 100:
                        content = content[:100] + "..."
                    lines.append(f"  {msg.role}: {content}")
        elif hasattr(self, "query_input") and self.query_input:
            content = str(self.query_input)
            if len(content) > 100:
                content = content[:100] + "..."
            lines.append(f"  user: {content}")

        # Show parameters
        params_line = []
        if hasattr(self, "temperature") and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                params_line.append(f"Parser: {parser_name.__name__}")

        if params_line:
            lines.append("Parameters: " + " • ".join(params_line))

        return "\n".join(lines)

    def _format_params_complete_plain(self) -> str:
        """Format complete messages for COMPLETE level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(
            f"[{timestamp}] REQUEST {getattr(self, 'model', 'unknown')} (Complete)"
        )
        lines.append("Complete Messages:")

        # Show full messages
        if hasattr(self, "messages") and self.messages:
            for msg in self.messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    content = str(msg.content)
                    lines.append(f"  {msg.role}: {content}")
        elif hasattr(self, "query_input") and self.query_input:
            content = str(self.query_input)
            lines.append(f"  user: {content}")

        # Show all parameters
        params_line = []
        if hasattr(self, "temperature") and self.temperature is not None:
            params_line.append(f"Temperature: {self.temperature}")
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                params_line.append(f"Parser: {parser_name.__name__}")
        if hasattr(self, "stream"):
            params_line.append(f"Stream: {self.stream}")

        if params_line:
            lines.append("Parameters: " + " • ".join(params_line))

        return "\n".join(lines)

    def _format_params_debug_plain(self) -> str:
        """Format full JSON debug for DEBUG level."""
        from datetime import datetime
        import json

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(f"[{timestamp}] DEBUG REQUEST {getattr(self, 'model', 'unknown')}")

        # Create debug dictionary
        debug_data = {
            "model": getattr(self, "model", "unknown"),
            "timestamp": datetime.now().isoformat(),
        }

        # Add messages
        if hasattr(self, "messages") and self.messages:
            debug_data["messages"] = []
            for msg in self.messages:
                if hasattr(msg, "role") and hasattr(msg, "content"):
                    debug_data["messages"].append(
                        {"role": str(msg.role), "content": str(msg.content)}
                    )
        elif hasattr(self, "query_input") and self.query_input:
            debug_data["messages"] = [
                {"role": "user", "content": str(self.query_input)}
            ]

        # Add all other parameters
        for attr in ["temperature", "stream", "verbose"]:
            if hasattr(self, attr):
                value = getattr(self, attr)
                if value is not None:
                    # Handle Verbosity enum serialization
                    if hasattr(value, "to_json_serializable"):
                        debug_data[attr] = value.to_json_serializable()
                    elif hasattr(value, "name"):  # Other enums
                        debug_data[attr] = f"{value.name} ({value.value})"
                    else:
                        debug_data[attr] = (
                            str(value)
                            if not isinstance(value, (int, float, bool))
                            else value
                        )

        # Add parser info
        if hasattr(self, "parser") and self.parser:
            parser_name = getattr(self.parser, "pydantic_model", {})
            if hasattr(parser_name, "__name__"):
                debug_data["parser"] = parser_name.__name__

        # Add client params if available
        if hasattr(self, "client_params") and self.client_params:
            debug_data["client_params"] = safe_json_serialize(self.client_params)

        # Make the entire debug_data object safe for JSON serialization
        safe_debug_data = safe_json_serialize(debug_data)

        # Format as JSON
        json_content = json.dumps(safe_debug_data, indent=2, ensure_ascii=False)
        lines.append("Complete Parameters:")
        lines.append(json_content)

        return "\n".join(lines)


class RichDisplayResponseMixin(RichDisplayMixin):
    """Rich display mixin for Response objects."""

    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Format Response object for Rich console display."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text

        if verbosity == Verbosity.SUMMARY:
            return self._format_response_summary_rich()
        elif verbosity == Verbosity.DETAILED:
            return self._format_response_detailed_rich()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_response_complete_rich()
        elif verbosity == Verbosity.DEBUG:
            return self._format_response_debug_rich()
        else:
            return Text("")

    def _format_response_summary_rich(self) -> "Panel":
        """Format basic response info for SUMMARY level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        content = Text()
        duration = getattr(self, "duration", 0)

        # Show response content (truncated)
        response_content = str(getattr(self, "content", "No content"))
        if len(response_content) > 100:
            response_content = response_content[:100] + "..."

        content.append(response_content, style="blue")

        # Show metadata if available
        if hasattr(self, "model"):
            content.append(f"\nModel: {self.model}", style="dim")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title=f"✓ RESPONSE {duration:.1f}s",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
        )

    def _format_response_detailed_rich(self) -> "Panel":
        """Format truncated response for DETAILED level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        content = Text()
        duration = getattr(self, "duration", 0)

        # NEW: Show user prompt first at DETAILED level and above
        if hasattr(self, "request"):
            user_prompt = extract_user_prompt(self.request)
            if user_prompt:
                content.append("User: ", style="bold yellow")
                # Truncate user prompt to ~200 chars for detailed view
                if len(user_prompt) > 200:
                    user_prompt = user_prompt[:200] + "..."
                content.append(f"{user_prompt}\n\n", style="yellow")

        # Show response content (truncated to ~200 chars for detailed)
        content.append("Assistant: ", style="bold blue")
        response_content = str(getattr(self, "content", "No content"))
        if len(response_content) > 200:
            response_content = response_content[:200] + "..."
        content.append(response_content, style="blue")

        # Show metadata
        content.append("\n", style="")
        metadata_lines = []
        if hasattr(self, "model"):
            metadata_lines.append(f"Model: {self.model}")
        if hasattr(self, "request") and self.request:
            if (
                hasattr(self.request, "temperature")
                and self.request.temperature is not None
            ):
                metadata_lines.append(f"Temperature: {self.request.temperature}")

        if metadata_lines:
            content.append(" • ".join(metadata_lines), style="dim")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title=f"✓ CONVERSATION {duration:.1f}s",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
        )

    def _format_response_complete_rich(self) -> "Panel":
        """Format complete response for COMPLETE level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        content = Text()
        duration = getattr(self, "duration", 0)

        # NEW: Show full user prompt at COMPLETE level
        if hasattr(self, "request"):
            user_prompt = extract_user_prompt(self.request)
            if user_prompt:
                content.append("User: ", style="bold yellow")
                content.append(f"{user_prompt}\n\n", style="yellow")

        # Show full response content (no truncation)
        content.append("Assistant: ", style="bold blue")
        response_content = str(getattr(self, "content", "No content"))
        content.append(response_content, style="blue")

        # Show detailed metadata
        content.append("\n\n", style="")
        metadata_lines = []
        if hasattr(self, "model"):
            metadata_lines.append(f"Model: {self.model}")
        if hasattr(self, "request") and self.request:
            if (
                hasattr(self.request, "temperature")
                and self.request.temperature is not None
            ):
                metadata_lines.append(f"Temperature: {self.request.temperature}")
            if hasattr(self.request, "response_model") and self.request.response_model:
                parser_name = getattr(
                    self.request.response_model,
                    "__name__",
                    str(self.request.response_model),
                )
                metadata_lines.append(f"Parser: {parser_name}")

        if hasattr(self, "timestamp"):
            metadata_lines.append(f"Timestamp: {self.timestamp}")

        if metadata_lines:
            content.append(" • ".join(metadata_lines), style="dim")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title=f"✓ FULL CONVERSATION {duration:.1f}s",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
        )

    def _format_response_debug_rich(self) -> "Panel":
        """Format full JSON debug for DEBUG level."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from datetime import datetime
        import json

        duration = getattr(self, "duration", 0)

        # Create debug dictionary with conversation context
        debug_data = {
            "type": "ConversationDebug",
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "model": getattr(self, "model", "unknown"),
        }

        # Add user prompt
        if hasattr(self, "request"):
            user_prompt = extract_user_prompt(self.request)
            if user_prompt:
                debug_data["user_prompt"] = user_prompt

        # Add response content
        content = getattr(self, "content", None)
        if content is not None:
            try:
                if hasattr(content, "model_dump"):
                    debug_data["assistant_response"] = content.model_dump()
                elif hasattr(content, "__dict__"):
                    debug_data["assistant_response"] = content.__dict__
                else:
                    debug_data["assistant_response"] = str(content)
            except:
                debug_data["assistant_response"] = str(content)

        # Add full request if available
        if hasattr(self, "request") and self.request:
            debug_data["full_request"] = safe_json_serialize(self.request)

        # Make the entire debug_data object safe for JSON serialization
        safe_debug_data = safe_json_serialize(debug_data)

        json_content = json.dumps(safe_debug_data, indent=2, ensure_ascii=False)
        syntax = Syntax(json_content, "json", line_numbers=True, theme="monokai")

        return Panel(
            syntax,
            title=f"🐛 CONVERSATION DEBUG {duration:.1f}s",
            title_align="left",
            subtitle=f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
            subtitle_align="right",
        )


class PlainDisplayResponseMixin(PlainDisplayMixin):
    """Plain text display mixin for Response objects."""

    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Format Response object for plain text display."""
        if verbosity == Verbosity.SUMMARY:
            return self._format_response_summary_plain()
        elif verbosity == Verbosity.DETAILED:
            return self._format_response_detailed_plain()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_response_complete_plain()
        elif verbosity == Verbosity.DEBUG:
            return self._format_response_debug_plain()
        else:
            return ""

    def _format_response_summary_plain(self) -> str:
        """Format basic response info for SUMMARY level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        duration = getattr(self, "duration", 0)

        # Header
        lines.append(f"[{timestamp}] RESPONSE {duration:.1f}s")

        # Show response content (truncated)
        response_content = str(getattr(self, "content", "No content"))
        if len(response_content) > 100:
            response_content = response_content[:100] + "..."
        lines.append(response_content)

        return "\n".join(lines)

    def _format_response_detailed_plain(self) -> str:
        """Format truncated response for DETAILED level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        duration = getattr(self, "duration", 0)

        # Header
        lines.append(f"[{timestamp}] CONVERSATION {duration:.1f}s (Detailed)")

        # NEW: Show user prompt
        if hasattr(self, "request"):
            user_prompt = extract_user_prompt(self.request)
            if user_prompt:
                if len(user_prompt) > 200:
                    user_prompt = user_prompt[:200] + "..."
                lines.append(f"User: {user_prompt}")

        # Show response content (truncated to ~200 chars)
        response_content = str(getattr(self, "content", "No content"))
        if len(response_content) > 200:
            response_content = response_content[:200] + "..."
        lines.append(f"Assistant: {response_content}")

        # Show metadata
        metadata_lines = []
        if hasattr(self, "model"):
            metadata_lines.append(f"Model: {self.model}")
        if hasattr(self, "request") and self.request:
            if (
                hasattr(self.request, "temperature")
                and self.request.temperature is not None
            ):
                metadata_lines.append(f"Temperature: {self.request.temperature}")

        if metadata_lines:
            lines.append("Metadata: " + " • ".join(metadata_lines))

        return "\n".join(lines)

    def _format_response_complete_plain(self) -> str:
        """Format complete response for COMPLETE level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        duration = getattr(self, "duration", 0)

        # Header
        lines.append(f"[{timestamp}] FULL CONVERSATION {duration:.1f}s")

        # NEW: Show full user prompt
        if hasattr(self, "request"):
            user_prompt = extract_user_prompt(self.request)
            if user_prompt:
                lines.append(f"User: {user_prompt}")

        # Show full response content
        response_content = str(getattr(self, "content", "No content"))
        lines.append(f"Assistant: {response_content}")

        # Show detailed metadata
        metadata_lines = []
        if hasattr(self, "model"):
            metadata_lines.append(f"Model: {self.model}")
        if hasattr(self, "request") and self.request:
            if (
                hasattr(self.request, "temperature")
                and self.request.temperature is not None
            ):
                metadata_lines.append(f"Temperature: {self.request.temperature}")
            if hasattr(self.request, "response_model") and self.request.response_model:
                parser_name = getattr(
                    self.request.response_model,
                    "__name__",
                    str(self.request.response_model),
                )
                metadata_lines.append(f"Parser: {parser_name}")

        if hasattr(self, "timestamp"):
            metadata_lines.append(f"Timestamp: {self.timestamp}")

        if metadata_lines:
            lines.append("Metadata: " + " • ".join(metadata_lines))

        return "\n".join(lines)

    def _format_response_debug_plain(self) -> str:
        """Format full JSON debug for DEBUG level."""
        from datetime import datetime
        import json

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")
        duration = getattr(self, "duration", 0)

        # Header
        lines.append(f"[{timestamp}] CONVERSATION DEBUG {duration:.1f}s")

        # Create debug dictionary
        debug_data = {
            "type": "ConversationDebug",
            "timestamp": datetime.now().isoformat(),
            "duration": duration,
            "model": getattr(self, "model", "unknown"),
        }

        # Add user prompt
        if hasattr(self, "request"):
            user_prompt = extract_user_prompt(self.request)
            if user_prompt:
                debug_data["user_prompt"] = user_prompt

        # Add response content
        content = getattr(self, "content", None)
        if content is not None:
            try:
                if hasattr(content, "model_dump"):
                    debug_data["assistant_response"] = content.model_dump()
                elif hasattr(content, "__dict__"):
                    debug_data["assistant_response"] = content.__dict__
                else:
                    debug_data["assistant_response"] = str(content)
            except:
                debug_data["assistant_response"] = str(content)

        # Add full request if available
        if hasattr(self, "request") and self.request:
            debug_data["full_request"] = safe_json_serialize(self.request)

        # Make the entire debug_data object safe for JSON serialization
        safe_debug_data = safe_json_serialize(debug_data)

        # Format as JSON
        json_content = json.dumps(safe_debug_data, indent=2, ensure_ascii=False)
        lines.append("Complete Conversation Object:")
        lines.append(json_content)

        return "\n".join(lines)


class RichDisplayConduitErrorMixin(RichDisplayMixin):
    """Rich display mixin for ConduitError objects."""

    def _to_rich_impl(self, verbosity: Verbosity) -> "RenderableType":
        """Format ConduitError object for Rich console display."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from rich.text import Text

        if verbosity == Verbosity.SUMMARY:
            return self._format_error_summary_rich()
        elif verbosity == Verbosity.DETAILED:
            return self._format_error_detailed_rich()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_error_complete_rich()
        elif verbosity == Verbosity.DEBUG:
            return self._format_error_debug_rich()
        else:
            return Text("")

    def _format_error_summary_rich(self) -> "Panel":
        """Format basic error info for SUMMARY level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        content = Text()

        # Show error info
        if hasattr(self, "info"):
            content.append(f"Error: {self.info.code}\n", style="bold red")
            content.append(f"{self.info.message}\n", style="red")
            content.append(f"Category: {self.info.category}", style="dim red")
        else:
            content.append("Unknown error", style="red")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title="✗ ERROR",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
            border_style="red",
        )

    def _format_error_detailed_rich(self) -> "Panel":
        """Format detailed error for DETAILED level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        content = Text()

        # Show error info with more details
        if hasattr(self, "info"):
            content.append(f"Error Code: {self.info.code}\n", style="bold red")
            content.append(f"Message: {self.info.message}\n", style="red")
            content.append(f"Category: {self.info.category}\n", style="dim red")
            content.append(f"Timestamp: {self.info.timestamp}\n", style="dim")
        else:
            content.append("Unknown error", style="red")

        # Show additional detail if available
        if hasattr(self, "detail") and self.detail:
            content.append(
                f"\nException Type: {self.detail.exception_type}", style="dim red"
            )
            if self.detail.request_params:
                content.append(
                    f"\nRequest had {len(self.detail.request_params)} parameters",
                    style="dim",
                )

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title="✗ ERROR (Detailed)",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
            border_style="red",
        )

    def _format_error_complete_rich(self) -> "Panel":
        """Format complete error for COMPLETE level."""
        from rich.panel import Panel
        from rich.text import Text
        from datetime import datetime

        content = Text()

        # Show comprehensive error info
        if hasattr(self, "info"):
            content.append(f"Error Code: {self.info.code}\n", style="bold red")
            content.append(f"Message: {self.info.message}\n", style="red")
            content.append(f"Category: {self.info.category}\n", style="dim red")
            content.append(f"Timestamp: {self.info.timestamp}\n", style="dim")
        else:
            content.append("Unknown error\n", style="red")

        # Show complete detail if available
        if hasattr(self, "detail") and self.detail:
            content.append(f"\nException Details:\n", style="bold")
            content.append(f"Type: {self.detail.exception_type}\n", style="dim red")

            if self.detail.request_params:
                content.append(
                    f"Request Parameters: {len(self.detail.request_params)} items\n",
                    style="dim",
                )

            if self.detail.retry_count is not None:
                content.append(f"Retry Count: {self.detail.retry_count}\n", style="dim")

            # Show truncated stack trace if available
            if self.detail.stack_trace:
                stack_lines = self.detail.stack_trace.split("\n")
                # Show first few and last few lines
                if len(stack_lines) > 10:
                    shown_lines = (
                        stack_lines[:3] + ["  ... (truncated) ..."] + stack_lines[-3:]
                    )
                else:
                    shown_lines = stack_lines
                content.append("\nStack Trace (truncated):\n", style="bold dim")
                content.append("\n".join(shown_lines), style="dim red")

        timestamp = datetime.now().strftime("%H:%M:%S")
        return Panel(
            content,
            title="✗ ERROR (Complete)",
            title_align="left",
            subtitle=f"[dim]{timestamp}[/dim]",
            subtitle_align="right",
            border_style="red",
        )

    def _format_error_debug_rich(self) -> "Panel":
        """Format full JSON debug for DEBUG level."""
        from rich.panel import Panel
        from rich.syntax import Syntax
        from datetime import datetime
        import json

        # Create debug dictionary with all error information
        debug_data = {
            "type": "ConduitError",
            "timestamp": datetime.now().isoformat(),
        }

        # Add error info
        if hasattr(self, "info"):
            debug_data["info"] = {
                "code": self.info.code,
                "message": self.info.message,
                "category": self.info.category,
                "timestamp": self.info.timestamp.isoformat()
                if hasattr(self.info.timestamp, "isoformat")
                else str(self.info.timestamp),
            }

        # Add error detail
        if hasattr(self, "detail") and self.detail:
            debug_data["detail"] = {
                "exception_type": self.detail.exception_type,
                "has_stack_trace": bool(self.detail.stack_trace),
                "stack_trace_lines": len(self.detail.stack_trace.split("\n"))
                if self.detail.stack_trace
                else 0,
                "request_params": self.detail.request_params,
                "retry_count": self.detail.retry_count,
                "raw_response": str(self.detail.raw_response)
                if self.detail.raw_response
                else None,
            }

            # Include full stack trace in debug mode
            if self.detail.stack_trace:
                debug_data["detail"]["full_stack_trace"] = self.detail.stack_trace

        # Make the entire debug_data object safe for JSON serialization
        safe_debug_data = safe_json_serialize(debug_data)

        json_content = json.dumps(safe_debug_data, indent=2, ensure_ascii=False)
        syntax = Syntax(json_content, "json", line_numbers=True, theme="monokai")

        return Panel(
            syntax,
            title="🐛 FULL DEBUG ERROR",
            title_align="left",
            subtitle=f"[dim]{datetime.now().strftime('%H:%M:%S')}[/dim]",
            subtitle_align="right",
            border_style="red",
        )


class PlainDisplayConduitErrorMixin(PlainDisplayMixin):
    """Plain text display mixin for ConduitError objects."""

    def _to_plain_impl(self, verbosity: Verbosity) -> str:
        """Format ConduitError object for plain text display."""
        if verbosity == Verbosity.SUMMARY:
            return self._format_error_summary_plain()
        elif verbosity == Verbosity.DETAILED:
            return self._format_error_detailed_plain()
        elif verbosity == Verbosity.COMPLETE:
            return self._format_error_complete_plain()
        elif verbosity == Verbosity.DEBUG:
            return self._format_error_debug_plain()
        else:
            return ""

    def _format_error_summary_plain(self) -> str:
        """Format basic error info for SUMMARY level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(f"[{timestamp}] ERROR")

        # Show error info
        if hasattr(self, "info"):
            lines.append(f"Error: {self.info.code}")
            lines.append(f"{self.info.message}")
            lines.append(f"Category: {self.info.category}")
        else:
            lines.append("Unknown error")

        return "\n".join(lines)

    def _format_error_detailed_plain(self) -> str:
        """Format detailed error for DETAILED level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(f"[{timestamp}] ERROR (Detailed)")

        # Show error info with more details
        if hasattr(self, "info"):
            lines.append(f"Error Code: {self.info.code}")
            lines.append(f"Message: {self.info.message}")
            lines.append(f"Category: {self.info.category}")
            lines.append(f"Timestamp: {self.info.timestamp}")
        else:
            lines.append("Unknown error")

        # Show additional detail if available
        if hasattr(self, "detail") and self.detail:
            lines.append(f"Exception Type: {self.detail.exception_type}")
            if self.detail.request_params:
                lines.append(
                    f"Request had {len(self.detail.request_params)} parameters"
                )

        return "\n".join(lines)

    def _format_error_complete_plain(self) -> str:
        """Format complete error for COMPLETE level."""
        from datetime import datetime

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(f"[{timestamp}] ERROR (Complete)")

        # Show comprehensive error info
        if hasattr(self, "info"):
            lines.append(f"Error Code: {self.info.code}")
            lines.append(f"Message: {self.info.message}")
            lines.append(f"Category: {self.info.category}")
            lines.append(f"Timestamp: {self.info.timestamp}")
        else:
            lines.append("Unknown error")

        # Show complete detail if available
        if hasattr(self, "detail") and self.detail:
            lines.append("\nException Details:")
            lines.append(f"Type: {self.detail.exception_type}")

            if self.detail.request_params:
                lines.append(
                    f"Request Parameters: {len(self.detail.request_params)} items"
                )

            if self.detail.retry_count is not None:
                lines.append(f"Retry Count: {self.detail.retry_count}")

            # Show truncated stack trace if available
            if self.detail.stack_trace:
                lines.append("\nStack Trace (truncated):")
                stack_lines = self.detail.stack_trace.split("\n")
                # Show first few and last few lines
                if len(stack_lines) > 10:
                    shown_lines = (
                        stack_lines[:3] + ["  ... (truncated) ..."] + stack_lines[-3:]
                    )
                else:
                    shown_lines = stack_lines
                lines.extend(shown_lines)

        return "\n".join(lines)

    def _format_error_debug_plain(self) -> str:
        """Format full JSON debug for DEBUG level."""
        from datetime import datetime
        import json

        lines = []
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Header
        lines.append(f"[{timestamp}] DEBUG ERROR")

        # Create debug dictionary
        debug_data = {
            "type": "ConduitError",
            "timestamp": datetime.now().isoformat(),
        }

        # Add error info
        if hasattr(self, "info"):
            debug_data["info"] = {
                "code": self.info.code,
                "message": self.info.message,
                "category": self.info.category,
                "timestamp": self.info.timestamp.isoformat()
                if hasattr(self.info.timestamp, "isoformat")
                else str(self.info.timestamp),
            }

        # Add error detail
        if hasattr(self, "detail") and self.detail:
            debug_data["detail"] = {
                "exception_type": self.detail.exception_type,
                "has_stack_trace": bool(self.detail.stack_trace),
                "stack_trace_lines": len(self.detail.stack_trace.split("\n"))
                if self.detail.stack_trace
                else 0,
                "request_params": self.detail.request_params,
                "retry_count": self.detail.retry_count,
                "raw_response": str(self.detail.raw_response)
                if self.detail.raw_response
                else None,
            }

            # Include full stack trace in debug mode
            if self.detail.stack_trace:
                debug_data["detail"]["full_stack_trace"] = self.detail.stack_trace

        # Make the entire debug_data object safe for JSON serialization
        safe_debug_data = safe_json_serialize(debug_data)

        # Format as JSON
        json_content = json.dumps(safe_debug_data, indent=2, ensure_ascii=False)
        lines.append("Complete Error Object:")
        lines.append(json_content)

        return "\n".join(lines)
