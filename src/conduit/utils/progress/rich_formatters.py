from __future__ import annotations
import json
from conduit.domain.result.response import Response
from conduit.domain.result.error import ConduitError
from conduit.utils.progress.verbosity import Verbosity
from conduit.utils.progress.plain_formatters import _extract_user_prompt

from rich.console import RenderableType
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.syntax import Syntax
from rich import box


# --- Rich Formatters: Response ---
def format_response_rich(
    response: Response, verbosity: Verbosity
) -> RenderableType | None:
    """Entry point for formatting a Response object into a Rich Renderable."""
    if verbosity == Verbosity.SUMMARY:
        return _response_summary_rich(response)
    elif verbosity == Verbosity.DETAILED:
        return _response_detailed_rich(response)
    elif verbosity == Verbosity.COMPLETE:
        return _response_complete_rich(response)
    elif verbosity == Verbosity.DEBUG:
        return _response_debug_rich(response)
    return None


def _response_summary_rich(response: Response) -> Panel:
    content = Text()

    # Truncate content
    text = str(response.content or "No content")
    if len(text) > 150:
        text = text[:150] + "..."

    content.append(text, style="blue")

    # Add simple footer
    content.append("\n")
    content.append(f"Model: {response.request.params.model}", style="dim")

    return Panel(
        content,
        title="[bold blue]Response Summary[/bold blue]",
        border_style="blue",
        expand=False,
    )


def _response_detailed_rich(response: Response) -> Panel:
    """Detailed view: User prompt + Response content (truncated) + Metadata."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column("Label", style="bold yellow", justify="right")
    grid.add_column("Content")

    # User
    if response.request:
        user_prompt = _extract_user_prompt(response.request)
        if len(user_prompt) > 300:
            user_prompt = user_prompt[:300] + "..."
        grid.add_row("User:", user_prompt)

    # Spacer
    grid.add_row("", "")

    # Assistant
    content = str(response.content or "No content")
    if len(content) > 500:
        content = content[:500] + "..."
    grid.add_row("[bold blue]Assistant:[/bold blue]", content)

    # Footer Metadata
    meta_text = f"Model: {response.request.params.model}"
    if response.request.params.temperature:
        meta_text += f" • Temp: {response.request.params.temperature}"

    return Panel(
        grid,
        title="[bold]Conversation Detail[/bold]",
        subtitle=f"[dim]{meta_text}[/dim]",
        subtitle_align="right",
        border_style="blue",
    )


def _response_complete_rich(response: Response) -> Panel:
    """Complete view: Full messages, no truncation."""
    grid = Table.grid(padding=(0, 1))
    grid.add_column("Role", style="bold", width=10)
    grid.add_column("Content")

    if response.request:
        for msg in response.request.messages:
            role_style = "green" if msg.role == "system" else "yellow"
            grid.add_row(
                f"[{role_style}]{msg.role.upper()}[/{role_style}]", str(msg.content)
            )
            grid.add_row("", "")  # Spacer

    # Response
    grid.add_row("[blue]ASSISTANT[/blue]", str(response.content))

    return Panel(
        grid,
        title="[bold]Full Conversation[/bold]",
        border_style="green",
        box=box.ROUNDED,
    )


def _response_debug_rich(response: Response) -> Panel:
    """Debug view: Full JSON syntax highlighting."""
    debug_data = response.model_dump(mode="json", exclude_none=True)
    if response.request:
        debug_data["_user_prompt_preview"] = _extract_user_prompt(response.request)

    json_str = json.dumps(debug_data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)

    return Panel(
        syntax,
        title="[bold red]DEBUG: Response Object[/bold red]",
        border_style="red",
    )


# --- Rich Formatters: Errors ---


def format_error_rich(
    error: ConduitError, verbosity: Verbosity
) -> RenderableType | None:
    if verbosity == Verbosity.SUMMARY:
        return _error_summary_rich(error)
    elif verbosity >= Verbosity.DETAILED:
        # Use debug view for detailed/complete/debug on errors
        return _error_debug_rich(error)
    return None


def _error_summary_rich(error: ConduitError) -> Panel:
    content = Text()
    content.append(f"{error.info.message}", style="red")
    return Panel(
        content,
        title=f"[bold red]Error: {error.info.code}[/bold red]",
        border_style="red",
    )


def _error_debug_rich(error: ConduitError) -> Panel:
    debug_data = error.model_dump(mode="json", exclude_none=True)
    json_str = json.dumps(debug_data, indent=2)
    syntax = Syntax(json_str, "json", theme="monokai", word_wrap=True)
    return Panel(syntax, title="[bold red]Error Details[/bold red]", border_style="red")
