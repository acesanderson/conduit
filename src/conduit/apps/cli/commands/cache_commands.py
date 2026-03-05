from __future__ import annotations

import asyncio
import click
from typing import TYPE_CHECKING

from conduit.apps.cli.handlers.cache_handlers import CacheHandlers
from conduit.apps.cli.utils.duration import parse_duration

if TYPE_CHECKING:
    from conduit.apps.cli.utils.printer import Printer

handlers = CacheHandlers()


@click.group(invoke_without_command=True)
@click.pass_context
def cache(ctx: click.Context) -> None:
    """Inspect and manage the Postgres query cache."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@cache.command()
@click.argument("project_name", required=False, default=None)
@click.pass_context
def ls(ctx: click.Context, project_name: str | None) -> None:
    """List cache entries. Without PROJECT_NAME, lists all projects."""
    printer: Printer = ctx.obj["printer"]
    loop: asyncio.AbstractEventLoop = ctx.obj["loop"]
    db_name: str = ctx.obj["db_name"]

    handlers.handle_cache_ls(
        project_name=project_name,
        printer=printer,
        loop=loop,
        db_name=db_name,
    )


@cache.command()
@click.argument("project_name", required=False, default=None)
@click.option("--all", "all_projects", is_flag=True, help="Clear all cache namespaces.")
@click.option(
    "--older-than",
    "older_than_raw",
    type=str,
    default=None,
    help="Delete entries older than DURATION (e.g. 7d, 2w, 48h).",
)
@click.option("--force", is_flag=True, help="Skip confirmation prompt.")
@click.pass_context
def clear(
    ctx: click.Context,
    project_name: str | None,
    all_projects: bool,
    older_than_raw: str | None,
    force: bool,
) -> None:
    """Clear cache entries for PROJECT_NAME, or all projects with --all."""
    if not project_name and not all_projects:
        raise click.UsageError("Provide a PROJECT_NAME or use --all.")
    if project_name and all_projects:
        raise click.UsageError("PROJECT_NAME and --all are mutually exclusive.")
    if all_projects and older_than_raw is not None:
        raise click.UsageError("--all and --older-than are mutually exclusive.")

    older_than: str | None = None
    if older_than_raw is not None:
        older_than = parse_duration(older_than_raw)

    printer: Printer = ctx.obj["printer"]
    loop: asyncio.AbstractEventLoop = ctx.obj["loop"]
    db_name: str = ctx.obj["db_name"]

    handlers.handle_cache_clear(
        project_name=project_name,
        all_projects=all_projects,
        older_than=older_than,
        force=force,
        printer=printer,
        loop=loop,
        db_name=db_name,
    )


@cache.command()
@click.argument("project_name")
@click.pass_context
def inspect(ctx: click.Context, project_name: str) -> None:
    """Inspect the most recent cache entry for PROJECT_NAME."""
    printer: Printer = ctx.obj["printer"]
    loop: asyncio.AbstractEventLoop = ctx.obj["loop"]
    db_name: str = ctx.obj["db_name"]

    handlers.handle_cache_inspect(
        project_name=project_name,
        printer=printer,
        loop=loop,
        db_name=db_name,
    )
