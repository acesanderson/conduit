"""
Conduit Cache Management CLI

A standalone tool for inspecting and managing the Postgres-backed cache.
Uses raw SQL for all operations to bypass application-level constraints.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from typing import Any

from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

# Import only for connection pool access
from conduit.storage.cache.postgres_cache_async import AsyncPostgresCache

console = Console()


async def get_pool_with_timeout(cache: AsyncPostgresCache, timeout: float = 5.0):
    """
    Acquire a connection pool with timeout to prevent hangs.

    Args:
        cache: AsyncPostgresCache instance
        timeout: Timeout in seconds

    Returns:
        Connection pool

    Raises:
        asyncio.TimeoutError: If pool acquisition times out
    """
    try:
        return await asyncio.wait_for(cache._get_pool(), timeout=timeout)
    except asyncio.TimeoutError:
        console.print("[red]Error: Timeout acquiring database connection pool[/red]")
        raise


async def list_all_projects(pool) -> None:
    """
    List all projects (cache_name values) with statistics.

    Displays: Project Name, Count, Oldest Timestamp, Newest Timestamp
    """
    query = """
        SELECT 
            cache_name,
            COUNT(*) as entry_count,
            MIN(created_at) as oldest_timestamp,
            MAX(updated_at) as newest_timestamp
        FROM conduit_cache_entries
        GROUP BY cache_name
        ORDER BY cache_name
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)

    if not rows:
        console.print("[yellow]No projects found in cache[/yellow]")
        return

    table = Table(
        title="Conduit Cache Projects", show_header=True, header_style="bold magenta"
    )
    table.add_column("Project Name", style="cyan", no_wrap=True)
    table.add_column("Entry Count", justify="right", style="green")
    table.add_column("Oldest Timestamp", style="blue")
    table.add_column("Newest Timestamp", style="blue")

    for row in rows:
        table.add_row(
            row["cache_name"],
            str(row["entry_count"]),
            row["oldest_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
            row["newest_timestamp"].strftime("%Y-%m-%d %H:%M:%S"),
        )

    console.print(table)
    console.print(f"\n[bold]Total projects:[/bold] {len(rows)}")


async def list_project_entries(pool, project_name: str, limit: int = 50) -> None:
    """
    List cache entries for a specific project (most recent 50).

    Displays: Timestamp, Cache Key

    Args:
        pool: Database connection pool
        project_name: Project name to query
        limit: Maximum number of entries to display
    """
    query = """
        SELECT 
            cache_key,
            updated_at
        FROM conduit_cache_entries
        WHERE cache_name = $1
        ORDER BY updated_at DESC
        LIMIT $2
    """

    async with pool.acquire() as conn:
        rows = await conn.fetch(query, project_name, limit)

    if not rows:
        console.print(f"[yellow]No entries found for project: {project_name}[/yellow]")
        return

    table = Table(
        title=f"Cache Entries for '{project_name}' (Latest {limit})",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("Timestamp", style="blue", no_wrap=True)
    table.add_column("Cache Key", style="cyan")

    for row in rows:
        table.add_row(
            row["updated_at"].strftime("%Y-%m-%d %H:%M:%S"),
            row["cache_key"],
        )

    console.print(table)

    # Get total count
    count_query = "SELECT COUNT(*) FROM conduit_cache_entries WHERE cache_name = $1"
    async with pool.acquire() as conn:
        total_count = await conn.fetchval(count_query, project_name)

    console.print(f"\n[bold]Showing {len(rows)} of {total_count} total entries[/bold]")


async def wipe_project(pool, project_name: str) -> None:
    """
    Delete all cache entries for a project with user confirmation.

    Args:
        pool: Database connection pool
        project_name: Project name to wipe
    """
    # First, get the count
    count_query = "SELECT COUNT(*) FROM conduit_cache_entries WHERE cache_name = $1"
    async with pool.acquire() as conn:
        count = await conn.fetchval(count_query, project_name)

    if count == 0:
        console.print(f"[yellow]No entries found for project: {project_name}[/yellow]")
        return

    # Ask for confirmation
    console.print(
        f"\n[bold red]WARNING:[/bold red] This will delete {count} cache entries for project '{project_name}'"
    )

    # Use input() for confirmation (blocking, but appropriate for CLI)
    try:
        response = input("Are you sure you want to proceed? (y/N): ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[yellow]Operation cancelled[/yellow]")
        return

    if response != "y":
        console.print("[yellow]Operation cancelled[/yellow]")
        return

    # Perform the deletion
    delete_query = "DELETE FROM conduit_cache_entries WHERE cache_name = $1"
    async with pool.acquire() as conn:
        result = await conn.execute(delete_query, project_name)

    # Parse the result (format is like "DELETE 42")
    deleted_count = int(result.split()[-1]) if result else 0
    console.print(
        f"[green]✓ Successfully deleted {deleted_count} entries for project '{project_name}'[/green]"
    )


async def show_last_entry(pool, project_name: str) -> None:
    """
    Fetch and display the payload of the most recent cache entry.

    Args:
        pool: Database connection pool
        project_name: Project name to query
    """
    query = """
        SELECT 
            cache_key,
            payload,
            updated_at
        FROM conduit_cache_entries
        WHERE cache_name = $1
        ORDER BY updated_at DESC
        LIMIT 1
    """

    async with pool.acquire() as conn:
        row = await conn.fetchrow(query, project_name)

    if row is None:
        console.print(f"[yellow]No entries found for project: {project_name}[/yellow]")
        return

    console.print(f"\n[bold]Project:[/bold] {project_name}")
    console.print(f"[bold]Cache Key:[/bold] {row['cache_key']}")
    console.print(
        f"[bold]Timestamp:[/bold] {row['updated_at'].strftime('%Y-%m-%d %H:%M:%S')}"
    )
    console.print("\n[bold]Payload:[/bold]")

    # Parse and pretty-print the JSON payload
    try:
        payload_data = (
            json.loads(row["payload"])
            if isinstance(row["payload"], str)
            else row["payload"]
        )
        payload_json = json.dumps(payload_data, indent=2)

        syntax = Syntax(payload_json, "json", theme="monokai", line_numbers=True)
        console.print(syntax)
    except (json.JSONDecodeError, TypeError) as e:
        console.print(f"[red]Error parsing payload: {e}[/red]")
        console.print(row["payload"])


async def main_flow() -> int:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description="Conduit Cache Management CLI - Inspect and manage Postgres-backed cache",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                    # List all projects
  %(prog)s -p my_project      # List entries for 'my_project'
  %(prog)s -p my_project -l   # Show last entry for 'my_project'
  %(prog)s -p my_project -w   # Wipe all entries for 'my_project'
        """,
    )

    parser.add_argument(
        "-p", "--project", type=str, help="Project name (cache_name) to operate on"
    )
    parser.add_argument(
        "-w",
        "--wipe",
        action="store_true",
        help="Wipe all cache entries for the specified project (requires confirmation)",
    )
    parser.add_argument(
        "-l",
        "--last",
        action="store_true",
        help="Show the payload of the most recent entry for the specified project",
    )
    parser.add_argument(
        "-d",
        "--database",
        type=str,
        default="conduit",
        help="Database name (default: conduit)",
    )

    args = parser.parse_args()

    # Validate argument combinations
    if (args.wipe or args.last) and not args.project:
        console.print("[red]Error: -w/--wipe and -l/--last require -p/--project[/red]")
        return 1

    if args.wipe and args.last:
        console.print("[red]Error: Cannot use -w/--wipe and -l/--last together[/red]")
        return 1

    # Create a cache instance solely for pool access
    # Use a dummy project name since we're bypassing the class methods
    cache = AsyncPostgresCache(project_name="_cli_dummy", db_name=args.database)
    pool = None

    try:
        # Acquire pool with timeout
        pool = await get_pool_with_timeout(cache, timeout=5.0)

        # Execute the requested command
        if args.project:
            if args.wipe:
                await wipe_project(pool, args.project)
            elif args.last:
                await show_last_entry(pool, args.project)
            else:
                await list_project_entries(pool, args.project)
        else:
            # Default: list all projects
            await list_all_projects(pool)

        return 0

    except asyncio.TimeoutError:
        console.print("[red]Error: Database connection timed out[/red]")
        return 1
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback

        if "--debug" in sys.argv:
            console.print(traceback.format_exc())
        return 1
    finally:
        # Always close the pool
        if pool is not None:
            try:
                await cache.aclose()
            except Exception as e:
                console.print(f"[yellow]Warning: Error closing pool: {e}[/yellow]")


def main() -> None:
    """Synchronous entry point for console script."""
    try:
        exit_code = asyncio.run(main_flow())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)


if __name__ == "__main__":
    main()
