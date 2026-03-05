from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from conduit.apps.cli.utils.printer import Printer

logger = logging.getLogger(__name__)


class CacheHandlers:
    @staticmethod
    def handle_cache_ls(
        project_name: str | None,
        printer: Printer,
        loop: asyncio.AbstractEventLoop,
        db_name: str,
    ) -> None:
        from conduit.storage.cache.postgres_cache_async import AsyncPostgresCache
        from rich.table import Table

        # Use an empty string as the project_name placeholder for ls_all;
        # ls_all queries across all cache_names so the instance value is unused.
        cache = AsyncPostgresCache(project_name=project_name or "", db_name=db_name)

        try:
            if project_name is None:
                rows = loop.run_until_complete(cache.ls_all())
                if not rows:
                    printer.print_pretty("No cached entries found.")
                    return

                table = Table(title="Cache Entries (all projects)")
                table.add_column("Cache Name")
                table.add_column("Entries", justify="right")
                table.add_column("Size", justify="right")
                table.add_column("Oldest")
                table.add_column("Newest")

                for row in rows:
                    table.add_row(
                        str(row["cache_name"]),
                        str(row["total_entries"]),
                        _format_size(int(row["total_size_bytes"])),
                        str(row["oldest_record"] or ""),
                        str(row["latest_record"] or ""),
                    )

                printer.print_pretty(table)

            else:
                stats = loop.run_until_complete(cache.cache_stats())
                if stats["total_entries"] == 0:
                    printer.print_pretty(
                        f"No cached entries for project '{project_name}'."
                    )
                    return

                table = Table(title=f"Cache Entries: {project_name}")
                table.add_column("Cache Name")
                table.add_column("Entries", justify="right")
                table.add_column("Size", justify="right")
                table.add_column("Oldest")
                table.add_column("Newest")

                table.add_row(
                    str(stats["cache_name"]),
                    str(stats["total_entries"]),
                    _format_size(int(stats["total_size_bytes"])),
                    str(stats["oldest_record"] or ""),
                    str(stats["latest_record"] or ""),
                )

                printer.print_pretty(table)

        except TimeoutError as e:
            printer.print_pretty(f"[red]Connection timed out: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            logger.error(f"cache ls failed: {e}")
            printer.print_pretty(f"[red]Error: {e}[/red]")
            sys.exit(1)

    @staticmethod
    def handle_cache_clear(
        project_name: str | None,
        all_projects: bool,
        older_than: str | None,
        force: bool,
        printer: Printer,
        loop: asyncio.AbstractEventLoop,
        db_name: str,
    ) -> None:
        from conduit.storage.cache.postgres_cache_async import AsyncPostgresCache

        cache = AsyncPostgresCache(project_name=project_name or "", db_name=db_name)

        if not force:
            from rich.prompt import Confirm

            try:
                confirmed = Confirm.ask(
                    "Are you sure you want to clear the cache?", default=False
                )
            except EOFError:
                printer.print_pretty("Operation cancelled.")
                return
            except KeyboardInterrupt:
                printer.print_pretty("Operation cancelled.")
                return

            if not confirmed:
                printer.print_pretty("Operation cancelled.")
                return

        try:
            if older_than is not None:
                deleted = loop.run_until_complete(cache.delete_older_than(older_than))
            elif all_projects:
                deleted = loop.run_until_complete(cache.wipe_all())
            else:
                stats = loop.run_until_complete(cache.cache_stats())
                deleted = int(stats["total_entries"])
                loop.run_until_complete(cache.wipe())

            printer.print_pretty(f"Cleared {deleted} entries.")

        except TimeoutError as e:
            printer.print_pretty(f"[red]Connection timed out: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            logger.error(f"cache clear failed: {e}")
            printer.print_pretty(f"[red]Error: {e}[/red]")
            sys.exit(1)

    @staticmethod
    def handle_cache_inspect(
        project_name: str,
        printer: Printer,
        loop: asyncio.AbstractEventLoop,
        db_name: str,
    ) -> None:
        from conduit.storage.cache.postgres_cache_async import AsyncPostgresCache
        from rich.syntax import Syntax

        cache = AsyncPostgresCache(project_name=project_name, db_name=db_name)

        try:
            entry = loop.run_until_complete(cache.inspect_latest())
        except TimeoutError as e:
            printer.print_pretty(f"[red]Connection timed out: {e}[/red]")
            sys.exit(1)
        except Exception as e:
            logger.error(f"cache inspect failed: {e}")
            printer.print_pretty(f"[red]Error: {e}[/red]")
            sys.exit(1)

        if entry is None:
            printer.print_pretty(
                f"No cached entries for project '{project_name}'."
            )
            return

        cache_name = entry["cache_name"]
        cache_key = entry["cache_key"]
        updated_at = entry["updated_at"]
        payload = entry["payload"]

        ts_str = updated_at.strftime("%Y-%m-%d %H:%M:%S") if hasattr(updated_at, "strftime") else str(updated_at)

        printer.print_pretty(f"Cache Name: {cache_name}")
        printer.print_pretty(f"Cache Key: {cache_key}")
        printer.print_pretty(f"Timestamp: {ts_str}")

        try:
            json_str = json.dumps(payload, indent=2)
        except TypeError:
            printer.print_pretty("[yellow]Warning: payload is not JSON-serializable. Showing raw repr.[/yellow]")
            printer.print_pretty(repr(payload))
            return

        syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
        printer.print_pretty(syntax)


def _format_size(size_bytes: int) -> str:
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1_048_576:
        return f"{size_bytes / 1024:.1f} KiB"
    else:
        return f"{size_bytes / 1_048_576:.1f} MiB"
