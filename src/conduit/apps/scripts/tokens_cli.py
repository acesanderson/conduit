#!/usr/bin/env python3
"""
Odometer Snapshot - Shows usage statistics from the persistent odometer

Usage:
    python snapshot.py
"""

import asyncio
import sys
from datetime import date
from rich.console import Console
from rich.table import Table
from conduit.storage.odometer.pgres.postgres_backend_async import AsyncPostgresOdometer


def format_large_number(num):
    """Format large numbers with commas"""
    if num is None:
        return "0"
    return f"{num:,}"


def print_usage_stats(console, stats):
    """Print usage statistics as a single line"""
    if not stats or stats.get("requests", 0) == 0:
        console.print("[red]No usage data found[/red]")
        return

    console.print("[bold gold3]Usage Statistics[/bold gold3]")
    console.print(
        f"[cyan]Requests:[/cyan] {format_large_number(stats['requests'])}    "
        f"[cyan]Tokens:[/cyan] {format_large_number(stats['total_tokens'])}    "
        f"[cyan]Input:[/cyan] [green]{format_large_number(stats['input'])}[/green]    "
        f"[cyan]Output:[/cyan] [yellow]{format_large_number(stats['output'])}[/yellow]    "
        f"[cyan]Providers:[/cyan] {stats['providers']}    "
        f"[cyan]Models:[/cyan] {stats['models']}"
    )


def print_provider_table(console, provider_stats):
    """Print clean provider table"""
    if not provider_stats:
        return

    console.print(f"\n[bold gold3]Usage by Provider[/bold gold3]")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Provider", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Input", justify="right", style="green")
    table.add_column("Output", justify="right", style="yellow")
    table.add_column("Total", justify="right", style="bold")

    sorted_providers = sorted(
        provider_stats.items(), key=lambda x: x[1]["total"], reverse=True
    )

    for provider, stats in sorted_providers:
        table.add_row(
            provider,
            format_large_number(stats["events"]),
            format_large_number(stats["input"]),
            format_large_number(stats["output"]),
            format_large_number(stats["total"]),
        )

    console.print(table)


def print_models_table(console, model_stats, limit=10, title="Top 10 Models"):
    """Print clean top models table"""
    if not model_stats:
        return

    console.print(f"\n[bold gold3]{title}[/bold gold3]")

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("Rank", justify="center", width=4)
    table.add_column("Model", style="cyan")
    table.add_column("Requests", justify="right")
    table.add_column("Total Tokens", justify="right", style="bold")

    sorted_models = sorted(
        model_stats.items(), key=lambda x: x[1]["total"], reverse=True
    )[:limit]

    for i, (model, stats) in enumerate(sorted_models, 1):
        display_model = model[:35] + "..." if len(model) > 38 else model
        rank_text = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else str(i)

        table.add_row(
            rank_text,
            display_model,
            format_large_number(stats["events"]),
            format_large_number(stats["total"]),
        )

    console.print(table)


async def async_main():
    """Main function to generate the snapshot"""
    console = Console()

    try:
        with console.status("Connecting...", spinner="dots"):
            backend = AsyncPostgresOdometer()

            # Fetch all data concurrently
            overall_task = backend.get_overall_stats()
            provider_task = backend.get_aggregates("provider")
            all_time_task = backend.get_aggregates("model")

            today = date.today()
            daily_task = backend.get_aggregates(
                "model", start_date=today, end_date=today
            )

            # Await all results
            stats = await overall_task
            provider_stats = await provider_task
            all_time_stats = await all_time_task
            daily_stats = await daily_task

        print_usage_stats(console, stats)
        print_provider_table(console, provider_stats)
        print_models_table(console, all_time_stats)

        if daily_stats:
            print_models_table(console, daily_stats, title="Top 10 Models (Today)")
        else:
            console.print(f"\n[bold gold3]No usage today[/bold gold3]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled[/yellow]")
    except Exception as e:
        console.print(f"❌ [red]Error: {e}[/red]")
        # import traceback
        # traceback.print_exc()
        sys.exit(1)


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
