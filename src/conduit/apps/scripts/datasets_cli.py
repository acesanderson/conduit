#!/usr/bin/env python3
"""
conduit-dataset CLI — inventory snapshots for ConduitDataset projects.

Commands:
    status                       List all projects with counts per stage.
    inspect <project>            Detailed breakdown for one project.
    inspect <project> --scores   Same + mean/min/max per eval_function.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "evals"))
sys.path.insert(
    0,
    str(
        Path(__file__).parent.parent.parent.parent.parent.parent
        / "dbclients-project"
        / "src"
    ),
)

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def _counts(pool, project: str) -> dict:
    async with pool.acquire() as conn:
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1", project
        )
        gold = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1 AND reference IS NOT NULL",
            project,
        )
        runs = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project = $1", project
        )
        evals = await conn.fetchval(
            "SELECT COUNT(*) FROM eval_results WHERE project = $1", project
        )
    return {"docs": docs, "gold": gold, "runs": runs, "evals": evals}


async def _scores(pool, project: str) -> list[dict]:
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT strategy, eval_function,
                   ROUND(AVG(score)::numeric, 3) AS mean,
                   ROUND(MIN(score)::numeric, 3) AS min,
                   ROUND(MAX(score)::numeric, 3) AS max
            FROM   eval_results
            WHERE  project = $1
            GROUP  BY strategy, eval_function
            ORDER  BY strategy, eval_function
            """,
            project,
        )
    return [dict(r) for r in rows]


async def cmd_status() -> None:
    from persist import _get_pool, ensure_tables

    try:
        pool = await _get_pool()
    except Exception as e:
        raise ConnectionError(str(e)) from e

    await ensure_tables()

    from dataset import ConduitDatasetAsync

    projects = await ConduitDatasetAsync.list_projects(pool=pool)

    if not projects:
        console.print("[dim]No projects found.[/dim]")
        return

    table = Table(title="ConduitDataset Status", show_lines=True)
    table.add_column("Project", style="cyan", no_wrap=True)
    table.add_column("Documents", justify="right")
    table.add_column("Gold Standards", justify="right")
    table.add_column("Runs", justify="right")
    table.add_column("Evals", justify="right")

    for project in projects:
        c = await _counts(pool, project)
        pending = c["docs"] - c["gold"]
        gold_str = f"{c['gold']} / {c['docs']}" + (
            f"  [yellow]({pending} pending)[/yellow]" if pending else ""
        )
        table.add_row(
            project, str(c["docs"]), gold_str, str(c["runs"]), str(c["evals"])
        )

    console.print(table)


async def cmd_inspect(project: str, show_scores: bool) -> None:
    from persist import _get_pool, ensure_tables

    try:
        pool = await _get_pool()
    except Exception as e:
        raise ConnectionError(str(e)) from e

    await ensure_tables()

    from dataset import ConduitDatasetAsync

    projects = await ConduitDatasetAsync.list_projects(pool=pool)
    if project not in projects:
        console.print(f"[red]Project '{project}' not found.[/red]")
        sys.exit(1)

    c = await _counts(pool, project)
    pending = c["docs"] - c["gold"]

    console.print(
        Panel(
            f"[bold cyan]{project}[/bold cyan]\n\n"
            f"  Documents : {c['docs']}  "
            f"({'[yellow]' + str(pending) + ' pending gold standard[/yellow]' if pending else '[green]all have gold standards[/green]'})\n"
            f"  Runs      : {c['runs']}\n"
            f"  Evals     : {c['evals']}",
            title="ConduitDataset Inspect",
            expand=False,
        )
    )

    if show_scores:
        score_rows = await _scores(pool, project)
        if not score_rows:
            console.print("[dim]No eval scores yet.[/dim]")
            return
        t = Table(title="Scores by Strategy x Eval Function", show_lines=True)
        t.add_column("Strategy", style="cyan")
        t.add_column("Eval Function", style="magenta")
        t.add_column("Mean", justify="right")
        t.add_column("Min", justify="right")
        t.add_column("Max", justify="right")
        for row in score_rows:
            t.add_row(
                row["strategy"],
                row["eval_function"],
                str(row["mean"]),
                str(row["min"]),
                str(row["max"]),
            )
        console.print(t)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="conduit-dataset",
        description="ConduitDataset inventory CLI",
    )
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="List all projects with stage counts")

    ins = sub.add_parser("inspect", help="Detailed breakdown for one project")
    ins.add_argument("project", help="Project name")
    ins.add_argument(
        "--scores", action="store_true", help="Include score stats per eval function"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    host = os.environ.get("POSTGRES_HOST", "localhost")
    loop = asyncio.new_event_loop()
    try:
        if args.command == "status":
            loop.run_until_complete(cmd_status())
        elif args.command == "inspect":
            loop.run_until_complete(cmd_inspect(args.project, args.scores))
    except SystemExit:
        raise
    except ConnectionError as e:
        console.print(
            f"[bold red]Cannot reach postgres at {host}:5432 (database: evals)[/bold red]\n"
            f"[dim]{e}[/dim]"
        )
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        sys.exit(2)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
