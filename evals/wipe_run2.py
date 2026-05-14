"""Wipe all run2 project data from the evals DB. Run once before a full restart."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PROJECT = "run2_strategy_comparison"


async def main() -> None:
    from dbclients.clients.postgres import get_postgres_client

    pool = await get_postgres_client("async", dbname="evals")
    async with pool.acquire() as conn:
        n_eval = await conn.fetchval(
            "SELECT COUNT(*) FROM eval_results WHERE project = $1", PROJECT
        )
        n_run = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project = $1", PROJECT
        )
        n_fail = await conn.fetchval(
            "SELECT COUNT(*) FROM run_failures WHERE project = $1", PROJECT
        )
        n_doc = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1", PROJECT
        )
        print(f"About to delete for project={PROJECT!r}:")
        print(f"  eval_results : {n_eval}")
        print(f"  run_results  : {n_run}  (eval_results cascade-deleted)")
        print(f"  run_failures : {n_fail}")
        print(f"  documents    : {n_doc}")

        confirm = input("Proceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

        await conn.execute("DELETE FROM run_results WHERE project = $1", PROJECT)
        await conn.execute("DELETE FROM run_failures WHERE project = $1", PROJECT)
        await conn.execute("DELETE FROM documents WHERE project = $1", PROJECT)
        print("Done.")

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
