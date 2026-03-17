"""
ConduitDataset — project-scoped handle for documents, runs, and evals.

Usage (async):
    cd = ConduitDatasetAsync("summarize")
    docs = await cd.documents.list()

Usage (sync / scripting):
    with ConduitDataset("summarize") as cd:
        cd.documents.save(items)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "dbclients-project" / "src"))

from evals import EvalResult, RunInput, RunOutput, RunResult

logger = logging.getLogger("conduit.dataset")


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class DocumentsHaveRunsError(Exception):
    """Raised when delete_all() or swap() is called but run_results exist."""


class GoldStandardExistsError(Exception):
    """Raised when save_gold_standards() would overwrite an existing reference."""


class DocumentNotFoundError(Exception):
    """Raised when save_gold_standards() references a source_id not in documents."""


class BatchSaveError(Exception):
    """Raised by evals.save() when one or more per-result transactions fail."""

    def __init__(self, failures: list[tuple[EvalResult, Exception]]):
        self.failures = failures
        super().__init__(f"{len(failures)} eval result(s) failed to save")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_config_id(config: dict) -> str:
    return hashlib.md5(
        json.dumps(config, sort_keys=True).encode()
    ).hexdigest()[:8]


def _doc_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


async def _get_pool_default():
    from persist import _get_pool
    return await _get_pool()


# ---------------------------------------------------------------------------
# DocumentsNamespace
# ---------------------------------------------------------------------------

class DocumentsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn

    async def list(self) -> list[RunInput]:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_id, text, reference, metadata
                FROM   documents
                WHERE  project = $1
                ORDER  BY source_id
                """,
                self._project,
            )
        return [
            RunInput(
                source_id=r["source_id"],
                data=r["text"],
                reference=r["reference"],
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            )
            for r in rows
        ]

    async def save(self, items: list[RunInput]) -> None:
        if not items:
            return
        pool = await self._pool_fn()
        rows = [
            (
                _doc_hash(item.data),
                self._project,
                item.source_id,
                item.data,
                item.reference,
                json.dumps(item.metadata) if item.metadata else None,
            )
            for item in items
        ]
        async with pool.acquire() as conn:
            await conn.executemany(
                """
                INSERT INTO documents
                    (doc_hash, project, source_id, text, reference, metadata)
                VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                ON CONFLICT (project, source_id) DO UPDATE SET
                    doc_hash = EXCLUDED.doc_hash,
                    text     = EXCLUDED.text,
                    metadata = EXCLUDED.metadata
                """,
                rows,
            )
        logger.debug("documents.save project=%s count=%d", self._project, len(items))

    async def swap(self, items: list[RunInput], force: bool = False) -> None:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            async with conn.transaction():
                run_count = await conn.fetchval(
                    "SELECT COUNT(*) FROM run_results WHERE project = $1",
                    self._project,
                )
                if run_count > 0 and not force:
                    raise DocumentsHaveRunsError(
                        f"Project '{self._project}' has {run_count} run(s). "
                        "Call runs.delete_all() first or pass force=True."
                    )
                if force:
                    await conn.execute(
                        "DELETE FROM run_results WHERE project = $1", self._project
                    )
                await conn.execute(
                    "DELETE FROM documents WHERE project = $1", self._project
                )
                if items:
                    rows = [
                        (
                            _doc_hash(item.data),
                            self._project,
                            item.source_id,
                            item.data,
                            item.reference,
                            json.dumps(item.metadata) if item.metadata else None,
                        )
                        for item in items
                    ]
                    await conn.executemany(
                        """
                        INSERT INTO documents
                            (doc_hash, project, source_id, text, reference, metadata)
                        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
                        """,
                        rows,
                    )
        logger.info("documents.swap project=%s count=%d", self._project, len(items))

    async def delete_all(self, force: bool = False) -> None:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            run_count = await conn.fetchval(
                "SELECT COUNT(*) FROM run_results WHERE project = $1", self._project
            )
            if run_count > 0 and not force:
                raise DocumentsHaveRunsError(
                    f"Project '{self._project}' has {run_count} run(s). "
                    "Call runs.delete_all() first or pass force=True."
                )
            if force:
                await conn.execute(
                    "DELETE FROM run_results WHERE project = $1", self._project
                )
            await conn.execute(
                "DELETE FROM documents WHERE project = $1", self._project
            )
        logger.warning(
            "documents.delete_all project=%s force=%s", self._project, force
        )

    async def needs_gold_standard(self) -> list[RunInput]:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT source_id, text, reference, metadata
                FROM   documents
                WHERE  project = $1 AND reference IS NULL
                ORDER  BY source_id
                """,
                self._project,
            )
        return [
            RunInput(
                source_id=r["source_id"],
                data=r["text"],
                reference=None,
                metadata=json.loads(r["metadata"]) if r["metadata"] else None,
            )
            for r in rows
        ]

    async def save_gold_standards(
        self, items: list[RunInput], force: bool = False
    ) -> None:
        to_write = [i for i in items if i.reference is not None]
        if not to_write:
            return

        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            source_ids = [i.source_id for i in to_write]
            existing_rows = await conn.fetch(
                """
                SELECT source_id, reference
                FROM   documents
                WHERE  project = $1 AND source_id = ANY($2::text[])
                """,
                self._project,
                source_ids,
            )
            existing = {r["source_id"]: r["reference"] for r in existing_rows}

            missing = [sid for sid in source_ids if sid not in existing]
            if missing:
                raise DocumentNotFoundError(
                    f"source_id(s) not found in project '{self._project}': {missing}"
                )

            conflicts = [sid for sid, ref in existing.items() if ref is not None]
            if conflicts and not force:
                logger.warning(
                    "save_gold_standards: existing references found project=%s ids=%s",
                    self._project,
                    conflicts,
                )
                raise GoldStandardExistsError(
                    f"Gold standards already exist for: {conflicts}. "
                    "Pass force=True to overwrite."
                )
            if conflicts and force:
                logger.warning(
                    "save_gold_standards: force=True, overwriting project=%s ids=%s",
                    self._project,
                    conflicts,
                )

            rows = [(i.reference, self._project, i.source_id) for i in to_write]
            await conn.executemany(
                """
                UPDATE documents SET reference = $1
                WHERE  project = $2 AND source_id = $3
                """,
                rows,
            )


# ---------------------------------------------------------------------------
# RunsNamespace
# ---------------------------------------------------------------------------

class RunsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn

    async def save(self, results: list[RunResult]) -> None:
        if not results:
            return
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            for r in results:
                config = r.config if isinstance(r.config, dict) else r.config.model_dump()
                config_id = _compute_config_id(config)
                await conn.execute(
                    """
                    INSERT INTO configs (config_id, config)
                    VALUES ($1, $2::jsonb)
                    ON CONFLICT (config_id) DO NOTHING
                    """,
                    config_id,
                    json.dumps(config),
                )
                await conn.execute(
                    """
                    INSERT INTO run_results
                        (project, strategy, config_id, source_id, reference_id,
                         output, output_metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                    ON CONFLICT (project, strategy, config_id, source_id) DO UPDATE SET
                        reference_id    = EXCLUDED.reference_id,
                        output          = EXCLUDED.output,
                        output_metadata = EXCLUDED.output_metadata,
                        created_at      = now()
                    """,
                    self._project,
                    r.strategy,
                    config_id,
                    r.source_id,
                    r.reference_id,
                    r.output.output,
                    json.dumps(r.output.metadata) if r.output.metadata else None,
                )
        logger.debug("runs.save project=%s count=%d", self._project, len(results))

    async def delete_all(self) -> None:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM run_results WHERE project = $1", self._project
            )
        logger.debug("runs.delete_all project=%s", self._project)

    async def list(
        self,
        strategy: str | None = None,
        config_id: str | None = None,
        source_id: str | None = None,
        config: dict | None = None,
    ) -> list[RunResult]:
        if config is not None and config_id is not None:
            raise ValueError(
                "config and config_id are mutually exclusive — provide one or neither"
            )
        if config is not None:
            config_id = _compute_config_id(config)

        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            conditions = ["r.project = $1"]
            params: list[Any] = [self._project]
            idx = 2
            if strategy is not None:
                conditions.append(f"r.strategy = ${idx}")
                params.append(strategy)
                idx += 1
            if config_id is not None:
                conditions.append(f"r.config_id = ${idx}")
                params.append(config_id)
                idx += 1
            if source_id is not None:
                conditions.append(f"r.source_id = ${idx}")
                params.append(source_id)
                idx += 1

            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"""
                SELECT r.strategy, r.config_id, r.source_id, r.reference_id,
                       r.output, r.output_metadata, c.config
                FROM   run_results r
                JOIN   configs c USING (config_id)
                WHERE  {where}
                """,
                *params,
            )
        return [
            RunResult(
                strategy=r["strategy"],
                config_id=r["config_id"],
                source_id=r["source_id"],
                reference_id=r["reference_id"],
                config=json.loads(r["config"]),
                output=RunOutput(
                    output=r["output"],
                    metadata=json.loads(r["output_metadata"]) if r["output_metadata"] else {},
                ),
            )
            for r in rows
        ]

    async def list_configs(self) -> list[dict]:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT c.config, c.created_at
                FROM   run_results r
                JOIN   configs c USING (config_id)
                WHERE  r.project = $1
                ORDER  BY c.created_at ASC
                """,
                self._project,
            )
        return [json.loads(r["config"]) for r in rows]


# ---------------------------------------------------------------------------
# EvalsNamespace
# ---------------------------------------------------------------------------

class EvalsNamespace:
    def __init__(self, project: str, pool_fn: Callable):
        self._project = project
        self._pool_fn = pool_fn

    async def save(
        self, results: list[EvalResult], eval_function: str
    ) -> None:
        if not results:
            return
        pool = await self._pool_fn()
        failures: list[tuple[EvalResult, Exception]] = []

        for er in results:
            r = er.run_result
            config = r.config if isinstance(r.config, dict) else r.config.model_dump()
            # Compute hash from the config dict for configs table upsert.
            # Use r.config_id directly for run_results/eval_results so that a
            # mismatched config_id triggers an FK violation (enables AC12 testing).
            computed_config_id = _compute_config_id(config)
            try:
                async with pool.acquire() as conn:
                    async with conn.transaction():
                        await conn.execute(
                            """
                            INSERT INTO configs (config_id, config)
                            VALUES ($1, $2::jsonb)
                            ON CONFLICT (config_id) DO NOTHING
                            """,
                            computed_config_id,
                            json.dumps(config),
                        )
                        await conn.execute(
                            """
                            INSERT INTO run_results
                                (project, strategy, config_id, source_id,
                                 reference_id, output, output_metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                            ON CONFLICT (project, strategy, config_id, source_id)
                            DO UPDATE SET
                                output          = EXCLUDED.output,
                                output_metadata = EXCLUDED.output_metadata,
                                created_at      = now()
                            """,
                            self._project,
                            r.strategy,
                            r.config_id,
                            r.source_id,
                            r.reference_id,
                            r.output.output,
                            json.dumps(r.output.metadata) if r.output.metadata else None,
                        )
                        await conn.execute(
                            """
                            INSERT INTO eval_results
                                (project, strategy, config_id, source_id,
                                 eval_function, score, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6, $7::jsonb)
                            ON CONFLICT (project, strategy, config_id,
                                         source_id, eval_function)
                            DO UPDATE SET
                                score      = EXCLUDED.score,
                                metadata   = EXCLUDED.metadata,
                                created_at = now()
                            """,
                            self._project,
                            r.strategy,
                            r.config_id,
                            r.source_id,
                            eval_function,
                            er.score,
                            None,
                        )
            except Exception as exc:
                logger.error(
                    "evals.save transaction failed project=%s strategy=%s "
                    "config_id=%s source_id=%s error=%s",
                    self._project,
                    r.strategy,
                    r.config_id,
                    r.source_id,
                    exc,
                )
                failures.append((er, exc))

        if failures:
            logger.error(
                "evals.save BatchSaveError project=%s failures=%d attempted=%d",
                self._project,
                len(failures),
                len(results),
            )
            raise BatchSaveError(failures)

    async def list(
        self,
        eval_function: str | None = None,
        strategy: str | None = None,
    ) -> list[EvalResult]:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            conditions = ["e.project = $1"]
            params: list[Any] = [self._project]
            idx = 2
            if eval_function is not None:
                conditions.append(f"e.eval_function = ${idx}")
                params.append(eval_function)
                idx += 1
            if strategy is not None:
                conditions.append(f"e.strategy = ${idx}")
                params.append(strategy)
                idx += 1
            where = " AND ".join(conditions)
            rows = await conn.fetch(
                f"""
                SELECT e.strategy, e.config_id, e.source_id,
                       e.eval_function, e.score, e.metadata,
                       r.reference_id, r.output, r.output_metadata,
                       c.config
                FROM   eval_results e
                JOIN   run_results r
                       ON  r.project   = e.project
                       AND r.strategy  = e.strategy
                       AND r.config_id = e.config_id
                       AND r.source_id = e.source_id
                JOIN   configs c ON c.config_id = e.config_id
                WHERE  {where}
                """,
                *params,
            )
        return [
            EvalResult(
                run_result=RunResult(
                    strategy=r["strategy"],
                    config_id=r["config_id"],
                    source_id=r["source_id"],
                    reference_id=r["reference_id"],
                    config=json.loads(r["config"]),
                    output=RunOutput(
                        output=r["output"],
                        metadata=json.loads(r["output_metadata"])
                        if r["output_metadata"]
                        else {},
                    ),
                ),
                score=r["score"],
            )
            for r in rows
        ]

    async def delete_all(self) -> None:
        pool = await self._pool_fn()
        async with pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM eval_results WHERE project = $1", self._project
            )


# ---------------------------------------------------------------------------
# ConduitDatasetAsync
# ---------------------------------------------------------------------------

class ConduitDatasetAsync:
    def __init__(self, project: str, pool=None):
        self._project = project
        _p = pool

        async def pool_fn():
            if _p is not None:
                return _p
            return await _get_pool_default()

        self.documents = DocumentsNamespace(project, pool_fn)
        self.runs = RunsNamespace(project, pool_fn)
        self.evals = EvalsNamespace(project, pool_fn)

    @classmethod
    async def list_projects(cls, pool=None) -> list[str]:
        if pool is None:
            pool = await _get_pool_default()
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT DISTINCT project FROM documents ORDER BY project"
            )
        return [r["project"] for r in rows]


# ---------------------------------------------------------------------------
# _SyncProxy + ConduitDatasetSync
# ---------------------------------------------------------------------------

class _SyncProxy:
    """Wraps an async namespace, making every coroutine method blocking."""

    def __init__(self, async_ns: Any, loop: asyncio.AbstractEventLoop, closed_fn: Callable):
        self._ns = async_ns
        self._loop = loop
        self._closed_fn = closed_fn

    def __getattr__(self, name: str):
        if self._closed_fn():
            raise RuntimeError("ConduitDataset is closed")
        attr = getattr(self._ns, name)
        if asyncio.iscoroutinefunction(attr):
            def wrapper(*args, **kwargs):
                if self._closed_fn():
                    raise RuntimeError("ConduitDataset is closed")
                return self._loop.run_until_complete(attr(*args, **kwargs))
            return wrapper
        return attr


class ConduitDatasetSync:
    """
    Sync wrapper around ConduitDatasetAsync.
    Creates a new event loop at __init__. Raises RuntimeError if a loop is
    already running (i.e. called from an async context).
    Supports use as a context manager.
    """

    def __init__(self, project: str, pool=None):
        try:
            running = asyncio.get_event_loop().is_running()
        except RuntimeError:
            running = False
        if running:
            raise RuntimeError("Use ConduitDatasetAsync in async contexts")

        self._loop = asyncio.new_event_loop()
        self._closed = False
        self._async = ConduitDatasetAsync(project, pool=pool)

        def is_closed():
            return self._closed

        self.documents = _SyncProxy(self._async.documents, self._loop, is_closed)
        self.runs = _SyncProxy(self._async.runs, self._loop, is_closed)
        self.evals = _SyncProxy(self._async.evals, self._loop, is_closed)

    def close(self) -> None:
        if not self._closed:
            self._loop.close()
            self._closed = True

    def __enter__(self) -> ConduitDatasetSync:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    @classmethod
    def list_projects(cls) -> list[str]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(ConduitDatasetAsync.list_projects())
        finally:
            loop.close()


ConduitDataset = ConduitDatasetSync
