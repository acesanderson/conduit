# evals/runner.py
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sys
import traceback
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dataset import ConduitDatasetAsync
    from evals import EvalResult, RunInput, RunResult

logger = logging.getLogger(__name__)

EVAL_FUNCTION = "gemini_judge"


def _config_id(config: dict) -> str:
    from evals import config_to_canonical_dict

    canonical = config_to_canonical_dict(config)
    return hashlib.md5(json.dumps(canonical, sort_keys=True).encode()).hexdigest()[:8]


def _classify_error(exc: BaseException) -> str:
    msg = str(exc)
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if "NETWORK_ERROR" in msg:
        return "network_error"
    if "INTERNAL_ERROR" in msg:
        if "empty response" in msg.lower() or "num_ctx" in msg.lower():
            return "context_overflow"
        return "internal_error"
    if isinstance(exc, ConnectionError) or "Cannot connect" in msg:
        return "connection_error"
    return type(exc).__name__


_FAILURES_DDL = """
CREATE TABLE IF NOT EXISTS run_failures (
    project       TEXT        NOT NULL,
    strategy      TEXT        NOT NULL,
    config_id     TEXT        NOT NULL,
    source_id     TEXT        NOT NULL,
    error_type    TEXT        NOT NULL,
    error_message TEXT,
    token_count   INTEGER,
    traceback     TEXT,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_run_failures_project ON run_failures (project);
CREATE INDEX IF NOT EXISTS idx_run_failures_error   ON run_failures (error_type);
CREATE INDEX IF NOT EXISTS idx_run_failures_source  ON run_failures (source_id);
"""


class ServerCircuitBreaker:
    """
    Opens after `threshold` consecutive failures; blocks new dispatches until
    `cooldown_s` has elapsed.
    """

    def __init__(self, server: str, threshold: int = 5, cooldown_s: float = 60.0):
        self.server = server
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self._consecutive = 0
        self._opened_at: float | None = None
        self._lock = asyncio.Lock()

    async def record_success(self) -> None:
        async with self._lock:
            self._consecutive = 0
            if self._opened_at is not None:
                logger.info("circuit %s: CLOSED on success", self.server)
                self._opened_at = None

    async def record_failure(self) -> None:
        async with self._lock:
            self._consecutive += 1
            if self._consecutive >= self.threshold and self._opened_at is None:
                self._opened_at = asyncio.get_event_loop().time()
                logger.warning(
                    "circuit %s: OPEN after %d consecutive failures — %.0fs cooldown",
                    self.server, self._consecutive, self.cooldown_s,
                )

    async def wait_if_open(self) -> None:
        while True:
            async with self._lock:
                if self._opened_at is None:
                    return
                elapsed = asyncio.get_event_loop().time() - self._opened_at
                if elapsed >= self.cooldown_s:
                    logger.info("circuit %s: RESET after %.0fs cooldown", self.server, elapsed)
                    self._opened_at = None
                    self._consecutive = 0
                    return
                remaining = self.cooldown_s - elapsed
            logger.info("circuit %s: open, waiting %.0fs", self.server, remaining)
            await asyncio.sleep(min(remaining, 5.0))


class EvalRunner:
    def __init__(
        self,
        run_matrix: list[dict],
        dataset_loader: Callable[[], list[RunInput]],
        judge_factory: Callable[[dict], Any],
        project: str,
        log_path: Path,
        status_path: Path,
        smoke_gate: int = 2,
    ) -> None:
        self._run_matrix = run_matrix
        self._dataset_loader = dataset_loader
        self._judge_factory = judge_factory
        self._project = project
        self._log_path = log_path
        self._status_path = status_path
        self._smoke_gate = smoke_gate
        # Derive unique servers + a representative model per server for warmup
        self._servers: dict[str, str] = {}
        for entry in run_matrix:
            alias = entry["server"]
            if alias not in self._servers:
                self._servers[alias] = entry["config"]["model"]
        self._circuit_breakers: dict[str, ServerCircuitBreaker] = {
            alias: ServerCircuitBreaker(alias) for alias in self._servers
        }

    def _setup_logging(self) -> None:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
        # force=True so we win against any library (asyncpg, headwater, etc.)
        # that emitted at import time and locked the root logger into its own
        # handler — that was producing 0-byte log files because the FileHandler
        # was never attached.
        file_handler = logging.FileHandler(self._log_path, mode="w", encoding="utf-8")
        # Flush on every record so SIGKILL at the Cronicle timeout still leaves
        # us a useful tail. Without this we lose the last block-buffered window.
        original_emit = file_handler.emit

        def flushing_emit(record: logging.LogRecord) -> None:
            original_emit(record)
            file_handler.flush()

        file_handler.emit = flushing_emit  # type: ignore[assignment]
        logging.basicConfig(
            level=logging.INFO,
            format=fmt,
            handlers=[
                logging.StreamHandler(sys.stdout),
                file_handler,
            ],
            force=True,
        )

    def _print_matrix(self, docs_count: int | None = None) -> None:
        print(f"Project:   {self._project}")
        if docs_count is not None:
            print(f"Docs:      {docs_count}")
        print(f"Matrix ({len(self._run_matrix)} entries):")
        for e in self._run_matrix:
            cap = f"  cap={e['max_token_count'] // 1000}K" if e.get("max_token_count") else ""
            model = e["config"].get("model", "")
            print(f"  {e['strategy_cls'].__name__:<30} {model:<20} [{e['server']}]{cap}")

    async def _warmup_server(self, alias: str) -> bool:
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        from headwater_api.classes import BatchRequest
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.utils.progress.verbosity import Verbosity

        model = self._servers[alias]
        try:
            params = GenerationParams(model=model, temperature=0.0)
            options = ConduitOptions(
                project_name="warmup",
                include_history=False,
                verbosity=Verbosity.SILENT,
            )
            batch_req = BatchRequest(
                prompt_strings_list=["Hi"],
                params=params,
                options=options,
            )
            async with HeadwaterAsyncClient(host_alias=alias) as client:
                resp = await asyncio.wait_for(
                    client.conduit.query_batch(batch_req), timeout=30.0
                )
            return bool(resp and resp.results)
        except Exception as exc:
            print(f"[cron] {alias} warmup failed: {exc}")
            return False

    async def _health_check(self) -> bool:
        from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
        for alias in self._servers:
            try:
                async with HeadwaterAsyncClient(host_alias=alias) as client:
                    ok = await client.ping()
                if not ok:
                    print(f"[cron] {alias} ping returned False — aborting")
                    return False
            except Exception as exc:
                print(f"[cron] {alias} unreachable: {exc} — aborting")
                return False
            if not await self._warmup_server(alias):
                return False
        return True

    async def _ensure_failures_table(self) -> None:
        from persist import _get_pool
        pool = await _get_pool()
        async with pool.acquire() as conn:
            await conn.execute(_FAILURES_DDL)

    async def _save_failure(
        self,
        strategy: str,
        config_id: str,
        source_id: str,
        error_type: str,
        error_message: str,
        token_count: int | None,
        tb_str: str,
    ) -> None:
        from persist import _get_pool
        pool = await _get_pool()
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO run_failures
                        (project, strategy, config_id, source_id, error_type,
                         error_message, token_count, traceback)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                    """,
                    self._project, strategy, config_id, source_id,
                    error_type, error_message[:500], token_count, tb_str[:4000],
                )
        except Exception as db_exc:
            logger.warning("failed to persist failure record: %s", db_exc)

    def _write_status(self, status: dict) -> None:
        self._status_path.write_text(json.dumps(status, indent=2, default=str))

    def _notify(self, title: str, message: str) -> None:
        logger.info("[notify] %s — %s", title, message)

    async def _run_inference_incremental(
        self,
        docs: list[RunInput],
        config: dict,
        strategy: Any,
        ds: ConduitDatasetAsync,
        timeout_s: int,
        circuit_breaker: ServerCircuitBreaker,
        concurrency: int,
    ) -> list[RunResult]:
        from evals import run_eval
        sem = asyncio.Semaphore(concurrency)
        strategy_name = strategy.__class__.__name__
        cid = _config_id(config)
        inflight: list[int] = [0]

        async def run_and_save(doc: RunInput) -> RunResult | None:
            await circuit_breaker.wait_if_open()
            async with sem:
                inflight[0] += 1
                token_count: int | None = (doc.metadata or {}).get("token_count")
                try:
                    result = await asyncio.wait_for(
                        run_eval(doc, config, strategy), timeout=timeout_s
                    )
                    await ds.runs.save([result])
                    await circuit_breaker.record_success()
                    return result
                except asyncio.TimeoutError as exc:
                    tb_str = traceback.format_exc()
                    logger.warning(
                        "timeout source_id=%s tokens=%s inflight=%d after %ds\n%s",
                        doc.source_id, token_count, inflight[0], timeout_s, tb_str,
                    )
                    await circuit_breaker.record_failure()
                    await self._save_failure(strategy_name, cid, doc.source_id,
                                             "timeout", str(exc), token_count, tb_str)
                    return None
                except Exception as exc:
                    error_type = _classify_error(exc)
                    tb_str = traceback.format_exc()
                    logger.error(
                        "run_eval failed source_id=%s tokens=%s inflight=%d "
                        "strategy=%s error=%s: %s\n%s",
                        doc.source_id, token_count, inflight[0],
                        strategy_name, error_type, exc, tb_str,
                    )
                    await circuit_breaker.record_failure()
                    await self._save_failure(strategy_name, cid, doc.source_id,
                                             error_type, str(exc), token_count, tb_str)
                    return None
                finally:
                    inflight[0] -= 1

        tasks = [asyncio.create_task(run_and_save(doc)) for doc in docs]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    async def _score_missing(
        self,
        ds: ConduitDatasetAsync,
        strategy_name: str,
        cid: str,
        judge: Any,
    ) -> list[EvalResult]:
        from evals import CONCURRENCY_LIMIT, EvalResult
        all_runs = await ds.runs.list(strategy=strategy_name, config_id=cid)
        existing_evals = await ds.evals.list(strategy=strategy_name, config_id=cid)
        scored_ids = {er.run_result.source_id for er in existing_evals}
        unscored = [r for r in all_runs if r.source_id not in scored_ids]

        if not unscored:
            return []

        print(f"  Scoring {len(unscored)} unscored results ...")
        sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

        async def score_and_save(run_result: RunResult) -> EvalResult | None:
            async with sem:
                try:
                    score = await judge(run_result)
                    er = EvalResult(run_result=run_result, score=score)
                    await ds.evals.save([er], eval_function=EVAL_FUNCTION)
                    return er
                except Exception as exc:
                    logger.error(
                        "scoring failed source_id=%s: %s: %s",
                        run_result.source_id, type(exc).__name__, exc,
                    )
                    return None

        tasks = [asyncio.create_task(score_and_save(r)) for r in unscored]
        raw = await asyncio.gather(*tasks)
        return [r for r in raw if r is not None]

    @staticmethod
    def _filter_docs(entry: dict, docs: list[RunInput]) -> list[RunInput]:
        max_tc = entry.get("max_token_count")
        if max_tc is not None:
            docs = [d for d in docs if (d.metadata or {}).get("token_count", 0) <= max_tc]
        predicate = entry.get("doc_predicate")
        if predicate is not None:
            docs = [d for d in docs if predicate(d)]
        return docs

    async def _run_entry(
        self,
        ds: ConduitDatasetAsync,
        entry: dict,
        docs: list[RunInput],
        judge: Any,
        smoke_tested: set[tuple[str, str]],
    ) -> list[RunResult]:
        from evals import generate_runs
        strategy = entry["strategy_cls"]()
        config = entry["config"]
        strategy_name = strategy.__class__.__name__
        server = entry["server"]
        cid = _config_id(config)
        docs = self._filter_docs(entry, docs)
        n_total = len(docs)

        done_ids = {r.source_id for r in await ds.runs.list(strategy=strategy_name, config_id=cid)}
        if len(done_ids) >= n_total:
            await self._score_missing(ds, strategy_name, cid, judge)
            return []

        remaining = [d for d in docs if d.source_id not in done_ids]
        if done_ids:
            print(f"  RESUME {strategy_name}/{cid}: {len(done_ids)}/{n_total} done, {len(remaining)} remaining")
        else:
            print(f"  START  {strategy_name}/{cid} x {n_total} docs  [{server}]")

        smoke_key = (strategy_name, server)
        if not done_ids and smoke_key not in smoke_tested:
            print(f"  Smoke: {strategy_name} on {server} x {self._smoke_gate} docs ...")
            try:
                smoke_results = await generate_runs(
                    inputs=remaining[:self._smoke_gate],
                    configs=[config],
                    strategy=strategy,
                )
                if len(smoke_results) < self._smoke_gate:
                    print(f"  ABORT  {strategy_name}/{server} — smoke test failed "
                          f"({len(smoke_results)}/{self._smoke_gate} succeeded)")
                    return []
                print("  Smoke passed.")
                smoke_tested.add(smoke_key)
            except Exception as exc:
                print(f"  ABORT  {strategy_name}/{server} — smoke test error: {exc}")
                logger.exception("smoke test error strategy=%s server=%s", strategy_name, server)
                return []

        run_results = await self._run_inference_incremental(
            docs=remaining,
            config=config,
            strategy=strategy,
            ds=ds,
            timeout_s=entry["timeout_s"],
            circuit_breaker=self._circuit_breakers[server],
            concurrency=entry.get("concurrency", 1),
        )
        print(f"  Done.  {strategy_name}/{cid}: {len(run_results)}/{len(remaining)} succeeded.")
        await self._score_missing(ds, strategy_name, cid, judge)
        return run_results

    async def publish(self, ds: ConduitDatasetAsync, doc_meta: dict[str, dict]) -> None:
        pass

    async def run(
        self,
        *,
        limit: int | None = None,
        cron: bool = False,
        dry_run: bool = False,
    ) -> None:
        from dataset import ConduitDatasetAsync

        self._setup_logging()

        if dry_run:
            self._print_matrix()
            return

        if cron and not await self._health_check():
            sys.exit(0)

        docs = self._dataset_loader()
        if limit:
            docs = docs[:limit]

        self._print_matrix(docs_count=len(docs))

        ds = ConduitDatasetAsync(self._project)
        await ds.documents.save(docs)
        await self._ensure_failures_table()

        references = {doc.source_id: doc.reference for doc in docs}
        judge = self._judge_factory(references)
        doc_meta = {doc.source_id: doc.metadata for doc in docs}

        smoke_tested: set[tuple[str, str]] = set()
        all_results: list[RunResult] = []
        started_at = datetime.now()

        # Group entries by server; run server groups concurrently, entries within
        # each group sequentially (avoids overloading a single Ollama instance).
        by_server: dict[str, list[dict]] = {}
        for entry in self._run_matrix:
            by_server.setdefault(entry["server"], []).append(entry)

        async def run_server_entries(entries: list[dict]) -> list[RunResult]:
            results = []
            for entry in entries:
                batch = await self._run_entry(ds, entry, docs, judge, smoke_tested)
                results.extend(batch)
            return results

        try:
            server_tasks = [
                asyncio.create_task(run_server_entries(entries))
                for entries in by_server.values()
            ]
            server_results = await asyncio.gather(*server_tasks)
            for batch in server_results:
                all_results.extend(batch)

            print(f"\nTotal new results this run: {len(all_results)}")
            await self.publish(ds, doc_meta)

            self._write_status({
                "result": "ok",
                "started_at": started_at.isoformat(),
                "completed_at": datetime.now().isoformat(),
                "new_results": len(all_results),
            })
            self._notify(f"{self._project} complete", f"{len(all_results)} new results")

        except Exception as exc:
            self._write_status({
                "result": "failed",
                "started_at": started_at.isoformat(),
                "failed_at": datetime.now().isoformat(),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            })
            self._notify(f"{self._project} FAILED", str(exc)[:100])
            raise
