# Eval Runner Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `evals/run2.py` safe for unattended 2 AM cron runs by adding per-doc incremental saves, decoupled scoring, Gemini judge retry logic, per-strategy timeouts, inference warmup, and completion notifications.

**Architecture:** Six targeted changes across two files (`evals/scorer.py`, `evals/run2.py`). Core principle: "save immediately, fail locally." Inference results are committed to DB per-doc rather than batched; scoring is a separate idempotent pass; any single-point failure (Gemini timeout, network blip) is retried or skipped without crashing the run.

**Tech Stack:** Python 3.12+, asyncio, asyncpg, pytest-asyncio (already configured with `asyncio_mode = "auto"`), osascript (macOS notification)

**Immediate goal:** Robust enough to schedule with Cronicle tonight. The `--cron` flag is already in `run2.py`; the `health_check()` added in Task 5 is the entry guard Cronicle needs.

**Longer-term context:** The patterns here — incremental saves, decoupled scoring, health gate, status file — are a first draft of a general "eval event packaging" layer that any eval runner in this framework can adopt. Future work will extract these into reusable infrastructure so new eval pipelines (different tasks, different strategies) can slot into Cronicle scheduling with minimal boilerplate.

---

## File Map

**Modified:**
- `evals/scorer.py` — add `_call_with_retry()` and constants; update `gemini_judge` to use them
- `evals/run2.py` — add `_run_inference_incremental()`, `score_missing()`, `warmup_server()`, `health_check()`, `_write_status()`, `_notify()`, `STATUS_PATH`; update imports, `RUN_MATRIX`, `run_entry()`, `main()`

**Created:**
- `evals/test_resilience.py` — all tests for new functions

**Unchanged:**
- `evals/evals.py` — `generate_runs()` and `evaluate()` stay as-is (smoke test still uses `generate_runs`)
- `evals/dataset.py` — DB layer unchanged; per-doc save works fine with existing `runs.save([single])`

---

## Prerequisites

Before Task 1, verify `pytest-asyncio` is present and configured:

```bash
grep -E "pytest-asyncio|asyncio_mode" pyproject.toml
```

Expected output includes `pytest-asyncio` in deps and `asyncio_mode = "auto"` in `[tool.pytest.ini_options]`. Both are already present — no action needed.

---

### Task 1: Retry logic in `scorer.py`

**Files:**
- Modify: `evals/scorer.py`
- Test: `evals/test_resilience.py`

- [ ] **Step 1: Create test file with failing tests for `_call_with_retry`**

```python
# evals/test_resilience.py
from __future__ import annotations
import asyncio
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.mark.asyncio
async def test_retry_succeeds_on_second_attempt():
    """Transient failure on attempt 1, success on attempt 2."""
    from scorer import _call_with_retry

    call_count = 0

    async def flaky():
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise ValueError("transient")
        return 0.9

    with patch("scorer.asyncio.sleep", new_callable=AsyncMock):
        result = await _call_with_retry(flaky)

    assert result == 0.9
    assert call_count == 2


@pytest.mark.asyncio
async def test_retry_raises_after_exhaustion():
    """All retries exhausted — original exception propagates."""
    from scorer import _call_with_retry

    async def always_fails():
        raise ConnectionError("down")

    with patch("scorer.asyncio.sleep", new_callable=AsyncMock):
        with pytest.raises(ConnectionError, match="down"):
            await _call_with_retry(always_fails)


@pytest.mark.asyncio
async def test_retry_applies_per_call_timeout():
    """Each individual call is bounded by _JUDGE_TIMEOUT."""
    from scorer import _call_with_retry

    async def slow():
        await asyncio.sleep(9999)

    with patch("scorer.asyncio.sleep", new_callable=AsyncMock):
        with patch("scorer._JUDGE_TIMEOUT", 0.01):
            with pytest.raises(Exception):  # TimeoutError or its wrapper
                await _call_with_retry(slow)
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest evals/test_resilience.py::test_retry_succeeds_on_second_attempt evals/test_resilience.py::test_retry_raises_after_exhaustion evals/test_resilience.py::test_retry_applies_per_call_timeout -v
```

Expected: `ImportError` — `_call_with_retry` not yet defined.

- [ ] **Step 3: Add `_call_with_retry` and constants to `scorer.py`**

Add after the existing imports (before `_JUDGE_PROMPT`):

```python
import asyncio

_RETRY_DELAYS = [2.0, 8.0, 32.0]
_JUDGE_TIMEOUT = 45.0


async def _call_with_retry(coro_fn):
    last_exc: BaseException | None = None
    for attempt, pre_delay in enumerate([0.0] + _RETRY_DELAYS):
        if pre_delay:
            await asyncio.sleep(pre_delay)
        try:
            return await asyncio.wait_for(coro_fn(), timeout=_JUDGE_TIMEOUT)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "judge attempt %d/%d failed: %s: %s",
                attempt + 1,
                len(_RETRY_DELAYS) + 1,
                type(exc).__name__,
                exc,
            )
    raise last_exc
```

- [ ] **Step 4: Update `gemini_judge` to use `_call_with_retry`**

Replace the `response = await model.query(...)` line and `return _parse_score(...)` with:

```python
        async def _query():
            response = await model.query(query_input=rendered, params=params, options=options)
            return _parse_score(str(response.content))

        return await _call_with_retry(_query)
```

Full updated `gemini_judge` body for reference:

```python
    async def gemini_judge(run_result) -> float:
        from conduit.core.model.model_async import ModelAsync
        from conduit.core.prompt.prompt import Prompt
        from conduit.domain.request.generation_params import GenerationParams
        from conduit.domain.config.conduit_options import ConduitOptions
        from conduit.utils.progress.verbosity import Verbosity

        reference = references.get(run_result.source_id)
        if not reference:
            logger.warning("No reference found for source_id=%s", run_result.source_id)
            return 0.0

        rendered = Prompt(_JUDGE_PROMPT).render({
            "reference": reference,
            "generated": run_result.output.output,
        })
        model = ModelAsync(model="gemini3")
        params = GenerationParams(model="gemini3", temperature=0.0)
        options = ConduitOptions(
            project_name="summarization_eval",
            verbosity=Verbosity.SILENT,
        )

        async def _query():
            response = await model.query(query_input=rendered, params=params, options=options)
            return _parse_score(str(response.content))

        return await _call_with_retry(_query)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest evals/test_resilience.py::test_retry_succeeds_on_second_attempt evals/test_resilience.py::test_retry_raises_after_exhaustion evals/test_resilience.py::test_retry_applies_per_call_timeout -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add evals/scorer.py evals/test_resilience.py
git commit -m "feat(evals): add retry with exponential backoff to gemini judge (3 retries, 45s per call)"
```

---

### Task 2: Per-strategy timeout in `RUN_MATRIX`

**Files:**
- Modify: `evals/run2.py`
- Test: `evals/test_resilience.py`

- [ ] **Step 1: Write failing test**

```python
# add to evals/test_resilience.py

def test_run_matrix_has_timeout_s():
    """Every RUN_MATRIX entry must declare an explicit timeout_s."""
    from run2 import RUN_MATRIX

    for entry in RUN_MATRIX:
        name = entry["strategy_cls"].__name__
        assert "timeout_s" in entry, f"Missing timeout_s: {name}"
        assert isinstance(entry["timeout_s"], int), f"timeout_s must be int: {name}"
        assert entry["timeout_s"] > 0, f"timeout_s must be positive: {name}"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
uv run pytest evals/test_resilience.py::test_run_matrix_has_timeout_s -v
```

Expected: FAIL — `KeyError: 'timeout_s'`.

- [ ] **Step 3: Replace `RUN_MATRIX` in `run2.py`**

```python
RUN_MATRIX = [
    # Deepwater (qwen3.6)
    {"strategy_cls": RecursiveSummarizer,        "config": _QWEN_RECURSIVE, "server": "deepwater", "timeout_s": 600},
    {"strategy_cls": RollingRefineSummarizer,    "config": _QWEN,           "server": "deepwater", "timeout_s": 1800},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _QWEN,           "server": "deepwater", "timeout_s": 900},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _QWEN,           "server": "deepwater", "timeout_s": 900},
    # Bywater (gpt-oss)
    {"strategy_cls": RecursiveSummarizer,        "config": _GPT,            "server": "bywater",   "timeout_s": 600},
    {"strategy_cls": RollingRefineSummarizer,    "config": _GPT,            "server": "bywater",   "timeout_s": 1800},
    {"strategy_cls": MapDedupeReduceSummarizer,  "config": _GPT,            "server": "bywater",   "timeout_s": 900},
    {"strategy_cls": HierarchicalTreeSummarizer, "config": _GPT,            "server": "bywater",   "timeout_s": 900},
]
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
uv run pytest evals/test_resilience.py::test_run_matrix_has_timeout_s -v
```

Expected: PASSED.

- [ ] **Step 5: Commit**

```bash
git add evals/run2.py evals/test_resilience.py
git commit -m "feat(evals): add per-strategy timeout_s to RUN_MATRIX (600–1800s)"
```

---

### Task 3: Streaming inference with per-doc saves

**Files:**
- Modify: `evals/run2.py`
- Test: `evals/test_resilience.py`

- [ ] **Step 1: Write failing tests**

```python
# add to evals/test_resilience.py

@pytest.mark.asyncio
async def test_incremental_save_called_per_doc():
    """ds.runs.save is called once per successful doc, not once for all."""
    from run2 import _run_inference_incremental
    from evals import RunInput, RunResult, RunOutput

    docs = [
        RunInput(source_id="d1", data="text1"),
        RunInput(source_id="d2", data="text2"),
        RunInput(source_id="d3", data="text3"),
    ]
    config = {"model": "gpt-oss:latest"}

    def make_result(source_id):
        return RunResult(
            strategy="MockStrategy",
            config_id="abcd1234",
            source_id=source_id,
            config=config,
            output=RunOutput(output="summary", metadata={}),
        )

    save_calls: list[str] = []

    async def mock_run_eval(doc, cfg, strategy):
        return make_result(doc.source_id)

    mock_ds = MagicMock()
    mock_ds.runs.save = AsyncMock(
        side_effect=lambda results: save_calls.append(results[0].source_id)
    )

    with patch("run2.run_eval", side_effect=mock_run_eval):
        results = await _run_inference_incremental(
            docs, config, MagicMock(), mock_ds, timeout_s=60
        )

    assert len(results) == 3
    assert len(save_calls) == 3  # one save per doc, not one batch
    assert set(save_calls) == {"d1", "d2", "d3"}


@pytest.mark.asyncio
async def test_timeout_skips_doc_not_crash():
    """A timed-out doc is skipped; the rest still complete and save."""
    from run2 import _run_inference_incremental
    from evals import RunInput, RunResult, RunOutput

    docs = [
        RunInput(source_id="fast", data="short"),
        RunInput(source_id="slow", data="huge"),
    ]
    config = {"model": "gpt-oss:latest"}

    async def mock_run_eval(doc, cfg, strategy):
        if doc.source_id == "slow":
            await asyncio.sleep(9999)
        return RunResult(
            strategy="MockStrategy",
            config_id="abcd1234",
            source_id=doc.source_id,
            config=cfg,
            output=RunOutput(output="ok", metadata={}),
        )

    mock_ds = MagicMock()
    mock_ds.runs.save = AsyncMock()

    with patch("run2.run_eval", side_effect=mock_run_eval):
        results = await _run_inference_incremental(
            docs, config, MagicMock(), mock_ds, timeout_s=0.01
        )

    assert len(results) == 1
    assert results[0].source_id == "fast"
    assert mock_ds.runs.save.call_count == 1
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest evals/test_resilience.py::test_incremental_save_called_per_doc evals/test_resilience.py::test_timeout_skips_doc_not_crash -v
```

Expected: FAIL — `ImportError: cannot import name '_run_inference_incremental' from 'run2'`.

- [ ] **Step 3: Update the `evals` import line in `run2.py`**

Replace:
```python
from evals import RunInput, RunResult, generate_runs, evaluate
```
With:
```python
from evals import (
    CONCURRENCY_LIMIT,
    EvalResult,
    RunInput,
    RunResult,
    evaluate,
    generate_runs,
    run_eval,
)
```

- [ ] **Step 4: Add `_run_inference_incremental` to `run2.py`**

Add after `get_done_ids`:

```python
async def _run_inference_incremental(
    docs: list[RunInput],
    config: dict,
    strategy,
    ds: ConduitDatasetAsync,
    timeout_s: int,
) -> list[RunResult]:
    sem = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def run_and_save(doc: RunInput) -> RunResult | None:
        async with sem:
            try:
                result = await asyncio.wait_for(
                    run_eval(doc, config, strategy), timeout=timeout_s
                )
                await ds.runs.save([result])
                return result
            except asyncio.TimeoutError:
                logger.warning(
                    "timeout source_id=%s after %ds", doc.source_id, timeout_s
                )
                return None
            except Exception as exc:
                logger.error(
                    "run_eval failed source_id=%s: %s: %s",
                    doc.source_id,
                    type(exc).__name__,
                    exc,
                )
                return None

    tasks = [asyncio.create_task(run_and_save(doc)) for doc in docs]
    raw = await asyncio.gather(*tasks)
    return [r for r in raw if r is not None]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest evals/test_resilience.py::test_incremental_save_called_per_doc evals/test_resilience.py::test_timeout_skips_doc_not_crash -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Commit**

```bash
git add evals/run2.py evals/test_resilience.py
git commit -m "feat(evals): add streaming inference with per-doc saves to run_entry"
```

---

### Task 4: `score_missing()` + decouple scoring from `run_entry`

**Files:**
- Modify: `evals/run2.py`
- Test: `evals/test_resilience.py`

- [ ] **Step 1: Write failing tests**

```python
# add to evals/test_resilience.py

@pytest.mark.asyncio
async def test_score_missing_skips_already_scored():
    """Docs with existing eval_results are not re-scored."""
    from run2 import score_missing
    from evals import RunResult, RunOutput, EvalResult

    def make_run(sid):
        return RunResult(
            strategy="S", config_id="c1", source_id=sid,
            config={}, output=RunOutput(output="x", metadata={})
        )

    mock_ds = MagicMock()
    mock_ds.runs.list = AsyncMock(
        return_value=[make_run("d1"), make_run("d2"), make_run("d3")]
    )
    mock_ds.evals.list = AsyncMock(
        return_value=[
            EvalResult(run_result=make_run("d1"), score=0.8),
            EvalResult(run_result=make_run("d2"), score=0.8),
        ]
    )
    mock_ds.evals.save = AsyncMock()
    judge = AsyncMock(return_value=0.7)

    results = await score_missing(mock_ds, "S", "c1", judge)

    assert len(results) == 1
    assert results[0].run_result.source_id == "d3"
    assert judge.call_count == 1


@pytest.mark.asyncio
async def test_score_missing_saves_per_doc():
    """ds.evals.save is called once per doc, not batched."""
    from run2 import score_missing
    from evals import RunResult, RunOutput

    def make_run(sid):
        return RunResult(
            strategy="S", config_id="c1", source_id=sid,
            config={}, output=RunOutput(output="x", metadata={})
        )

    mock_ds = MagicMock()
    mock_ds.runs.list = AsyncMock(return_value=[make_run("d1"), make_run("d2")])
    mock_ds.evals.list = AsyncMock(return_value=[])
    mock_ds.evals.save = AsyncMock()
    judge = AsyncMock(return_value=0.6)

    await score_missing(mock_ds, "S", "c1", judge)

    assert mock_ds.evals.save.call_count == 2  # one per doc


@pytest.mark.asyncio
async def test_score_missing_handles_judge_failure():
    """A judge failure on one doc does not prevent others from scoring."""
    from run2 import score_missing
    from evals import RunResult, RunOutput

    def make_run(sid):
        return RunResult(
            strategy="S", config_id="c1", source_id=sid,
            config={}, output=RunOutput(output="x", metadata={})
        )

    mock_ds = MagicMock()
    mock_ds.runs.list = AsyncMock(return_value=[make_run("d1"), make_run("d2")])
    mock_ds.evals.list = AsyncMock(return_value=[])
    mock_ds.evals.save = AsyncMock()

    async def flaky_judge(run_result):
        if run_result.source_id == "d1":
            raise RuntimeError("gemini down")
        return 0.8

    results = await score_missing(mock_ds, "S", "c1", flaky_judge)

    assert len(results) == 1
    assert results[0].run_result.source_id == "d2"
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest evals/test_resilience.py::test_score_missing_skips_already_scored evals/test_resilience.py::test_score_missing_saves_per_doc evals/test_resilience.py::test_score_missing_handles_judge_failure -v
```

Expected: FAIL — `ImportError: cannot import name 'score_missing' from 'run2'`.

- [ ] **Step 3: Add `score_missing` to `run2.py`**

Add after `_run_inference_incremental`:

```python
async def score_missing(
    ds: ConduitDatasetAsync,
    strategy_name: str,
    cid: str,
    judge,
) -> list[EvalResult]:
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
                    run_result.source_id,
                    type(exc).__name__,
                    exc,
                )
                return None

    tasks = [asyncio.create_task(score_and_save(r)) for r in unscored]
    raw = await asyncio.gather(*tasks)
    return [r for r in raw if r is not None]
```

- [ ] **Step 4: Update `run_entry` to use `_run_inference_incremental` and `score_missing`**

Replace the block from `run_results = await generate_runs(...)` through the end of `run_entry` with:

```python
    run_results = await _run_inference_incremental(
        docs=remaining,
        config=config,
        strategy=strategy,
        ds=ds,
        timeout_s=entry["timeout_s"],
    )
    print(f"  Done.  {strategy_name}/{cid}: {len(run_results)}/{len(remaining)} succeeded.")

    await score_missing(ds, strategy_name, cid, judge)

    return run_results
```

Replace the `if len(done_ids) >= n_total:` branch with:

```python
    if len(done_ids) >= n_total:
        await score_missing(ds, strategy_name, cid, judge)
        return []
```

The complete updated `run_entry` for reference:

```python
async def run_entry(
    ds: ConduitDatasetAsync,
    entry: dict,
    docs: list[RunInput],
    judge,
    smoke_tested: set[tuple[str, str]],
) -> list[RunResult]:
    strategy = entry["strategy_cls"]()
    config = entry["config"]
    strategy_name = strategy.__class__.__name__
    server = entry["server"]
    cid = _config_id(config)
    n_total = len(docs)

    done_ids = await get_done_ids(ds, strategy_name, cid)
    if len(done_ids) >= n_total:
        await score_missing(ds, strategy_name, cid, judge)
        return []

    remaining = [d for d in docs if d.source_id not in done_ids]
    if done_ids:
        print(f"  RESUME {strategy_name}/{cid}: {len(done_ids)}/{n_total} done, {len(remaining)} remaining")
    else:
        print(f"  START  {strategy_name}/{cid} × {n_total} docs  [{server}]")

    smoke_key = (strategy_name, server)
    if not done_ids and smoke_key not in smoke_tested:
        print(f"  Smoke: {strategy_name} on {server} × 2 docs ...")
        try:
            smoke_results = await generate_runs(inputs=remaining[:2], configs=[config], strategy=strategy)
            if len(smoke_results) < 2:
                print(f"  ABORT  {strategy_name}/{server} — smoke test failed ({len(smoke_results)}/2 succeeded)")
                return []
            print(f"  Smoke passed.")
            smoke_tested.add(smoke_key)
        except Exception as exc:
            print(f"  ABORT  {strategy_name}/{server} — smoke test error: {exc}")
            logger.exception("smoke test error strategy=%s server=%s", strategy_name, server)
            return []

    run_results = await _run_inference_incremental(
        docs=remaining,
        config=config,
        strategy=strategy,
        ds=ds,
        timeout_s=entry["timeout_s"],
    )
    print(f"  Done.  {strategy_name}/{cid}: {len(run_results)}/{len(remaining)} succeeded.")

    await score_missing(ds, strategy_name, cid, judge)

    return run_results
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest evals/test_resilience.py::test_score_missing_skips_already_scored evals/test_resilience.py::test_score_missing_saves_per_doc evals/test_resilience.py::test_score_missing_handles_judge_failure -v
```

Expected: 3 PASSED.

- [ ] **Step 6: Commit**

```bash
git add evals/run2.py evals/test_resilience.py
git commit -m "feat(evals): decouple scoring from inference via score_missing(); update run_entry"
```

---

### Task 5: Cron warmup

**Files:**
- Modify: `evals/run2.py`
- Test: `evals/test_resilience.py`

- [ ] **Step 1: Write failing tests**

```python
# add to evals/test_resilience.py

@pytest.mark.asyncio
async def test_warmup_server_returns_true_on_success():
    from run2 import warmup_server

    mock_resp = MagicMock()
    mock_resp.results = ["result"]
    mock_client = AsyncMock()
    mock_client.conduit.query_batch = AsyncMock(return_value=mock_resp)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("run2.HeadwaterAsyncClient", return_value=mock_client):
        result = await warmup_server("bywater")

    assert result is True


@pytest.mark.asyncio
async def test_warmup_server_returns_false_on_exception():
    from run2 import warmup_server

    mock_client = AsyncMock()
    mock_client.__aenter__ = AsyncMock(side_effect=ConnectionRefusedError("no server"))
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch("run2.HeadwaterAsyncClient", return_value=mock_client):
        result = await warmup_server("deepwater")

    assert result is False
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest evals/test_resilience.py::test_warmup_server_returns_true_on_success evals/test_resilience.py::test_warmup_server_returns_false_on_exception -v
```

Expected: FAIL — `ImportError: cannot import name 'warmup_server' from 'run2'`.

- [ ] **Step 3: Add module-level import for `HeadwaterAsyncClient` to `run2.py`**

Add near the top of `run2.py` (after the stdlib imports, before the local imports):

```python
from headwater_client.client.headwater_client_async import HeadwaterAsyncClient
```

Remove the lazy `from headwater_client...` import that was inside `ping_servers()`.

- [ ] **Step 4: Add `warmup_server` and replace `ping_servers` with `health_check` in `run2.py`**

Delete `ping_servers`. Add in its place:

```python
async def warmup_server(alias: str) -> bool:
    from headwater_api.classes import BatchRequest
    from conduit.domain.request.generation_params import GenerationParams
    from conduit.domain.config.conduit_options import ConduitOptions
    from conduit.utils.progress.verbosity import Verbosity

    model = "qwen3.6:latest" if alias == "deepwater" else "gpt-oss:latest"
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


async def health_check() -> bool:
    for alias in ("deepwater", "bywater"):
        try:
            async with HeadwaterAsyncClient(host_alias=alias) as client:
                ok = await client.ping()
            if not ok:
                print(f"[cron] {alias} ping returned False — aborting")
                return False
        except Exception as exc:
            print(f"[cron] {alias} unreachable: {exc} — aborting")
            return False

        if not await warmup_server(alias):
            print(f"[cron] {alias} warmup failed — aborting")
            return False

    return True
```

Update `main()` to call `health_check()` instead of `ping_servers()`:

```python
    if args.cron:
        if not await health_check():
            sys.exit(0)
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest evals/test_resilience.py::test_warmup_server_returns_true_on_success evals/test_resilience.py::test_warmup_server_returns_false_on_exception -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Commit**

```bash
git add evals/run2.py evals/test_resilience.py
git commit -m "feat(evals): add inference warmup to cron health gate; replace ping_servers with health_check"
```

---

### Task 6: Completion notification and status file

**Files:**
- Modify: `evals/run2.py`
- Test: `evals/test_resilience.py`

- [ ] **Step 1: Write failing tests**

```python
# add to evals/test_resilience.py

def test_write_status_creates_json_file(tmp_path):
    from run2 import _write_status

    status_path = tmp_path / "run2_status.json"
    with patch("run2.STATUS_PATH", status_path):
        _write_status({"result": "ok", "new_results": 42})

    written = json.loads(status_path.read_text())
    assert written["result"] == "ok"
    assert written["new_results"] == 42


def test_notify_calls_osascript():
    from run2 import _notify

    with patch("run2.subprocess.run") as mock_run:
        _notify("run2 complete", "200 new results")

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "osascript"
    assert "run2 complete" in args[2]
    assert "200 new results" in args[2]
```

- [ ] **Step 2: Run tests to confirm they fail**

```bash
uv run pytest evals/test_resilience.py::test_write_status_creates_json_file evals/test_resilience.py::test_notify_calls_osascript -v
```

Expected: FAIL — `ImportError` for `_write_status` and `STATUS_PATH`.

- [ ] **Step 3: Add imports, constants, and helpers to `run2.py`**

Add to the stdlib imports at the top:

```python
import json
import subprocess
import traceback
from datetime import datetime
```

Add before `main()`:

```python
STATUS_PATH = Path(__file__).parent / "run2_status.json"


def _write_status(status: dict) -> None:
    STATUS_PATH.write_text(json.dumps(status, indent=2, default=str))


def _notify(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    subprocess.run(["osascript", "-e", script], capture_output=True)
```

- [ ] **Step 4: Wrap the run body in `main()` with try/except**

Replace the body of `main()` from `ds = ConduitDatasetAsync(...)` through `await print_results(...)` with:

```python
    ds = ConduitDatasetAsync(args.project)

    print("\nSeeding documents to DB...")
    await ds.documents.save(docs)

    references = {doc.source_id: doc.reference for doc in docs}
    judge = make_gemini_judge(references)
    doc_meta = {doc.source_id: doc.metadata for doc in docs}

    smoke_tested: set[tuple[str, str]] = set()
    all_results: list[RunResult] = []
    started_at = datetime.now()

    try:
        async def run_bywater() -> list[RunResult]:
            results = []
            for entry in bywater_entries:
                batch = await run_entry(ds, entry, docs, judge, smoke_tested)
                results.extend(batch)
            return results

        print("\nRunning summarizations:")
        bywater_task = asyncio.create_task(run_bywater())

        for entry in deepwater_entries:
            batch = await run_entry(ds, entry, docs, judge, smoke_tested)
            all_results.extend(batch)

        bywater_results = await bywater_task
        all_results.extend(bywater_results)

        print(f"\nTotal new results this run: {len(all_results)}")
        await print_results(ds, doc_meta)

        _write_status({
            "result": "ok",
            "started_at": started_at.isoformat(),
            "completed_at": datetime.now().isoformat(),
            "new_results": len(all_results),
        })
        _notify("run2 complete", f"{len(all_results)} new results")

    except Exception as exc:
        _write_status({
            "result": "failed",
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now().isoformat(),
            "error": str(exc),
            "traceback": traceback.format_exc(),
        })
        _notify("run2 FAILED", str(exc)[:100])
        raise
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
uv run pytest evals/test_resilience.py::test_write_status_creates_json_file evals/test_resilience.py::test_notify_calls_osascript -v
```

Expected: 2 PASSED.

- [ ] **Step 6: Run full test suite**

```bash
uv run pytest evals/test_resilience.py -v
```

Expected: All 14 tests PASSED.

- [ ] **Step 7: Commit**

```bash
git add evals/run2.py evals/test_resilience.py
git commit -m "feat(evals): add completion notification and status file on success/failure"
```

---

## Self-Review

**Spec coverage:**

| Problem | Task |
|---------|------|
| Gemini judge timeouts crash the run | Task 1 — retry with 45s per-call timeout, 3 retries |
| No inference/scoring separation | Task 4 — `score_missing()` extracted; `run_entry` calls it separately |
| Resumability mismatch | Task 4 — `score_missing()` re-queries DB for unscored; safe to call repeatedly |
| 600s timeout too blunt | Task 2 — per-strategy `timeout_s`; Task 3 — passed to `_run_inference_incremental` |
| No per-run persistence | Task 3 — `ds.runs.save([result])` called immediately after each doc |
| Cron health gate only pings | Task 5 — `warmup_server()` fires a 1-token inference request before committing to a run |
| No alerting | Task 6 — `_write_status()` + `_notify()` on both success and failure |

**Placeholder scan:** None.

**Type consistency:**
- `EvalResult` imported in Task 3 update; used in `score_missing()` return type (Task 4) ✓
- `run_eval` imported in Task 3; used in `_run_inference_incremental` (Task 3) ✓
- `CONCURRENCY_LIMIT` imported in Task 3; used in both Task 3 and Task 4 functions ✓
- `HeadwaterAsyncClient` moved to module-level import in Task 5; patched as `run2.HeadwaterAsyncClient` in tests ✓
- `STATUS_PATH` defined as module-level in Task 6; patched as `run2.STATUS_PATH` in test ✓
- `entry["timeout_s"]` added in Task 2; consumed in Task 3 via `_run_inference_incremental(timeout_s=entry["timeout_s"])` ✓
