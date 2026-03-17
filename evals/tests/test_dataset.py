from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from evals import EvalResult, RunInput, RunOutput, RunResult
from dataset import (
    BatchSaveError,
    ConduitDataset,
    ConduitDatasetAsync,
    DocumentNotFoundError,
    DocumentsHaveRunsError,
    DocumentsNamespace,
    GoldStandardExistsError,
    _SyncProxy,
    _compute_config_id,
)


# ── AC 1 ──────────────────────────────────────────────────────────────────────
async def test_documents_list_returns_run_inputs_with_reference(cd):
    """AC 1"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text one", reference="ref one"),
        RunInput(source_id="doc2", data="text two", reference="ref two"),
        RunInput(source_id="doc3", data="text three"),
    ])
    result = await cd.documents.list()
    assert len(result) == 3
    by_id = {r.source_id: r for r in result}
    assert by_id["doc1"].reference == "ref one"
    assert by_id["doc2"].reference == "ref two"
    assert by_id["doc3"].reference is None
    assert by_id["doc1"].data == "text one"


# ── AC 2 ──────────────────────────────────────────────────────────────────────
async def test_documents_swap_is_atomic(cd):
    """AC 2"""
    await cd.documents.save([RunInput(source_id="original", data="original text")])
    with pytest.raises(Exception):
        await cd.documents.swap([
            RunInput(source_id="good", data="good text"),
            RunInput(source_id="bad", data=None),  # NOT NULL violation
        ])
    remaining = await cd.documents.list()
    assert len(remaining) == 1
    assert remaining[0].source_id == "original"


# ── AC 3 ──────────────────────────────────────────────────────────────────────
async def test_documents_delete_all_raises_if_runs_exist(cd, pool, project):
    """AC 3 — guard"""
    await cd.documents.save([RunInput(source_id="doc1", data="text")])
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO configs (config_id, config) VALUES ('aaaaaaaa', '{}'::jsonb)"
            " ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            """
            INSERT INTO run_results (project, strategy, config_id, source_id, output)
            VALUES ($1, 'Strat', 'aaaaaaaa', 'doc1', 'out')
            ON CONFLICT DO NOTHING
            """,
            project,
        )
    with pytest.raises(DocumentsHaveRunsError):
        await cd.documents.delete_all()


async def test_documents_delete_all_force_clears_all(cd, pool, project):
    """AC 3 — force=True"""
    await cd.documents.save([RunInput(source_id="doc1", data="text")])
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO configs (config_id, config) VALUES ('aaaaaaaa', '{}'::jsonb)"
            " ON CONFLICT DO NOTHING"
        )
        await conn.execute(
            """
            INSERT INTO run_results (project, strategy, config_id, source_id, output)
            VALUES ($1, 'Strat', 'aaaaaaaa', 'doc1', 'out')
            ON CONFLICT DO NOTHING
            """,
            project,
        )
    await cd.documents.delete_all(force=True)
    async with pool.acquire() as conn:
        docs = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1", project
        )
        runs = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project = $1", project
        )
        evals = await conn.fetchval(
            "SELECT COUNT(*) FROM eval_results WHERE project = $1", project
        )
    assert docs == 0
    assert runs == 0
    assert evals == 0


# ── AC 4 ──────────────────────────────────────────────────────────────────────
async def test_documents_save_does_not_overwrite_reference(cd):
    """AC 4"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text", reference="keep this")
    ])
    await cd.documents.save([
        RunInput(source_id="doc1", data="text updated")
    ])
    result = await cd.documents.list()
    assert result[0].reference == "keep this"


# ── AC 5 ──────────────────────────────────────────────────────────────────────
async def test_save_gold_standards_raises_if_reference_exists(cd):
    """AC 5 — pre-validate, no partial writes"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text 1", reference="existing"),
        RunInput(source_id="doc2", data="text 2"),
    ])
    with pytest.raises(GoldStandardExistsError):
        await cd.documents.save_gold_standards([
            RunInput(source_id="doc1", data="text 1", reference="new gold"),
            RunInput(source_id="doc2", data="text 2", reference="new gold 2"),
        ])
    result = await cd.documents.list()
    by_id = {r.source_id: r for r in result}
    assert by_id["doc2"].reference is None


async def test_save_gold_standards_force_overwrites(cd):
    """AC 5 — force=True"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text", reference="old ref")
    ])
    await cd.documents.save_gold_standards(
        [RunInput(source_id="doc1", data="text", reference="new ref")],
        force=True,
    )
    result = await cd.documents.list()
    assert result[0].reference == "new ref"


# ── AC 6 ──────────────────────────────────────────────────────────────────────
async def test_save_gold_standards_raises_if_source_id_missing(cd):
    """AC 6"""
    await cd.documents.save([RunInput(source_id="doc1", data="text")])
    with pytest.raises(DocumentNotFoundError):
        await cd.documents.save_gold_standards([
            RunInput(source_id="doc1", data="text", reference="gold 1"),
            RunInput(source_id="missing", data="x", reference="gold missing"),
        ])
    result = await cd.documents.list()
    assert result[0].reference is None


# ── AC 7 ──────────────────────────────────────────────────────────────────────
async def test_save_gold_standards_skips_none_reference(cd):
    """AC 7"""
    await cd.documents.save([
        RunInput(source_id="doc1", data="text 1"),
        RunInput(source_id="doc2", data="text 2"),
    ])
    await cd.documents.save_gold_standards([
        RunInput(source_id="doc1", data="text 1", reference="gold 1"),
        RunInput(source_id="doc2", data="text 2", reference=None),
    ])
    result = await cd.documents.list()
    by_id = {r.source_id: r for r in result}
    assert by_id["doc1"].reference == "gold 1"
    assert by_id["doc2"].reference is None


# ── AC 8 ──────────────────────────────────────────────────────────────────────
async def test_runs_list_filters_by_config_dict(cd, project):
    """AC 8"""
    config_a = {"model": "gpt-oss:latest"}
    config_b = {"model": "deepseek:latest"}
    id_a = _compute_config_id(config_a)
    id_b = _compute_config_id(config_b)

    await cd.runs.save([
        RunResult(strategy="S", config_id=id_a, source_id="doc1",
                  config=config_a, output=RunOutput(output="out", metadata={})),
        RunResult(strategy="S", config_id=id_b, source_id="doc1",
                  config=config_b, output=RunOutput(output="out", metadata={})),
    ])
    results = await cd.runs.list(config=config_a)
    assert len(results) == 1
    assert results[0].config_id == id_a


# ── AC 9 ──────────────────────────────────────────────────────────────────────
async def test_runs_list_unknown_config_returns_empty(cd):
    """AC 9"""
    result = await cd.runs.list(config={"model": "never-saved"})
    assert result == []


# ── AC 10 ─────────────────────────────────────────────────────────────────────
async def test_runs_list_raises_on_conflicting_config_args(cd):
    """AC 10"""
    with pytest.raises(ValueError, match="mutually exclusive"):
        await cd.runs.list(config={"model": "x"}, config_id="abc12345")


# ── AC 11 ─────────────────────────────────────────────────────────────────────
async def test_runs_list_configs_ordered_by_created_at(cd):
    """AC 11"""
    config_a = {"model": "first"}
    config_b = {"model": "second"}
    id_a = _compute_config_id(config_a)
    id_b = _compute_config_id(config_b)

    await cd.runs.save([
        RunResult(strategy="S", config_id=id_a, source_id="doc1",
                  config=config_a, output=RunOutput(output="o", metadata={})),
    ])
    await cd.runs.save([
        RunResult(strategy="S", config_id=id_b, source_id="doc2",
                  config=config_b, output=RunOutput(output="o", metadata={})),
    ])
    configs = await cd.runs.list_configs()
    assert len(configs) == 2
    assert configs[0] == config_a
    assert configs[1] == config_b
    assert len(configs) == len({json.dumps(c, sort_keys=True) for c in configs})


# ── AC 12 ─────────────────────────────────────────────────────────────────────
async def test_evals_save_per_result_transaction_and_batch_error(cd, pool, project):
    """AC 12"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)

    good = EvalResult(
        run_result=RunResult(
            strategy="S", config_id=config_id, source_id="doc1",
            config=config, output=RunOutput(output="out", metadata={}),
        ),
        score=0.9,
    )
    # bad: r.config_id="deadbeef" won't exist in configs (computed hash is different)
    # → FK violation on run_results INSERT
    bad_run = RunResult(
        strategy="S",
        config_id="deadbeef",
        source_id="doc2",
        config={"model": "other"},
        output=RunOutput(output="out", metadata={}),
    )
    bad = EvalResult(run_result=bad_run, score=0.5)

    with pytest.raises(BatchSaveError) as exc_info:
        await cd.evals.save([good, bad], eval_function="test_scorer")

    assert len(exc_info.value.failures) == 1
    assert exc_info.value.failures[0][0] is bad

    async with pool.acquire() as conn:
        good_run = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project=$1 AND source_id='doc1'",
            project,
        )
        bad_run_count = await conn.fetchval(
            "SELECT COUNT(*) FROM run_results WHERE project=$1 AND source_id='doc2'",
            project,
        )
    assert good_run == 1
    assert bad_run_count == 0


# ── AC 13 ─────────────────────────────────────────────────────────────────────
async def test_runs_save_deduplicates_configs(cd, pool, project):
    """AC 13"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)
    results = [
        RunResult(
            strategy="OneShotSummarizer",
            config_id=config_id,
            source_id=f"doc{i}",
            config=config,
            output=RunOutput(output=f"summary {i}", metadata={}),
        )
        for i in range(3)
    ]
    await cd.runs.save(results)
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM configs WHERE config_id = $1", config_id
        )
    assert count == 1


# ── AC 14 ─────────────────────────────────────────────────────────────────────
async def test_sync_raises_when_loop_is_running():
    """AC 14"""
    with pytest.raises(RuntimeError, match="Use ConduitDatasetAsync"):
        ConduitDataset("test_project")


# ── AC 15 ─────────────────────────────────────────────────────────────────────
def test_sync_raises_after_close():
    """AC 15"""
    cd = ConduitDataset.__new__(ConduitDataset)
    cd._loop = asyncio.new_event_loop()
    cd._closed = False
    cd._async = ConduitDatasetAsync.__new__(ConduitDatasetAsync)
    cd._async.documents = DocumentsNamespace.__new__(DocumentsNamespace)

    def is_closed():
        return cd._closed

    cd.documents = _SyncProxy(cd._async.documents, cd._loop, is_closed)
    cd.close()

    with pytest.raises(RuntimeError, match="ConduitDataset is closed"):
        cd.documents.list()


# ── AC 16 ─────────────────────────────────────────────────────────────────────
def test_cli_status_exits_1_when_db_unreachable():
    """AC 16"""
    import os
    import subprocess

    env = os.environ.copy()
    env.pop("POSTGRES_PASSWORD", None)  # missing password → connection failure
    result = subprocess.run(
        [
            sys.executable,
            str(
                Path(__file__).parent.parent.parent
                / "src/conduit/apps/scripts/datasets_cli.py"
            ),
            "status",
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    assert result.returncode == 1
    output = result.stdout + result.stderr
    assert "Cannot reach postgres" in output
    assert "evals" in output


# ── AC 17 — documents ─────────────────────────────────────────────────────────
async def test_documents_save_is_idempotent(cd, pool, project):
    """AC 17 — documents"""
    item = RunInput(source_id="doc1", data="text")
    await cd.documents.save([item])
    await cd.documents.save([item])
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            "SELECT COUNT(*) FROM documents WHERE project = $1 AND source_id = $2",
            project,
            "doc1",
        )
    assert count == 1


# ── AC 17 — runs ──────────────────────────────────────────────────────────────
async def test_runs_save_is_idempotent(cd, pool, project):
    """AC 17 — runs"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)
    result = RunResult(
        strategy="OneShotSummarizer",
        config_id=config_id,
        source_id="doc1",
        config=config,
        output=RunOutput(output="summary", metadata={}),
    )
    await cd.runs.save([result])
    await cd.runs.save([result])
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM run_results
            WHERE project = $1 AND strategy = $2 AND config_id = $3 AND source_id = $4
            """,
            project,
            "OneShotSummarizer",
            config_id,
            "doc1",
        )
    assert count == 1


# ── AC 17 — evals ─────────────────────────────────────────────────────────────
async def test_evals_save_is_idempotent(cd, pool, project):
    """AC 17 — evals"""
    config = {"model": "gpt-oss:latest"}
    config_id = _compute_config_id(config)
    er = EvalResult(
        run_result=RunResult(
            strategy="S",
            config_id=config_id,
            source_id="doc1",
            config=config,
            output=RunOutput(output="out", metadata={}),
        ),
        score=0.8,
    )
    await cd.evals.save([er], eval_function="scorer")
    await cd.evals.save([er], eval_function="scorer")
    async with pool.acquire() as conn:
        count = await conn.fetchval(
            """
            SELECT COUNT(*) FROM eval_results
            WHERE project=$1 AND eval_function='scorer'
            """,
            project,
        )
    assert count == 1


# ── AC 18 ─────────────────────────────────────────────────────────────────────
def test_sync_context_manager_closes_on_exit():
    """AC 18 — normal exit"""
    loop = asyncio.new_event_loop()

    class _FakeAsync:
        documents = object()
        runs = object()
        evals = object()

    cd = object.__new__(ConduitDataset)
    cd._loop = loop
    cd._closed = False
    cd._async = _FakeAsync()
    cd.documents = _SyncProxy(cd._async.documents, loop, lambda: cd._closed)
    cd.runs = _SyncProxy(cd._async.runs, loop, lambda: cd._closed)
    cd.evals = _SyncProxy(cd._async.evals, loop, lambda: cd._closed)

    with cd:
        assert not cd._closed

    assert cd._closed
    assert loop.is_closed()


def test_sync_context_manager_closes_on_exception():
    """AC 18 — exception exit"""
    loop = asyncio.new_event_loop()

    class _FakeAsync:
        documents = object()
        runs = object()
        evals = object()

    cd = object.__new__(ConduitDataset)
    cd._loop = loop
    cd._closed = False
    cd._async = _FakeAsync()
    cd.documents = _SyncProxy(cd._async.documents, loop, lambda: cd._closed)
    cd.runs = _SyncProxy(cd._async.runs, loop, lambda: cd._closed)
    cd.evals = _SyncProxy(cd._async.evals, loop, lambda: cd._closed)

    with pytest.raises(ValueError):
        with cd:
            raise ValueError("body error")

    assert cd._closed
