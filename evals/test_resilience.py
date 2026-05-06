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
        with patch("scorer._JUDGE_TIMEOUT", 0.0):
            with pytest.raises(asyncio.TimeoutError):
                await _call_with_retry(slow)


def test_run_matrix_has_timeout_s():
    """Every RUN_MATRIX entry must declare an explicit timeout_s."""
    from run2 import RUN_MATRIX

    for entry in RUN_MATRIX:
        name = entry["strategy_cls"].__name__
        assert "timeout_s" in entry, f"Missing timeout_s: {name}"
        assert isinstance(entry["timeout_s"], int), f"timeout_s must be int: {name}"
        assert entry["timeout_s"] > 0, f"timeout_s must be positive: {name}"


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
    assert len(save_calls) == 3
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
            docs, config, MagicMock(), mock_ds, timeout_s=0.05
        )

    assert len(results) == 1
    assert results[0].source_id == "fast"
    assert mock_ds.runs.save.call_count == 1
