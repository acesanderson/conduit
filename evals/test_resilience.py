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
            with pytest.raises(Exception):
                await _call_with_retry(slow)
