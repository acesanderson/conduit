# evals/tests/test_runner.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from runner import ServerCircuitBreaker, _classify_error, _config_id


def test_config_id_is_deterministic():
    cfg = {"model": "qwen3.6:latest", "use_remote": True, "host_alias": "deepwater"}
    assert _config_id(cfg) == _config_id(cfg)


def test_config_id_differs_for_different_configs():
    assert _config_id({"model": "qwen3.6:latest"}) != _config_id({"model": "gpt-oss:latest"})


def test_classify_timeout():
    assert _classify_error(asyncio.TimeoutError()) == "timeout"


def test_classify_network_error():
    assert _classify_error(Exception("NETWORK_ERROR: connection refused")) == "network_error"


def test_classify_context_overflow():
    assert _classify_error(Exception("INTERNAL_ERROR: empty response from model")) == "context_overflow"


def test_classify_context_overflow_num_ctx():
    assert _classify_error(Exception("INTERNAL_ERROR: num_ctx exceeded")) == "context_overflow"


def test_classify_internal_error():
    assert _classify_error(Exception("INTERNAL_ERROR: something else")) == "internal_error"


def test_classify_connection_error():
    assert _classify_error(ConnectionError("Cannot connect to server")) == "connection_error"


def test_classify_unknown():
    assert _classify_error(ValueError("some random error")) == "ValueError"


def test_circuit_breaker_opens_after_threshold():
    cb = ServerCircuitBreaker("testserver", threshold=3, cooldown_s=60.0)

    async def run():
        for _ in range(3):
            await cb.record_failure()
        assert cb._opened_at is not None

    asyncio.run(run())


def test_circuit_breaker_closes_on_success():
    cb = ServerCircuitBreaker("testserver", threshold=3, cooldown_s=60.0)

    async def run():
        for _ in range(3):
            await cb.record_failure()
        assert cb._opened_at is not None
        await cb.record_success()
        assert cb._opened_at is None

    asyncio.run(run())


def test_circuit_breaker_does_not_open_below_threshold():
    cb = ServerCircuitBreaker("testserver", threshold=3, cooldown_s=60.0)

    async def run():
        for _ in range(2):
            await cb.record_failure()
        assert cb._opened_at is None

    asyncio.run(run())


def test_circuit_breaker_resets_consecutive_on_success():
    cb = ServerCircuitBreaker("testserver", threshold=5, cooldown_s=60.0)

    async def run():
        for _ in range(2):
            await cb.record_failure()
        await cb.record_success()
        assert cb._consecutive == 0

    asyncio.run(run())
