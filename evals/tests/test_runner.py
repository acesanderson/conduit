# evals/tests/test_runner.py
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from runner import EvalRunner, ServerCircuitBreaker, _classify_error, _config_id


def test_setup_logging_attaches_file_handler_even_if_root_preconfigured(tmp_path):
    """Regression: 0-byte NAS logs.

    If anything (library import, prior call) attaches a handler to the root
    logger first, `logging.basicConfig` is a no-op — the FileHandler never
    fires, and the NAS log file ends up 0 bytes. _setup_logging must use
    force=True so it wins.
    """
    pre_existing = logging.StreamHandler()
    logging.getLogger().addHandler(pre_existing)
    try:
        log_path = tmp_path / "x.log"
        runner = EvalRunner.__new__(EvalRunner)
        runner._log_path = log_path
        runner._setup_logging()
        logging.getLogger("test_lognot0").info("hello world")

        for h in logging.getLogger().handlers:
            try:
                h.flush()
            except Exception:
                pass

        assert log_path.exists()
        assert log_path.stat().st_size > 0
        assert "hello world" in log_path.read_text()
    finally:
        logging.getLogger().removeHandler(pre_existing)


def _make_doc(source_id: str, token_count: int = 0):
    from evals import RunInput
    return RunInput(source_id=source_id, data="", metadata={"token_count": token_count})


def test_filter_docs_no_filters_returns_all():
    docs = [_make_doc("a"), _make_doc("b"), _make_doc("c")]
    assert EvalRunner._filter_docs({}, docs) == docs


def test_filter_docs_max_token_count_drops_oversized():
    docs = [_make_doc("a", 1000), _make_doc("b", 5000), _make_doc("c", 9000)]
    filtered = EvalRunner._filter_docs({"max_token_count": 4000}, docs)
    assert [d.source_id for d in filtered] == ["a"]


def test_filter_docs_predicate_only_keeps_matching():
    docs = [_make_doc("a"), _make_doc("b"), _make_doc("c"), _make_doc("d")]
    filtered = EvalRunner._filter_docs(
        {"doc_predicate": lambda d: d.source_id in {"a", "c"}}, docs,
    )
    assert [d.source_id for d in filtered] == ["a", "c"]


def test_filter_docs_predicate_is_deterministic_for_partitioning():
    docs = [_make_doc(s) for s in ["aa", "bb", "cc", "dd", "ee", "ff"]]
    even = lambda d: int(d.source_id, 36) % 2 == 0
    odd  = lambda d: int(d.source_id, 36) % 2 == 1
    e = [d.source_id for d in EvalRunner._filter_docs({"doc_predicate": even}, docs)]
    o = [d.source_id for d in EvalRunner._filter_docs({"doc_predicate": odd},  docs)]
    assert set(e).isdisjoint(set(o))
    assert set(e) | set(o) == {d.source_id for d in docs}


def test_filter_docs_combines_max_tokens_and_predicate():
    docs = [
        _make_doc("a", 1000), _make_doc("b", 5000),
        _make_doc("c", 1000), _make_doc("d", 9000),
    ]
    entry = {"max_token_count": 4000, "doc_predicate": lambda d: d.source_id in {"a", "b"}}
    filtered = EvalRunner._filter_docs(entry, docs)
    assert [d.source_id for d in filtered] == ["a"]


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
