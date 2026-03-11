from __future__ import annotations
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from conduit.core.workflow.step import StepWrapper
from conduit.core.workflow.harness import ConduitHarness
from conduit.domain.message.message import AssistantMessage
from conduit.domain.conversation.conversation import Conversation
from conduit.strategies.document_edits.models import DocumentEdits, EditOp, EditType
from conduit.strategies.document_edits.strategy import DocumentEditStrategy


def _make_fake_conversation(parsed_obj: DocumentEdits) -> MagicMock:
    msg = AssistantMessage(parsed=parsed_obj)
    conv = MagicMock(spec=Conversation)
    conv.last = msg
    return conv


def test_strategy_call_is_wrapped_with_step():
    strategy = DocumentEditStrategy()
    assert isinstance(DocumentEditStrategy.__call__, StepWrapper)


async def test_strategy_builds_params_with_structured_response_output_type():
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    fake_edits = DocumentEdits(edits=[], summary="no changes")
    fake_conv = _make_fake_conversation(fake_edits)

    with patch("conduit.core.conduit.conduit_async.ConduitAsync.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = fake_conv
        await harness.run(strategy, document="hello world", user_prompt="do nothing")

    call_kwargs = mock_run.call_args.kwargs
    params = call_kwargs["params"]
    assert params.output_type == "structured_response"
    assert params.response_model is DocumentEdits


async def test_strategy_builds_options_with_include_history_false():
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    fake_edits = DocumentEdits(edits=[], summary="no changes")
    fake_conv = _make_fake_conversation(fake_edits)

    with patch("conduit.core.conduit.conduit_async.ConduitAsync.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = fake_conv
        await harness.run(strategy, document="hello world", user_prompt="do nothing")

    call_kwargs = mock_run.call_args.kwargs
    options = call_kwargs["options"]
    assert options.include_history is False


async def test_strategy_returns_document_with_edits_applied():
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    fake_edits = DocumentEdits(
        edits=[EditOp(type=EditType.replace, search="hello", replace="goodbye")],
        summary="replaced greeting",
    )
    fake_conv = _make_fake_conversation(fake_edits)

    with patch("conduit.core.conduit.conduit_async.ConduitAsync.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = fake_conv
        result = await harness.run(
            strategy, document="hello world", user_prompt="replace greeting"
        )

    assert result == "goodbye world"


async def test_strategy_raises_type_error_when_parsed_is_not_document_edits():
    strategy = DocumentEditStrategy()
    harness = ConduitHarness(use_defaults=True)

    bad_msg = AssistantMessage(content="not json")
    bad_conv = MagicMock(spec=Conversation)
    bad_conv.last = bad_msg

    with patch("conduit.core.conduit.conduit_async.ConduitAsync.run", new_callable=AsyncMock) as mock_run:
        mock_run.return_value = bad_conv
        with pytest.raises(TypeError, match="Expected DocumentEdits"):
            await harness.run(
                strategy, document="hello world", user_prompt="do something"
            )
