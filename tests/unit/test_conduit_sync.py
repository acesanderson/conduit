from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from conduit.core.conduit.conduit_sync import ConduitSync
from conduit.core.prompt.prompt import Prompt
from conduit.domain.conversation.conversation import Conversation


def make_conduit_sync() -> ConduitSync:
    """Construct a minimal ConduitSync with mocked params and options."""
    prompt = Prompt("hello")
    conduit = ConduitSync(prompt=prompt)
    conduit.params = MagicMock()
    conduit.options = MagicMock()
    return conduit


def test_pipe_sync_calls_impl_pipe_with_self_params_and_options():
    """AC6: pipe_sync delegates to _impl.pipe with self.params and self.options."""
    conduit = make_conduit_sync()
    conversation = Conversation()

    mock_result = MagicMock(spec=Conversation)
    conduit._impl.pipe = AsyncMock(return_value=mock_result)

    with patch.object(conduit, "_run_sync", side_effect=lambda coro: mock_result):
        result = conduit.pipe_sync(conversation)

    conduit._impl.pipe.assert_called_once_with(
        conversation, conduit.params, conduit.options
    )
    assert result is mock_result
