from unittest.mock import patch, AsyncMock

import pytest

from tests.factories import ConversationFactory, UserMessageFactory, AssistantMessageFactory
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.request.generation_params import GenerationParams
from conduit.apps.chat.app import ChatApp
from conduit.apps.chat.engine.async_engine import ChatEngine
from conduit.apps.chat.ui.async_input import AsyncInput
from conduit.core.model.models.modelstore import ModelStore


def test_conversation_factory_creates_a_conversation():
    """
    Tests that the ConversationFactory can create a valid Conversation object.
    """
    conversation = ConversationFactory()
    assert conversation is not None
    assert conversation.messages == []


@pytest.mark.asyncio
async def test_async_input_retrieves_user_text():
    """
    Tests that the AsyncInput class can retrieve user input asynchronously.
    """
    with patch("prompt_toolkit.shortcuts.PromptSession.prompt_async", return_value="Hello, world!") as mock_prompt:
        input_handler = AsyncInput()
        user_input = await input_handler.get_input()
        assert user_input == "Hello, world!"
        mock_prompt.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_app_handles_natural_language_query():
    """
    Tests that the ChatApp correctly handles a natural language query.
    """
    # 1. Arrange
    mock_input = AsyncMock()
    mock_input.get_input.return_value = "Hello"

    mock_engine = AsyncMock(spec=ChatEngine) # Mock the entire engine

    app = ChatApp(engine=mock_engine, input_interface=mock_input)

    # 2. Act
    await app.run_once()

    # 3. Assert
    mock_engine.handle_query.assert_awaited_once_with("Hello")


@pytest.mark.asyncio
async def test_chat_app_handles_wipe_command():
    """
    Tests that the ChatApp correctly handles the /wipe command.
    """
    # 1. Arrange
    mock_input = AsyncMock()
    mock_input.get_input.return_value = "/wipe"

    engine = ChatEngine()
    engine.conversation = ConversationFactory(messages=[UserMessageFactory()])
    
    app = ChatApp(engine=engine, input_interface=mock_input)

    # 2. Act
    await app.run_once()

    # 3. Assert
    assert app.engine.conversation.messages == []


@pytest.mark.asyncio
async def test_chat_app_exits_on_exit_command():
    """
    Tests that the ChatApp exits on the /exit command.
    """
    # 1. Arrange
    mock_input = AsyncMock()
    mock_input.get_input.return_value = "/exit"

    engine = ChatEngine()
    
    app = ChatApp(engine=engine, input_interface=mock_input)

    # 2. Act
    await app.run_once()

    # 3. Assert
    assert not app.is_running


@pytest.mark.asyncio
async def test_chat_app_handles_help_command():
    """
    Tests that the ChatApp correctly handles the /help command.
    """
    # 1. Arrange
    mock_input = AsyncMock()
    mock_input.get_input.return_value = "/help"

    engine = ChatEngine()
    
    app = ChatApp(engine=engine, input_interface=mock_input)

    # 2. Act
    with patch("conduit.apps.chat.app.display") as mock_display:
        await app.run_once()

        # 3. Assert
        mock_display.assert_called_once_with("Available commands:\n/help\n/wipe\n/exit\n/history\n/models\n/model\n/set model <model_name>")


@pytest.mark.asyncio
async def test_chat_app_handles_history_command():
    """
    Tests that the ChatApp correctly handles the /history command.
    """
    # 1. Arrange
    mock_input = AsyncMock()
    mock_input.get_input.return_value = "/history"

    engine = ChatEngine()
    engine.conversation = ConversationFactory(messages=[UserMessageFactory(), AssistantMessageFactory()])

    app = ChatApp(engine=engine, input_interface=mock_input)

    # 2. Act
    with patch("conduit.domain.conversation.conversation.Conversation.print_history") as mock_print_history:
        await app.run_once()

        # 3. Assert
        mock_print_history.assert_called_once()











@pytest.mark.asyncio





async def test_chat_app_handles_set_model_command():





    """





    Tests that the ChatApp correctly handles the /set model command.





    """





    # 1. Arrange





    mock_input = AsyncMock()





    mock_input.get_input.return_value = "/set model new-test-model"











    initial_model = "old-test-model"





    engine = ChatEngine(params=GenerationParams(model=initial_model))





    





    app = ChatApp(engine=engine, input_interface=mock_input)











    # 2. Act





    await app.run_once()











    # 3. Assert





    assert app.engine.params.model == "new-test-model"







