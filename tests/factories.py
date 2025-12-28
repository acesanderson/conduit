import factory
from conduit.domain.conversation.conversation import Conversation
from conduit.domain.message.message import (
    SystemMessage,
    UserMessage,
    AssistantMessage,
    ToolMessage,
    ToolCall,
)
from conduit.domain.request.generation_params import GenerationParams
from conduit.domain.config.conduit_options import ConduitOptions


class SystemMessageFactory(factory.Factory):
    class Meta:
        model = SystemMessage

    content = "You are a helpful assistant."


class UserMessageFactory(factory.Factory):
    class Meta:
        model = UserMessage

    content = "Hello, world!"


class ToolCallFactory(factory.Factory):
    class Meta:
        model = ToolCall

    function_name = "get_weather"
    arguments = {"location": "San Francisco"}


class AssistantMessageFactory(factory.Factory):
    class Meta:
        model = AssistantMessage

    content = "I can help with that."
    tool_calls = factory.List([factory.SubFactory(ToolCallFactory)])


class ToolMessageFactory(factory.Factory):
    class Meta:
        model = ToolMessage

    tool_call_id = factory.LazyAttribute(lambda o: str(o.factory_parent.tool_calls[0].tool_call_id))
    content = '{"temperature": "72F"}'


class ConversationFactory(factory.Factory):
    class Meta:
        model = Conversation

    topic = "Test Conversation"
    messages = factory.List([])


class GenerationParamsFactory(factory.Factory):
    class Meta:
        model = GenerationParams

    model = "gpt-4"
    temperature = 0.7
    top_p = 1.0
    max_tokens = 100
    stream = False


class ConduitOptionsFactory(factory.Factory):
    class Meta:
        model = ConduitOptions

    project_name = "test-project"
    use_cache = False
    persistence_mode = "resume"