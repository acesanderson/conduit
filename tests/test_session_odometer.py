from conduit.odometer.ConversationOdometer import ConversationOdometer
from conduit.message.messagestore import MessageStore
from conduit.message.textmessage import TextMessage
from pathlib import Path

SYSTEM_MESSAGE = "You are a helpful assistant that helps people find information."
HISTORY_FILE = Path(__file__).parent / "test_history.jsonl"

ms = MessageStore(history_file=HISTORY_FILE)
ms.ensure_system_message(SYSTEM_MESSAGE)
ms.append(TextMessage(role="user", content="Hello, how are you?"))
ms.append(
    TextMessage(
        role="assistant", content="I'm good, thank you! How can I assist you today?"
    )
)
ms.append(TextMessage(role="user", content="Can you tell me a joke?"))
ms.append(
    TextMessage(
        role="assistant",
        content="Sure! Why did the scarecrow win an award? Because he was outstanding in his field!",
    )
)
ms.append(TextMessage(role="user", content="That's funny! Tell me another one."))


print(ms.messages)
