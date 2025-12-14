from enum import Enum


class PersistenceMode(Enum):
    # Always create new conversation; discard existing
    OVERWRITE = "overwrite"

    # Load conversation if exists, else create new; start from last message
    RESUME = "resume"

    # Resume but also: generate a conversation title if missing
    CHAT = "chat"
