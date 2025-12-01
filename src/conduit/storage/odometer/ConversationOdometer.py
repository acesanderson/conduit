from conduit.odometer.Odometer import Odometer
from pydantic import Field


class ConversationOdometer(Odometer):
    """
    Attaches to ModelStore on instance level. (self.conversation_odometer)
    Routed to by TokenEvent.host.
    """

    conversation_id: str = Field(
        ..., description="Unique identifier for the conversation."
    )
    # model_name: str = Field(
    #     ...,
    #     description="Name of the model used in this conversation. Assumes one model for entire MessageStore, since how else can you track context window?",
    # )
    # provider: str = Field(
    #     ...,
    #     description="Provider of the model used in this conversation (OpenAI, Anthropic, etc.).",
    # )
    # context_window: int = Field(
    #     ..., description="Context window size of the model used in this conversation."
    # )
    # start_time: datetime = Field(
    #     default_factory=datetime.now, description="Start time of the conversation."
    # )
    #
    # # Methods:
    # def context_utilization(self) -> float:
    #     # Calculate percentage of context window used
    #     pass
    #
    # def should_warn_context_limit(self) -> bool:
    #     # Check if approaching context limits
    #     pass
