from pydantic import BaseModel, Field
from datetime import datetime
from uuid import uuid4


class GeneratedSummary(BaseModel):
    """
    A data model representing a generated summary along with its metadata.
    Does not include the original input text; this object is paired with its source entry within a GoldStandardDatum object.
    """

    # Required fields
    summary: str
    token_count: int

    # Model/Engine Metadata
    config_dict: dict = Field(
        default_factory=dict,
        description="The run params, may include: model, temperature, chunksize, etc. This is the config dict you passed to your harness.",
    )
    trace: dict = Field(
        default_factory=dict,
        description="The full dictionary from your harness for deep debugging, may include: token usage, latency, etc.",
    )

    # Computed Metadata
    trace_id: str = Field(
        default_factory=lambda: str(uuid4()),
        description="A unique identifier for this trace, useful for debugging and correlation.",
    )
    generated_at: datetime = Field(
        default_factory=datetime.now,
        description="The timestamp when the summary was generated.",
    )
