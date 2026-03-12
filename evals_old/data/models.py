import hashlib
import json
from datetime import datetime
from uuid import UUID, uuid4
from pydantic import BaseModel, Field, model_validator


class EvalConfig(BaseModel):
    """
    The 'Recipe'. Represents a unique set of hyperparameters.
    """

    params: dict
    id: UUID = Field(default_factory=uuid4)
    checksum: str = Field(default="")
    created_at: datetime = Field(default_factory=datetime.now)

    @model_validator(mode="after")
    def generate_checksum(self) -> "EvalConfig":
        # Canonicalize: sort keys and remove whitespace for a stable hash
        # This ensures prompt tweaks in config.json trigger a new config ID
        canonical = json.dumps(self.params, sort_keys=True, separators=(",", ":"))
        self.checksum = hashlib.sha256(canonical.encode()).hexdigest()
        return self


class DatasetItem(BaseModel):
    """
    Mirror of your GoldStandardDatum.
    Stored once to avoid duplicating 50 documents 1000x.
    """

    source_id: str  # The unique slug from your GoldStandardEntry
    category: str
    content: dict  # Stores the full GoldStandardDatum as JSONB
    created_at: datetime = Field(default_factory=datetime.now)


class RunGroup(BaseModel):
    """
    The 'Experiment'. Context for why you are running these evals.
    """

    id: UUID = Field(default_factory=uuid4)
    project_name: str
    name: str
    git_commit: str | None = None
    metadata: dict = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)


class EvalRun(BaseModel):
    """
    The 'Result'. Connects Config + DatasetItem + Group.
    """

    id: UUID = Field(default_factory=uuid4)
    config_id: UUID
    group_id: UUID
    source_id: str

    output_summary: str
    metrics: dict[str, float]  # L_fact, L_entity, L_total, etc.
    trace: dict  # Full execution log

    latency_ms: float | None = None
    created_at: datetime = Field(default_factory=datetime.now)
