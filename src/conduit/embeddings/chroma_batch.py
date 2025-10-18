from pydantic import BaseModel, Field
from typing import Any


class ChromaBatch(BaseModel):
    # Mandatory fields
    ids: list[str] = Field(
        ..., description="List of unique identifiers for each item in the batch."
    )
    documents: list[str] = Field(
        ..., description="List of documents or text associated with each item."
    )
    # Optional fields
    embeddings: list[list[float]] | None = Field(
        default=None, description="List of embeddings corresponding to each item."
    )
    metadatas: list[dict[str, Any]] | None = Field(
        default=None, description="List of metadata dictionaries for each item."
    )
