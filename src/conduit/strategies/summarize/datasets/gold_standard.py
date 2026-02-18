from __future__ import annotations
from pydantic import BaseModel, Field, model_validator
from conduit.strategies.summarize.compression import get_target_summary_length


class GoldStandardEntry(BaseModel):
    category: str = Field(
        ..., description="The category of the text (e.g., GovReport, BillSum, WikiHow)"
    )
    source_id: str = Field(
        ..., description="A unique identifier for the source document"
    )
    text: str = Field(..., description="The original text to be summarized")
    token_count: int = Field(
        ..., description="The number of tokens in the original text"
    )
    # Note: this field is constructed post-init, using get_target_summary_length based on token_count
    expected_summary_length: int | None = Field(
        default=None,
        description="The target token length for the summary, based on the original text length",
    )

    # Construct expected summary length based on token count of the original text, post-init
    @model_validator(mode="after")
    def set_expected_summary_length(self) -> GoldStandardEntry:
        self.expected_summary_length = get_target_summary_length(self.token_count)
        return self


class GoldStandardSummary(BaseModel):
    """
    Standardized Ground Truth for high-fidelity summarization.
    Designed to facilitate automated recall and faithfulness scoring.
    """

    main_theme: str = Field(
        description="A single sentence defining the primary topic and scope of the document."
    )

    summary: str = Field(
        description="""
        A dense, coherent narrative summary. 
        Must maintain the logical progression of the source. 
        No meta-commentary (e.g., avoid 'The author says').
        """
    )

    key_facts: list[str] = Field(
        description="""
        A list of 10-15 Atomic Facts extracted from the text. 
        An Atomic Fact is a standalone statement of truth that 
        contains exactly one core piece of information.
        """,
        min_length=10,
        max_length=20,
    )

    logical_outline: list[str] = Field(
        description="""
        A high-level sequence of the document's progression. 
        Used to validate that the summary maintains correct structural flow.
        """
    )

    entity_list: list[str] = Field(
        description="A list of primary entities (People, Organizations, Specific Technologies, or Laws) mentioned."
    )


class GoldStandardSummaryWithMetadata(GoldStandardSummary):
    summary_length: int = Field(
        description="The token count of the summary, used for recall evaluation against target summary length."
    )
    summary_embeddings: list[float] = Field(
        description="A dense vector representation of the summary, used for semantic similarity and recall evaluation."
    )
    entity_list_embeddings: list[list[float]] = Field(
        description="A list of dense vector representations for each entity in the entity list."
    )


class GoldStandardDatum(BaseModel):
    entry: GoldStandardEntry
    summary: GoldStandardSummaryWithMetadata
