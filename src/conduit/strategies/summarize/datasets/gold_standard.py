from pydantic import BaseModel, Field


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


class GoldStandardDatum(BaseModel):
    entry: GoldStandardEntry
    summary: GoldStandardSummary
