# Summarization Loss Function: The "Tri-Gate" Architecture

## 1. Design Philosophy

We reject the notion that we must manually engineer features (extracting facts, listing entities) to evaluate quality. Instead, we use a **Cascading Gate** approach.

We treat the evaluation as a funnel. A summary must pass cheap, mechanical checks before we spend compute on expensive, cognitive checks.

## 2. The Three Tiers

### Tier 1: Syntax (The "Hard" Gate)

**Objective:** Instant rejection of failures that don't meet the basic contract.

* **Metric:** `Length Adherence` & `Hallucination Format`
* **Cost:** Free (0ms, no API calls)
* **Logic:**
1. **Length:** Did the model produce  the requested token count? (e.g., within ).
2. **Format:** Did the model output "Here is your summary:" or other chatty conversational filler? (Regex check).


* **Action:** If failed, **Score = 0**. Stop immediately.

### Tier 2: Semantics (The "Drift" Gate)

**Objective:** Cheap rejection of summaries that are on the wrong topic entirely.

* **Metric:** `Cosine Similarity`
* **Cost:** Low (1 embedding call ~ $0.0001)
* **Logic:**
* Compare  vs .
* If similarity  (adjustable threshold), the model has hallucinated a different document or lost the plot.


* **Action:** If failed, **Score = Similarity**. Stop immediately.

### Tier 3: Cognition (The "Judge" Gate)

**Objective:** High-fidelity assessment of *faithfulness* and *completeness*.

* **Metric:** `LLM-as-a-Judge Score` (1-5 Scale)
* **Cost:** High (LLM Inference)
* **Logic:**
* We pass the **Gold Standard Text** and the **Generated Text** to a superior model (e.g., Gemini 1.5 Pro).
* The Judge compares them directly for:
* **Factual Recall:** Did we capture the core truths?
* **Entity Precision:** Did we name the right people/orgs?
* **Flow:** Is the narrative structure preserved?





---

## 3. Simplified Data Model

We delete the `entity_list`, `key_facts`, and `logical_outline`. The **Gold Standard** is now just the perfect summary and its mathematical signature.

```python
from pydantic import BaseModel, Field

class GoldStandardSummary(BaseModel):
    """
    The Single Source of Truth.
    """
    
    # 1. The Reference Text (For Tier 3 Judge)
    summary_text: str = Field(
        ..., 
        description="The ideal summary written by a human or superior model."
    )

    # 2. The Semantic Signature (For Tier 2 Gate)
    summary_embedding: list[float] = Field(
        ...,
        description="Pre-computed vector of the summary_text."
    )

    # 3. The Constraint (For Tier 1 Gate)
    target_token_count: int = Field(
        ...,
        description="The length of the gold summary (used to set the +/- 20% bounds)."
    )

```

---

## 4. The Judge's Rubric (Tier 3)

This prompt replaces your 200 lines of Python code. It effectively instructs the Judge to perform the "Feature Extraction" and "Comparison" in real-time.

**System Prompt:**

> You are an expert Editor-in-Chief. You will be given a [Gold Standard Summary] (the ground truth) and a [Candidate Summary] (written by a junior writer).
> Your goal is to score the Candidate on a scale of 1-5 based heavily on **Information Recall**.
> **Evaluation Criteria:**
> 1. **Facts & Entities:** Does the Candidate mention the same key actors, numbers, and events as the Gold Standard? (It doesn't need to match word-for-word, but the *data* must be there).
> 2. **No Hallucinations:** Does the Candidate include claims *not* supported by the Gold Standard? (Penalize heavily).
> 3. **Structure:** Does the Candidate follow the logical progression of the Gold Standard?
> 
> 
> **Ignore Style:** If the Gold Standard is "dry" and the Candidate is "lively," do NOT penalize, provided the facts are identical.
> **Output JSON:** `{ "score": int, "missing_facts": list[str], "reasoning": str }`

---

## 5. Scoring Formula (The "Loss")

The final loss is simply the **inverse** of the accumulated quality.

* : 1.0 if within bounds, 0.0 if outside.
* : The raw cosine similarity (0.0 to 1.0).
* : The normalized LLM score (Score / 5.0).

**Why this weighting?**

* **50%** of the score comes from the Judge (the smartest part of the system).
* **30%** comes from Embeddings (ensures we don't drift too far in vocabulary).
* **20%** ensures we respect the length constraint (critical for your use case).

### Appendix 1: Compression function

```python
from dataclasses import dataclass


@dataclass
class CompressionTier:
    max_input: int
    ratio: float
    min_tokens: int
    max_tokens: int
    # tolerance is (lower_bound_multiplier, upper_bound_multiplier)
    tolerance: tuple[float, float] = (0.75, 1.25)

    def is_valid_summary(
        self, target: int, summary_tokens: int, original_tokens: int
    ) -> bool:
        """
        Checks if summary tokens fall within the tier-specific tolerance.
        Adjusts the lower bound for documents shorter than the target floor.
        """
        low_m, high_m = self.tolerance

        # Avoid the 'floor trap': if original text < target floor,
        # use original text as the baseline for the lower bound calculation.
        lower_baseline = min(target, original_tokens)

        lower_bound = lower_baseline * low_m
        upper_bound = target * high_m

        return lower_bound <= summary_tokens <= upper_bound


# Define the formal compression mapping
COMPRESSION_SCHEDULE = [
    # Tier A: Detailed (Small docs)
    CompressionTier(2000, 0.15, 300, 400, (0.75, 1.25)),
    # Tier B: Narrative (Standard docs)
    CompressionTier(10000, 0.10, 400, 1000, (0.70, 1.20)),
    # Tier C: Strategic (Large docs)
    CompressionTier(40000, 0.05, 1000, 2000, (0.65, 1.20)),
]

GLOBAL_MAX_TOKENS = 2500
GLOBAL_TOLERANCE = (0.50, 1.20)


def get_target_summary_length(input_tokens: int) -> int:
    """
    Calculates the target summary length based on input token volume.
    Applies a tiered ratio with a hard saturation cap.
    """
    selected_tier = None
    for tier in COMPRESSION_SCHEDULE:
        if input_tokens <= tier.max_input:
            selected_tier = tier
            break

    if selected_tier:
        raw_target = int(input_tokens * selected_tier.ratio)
        return max(selected_tier.min_tokens, min(raw_target, selected_tier.max_tokens))

    return GLOBAL_MAX_TOKENS


def is_within_threshold(original_tokens: int, summary_tokens: int) -> bool:
    """
    Validates if summary length is acceptable relative to the compression target.
    """
    target = get_target_summary_length(original_tokens)

    selected_tier = next(
        (t for t in COMPRESSION_SCHEDULE if original_tokens <= t.max_input), None
    )

    if selected_tier:
        return selected_tier.is_valid_summary(target, summary_tokens, original_tokens)

    # Global fallback for very large docs
    low_m, high_m = GLOBAL_TOLERANCE
    # Apply same floor-trap logic for consistency, though rare at global scale
    lower_bound = min(target, original_tokens) * low_m
    upper_bound = target * high_m

    return lower_bound <= summary_tokens <= upper_bound


if __name__ == "__main__":
    # Test visualization
    # Case 50: Original 101, Summary 187, Target 300.
    # Old logic failed because 187 < (300 * 0.75 = 225).
    # New logic passes because 187 > (101 * 0.75 = 75).
    orig_50, summ_50 = 101, 187
    print(f"Datum 50 Pass: {is_within_threshold(orig_50, summ_50)}")
```

### Appendix 2: Gold Standard dataset model

```python
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
```
