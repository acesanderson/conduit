# Summarization Loss Function: Multi-Factor Recall () - V4

## 1. Objective

To quantify the delta between a generated summary and a Gemini 3 "Golden" reference. This version utilizes **pre-computed embeddings** within the `GoldStandardSummary` to minimize runtime latency and ensure consistency across large-scale evaluations.

## 2. Mathematical Definition

The total loss is a weighted sum of five normalized components:

### Updated Weight Schedule

| Component | Weight () | Metric | Data Source |
| --- | --- | --- | --- |
| ** (Facts)** | **0.40** | Bidirectional NLI ( Score) | `key_facts` |
| ** (Entities)** | **0.20** | Cosine Similarity Matrix | `entity_list_embeddings` |
| ** (Flow)** | **0.20** | Kendall’s Tau / Inversion Count | `logical_outline` |
| ** (Semantic)** | **0.15** | Cosine Distance | `summary_embedding` |
| ** (Length)** | **0.05** | Sublinear Tiered Penalty | `token_count` |

---

## 3. Component Breakdown (Optimized)

### A. Semantic Similarity ()

Direct comparison between the pre-computed golden vector and the live-generated vector.

* **Calculation:** 
* **Model:** `google/embeddinggemma-300m` (bfloat16).

### B. Entity Preservation ()

Validates presence and context of key actors using a many-to-many similarity matrix.

* **Logic:** For each vector in `entity_list_embeddings`, find the max similarity in the generated text's entity space.
* **Verification:** A match is only valid if .

### C. Bidirectional Atomic Fact Recall ()

*Requires dynamic NLI Cross-Encoding.*

* **Recall:** Do the `key_facts` exist in the generation?
* **Precision:** Does the generation contain facts not supported by the gold summary?

### D. Structural Monotonicity ()

Validates the `logical_outline` sequence.

* **Scoring:** Uses NLI to locate outline points in the generation, then calculates the **Inversion Count** to penalize out-of-order information.

---

## 4. Operational Guardrails (Hard Fails)

The evaluator returns  (Max Loss) if:

1. **Meta-Commentary:** Presence of "This summary..." or "The author...".
2. **Truncation:** String ends without valid terminal punctuation.
3. **Embedding Mismatch:** If the generated vector dimensions do not match the `summary_embedding` dimensions (indicating a model version conflict).

---

## 5. Updated Implementation Roadmap

1. **Phase 1: Guardrails & Length:** Immediate filter based on regex and `get_target_summary_length`.
2. **Phase 2: Live Vectorization:** Generate  and local entity vectors for the *generated* text only.
3. **Phase 3: Static Alignment:** Compute  and  by comparing live vectors against the `GoldStandardSummary` embeddings.
4. **Phase 4: Batched NLI:** Run the Cross-Encoder for  and .
5. **Phase 5: Aggregation:** Final weighted scalar calculation.

### Gold Standard data model
```python
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

    # Embeddings
    summary_embedding: list[float] | None = Field(
        default=None,
        description="A dense vector representation of the summary, used for semantic similarity and recall evaluation.",
    )
    entity_list_embeddings: list[list[float]] | None = Field(
        default=None,
        description="A list of dense vector representations for each entity in the entity list.",
    )


class GoldStandardDatum(BaseModel):
    entry: GoldStandardEntry
    summary: GoldStandardSummary
```
### Compression function
```python
from dataclasses import dataclass


@dataclass
class CompressionTier:
    max_input: int
    ratio: float
    min_tokens: int
    max_tokens: int


# Define the formal compression mapping
# Tier A: Detailed (20% for very short, 10% for standard)
# Tier B: Narrative (5% for long technical docs)
# Tier C: Strategic (Fixed cap for massive datasets)
COMPRESSION_SCHEDULE = [
    CompressionTier(max_input=2000, ratio=0.15, min_tokens=300, max_tokens=400),
    CompressionTier(max_input=10000, ratio=0.10, min_tokens=400, max_tokens=1000),
    CompressionTier(max_input=40000, ratio=0.05, min_tokens=1000, max_tokens=2000),
]

GLOBAL_MAX_TOKENS = 2500


def get_target_summary_length(input_tokens: int) -> int:
    """
    Calculates the target summary length based on input token volume.
    Applies a tiered ratio with a hard saturation cap.
    """
    target = 0

    # Find the appropriate tier
    selected_tier = None
    for tier in COMPRESSION_SCHEDULE:
        if input_tokens <= tier.max_input:
            selected_tier = tier
            break

    if selected_tier:
        raw_target = int(input_tokens * selected_tier.ratio)
        # Clamp between tier min/max
        target = max(
            selected_tier.min_tokens, min(raw_target, selected_tier.max_tokens)
        )
    else:
        # Default for anything above 40k
        target = GLOBAL_MAX_TOKENS

    return target


if __name__ == "__main__":
    # Standalone test to visualize the map
    test_values = [500, 1500, 5000, 15000, 50000, 100000]
    print(f"{'Input Tokens':<15} | {'Target Tokens':<15} | {'Effective Ratio':<15}")
    print("-" * 50)
    for val in test_values:
        t = get_target_summary_length(val)
        ratio = (t / val) * 100
        print(f"{val:<15} | {t:<15} | {ratio:>5.1f}%")
```
