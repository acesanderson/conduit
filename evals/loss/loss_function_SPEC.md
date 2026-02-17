## `evals/LOSS_FUNCTION_SPEC_V3_FINAL.md`

# Summarization Loss Function: Multi-Factor Recall ()

## 1. Objective

To quantify the delta between a generated summary and a Gemini 3 "Golden" reference. This loss function prioritizes **Information Recall** and **Logical Flow** while acting as a guardrail against hallucinations and length deviance.

## 2. Mathematical Definition

The total loss is the weighted sum of five independent, normalized components:

### Final Weight Schedule ()

| Component | Weight | Focus | Metric |
| --- | --- | --- | --- |
| ** (Facts)** | **0.40** | Content Accuracy | Bidirectional NLI ( Score) |
| ** (Entities)** | **0.20** | Key Actors | Fuzzy String Match + Context Check |
| ** (Flow)** | **0.20** | Logical Sequence | Monotonicity of Outline |
| ** (Semantic)** | **0.15** | Narrative "Vibe" | Cosine Distance (Vector Space) |
| ** (Length)** | **0.05** | Conciseness | Sublinear Tiered Penalty |

---

## 3. Component Breakdown

### A. Bidirectional Atomic Fact Recall ()

Measures the overlap of discrete truth units between the gold standard and the generation.

* **Recall ():** . (Is the info present?)
* **Precision ():** . (Is the summary faithful?)
* **Calculation:** 

### B. Entity Preservation ()

Ensures People, Organizations, and Technologies are correctly identified.

* **Logic:** Uses fuzzy matching (threshold > 0.9). To prevent false positives (e.g., "Washington State" vs "George Washington"), a match is only valid if the surrounding context window has a semantic similarity  to the source.

### C. Structural Monotonicity ()

Validates that the summary visits the `logical_outline` points in the correct order.

* **Scoring:** * **Coverage (70%):** Percentage of outline points found in the summary via NLI.
* **Order (30%):** Penalty applied if Point  appears before Point .



### D. Semantic Similarity ()

A high-level proximity check using sentence embeddings.

* **Method:** Cosine distance () between the global mean-pooled vector of the generated summary and the golden narrative.

### E. Sublinear Length Penalty ()

A gentle guardrail against extreme length deviance.

* **Target:** Defined by `get_target_summary_length(input_tokens)`.
* **Calculation:** . This ensures that being 10% off doesn't ruin an otherwise perfect factual score.

---

## 4. Operational Guardrails

### Hard Fails ()

The evaluator immediately returns maximum loss if:

1. **Meta-Commentary:** Contains phrases like "This summary covers..." or "The document states...".
2. **JSON Failure:** If structured output was requested and is unparseable.
3. **Truncation:** The text ends mid-sentence.

---

## 5. Technical Implementation Roadmap

1. **Phase 1: Heuristics (Regex/Len)** - Filter out "garbage" outputs immediately.
2. **Phase 2: Vectorization** - Embed the gold and gen texts once. Compute .
3. **Phase 3: Batched NLI** - Collate all pairs (Fact-Sentence, Sentence-Fact, Outline-Sentence) into a single tensor.
4. **Phase 4: Aggregation** - Weight the scores and log individual component values for debugging.

## Supporting context
### Gold Standard Model

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


class GoldStandardDatum(BaseModel):
    entry: GoldStandardEntry
    summary: GoldStandardSummary
```

### Compression Ratio Function

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
