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
