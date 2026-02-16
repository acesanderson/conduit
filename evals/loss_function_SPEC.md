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

