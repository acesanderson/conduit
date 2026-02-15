## `evals/LOSS_FUNCTION_SPEC.md`

# Summarization Loss Function: Multi-Factor Recall ()

## 1. Objective

To quantify the delta between a locally generated summary and the Gemini 3 "Golden" reference. This loss function is the primary metric for identifying the "Golden Config" through hyperparameter optimization.

## 2. Mathematical Definition

The total loss is a weighted sum of four normalized components. We define success as the minimization of :

### Weights ()

The weights are tuned to prioritize **Recall** of information over stylistic alignment:

* ** (Fact Weight):** 0.40
* ** (Entity Weight):** 0.25
* ** (Semantic Weight):** 0.20
* ** (Penalty Weight):** 0.15

---

## 3. Component Breakdown

### A. Atomic Fact Recall ()

Measures the preservation of discrete truth units.

* **Source:** `summary.key_facts`
* **Method:** Natural Language Inference (NLI) for Entailment.
* **Calculation:** 


where  is the total number of golden facts.

### B. Entity Preservation ()

Ensures key actors, places, and technologies are maintained.

* **Source:** `summary.entity_list`
* **Method:** Fuzzy string matching (Levenshtein distance).
* **Calculation:** 


### C. Semantic Similarity ()

A high-level "vibe check" using vector space proximity.

* **Source:** `summary.summary` (Narrative)
* **Method:** Cosine similarity of sentence embeddings.
* **Calculation:** 


where  and  are the embedding vectors of the local and golden summaries.

### D. Length Penalty ()

Punishes deviation from the user-defined token target.

* **Calculation:** 


---

## 4. Evaluation Constraints

### Hard Fails (Score = 1.0)

The Harness will assign a maximum loss if any of the following triggers occur:

1. **JSON Malformation:** Failure to parse structured output (if requested).
2. **Meta-Commentary:** Inclusion of "AI-speak" (e.g., "In this summary...", "The document states...").
3. **Truncation:** If the summary ends mid-sentence due to context window limits.

---

## 5. Implementation Roadmap

The evaluation harness must execute these metrics in order of computational cost:

1. **Heuristics** (Length, Regex) -> *Fastest*
2. **Deterministic** (Entities)
3. **Semantic** (Embeddings)
4. **Inference-based** (Fact NLI) -> *Slowest*
---
