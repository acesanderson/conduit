# Summarization Eval Strategy

## Goal

Find the best summarization configuration (strategy × model × prompt) for each Siphon SourceType,
using Gemini3 one-shot summaries as gold standard. Eventually "publish" a named config per SourceType
that Siphon can route to at ingest time.

---

## Gold Standard

- **Source**: 200 Siphon documents across all SourceTypes, token range 503–181K
- **Reference summaries**: Gemini3 one-shot (treated as ceiling quality)
- **Location**: parquet in `evals/` loaded via `load_datasets.py`
- **Rationale**: Gemini3 with a 1M context window can one-shot anything in the corpus. Local models
  with smaller context windows need strategies to handle long documents — the eval measures how close
  they get to Gemini quality.

---

## Swappable Dimensions

| Dimension | Current | Notes |
|-----------|---------|-------|
| **Documents** | 200 Siphon docs | Swap via `load_golden_dataset()` |
| **Models** | command-r, gpt-oss, qwen3.6, gemma4 | `MODELS` list in `run.py` |
| **Strategy** | `RecursiveSummarizer` | Any `SummarizationStrategy` subclass |
| **Params** | Per-strategy `Config` (Pydantic) | Pass as config dict; schema validated per strategy |
| **Eval function** | Gemini3 LLM-as-judge (0–1 score) | Any `async (RunResult) -> float` callable |

---

## Strategies Implemented

| Strategy | Approach | Prompt influence | Best for |
|----------|----------|-----------------|----------|
| `OneShotSummarizer` | Single LLM call | High | Docs that fit in context |
| `RecursiveSummarizer` | One-shot if fits, else MapReduce + recurse | High | General fallback (current default) |
| `RollingRefine` | Chunk linearly, iteratively refine running summary | High — prompt fires every chunk | Narrative / linear content |
| `MapReduce` | Summarize chunks in parallel, combine | High | General long-form |
| `MapDedupeReduce` | Map + explicit deduplication pass + reduce | High | Repetitive / legal content |
| `HierarchicalTree` | RAPTOR-style bottom-up tree | Medium — algorithm controls structure | Very long docs |
| `ChainOfDensity` | One-shot then iteratively pack in missing entities | Medium — entity focus is prompt-driven | Dense factual content |
| `ClusterSelect` | K-means on embeddings, pick centroids, one-shot | Low at selection, high at final | Redundant / noisy docs |
| `ExtractivePreFilter` | Keep top 30% chunks by centroid similarity | Low at selection, high at final | Info-sparse docs (YouTube) |
| `AtomicProposition` | Decompose to atomic facts, dedupe, list | Low | Fact extraction, not summaries |
| `SchemaExtraction` | Extract structured Pydantic objects from chunks | Low | Structured data extraction |

**Gut prediction for single best strategy across mixed corpus**: `RollingRefine` — produces the most
coherent output because each chunk is contextualized against the evolving summary, avoiding the
concatenation problem of parallel map strategies.

---

## Prompt Influence by Strategy

Prompt differentiation per SourceType is a first-class goal (YouTube summaries should feel different
from academic paper summaries). Strategies vary in how much the prompt actually propagates:

- **Full influence**: `OneShot`, `RollingRefine`, `MapReduce` — the summarization prompt fires on
  every meaningful LLM call. Per-SourceType prompts propagate throughout.
- **Partial influence**: `ChainOfDensity`, `HierarchicalTree` — algorithm enforces structure;
  prompt shapes focus but not form.
- **Selection only**: `ClusterSelect`, `ExtractivePreFilter` — embedding-based selection has no
  prompt; influence is limited to the final one-shot.

**Implication**: if per-SourceType customization matters, prefer strategies in the "full influence"
category.

---

## Current Eval Run

**Run matrix**: 200 docs × 4 models × `RecursiveSummarizer` = 800 runs

Models and routing:
- `qwen3.6:latest` → deepwater (AlphaBlue, RTX 5090) only
- `gemma4:latest` → deepwater only
- `command-r:latest` → deepwater (133 docs) + bywater (67 docs)
- `gpt-oss:latest` → deepwater (133 docs) + bywater (67 docs)

Deepwater-only models run sequentially (one model completes before the next starts) to avoid Ollama
VRAM contention between the 14B and 27B models. Bywater starts in parallel as a background task.

**Scoring**: Gemini3 judge rates each output 0–1 against the reference summary.

**Persistence**: runs and scores saved to Postgres `evals` DB via `ConduitDatasetAsync`.

---

## Planned Runs

### Run 1 — Model baseline (current, overnight)
**Question**: Which model produces the best summaries with RecursiveSummarizer?
- Strategy: `RecursiveSummarizer`
- Models: `gpt-oss:latest` (bywater), `qwen3.6:latest` (deepwater), `gemma4:latest` (deepwater)
- Docs: 200
- Total runs: 600
- **Decision gate**: identify top 1–2 models; drop the rest for subsequent runs

### Run 2 — Strategy comparison
**Question**: Does a better strategy beat RecursiveSummarizer on the winning model(s)?
- Strategy: `RollingRefine`, `HierarchicalTree`, `MapDedupeReduce`
- Models: top 1–2 from Run 1
- Docs: 200
- **Decision gate**: pick winning strategy per SourceType category; check if one strategy
  dominates or if genre-specific routing is warranted

### Run 3 — Prompt tuning
**Question**: How much does a SourceType-specific prompt move the needle?
- Strategy: winner from Run 2
- Models: winner(s) from Run 1
- Variants: generic prompt vs. per-SourceType prompts (already exist in Siphon)
- **Decision gate**: if delta > 0.05 score points, prompt differentiation is worth the
  complexity; wire per-SourceType prompts into the registry

### Run 4 — Publish validation
**Question**: Does the final config generalize to unseen docs?
- Strategy + model: published config from Runs 1–3
- Docs: holdout set (not yet assembled — separate from the 200-doc eval set)
- **Decision gate**: if scores hold, ship to Siphon ingest

---

## Roadmap

### Phase 1 (current): Baseline
- [x] Gold standard dataset (200 docs, Gemini3 references)
- [x] Eval harness with Postgres persistence and CSV output
- [x] Run 1: `RecursiveSummarizer` × 3 models

### Phase 2: Strategy comparison
- [ ] Run 2: `RollingRefine`, `HierarchicalTree`, `MapDedupeReduce` on top model(s)
- [ ] Slice all results by SourceType/category to find per-genre winners
- [ ] Run 3: per-SourceType prompt tuning on winning strategy × model

### Phase 3: Publish
- [ ] Make `RecursiveSummarizer` accept injectable overflow strategy (currently hardwired to MapReduce)
- [ ] Define `SummarizationProfile(strategy, model, prompt)` abstraction
- [ ] Build `SUMMARIZER_REGISTRY` keyed by SourceType
- [ ] Assemble holdout set for Run 4
- [ ] Run 4: validate final config on holdout
- [ ] Wire Siphon ingest to look up profile by SourceType

---

## Infrastructure Notes

- `RecursiveSummarizer` overflow strategy is currently hardwired to `MapReduceSummarizer` — needs
  to accept an injectable strategy before Phase 3 is possible.
- All strategies support `use_remote=True` + `host_alias` for remote model routing.
- Context window sizes configured in `~/.config/conduit/ollama_context_sizes.json` (all 4 eval
  models set to 128K).
- `evals.py` uses `return_exceptions=True` in gather — individual failures are logged and skipped,
  the run continues.
