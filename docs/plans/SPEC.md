# Summarization Strategies — Implementation Spec

You are implementing five new summarization strategies for the Conduit project.
Working directory is the project root. All source is under src/conduit/.

## Setup: Work in a Worktree

Before writing any code, create an isolated git worktree and do all work there:

```bash
git worktree add ~/.config/superpowers/worktrees/conduit-project/summarizers -b feat/summarizers
```

Then set your working directory to that worktree for all subsequent file operations.
Do not touch the main branch or the original working directory.

When implementation is complete and all checks pass, stop. Do not merge or push.
Leave the branch in the worktree for the user to review.

## Context

There are five existing, fully-implemented summarizers to use as reference:
  src/conduit/strategies/summarize/summarizers/one_shot.py
  src/conduit/strategies/summarize/summarizers/map_reduce.py
  src/conduit/strategies/summarize/summarizers/recursive.py
  src/conduit/strategies/summarize/summarizers/rolling_refine.py
  src/conduit/strategies/summarize/summarizers/extractive_pre_filter.py
  src/conduit/strategies/summarize/summarizers/hierarchical_tree.py

Read ALL of them before writing anything. They define the full contract:
- Base class: `SummarizationStrategy` from `conduit.strategies.summarize.strategy`
- Interface: `async def __call__(self, input: Any, config: dict) -> str`
- Decorators: `@step` and `@override` on `__call__`
- Context management: acquire/reset `context.config` and `context.use_defaults` in a try/finally
- Param access: `get_param("key", default=value)` — never read config dict directly
- Metadata: `add_metadata("key", value)` to record trace data
- Inner text type: `_TextInput` from `conduit.strategies.summarize.strategy`
- Embedding access: `HeadwaterAsyncClient` (see extractive_pre_filter.py for the exact pattern)
- Each file must include a runnable `if __name__ == "__main__":` block matching the style
  of the reference files

Also read:
  src/conduit/strategies/summarize/summarizers/chunker.py
  src/conduit/strategies/summarize/compression.py
  src/conduit/core/workflow/step.py
  src/conduit/core/workflow/context.py
  src/conduit/core/workflow/harness.py

There are stub files in `src/conduit/strategies/summarize/summarizers/future/` describing
each strategy's intent. IMPORTANT: those stubs use a stale import path
(`conduit.extensions.summarize.strategy`) and a stale interface (`def summarize(...)`).
Ignore their code. Use only their docstrings as a description of the intended behavior.

## Model

Use `gpt3` as the model for all LLM calls throughout development. This applies to all
`get_param("model", default=...)` defaults and all `__main__` example configs.
Do not use any other model.

## Strategies to Implement

Write each as a new file directly in `src/conduit/strategies/summarize/summarizers/`
(not in `future/`). Do not modify or delete the future/ stubs.

### 1. chain_of_density.py — ChainOfDensitySummarizer
Iterative densification. Workflow:
1. Generate an initial verbose summary (Draft 1) via OneShotSummarizer.
2. In a loop (default N=3 iterations):
   a. Prompt the model to identify 5-10 specific entities present in the source text
      but missing or underrepresented in the current draft.
   b. Prompt the model to fuse those entities into the draft without increasing word count.
3. Return the final densified summary.
Config params: model, iterations (default 3), max_tokens, temperature.
Track metadata: iterations_run, input_tokens_total, output_tokens_total.

### 2. cluster_select.py — ClusterSelectSummarizer
K-Means cluster-based chunk selection. Workflow:
1. Chunk the text via Chunker.
2. Embed all chunks via HeadwaterAsyncClient (same pattern as extractive_pre_filter.py).
3. Run K-Means clustering (`from sklearn.cluster import KMeans`) to find K semantic
   clusters. K = config param, default = min(10, total_chunks).
4. For each cluster, select the chunk whose embedding is closest to the cluster centroid
   (cosine similarity).
5. Pass the K selected chunks (in original document order) to OneShotSummarizer.
Config params: model, k (default min(10, total_chunks)), embedding_model.
Track metadata: num_chunks, k, chunks_selected.
sklearn is already in the project venv — do not add it to pyproject.toml.

### 3. atomic_proposition.py — AtomicPropositionSummarizer
Propositional decomposition. Workflow:
1. Chunk the text via Chunker.
2. For each chunk in parallel, prompt the model to decompose it into a list of
   independent, self-contained atomic statements. Each statement must dereference
   all pronouns (replace "he", "it", "the company" with the full named entity).
3. Collect all atomic propositions across all chunks.
4. Deduplicate by running a final LLM pass that removes near-duplicate propositions
   and returns the canonical set.
5. Return the propositions joined as a numbered list (one per line).
Config params: model, max_tokens, temperature, concurrency_limit (default 5).
Track metadata: num_chunks, total_propositions_before_dedupe, total_propositions_after_dedupe.
Note: return type is still str (a formatted list), consistent with the Strategy protocol.

### 4. map_dedupe_reduce.py — MapDedupeReduceSummarizer
Map-reduce with explicit deduplication pass. Workflow:
1. Chunk the text.
2. Map (parallel): run an extraction prompt on every chunk — default prompt extracts
   key facts, entities, and decisions. Honor a config param `extraction_prompt` override.
3. Collect all extracted lists from all chunks into one combined string.
4. Dedupe pass: prompt the model to merge duplicates, normalize format, and remove
   near-identical entries.
5. Final reduce: pass the deduplicated list to OneShotSummarizer for a coherent summary.
Config params: model, extraction_prompt (optional override), max_tokens, temperature,
concurrency_limit (default 5).
Track metadata: num_chunks, map_input_tokens, map_output_tokens, dedupe_input_tokens,
dedupe_output_tokens.

### 5. schema_extraction.py — SchemaExtractionSummarizer
Note: the stub has a typo ("schema_extration.py") — name the new file schema_extraction.py.
Structured extraction using a caller-supplied Pydantic schema. Workflow:
1. Accept a `schema` config param — a Pydantic BaseModel class (passed as the class
   itself, not an instance). If absent, raise ValueError with a clear message.
2. Chunk the text via Chunker.
3. For each chunk in parallel, call the LLM with a prompt instructing it to extract
   data matching the schema as JSON. Parse and validate each response via the schema.
   On parse failure, log a warning and skip that chunk (do not crash).
4. Merge all valid extracted instances: for list fields, concatenate; for scalar fields,
   prefer the first non-None value.
5. Return `json.dumps(merged.model_dump(), indent=2)` as the output string.
Config params: model, schema (required), max_tokens, temperature, concurrency_limit (default 5).
Track metadata: num_chunks, chunks_parsed_successfully, chunks_failed.

## Verification

After implementing all five, run:
```bash
python -m pytest tests/ -x -q 2>&1 | head -60
```

If tests pass, do a smoke-check on each strategy by running its `__main__` block:
```bash
python src/conduit/strategies/summarize/summarizers/chain_of_density.py
python src/conduit/strategies/summarize/summarizers/cluster_select.py
python src/conduit/strategies/summarize/summarizers/atomic_proposition.py
python src/conduit/strategies/summarize/summarizers/map_dedupe_reduce.py
python src/conduit/strategies/summarize/summarizers/schema_extraction.py
```

These require a running model (gpt3 via Ollama) and Headwater. If either is unavailable,
note it and stop — do not attempt workarounds or retry loops.

Do not modify any existing files outside of the new strategy files.
Do not add dependencies to pyproject.toml.
