# Summarization Eval Strategy

## Goal

Find the best summarization configuration (strategy × model × prompt) for each Siphon SourceType,
using Gemini3 one-shot summaries as gold standard. Eventually "publish" a named config per SourceType
that Siphon can route to at ingest time.

---

## Status — 2026-06-01

**Phases 1 and 2 complete. Phase 3 substantially complete on the conduit side.** ArticleEnricher wired in Siphon as proof-of-concept and live-validated. Siphon-side retrieval/embedding design now lives in `siphon-server/dev/retrieval.md`; Siphon enrichment patterns live in `siphon-server/dev/summarization.md`.

- **qwen3.6 dropped from production routing** — quality strong but latency unworkable for online ingest (RollingRefine medians 8–28 min on long docs).
- **Production routing decided** (see "Published Routing Decision" below). Held as data in `PRODUCTION_ROUTING` (commit `b3f3157`), swappable post-rerun with a one-line edit.
- **RoutingSummarizer + SummarizationProfile shipped** (commit `b3f3157`). Token-count routing via tiktoken `cl100k_base`. The router is itself a `SummarizationStrategy`, so it plugs into the eval matrix as one row and Siphon calls it through the same surface as any concrete strategy.
- **Per-call guideline plumbing shipped** as `_TextInput.guideline`. Configs stay the published-artifact surface; guidelines are out-of-band per-call directives. OneShot applies guideline inline; RollingRefine applies it only at a post-loop format pass (intermediate refinement stays guideline-free to avoid premature format-locking).
- **ArticleEnricher shipped Siphon-side** (commit `040cfe3`, deployed 2026-06-01). Tier1 routed to gpt-oss/bywater, structured markdown output matched the guideline.
- **NAS-backed artifact storage is standard** (commit `0c03c08`). Results, status, logs go to `$NAS/evals/<project>/<eval>/`. See `evals/nas.py` and `evals/ARCHITECTURE.md`.

**Rerun status**: Cronicle event `emothl43a01` has been timing out at the 4h Cronicle timeout for the last several nightly attempts. Lower-priority since the strategy doc treats the Tier 3 swap as a one-line change post-rerun and Phase 3 was unblocked on it. Independently worth fixing: bump or remove the 4h timeout, and fix the 0-byte NAS log issue.

**Next concrete steps**:
1. Reshape description generation per `siphon-server/dev/retrieval.md`: description becomes a HyDE-shaped retrieval artifact generated as a one-shot pass on top of the summary, not from raw text. Description guideline lives Siphon-side at `sources/<source>/description_guideline.jinja2`.
2. Roll RoutingSummarizer out to the other Siphon enrichers. Of the 11 non-article sources, 9 follow the standard pattern (arxiv, audio, doc, email, github, image, obsidian, video, youtube; doc is multi-variant by MIME type). `drive` is a `NotImplementedError` stub; `podcasts` has no `enricher.py` — both skipped pending the upstream work that would make them enrichable.
3. Resolve the rerun timeout so Tier 3 can settle. Either cut the matrix to Tier 3 candidates only, or remove the Cronicle timeout. Also fix the 0-byte NAS logs so future failed runs are debuggable.

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
| **Models** | gpt-oss, gemma4 (qwen3.6 dropped — latency) | Configured per-entry in `jobs/<eval>.py` `RUN_MATRIX` |
| **Strategy** | `OneShotSummarizer`, `RollingRefineSummarizer` (production); others in eval rotation | Any `SummarizationStrategy` subclass |
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

## Published Routing Decision

Token breakpoints based on Effective Context Window (ECW) cliffs from the OneShot quality-by-bin sweep. Routing held as data; swappable post-rerun.

| Tier | Token range | Strategy | Model | Host | Quality (mean) | Notes |
|------|-------------|----------|-------|------|----------------|-------|
| 1 | `<12K`     | OneShot       | `gpt-oss:latest` | bywater   | 0.72 at <5K, 0.13 at 5K-12K | gpt-oss ECW cliff at ~5K; viable for short docs only |
| 2 | `12K–30K`  | OneShot       | `gemma4:latest`  | deepwater | 0.60 across range | gemma4 ECW cliff at ~30K |
| 3 | `≥30K`     | RollingRefine | `gemma4:latest`  | deepwater | 0.51–0.62 | Subject to revision pending rerun (hybrid candidate) |

**Models considered and dropped**:
- `qwen3.6:latest` — best quality at every bin but RollingRefine median 8–28 min on long docs. Out of latency budget. Available for overnight batch jobs, not online routing.
- `command-r:latest` — early eval; dropped after Run 1.
- `MapDedupeReduceSummarizer` — measured no quality or speed niche vs. RollingRefine at any bin. No intermediate tier.
- `HierarchicalTreeSummarizer` — fast at 12K-30K (9s, 0.568) but already in OneShot's range; collapses past 30K.

---

## Current Eval Run

**Focused rerun in flight** (Cronicle event `emothl43a01`, deploys from commit `0c03c08`).

Matrix (5 entries, all `use_cache=False` for clean duration capture):

| Tier | Strategy | Model | Server | Max tokens |
|------|----------|-------|--------|------------|
| 1 | OneShot          | gpt-oss          | bywater   | 12K |
| 2 | OneShot          | gemma4           | deepwater | 60K |
| 3 baseline | RollingRefine | gemma4 | deepwater | none |
| 3 candidate A | MapDedupeReduceHybrid | gpt-oss chunks + gemma4 reduce | cross-host | none |
| 3 candidate B | MapReduce (plain) | gemma4 | deepwater | none |

**Question this rerun answers**: does the cross-host hybrid beat RollingRefine on wall-clock at acceptable quality for Tier 3? Plain MapReduce isolates whether MDR's dedupe pass is the cost driver.

**Why a rerun was needed**:
- Run 2's duration capture used `trace[-1]["duration"]` (last LLM call only) — understated chunked-strategy wall time. Fixed in commit `b566de6` (now sums over all trace entries).
- ECW sweep CSV had no duration column at all — Tier 1/2 timing was unmeasured.
- `MapDedupeReduceHybrid` was in the matrix as of commit `e0e508d` but never executed; zero rows in DB.

**Scoring**: Gemini3 judge rates each output 0–1 against the reference summary.

**Persistence**: runs + scores saved to Postgres `evals` DB via `ConduitDatasetAsync`. Per-run CSV / status / log artifacts go to `$NAS/evals/conduit/run2/` per the NAS convention.

---

## Planned Runs

### Run 1 — Model baseline — DONE (2026-05)
**Question**: Which model produces the best summaries with RecursiveSummarizer?
**Outcome**: qwen3.6 highest quality, gemma4 second, gpt-oss collapses past 5K tokens, command-r dropped. Quality table from OneShot ECW sweep:

| model | <5K | 5K-12K | 12K-30K | 30K-60K | 60K-100K |
|-------|-----|--------|---------|---------|----------|
| qwen3.6 | 0.80 | 0.75 | 0.76 | 0.27 | 0.22 |
| gemma4  | 0.66 | 0.60 | 0.60 | 0.07 | 0.04 |
| gpt-oss | 0.72 | 0.13 | 0.09 | 0.07 | 0.08 |

### Run 2 — Strategy comparison — DONE (2026-05)
**Question**: Does a better strategy beat RecursiveSummarizer on the winning model(s)?
**Outcome**: RollingRefine wins for multi-chunk docs (>12K) on both qwen3.6 and gemma4. MapDedupeReduce doesn't earn an intermediate tier (slower or tied + lower quality at every bin). HierarchicalTree collapses past 30K. Strategy × model mean scores (multi-chunk docs only):

| strategy + model | quality |
|------------------|---------|
| RollingRefine + qwen3.6 | 0.694 |
| RollingRefine + gemma4 | 0.560 |
| MapDedupeReduce + qwen3.6 | 0.538 |
| MapDedupeReduce + gemma4 | 0.429 |
| (all gpt-oss strategies) | 0.07–0.15 |

### Run 2b — Focused rerun — IN FLIGHT (2026-05-31)
See "Current Eval Run" above. Settles hybrid vs RollingRefine for Tier 3 + captures missing Tier 1/2 durations.

### Run 3 — Prompt / guideline tuning — DEFERRED
**Question**: How much does a SourceType-specific guideline move the needle?
- Strategy: winner per tier (currently OneShot + RollingRefine)
- Models: per Published Routing Decision
- Variants: generic prompt vs. per-SourceType guidelines injected via `SummarizationProfile.guideline`
- **Decision gate**: if delta > 0.05 score points, guideline differentiation is worth the complexity; wire per-SourceType guidelines into the Siphon-side `EnricherStrategy`
- **Prerequisite**: scaffold the guideline injection point in conduit's `RoutingSummarizer` (Phase 3 next-session task)

### Run 4 — Publish validation — DEFERRED
**Question**: Does the final config generalize to unseen docs?
- Strategy + model: published config from Runs 1–3
- Docs: holdout set (not yet assembled — separate from the 200-doc eval set)
- **Decision gate**: if scores hold, ship to Siphon ingest

---

## Roadmap

### Phase 1 — Baseline — DONE
- [x] Gold standard dataset (200 docs, Gemini3 references)
- [x] Eval harness with Postgres persistence and CSV output
- [x] Run 1: model baseline across `gpt-oss`, `qwen3.6`, `gemma4`, `command-r`

### Phase 2 — Strategy comparison — DONE
- [x] Run 2: `RollingRefine`, `HierarchicalTree`, `MapDedupeReduce` on gemma4 + qwen3.6
- [x] Strategy × model breakdown by token bin (per-tier quality cliff identified)
- [x] Routing decision: OneShot ≤ 30K / RollingRefine > 30K, gemma4 production model

### Phase 3 — Publish — SUBSTANTIALLY COMPLETE (conduit side)

**Done**:
- [x] Production routing decided (see "Published Routing Decision")
- [x] NAS-backed artifact storage (`evals/nas.py`, commit `0c03c08`)
- [x] Focused rerun matrix to settle hybrid vs RollingRefine (commit `b566de6`, parked at 4h timeout)
- [x] `RoutingSummarizer` + `SummarizationProfile` data model shipped (commit `b3f3157`). Routing held as data; Tier 3 swap is one-line.
- [x] `_TextInput.guideline` per-call directive shipped. OneShot inline, RollingRefine post-loop format pass (commit `b3f3157`). Documented convention: any future strategy added to `PRODUCTION_ROUTING` must opt in to guideline at its final user-facing call only, or Siphon-supplied guidelines silently drop.
- [x] `ArticleEnricher` shipped and live-validated end-to-end against a real article (commit `040cfe3`, deployed 2026-06-01). Tier1 routing confirmed, guideline applied correctly, structured markdown output preserves the intended format.
- [x] `guideline.jinja2` scaffold for article (`siphon-server/src/siphon_server/sources/article/guideline.jinja2`). Convention: `.jinja2` extension for templated guidelines, `.md` for un-templated stubs.

**Open (in priority order)**:
- [ ] **Description workflow redesign per `siphon-server/dev/retrieval.md`**: description becomes a HyDE-shaped retrieval-only artifact, generated by a one-shot pass on the summary (not the raw text). Hands a bounded input to gpt-oss; eliminates the long-input description problem permanently. Article first (Phase R1); other 9 sources follow per Phase R5.
- [ ] Aggregate rerun results once the Cronicle timeout is fixed. Tier 3 (RollingRefine vs hybrid) remains an open question, but doesn't block production routing.
- [ ] Roll routing summarizer to the 9 remaining standard Siphon enrichers (arxiv, audio, doc, email, github, image, obsidian, video, youtube). `drive` (NotImplementedError stub) and `podcasts` (no enricher) skipped. Each is a near-identical pattern; see `siphon-server/dev/summarization.md`.
- [ ] Author actual per-SourceType guideline content (dedicated sessions with user input per source).
- [ ] Run 3: per-SourceType guideline tuning eval.
- [ ] Assemble holdout set for Run 4; validate published config on holdout.

**Cross-references**:
- Siphon enrichment architecture: `siphon-server/dev/summarization.md`
- Siphon retrieval architecture (embedding model migration, HyDE, RRF): `siphon-server/dev/retrieval.md`

---

## Infrastructure Notes

- **NAS-backed artifacts** (`$NAS/evals/<project>/<eval>/`) are the standard for results, status, and logs. Filenames encode provenance: `<eval>__<utc_ts>__<host>__<artifact>.<ext>`. Use `evals/nas.py:artifact_paths()`. Fail-fast if `$NAS` is unset or unmounted. See `evals/ARCHITECTURE.md` for the full spec.
- **Routing held as data**: the production `RoutingSummarizer` reads its routing from a module-level list of `(token_max, SummarizationProfile)` tuples. When the rerun settles or future evals shift the breakpoints, the swap is a one-line change — no refactor.
- **`RecursiveSummarizer` overflow strategy** is currently hardwired to `MapReduceSummarizer`. Not blocking the current routing (Recursive isn't in the production tiers), but worth fixing if future evals want it back.
- All strategies support `use_remote=True` + `host_alias` for remote model routing via Headwater.
- Context window sizes configured in `~/.config/conduit/ollama_context_sizes.json` (eval models set to 128K nominal; effective context per ECW data is much smaller — see Published Routing Decision).
- `evals.py` uses `return_exceptions=True` in gather — individual failures are logged and skipped, the run continues.
- **Cronicle events for this project**:
  - `emothl43a01` — "summarization evals" → `jobs/run2.py` (strategy + Tier 1/2 timing rerun)
  - `empgbar020l` — "ECW sweep (multi-model)" → `jobs/ecw_sweep.py`
  - `emp50harw0e` — "Session Summary Quality Check — gemma4:latest" → `jobs/summarize_session_quality.py` (TBD, blocked on long-text harness — see CLAUDE.md)
