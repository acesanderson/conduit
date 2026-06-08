# Progress

Session handoff. Most recent first. Each entry: what shipped + what remains.

## 2026-06-08

### Shipped

**#4 — RoutingSummarizer eval-matrix gap closed** (conduit `cc9543d`)
- `config_to_canonical_dict()` normalizes Pydantic BaseModels and `type`
  objects so configs with non-scalar fields can hash and persist. Wired
  into `evals.run_eval` and `runner._config_id`.
- `RoutingSummarizer` can now appear in a matrix as a single row.
- 8 tests including a regression that exercises `PRODUCTION_ROUTING`.

**#3 — 0-byte NAS log fix** (conduit `cc9543d`, same commit)
- `EvalRunner._setup_logging` uses `force=True` so it wins against any
  library that attached a handler to root at import time. Per-record
  flush so SIGKILL at the Cronicle timeout still leaves a useful tail.
- Regression test included.

**#1a — Siphon Phase R2–R4 scaffolding** (siphon `56a47a7`, code only,
not deployed)
- `models.py`: `fts_doc` tsvector generated column + GIN index
  `ix_pc_fts`. Additive; existing queries unaffected.
- `setup.py`: `ensure_fts_column()` idempotent helper for backfilling
  the column on the live DB.
- `repository.py`: `get_embed_descriptions`, `search_fts`,
  `search_semantic`, `search_hybrid` (RRF default), `rrf_fuse`,
  `reenrich_row`, `list_uris_for_reenrichment`.
- `migrate_embedding_dim.py`: destructive migration script. Drops HNSW,
  NULLs embeddings, ALTERs `vector(384) → vector(768)`, recreates HNSW.
  Requires `--confirm 'YES I HAVE A BACKUP'`.
- `EMBED_DIM` still 384. Bump deferred until after the destructive
  migration runs.

**#1b — Headwater v2 embed path added, kept off** (headwater `ef2822c`,
code only, not deployed)
- `SIPHON_EMBED_MODEL_V1` = all-MiniLM (current default).
- `SIPHON_EMBED_MODEL_V2` = nomic-embed-text-v1.5 (headwater has v1.5,
  not v2 — retrieval.md says v2; doc needs correction).
- `embed_batch_siphon_service` branches on model name. v1 keeps the
  `(title, summary)` concat; v2 uses description only.
- `SIPHON_EMBED_MODEL = SIPHON_EMBED_MODEL_V1`. One-line flip switches
  the whole pipeline atomically.

**#2 — Re-enrichment job written** (siphon `56a47a7`, code only, not run)
- `siphon-server/scripts/reenrich.py`. Walks rows, reconstructs
  `ContentData`, runs the matching enricher, calls `reenrich_row()`
  which also NULLs the embedding.
- Idempotent via `content_metadata._enrichment_version` tag.
- Cronicle-friendly progress reporting + SIGTERM handling.
- Concurrency-bounded. Requires `--confirm 'YES PROCEED'`.

### Manual actions left

1. **Cronicle timeout flip** for event `emothl43a01`: 14400 → 0 in the
   UI. Auto-classifier blocked the API call as lifecycle.
2. **Destructive sequence** (decided, not fired). Order:
   - `ensure_fts_column()` on live DB (additive).
   - `reenrich.py --dry-run --tag hyde_2026_06` for row counts per source.
   - Postgres snapshot.
   - `reenrich.py --confirm 'YES PROCEED' --tag hyde_2026_06`. Hours
     of LLM calls.
   - `migrate_embedding_dim.py --new-dim 768 --confirm 'YES I HAVE A BACKUP'`.
   - Bump `EMBED_DIM = 768` in siphon `models.py`; deploy siphon.
   - Flip `SIPHON_EMBED_MODEL = SIPHON_EMBED_MODEL_V2` in headwater;
     deploy headwater.
   - Embed-batch with `force=True` across all URIs.

### Open

- `retrieval.md` says nomic-embed-text-v2; headwater only ships v1.5
  (also 768d, schema math holds). Doc correction.
- HyDE query layer (RRF query CLI, `--semantic-only`, `--bm25-only`,
  `--no-hyde` flags) — not started. Phase R4's query side.
- Per-source description guideline tuning (Run 3 of STRATEGY.md) —
  guidelines are currently generic answer-voice; tuning awaits an eval
  cycle on real corpus.

---

## 2026-06-05

### Shipped

**HyDE description rollout to 9 sources (Phase R5)** (siphon `11cbab6`,
deployed)
- Per-source `description_guideline.jinja2` for arxiv, audio, doc (4
  variants: code/data/presentation/prose), email, github, image,
  obsidian, video, youtube.
- Each enricher now runs `summary → description → title` sequentially.
- Smoke-tested obsidian against a synthetic RRF note: 170-word
  answer-voice description, entities preserved, three sequential calls
  fired as designed.
- `drive` (NotImplementedError stub) and `podcasts` (no enricher.py)
  remain skipped.

**Strategy doc sync** (conduit `534985d`, deployed)
- Marked Phase R5 done; next concrete steps are embedding swap and
  re-enrichment of stored rows.

---

## 2026-06-04

### Shipped

**HyDE description for article (Phase R1)** (siphon `83fbb23`, deployed)
- `ArticleEnricher._describe()` runs gpt-oss/bywater one-shot over the
  summary with `description_guideline.jinja2`.
- `enrich()` sequences summary → description (no asyncio.gather).
- Smoke-tested against `0xsid.com/blog/meta-account-takeover-fiasco`:
  answer-voice, entities preserved, no meta-framing.

**RoutingSummarizer rollout to 9 enrichers** (siphon `2b39ca9`, deployed)
- Summary path through `RoutingSummarizer + PRODUCTION_ROUTING` for
  arxiv, audio, doc (4 variants), email, github, image, obsidian,
  video, youtube.
- Each enricher's legacy `<source>_summary.jinja2` became
  `guideline.jinja2` (text block stripped, metadata block retained).
- Description path stayed legacy at this point — replaced by the Phase
  R5 work on 2026-06-05.

**Strategy doc sync** (conduit `fcd17a7`, deployed)
- Article enricher was already shipped (commit `040cfe3`, 2026-06-01)
  but `evals/STRATEGY.md` still said "not yet committed." Corrected.
- Rollout target corrected: 9 implementable enrichers, not 10. `drive`
  and `podcasts` flagged as not-migrate-able pending upstream work.
