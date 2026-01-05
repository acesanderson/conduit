# RAG Strategy Specification

## Overview

RAG (Retrieval-Augmented Generation) is fundamentally more complex than Summarization. Where Summarization is a **stateless transformation** (`text -> summary`), RAG involves **stateful infrastructure**: you build indexes, persist them, query them repeatedly, and update them as sources change.

This complexity manifests in multiple dimensions:

| Dimension | Summarization | RAG |
|-----------|---------------|-----|
| Data flow | Input → Output | Source → Index → Query → Results |
| State | Stateless | Stateful (persistent indexes) |
| Lifecycle | Single invocation | Build once, query many |
| Components | Strategy only | Corpus + Pipeline + Retriever + Reranker |
| Evaluation surface | Output quality | Indexing quality + Retrieval quality + Output quality |

RAG requires separating **build-time concerns** (indexing, embedding, chunking) from **query-time concerns** (retrieval, reranking). This spec defines abstractions for both.

---

## Architecture

### Core Concepts

| Concept | Role | Stateful? |
|---------|------|-----------|
| **Source** | Raw access to files/records (e.g., Obsidian vault, Siphon DB) | No (just a pointer) |
| **Pipeline** | Transforms Source into queryable Corpus; `@step`-decorated but not a Strategy | Yes (creates/updates artifacts) |
| **Corpus** | Interface to indexed content; protocol-based, swappable | No (reads from artifacts) |
| **RetrieverStrategy** | Finds relevant chunks from a Corpus | No |
| **RerankerStrategy** | Reorders/filters retrieved chunks | No |

### Data Flow

```
Source → Pipeline → Corpus → RetrieverStrategy → list[Chunk] → RerankerStrategy → list[Chunk]
```

Key insight: **Corpus is the boundary**. Pipelines produce them, Strategies consume them. You can build a Corpus once and query it many times. You can swap how it was built without changing how it's queried.

### Relationship to Existing Code

This spec integrates with existing Conduit components:

| Existing Component | Role in RAG |
|--------------------|-------------|
| `@step` decorator | Wraps Pipeline.sync(), Retriever.__call__(), Reranker.__call__() for tracing |
| `get_param()` | Runtime configuration for all tunable parameters |
| `ConduitHarness` | Manages trace context for RAG workflows |
| `Chunker` (from summarizers) | Reused for chunk boundary detection |
| `EmbeddingModel` (from headwater) | Reused for vector generation |
| ChromaDB services | Backend for ChromaCorpus |

---

## Protocol Definitions

### Chunk

The universal handoff format between retrievers and rerankers:

```python
@dataclass
class Chunk:
    id: str
    content: str
    score: float | None  # None before scoring, populated by retriever/reranker
    metadata: dict[str, Any]  # source file, position, timestamps, tags, etc.
```

### Corpus

```python
class Corpus(Protocol):
    def list_documents(self) -> list[str]:
        """Return all document IDs in the corpus."""
        ...
    
    def get_document(self, id: str) -> str:
        """Retrieve full document content by ID."""
        ...
    
    def get_all_text(self) -> Iterable[Chunk]:
        """Yield all chunks for full-text scanning (grep-style)."""
        ...


class SupportsEmbeddingQuery(Protocol):
    def query_embeddings(self, vector: list[float], n: int) -> list[Chunk]:
        """Return n most similar chunks to the query vector."""
        ...


class SupportsTokenEmbeddings(Protocol):
    def query_token_embeddings(self, vectors: list[list[float]], n: int) -> list[Chunk]:
        """ColBERT-style MaxSim scoring against token-level vectors."""
        ...
```

### Pipeline

```python
class Pipeline(Protocol):
    async def sync(self) -> None:
        """
        Idempotent sync: update target corpus to reflect current source state.
        Subsequent calls only process changed documents.
        """
        ...
```

### RetrieverStrategy

```python
class RetrieverStrategy(Protocol):
    async def __call__(self, query: str, n: int) -> list[Chunk]:
        """Retrieve n relevant chunks for the query."""
        ...
```

### RerankerStrategy

```python
class RerankerStrategy(Protocol):
    async def __call__(self, query: str, chunks: list[Chunk], n: int) -> list[Chunk]:
        """Rerank chunks and return top n."""
        ...
```

---

## Configuration Surface

All parameters tunable via `get_param()` with standard scoping rules (`{step_name}.{key}` → `{key}` → default):

### Pipeline Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `embedding_model` | Model for vector generation | `sentence-transformers/all-MiniLM-L6-v2` |
| `chunk_size` | Target tokens per chunk | `12000` |
| `overlap` | Token overlap between chunks | `500` |
| `chunking_mode` | `traditional` or `late` | `traditional` |

### Retriever Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `n_results` | Number of chunks to retrieve | `10` |
| `embedding_model` | Must match pipeline's model | (inherited) |

### Reranker Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `reranker_model` | Cross-encoder model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `n_results` | Final number to return | `3` |

---

## Directory Structure

```
strategies/
  rag/
    corpus/
      protocol.py           # Corpus, SupportsEmbeddingQuery, SupportsTokenEmbeddings, Chunk
      obsidian.py           # ObsidianVault
      chroma.py             # ChromaCorpus (chunk vectors)
      siphon.py             # SiphonCorpus (wraps ProcessedContent + pgvector)
      colbert.py            # ColBERTCorpus (token vectors; wraps RAGatouille)
      future/
        future.md

    pipelines/
      pipeline.py           # Pipeline base, ChunkingMode enum (TRADITIONAL, LATE, COLBERT)
      vector.py             # VectorPipeline (traditional + late chunking via config)
      colbert.py            # ColBERTPipeline (token-level indexing)
      future/
        future.md

    retrievers/
      strategy.py           # RetrieverStrategy protocol
      grep.py               # GrepRetriever (full-text scan)
      semantic.py           # SemanticRetriever (chunk-level similarity)
      colbert.py            # ColBERTRetriever (MaxSim token scoring)
      future/
        future.md

    rerankers/
      strategy.py           # RerankerStrategy protocol
      cross_encoder.py      # CrossEncoderReranker
      future/
        future.md
```

---

## MVP Implementation

### Scope

- ObsidianVault corpus (read markdown files)
- VectorPipeline with traditional chunking
- ChromaCorpus for storage
- SemanticRetriever for embedding-based search
- GrepRetriever for full-text search
- CrossEncoderReranker for result refinement

### Usage Example

```python
from strategies.rag.corpus.obsidian import ObsidianVault
from strategies.rag.corpus.chroma import ChromaCorpus
from strategies.rag.pipelines.vector import VectorPipeline
from strategies.rag.retrievers.semantic import SemanticRetriever
from strategies.rag.rerankers.cross_encoder import CrossEncoderReranker
import os

# 1. Define source and target
vault = ObsidianVault(path=os.getenv("OBSIDIAN_PATH"))
corpus = ChromaCorpus(collection="obsidian-notes")

# 2. Build index (run once, or periodically)
pipeline = VectorPipeline(source=vault, target=corpus)
await pipeline.sync()

# 3. Retrieve candidates
retriever = SemanticRetriever(corpus=corpus)
candidates = await retriever(query="gift ideas for family members", n=20)

# 4. Rerank and return top 3
reranker = CrossEncoderReranker()
results = await reranker(query="gift ideas for family members", chunks=candidates, n=3)
```

### Notes

- `pipeline.sync()` is idempotent; subsequent calls update only changed files
- Retriever and reranker are stateless; instantiate once, reuse across queries
- All components are `@step`-decorated for tracing via `ConduitHarness`
- Configuration injectable via harness config for eval sweeps

---

## Middle-Term: Embedding Strategies

Three approaches to embedding, trading off simplicity, storage cost, and retrieval quality:

### Traditional Chunking (MVP)

Split documents into chunks, embed each independently.

- **Pros:** Simple, works with any embedding model, low storage
- **Cons:** Chunks lose surrounding context
- **Implementation:** `VectorPipeline` with `chunking_mode="traditional"`

### Late Chunking

Run embedding model over full document, then pool token embeddings into chunk vectors.

- **Pros:** Chunks retain cross-document context, same storage cost
- **Cons:** Bounded by model's max sequence length, more complex
- **Implementation:** `VectorPipeline` with `chunking_mode="late"`

```python
pipeline = VectorPipeline(
    source=vault,
    target=corpus,
    chunking_mode="late",  # embed full doc, pool into chunks
)
```

### ColBERT (Late Interaction)

Store one vector per token, score by summing best token-level matches.

- **Pros:** Best generalization, fine-grained matching
- **Cons:** Much higher storage, requires ColBERT-specific indexing
- **Implementation:** Separate `ColBERTPipeline` + `ColBERTCorpus` + `ColBERTRetriever`

```python
from strategies.rag.corpus.colbert import ColBERTCorpus
from strategies.rag.pipelines.colbert import ColBERTPipeline
from strategies.rag.retrievers.colbert import ColBERTRetriever

corpus = ColBERTCorpus(index_path="./colbert_index")
pipeline = ColBERTPipeline(source=vault, target=corpus)
await pipeline.sync()

retriever = ColBERTRetriever(corpus=corpus)
results = await retriever(query="gift ideas", n=10)
```

**Rule of thumb:** Traditional → Late Chunking → ColBERT represents increasing quality and complexity. Move up the ladder when retrieval quality is demonstrably your bottleneck.

---

## Futures

### corpus/future/future.md

| Name | Description |
|------|-------------|
| `multi_corpus` | Federated search across multiple corpora; union/intersection semantics |
| `temporal_corpus` | Time-aware corpus; supports queries like "what did I write last month" |
| `versioned_corpus` | Git-like semantics; query against historical snapshots, diff between versions |
| `hierarchical_corpus` | Parent-child relationships; notes link to sub-notes, books contain chapters |
| `filtered_corpus` | Wrapper that applies predicates (by tag, folder, date range) before search |
| `cached_corpus` | Wrapper that caches frequent queries; LRU or TTL-based invalidation |
| `lazy_corpus` | Deferred loading; only fetches content when accessed |
| `hybrid_corpus` | Exposes both full-text and embedding query; merges results with configurable fusion |

### pipelines/future/future.md

| Name | Description |
|------|-------------|
| `incremental_pipeline` | Watches source for changes; only re-indexes modified documents |
| `scheduled_pipeline` | Cron-like; runs sync at intervals |
| `multi_index_pipeline` | Single source → multiple corpora (one fine-grain, one whole-document) |
| `summary_augmented_pipeline` | Generates synthetic summaries per chunk; embeds both original and summary |
| `hypothetical_document_pipeline` | HyDE-style; generates hypothetical answers for each chunk, embeds those |
| `parent_document_pipeline` | Indexes chunks but stores references to full parent documents |
| `multimodal_pipeline` | Handles images, tables, diagrams; extracts text descriptions |
| `deduplication_pipeline` | Detects near-duplicate content across notes; merges or flags |
| `entity_extraction_pipeline` | Extracts named entities; stores as structured metadata |
| `knowledge_graph_pipeline` | Entities + relationships; builds queryable graph structure |
| `graph_rag_pipeline` | Full GraphRAG: entities → relationships → communities → community summaries |

### retrievers/future/future.md

| Name | Description |
|------|-------------|
| `hybrid_retriever` | Combines BM25 + embedding similarity; reciprocal rank fusion |
| `mmr_retriever` | Maximal Marginal Relevance; diversity-aware to reduce redundancy |
| `query_expansion_retriever` | LLM rewrites query into variants; unions results |
| `hyde_retriever` | Generates hypothetical answer, embeds that, searches for similar |
| `recursive_retriever` | Retrieves chunks, then follows links to pull related documents |
| `agent_retriever` | LLM decides which corpus to query, what filters, iterates until satisfied |
| `sparse_retriever` | BM25/TF-IDF only; baseline for comparison |
| `metadata_retriever` | Filters by structured fields before or instead of semantic search |
| `tree_retriever` | RAPTOR-style; traverses hierarchical summaries from root to leaves |
| `ensemble_retriever` | Runs multiple retrievers in parallel; merges with learned weights |
| `graph_traversal_retriever` | Finds entities, walks edges to related chunks |
| `community_retriever` | Matches query to community summaries, returns member chunks |
| `graph_rag_retriever` | Orchestrates local (traversal) vs global (community) based on query |

### rerankers/future/future.md

| Name | Description |
|------|-------------|
| `llm_reranker` | Asks LLM to rank candidates by relevance; expensive but high quality |
| `cohere_reranker` | Wraps Cohere's rerank API |
| `diversity_reranker` | MMR-style post-retrieval; penalizes redundancy among top results |
| `recency_reranker` | Boosts recent documents; tunable decay curve |
| `personalized_reranker` | Learns user preferences; upweights topics they engage with |
| `chain_reranker` | Composes multiple rerankers sequentially; coarse filter → fine rerank |
| `threshold_reranker` | Drops candidates below score threshold; variable-length results |
| `contextual_reranker` | Considers conversation history, not just current query |
| `calibrated_reranker` | Outputs interpretable confidence scores; enables "I don't know" detection |

---

## Open Questions

Decisions deferred for future resolution:

### Chunking & Late Chunking

- Where exactly does chunking happen for late chunking mode? Pipeline needs chunk boundaries before embedding but must embed full doc first. Likely: chunk for boundaries → embed full doc → pool by boundaries.

### ObsidianVault Specifics

- How to handle frontmatter? Parse as metadata, strip from content, or both?
- How to handle tags (`#tag`)? Extract to metadata?
- How to handle wikilinks (`[[Other Note]]`)? Resolve to content, treat as metadata, or ignore?
- How to handle embeds (`![[Embedded Note]]`)? Inline the content?

### Incremental Indexing

- Cache invalidation strategy? Content hash? Modified timestamp?
- How to handle deletions? Tombstones? Full reconciliation?

### Evaluation

- What metrics for retrieval quality? Precision@k, Recall@k, MRR, NDCG?
- How to construct ground truth for Obsidian vault? Manual labeling? Synthetic queries?

### Multi-Corpus

- How to handle different embedding models across corpora?
- Score normalization when merging results from different sources?
