from conduit.extensions.summarize.strategy import ChunkingStrategy
from conduit.core.workflow.workflow import step, get_param, add_metadata
from typing import override
import semchunk
import tiktoken
import statistics


class Chunker(ChunkingStrategy):
    """
    A Workflow Step that splits text using semantic boundaries and token limits.
    Wraps the 'semchunk' library.

    OPINIONATED DEFAULTS:
    This strategy is tuned for 'Maximal Context Stuffing' in summarization pipelines.
    It defaults to large chunks (12k tokens) with fixed overlap (500 tokens)
    to minimize the number of API calls and preserve narrative arcs.
    """

    @step
    @override
    async def __call__(self, text: str, **kwargs) -> list[str]:
        # --- 1. CONFIGURATION ---

        # Target max tokens per chunk.
        # DEFAULT: 12,000.
        # Rationale: Fits comfortably within a 16k context window (common safety limit),
        # leaving ~4k buffer for System Prompt + Summary Output.
        # For 128k context models, override this to ~120,000 via config.
        chunk_size = get_param("chunk_size", default=12000)

        # Overlap size.
        # DEFAULT: 500 (Fixed Token Count).
        # Rationale: 500 tokens (~300 words) captures roughly 2-3 paragraphs.
        # This ensures that if a split happens mid-topic, the next chunk has enough
        # 'lookback' to restore the context. Avoid percentages for large chunks.
        overlap = get_param("overlap", default=500)

        # Tokenizer model string.
        # DEFAULT: "gpt-4o".
        # Rationale: Uses the 'o200k_base' encoding, which is the standard for
        # modern high-performance models (GPT-4o, etc).
        model_name = get_param("tokenizer_model", default="gpt-4o")

        # Optimization flag.
        memoize = get_param("memoize", default=True)

        # --- 2. SETUP ---
        try:
            tokenizer = tiktoken.encoding_for_model(model_name)
        except KeyError:
            # Fallback for newer/unknown models (e.g. gpt-oss specific tags)
            # to the standard modern encoding.
            tokenizer = tiktoken.get_encoding("o200k_base")

        # Create the closure for semchunk (it requires a callable, not an object)
        def count_tokens(t: str) -> int:
            return len(tokenizer.encode(t))

        # --- 3. EXECUTION ---
        chunks = semchunk.chunk(
            text,
            chunk_size=chunk_size,
            overlap=overlap,
            token_counter=count_tokens,
            memoize=memoize,
        )

        # --- 4. TELEMETRY ---
        add_metadata("num_chunks", len(chunks))
        add_metadata("original_text_chars", len(text))

        if chunks:
            avg_chars = statistics.mean(len(c) for c in chunks)
            add_metadata("avg_chunk_chars", int(avg_chars))
            # Track efficiency: How full is the final chunk?
            # Tiny final chunks can sometimes be merged or ignored in post-processing.
            add_metadata("last_chunk_token_est", count_tokens(chunks[-1]))

        return chunks
