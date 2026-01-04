# === CONSTANTS ===
SAFE_INPUT_WINDOW = 16_000  # tokens we trust model to handle well
CHUNK_SIZE = 14_000  # leaves room for system prompt + output
TARGET_COMPRESSION = 10  # rough ratio, tunable per source type


# === CORE FUNCTIONS ===


def token_count(text: str) -> int:
    """Return token count for text."""
    ...


def chunk_naive(text: str, chunk_size: int) -> list[str]:
    """
    Split text into chunks of roughly chunk_size tokens.
    Snap to sentence boundaries to avoid mid-sentence splits.
    """
    ...


def summarize_chunk(chunk: str, chunk_index: int, total_chunks: int) -> str:
    """
    Summarize a single chunk. Returns structured output:

    [SECTION {chunk_index + 1} OF {total_chunks}]
    {summary text}
    """
    prompt = f"""
    Summarize the following content. This is section {chunk_index + 1} of {total_chunks}.
    Preserve key facts, entities, and relationships.
    Target ~{len(chunk) // TARGET_COMPRESSION} tokens.
    
    {chunk}
    """
    return llm_call(prompt)


def join_structured(summaries: list[str]) -> str:
    """Join structured chunk summaries with blank line separators."""
    return "\n\n".join(summaries)


# === STRATEGY 1: SINGLE PASS ===


def single_pass(text: str) -> str:
    """
    Direct summarization when input fits in window.
    This IS the polish step for short content.
    """
    assert token_count(text) <= SAFE_INPUT_WINDOW
    return text  # pass through to polish step unchanged


# === STRATEGY 2: MAP-REDUCE ===


def map_reduce(text: str) -> str:
    """
    Chunk -> summarize each -> join.
    Use when: input exceeds window, but chunk summaries fit in window.
    """
    chunks = chunk_naive(text, CHUNK_SIZE)

    # Map: summarize each chunk in parallel
    summaries = parallel_map(
        lambda i, chunk: summarize_chunk(chunk, i, len(chunks)), enumerate(chunks)
    )

    # Reduce: join structured summaries
    combined = join_structured(summaries)

    assert token_count(combined) <= SAFE_INPUT_WINDOW, (
        "Combined summaries exceed window — use hierarchical instead"
    )

    return combined


# === STRATEGY 3: HIERARCHICAL ===


def hierarchical(text: str) -> str:
    """
    Recursively condense until output fits in window.
    Use when: even chunk summaries exceed window.
    """
    current = text
    iteration = 0
    max_iterations = 5  # safety bound

    while token_count(current) > SAFE_INPUT_WINDOW:
        if iteration >= max_iterations:
            raise RuntimeError("Failed to condense within iteration limit")

        chunks = chunk_naive(current, CHUNK_SIZE)

        summaries = parallel_map(
            lambda i, chunk: summarize_chunk(chunk, i, len(chunks)), enumerate(chunks)
        )

        current = join_structured(summaries)
        iteration += 1

    return current


# === ROUTER ===


def condense(text: str) -> str:
    """
    Route to appropriate strategy based on input size.
    Returns text ready for polish step.
    """
    input_tokens = token_count(text)

    if input_tokens <= SAFE_INPUT_WINDOW:
        return single_pass(text)

    # Estimate output size after one map-reduce pass
    chunks = chunk_naive(text, CHUNK_SIZE)
    estimated_output = len(chunks) * (CHUNK_SIZE // TARGET_COMPRESSION)

    if estimated_output <= SAFE_INPUT_WINDOW:
        return map_reduce(text)
    else:
        return hierarchical(text)


# === FULL PIPELINE ===


def summarize(text: str, polish_prompt: str) -> str:
    """
    Full two-step pipeline: condense (if needed) -> polish.

    Args:
        text: raw input content
        polish_prompt: source-specific jinja2 template (your existing prompts)

    Returns:
        Final polished summary
    """
    condensed = condense(text)
    polished = llm_call(polish_prompt.render({"text": condensed}))
    return polished
