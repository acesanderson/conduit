from conduit.remote import RemoteModel, Response

# CONFIG
PREFERRED_MODEL = "gpt-oss:latest"
model = RemoteModel(PREFERRED_MODEL)
CHUNK_SIZE = 20000  # ~5k tokens (Safe zone)
OVERLAP = 1000  # Preserves context between cuts


def query_ollama(prompt_str: str) -> str:
    response = model.query(prompt_str)
    assert isinstance(response, Response)
    return str(response.content)


def create_chunks(text: str, chunk_size: int, overlap: int):
    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        # Move start forward, but backtrack by overlap amount
        start += chunk_size - overlap

    return chunks


def map_reduce_summarize(long_transcript: str):
    # 1. MAP STEP: Summarize chunks
    chunks = create_chunks(long_transcript, CHUNK_SIZE, OVERLAP)
    chunk_summaries = []

    print(f"Processing {len(chunks)} chunks...")

    for i, chunk in enumerate(chunks):
        prompt = f"""
        You are a summarization engine. Summarize the following video transcript segment. 
        Focus on factual claims, arguments, and key details. Do not introduce outside info.
        
        TRANSCRIPT SEGMENT:
        {chunk}
        """
        print(f"Summarizing chunk {i + 1}/{len(chunks)}...")
        summary = query_ollama(prompt)
        chunk_summaries.append(summary)

    # 2. REDUCE STEP: Consolidate
    print("Consolidating final summary...")
    combined_text = "\n\n".join(chunk_summaries)

    final_prompt = f"""
    You are an executive editor. Below is a series of rough notes from video segments.
    Compile them into a single, coherent Executive Summary with a 'Section Insights' list.
    Resolve any repetitive information.
    
    SEGMENT NOTES:
    {combined_text}
    """

    return query_ollama(final_prompt)


final_summary = map_reduce_summarize(huge_transcript_string)
print(final_summary)
