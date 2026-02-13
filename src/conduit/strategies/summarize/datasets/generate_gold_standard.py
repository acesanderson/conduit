import os
import json
import asyncio
import google.generativeai as genai
from tqdm import tqdm
from conduit.config import settings
from conduit.strategies.summarize.datasets.corpus import fetch_siphon_stratified
from conduit.core.prompt.prompt_loader import PromptLoader
from pathlib import Path

MODEL_NAME = "gemini3"
OUTPUT_FILE = settings.paths["DATA_DIR"] / "gold_standard.json"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
PROMPT_LOADER = PromptLoader(PROMPTS_DIR)


async def generate_gold_summary(model, text):
    """Sends a single doc to Gemini Cloud."""
    try:
        response = await model.generate_content_async(
            GOLDEN_PROMPT.format(text=text),
            generation_config={"response_mime_type": "application/json"},
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"❌ Error generating summary: {e}")
        return None


async def main():
    # 1. Setup
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Please set GOOGLE_API_KEY environment variable.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(MODEL_NAME)

    # 2. Build Data
    dataset = build_composite_dataset()
    print(f"📦 Total Documents: {len(dataset)}")

    confirm = input("   Proceed? (y/n): ")
    if confirm.lower() != "y":
        return

    # 3. Execution Loop
    results = []
    print("🚀 Starting Generation...")

    # Semaphore to prevent hitting rate limits (5 concurrent requests)
    sem = asyncio.Semaphore(5)

    async def process_doc(doc):
        async with sem:
            res = await generate_gold_summary(model, doc["text"])
            if res:
                return {
                    "id": doc["source_id"],
                    "category": doc["category"],
                    "source_text": doc["text"],  # Save source for reference
                    "gold_summary": res["summary"],
                    "gold_facts": res["key_facts"],
                    "theme": res["main_theme"],
                }
            return None

    # Run tasks
    tasks = [process_doc(doc) for doc in dataset]
    completed_docs = []

    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result = await f
        if result:
            completed_docs.append(result)

    # 4. Save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(completed_docs, f, indent=2)

    print(f"\n✅ Done! Saved {len(completed_docs)} gold records to {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
