"""
OpenAI research example with web search tool.
Expand to Gemini deep research.
Try out exa.ai.
"""

from openai import OpenAI
import os
import time
from rich.console import Console
from pathlib import Path

console = Console()


api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
QUERY = "What are concrete ways people are starting to implement skills validation / assessment with generative AI and related technologies? Find concrete case studies that could be relevant to a corporate training context."
SAVE_FILE = Path("research_response.md")

# Progress spinner
with console.status("[bold green]Sending requests...") as status:
    response = client.responses.create(
        model="o4-mini-deep-research-2025-06-26",
        input=[
            {"role": "system", "content": "You are an expert research assistant."},
            {
                "role": "user",
                "content": QUERY,
            },
        ],
        tools=[
            {"type": "web_search_preview"},
        ],
        reasoning={"effort": "medium"},  # optional parameter for complex research
        background=True,
    )
response_id = response.id

with console.status("[bold green]Waiting for response...") as status:
    start_time = time.time()
    while True:
        elapsed = int(time.time() - start_time)
        status.update(f"[bold green]Researching... {elapsed}s elapsed")
        status_check = client.responses.retrieve(response_id)
        if status_check.status == "completed":
            print(status_check.output)
            break
        elif status_check.status == "failed":
            print(f"Error: {status_check.error}")
            break
        time.sleep(2)  # Wait before checking again

console.print(response.output)
_ = SAVE_FILE.write_text(response.output)
