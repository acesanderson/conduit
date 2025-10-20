"""
JSON Schema:
    Start a task:
        {
            "model": "exa-research",
            "instructions": "<string>"
        }
    Response:
        {
          "researchId": "01jszdfs0052sg4jc552sg4jc5",
          "model": "exa-research",
          "instructions": "What species of ant are similar to honeypot ants?",
          "status": "running"
        }

    Get a task (by researchId):
        https://api.exa.ai/research/v1/{researchId}

    Completed task:
        {
          "researchId": "01jszdfs0052sg4jc552sg4jc5",
          "model": "exa-research",
          "instructions": "What species of ant are similar to honeypot ants?",
          "status": "completed",
          "output": "<string>",
          "costDollars": {
            "total": 0.12,
            "input": 0.05,
            "output": 0.07
          },
          "createdAt": "2023-10-05T14:48:00.000Z"
        }
"""

import requests
import os
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path
from conduit.sync import Model, ConduitCache

console = Console()
model = Model("gpt")
cache = ConduitCache()
Model.conduit_cache = cache
EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY environment variable is not set.")
QUERY_STRING = (Path(__file__).parent / "query_string.jinja2").read_text()
TRUNCATED_QUERY_STRING = str(
    model.query(
        query_input="Shorten this query to around 4000 characters at most. Return ONLY the amended query.\n\n"
        + "<query>"
        + QUERY_STRING
        + "</query>",
    ).content
)
OBSIDIAN_PATH = os.getenv("OBSIDIAN_PATH")
if not OBSIDIAN_PATH:
    raise ValueError("OBSIDIAN_PATH environment variable is not set.")
OBSIDIAN_RESEARCH_NOTE = Path(OBSIDIAN_PATH) / "Exa Research Example.md"


def start_research_task(query: str) -> dict:
    url = "https://api.exa.ai/research/v1"
    payload = {"model": "exa-research", "instructions": TRUNCATED_QUERY_STRING}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EXA_API_KEY}",
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json()


def get_research_task(researchId: str) -> dict:
    url = f"https://api.exa.ai/research//v1/{researchId}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EXA_API_KEY}",
    }
    response = requests.get(url, headers=headers)
    return response.json()


if __name__ == "__main__":
    # {'researchId': 'r_01k7z7yn9nzrrw6q3w72sjdbgx',
    retrieved_task = get_research_task("r_01k7z7yn9nzrrw6q3w72sjdbgx")
    output = retrieved_task["output"]["content"]
    console.print(Markdown(output))
    with open(OBSIDIAN_RESEARCH_NOTE, "w") as f:
        f.write(output)
