"""
OpenAI research example with web search tool.

Docs: https://platform.openai.com/docs/guides/deep-research

Of course. Based on the object you provided, here is a simplified tree structure outlining how to access its data.

The top-level object is a **list** containing various event objects that occur in sequence. The most important ones for accessing data are `ResponseFunctionWebSearch` and `ResponseOutputMessage`.

```
list [
  ├── ResponseReasoningItem
  │
  ├── ResponseFunctionWebSearch
  │   └── action
  │       ├── ActionSearch
  │       │   └── query: str
  │       │
  │       ├── ActionOpenPage
  │       │   └── url: str
  │       │
  │       └── ActionFindInPage
  │           ├── pattern: str
  │           └── url: str
  │
  └── ResponseOutputMessage      // This is the final generated content
      └── content: list [
          └── ResponseOutputText
              ├── text: str       // The complete text of the response
              └── annotations: list [
                  └── AnnotationURLCitation
                      ├── start_index: int
                      ├── end_index: int
                      ├── title: str
                      └── url: str
              ]
      ]
]
```
"""

from openai import OpenAI
import os
import time
from rich.console import Console
from pathlib import Path
import shelve

console = Console()

# Pass it to OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Constants
QUERY_STRING = (Path(__file__).parent / "query_string.jinja2").read_text()
OBSIDIAN_PATH = os.getenv("OBSIDIAN_PATH")
if not OBSIDIAN_PATH:
    raise ValueError("OBSIDIAN_PATH environment variable is not set.")
OBSIDIAN_RESEARCH_NOTE = Path(OBSIDIAN_PATH) / "OpenAI Research Example.md"
assert Path(OBSIDIAN_PATH).exists(), "Obsidian path does not exist."


# Progress spinner
def start_research_task(query: str) -> str:
    with console.status("[bold green]Sending requests...") as status:
        response = client.responses.create(
            model="o4-mini-deep-research-2025-06-26",
            input=[
                {"role": "system", "content": "You are an expert research assistant."},
                {
                    "role": "user",
                    "content": QUERY_STRING,
                },
            ],
            tools=[
                {"type": "web_search_preview"},
            ],
            reasoning={"effort": "medium"},  # optional parameter for complex research
            background=True,
        )
    response_id = response.id
    return response_id


def get_research_task(response_id: str):
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
    return status_check


def retrieve_output_text(response) -> str:
    """
    Not implemented yet.
    """
    raise NotImplementedError
    output = response.output
    for item in output:
        if item["type"] == "ResponseOutputMessage":
            content = item["content"]
            for content_item in content:
                if content_item["type"] == "ResponseOutputText":
                    return content_item["text"]
            annotations = []


if __name__ == "__main__":
    response_id = start_research_task(QUERY_STRING)
    response = get_research_task(response_id)
    console.print(response.output)
    _ = OBSIDIAN_RESEARCH_NOTE.write_text(str(response.output))
    # Shelve response, query as key, + response object as value
    with shelve.open("research_shelve.db") as db:
        db[QUERY_STRING] = response
    console.print(f"Research note saved to {OBSIDIAN_RESEARCH_NOTE}")
    console.print("Research response shelved in research_shelve.db")
