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
from openai.types.responses.response_output_message import ResponseOutputMessage
import os
import time
from rich.console import Console
from pathlib import Path

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


def retrieve_content(response) -> tuple[str, list]:
    """
    Extract text and citations from the response output.

    Returns
    -------
    tuple[str, list]
        The extracted text and list of citations.
    """
    output = response.output
    for o in output:
        if isinstance(o, ResponseOutputMessage):
            content = o.content[0]  # One item in list
            text = content.text
            citations = content.annotations
            break
    return text, citations


def format_citations(citations: list) -> str:
    """
    Format citations into a single markdown string.
    """
    output = ""
    citation_lines = []
    for cite in citations:
        citation_lines.append(f"- [{cite.title}]({cite.url})")
    output += "# Citations\n"
    output += "\n".join(citation_lines)
    return output


def retrieve_answer(response) -> str:
    """
    Extract the final answer text from the response output.
    """
    text, citations = retrieve_content(response)
    output = ""
    output += "# Research Answer\n"
    output += text + "\n\n"
    output += format_citations(citations)
    return output


if __name__ == "__main__":
    response_id = start_research_task(QUERY_STRING)
    response = get_research_task(response_id)
    console.print(response.output)
    _ = OBSIDIAN_RESEARCH_NOTE.write_text(retrieve_answer(response))
