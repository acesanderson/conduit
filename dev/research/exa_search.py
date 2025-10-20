"""
Functions:
    Search:
        query (str): The query string.
        num_results (int, optional): Number of search results to return (default 10).
        include_domains (List[str], optional): Domains to include in the search.
        exclude_domains (List[str], optional): Domains to exclude from the search.
        start_crawl_date (str, optional): Only links crawled after this date.
        end_crawl_date (str, optional): Only links crawled before this date.
        start_published_date (str, optional): Only links published after this date.
        end_published_date (str, optional): Only links published before this date.
        include_text (List[str], optional): Strings that must appear in the page text.
        exclude_text (List[str], optional): Strings that must not appear in the page text.
        use_autoprompt (bool, optional): Convert query to Exa (default False).
        type (str, optional): 'keyword', 'neural', 'hybrid', 'fast', 'deep', or 'auto' (default 'auto').
        category (str, optional): e.g. 'company'
        flags (List[str], optional): Experimental flags for Exa usage.
        moderation (bool, optional): If True, the search results will be moderated for safety.
        user_location (str, optional): Two-letter ISO country code of the user (e.g. US).

Classes:
    SearchResponse:
        results (List[Result]): A list of search results.
        autoprompt_string (str, optional): The Exa query created by autoprompt.
        resolved_search_type (str, optional): 'neural' or 'keyword' if auto.
        auto_date (str, optional): A date for filtering if autoprompt found one.
        context (str, optional): Combined context string when requested via contents.context.
        statuses (List[ContentStatus], optional): Status list from get_contents.
        cost_dollars (CostDollars, optional): Cost breakdown.

    Result:
        title (str): The title of the search result.
        url (str): The URL of the search result.
        id (str): The temporary ID for the document.
        score (float, optional): A number from 0 to 1 representing similarity.
        published_date (str, optional): An estimate of the creation date, from parsing HTML content.
        author (str, optional): The author of the content (if available).
        image (str, optional): A URL to an image associated with the content (if available).
        favicon (str, optional): A URL to the favicon (if available).
        subpages (List[_Result], optional): Subpages of main page
        extras (Dict, optional): Additional metadata; e.g. links, images.

"""

import os
from exa_py import Exa
from urllib.parse import urlparse
from rich.console import Console
from rich.markdown import Markdown
from pathlib import Path

console = Console()
EXA_API_KEY = os.getenv("EXA_API_KEY")
if not EXA_API_KEY:
    raise ValueError("EXA_API_KEY environment variable is not set.")
QUERY_STRING = (Path(__file__).parent / "query_string.jinja2").read_text()

exa = Exa(api_key=EXA_API_KEY)


def get_domain_from_url(url: str):
    parsed_url = urlparse(url)
    return parsed_url.netloc


if __name__ == "__main__":
    result = exa.search_and_contents(QUERY_STRING, text=True, type="auto")
    results = result.results

    output: list[dict[str, str]] = []
    for index, r in enumerate(results):
        print(f"Result {index + 1}/{len(results)}: {r.url}")
        title = r.title
        content = r.text
        markdown_output = f"# {title}\n\n{content}"
        console.clear()
        console.print(Markdown(markdown_output))
        input("Press Enter to continue to the next result...")
