import requests
import requests_cache
import os
from pydantic import BaseModel, Field
from functools import cached_property
from conduit.sync import Model, Prompt, Conduit, ConduitCache, Verbosity, Response
from newspaper import Article
from rich.console import Console
import logging

# Configs
requests_cache.install_cache("brave_search_cache", expire_after=86400)
Model.conduit_cache = ConduitCache("research")
Model.console = Console()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY")
# EXAMPLE_SEARCH = (
#     "I need the earnings call transcript for the latest earnings call from Udemy."
# )
EXAMPLE_SEARCH = (
    "I need the earnings call transcript for the latest earnings call from Coursera."
)
VERBOSITY = Verbosity.PROGRESS
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
}
PREFERRED_MODEL = "haiku"
WHITE_LIST = {
    "finance.yahoo.com",
    "motleyfool.com",
    "investorplace.com",
    "insidermonkey.com",
    "thestreet.com",
}
BLACK_LIST = {"seekingalpha.com"}


# Data classes
class BraveSearchResult(BaseModel):
    title: str = Field(..., description="The title of the search result.")
    url: str = Field(..., description="The URL of the search result.")
    is_source_local: bool = Field(
        default=False, description="Indicates if the source is local."
    )
    is_source_both: bool = Field(
        default=False, description="Indicates if the source is both local and global."
    )
    description: str = Field(
        ..., description="A brief description of the search result."
    )
    page_age: str | None = Field(default=None, description="The age of the page.")
    profile: dict = Field(
        default={}, description="Profile information related to the search result."
    )
    type: str = Field(..., description="The type of the search result.")
    subtype: str = Field(..., description="The subtype of the search result.")
    meta_url: dict = Field(default={}, description="Metadata URL information.")
    thumbnail: dict = Field(
        default={}, description="Thumbnail image URL of the search result."
    )
    age: str | None = Field(default=None, description="The age rating of the content.")

    @cached_property
    def llm_context(self) -> str:
        """
        Generate an XML string suitable for LLM input.
        """
        context = f"""<search_result>
        <title>{self.title}</title>
        <url>{self.url}</url>
        <description>{self.description}</description>
        </search_result>\n"""
        return context


# Prompt strings
selector_prompt = """
Output only the best matching URL from search_results for the query. Nothing else.

<query>
{{query}}
</query>

<search_results>
{{search_results}}
</search_results>
""".strip()


def construct_earnings_call_query(company: str) -> str:
    """
    Construct a search query for the latest earnings call transcript of a company.
    """
    query = f"I need the earnings call transcript for the latest earnings call from {company}."
    # Build whitelist query properly
    whitelist_query = " OR ".join([f"site:{site}" for site in WHITE_LIST])
    query = f"{query} ({whitelist_query})"
    # Then add blacklist
    for site in BLACK_LIST:
        query += f" NOT site:{site}"  # Exclude blacklisted sites
    logger.info(f"Constructed query: {query}")
    return query


def is_valid_url(response_str: str) -> bool:
    """
    Should be a valid URL starting with http or https.
    """
    return response_str.startswith("http://") or response_str.startswith("https://")


def route_through_archive(url: str) -> str:
    """
    Check the Wayback Machine for an archived version of the URL.
    """
    archive_api_url = "https://archive.org/wayback/available"
    truncated_url = url.replace("https://", "").replace("http://", "")
    params = {"url": truncated_url}
    response = requests.get(archive_api_url, params=params)
    response.raise_for_status()
    data = response.json()
    breakpoint()
    snapshots = data.get("archived_snapshots", {})
    closest = snapshots.get("closest", {})
    if closest.get("available"):
        return closest.get("url")
    return url


def download_article(url) -> Article | None:
    """
    Feed this a url for an article, and it will return the text of the article.
    """
    # First, validate and possibly route through archive
    if not is_valid_url(url):
        raise ValueError(f"Invalid URL: {url}")
    # Download the article
    article = Article(url, request_headers=HEADERS)
    article.download()
    # Check status
    if article.download_state == 1:
        print(f"Failed to download article from original URL: {url}; trying archive...")
        # Try Wayback Machine
        archived_url = route_through_archive(url)
        if archived_url == url:
            raise ValueError(f"No archived version found for URL: {url}")
        article = Article(archived_url, request_headers=HEADERS)
        article.download()
        if article.download_state == 2:
            article.parse()
            return article
        else:
            raise ValueError(
                f"Failed to download article from both original and archived URLs: {url}"
            )
    elif article.download_state == 2:
        article.parse()
        return article


def select_best_url(query: str, search_results: list[BraveSearchResult]) -> str:
    """
    Given a query and a list of BraveSearchResult, select the best URL using an LLM.
    """
    logger.info(f"Selecting best URL for query: {query}")
    input_variables = {
        "query": query,
        "search_results": "\n".join([result.llm_context for result in search_results]),
    }
    logger.info(f"Input variables for URL selection: {input_variables}")
    for index in range(3):
        logger.info(f"Attempting to select best URL, try {index + 1}/3")
        prompt = Prompt(selector_prompt)
        model = Model(PREFERRED_MODEL)
        conduit = Conduit(model=model, prompt=prompt)
        response = conduit.run(
            input_variables=input_variables,
            verbose=VERBOSITY,
        )
        assert isinstance(response, Response), (
            f"Expected response to be of type Response, got {type(response)}"
        )
        if is_valid_url(str(response.content).strip()):
            logger.info(f"Selected best URL: {response.content}")
            return str(response.content).strip()
    raise ValueError("Failed to select a valid URL from search results.")


def brave_search(query, num_results=10) -> list[BraveSearchResult]:
    """
    Use the curl example above to create the equivalent requests code to perform a search on Brave Search.
    """
    # Exclude seekingalpha.com
    query += " NOT site:seekingalpha.com"
    logger.info(f"Performing Brave search for query: {query}")
    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": BRAVE_API_KEY,
    }
    params = {
        "q": query,
        "size": num_results,
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    logger.info(f"Brave search request successful, status code: {response.status_code}")
    response_dict: dict = response.json()
    results: list[dict] = response_dict["web"]["results"]
    objs = []
    for result in results:
        obj = BraveSearchResult(**result)
        logger.info(f"Brave Search Result: {obj.title} - {obj.url}")
        objs.append(obj)
    return objs


def get_earnings_call(company: str) -> Article:
    """
    Main function to get the earnings call transcript for a given company.
    """
    query = construct_earnings_call_query(company)
    search_results = brave_search(query)
    best_url = select_best_url(query, search_results)
    article = download_article(best_url)
    if article:
        return article
    else:
        raise ValueError("Failed to download the article.")


if __name__ == "__main__":
    company_name = "Google"
    article = get_earnings_call(company_name)
    print(f"Earnings Call Transcript for {company_name}:\n")
    print(article.text)
