from conduit.tools.tool import ToolCall, ToolCallError, Tool
from pydantic import BaseModel, Field, field_validator
from typing import Literal
import markdownify
import readabilipy


# Our pydantic classes
class FetchUrlParameters(BaseModel):
    url: str = Field(description="The URL of the web page to fetch")

    @field_validator("url")
    def validate_url(cls, v: str) -> str:
        if not v.startswith("http://") and not v.startswith("https://"):
            raise ValueError("URL must start with http:// or https://")
        return v


class FetchUrlToolCall(ToolCall):
    """
    Fetch the content of a web page.
    """

    tool_name: Literal["fetch_url"] = "fetch_url"
    parameters: FetchUrlParameters


def extract_content_from_html(html: str) -> str:
    """Extract and convert HTML content to Markdown format.

    Args:
        html: Raw HTML content to process

    Returns:
        Simplified markdown version of the content
    """
    ret = readabilipy.simple_json.simple_json_from_html_string(
        html, use_readability=True
    )
    if not ret["content"]:
        return "<error>Page failed to be simplified from HTML</error>"
    content = markdownify.markdownify(
        ret["content"],
        heading_style=markdownify.ATX,
    )
    return content


async def fetch_url(
    url: str, user_agent: str, force_raw: bool = False, proxy_url: str | None = None
) -> tuple[str, str]:
    """
    Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
    """
    from httpx import AsyncClient, HTTPError

    async with AsyncClient() as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={"User-Agent": user_agent},
                timeout=30,
            )
        except HTTPError as e:
            raise ToolCallError(f"Failed to fetch {url}: {e}")
        if response.status_code >= 400:
            raise ToolCallError(
                f"Failed to fetch {url} - status code {response.status_code}"
            )

        page_raw = response.text

    content_type = response.headers.get("content-type", "")
    is_page_html = (
        "<html" in page_raw[:100] or "text/html" in content_type or not content_type
    )

    if is_page_html and not force_raw:
        return extract_content_from_html(page_raw), ""

    return (
        page_raw,
        f"Content type {content_type} cannot be simplified to markdown, but here is the raw content:\n",
    )


example_query = "Fetch this web page: https://example.com/article"
example_params = {"url": "https://example.com/article"}


FetchUrlTool = Tool(
    tool_call_schema=FetchUrlToolCall,
    function=fetch_url,
    example_query=example_query,
    example_params=example_params,
)
