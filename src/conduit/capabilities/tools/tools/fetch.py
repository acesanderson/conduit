from conduit.domain.exceptions.exceptions import ToolError
from typing import Annotated


def extract_content_from_html(html: str) -> str:
    """Extract and convert HTML content to Markdown format.

    Args:
        html: Raw HTML content to process

    Returns:
        Simplified markdown version of the content
    """
    import markdownify
    import readabilipy

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


async def fetch_url(url: Annotated[str, "The URL to fetch"]) -> dict[str, str]:
    """
    Fetch the URL and return the content in a form ready for the LLM, as well as a prefix string with status information.
    """
    from httpx import AsyncClient, HTTPError

    async with AsyncClient() as client:
        try:
            response = await client.get(
                url,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
                },
                timeout=30,
            )
        except HTTPError as e:
            raise ToolError(f"Failed to fetch {url}: {e}")
        if response.status_code >= 400:
            raise ToolError(
                f"Failed to fetch {url} - status code {response.status_code}"
            )

        page_raw = response.text

    content_type = response.headers.get("content-type", "")
    is_page_html = (
        "<html" in page_raw[:100] or "text/html" in content_type or not content_type
    )

    if is_page_html:
        return {
            "type": "html",
            "url": url,
            "content": extract_content_from_html(page_raw),
        }

    return {
        "type": "text",
        "url": url,
        "content": page_raw,
    }


if __name__ == "__main__":
    # Test the most basic functionality
    example_url = "https://www.example.com"
    import asyncio

    result = asyncio.run(fetch_url(example_url))
    print(result)


async def web_search(
    query: Annotated[str, "The search query to find information online."],
) -> dict[str, str]:
    """
    Performs a web search using the Brave Search API to find documentation, solutions, or current events.
    """
    import os
    import httpx
    import json
    from conduit.domain.exceptions.exceptions import ToolError

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ToolError("Missing BRAVE_API_KEY environment variable.")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
        "X-Subscription-Token": api_key,
    }
    params = {
        "q": query,
        "count": 5,  # Limit results to save tokens
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers=headers, params=params, timeout=10.0
            )

            if response.status_code == 401:
                raise ToolError("Invalid Brave Search API key.")
            if response.status_code == 429:
                raise ToolError("Rate limit exceeded for Brave Search.")

            response.raise_for_status()
            data = response.json()

        # Parse and format the results for the LLM
        results = []

        # Brave puts results in ['web']['results']
        web_results = data.get("web", {}).get("results", [])

        if not web_results:
            return {"result": "No results found."}

        for item in web_results:
            title = item.get("title", "No Title")
            link = item.get("url", "")
            description = item.get("description", "")
            # Only include pub date if available, helpful for "latest news" queries
            age = item.get("age", "")

            entry = f"Title: {title}\nURL: {link}\nDescription: {description}"
            if age:
                entry += f"\nPublished: {age}"
            results.append(entry)

        return {"result": "\n---\n".join(results)}

    except httpx.TimeoutException:
        return {"error": "Search request timed out."}
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


__all__ = ["fetch_url", "web_search"]
