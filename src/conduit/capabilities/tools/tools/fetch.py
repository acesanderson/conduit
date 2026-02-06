import io
import logging
from typing import Annotated, Any
from conduit.domain.exceptions.exceptions import ToolError
from functools import lru_cache

logger = logging.getLogger(__name__)


# CONVERSION FUNCTIONS
def _convert_html_to_md(html_text: str) -> str:
    """Standard HTML denoising using readabilipy and markdownify."""
    import markdownify
    import readabilipy

    ret = readabilipy.simple_json_from_html_string(html_text, use_readability=True)
    if not ret["content"]:
        return "<error>Page failed to be simplified from HTML</error>"

    return markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)


def _convert_binary_to_md(content_bytes: bytes, extension: str) -> str:
    """Uses MarkItDown locally to handle PDFs, Office docs, etc."""
    from markitdown import MarkItDown

    md = MarkItDown()
    stream = io.BytesIO(content_bytes)

    # We provide the extension hint to help MarkItDown select the right parser
    try:
        result = md.convert_stream(stream, file_extension=extension)
        return result.text_content
    except Exception as e:
        return f"<error>MarkItDown failed to convert {extension} file: {str(e)}</error>"


async def _fetch_youtube_via_siphon(url: str) -> str:
    """
    Lazy loads Siphon/Headwater client to extract YouTube transcripts
    via the remote Siphon server (Stallion).
    """
    try:
        from headwater_client.client.headwater_client import HeadwaterClient
        from siphon_api.api.siphon_request import SiphonRequestParams
        from siphon_api.api.to_siphon_request import create_siphon_request
        from siphon_api.enums import ActionType
        from siphon_api.models import ContentData

        # 1. Build the ephemeral extraction request
        params = SiphonRequestParams(action=ActionType.EXTRACT)
        request = create_siphon_request(source=url, request_params=params)

        # 2. Process via the Headwater/Siphon server
        client = HeadwaterClient()
        response = client.siphon.process(request)

        payload = response.payload
        if isinstance(payload, ContentData):
            return payload.text
        return (
            f"<error>Siphon returned unexpected payload type: {type(payload)}</error>"
        )

    except ImportError:
        return (
            "<error>YouTube support requires 'siphon-client' and 'headwater-client' "
            "to be installed. Please install them to use this feature.</error>"
        )
    except Exception as e:
        return f"<error>Siphon YouTube extraction failed: {str(e)}</error>"


def _paginate_content(full_content: str, url: str, page: int) -> dict[str, Any]:
    """Unified pagination and TOC logic for any markdown content."""
    lines = full_content.splitlines()

    # Generate Table of Contents (Map)
    toc = [
        {"text": line, "line": i + 1}
        for i, line in enumerate(lines)
        if line.strip().startswith("#")
    ]

    # Viewport Slicing (~2000 tokens)
    chars_per_page = 8000
    total_chars = len(full_content)
    total_pages = (total_chars // chars_per_page) + 1

    start_idx = (page - 1) * chars_per_page
    end_idx = start_idx + chars_per_page
    viewport_text = full_content[start_idx:end_idx]

    if start_idx >= total_chars and total_chars > 0:
        return {
            "error": f"Page {page} out of bounds. Total pages: {total_pages}",
            "url": url,
        }

    return {
        "metadata": {
            "url": url,
            "current_page": page,
            "total_pages": total_pages,
            "total_characters": total_chars,
            "is_truncated": page < total_pages,
        },
        "table_of_contents": toc,
        "content": viewport_text,
        "instructions": (
            f"Showing page {page} of {total_pages}. "
            "Use the TOC to jump to sections using the 'page' parameter."
        ),
    }


# --- MAIN TOOLS ---


@lru_cache
async def fetch_url(
    url: Annotated[str, "The URL to fetch"],
    page: Annotated[int, "The page number to view (1-indexed)."] = 1,
) -> dict[str, Any]:
    """
    Fetch a URL and convert it to clean Markdown.
    Supports HTML, PDF, Office documents, and YouTube transcripts.
    """
    from curl_cffi.requests import AsyncSession
    from urllib.parse import urlparse
    import mimetypes
    import asyncio

    # 1. Domain Fork: YouTube
    if "youtube.com" in url or "youtu.be" in url:
        logger.info(f"Routing YouTube URL to Siphon: {url}")
        full_md = await _fetch_youtube_via_siphon(url)
        return _paginate_content(full_md, url, page)

    # 2. Standard Fetch with Session Priming + Retry Logic
    max_retries = 3
    response = None

    async with AsyncSession(impersonate="chrome120") as session:
        for attempt in range(max_retries):
            try:
                # Prime the session by visiting the homepage first
                if attempt == 0:
                    parsed = urlparse(url)
                    base_url = f"{parsed.scheme}://{parsed.netloc}"
                    logger.info(f"Priming session with {base_url}")
                    try:
                        await session.get(base_url, timeout=10)
                    except Exception:
                        pass  # Continue even if priming fails

                    # Small delay after priming
                    await asyncio.sleep(1)

                # Now fetch the actual URL
                logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching {url}")
                response = await session.get(url, timeout=30)

                # Check for blocks
                if response.status_code in [403, 429]:
                    logger.warning(f"Blocked with {response.status_code}, retrying...")
                    await asyncio.sleep(2**attempt)  # Exponential backoff
                    continue

                if response.status_code >= 400:
                    raise Exception(f"HTTP {response.status_code}")

                # Success
                break

            except Exception as e:
                if attempt == max_retries - 1:
                    raise ToolError(
                        f"Failed to fetch {url} after {max_retries} attempts: {e}"
                    )
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                await asyncio.sleep(2**attempt)

        if not response or response.status_code >= 400:
            raise ToolError(
                f"Failed to fetch {url}: HTTP {response.status_code if response else 'No response'}"
            )

    # 3. MIME Type Dispatcher
    content_type = response.headers.get("content-type", "").lower().split(";")[0]
    extension = mimetypes.guess_extension(content_type) or ""

    try:
        match content_type:
            case "text/html" | "application/xhtml+xml":
                full_md = _convert_html_to_md(response.text)

            case (
                "application/pdf"
                | "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                | "application/vnd.openxmlformats-officedocument.presentationml.presentation"
                | "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                | "application/rtf"
            ):
                logger.info(
                    f"Processing binary document ({content_type}) via MarkItDown"
                )
                full_md = _convert_binary_to_md(response.content, extension)

            case "application/json":
                full_md = f"```json\n{response.text}\n```"

            case _:
                # Sniff for HTML if MIME is missing/generic
                if "<html" in response.text[:100].lower():
                    full_md = _convert_html_to_md(response.text)
                else:
                    full_md = response.text

    except Exception as e:
        raise ToolError(f"Conversion error for {url}: {str(e)}")

    return _paginate_content(full_md, url, page)


async def web_search(
    query: Annotated[str, "The search query to find information online."],
) -> dict[str, str]:
    """
    Performs a web search using the Brave Search API.
    """
    import os
    import httpx

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ToolError("Missing BRAVE_API_KEY environment variable.")

    url = "https://api.search.brave.com/res/v1/web/search"
    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }
    params = {"q": query, "count": 5}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url, headers=headers, params=params, timeout=10.0
            )
            response.raise_for_status()
            data = response.json()

        results = []
        web_results = data.get("web", {}).get("results", [])

        if not web_results:
            return {"result": "No results found."}

        for i, item in enumerate(web_results, 1):
            title = item.get("title", "No Title")
            link = item.get("url", "")
            description = item.get("description", "")
            results.append(
                f"[{i}] Title: {title}\n    URL: {link}\n    Snippet: {description}"
            )

        return {
            "result": "\n---\n".join(results),
            "next_step_hint": "Use fetch_url to see full page content or transcripts.",
        }
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


__all__ = ["fetch_url", "web_search"]
