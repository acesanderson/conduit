import io
import logging
from typing import Annotated, Any
from collections import defaultdict
from urllib.parse import urlparse
from conduit.domain.exceptions.exceptions import (
    ToolExecutionError,
    ToolConfigurationError,
)

logger = logging.getLogger(__name__)


# --- SESSION POOL ---
class SessionPool:
    """Manages persistent sessions per domain for performance and anti-bot consistency."""

    def __init__(self):
        self._sessions = {}  # domain -> AsyncSession
        self._locks = defaultdict(lambda: None)  # domain -> asyncio.Lock (lazy init)

    async def get_session(self, domain: str):
        """Get or create a persistent session for the given domain."""
        from curl_cffi.requests import AsyncSession
        import asyncio

        # Lazy-initialize lock for this domain
        if self._locks[domain] is None:
            self._locks[domain] = asyncio.Lock()

        async with self._locks[domain]:
            if domain not in self._sessions:
                logger.info(f"Creating new session for {domain}")
                self._sessions[domain] = AsyncSession(impersonate="chrome120")
            return self._sessions[domain]

    async def reset_session(self, domain: str):
        """Force-reset a session (useful after 403/429 blocks)."""
        import asyncio

        if self._locks[domain] is None:
            self._locks[domain] = asyncio.Lock()

        async with self._locks[domain]:
            if domain in self._sessions:
                try:
                    await self._sessions[domain].close()
                except Exception:
                    pass
                del self._sessions[domain]
                logger.info(f"Reset session for {domain}")

    async def cleanup(self):
        """Close all sessions (call on shutdown)."""
        for session in self._sessions.values():
            try:
                await session.close()
            except Exception:
                pass
        self._sessions.clear()


# Global session pool instance
_session_pool = SessionPool()


# CONVERSION FUNCTIONS
def _convert_html_to_md(html_text: str) -> str:
    """Standard HTML denoising using readabilipy and markdownify."""
    try:
        import markdownify
        import readabilipy
    except ImportError as e:
        raise ToolConfigurationError(
            f"Missing required dependency: {e.name}. Install with: pip install markdownify readabilipy"
        )

    ret = readabilipy.simple_json_from_html_string(html_text, use_readability=True)
    if not ret["content"]:
        raise ToolExecutionError(
            "Page failed to be simplified from HTML. The page may be dynamically generated or blocked."
        )

    return markdownify.markdownify(ret["content"], heading_style=markdownify.ATX)


def _convert_binary_to_md(content_bytes: bytes, extension: str) -> str:
    """Uses MarkItDown locally to handle PDFs, Office docs, etc."""
    try:
        from markitdown import MarkItDown
    except ImportError:
        raise ToolConfigurationError(
            "MarkItDown is not installed. Install with: pip install markitdown"
        )

    md = MarkItDown()
    stream = io.BytesIO(content_bytes)

    try:
        result = md.convert_stream(stream, file_extension=extension)
        return result.text_content
    except Exception as e:
        raise ToolExecutionError(
            f"Failed to convert {extension} file: {str(e)}. The file may be encrypted, corrupted, or in an unsupported format."
        )


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
    except ImportError:
        raise ToolConfigurationError(
            "YouTube support requires 'siphon-client' and 'headwater-client' packages. Install them to use this feature."
        )

    try:
        # 1. Build the ephemeral extraction request
        params = SiphonRequestParams(action=ActionType.EXTRACT)
        request = create_siphon_request(source=url, request_params=params)

        # 2. Process via the Headwater/Siphon server
        client = HeadwaterClient()
        response = client.siphon.process(request)

        payload = response.payload
        if isinstance(payload, ContentData):
            return payload.text

        raise ToolExecutionError(
            f"Siphon returned unexpected payload type: {type(payload)}"
        )

    except Exception as e:
        if isinstance(e, (ToolConfigurationError, ToolExecutionError)):
            raise
        raise ToolExecutionError(f"YouTube transcript extraction failed: {str(e)}")


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


async def fetch_url(
    url: Annotated[str, "The URL to fetch"],
    page: Annotated[int, "The page number to view (1-indexed)."] = 1,
) -> dict[str, Any]:
    """
    Fetch a URL and convert it to clean Markdown.
    Supports HTML, PDF, Office documents, and YouTube transcripts.
    Uses persistent sessions per domain for performance and anti-bot consistency.
    """
    import mimetypes
    import asyncio

    # Validate inputs
    if not url or not url.strip():
        raise ToolConfigurationError("URL parameter cannot be empty")

    if page < 1:
        raise ToolConfigurationError(f"Page number must be >= 1, got {page}")

    # 1. Domain Fork: YouTube
    if "youtube.com" in url or "youtu.be" in url:
        logger.info(f"Routing YouTube URL to Siphon: {url}")
        full_md = await _fetch_youtube_via_siphon(url)
        return _paginate_content(full_md, url, page)

    # 2. Extract domain and get persistent session
    try:
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            raise ToolExecutionError(
                f"Invalid URL format: {url}. URLs must include scheme (http:// or https://) and domain."
            )

        domain = parsed.netloc
        base_url = f"{parsed.scheme}://{domain}"
    except Exception as e:
        raise ToolExecutionError(f"Failed to parse URL '{url}': {str(e)}")

    session = await _session_pool.get_session(domain)

    # 3. Fetch with Session Priming + Retry Logic
    max_retries = 3
    response = None
    primed = False

    for attempt in range(max_retries):
        try:
            # Prime the session on first attempt (if not already primed for this session)
            if attempt == 0 and not primed:
                logger.info(f"Priming session for {domain}")
                try:
                    await session.get(base_url, timeout=10)
                    primed = True
                except Exception:
                    pass  # Continue even if priming fails

                # Small delay after priming
                await asyncio.sleep(1)

            # Now fetch the actual URL
            logger.info(f"Attempt {attempt + 1}/{max_retries}: Fetching {url}")
            response = await session.get(url, timeout=30)

            # Check for blocks
            if response.status_code == 403:
                logger.warning(f"Access denied (403) for {url}")
                if attempt < max_retries - 1:
                    await _session_pool.reset_session(domain)
                    session = await _session_pool.get_session(domain)
                    primed = False
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    raise ToolExecutionError(
                        f"Access denied to {url}. The site may require authentication, is blocking bots, or is behind a firewall. Try a different URL or search for alternative sources."
                    )

            if response.status_code == 404:
                raise ToolExecutionError(
                    f"Page not found (404): {url}. The URL may be incorrect or the page may have been removed."
                )

            if response.status_code == 429:
                logger.warning(f"Rate limited (429) for {url}")
                if attempt < max_retries - 1:
                    await _session_pool.reset_session(domain)
                    session = await _session_pool.get_session(domain)
                    primed = False
                    await asyncio.sleep(2**attempt)
                    continue
                else:
                    raise ToolExecutionError(
                        f"Rate limited by {domain}. The site is blocking too many requests. Try again later or use a different source."
                    )

            if response.status_code >= 500:
                raise ToolExecutionError(
                    f"Server error ({response.status_code}): {url}. The website is experiencing technical difficulties. Try again later or use a different source."
                )

            if response.status_code >= 400:
                raise ToolExecutionError(
                    f"HTTP error {response.status_code} for {url}. The request failed for an unknown reason."
                )

            # Success
            break

        except ToolExecutionError:
            raise
        except asyncio.TimeoutError:
            if attempt == max_retries - 1:
                raise ToolExecutionError(
                    f"Request timeout for {url}. The site is too slow or unresponsive. Try a different source."
                )
            logger.warning(f"Timeout on attempt {attempt + 1}")
            await asyncio.sleep(2**attempt)
        except Exception as e:
            if attempt == max_retries - 1:
                raise ToolExecutionError(
                    f"Network error while fetching {url}: {str(e)}. Check your internet connection or try a different URL."
                )
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            await asyncio.sleep(2**attempt)

    if not response:
        raise ToolExecutionError(f"Failed to fetch {url} after {max_retries} attempts")

    # 4. Check content size
    content_length = response.headers.get("content-length")
    if content_length:
        size_bytes = int(content_length)
        size_mb = size_bytes / (1024 * 1024)
        if size_mb > 50:
            raise ToolExecutionError(
                f"Content too large: {size_mb:.1f}MB exceeds 50MB limit. Try a different source or page."
            )

    # 5. MIME Type Dispatcher
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

    except ToolExecutionError:
        raise
    except ToolConfigurationError:
        raise
    except Exception as e:
        raise ToolExecutionError(f"Failed to process content from {url}: {str(e)}")

    return _paginate_content(full_md, url, page)


async def web_search(
    query: Annotated[str, "The search query to find information online."],
) -> dict[str, str]:
    """
    Performs a web search using the Brave Search API.
    """
    import os
    import httpx

    # Validate input
    if not query or not query.strip():
        raise ToolConfigurationError("Search query cannot be empty")

    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise ToolConfigurationError(
            "BRAVE_API_KEY environment variable not set. Configure your API key to use web search."
        )

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

    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            raise ToolConfigurationError(
                "Invalid Brave Search API key. Check your BRAVE_API_KEY configuration."
            )
        elif e.response.status_code == 429:
            raise ToolExecutionError(
                "Brave Search rate limit exceeded. Try again in a few moments."
            )
        else:
            raise ToolExecutionError(
                f"Brave Search API error ({e.response.status_code}): {str(e)}"
            )

    except httpx.TimeoutException:
        raise ToolExecutionError(
            "Search request timed out. Check your internet connection or try again."
        )

    except Exception as e:
        raise ToolExecutionError(f"Search failed: {str(e)}")


__all__ = ["fetch_url", "web_search"]
