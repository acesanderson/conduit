"""
Registry of Ollama inference servers and utilities for fetching their model lists.

Three servers:
  deepwater  — AlphaBlue  heavy inference     (transport host_alias: "headwater")
  bywater    — Caruana    daily driver        (transport host_alias: "bywater")
  backwater  — Cheet      embeddings / light  (transport host_alias: "backwater")

Model lists are fetched from each server's HeadwaterServer instance via
GET /v1/models (port 8080). Ollama itself is not exposed externally.

The cache file (OLLAMA_MODELS_PATH) stores per-server entries:
    {
        "deepwater": {"models": [...], "updated": "2026-03-30T14:23:00"},
        "bywater":   {"models": [...], "updated": "2026-03-29T09:11:00"},
        "backwater": {"models": [...], "updated": "2026-03-28T17:45:00"}
    }
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Conduit server name → HeadwaterAsyncTransport host_alias
OLLAMA_SERVERS: dict[str, str] = {
    "deepwater": "headwater",   # AlphaBlue
    "bywater": "bywater",       # Caruana
    "backwater": "backwater",   # Cheet
}

HEADWATER_PORT = 8080
FETCH_TIMEOUT = 5.0


class ServerModelResult(NamedTuple):
    server: str
    models: list[str]
    from_cache: bool
    cached_at: str | None  # ISO timestamp, or None if live


def _models_url(host_alias: str) -> str:
    from dbclients.discovery.host import get_network_context

    ctx = get_network_context()
    match host_alias:
        case "headwater":
            ip = ctx.headwater_server
        case "bywater":
            ip = ctx.bywater_server
        case "backwater":
            ip = ctx.backwater_server
        case _:
            raise ValueError(f"Unknown host_alias: {host_alias!r}")
    return f"http://{ip}:{HEADWATER_PORT}/v1/models"


async def fetch_server_models(server_name: str) -> list[str]:
    """Fetch model list from a server's HeadwaterServer /v1/models endpoint."""
    import httpx

    host_alias = OLLAMA_SERVERS[server_name]
    url = _models_url(host_alias)
    async with httpx.AsyncClient(timeout=FETCH_TIMEOUT) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()
    return [item["id"] for item in data.get("data", [])]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def read_cache(cache_path: Path) -> dict:
    """Read the ollama_models.json cache. Returns {} on any error."""
    if not cache_path.exists():
        return {}
    try:
        with open(cache_path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def write_server_to_cache(
    cache_path: Path, server_name: str, models: list[str]
) -> None:
    """Update a single server's entry in the cache file."""
    cache = read_cache(cache_path)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    cache[server_name] = {"models": models, "updated": now}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(cache, f, indent=2)


def get_cached_server_models(
    cache_path: Path, server_name: str
) -> tuple[list[str], str | None]:
    """Returns (model_list, iso_timestamp) or ([], None) if not cached."""
    entry = read_cache(cache_path).get(server_name)
    if not entry or not isinstance(entry, dict):
        return [], None
    return entry.get("models", []), entry.get("updated")


def all_cached_models(cache_path: Path) -> list[str]:
    """
    Return a flat, deduplicated list of all Ollama models across all servers.

    Handles both formats:
      - New: {"deepwater": {"models": [...], "updated": "..."}, ...}
      - Legacy: {"ollama": [...]}
    """
    cache = read_cache(cache_path)
    seen: set[str] = set()
    result: list[str] = []
    for key, value in cache.items():
        if key == "ollama" and isinstance(value, list):
            # legacy format
            models = value
        elif isinstance(value, dict):
            models = value.get("models", [])
        else:
            continue
        for m in models:
            if m not in seen:
                seen.add(m)
                result.append(m)
    return result


# ---------------------------------------------------------------------------
# Fetch with cache fallback
# ---------------------------------------------------------------------------

async def fetch_with_cache_fallback(
    server_name: str, cache_path: Path
) -> ServerModelResult:
    """Try live fetch; fall back to cache on any failure."""
    try:
        models = await fetch_server_models(server_name)
        return ServerModelResult(
            server=server_name, models=models, from_cache=False, cached_at=None
        )
    except Exception as exc:
        logger.warning("Failed to fetch models from %s: %s", server_name, exc)
        cached_models, cached_at = get_cached_server_models(cache_path, server_name)
        return ServerModelResult(
            server=server_name,
            models=cached_models,
            from_cache=True,
            cached_at=cached_at,
        )


async def fetch_all_servers(cache_path: Path) -> list[ServerModelResult]:
    """Fetch all servers concurrently."""
    return list(
        await asyncio.gather(
            *[
                fetch_with_cache_fallback(server, cache_path)
                for server in OLLAMA_SERVERS
            ]
        )
    )
