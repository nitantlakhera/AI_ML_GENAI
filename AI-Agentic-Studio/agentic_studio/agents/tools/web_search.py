"""Web search with three interchangeable backends.

`offline` searches the local corpus instead of the internet, so agent demos and
tests behave identically without network access or an API key.
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

from agentic_studio.agents.tools.registry import tool
from agentic_studio.observability.logs import get_logger
from agentic_studio.settings import get_settings

logger = get_logger("tools.web_search")


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}


def search_tavily(query: str, max_results: int, api_key: str, timeout: float = 15.0) -> list[SearchResult]:
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
        "search_depth": "basic",
        "include_answer": False,
    }
    request = urllib.request.Request(
        "https://api.tavily.com/search",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return [
        SearchResult(
            title=item.get("title", ""),
            url=item.get("url", ""),
            snippet=(item.get("content") or "")[:500],
        )
        for item in data.get("results", [])[:max_results]
    ]


def search_duckduckgo(query: str, max_results: int, timeout: float = 15.0) -> list[SearchResult]:
    """DuckDuckGo instant-answer API: no key, but only returns topic summaries."""
    url = "https://api.duckduckgo.com/?" + urllib.parse.urlencode(
        {"q": query, "format": "json", "no_html": 1, "skip_disambig": 1}
    )
    request = urllib.request.Request(url, headers={"User-Agent": "ai-agentic-studio/1.0"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))

    results: list[SearchResult] = []
    if data.get("AbstractText"):
        results.append(
            SearchResult(
                title=data.get("Heading") or query,
                url=data.get("AbstractURL", ""),
                snippet=data["AbstractText"][:500],
            )
        )
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if topic.get("Text"):
            results.append(
                SearchResult(
                    title=topic.get("Text", "")[:80],
                    url=topic.get("FirstURL", ""),
                    snippet=topic.get("Text", "")[:500],
                )
            )
    return results[:max_results]


def search_offline(query: str, max_results: int) -> list[SearchResult]:
    """Search the ingested corpus; the honest fallback when there is no network."""
    try:
        from agentic_studio.rag.pipeline import get_pipeline

        hits, _ = get_pipeline().retrieve(query, top_k=max_results)
    except Exception as exc:
        logger.warning("offline search failed: %s", exc)
        return []
    return [
        SearchResult(
            title=str(hit.chunk.metadata.get("title") or hit.chunk.source),
            url=f"local://{hit.chunk.source}",
            snippet=hit.text[:500],
        )
        for hit in hits
    ]


def run_search(query: str, max_results: int = 5, provider: str | None = None) -> dict[str, Any]:
    settings = get_settings().tools
    provider = (provider or settings.search_provider).lower()

    try:
        if provider == "tavily" and settings.tavily_api_key:
            results = search_tavily(query, max_results, settings.tavily_api_key)
        elif provider == "duckduckgo":
            results = search_duckduckgo(query, max_results)
        else:
            provider = "offline"
            results = search_offline(query, max_results)
    except Exception as exc:
        logger.warning("%s search failed (%s); falling back to offline corpus", provider, exc)
        provider = "offline"
        results = search_offline(query, max_results)

    return {
        "provider": provider,
        "query": query,
        "count": len(results),
        "results": [result.to_dict() for result in results],
    }


@tool(name="web_search", tags=("research", "network"))
def web_search(query: str, max_results: int = 5) -> dict[str, Any]:
    """Search the web (or the local corpus when offline) and return ranked snippets.

    Args:
        query: What to search for, phrased as a search query.
        max_results: How many results to return, 1-10.
    """
    return run_search(query, max_results=max(1, min(int(max_results), 10)))
