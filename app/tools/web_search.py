"""Web Search Tool: retrieves information from the internet.

Uses DuckDuckGo as the search backend — no API key required.
Returns a list of text snippets suitable for LLM consumption.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def web_search(query: str, max_results: int = 3) -> list[str]:
    """Search the web and return text snippets."""
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))

        snippets: list[str] = []
        for r in results:
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            snippets.append(f"[{title}]({href})\n{body}")

        return snippets

    except Exception as exc:
        logger.warning("Web search failed: %s", exc)
        return [f"Web search unavailable: {exc}"]
