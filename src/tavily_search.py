"""Optional web search worker using the Tavily Search API."""

import time
from typing import Callable
from urllib.parse import urlsplit

from httpx import Client as HTTPClient


class TavilySearchAgent:
    name = "tavily-search"
    endpoint = "https://api.tavily.com/search"

    def __init__(self, api_key: str, reserve_call: Callable[[], None]):
        self._api_key = api_key.strip()
        if not self._api_key:
            raise ValueError("TAVILY_API_KEY is required for web search.")
        self._reserve_call = reserve_call
        self.calls_made = 0
        self.known_credits = 0

    def search(self, query: str) -> dict:
        query = query.strip()
        if not query:
            raise ValueError("Web search needs a non-empty query.")
        with HTTPClient(timeout=30.0, follow_redirects=False) as client:
            self._reserve_call()
            self.calls_made += 1
            started = time.perf_counter()
            response = client.post(
                self.endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "chunks_per_source": 1,
                    "include_answer": "basic",
                    "include_usage": True,
                },
            )
            latency = time.perf_counter() - started
            response.raise_for_status()
            body = response.json()

        if not isinstance(body, dict) or not isinstance(body.get("results"), list):
            raise ValueError("Tavily returned an invalid search response.")
        usage = body.get("usage")
        credits = usage.get("credits") if isinstance(usage, dict) else None
        if type(credits) is int and credits >= 0:
            self.known_credits += credits
        else:
            credits = None
        sources = []
        for row in body["results"][:5]:
            if not isinstance(row, dict):
                continue
            url = row.get("url")
            if not isinstance(url, str) or any(char.isspace() or not char.isprintable() for char in url):
                continue
            try:
                parts = urlsplit(url)
                if parts.scheme not in ("https", "http") or not parts.hostname or parts.username is not None:
                    continue
            except ValueError:
                continue
            title = " ".join(_clean_text(row.get("title"), 160).split()) or url
            snippet = _clean_text(row.get("content"), 500)
            sources.append(f"{len(sources) + 1}. {title}\n{url}" + (f"\n{snippet}" if snippet else ""))

        answer = _clean_text(body.get("answer"), 4000)
        if sources:
            output = "Sources:\n" + "\n\n".join(sources)
            if answer:
                output = f"{answer}\n\n{output}"
        else:
            output = "No web sources found."
        return {
            "output": output, "model_name": self.name, "success": True,
            "latency": latency, "search_credits": credits, "result_available": bool(sources),
        }


def _clean_text(value: object, limit: int) -> str:
    if not isinstance(value, str):
        return ""
    return "".join(char for char in value[:limit] if char.isprintable() or char in "\n\t").strip()
