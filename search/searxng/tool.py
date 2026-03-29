"""Agent-facing tool wrapper for the SearXNG client."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from .cache import CacheBackend, InMemoryTTLCache, make_cache_key
from .client import SearxngClient
from .config import SearxngConfig
from .models import SearchToolBriefResult, SearchToolResult, SearxngSearchRequest
from .normalize import normalize_search_response


class SearxngSearchTool:
    """High-level SearXNG-backed web-search tool for agent runtimes."""

    def __init__(
        self,
        config: SearxngConfig,
        *,
        client: SearxngClient | None = None,
        cache: CacheBackend[SearchToolResult] | None = None,
        logger: logging.Logger | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        self._owns_client = client is None
        self.client = client or SearxngClient(config=config, logger=self._logger)
        self.cache = cache if cache is not None else InMemoryTTLCache()

    @classmethod
    def from_config(cls, config: SearxngConfig | None = None) -> "SearxngSearchTool":
        """Create a tool from an explicit config or environment defaults."""

        resolved = config or SearxngConfig.from_env()
        return cls(config=resolved)

    def close(self) -> None:
        """Close owned resources."""

        if self._owns_client:
            self.client.close()

    def search_web(
        self,
        query: str,
        *,
        max_results: int = 5,
        categories: list[str] | tuple[str, ...] | None = None,
        engines: list[str] | tuple[str, ...] | None = None,
        language: str | None = None,
        page: int = 1,
        use_cache: bool = True,
    ) -> SearchToolResult:
        """Search the web and return compact normalized structured results."""

        request = SearxngSearchRequest(
            query=query,
            page=page,
            max_results=min(max_results, self.config.max_results),
            language=language if language is not None else self.config.language,
            categories=categories if categories is not None else self.config.categories,
            engines=engines if engines is not None else self.config.engines,
            safe_search=self.config.safe_search,
            output_format=self.config.default_format,
        )

        cache_key = make_cache_key("searxng.search_web", request.cache_identity())
        if use_cache and self.config.cache_ttl_seconds > 0:
            cached = self.cache.get(cache_key)
            if cached is not None:
                self._log_event("cache_hit", cache_key=cache_key)
                return cached.model_copy(update={"cached": True})
            self._log_event("cache_miss", cache_key=cache_key)

        response = self.client.search(request)
        normalized = normalize_search_response(response)
        truncated = normalized[: request.max_results]
        warnings = tuple(str(item) for item in response.unresponsive_engines)

        result = SearchToolResult(
            query=request.query,
            page=request.page,
            total_results=len(truncated),
            provider_total_results=response.number_of_results,
            language=request.language,
            categories=request.categories,
            engines=request.engines,
            cached=False,
            warnings=warnings,
            results=truncated,
        )

        if use_cache and self.config.cache_ttl_seconds > 0:
            self.cache.set(cache_key, result, ttl_seconds=self.config.cache_ttl_seconds)

        return result

    def search_web_brief(
        self,
        query: str,
        *,
        max_results: int = 5,
        categories: list[str] | tuple[str, ...] | None = None,
        engines: list[str] | tuple[str, ...] | None = None,
        language: str | None = None,
        page: int = 1,
        use_cache: bool = True,
    ) -> SearchToolBriefResult:
        """Return a compact text summary plus the full structured result."""

        result = self.search_web(
            query,
            max_results=max_results,
            categories=categories,
            engines=engines,
            language=language,
            page=page,
            use_cache=use_cache,
        )
        summary = _build_brief_summary(result)
        return SearchToolBriefResult(summary=summary, result=result)

    def _log_event(self, event: str, **fields: Any) -> None:
        self._logger.info(
            "searxng.%s %s",
            event,
            json.dumps(fields, sort_keys=True, default=str),
        )


def _build_brief_summary(result: SearchToolResult) -> str:
    if not result.results:
        return f"No SearXNG results found for '{result.query}'."

    lines = [f"Top {len(result.results)} SearXNG results for '{result.query}':"]
    for index, item in enumerate(result.results, start=1):
        snippet = item.snippet or "No snippet available."
        if len(snippet) > 140:
            snippet = f"{snippet[:137]}..."
        lines.append(f"{index}. {item.title} | {item.url} | {snippet}")
    return "\n".join(lines)


@lru_cache(maxsize=1)
def get_default_search_tool() -> SearxngSearchTool:
    """Return a process-local default search tool built from environment config."""

    return SearxngSearchTool.from_config(SearxngConfig.from_env())


def search_web(
    query: str,
    *,
    max_results: int = 5,
    categories: list[str] | tuple[str, ...] | None = None,
    engines: list[str] | tuple[str, ...] | None = None,
    language: str | None = None,
    page: int = 1,
    use_cache: bool = True,
    tool: SearxngSearchTool | None = None,
) -> SearchToolResult:
    """Convenience wrapper around the default or provided SearXNG tool."""

    active_tool = tool or get_default_search_tool()
    return active_tool.search_web(
        query,
        max_results=max_results,
        categories=categories,
        engines=engines,
        language=language,
        page=page,
        use_cache=use_cache,
    )


def search_web_brief(
    query: str,
    *,
    max_results: int = 5,
    categories: list[str] | tuple[str, ...] | None = None,
    engines: list[str] | tuple[str, ...] | None = None,
    language: str | None = None,
    page: int = 1,
    use_cache: bool = True,
    tool: SearxngSearchTool | None = None,
) -> SearchToolBriefResult:
    """Convenience wrapper returning a concise LLM-friendly summary."""

    active_tool = tool or get_default_search_tool()
    return active_tool.search_web_brief(
        query,
        max_results=max_results,
        categories=categories,
        engines=engines,
        language=language,
        page=page,
        use_cache=use_cache,
    )
