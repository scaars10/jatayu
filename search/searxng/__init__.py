"""SearXNG integration for Jatayu."""

from .cache import CacheBackend, InMemoryTTLCache, make_cache_key
from .client import (
    SearxngClient,
    SearxngConfigurationError,
    SearxngError,
    SearxngParsingError,
    SearxngRequestError,
    SearxngResponseError,
)
from .config import SearxngConfig
from .models import (
    NormalizedSearchResult,
    SearchToolBriefResult,
    SearchToolResult,
    SearxngRawResult,
    SearxngSearchRequest,
    SearxngSearchResponse,
)
from .normalize import normalize_result, normalize_results, normalize_search_response
from .tool import SearxngSearchTool, get_default_search_tool, search_web, search_web_brief

__all__ = [
    "CacheBackend",
    "InMemoryTTLCache",
    "NormalizedSearchResult",
    "SearchToolBriefResult",
    "SearchToolResult",
    "SearxngClient",
    "SearxngConfig",
    "SearxngConfigurationError",
    "SearxngError",
    "SearxngParsingError",
    "SearxngRawResult",
    "SearxngRequestError",
    "SearxngResponseError",
    "SearxngSearchRequest",
    "SearxngSearchResponse",
    "SearxngSearchTool",
    "get_default_search_tool",
    "make_cache_key",
    "normalize_result",
    "normalize_results",
    "normalize_search_response",
    "search_web",
    "search_web_brief",
]
