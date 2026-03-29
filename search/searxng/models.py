"""Typed request, response, and normalized models for SearXNG."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator


def _normalize_text(value: str) -> str:
    return " ".join(value.split())


def _tuple_or_empty(values: Sequence[str] | None) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(item.strip() for item in values if item and item.strip())


class SearxngSearchRequest(BaseModel):
    """Typed search options accepted by the SearXNG client."""

    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=1)
    page: int = Field(default=1, ge=1)
    max_results: int = Field(default=5, ge=1, le=50)
    language: str | None = None
    categories: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    safe_search: int | None = Field(default=None, ge=0, le=2)
    output_format: Literal["json"] = "json"

    @field_validator("query")
    @classmethod
    def _validate_query(cls, value: str) -> str:
        normalized = _normalize_text(value)
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @field_validator("categories", "engines", mode="before")
    @classmethod
    def _normalize_sequences(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return _tuple_or_empty(value.split(","))
        return _tuple_or_empty(value)

    def to_http_params(self) -> dict[str, str | int]:
        """Convert the request into SearXNG query parameters."""

        params: dict[str, str | int] = {
            "q": self.query,
            "format": self.output_format,
            "pageno": self.page,
        }
        if self.language:
            params["language"] = self.language
        if self.categories:
            params["categories"] = ",".join(self.categories)
        if self.engines:
            params["engines"] = ",".join(self.engines)
        if self.safe_search is not None:
            params["safesearch"] = self.safe_search
        return params

    def cache_identity(self) -> dict[str, Any]:
        """Return a normalized payload suitable for cache-key generation."""

        return {
            "query": self.query,
            "page": self.page,
            "max_results": self.max_results,
            "language": self.language,
            "categories": sorted(self.categories),
            "engines": sorted(self.engines),
            "safe_search": self.safe_search,
            "output_format": self.output_format,
        }


class SearxngRawResult(BaseModel):
    """Permissive near-raw result model for variable engine payloads."""

    model_config = ConfigDict(frozen=True, extra="allow")

    title: str | None = None
    url: str | None = None
    content: str | None = None
    engine: str | None = None
    engines: tuple[str, ...] = ()
    category: str | None = None
    score: float | None = None
    favicon: str | None = Field(
        default=None,
        validation_alias=AliasChoices("favicon", "favicon_url"),
    )
    thumbnail: str | None = Field(
        default=None,
        validation_alias=AliasChoices("thumbnail", "thumbnail_src", "img_src"),
    )
    source: str | None = None
    published_date_raw: str | int | float | None = Field(
        default=None,
        validation_alias=AliasChoices("publishedDate", "published_date", "date"),
    )

    @field_validator("engines", mode="before")
    @classmethod
    def _normalize_engines(
        cls,
        value: Sequence[str] | str | None,
    ) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            return _tuple_or_empty(value.split(","))
        return _tuple_or_empty(value)


class SearxngSearchResponse(BaseModel):
    """Typed response returned by the SearXNG client."""

    model_config = ConfigDict(frozen=True, extra="allow")

    query: str | None = None
    number_of_results: int | None = None
    results: tuple[SearxngRawResult, ...] = ()
    answers: tuple[Any, ...] = ()
    corrections: tuple[Any, ...] = ()
    infoboxes: tuple[Any, ...] = ()
    suggestions: tuple[str, ...] = ()
    unresponsive_engines: tuple[Any, ...] = ()


class NormalizedSearchResult(BaseModel):
    """Stable internal result schema for agent consumption."""

    model_config = ConfigDict(frozen=True)

    title: str
    url: str
    snippet: str | None = None
    engine: str | None = None
    category: str | None = None
    score: float | None = None
    published_date: datetime | None = None
    source: str | None = None
    favicon: str | None = None
    thumbnail: str | None = None
    domain: str | None = None
    position: int | None = None


class SearchToolResult(BaseModel):
    """Compact structured output for the agent runtime."""

    model_config = ConfigDict(frozen=True)

    query: str
    provider: Literal["searxng"] = "searxng"
    page: int = 1
    total_results: int
    provider_total_results: int | None = None
    language: str | None = None
    categories: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    cached: bool = False
    warnings: tuple[str, ...] = ()
    results: tuple[NormalizedSearchResult, ...] = ()


class SearchToolBriefResult(BaseModel):
    """LLM-oriented summary with the full structured result attached."""

    model_config = ConfigDict(frozen=True)

    summary: str
    result: SearchToolResult
