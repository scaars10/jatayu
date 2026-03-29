"""Configuration for the SearXNG integration."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value from {value!r}")


def _parse_list(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


class SearxngConfig(BaseModel):
    """Runtime configuration for a SearXNG-backed search provider."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    base_url: str = Field(default="http://localhost:8080")
    timeout_seconds: float = Field(default=10.0, gt=0)
    max_results: int = Field(default=8, ge=1, le=50)
    safe_search: int = Field(default=1, ge=0, le=2)
    language: str | None = Field(default="en")
    categories: tuple[str, ...] = ()
    engines: tuple[str, ...] = ()
    default_format: Literal["json"] = "json"
    user_agent: str = Field(default="jatayu-searxng/1.0")
    retry_count: int = Field(default=2, ge=0, le=10)
    backoff_seconds: float = Field(default=0.5, ge=0)
    verify_ssl: bool = True
    cache_ttl_seconds: int = Field(default=60, ge=0)
    enabled: bool = True

    @field_validator("base_url")
    @classmethod
    def _normalize_base_url(cls, value: str) -> str:
        normalized = value.strip().rstrip("/")
        if not normalized:
            raise ValueError("base_url must not be empty")
        return normalized

    @field_validator("language")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> Self:
        """Load configuration from process environment variables."""

        source = os.environ if env is None else env
        data: dict[str, object] = {}

        aliases = {
            "base_url": ("SEARXNG_BASE_URL", "JATAYU_SEARXNG_BASE_URL"),
            "timeout_seconds": (
                "SEARXNG_TIMEOUT_SECONDS",
                "JATAYU_SEARXNG_TIMEOUT_SECONDS",
            ),
            "max_results": ("SEARXNG_MAX_RESULTS", "JATAYU_SEARXNG_MAX_RESULTS"),
            "safe_search": ("SEARXNG_SAFE_SEARCH", "JATAYU_SEARXNG_SAFE_SEARCH"),
            "language": ("SEARXNG_LANGUAGE", "JATAYU_SEARXNG_LANGUAGE"),
            "categories": ("SEARXNG_CATEGORIES", "JATAYU_SEARXNG_CATEGORIES"),
            "engines": ("SEARXNG_ENGINES", "JATAYU_SEARXNG_ENGINES"),
            "default_format": (
                "SEARXNG_DEFAULT_FORMAT",
                "JATAYU_SEARXNG_DEFAULT_FORMAT",
            ),
            "user_agent": ("SEARXNG_USER_AGENT", "JATAYU_SEARXNG_USER_AGENT"),
            "retry_count": ("SEARXNG_RETRY_COUNT", "JATAYU_SEARXNG_RETRY_COUNT"),
            "backoff_seconds": (
                "SEARXNG_BACKOFF_SECONDS",
                "JATAYU_SEARXNG_BACKOFF_SECONDS",
            ),
            "verify_ssl": ("SEARXNG_VERIFY_SSL", "JATAYU_SEARXNG_VERIFY_SSL"),
            "cache_ttl_seconds": (
                "SEARXNG_CACHE_TTL_SECONDS",
                "JATAYU_SEARXNG_CACHE_TTL_SECONDS",
            ),
            "enabled": ("SEARXNG_ENABLED", "JATAYU_SEARXNG_ENABLED"),
        }

        for field_name, keys in aliases.items():
            raw_value = next(
                (
                    source[key]
                    for key in keys
                    if key in source and source[key].strip() != ""
                ),
                None,
            )
            if raw_value is None:
                continue

            if field_name in {"max_results", "safe_search", "retry_count", "cache_ttl_seconds"}:
                data[field_name] = int(raw_value)
            elif field_name in {"timeout_seconds", "backoff_seconds"}:
                data[field_name] = float(raw_value)
            elif field_name in {"verify_ssl", "enabled"}:
                data[field_name] = _parse_bool(raw_value)
            elif field_name in {"categories", "engines"}:
                data[field_name] = _parse_list(raw_value)
            else:
                data[field_name] = raw_value

        return cls(**data)
