"""Normalization helpers for variable SearXNG result payloads."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from .models import NormalizedSearchResult, SearxngRawResult, SearxngSearchResponse


def _collapse_whitespace(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.split())
    return normalized or None


def _parse_datetime(value: str | int | float | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)

    raw = value.strip()
    if not raw:
        return None

    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError:
        pass

    try:
        parsed = parsedate_to_datetime(raw)
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except (TypeError, ValueError, IndexError):
        pass

    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            parsed = datetime.strptime(raw, fmt)
            return parsed.replace(tzinfo=UTC)
        except ValueError:
            continue

    return None


def canonicalize_url(url: str) -> str:
    """Canonicalize a URL enough for deterministic deduplication."""

    parts = urlsplit(url.strip())
    if not parts.scheme or not parts.netloc:
        return url.strip()

    hostname = parts.hostname.lower() if parts.hostname else ""
    port = parts.port
    if port and (
        (parts.scheme == "http" and port != 80)
        or (parts.scheme == "https" and port != 443)
    ):
        netloc = f"{hostname}:{port}"
    else:
        netloc = hostname

    path = parts.path or "/"
    if path != "/" and path.endswith("/"):
        path = path.rstrip("/")

    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, ""))


def _first_extra(raw: SearxngRawResult, *names: str) -> Any:
    extra = raw.model_extra or {}
    for name in names:
        if name in extra and extra[name] not in (None, ""):
            return extra[name]
    return None


def normalize_result(
    raw: SearxngRawResult,
    *,
    position: int | None = None,
) -> NormalizedSearchResult | None:
    """Normalize one raw SearXNG result into a stable schema."""

    url = _collapse_whitespace(raw.url)
    if not url:
        return None

    canonical_url = canonicalize_url(url)
    domain = urlsplit(canonical_url).hostname
    title = (
        _collapse_whitespace(raw.title)
        or _collapse_whitespace(_first_extra(raw, "pretty_url"))
        or canonical_url
    )
    snippet = _collapse_whitespace(raw.content or _first_extra(raw, "snippet", "description"))
    engine = _collapse_whitespace(raw.engine) or next(iter(raw.engines), None)
    category = _collapse_whitespace(raw.category)
    source = _collapse_whitespace(raw.source) or domain
    favicon = _collapse_whitespace(raw.favicon or _first_extra(raw, "favicon_src"))
    thumbnail = _collapse_whitespace(raw.thumbnail or _first_extra(raw, "thumbnail_src", "img_src"))
    published_date = _parse_datetime(raw.published_date_raw)

    return NormalizedSearchResult(
        title=title,
        url=canonical_url,
        snippet=snippet,
        engine=engine,
        category=category,
        score=raw.score,
        published_date=published_date,
        source=source,
        favicon=favicon,
        thumbnail=thumbnail,
        domain=domain,
        position=position,
    )


def normalize_results(raw_results: Iterable[SearxngRawResult]) -> tuple[NormalizedSearchResult, ...]:
    """Normalize and deduplicate a sequence of raw results."""

    normalized: list[NormalizedSearchResult] = []
    seen_urls: set[str] = set()

    for raw in raw_results:
        item = normalize_result(raw, position=len(normalized) + 1)
        if item is None:
            continue
        if item.url in seen_urls:
            continue

        seen_urls.add(item.url)
        normalized.append(item)

    return tuple(normalized)


def normalize_search_response(
    response: SearxngSearchResponse,
) -> tuple[NormalizedSearchResult, ...]:
    """Normalize all response results into the agent-facing schema."""

    return normalize_results(response.results)
