"""Typed SearXNG HTTP client."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from hashlib import sha1
from typing import Any

import httpx
from pydantic import ValidationError

from .config import SearxngConfig
from .models import SearxngSearchRequest, SearxngSearchResponse


class SearxngError(RuntimeError):
    """Base class for SearXNG integration failures."""


class SearxngConfigurationError(SearxngError):
    """Raised when the client is disabled or misconfigured."""


class SearxngRequestError(SearxngError):
    """Raised when a request cannot be completed."""


class SearxngResponseError(SearxngError):
    """Raised when SearXNG returns an unexpected HTTP response."""


class SearxngParsingError(SearxngError):
    """Raised when SearXNG returns malformed JSON or unexpected payloads."""


class SearxngClient:
    """Synchronous typed client for a self-hosted SearXNG instance."""

    _RETRYABLE_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

    def __init__(
        self,
        config: SearxngConfig,
        *,
        http_client: httpx.Client | None = None,
        logger: logging.Logger | None = None,
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self._logger = logger or logging.getLogger(__name__)
        self._sleep = sleep_fn or time.sleep
        self._owns_client = http_client is None
        self._client = http_client or httpx.Client(
            base_url=self.config.base_url,
            headers={"User-Agent": self.config.user_agent},
            timeout=self.config.timeout_seconds,
            verify=self.config.verify_ssl,
        )

    def close(self) -> None:
        """Close the underlying HTTP client if owned by this instance."""

        if self._owns_client:
            self._client.close()

    def __enter__(self) -> "SearxngClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def search(
        self,
        request: SearxngSearchRequest | str,
        *,
        page: int = 1,
        max_results: int = 5,
        language: str | None = None,
        categories: list[str] | tuple[str, ...] | None = None,
        engines: list[str] | tuple[str, ...] | None = None,
        safe_search: int | None = None,
    ) -> SearxngSearchResponse:
        """Execute a search request and return a typed parsed response."""

        if not self.config.enabled:
            raise SearxngConfigurationError("SearXNG search is disabled in configuration")

        if isinstance(request, str):
            request_model = SearxngSearchRequest(
                query=request,
                page=page,
                max_results=max_results,
                language=language if language is not None else self.config.language,
                categories=categories if categories is not None else self.config.categories,
                engines=engines if engines is not None else self.config.engines,
                safe_search=safe_search if safe_search is not None else self.config.safe_search,
                output_format=self.config.default_format,
            )
        else:
            request_model = request

        params = request_model.to_http_params()
        query_hash = sha1(request_model.query.encode("utf-8")).hexdigest()[:12]
        attempts = self.config.retry_count + 1

        for attempt in range(1, attempts + 1):
            started = time.monotonic()
            self._log_event(
                logging.INFO,
                "request_started",
                attempt=attempt,
                endpoint="/search",
                page=request_model.page,
                query_hash=query_hash,
                categories=list(request_model.categories),
                engines=list(request_model.engines),
            )
            try:
                response = self._client.get("/search", params=params)
            except httpx.RequestError as exc:
                duration_ms = round((time.monotonic() - started) * 1000, 2)
                self._log_event(
                    logging.WARNING,
                    "request_failed",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error_type=exc.__class__.__name__,
                    retryable=self._is_retryable_exception(exc),
                )
                if self._is_retryable_exception(exc) and attempt < attempts:
                    self._sleep_before_retry(attempt, error_type=exc.__class__.__name__)
                    continue
                raise SearxngRequestError("Failed to contact SearXNG") from exc

            duration_ms = round((time.monotonic() - started) * 1000, 2)
            if response.status_code != 200:
                self._log_event(
                    logging.WARNING,
                    "request_completed",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    status_code=response.status_code,
                )
                if response.status_code in self._RETRYABLE_STATUS_CODES and attempt < attempts:
                    self._sleep_before_retry(attempt, status_code=response.status_code)
                    continue
                raise SearxngResponseError(
                    f"SearXNG returned HTTP {response.status_code}"
                )

            try:
                payload = response.json()
            except json.JSONDecodeError as exc:
                self._log_event(
                    logging.ERROR,
                    "request_failed",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error_type=exc.__class__.__name__,
                    reason="invalid_json",
                )
                raise SearxngParsingError("SearXNG returned invalid JSON") from exc

            try:
                parsed = SearxngSearchResponse.model_validate(payload)
            except ValidationError as exc:
                self._log_event(
                    logging.ERROR,
                    "request_failed",
                    attempt=attempt,
                    duration_ms=duration_ms,
                    error_type=exc.__class__.__name__,
                    reason="payload_validation",
                )
                raise SearxngParsingError("SearXNG payload could not be parsed") from exc

            self._log_event(
                logging.INFO,
                "request_completed",
                attempt=attempt,
                duration_ms=duration_ms,
                status_code=response.status_code,
                result_count=len(parsed.results),
            )
            return parsed

        raise SearxngRequestError("SearXNG request exhausted retries")

    def _is_retryable_exception(self, exc: httpx.RequestError) -> bool:
        return isinstance(
            exc,
            (
                httpx.TimeoutException,
                httpx.NetworkError,
                httpx.RemoteProtocolError,
            ),
        )

    def _sleep_before_retry(self, attempt: int, **fields: Any) -> None:
        delay_seconds = self.config.backoff_seconds * (2 ** (attempt - 1))
        if delay_seconds <= 0:
            return

        self._log_event(
            logging.INFO,
            "request_retry_scheduled",
            attempt=attempt,
            delay_seconds=delay_seconds,
            **fields,
        )
        self._sleep(delay_seconds)

    def _log_event(self, level: int, event: str, **fields: Any) -> None:
        self._logger.log(
            level,
            "searxng.%s %s",
            event,
            json.dumps(fields, sort_keys=True, default=str),
        )
