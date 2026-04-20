from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from pydantic import BaseModel, Field
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.gemini_model import gemini_model
from config.env_config import get_env


logger = logging.getLogger(__name__)

TRACKING_QUERY_PARAMS = {
    "utm_source",
    "utm_medium",
    "utm_campaign",
    "utm_term",
    "utm_content",
    "gclid",
    "fbclid",
}
GEMINI_SEARCH_RETRYABLE_MARKERS = (
    "429",
    "500",
    "502",
    "503",
    "504",
    "connection reset",
    "deadline exceeded",
    "internal",
    "overloaded",
    "rate limit",
    "resource exhausted",
    "service unavailable",
    "temporarily unavailable",
    "timeout",
    "unavailable",
)
DEFAULT_GEMINI_SEARCH_RETRY_COUNT = 4
DEFAULT_GEMINI_SEARCH_BACKOFF_SECONDS = 1.0
DEFAULT_GEMINI_SEARCH_MAX_BACKOFF_SECONDS = 8.0
DEFAULT_GEMINI_SEARCH_503_MIN_RETRY_SECONDS = 300.0
DEFAULT_WEB_SEARCH_MAX_RESULTS = 5

DEEP_RESEARCH_SYSTEM_PROMPT = """
You are an autonomous Deep Research Assistant.
Your objective is to comprehensively research the given topic and answer any specific questions.
Use the web_search tool to find relevant sources and web_fetch tool to read articles and documents in depth.
Iterate your search queries to gather comprehensive, high-quality, and up-to-date information.
Once you have enough information, synthesize your findings into a detailed, well-structured Markdown report.
Ensure you cite your sources (URLs) within the report.
"""

GATHER_SOURCES_SYSTEM_PROMPT = (
    DEEP_RESEARCH_SYSTEM_PROMPT.strip()
    + "\n\n"
    + "When using web_search, evaluate whether the returned results satisfy the research criteria before "
    + "selecting URLs. Check relevance, credibility, recency, coverage of the focus questions, and result "
    + "diversity. If the search results are weak, redundant, off-topic, or do not satisfy the criteria, "
    + "refine the query and search again. Only keep URLs that are strong candidates for the next reading step."
)


class WebSearchCandidate(BaseModel):
    title: str
    url: str
    snippet: str = ""
    source: str | None = None
    published_at: datetime | None = None
    provider: str
    position: int | None = None


class WebSearchResult(BaseModel):
    query: str
    answer: str = ""
    candidates: list[WebSearchCandidate] = Field(default_factory=list)
    providers_tried: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


def _parse_int_env(name: str, default: int) -> int:
    raw_value = get_env(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return int(raw_value)
    except ValueError:
        logger.warning("Invalid integer for %s=%r; using default %s", name, raw_value, default)
        return default


def _parse_float_env(name: str, default: float) -> float:
    raw_value = get_env(name)
    if raw_value is None or raw_value.strip() == "":
        return default
    try:
        return float(raw_value)
    except ValueError:
        logger.warning("Invalid float for %s=%r; using default %s", name, raw_value, default)
        return default


def _gemini_search_retry_count() -> int:
    return max(1, _parse_int_env("JATAYU_GEMINI_SEARCH_RETRY_COUNT", DEFAULT_GEMINI_SEARCH_RETRY_COUNT))


def _gemini_search_backoff_seconds() -> float:
    return max(0.0, _parse_float_env("JATAYU_GEMINI_SEARCH_BACKOFF_SECONDS", DEFAULT_GEMINI_SEARCH_BACKOFF_SECONDS))


def _gemini_search_max_backoff_seconds() -> float:
    return max(0.0, _parse_float_env("JATAYU_GEMINI_SEARCH_MAX_BACKOFF_SECONDS", DEFAULT_GEMINI_SEARCH_MAX_BACKOFF_SECONDS))


def _gemini_search_503_min_retry_seconds() -> float:
    return max(
        0.0,
        _parse_float_env(
            "JATAYU_GEMINI_SEARCH_503_MIN_RETRY_SECONDS",
            DEFAULT_GEMINI_SEARCH_503_MIN_RETRY_SECONDS,
        ),
    )


def _normalize_search_url(url: str) -> str:
    parsed = urlparse(url.strip())
    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key.lower() not in TRACKING_QUERY_PARAMS
    ]
    normalized = parsed._replace(
        scheme=(parsed.scheme or "https").lower(),
        netloc=parsed.netloc.lower(),
        path=parsed.path.rstrip("/") or "/",
        query=urlencode(filtered_query, doseq=True),
        fragment="",
    )
    return urlunparse(normalized)


def _is_retryable_gemini_search_error(exc: Exception) -> bool:
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return any(marker in text for marker in GEMINI_SEARCH_RETRYABLE_MARKERS)


def _is_503_gemini_search_error(exc: Exception) -> bool:
    text = f"{exc.__class__.__name__}: {exc}".lower()
    return "503" in text or "service unavailable" in text


def _gemini_search_retry_delay(attempt: int) -> float:
    base_delay = _gemini_search_backoff_seconds()
    if base_delay <= 0:
        return 0.0
    return min(base_delay * (2 ** max(attempt - 1, 0)), _gemini_search_max_backoff_seconds())


def _response_text(candidate: object) -> str:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return "\n".join(texts)


def _append_unique_candidates(
    target: list[WebSearchCandidate],
    incoming: list[WebSearchCandidate],
    *,
    seen_urls: set[str],
    max_results: int | None,
) -> None:
    for candidate in incoming:
        normalized_url = _normalize_search_url(candidate.url)
        if normalized_url in seen_urls:
            continue
        seen_urls.add(normalized_url)
        target.append(candidate.model_copy(update={"url": normalized_url, "position": len(target) + 1}))
        if max_results is not None and len(target) >= max_results:
            return


def _build_search_text(result: WebSearchResult) -> str:
    lines: list[str] = []
    if result.answer:
        lines.append(f"Gemini Grounded Answer: {result.answer}")
    if result.candidates:
        lines.append("Sources:")
        for index, candidate in enumerate(result.candidates, start=1):
            line = f"{index}. {candidate.title} | {candidate.url}"
            if candidate.snippet:
                line = f"{line} | {candidate.snippet}"
            lines.append(line)
    if lines:
        return "\n".join(lines)
    if result.warnings:
        return f"Search failed: {'; '.join(result.warnings)}"
    return "Search failed: no search results returned"


def _build_gather_sources_prompt(topic: str, specific_questions: list[str]) -> str:
    focus_questions = [question.strip() for question in specific_questions if question.strip()]
    prompt_parts = [
        f"Gather a list of relevant URLs for research on '{topic}'.",
        "Evaluate each search result before selecting it. A good result must satisfy these criteria:",
        "- Relevance: directly addresses the topic or a focus question.",
        "- Credibility: prefer official sources, primary documents, or reputable reporting.",
        "- Recency: prefer current information when the topic is time-sensitive.",
        "- Coverage: the final set of URLs should collectively answer the important questions.",
        "- Diversity: avoid redundant pages and unnecessary repeats from the same domain.",
        "If the current search results do not satisfy the criteria, refine the query and search again.",
    ]
    if focus_questions:
        prompt_parts.extend(
            [
                "Focus questions the selected URLs should help answer:",
                *[f"- {question}" for question in focus_questions],
            ]
        )
    prompt_parts.extend(
        [
            "Exclude low-value results such as duplicates, thin aggregator pages, or pages that do not materially help answer the research task.",
            "Return ONLY a JSON list of strings, where each string is a URL. Do not include markdown formatting or objects.",
        ]
    )
    return "\n".join(prompt_parts)


async def _search_with_gemini(query: str, *, max_results: int | None) -> WebSearchResult:
    from google.genai import types

    from agent.gemini_model import get_client, gemini_model

    attempts = _gemini_search_retry_count()
    min_retry_503_seconds = _gemini_search_503_min_retry_seconds()
    warnings: list[str] = []
    client = get_client()
    started_at = time.monotonic()

    attempt = 0
    while True:
        attempt += 1
        try:
            response = await client.aio.models.generate_content(
                model=gemini_model.get_large_model(),
                contents=query,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                ),
            )
        except Exception as exc:
            is_503_error = _is_503_gemini_search_error(exc)
            elapsed_seconds = max(0.0, time.monotonic() - started_at)
            should_retry = False
            if is_503_error:
                should_retry = elapsed_seconds < min_retry_503_seconds
            elif attempt < attempts and _is_retryable_gemini_search_error(exc):
                should_retry = True

            if should_retry:
                delay = _gemini_search_retry_delay(attempt)
                logger.warning(
                    "Gemini search attempt %s%s failed for query %r with retryable error: %s",
                    attempt,
                    " (continuing retries for 503)" if is_503_error else "",
                    query,
                    exc,
                )
                if delay > 0:
                    await asyncio.sleep(delay)
                continue
            raise

        candidates: list[WebSearchCandidate] = []
        seen_urls: set[str] = set()
        answer = ""

        for candidate in getattr(response, "candidates", None) or []:
            if not answer:
                answer = _response_text(candidate)

            grounding_metadata = getattr(candidate, "grounding_metadata", None)
            grounding_chunks = getattr(grounding_metadata, "grounding_chunks", None) or []
            for chunk in grounding_chunks:
                web = getattr(chunk, "web", None)
                uri = getattr(web, "uri", None)
                if not isinstance(uri, str) or not uri.strip():
                    continue

                normalized_url = _normalize_search_url(uri)
                if normalized_url in seen_urls:
                    continue

                parsed = urlparse(normalized_url)
                title = getattr(web, "title", None) or parsed.netloc or normalized_url
                source = parsed.netloc.lower() or None
                if source and source.startswith("www."):
                    source = source[4:]

                candidates.append(
                    WebSearchCandidate(
                        title=str(title).strip() or normalized_url,
                        url=normalized_url,
                        source=source,
                        provider="gemini",
                        position=len(candidates) + 1,
                    )
                )
                seen_urls.add(normalized_url)
                if max_results is not None and len(candidates) >= max_results:
                    break

            if max_results is not None and len(candidates) >= max_results:
                break

        return WebSearchResult(
            query=query,
            answer=answer,
            candidates=candidates if max_results is None else candidates[:max_results],
            providers_tried=["gemini"],
            warnings=warnings,
        )


async def _search_with_searxng(query: str, *, max_results: int) -> WebSearchResult:
    from search.searxng import SearxngConfig, search_web

    config = SearxngConfig.from_env()
    if not config.enabled:
        return WebSearchResult(query=query, providers_tried=["searxng"])

    result = await asyncio.to_thread(search_web, query, max_results=max_results)
    candidates = [
        WebSearchCandidate(
            title=item.title,
            url=item.url,
            snippet=item.snippet or "",
            source=item.source or item.domain,
            published_at=item.published_date,
            provider="searxng",
            position=index,
        )
        for index, item in enumerate(result.results, start=1)
    ]
    return WebSearchResult(
        query=query,
        candidates=candidates[:max_results],
        providers_tried=["searxng"],
        warnings=list(result.warnings),
    )


async def _search_with_duckduckgo(query: str, *, max_results: int) -> WebSearchResult:
    from duckduckgo_search import DDGS

    def _run_search() -> list[dict]:
        with DDGS() as ddgs:
            return [item for item in ddgs.text(query, max_results=max_results)]

    raw_results = await asyncio.to_thread(_run_search)
    candidates: list[WebSearchCandidate] = []
    for index, item in enumerate(raw_results, start=1):
        url = str(item.get("href", "")).strip()
        if not url:
            continue
        normalized_url = _normalize_search_url(url)
        parsed = urlparse(normalized_url)
        source = parsed.netloc.lower() or None
        if source and source.startswith("www."):
            source = source[4:]
        candidates.append(
            WebSearchCandidate(
                title=str(item.get("title", "No Title")).strip() or normalized_url,
                url=normalized_url,
                snippet=str(item.get("body", "")).strip(),
                source=source,
                provider="duckduckgo",
                position=index,
            )
        )
    return WebSearchResult(
        query=query,
        candidates=candidates[:max_results],
        providers_tried=["duckduckgo"],
    )


def _should_use_fallback(candidate_count: int, *, max_results: int | None) -> bool:
    if max_results is None:
        return candidate_count == 0
    return candidate_count < max_results


def _fallback_result_budget(candidate_count: int, *, max_results: int | None) -> int:
    if max_results is None:
        return DEFAULT_WEB_SEARCH_MAX_RESULTS
    return max(0, max_results - candidate_count)


async def search_web_candidates(
    query: str,
    *,
    max_results: int | None = DEFAULT_WEB_SEARCH_MAX_RESULTS,
    use_fallback: bool = False,
) -> WebSearchResult:
    """Return structured web-search results with Gemini prioritized first."""

    result = WebSearchResult(query=query)
    seen_urls: set[str] = set()

    try:
        gemini_result = await _search_with_gemini(query, max_results=max_results)
        result.providers_tried.extend(gemini_result.providers_tried)
        result.warnings.extend(gemini_result.warnings)
        if gemini_result.answer:
            result.answer = gemini_result.answer
        _append_unique_candidates(
            result.candidates,
            gemini_result.candidates,
            seen_urls=seen_urls,
            max_results=max_results,
        )
    except Exception as exc:
        logger.warning("Gemini search failed for query %r after retries: %s", query, exc)
        result.providers_tried.append("gemini")
        result.warnings.append(f"Gemini search failed: {exc}")

    if not use_fallback:
        return result

    if _should_use_fallback(len(result.candidates), max_results=max_results):
        try:
            searxng_result = await _search_with_searxng(
                query,
                max_results=_fallback_result_budget(len(result.candidates), max_results=max_results),
            )
            result.providers_tried.extend(searxng_result.providers_tried)
            result.warnings.extend(searxng_result.warnings)
            _append_unique_candidates(
                result.candidates,
                searxng_result.candidates,
                seen_urls=seen_urls,
                max_results=max_results,
            )
        except Exception as exc:
            logger.warning("SearXNG search failed for query %r: %s", query, exc)
            result.providers_tried.append("searxng")
            result.warnings.append(f"SearXNG search failed: {exc}")

    if _should_use_fallback(len(result.candidates), max_results=max_results):
        try:
            duckduckgo_result = await _search_with_duckduckgo(
                query,
                max_results=_fallback_result_budget(len(result.candidates), max_results=max_results),
            )
            result.providers_tried.extend(duckduckgo_result.providers_tried)
            result.warnings.extend(duckduckgo_result.warnings)
            _append_unique_candidates(
                result.candidates,
                duckduckgo_result.candidates,
                seen_urls=seen_urls,
                max_results=max_results,
            )
        except Exception as exc:
            logger.warning("DuckDuckGo search failed for query %r: %s", query, exc)
            result.providers_tried.append("duckduckgo")
            result.warnings.append(f"DuckDuckGo search failed: {exc}")

    return result

async def gather_sources(topic: str, specific_questions: list[str]) -> list[str]:
    """Gather sources for a deep research task."""
    
    api_key = get_env("GEMINI_API_KEY", required=True)
    
    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=GATHER_SOURCES_SYSTEM_PROMPT,
        tools=[web_search],
    )

    prompt = _build_gather_sources_prompt(topic, specific_questions)

    result = await research_agent.run(prompt, message_history=[])
    
    output = result.output.strip()
    if output.startswith("```json"):
        output = output[7:]
    elif output.startswith("```"):
        output = output[3:]
    if output.endswith("```"):
        output = output[:-3]
    output = output.strip()
    
    try:
        parsed = json.loads(output)
        if isinstance(parsed, list):
            urls = []
            for item in parsed:
                if isinstance(item, str):
                    urls.append(item)
                elif isinstance(item, dict) and "url" in item:
                    urls.append(item["url"])
                elif isinstance(item, dict) and "link" in item:
                    urls.append(item["link"])
            return urls
    except json.JSONDecodeError:
        pass

    return []

async def read_sources(urls: list[str]) -> str:
    """Read the content of a list of URLs and return a concatenated string."""
    import asyncio
    api_key = get_env("GEMINI_API_KEY", required=True)
    
    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        tools=[web_fetch],
    )

    async def process_url(url: str) -> str:
        try:
            if url.endswith(".pdf"):
                text = await read_pdf(url)
                return f"Source: {url}\n{text}\n\n"
            else:
                prompt = f"Read the content of the following URL and extract the relevant information. Summarize the key findings related to the overall research topic.\nURL: {url}"
                result = await research_agent.run(prompt, message_history=[])
                return f"Source: {url}\n{result.output}\n\n"
        except Exception as e:
            return f"Source: {url}\nFailed to read or process: {e}\n\n"

    # Process up to 10 URLs concurrently
    tasks = [process_url(url) for url in urls[:10]]
    results = await asyncio.gather(*tasks)
        
    return "".join(results)

async def synthesize_report(topic: str, specific_questions: list[str], sources_content: str, feedback: str | None = None) -> str:
    """Synthesize a report from the content of the sources."""

    api_key = get_env("GEMINI_API_KEY", required=True)

    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
    )

    prompt = f"Synthesize a detailed, well-structured Markdown report on '{topic}'"
    if specific_questions:
        prompt += f" with a focus on:\n"
        for q in specific_questions:
            prompt += f"- {q}\n"

    if feedback:
        prompt += f"\n\nThe user provided the following feedback to guide this report. You MUST incorporate this feedback:\n{feedback}\n"

    prompt += f"\nUse the following content to write the report. Make sure to use inline citations linking back to the source URLs provided in the text:\n{sources_content}"

    result = await research_agent.run(prompt, message_history=[])
    return result.output

async def read_pdf(url: str) -> str:
    """Read the text content of a PDF file from a URL."""
    import httpx
    from pypdf import PdfReader
    import io

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()

    pdf_file = io.BytesIO(response.content)
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()

    return text[:200000]

async def web_search(query: str, use_fallback: bool = False) -> str:
    """Search the web for a given query.
    
    Priority:
    1. Gemini Google Search (Native Grounding)
    2. SearXNG (optional fallback)
    3. DuckDuckGo Search (optional fallback)

    By default this returns all grounded Gemini results and does not use
    fallback providers unless `use_fallback=True`.
    """
    result = await search_web_candidates(query, max_results=None, use_fallback=use_fallback)
    return _build_search_text(result)

async def web_fetch(url: str) -> str:
    """Fetch the content of a web page and extract its text."""
    import httpx
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        async with httpx.AsyncClient(headers=headers, follow_redirects=True) as client:
            response = await client.get(url, timeout=15.0)
            response.raise_for_status()
            
        from lxml import html
        tree = html.fromstring(response.content)
        # Remove script, style, nav, header, and footer elements
        for bad in tree.xpath("//script|//style|//nav|//header|//footer"):
            bad.drop_tree()
        text = tree.text_content()
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = "\n".join(chunk for chunk in chunks if chunk)
        # Limit to 200000 characters
        return text[:200000]
    except Exception as e:
        return f"Failed to fetch or parse {url}: {e}"
