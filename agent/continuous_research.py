import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.continuous_state import ContinuousTask, global_continuous_state
from agent.deep_research_agent import send_proactive_update
from agent.gemini_model import gemini_model
from agent.research_steps import read_pdf, web_fetch, web_search
from config.env_config import get_env
from models import TelegramMessageEvent

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("continuous_research_output")
FEEDBACK_PLACEHOLDER = "<!-- Write your feedback/instructions here for the agent -->\n"
EVIDENCE_FILE_NAME = "evidence.json"
MAX_SEARCH_QUERIES = 4
MAX_RESULTS_PER_QUERY = 5
MAX_FETCHED_SOURCES_PER_CYCLE = 5
MAX_SOURCE_EXCERPT_CHARS = 6000
MAX_KNOWN_SOURCES_IN_PROMPT = 10
SUMMARY_PROMPT_CHAR_LIMIT = 10000
PLAN_PROMPT_CHAR_LIMIT = 4000
AUTO_PAUSE_AFTER_EMPTY_CYCLES = 5
AUTO_PAUSE_AFTER_FAILURES = 3
TRACKING_QUERY_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}

_URL_RE = re.compile(r"https?://[^\s\]>)}]+")
_TASK_LOCKS: dict[str, asyncio.Lock] = {}


class CycleResult(BaseModel):
    found_new_info: bool = Field(description="True if new, relevant information was found in this cycle.")
    new_findings: str = Field(description="Detailed new findings formatted in markdown with source citations.")
    updated_summary: str = Field(description="An updated, concise summary of the key findings tracked so far.")
    updated_plan: str = Field(description="A concise markdown checklist of what has been checked and what should be checked next.")
    supporting_urls: list[str] = Field(default_factory=list, description="Primary URLs that support the new findings for this cycle.")
    suggested_file_name: str | None = Field(default=None, description="Optional file name for a distinct update report. Leave null for the default timestamped update file.")


class SearchCandidate(BaseModel):
    query: str
    title: str
    url: str
    snippet: str = ""
    source: str | None = None
    published_at: datetime | None = None
    position: int | None = None


class SourceDocument(BaseModel):
    query: str
    title: str
    url: str
    snippet: str = ""
    source: str | None = None
    published_at: datetime | None = None
    content_excerpt: str
    content_hash: str


class EvidenceRecord(BaseModel):
    url: str
    title: str
    source: str | None = None
    published_at: datetime | None = None
    snippet: str = ""
    last_query: str = ""
    first_seen_at: datetime
    last_seen_at: datetime
    last_checked_at: datetime
    content_hash: str
    latest_excerpt: str = ""


class EvidenceLedger(BaseModel):
    records: dict[str, EvidenceRecord] = Field(default_factory=dict)


@dataclass
class CycleResearchMaterial:
    queries: list[str] = field(default_factory=list)
    candidates: list[SearchCandidate] = field(default_factory=list)
    selected_candidates: list[SearchCandidate] = field(default_factory=list)
    fetched_documents: list[SourceDocument] = field(default_factory=list)
    new_or_changed_documents: list[SourceDocument] = field(default_factory=list)
    search_errors: list[str] = field(default_factory=list)
    fetch_errors: list[str] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _get_task_lock(task_id: str) -> asyncio.Lock:
    if task_id not in _TASK_LOCKS:
        _TASK_LOCKS[task_id] = asyncio.Lock()
    return _TASK_LOCKS[task_id]


def _find_existing_task_for_event(event: TelegramMessageEvent) -> ContinuousTask | None:
    for task in global_continuous_state.get_all_tasks():
        if task.status == "stopped":
            continue

        if task.source_event_id and task.source_event_id == event.event_id:
            return task

        if (
            not task.source_event_id
            and task.event_dict.get("event_id") == event.event_id
        ):
            return task

    return None


def _normalize_url(url: str) -> str:
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


def _trim_for_prompt(text: str, max_chars: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return f"{cleaned[: max_chars - 3].rstrip()}..."


def _default_plan(topic: str) -> str:
    return (
        f"- [ ] Re-check the newest primary sources for {topic}\n"
        "- [ ] Compare any newly fetched material against the tracked evidence ledger\n"
        "- [ ] Notify the user only when a change is genuinely new and relevant"
    )


def _ensure_feedback_file(task_dir: Path) -> Path:
    feedback_path = task_dir / "feedback.md"
    if not feedback_path.exists():
        feedback_path.write_text(FEEDBACK_PLACEHOLDER, encoding="utf-8")
    return feedback_path


def _load_evidence_ledger(task_dir: Path) -> EvidenceLedger:
    evidence_path = task_dir / EVIDENCE_FILE_NAME
    if not evidence_path.exists():
        return EvidenceLedger()
    try:
        return EvidenceLedger.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("Failed to load evidence ledger for %s: %s", task_dir.name, exc)
        return EvidenceLedger()


def _save_evidence_ledger(task_dir: Path, ledger: EvidenceLedger) -> None:
    evidence_path = task_dir / EVIDENCE_FILE_NAME
    evidence_path.write_text(ledger.model_dump_json(indent=2), encoding="utf-8")


def _build_search_queries(task: ContinuousTask) -> list[str]:
    query_candidates = [
        task.topic,
        f"{task.topic} latest updates",
        f"{task.topic} {task.instructions}".strip(),
    ]
    for line in task.plan.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\[\]xX\s]+", "", line).strip()
        if cleaned:
            query_candidates.append(f"{task.topic} {cleaned}")
            break

    queries: list[str] = []
    seen: set[str] = set()
    for raw_query in query_candidates:
        normalized = " ".join(raw_query.split())
        if not normalized:
            continue
        query_key = normalized.lower()
        if query_key in seen:
            continue
        seen.add(query_key)
        queries.append(normalized[:200])
        if len(queries) >= MAX_SEARCH_QUERIES:
            break

    return queries or [task.topic]


def _extract_source_candidates_from_text(
    raw_result: str,
    query: str,
    *,
    max_results: int = MAX_RESULTS_PER_QUERY,
) -> list[SearchCandidate]:
    candidates: list[SearchCandidate] = []
    seen_urls: set[str] = set()

    for line in raw_result.splitlines():
        stripped = line.strip()
        if not stripped or stripped.lower().startswith("gemini grounded answer"):
            continue

        match = _URL_RE.search(stripped)
        if match is None:
            continue

        url = match.group(0).rstrip(".,)")
        normalized_url = _normalize_url(url)
        if normalized_url in seen_urls:
            continue

        before_url = stripped[: match.start()].strip()
        before_url = re.sub(r"^[\-\*\d\.\)\s]+", "", before_url).strip(" |:-")
        after_url = stripped[match.end() :].strip(" |:-")
        parsed = urlparse(url)
        title = before_url or parsed.netloc or url

        candidates.append(
            SearchCandidate(
                query=query,
                title=title,
                url=url,
                snippet=after_url,
                source=parsed.netloc.lower() or None,
                position=len(candidates) + 1,
            )
        )
        seen_urls.add(normalized_url)

        if len(candidates) >= max_results:
            break

    return candidates


async def _search_candidates(query: str) -> tuple[list[SearchCandidate], list[str]]:
    search_errors: list[str] = []

    raw_result = await web_search(query)
    if not raw_result.startswith("Search failed:"):
        candidates = _extract_source_candidates_from_text(raw_result, query)
        if candidates:
            return candidates, search_errors
    else:
        search_errors.append(f"Search failed for '{query}': {raw_result.removeprefix('Search failed:').strip()}")

    try:
        from search.searxng import SearxngConfig, search_web

        config = SearxngConfig.from_env()
        if config.enabled:
            result = await asyncio.to_thread(search_web, query, max_results=MAX_RESULTS_PER_QUERY)
            candidates = [
                SearchCandidate(
                    query=query,
                    title=item.title,
                    url=item.url,
                    snippet=item.snippet or "",
                    source=item.source or item.domain,
                    published_at=item.published_date,
                    position=item.position,
                )
                for item in result.results
            ]
            if candidates:
                return candidates, search_errors
    except Exception as exc:
        search_errors.append(f"Structured search failed for '{query}': {exc}")

    return [], search_errors


def _prioritize_candidates(candidates: list[SearchCandidate], ledger: EvidenceLedger) -> list[SearchCandidate]:
    deduped: dict[str, SearchCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(_normalize_url(candidate.url), candidate)

    def sort_key(candidate: SearchCandidate) -> tuple[int, float, int]:
        is_seen = 1 if _normalize_url(candidate.url) in ledger.records else 0
        published_rank = 0.0
        if candidate.published_at is not None:
            published_rank = -candidate.published_at.timestamp()
        position = candidate.position or 999
        return (is_seen, published_rank, position)

    return sorted(deduped.values(), key=sort_key)


async def _fetch_source_document(candidate: SearchCandidate) -> tuple[SourceDocument | None, str | None]:
    try:
        if candidate.url.lower().endswith(".pdf"):
            content = await read_pdf(candidate.url)
        else:
            content = await web_fetch(candidate.url)
    except Exception as exc:
        return None, f"{candidate.url}: {exc}"

    if not content:
        return None, f"{candidate.url}: no content returned"
    if content.startswith("Failed to fetch or parse"):
        return None, content

    excerpt = _trim_for_prompt(content, MAX_SOURCE_EXCERPT_CHARS)
    content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()

    return (
        SourceDocument(
            query=candidate.query,
            title=candidate.title,
            url=candidate.url,
            snippet=candidate.snippet,
            source=candidate.source,
            published_at=candidate.published_at,
            content_excerpt=excerpt,
            content_hash=content_hash,
        ),
        None,
    )


def _merge_documents_into_ledger(
    ledger: EvidenceLedger,
    documents: list[SourceDocument],
) -> list[SourceDocument]:
    now = _utc_now()
    new_or_changed: list[SourceDocument] = []

    for document in documents:
        key = _normalize_url(document.url)
        record = ledger.records.get(key)
        if record is None:
            ledger.records[key] = EvidenceRecord(
                url=document.url,
                title=document.title,
                source=document.source,
                published_at=document.published_at,
                snippet=document.snippet,
                last_query=document.query,
                first_seen_at=now,
                last_seen_at=now,
                last_checked_at=now,
                content_hash=document.content_hash,
                latest_excerpt=document.content_excerpt,
            )
            new_or_changed.append(document)
            continue

        changed = record.content_hash != document.content_hash
        record.title = document.title or record.title
        record.source = document.source or record.source
        record.published_at = document.published_at or record.published_at
        record.snippet = document.snippet or record.snippet
        record.last_query = document.query
        record.last_seen_at = now
        record.last_checked_at = now
        record.content_hash = document.content_hash
        record.latest_excerpt = document.content_excerpt

        if changed:
            new_or_changed.append(document)

    return new_or_changed


def _render_known_sources_snapshot(ledger: EvidenceLedger) -> str:
    if not ledger.records:
        return "No previously tracked sources."

    sorted_records = sorted(
        ledger.records.values(),
        key=lambda record: record.last_checked_at,
        reverse=True,
    )

    lines: list[str] = []
    for record in sorted_records[:MAX_KNOWN_SOURCES_IN_PROMPT]:
        published = record.published_at.isoformat() if record.published_at else "unknown"
        lines.append(f"- {record.title} | {record.url} | published={published}")
    return "\n".join(lines)


def _render_source_documents(documents: list[SourceDocument]) -> str:
    if not documents:
        return "No new or changed source documents were detected this cycle."

    blocks: list[str] = []
    for index, document in enumerate(documents, start=1):
        published = document.published_at.isoformat() if document.published_at else "unknown"
        blocks.append(
            "\n".join(
                [
                    f"### Source {index}",
                    f"Title: {document.title}",
                    f"URL: {document.url}",
                    f"Source: {document.source or 'unknown'}",
                    f"Published: {published}",
                    f"Search Query: {document.query}",
                    f"Search Snippet: {document.snippet or 'No snippet available.'}",
                    "Fetched Excerpt:",
                    document.content_excerpt,
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_cycle_prompt(
    task: ContinuousTask,
    user_feedback: str,
    material: CycleResearchMaterial,
) -> str:
    prompt_parts = [
        f"Topic: {task.topic}",
        f"Instructions: {task.instructions}",
        "Current Plan:",
        _trim_for_prompt(task.plan or _default_plan(task.topic), PLAN_PROMPT_CHAR_LIMIT),
        "Previous Summary:",
        _trim_for_prompt(task.last_summary or "No prior findings.", SUMMARY_PROMPT_CHAR_LIMIT),
    ]

    if user_feedback:
        prompt_parts.extend(
            [
                "New User Feedback (highest priority):",
                user_feedback,
            ]
        )

    prompt_parts.extend(
        [
            "Queries Run This Cycle:",
            "\n".join(f"- {query}" for query in material.queries),
            "Selected Candidate URLs:",
            "\n".join(f"- {candidate.title} | {candidate.url}" for candidate in material.selected_candidates)
            or "No candidate URLs were selected.",
            "Known Evidence Ledger Snapshot:",
            _render_known_sources_snapshot(material.ledger),
        ]
    )

    if material.search_errors:
        prompt_parts.extend(
            [
                "Search Errors:",
                "\n".join(f"- {error}" for error in material.search_errors),
            ]
        )

    if material.fetch_errors:
        prompt_parts.extend(
            [
                "Fetch Errors:",
                "\n".join(f"- {error}" for error in material.fetch_errors),
            ]
        )

    prompt_parts.extend(
        [
            "New Or Changed Source Documents:",
            _render_source_documents(material.new_or_changed_documents),
            "Requirements:",
            "- Only set found_new_info=true if the new or changed source documents contain materially new, user-relevant information.",
            "- Use markdown links when citing sources.",
            "- Keep updated_summary concise and canonical rather than a running log.",
            "- Keep updated_plan to a short markdown checklist with next checks informed by this cycle.",
            "- If there are no new or changed source documents, set found_new_info=false.",
        ]
    )

    return "\n\n".join(prompt_parts)


async def _collect_cycle_material(task: ContinuousTask, task_dir: Path) -> CycleResearchMaterial:
    ledger = _load_evidence_ledger(task_dir)
    queries = _build_search_queries(task)

    candidates: list[SearchCandidate] = []
    search_errors: list[str] = []
    for query in queries:
        found_candidates, query_errors = await _search_candidates(query)
        candidates.extend(found_candidates)
        search_errors.extend(query_errors)

    selected_candidates = _prioritize_candidates(candidates, ledger)[:MAX_FETCHED_SOURCES_PER_CYCLE]
    fetch_errors: list[str] = []
    fetched_documents: list[SourceDocument] = []

    if selected_candidates:
        fetch_results = await asyncio.gather(
            *(_fetch_source_document(candidate) for candidate in selected_candidates)
        )
        for document, fetch_error in fetch_results:
            if fetch_error:
                fetch_errors.append(fetch_error)
            elif document is not None:
                fetched_documents.append(document)

    new_or_changed_documents = _merge_documents_into_ledger(ledger, fetched_documents)
    _save_evidence_ledger(task_dir, ledger)

    return CycleResearchMaterial(
        queries=queries,
        candidates=candidates,
        selected_candidates=selected_candidates,
        fetched_documents=fetched_documents,
        new_or_changed_documents=new_or_changed_documents,
        search_errors=search_errors,
        fetch_errors=fetch_errors,
        ledger=ledger,
    )


def get_cycle_agent() -> Agent:
    api_key = get_env("GEMINI_API_KEY", required=True)
    return Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=(
            "You are a Continuous Research Analyst. "
            "You review newly fetched source material for a monitoring task and decide whether anything is materially new. "
            "Be conservative about novelty. Repeated coverage, minor rewrites, or unchanged sources are not new information. "
            "When there is new information, summarize it crisply and cite the supporting URLs with markdown links. "
            "Always maintain a concise canonical summary and a short markdown plan for future monitoring."
        ),
        output_type=CycleResult,
    )


async def _run_continuous_cycle(task: ContinuousTask) -> str:
    logger.info("Running continuous cycle for task %s (%s)", task.task_id, task.topic)
    task_dir = OUTPUT_DIR / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = _ensure_feedback_file(task_dir)
    task_lock = _get_task_lock(task.task_id)

    if task_lock.locked():
        logger.info("Skipping overlapping cycle for task %s", task.task_id)
        return "skipped_already_running"

    async with task_lock:
        if task.status != "running":
            logger.info("Skipping cycle for task %s because status is %s", task.task_id, task.status)
            return "skipped_not_running"

        started_at = _utc_now()
        global_continuous_state.update_task(
            task.task_id,
            is_cycle_running=True,
            last_cycle_started_at=started_at,
        )
        feedback_snapshot = feedback_path.read_text(encoding="utf-8").strip()
        user_feedback = ""
        if feedback_snapshot and feedback_snapshot != FEEDBACK_PLACEHOLDER.strip():
            user_feedback = feedback_snapshot

        try:
            material = await _collect_cycle_material(task, task_dir)
            if not material.fetched_documents and (material.search_errors or material.fetch_errors):
                error_text = "\n".join(material.search_errors + material.fetch_errors)
                raise RuntimeError(f"Unable to collect usable source material this cycle.\n{error_text}")

            prompt = _build_cycle_prompt(task, user_feedback, material)
            result = await get_cycle_agent().run(prompt)
            output: CycleResult = getattr(result, "data", getattr(result, "output", None))
            if output is None:
                raise RuntimeError("Continuous research cycle returned no structured output.")

            task.last_summary = output.updated_summary
            task.plan = output.updated_plan
            (task_dir / "plan.md").write_text(task.plan, encoding="utf-8")
            (task_dir / "summary.md").write_text(task.last_summary, encoding="utf-8")

            if user_feedback:
                current_feedback = feedback_path.read_text(encoding="utf-8").strip()
                if current_feedback == feedback_snapshot:
                    feedback_path.write_text(FEEDBACK_PLACEHOLDER, encoding="utf-8")

            completed_at = _utc_now()
            if output.found_new_info:
                task.no_new_findings_count = 0
                updates_dir = task_dir / "updates"
                updates_dir.mkdir(exist_ok=True)

                timestamp_str = completed_at.astimezone().strftime("%Y-%m-%d_%H-%M-%S")
                file_name = output.suggested_file_name if output.suggested_file_name else f"update_{timestamp_str}.md"
                if not file_name.endswith(".md"):
                    file_name += ".md"

                update_lines = [f"# Update: {completed_at.astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", "", output.new_findings]
                if output.supporting_urls:
                    update_lines.extend(
                        [
                            "",
                            "## Supporting URLs",
                            *[f"- {url}" for url in output.supporting_urls],
                        ]
                    )
                (updates_dir / file_name).write_text("\n".join(update_lines).strip() + "\n", encoding="utf-8")

                global_continuous_state.update_task(
                    task.task_id,
                    last_summary=task.last_summary,
                    plan=task.plan,
                    no_new_findings_count=0,
                    cycle_count=task.cycle_count + 1,
                    failure_count=0,
                    last_error="",
                    is_cycle_running=False,
                    last_cycle_completed_at=completed_at,
                    last_new_info_at=completed_at,
                )

                event = TelegramMessageEvent(**task.event_dict)
                await send_proactive_update(
                    event,
                    (
                        f"Continuous Research ('{task.topic}') found new updates with refreshed source analysis.\n"
                        f"Check the `{task.task_id}/updates/` directory or ask me for a status summary."
                    ),
                )
                return "completed_with_updates"

            task.no_new_findings_count += 1
            status = "paused" if task.no_new_findings_count >= AUTO_PAUSE_AFTER_EMPTY_CYCLES else task.status
            global_continuous_state.update_task(
                task.task_id,
                last_summary=task.last_summary,
                plan=task.plan,
                no_new_findings_count=task.no_new_findings_count,
                cycle_count=task.cycle_count + 1,
                failure_count=0,
                last_error="",
                is_cycle_running=False,
                last_cycle_completed_at=completed_at,
                status=status,
            )

            if status == "paused":
                event = TelegramMessageEvent(**task.event_dict)
                await send_proactive_update(
                    event,
                    (
                        f"I haven't found any new updates for '{task.topic}' recently after checking fresh source material. "
                        "I have paused the continuous research. Let me know if you want to resume it or change the instructions."
                    ),
                )
            return "completed_without_updates"

        except Exception as exc:
            completed_at = _utc_now()
            failure_count = task.failure_count + 1
            status = "paused" if failure_count >= AUTO_PAUSE_AFTER_FAILURES else task.status
            global_continuous_state.update_task(
                task.task_id,
                failure_count=failure_count,
                last_error=str(exc),
                is_cycle_running=False,
                last_cycle_completed_at=completed_at,
                status=status,
            )
            logger.error("Error in continuous cycle for task %s: %s", task.task_id, exc)

            event = TelegramMessageEvent(**task.event_dict)
            if status == "paused":
                await send_proactive_update(
                    event,
                    (
                        f"Continuous research for '{task.topic}' hit repeated errors and has been paused.\n"
                        f"Latest error: {exc}"
                    ),
                )
            return "failed"


async def continuous_research_loop():
    """Background loop that periodically runs continuous research tasks."""
    logger.info("Starting continuous research background loop.")
    while True:
        try:
            tasks = global_continuous_state.get_all_tasks()
            for task in tasks:
                if task.status == "running":
                    await _run_continuous_cycle(task)
        except Exception as exc:
            logger.error("Error in continuous research loop: %s", exc)

        await asyncio.sleep(600)


async def start_continuous_research(ctx: RunContext[TelegramMessageEvent], topic: str, instructions: str) -> str:
    """Start a continuous background research task.

    Use this when the user wants to monitor a topic over time, like finding new
    apartment listings or tracking ongoing news. The task keeps a research plan,
    a canonical summary, and a structured evidence ledger on disk.
    """
    event = ctx.deps
    existing_task = _find_existing_task_for_event(event)
    if existing_task is not None:
        return (
            f"Continuous research is already tracking this request as task {existing_task.task_id} "
            f"for '{existing_task.topic}' (status: {existing_task.status})."
        )

    task_id = f"cr_{uuid4().hex[:8]}"

    task = ContinuousTask(
        task_id=task_id,
        topic=topic,
        instructions=instructions,
        status="running",
        source_event_id=event.event_id,
        source_message_id=event.message_id,
        source_channel_id=event.channel_id,
        event_dict=event.model_dump(mode="json"),
    )
    global_continuous_state.add_task(task)

    task_dir = OUTPUT_DIR / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    _ensure_feedback_file(task_dir)
    (task_dir / "updates").mkdir(exist_ok=True)
    _save_evidence_ledger(task_dir, EvidenceLedger())

    asyncio.create_task(_run_continuous_cycle(task))
    return (
        f"Started continuous research on '{topic}'. Task ID is {task_id}. "
        "I will keep looking in the background, track evidence, and update you when I find something genuinely new."
    )


async def stop_continuous_research(ctx: RunContext[TelegramMessageEvent], task_id: str) -> str:
    """Stop a continuous research task permanently."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    global_continuous_state.update_task(task_id, status="stopped")
    return f"Stopped continuous research task {task_id} ('{task.topic}')."


async def pause_continuous_research(ctx: RunContext[TelegramMessageEvent], task_id: str) -> str:
    """Pause a continuous research task. It can be resumed later."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    global_continuous_state.update_task(task_id, status="paused")
    return f"Paused continuous research task {task_id} ('{task.topic}')."


async def resume_continuous_research(
    ctx: RunContext[TelegramMessageEvent],
    task_id: str,
    new_instructions: str | None = None,
) -> str:
    """Resume a paused continuous research task, optionally with new instructions."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    updates: dict[str, object] = {
        "status": "running",
        "no_new_findings_count": 0,
        "failure_count": 0,
        "last_error": "",
    }
    if new_instructions:
        updates["instructions"] = new_instructions

    global_continuous_state.update_task(task_id, **updates)

    if task.is_cycle_running:
        return f"Continuous research task {task_id} ('{task.topic}') is already running a cycle."

    asyncio.create_task(_run_continuous_cycle(task))
    return f"Resumed continuous research task {task_id} ('{task.topic}')."


async def update_continuous_research_plan(
    ctx: RunContext[TelegramMessageEvent],
    task_id: str,
    feedback: str,
) -> str:
    """Provide feedback or alter the high-level plan for an active continuous research task."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    task_dir = OUTPUT_DIR / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = _ensure_feedback_file(task_dir)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(feedback_path, "a", encoding="utf-8") as f:
        f.write(f"\n\n### Feedback at {timestamp}:\n{feedback}\n")

    if task.status == "running":
        if task.is_cycle_running:
            return f"Wrote feedback for task {task_id} to '{feedback_path}'. It will be picked up by the current running cycle or the next one."
        asyncio.create_task(_run_continuous_cycle(task))
        return f"Wrote feedback for task {task_id} to '{feedback_path}'. Triggered a new research cycle to process it."

    return f"Wrote feedback for task {task_id} to '{feedback_path}'. The task is currently {task.status}. Resume it to process the feedback."


async def trigger_continuous_research_cycle(ctx: RunContext[TelegramMessageEvent], task_id: str) -> str:
    """Manually trigger an immediate research cycle for a specific continuous research task."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    if task.status != "running":
        return f"Task {task_id} is currently {task.status}. Please resume it first."

    if task.is_cycle_running:
        return f"Task {task_id} ('{task.topic}') is already processing a research cycle."

    asyncio.create_task(_run_continuous_cycle(task))
    return (
        f"Manually triggered a new research cycle for task {task_id} ('{task.topic}'). "
        "The agent will check for updates and process any feedback files now."
    )


async def get_continuous_research_status(ctx: RunContext[TelegramMessageEvent]) -> str:
    """Get the status of continuous research tasks."""
    tasks = global_continuous_state.get_all_tasks()
    if not tasks:
        return "There are no continuous research tasks currently."

    ordered_tasks = sorted(
        tasks,
        key=lambda task: (
            0 if task.status == "running" else 1,
            task.topic.lower(),
            task.task_id,
        ),
    )

    response_lines = ["Continuous Research Tasks:", ""]
    for task in ordered_tasks:
        response_lines.append(f"- **ID**: {task.task_id}")
        response_lines.append(f"  **Topic**: {task.topic}")
        response_lines.append(f"  **Status**: {task.status}")
        response_lines.append(f"  **Cycles Run**: {task.cycle_count}")
        response_lines.append(f"  **Cycle Running**: {'yes' if task.is_cycle_running else 'no'}")

        if task.last_cycle_completed_at:
            response_lines.append(f"  **Last Cycle Completed**: {task.last_cycle_completed_at.isoformat()}")
        if task.last_new_info_at:
            response_lines.append(f"  **Last New Info**: {task.last_new_info_at.isoformat()}")
        if task.plan:
            response_lines.append(
                f"  **Plan Preview**: {_trim_for_prompt(task.plan, 200)}"
            )
        if task.last_summary:
            response_lines.append(
                f"  **Latest Summary Preview**: {_trim_for_prompt(task.last_summary, 200)}"
            )
        if task.last_error:
            response_lines.append(f"  **Last Error**: {_trim_for_prompt(task.last_error, 200)}")
        response_lines.append("")

    return "\n".join(response_lines).rstrip()
