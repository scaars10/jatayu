import asyncio
import hashlib
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.continuous_state import ContinuousTask, global_continuous_state
from agent.deep_research_agent import send_proactive_update
from agent.gemini_model import gemini_model
from agent.research_steps import (
    _gemini_search_503_min_retry_seconds,
    read_pdf,
    search_web_candidates,
    web_fetch,
    web_search,
)
from config.env_config import get_env
from models import TelegramMessageEvent

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("continuous_research_output")
FEEDBACK_PLACEHOLDER = "<!-- Write your feedback/instructions here for the agent -->\n"
EVIDENCE_FILE_NAME = "evidence.json"
BRIEF_FILE_NAME = "brief.md"
WORKSPACE_MANIFEST_FILE_NAME = "workspace_manifest.md"
DECISION_LOG_FILE_NAME = "decision_log.md"
ACTIVITY_LOG_FILE_NAME = "activity.log"
CYCLE_STATE_FILE_NAME = "cycle_state.json"
CANDIDATE_POOL_FILE_NAME = "candidate_pool.json"
FETCHED_SOURCES_FILE_NAME = "fetched_sources.json"
ANALYSIS_RESULT_FILE_NAME = "analysis_result.json"
QUERY_HISTORY_FILE_NAME = "query_history.jsonl"
CYCLE_REPORTS_DIR_NAME = "cycle_reports"
QUESTIONS_FILE_NAME = "questions.md"
PLAN_FILE_NAME = "plan.md"
SUMMARY_FILE_NAME = "summary.md"
DETAILED_REPORT_FILE_NAME = "detailed_report.md"
SOURCE_POLICY_FILE_NAME = "source_policy.md"  # Legacy artifact; ignored by the active flow.
ASSESSMENT_FILE_NAME = "assessment.md"
EVIDENCE_DIR_NAME = "evidence"
CLAIM_REGISTER_JSON_FILE_NAME = "claim_register.json"
CLAIM_REGISTER_MD_FILE_NAME = "claim_register.md"
FOUNDATION_WORKSPACE_VERSION = "foundation_v1"
MAX_SEARCH_QUERIES = 4
MAX_RESULTS_PER_QUERY = 5
MAX_FETCHED_SOURCES_PER_CYCLE = 5
BOOTSTRAP_CYCLE_COUNT = 3
BOOTSTRAP_MAX_SEARCH_QUERIES = 12
BOOTSTRAP_MAX_RESULTS_PER_QUERY = 8
BOOTSTRAP_MAX_FETCHED_SOURCES_PER_CYCLE = 15
MAX_CONCURRENT_SEARCH_QUERIES = 4
MAX_CONCURRENT_FETCHES_PER_CYCLE = 6
SOURCE_RECHECK_COOLDOWN = timedelta(days=3)
SEARCH_STAGE_TIMEOUT = timedelta(minutes=3)
DEFAULT_SEARCH_STAGE_TIMEOUT = SEARCH_STAGE_TIMEOUT
SEARCH_STAGE_TIMEOUT_BUFFER = timedelta(minutes=1)
FETCH_STAGE_TIMEOUT = timedelta(minutes=4)
ANALYSIS_STAGE_TIMEOUT = timedelta(minutes=3)
ANALYSIS_TRANSIENT_RETRY_DELAYS_SECONDS = (5, 15)
STALE_CYCLE_GRACE_PERIOD = timedelta(minutes=15)
MAX_SOURCE_EXCERPT_CHARS = 6000
MAX_CHANGED_DOC_PROMPT_CHARS = 20000
MAX_CYCLE_REPORT_SOURCE_EXCERPT_CHARS = 1800
MAX_CYCLE_REPORT_SOURCES = 20
MAX_KNOWN_SOURCES_IN_PROMPT = 10
MAX_QUERY_HISTORY_IN_PROMPT = 12
MAX_SEARCH_OBSERVATION_ANSWER_CHARS = 1800
MAX_SEARCH_OBSERVATIONS_PROMPT_CHARS = 12000
MIN_SOURCE_RECHECK_HOURS = 1
MAX_SOURCE_RECHECK_HOURS = 168
NOTIFICATION_CONFIDENCE_THRESHOLD = 0.8
CANONICAL_SUMMARY_CONFIDENCE_THRESHOLD = 0.6
SUMMARY_PROMPT_CHAR_LIMIT = 10000
DETAILED_REPORT_PROMPT_CHAR_LIMIT = 24000
PLAN_PROMPT_CHAR_LIMIT = 4000
BRIEF_PROMPT_CHAR_LIMIT = 12000
DECISION_LOG_TAIL_CHAR_LIMIT = 4000
QUESTIONS_PROMPT_CHAR_LIMIT = 6000
SOURCE_POLICY_PROMPT_CHAR_LIMIT = 5000
CLAIM_REGISTER_PROMPT_CHAR_LIMIT = 7000
ASSESSMENT_PROMPT_CHAR_LIMIT = 5000
AUTO_PAUSE_AFTER_EMPTY_CYCLES = 5
AUTO_PAUSE_AFTER_FAILURES = 3
REVIEW_CYCLE_INTERVAL = 5
TRACKING_QUERY_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "gclid", "fbclid"}
NO_PRIOR_FINDINGS = "No prior findings yet."

SOURCE_TIERS = (
    "official",
    "regulatory",
    "primary_reporting",
    "secondary_reporting",
    "low_trust",
)
SOURCE_TIER_ORDER = {
    "official": 0,
    "regulatory": 1,
    "primary_reporting": 2,
    "secondary_reporting": 3,
    "low_trust": 4,
}
CLAIM_STATUSES = (
    "active",
    "resolved",
    "contradicted",
    "superseded",
    "deferred",
)

REGULATORY_DOMAIN_MARKERS = (
    ".gov",
    ".gov.",
    ".nic.",
    "rera",
    "sec.gov",
    "regulator",
)
LOW_TRUST_DOMAIN_MARKERS = (
    "reddit.com",
    "x.com",
    "twitter.com",
    "facebook.com",
    "instagram.com",
    "youtube.com",
    "quora.com",
    "medium.com",
    "blogspot.",
    "wordpress.",
    "substack.com",
    "forum",
    "forums.",
)
PRIMARY_REPORTING_DOMAIN_MARKERS = (
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "economictimes.",
    "thehindu.",
    "indianexpress.",
    "business-standard.",
    "press",
    "newsroom",
)
SECONDARY_REPORTING_DOMAIN_MARKERS = (
    "housing.com",
    "99acres.com",
    "magicbricks.com",
    "commonfloor.com",
    "proptiger.com",
    "nobroker.in",
    "propstory.com",
    "youtube.com",
)
QUESTION_STOPWORDS = {
    "about",
    "after",
    "again",
    "against",
    "also",
    "been",
    "being",
    "between",
    "does",
    "from",
    "have",
    "into",
    "latest",
    "near",
    "next",
    "only",
    "over",
    "that",
    "their",
    "them",
    "there",
    "these",
    "they",
    "this",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
}

_URL_RE = re.compile(r"https?://[^\s\]>)}]+")
_TASK_LOCKS: dict[str, asyncio.Lock] = {}


def _search_stage_timeout(query_count: int) -> timedelta:
    if SEARCH_STAGE_TIMEOUT != DEFAULT_SEARCH_STAGE_TIMEOUT:
        return SEARCH_STAGE_TIMEOUT

    concurrent_queries = max(1, MAX_CONCURRENT_SEARCH_QUERIES)
    normalized_query_count = max(1, query_count)
    query_batches = (normalized_query_count + concurrent_queries - 1) // concurrent_queries
    retry_window = timedelta(seconds=_gemini_search_503_min_retry_seconds())
    scaled_timeout = (retry_window * query_batches) + SEARCH_STAGE_TIMEOUT_BUFFER
    return max(SEARCH_STAGE_TIMEOUT, scaled_timeout)


class SearchQueryDecision(BaseModel):
    query: str
    reason: str = ""
    expected_signal: str = ""
    query_role: str = Field(
        default="new",
        description="Why this query is being spent now: new, repeat, refinement, corroboration, or contradiction_check.",
    )


class SearchPlan(BaseModel):
    queries: list[SearchQueryDecision] = Field(default_factory=list)
    strategy_summary: str = ""


class DocumentConfidence(BaseModel):
    url: str
    evidence_type: str = "unknown"
    confidence: float = 0.0
    confidence_reason: str = ""
    limitations: str = ""
    supports_claims: list[str] = Field(default_factory=list)
    contradicts_claims: list[str] = Field(default_factory=list)


class SourceMonitoringDecision(BaseModel):
    url: str
    suggested_recheck_after_hours: int = 24
    reason: str = ""


class QueryHistoryEntry(BaseModel):
    cycle_id: str = ""
    query: str
    candidate_count: int = 0
    selected_count: int = 0
    fetched_count: int = 0
    changed_count: int = 0
    errors: list[str] = Field(default_factory=list)
    top_domains: list[str] = Field(default_factory=list)
    outcome: str = "unknown"
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class CycleResult(BaseModel):
    found_new_info: bool = Field(description="True if new, relevant information was found in this cycle.")
    new_findings: str = Field(description="Detailed new findings formatted in markdown with source citations.")
    updated_summary: str = Field(description="An updated, concise summary of the key findings tracked so far.")
    updated_detailed_report: str = Field(
        default="",
        description=(
            "A comprehensive living markdown report that preserves useful details across cycles, "
            "including findings, comparisons, confidence notes, caveats, unresolved questions, and cited sources."
        ),
    )
    updated_plan: str = Field(description="A concise markdown checklist of what has been checked and what should be checked next.")
    updated_questions: str = Field(
        default="",
        description=(
            "Updated markdown for the living questions artifact. "
            "Maintain a concise set of active, resolved, and deferred questions for future cycles."
        ),
    )
    updated_brief: str = Field(
        default="",
        description=(
            "Optional full updated markdown for the task brief. "
            "If provided during review, preserve explicit user intent and only refine the other sections."
        ),
    )
    updated_source_policy: str = Field(
        default="",
        description=(
            "Deprecated. Do not use this field; source reliability is evaluated per document."
        ),
    )
    updated_claim_register_markdown: str = Field(
        default="",
        description="Optional analyst-facing markdown notes to accompany the derived claim register snapshot.",
    )
    updated_assessment: str = Field(
        default="",
        description=(
            "Updated markdown for the assessment artifact. "
            "It must explain what changed, what matters, what remains unresolved, and whether to notify."
        ),
    )
    should_notify: bool = Field(
        default=False,
        description="True only when this cycle should produce a user-facing notification after app-level gating.",
    )
    notification_reason: str = Field(
        default="",
        description="Short explanation for why the user should or should not be notified about this cycle.",
    )
    supporting_urls: list[str] = Field(default_factory=list, description="Primary URLs that support the new findings for this cycle.")
    suggested_file_name: str | None = Field(default=None, description="Optional file name for a distinct update report. Leave null for the default timestamped update file.")
    document_confidences: list[DocumentConfidence] = Field(
        default_factory=list,
        description="Per-document reliability and confidence evaluation based on fetched content and context.",
    )
    source_monitoring: list[SourceMonitoringDecision] = Field(
        default_factory=list,
        description="Optional per-source recheck timing suggestions based on source volatility and evidence needs.",
    )
    suggested_next_queries: list[SearchQueryDecision] = Field(
        default_factory=list,
        description="Optional factual search suggestions for a later cycle. The app may use these as planner context.",
    )
    claim_records: list["ClaimRecord"] = Field(
        default_factory=list,
        description="Full current claim register for the workspace-enabled task.",
    )


class SearchCandidate(BaseModel):
    query: str
    title: str
    url: str
    snippet: str = ""
    source: str | None = None
    published_at: datetime | None = None
    provider: str | None = None
    position: int | None = None


class SearchObservation(BaseModel):
    query: str
    answer: str = ""
    candidate_count: int = 0
    selected_count: int = 0
    fetched_count: int = 0
    changed_count: int = 0
    top_candidates: list[SearchCandidate] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


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
    next_check_after: datetime | None = None
    recheck_reason: str = ""
    change_count: int = 0
    unchanged_count: int = 0


class EvidenceLedger(BaseModel):
    records: dict[str, EvidenceRecord] = Field(default_factory=dict)


class ClaimRecord(BaseModel):
    id: str
    statement: str
    status: str
    confidence: float = 0.5
    confidence_reason: str = ""
    supporting_urls: list[str] = Field(default_factory=list)
    contradicting_urls: list[str] = Field(default_factory=list)
    source_tiers: list[str] = Field(default_factory=list)
    contradicts: list[str] = Field(default_factory=list)
    updated_at: datetime
    notes: str = ""


class ClaimRegister(BaseModel):
    claims: list[ClaimRecord] = Field(default_factory=list)


class CycleRuntimeState(BaseModel):
    task_id: str
    cycle_id: str = ""
    status: str = "idle"
    stage: str = "idle"
    started_at: datetime | None = None
    last_heartbeat_at: datetime | None = None
    completed_at: datetime | None = None
    current_query: str = ""
    current_url: str = ""
    query_count: int = 0
    selected_candidate_count: int = 0
    fetched_document_count: int = 0
    changed_document_count: int = 0
    last_event: str = ""
    notification_reason: str = ""
    error: str = ""
    result: str = ""
    stage_details: dict[str, object] = Field(default_factory=dict)


@dataclass
class SourcePolicyConfig:
    markdown: str = ""
    preferred_domains: set[str] = field(default_factory=set)
    discouraged_domains: set[str] = field(default_factory=set)
    domain_notes: str = ""


@dataclass
class CycleResearchMaterial:
    queries: list[str] = field(default_factory=list)
    search_observations: list[SearchObservation] = field(default_factory=list)
    candidates: list[SearchCandidate] = field(default_factory=list)
    selected_candidates: list[SearchCandidate] = field(default_factory=list)
    fetched_documents: list[SourceDocument] = field(default_factory=list)
    new_or_changed_documents: list[SourceDocument] = field(default_factory=list)
    search_errors: list[str] = field(default_factory=list)
    fetch_errors: list[str] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    selected_candidate_tiers: dict[str, str] = field(default_factory=dict)
    source_tier_counts: dict[str, int] = field(default_factory=dict)
    query_decisions: list[SearchQueryDecision] = field(default_factory=list)
    query_history: list[QueryHistoryEntry] = field(default_factory=list)


@dataclass
class WorkspaceArtifacts:
    user_request: str = ""
    explicit_user_intent: str = ""
    assistant_working_interpretation: str = ""
    known_constraints: str = ""
    inferred_assumptions: str = ""
    novelty_guidance: str = ""
    cadence_and_stop_conditions: str = ""
    feedback: str = ""
    questions: str = ""
    plan: str = ""
    summary: str = ""
    detailed_report: str = ""
    source_policy: str = ""
    claim_register_markdown: str = ""
    assessment: str = ""
    recent_decision_log: str = ""
    query_history: str = ""


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


def _default_summary() -> str:
    return NO_PRIOR_FINDINGS


def _default_detailed_report(task: ContinuousTask | None = None) -> str:
    topic = task.topic if task is not None and task.topic else "the tracked topic"
    return "\n".join(
        [
            "# Continuous Research Detailed Report",
            "",
            f"Topic: {topic}",
            "",
            "## Current Bottom Line",
            "- No detailed findings have been confirmed yet.",
            "",
            "## Key Findings",
            "- No findings recorded yet.",
            "",
            "## Evidence And Confidence",
            "- No evidence has been evaluated yet.",
            "",
            "## Comparisons And Tradeoffs",
            "- No comparisons recorded yet.",
            "",
            "## Watchlist",
            "- Continue monitoring based on the current plan.",
            "",
            "## Open Questions",
            "- No open questions recorded yet.",
            "",
            "## Source Notes",
            "- No sources recorded yet.",
            "",
        ]
    )


def _ensure_feedback_file(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = task_dir / "feedback.md"
    if not feedback_path.exists():
        feedback_path.write_text(FEEDBACK_PLACEHOLDER, encoding="utf-8")
    return feedback_path


def _brief_path(task_dir: Path) -> Path:
    return task_dir / BRIEF_FILE_NAME


def _workspace_manifest_path(task_dir: Path) -> Path:
    return task_dir / WORKSPACE_MANIFEST_FILE_NAME


def _decision_log_path(task_dir: Path) -> Path:
    return task_dir / DECISION_LOG_FILE_NAME


def _activity_log_path(task_dir: Path) -> Path:
    return task_dir / ACTIVITY_LOG_FILE_NAME


def _cycle_state_path(task_dir: Path) -> Path:
    return task_dir / CYCLE_STATE_FILE_NAME


def _candidate_pool_path(task_dir: Path) -> Path:
    return task_dir / CANDIDATE_POOL_FILE_NAME


def _fetched_sources_path(task_dir: Path) -> Path:
    return task_dir / FETCHED_SOURCES_FILE_NAME


def _analysis_result_path(task_dir: Path) -> Path:
    return task_dir / ANALYSIS_RESULT_FILE_NAME


def _query_history_path(task_dir: Path) -> Path:
    return task_dir / QUERY_HISTORY_FILE_NAME


def _cycle_reports_dir_path(task_dir: Path) -> Path:
    return task_dir / CYCLE_REPORTS_DIR_NAME


def _questions_path(task_dir: Path) -> Path:
    return task_dir / QUESTIONS_FILE_NAME


def _plan_path(task_dir: Path) -> Path:
    return task_dir / PLAN_FILE_NAME


def _summary_path(task_dir: Path) -> Path:
    return task_dir / SUMMARY_FILE_NAME


def _detailed_report_path(task_dir: Path) -> Path:
    return task_dir / DETAILED_REPORT_FILE_NAME


def _source_policy_path(task_dir: Path) -> Path:
    return task_dir / SOURCE_POLICY_FILE_NAME


def _assessment_path(task_dir: Path) -> Path:
    return task_dir / ASSESSMENT_FILE_NAME


def _evidence_dir_path(task_dir: Path) -> Path:
    return task_dir / EVIDENCE_DIR_NAME


def _claim_register_json_path(task_dir: Path) -> Path:
    return _evidence_dir_path(task_dir) / CLAIM_REGISTER_JSON_FILE_NAME


def _claim_register_markdown_path(task_dir: Path) -> Path:
    return _evidence_dir_path(task_dir) / CLAIM_REGISTER_MD_FILE_NAME


def _task_user_request(task: ContinuousTask, event: TelegramMessageEvent | None = None) -> str:
    if event is not None and event.message:
        return event.message.strip()

    message = task.event_dict.get("message", "")
    if isinstance(message, str):
        return message.strip()
    return ""


def _render_brief(task: ContinuousTask, user_request: str) -> str:
    assistant_interpretation_lines = [
        f"- Topic: {task.topic or 'Not set'}",
        f"- Instructions: {task.instructions or 'No assistant interpretation recorded.'}",
    ]

    return "\n".join(
        [
            "# Continuous Research Brief",
            "",
            "## User Request",
            user_request or "No original user request recorded.",
            "",
            "## Explicit User Intent",
            user_request or "No explicit user intent recorded beyond the original request.",
            "",
            "## Assistant Working Interpretation",
            *assistant_interpretation_lines,
            "",
            "## Known Constraints",
            "- No additional confirmed constraints recorded yet.",
            "",
            "## Inferred Assumptions",
            "- None recorded yet. Keep inferred assumptions labeled here until the user confirms them.",
            "",
            "## Novelty Guidance",
            "- Only treat changes as new when they are materially new and relevant to the user request.",
            "- Repeated coverage, minor rewrites, or unchanged listings/documents are not new information.",
            "",
            "## Cadence And Stop Conditions",
            "- Default cadence: run every 10 minutes while the task is active.",
            (
                f"- Auto-pause after {AUTO_PAUSE_AFTER_EMPTY_CYCLES} empty cycles or "
                f"{AUTO_PAUSE_AFTER_FAILURES} consecutive failures unless resumed."
            ),
            "",
        ]
    )


def _render_workspace_manifest() -> str:
    return "\n".join(
        [
            "# Continuous Research Workspace Manifest",
            "",
            "This is a human-readable index for the bounded-autonomy continuous research workspace.",
            "It is not executable configuration.",
            "",
            "## Agent-Visible Working Memory",
            f"- `{BRIEF_FILE_NAME}`",
            f"- `{QUESTIONS_FILE_NAME}`",
            f"- `{PLAN_FILE_NAME}`",
            f"- `{SUMMARY_FILE_NAME}`",
            f"- `{DETAILED_REPORT_FILE_NAME}`",
            f"- `{EVIDENCE_DIR_NAME}/{CLAIM_REGISTER_MD_FILE_NAME}` as compact existing-claim memory",
            f"- `{QUERY_HISTORY_FILE_NAME}` as compact search execution memory",
            "- `feedback.md`",
            "",
            "These files are assembled into a single Agent Working Context before each cycle. The agent should not infer behavior from file count or read order.",
            "`feedback.md` is a transient feedback inbox. The user may edit it directly, and assistant tools may append user feedback there. The next cycle treats it as high-priority input, updates the working memory files from it, and clears it if no newer feedback arrived while the cycle was running.",
            "",
            "## Diagnostic And Audit Artifacts",
            f"- `{WORKSPACE_MANIFEST_FILE_NAME}`",
            f"- `{DECISION_LOG_FILE_NAME}`",
            f"- `{ACTIVITY_LOG_FILE_NAME}`",
            f"- `{CYCLE_STATE_FILE_NAME}`",
            f"- `{CANDIDATE_POOL_FILE_NAME}`",
            f"- `{FETCHED_SOURCES_FILE_NAME}`",
            f"- `{ANALYSIS_RESULT_FILE_NAME}`",
            f"- `{ASSESSMENT_FILE_NAME}`",
            f"- `{EVIDENCE_FILE_NAME}`",
            f"- `{EVIDENCE_DIR_NAME}/{CLAIM_REGISTER_JSON_FILE_NAME}`",
            f"- `{CYCLE_REPORTS_DIR_NAME}/`",
            "- `updates/`",
            "",
            "## Authoritative Artifacts",
            f"- `{BRIEF_FILE_NAME}` is authoritative for user intent and assistant working interpretation.",
            f"- `{SUMMARY_FILE_NAME}` is the concise current answer.",
            f"- `{DETAILED_REPORT_FILE_NAME}` is the comprehensive current understanding.",
            f"- `{PLAN_FILE_NAME}` and `{QUESTIONS_FILE_NAME}` are the current work queue.",
            f"- `{EVIDENCE_DIR_NAME}/{CLAIM_REGISTER_MD_FILE_NAME}` is compact claim memory for continuity and contradiction checks.",
            f"- `{QUERY_HISTORY_FILE_NAME}` is factual search execution history, not a search policy.",
            "- `feedback.md` is the high-priority user/assistant feedback inbox for the next cycle.",
            "",
            "## Diagnostic Artifact Roles",
            f"- `{WORKSPACE_MANIFEST_FILE_NAME}` explains workspace shape.",
            f"- `{DECISION_LOG_FILE_NAME}` is the append-only audit history of prior cycles.",
            f"- `{ACTIVITY_LOG_FILE_NAME}` is the live execution trace for operator visibility and debugging.",
            f"- `{CYCLE_STATE_FILE_NAME}` is the live per-cycle execution state and heartbeat.",
            f"- `{EVIDENCE_DIR_NAME}/{CLAIM_REGISTER_JSON_FILE_NAME}` is the authoritative machine-readable claim register.",
            f"- `{ASSESSMENT_FILE_NAME}` records the latest materiality and notification assessment; it is not fed back as ordinary context.",
            f"- `{CYCLE_STATE_FILE_NAME}` shows the current stage, heartbeat, and in-progress details.",
            f"- `{CANDIDATE_POOL_FILE_NAME}` captures the latest search candidates, prioritization, and selection.",
            f"- `{FETCHED_SOURCES_FILE_NAME}` captures the latest fetched and changed source documents for the cycle.",
            f"- `{ANALYSIS_RESULT_FILE_NAME}` captures the last structured model output before final commit.",
            f"- `{CYCLE_REPORTS_DIR_NAME}/` captures per-cycle working reports, including non-notified and degraded cycles.",
            "",
            "## Reserved For Later Slices",
            "- `notes/`",
            "- `updates/` contents beyond user-facing reports",
            "",
        ]
    )


def _render_activity_log_header() -> str:
    return "\n".join(
        [
            "# Continuous Research Activity Log",
            "# format: <timestamp> | <level> | <event> | <details-json>",
            "",
        ]
    )


def _write_json_artifact(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def _default_cycle_runtime_state(task: ContinuousTask) -> CycleRuntimeState:
    return CycleRuntimeState(task_id=task.task_id)


def _load_cycle_runtime_state(task: ContinuousTask, task_dir: Path) -> CycleRuntimeState:
    cycle_state_path = _cycle_state_path(task_dir)
    if not cycle_state_path.exists():
        return _default_cycle_runtime_state(task)
    try:
        return CycleRuntimeState.model_validate_json(cycle_state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Continuous research cycle state at %s is unreadable: %s", cycle_state_path, exc)
        return _default_cycle_runtime_state(task)


def _save_cycle_runtime_state(task_dir: Path, cycle_state: CycleRuntimeState) -> None:
    _cycle_state_path(task_dir).write_text(cycle_state.model_dump_json(indent=2), encoding="utf-8")


def _empty_candidate_pool_payload(task: ContinuousTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "cycle_id": "",
        "updated_at": None,
        "queries": [],
        "search_errors": [],
        "query_decisions": [],
        "search_observations": [],
        "candidates": [],
        "prioritized_candidates": [],
        "selected_candidates": [],
    }


def _empty_fetched_sources_payload(task: ContinuousTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "cycle_id": "",
        "updated_at": None,
        "fetch_errors": [],
        "fetched_documents": [],
        "changed_documents": [],
    }


def _empty_analysis_result_payload(task: ContinuousTask) -> dict[str, object]:
    return {
        "task_id": task.task_id,
        "cycle_id": "",
        "updated_at": None,
        "status": "idle",
        "output": None,
    }


def _persist_candidate_pool_artifact(
    task: ContinuousTask,
    task_dir: Path,
    *,
    cycle_id: str,
    queries: list[str],
    query_decisions: list[SearchQueryDecision] | None = None,
    search_observations: list[SearchObservation] | None = None,
    candidates: list[SearchCandidate],
    prioritized_candidates: list[SearchCandidate],
    selected_candidates: list[SearchCandidate],
    search_errors: list[str],
) -> None:
    payload = {
        "task_id": task.task_id,
        "cycle_id": cycle_id,
        "updated_at": _utc_now().isoformat(),
        "queries": queries,
        "query_decisions": [decision.model_dump(mode="json") for decision in (query_decisions or [])],
        "search_observations": [observation.model_dump(mode="json") for observation in (search_observations or [])],
        "search_errors": search_errors,
        "candidates": [candidate.model_dump(mode="json") for candidate in candidates],
        "prioritized_candidates": [candidate.model_dump(mode="json") for candidate in prioritized_candidates],
        "selected_candidates": [candidate.model_dump(mode="json") for candidate in selected_candidates],
    }
    _write_json_artifact(_candidate_pool_path(task_dir), payload)


def _source_document_runtime_payload(document: SourceDocument) -> dict[str, object]:
    payload = document.model_dump(mode="json")
    payload["content_excerpt_preview"] = _trim_for_prompt(document.content_excerpt, 600)
    payload.pop("content_excerpt", None)
    return payload


def _persist_fetched_sources_artifact(
    task: ContinuousTask,
    task_dir: Path,
    *,
    cycle_id: str,
    fetch_errors: list[str],
    fetched_documents: list[SourceDocument],
    changed_documents: list[SourceDocument],
) -> None:
    payload = {
        "task_id": task.task_id,
        "cycle_id": cycle_id,
        "updated_at": _utc_now().isoformat(),
        "fetch_errors": fetch_errors,
        "fetched_documents": [_source_document_runtime_payload(document) for document in fetched_documents],
        "changed_documents": [_source_document_runtime_payload(document) for document in changed_documents],
    }
    _write_json_artifact(_fetched_sources_path(task_dir), payload)


def _persist_analysis_result_artifact(
    task: ContinuousTask,
    task_dir: Path,
    *,
    cycle_id: str,
    status: str,
    output: CycleResult | None = None,
    error: str = "",
) -> None:
    payload = {
        "task_id": task.task_id,
        "cycle_id": cycle_id,
        "updated_at": _utc_now().isoformat(),
        "status": status,
        "error": error,
        "output": output.model_dump(mode="json") if output is not None else None,
    }
    _write_json_artifact(_analysis_result_path(task_dir), payload)


def _load_query_history(task_dir: Path, *, limit: int = MAX_QUERY_HISTORY_IN_PROMPT) -> list[QueryHistoryEntry]:
    path = _query_history_path(task_dir)
    if not path.exists():
        return []

    entries: list[QueryHistoryEntry] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        logger.warning("Continuous research query history at %s is unreadable: %s", path, exc)
        return []

    for line in lines[-limit:]:
        if not line.strip():
            continue
        try:
            entries.append(QueryHistoryEntry.model_validate_json(line))
        except Exception:
            continue
    return entries


def _append_query_history(task_dir: Path, entries: list[QueryHistoryEntry]) -> None:
    if not entries:
        return
    path = _query_history_path(task_dir)
    try:
        with path.open("a", encoding="utf-8") as handle:
            for entry in entries:
                handle.write(entry.model_dump_json() + "\n")
    except Exception as exc:
        logger.warning("Failed to append continuous research query history to %s: %s", path, exc)


def _render_query_history_summary(entries: list[QueryHistoryEntry]) -> str:
    if not entries:
        return "No prior query history recorded."

    lines: list[str] = []
    for entry in entries[-MAX_QUERY_HISTORY_IN_PROMPT:]:
        errors = "; ".join(entry.errors[:2]) if entry.errors else "none"
        domains = ", ".join(entry.top_domains[:4]) if entry.top_domains else "none"
        lines.append(
            (
                f"- {entry.query} | outcome={entry.outcome} | candidates={entry.candidate_count} | "
                f"selected={entry.selected_count} | fetched={entry.fetched_count} | "
                f"changed={entry.changed_count} | domains={domains} | errors={errors}"
            )
        )
    return "\n".join(lines)


def _query_history_entries_for_cycle(
    *,
    cycle_id: str,
    queries: list[str],
    candidates: list[SearchCandidate],
    selected_candidates: list[SearchCandidate],
    fetched_documents: list[SourceDocument],
    changed_documents: list[SourceDocument],
    search_errors: list[str],
    fetch_errors: list[str],
    outcome: str,
) -> list[QueryHistoryEntry]:
    entries: list[QueryHistoryEntry] = []
    for query in queries:
        query_candidates = [candidate for candidate in candidates if candidate.query == query]
        query_selected = [candidate for candidate in selected_candidates if candidate.query == query]
        query_fetched = [document for document in fetched_documents if document.query == query]
        query_changed = [document for document in changed_documents if document.query == query]
        domain_counts = Counter(_candidate_domain(candidate) for candidate in query_candidates)
        errors = [
            error
            for error in search_errors + fetch_errors
            if query.lower() in error.lower()
        ]
        entries.append(
            QueryHistoryEntry(
                cycle_id=cycle_id,
                query=query,
                candidate_count=len(query_candidates),
                selected_count=len(query_selected),
                fetched_count=len(query_fetched),
                changed_count=len(query_changed),
                errors=errors[:3],
                top_domains=[domain for domain, _ in domain_counts.most_common(5)],
                outcome=outcome,
            )
        )
    return entries


def _annotate_search_observations(
    observations: list[SearchObservation],
    *,
    queries: list[str],
    candidates: list[SearchCandidate],
    selected_candidates: list[SearchCandidate],
    fetched_documents: list[SourceDocument],
    changed_documents: list[SourceDocument],
    search_errors: list[str],
    fetch_errors: list[str],
) -> list[SearchObservation]:
    observations_by_query = {observation.query: observation for observation in observations}
    candidate_counts = Counter(candidate.query for candidate in candidates)
    selected_counts = Counter(candidate.query for candidate in selected_candidates)
    fetched_counts = Counter(document.query for document in fetched_documents)
    changed_counts = Counter(document.query for document in changed_documents)

    annotated: list[SearchObservation] = []
    for query in queries:
        observation = observations_by_query.get(query) or SearchObservation(query=query)
        errors = [
            error
            for error in search_errors + fetch_errors
            if query.lower() in error.lower()
        ]
        annotated.append(
            observation.model_copy(
                update={
                    "candidate_count": candidate_counts[query] or observation.candidate_count,
                    "selected_count": selected_counts[query],
                    "fetched_count": fetched_counts[query],
                    "changed_count": changed_counts[query],
                    "errors": [*observation.errors, *errors][:5],
                }
            )
        )
    return annotated


def _cycle_state_stage_for_event(event: str) -> str | None:
    if event in {"cycle_started", "workspace_context_loaded", "legacy_context_loaded"}:
        return "starting"
    if event in {"query_planning_started", "query_planning_failed", "queries_built", "search_budget_prepared", "search_started", "search_completed"}:
        return "search"
    if event in {"candidate_prioritization_completed", "candidate_recheck_skipped", "candidates_selected_for_fetch"}:
        return "select"
    if event in {"fetch_started", "fetch_completed", "ledger_updated"}:
        return "fetch"
    if event in {"analysis_prompt_prepared", "analysis_started", "analysis_retry_scheduled", "analysis_timeout", "analysis_completed"}:
        return "analyze"
    if event in {"workspace_artifacts_updated", "legacy_artifacts_updated", "update_written", "notification_gate_evaluated", "cycle_report_written"}:
        return "commit"
    if event == "cycle_completed":
        return "completed"
    if event == "cycle_failed":
        return "failed"
    return None


def _update_cycle_runtime_state_from_trace(
    cycle_state: CycleRuntimeState,
    *,
    event: str,
    details: dict[str, object],
) -> CycleRuntimeState:
    updated = cycle_state.model_copy(deep=True)
    updated.last_heartbeat_at = _utc_now()
    updated.last_event = event
    derived_stage = _cycle_state_stage_for_event(event)
    if derived_stage is not None:
        updated.stage = derived_stage
    if "query" in details and isinstance(details["query"], str):
        updated.current_query = details["query"].strip()
    if "url" in details and isinstance(details["url"], str):
        updated.current_url = details["url"].strip()
    if "query_count" in details:
        updated.query_count = int(details["query_count"])
    if "selected_candidate_count" in details:
        updated.selected_candidate_count = int(details["selected_candidate_count"])
    if "selected_count" in details:
        updated.selected_candidate_count = int(details["selected_count"])
    if "fetched_document_count" in details:
        updated.fetched_document_count = int(details["fetched_document_count"])
    if "changed_document_count" in details:
        updated.changed_document_count = int(details["changed_document_count"])
    if "notification_reason" in details and isinstance(details["notification_reason"], str):
        updated.notification_reason = details["notification_reason"].strip()
    if "error" in details and isinstance(details["error"], str):
        updated.error = details["error"].strip()
    updated.stage_details = _trace_payload(details)
    return updated


def _recover_orphaned_cycle_state(task: ContinuousTask, task_dir: Path) -> None:
    if not task.is_cycle_running:
        return
    recovered_at = _utc_now()
    if task.last_cycle_started_at is not None and (recovered_at - task.last_cycle_started_at) >= STALE_CYCLE_GRACE_PERIOD:
        reason = "stale_cycle_recovered"
    else:
        reason = "orphaned_running_flag_recovered"

    global_continuous_state.update_task(
        task.task_id,
        is_cycle_running=False,
        last_error=reason,
        last_cycle_completed_at=recovered_at,
    )
    cycle_state = _load_cycle_runtime_state(task, task_dir)
    cycle_state.status = "recovered"
    cycle_state.stage = "idle"
    cycle_state.completed_at = recovered_at
    cycle_state.last_heartbeat_at = recovered_at
    cycle_state.last_event = reason
    cycle_state.error = reason
    cycle_state.result = reason
    _save_cycle_runtime_state(task_dir, cycle_state)
    _trace_continuous_research(
        task,
        task_dir,
        reason,
        previous_started_at=task.last_cycle_started_at.isoformat() if task.last_cycle_started_at else "",
    )


def _ensure_activity_log_file(task_dir: Path) -> Path:
    task_dir.mkdir(parents=True, exist_ok=True)
    activity_log_path = _activity_log_path(task_dir)
    if not activity_log_path.exists():
        activity_log_path.write_text(_render_activity_log_header(), encoding="utf-8")
    return activity_log_path


def _search_candidates_trace_payload(candidates: list[SearchCandidate], *, limit: int = 5) -> list[dict[str, object]]:
    return [
        {
            "title": _trim_for_prompt(candidate.title, 120),
            "url": candidate.url,
            "provider": candidate.provider or "unknown",
            "source": candidate.source or "unknown",
            "query": _trim_for_prompt(candidate.query, 120),
        }
        for candidate in candidates[:limit]
    ]


def _source_documents_trace_payload(documents: list[SourceDocument], *, limit: int = 5) -> list[dict[str, object]]:
    return [
        {
            "title": _trim_for_prompt(document.title, 120),
            "url": document.url,
            "source": document.source or "unknown",
            "query": _trim_for_prompt(document.query, 120),
            "content_hash": document.content_hash[:12],
        }
        for document in documents[:limit]
    ]


def _trace_detail_preview(text: str, *, limit: int = 400) -> str:
    return _trim_for_prompt(text, limit)


def _trace_payload(details: dict[str, object]) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in details.items():
        if value is None:
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            payload[key] = trimmed
            continue
        if isinstance(value, (list, dict, tuple)) and not value:
            continue
        payload[key] = value
    return payload


def _trace_continuous_research(
    task: ContinuousTask,
    task_dir: Path,
    event: str,
    *,
    level: int = logging.INFO,
    **details: object,
) -> None:
    _ensure_activity_log_file(task_dir)
    payload = _trace_payload(
        {
            "task_id": task.task_id,
            "topic": task.topic,
            **details,
        }
    )
    payload_text = json.dumps(payload, default=str, ensure_ascii=True, sort_keys=True)
    logger.log(level, "Continuous research trace [%s]: %s", event, payload_text)

    activity_log_path = _activity_log_path(task_dir)
    timestamp = _utc_now().astimezone().isoformat()
    level_name = logging.getLevelName(level)
    try:
        with activity_log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} | {level_name} | {event} | {payload_text}\n")
    except Exception as exc:
        logger.warning("Failed to append continuous research activity log entry to %s: %s", activity_log_path, exc)


def _render_questions(task: ContinuousTask, user_request: str) -> str:
    return "\n".join(
        [
            "# Continuous Research Questions",
            "",
            "## Active Questions",
            f"- What materially new information has appeared for {task.topic}?",
            f"- What is still unverified in the user request: {user_request or task.topic}?",
            "- Which source should be checked next to reduce uncertainty the most?",
            "",
            "## Recently Resolved",
            "- None yet.",
            "",
            "## Deferred Questions",
            "- None yet.",
            "",
        ]
    )


def _render_source_policy() -> str:
    return "\n".join(
        [
            "# Continuous Research Source Policy",
            "",
            "## Trust Tiers",
            "1. `official`",
            "2. `regulatory`",
            "3. `primary_reporting`",
            "4. `secondary_reporting`",
            "5. `low_trust`",
            "",
            "## Preferred Sources",
            "- None recorded yet.",
            "",
            "## Discouraged Sources",
            "- None recorded yet.",
            "",
            "## Domain Notes",
            "- Prefer primary and authoritative sources when they exist.",
            "- Low-trust sources may inform open questions but should not become canonical facts without stronger support.",
            "",
        ]
    )


def _render_assessment() -> str:
    return "\n".join(
        [
            "# Continuous Research Assessment",
            "",
            "## What Changed",
            "- No assessment yet.",
            "",
            "## What Is Genuinely New",
            "- No assessment yet.",
            "",
            "## What Matters To The User",
            "- No assessment yet.",
            "",
            "## What Remains Unresolved",
            "- No assessment yet.",
            "",
            "## Notification Decision",
            "- Notify: no",
            "- Reason: No material change has been assessed yet.",
            "",
        ]
    )


def _default_claim_register() -> ClaimRegister:
    return ClaimRegister()


def _render_claim_register_markdown(
    claim_register: ClaimRegister,
    *,
    notes: str = "",
) -> str:
    lines = [
        "# Continuous Research Claim Register",
        "",
        "## Active Claims",
    ]
    active_claims = [claim for claim in claim_register.claims if claim.status == "active"]
    if active_claims:
        for claim in active_claims:
            lines.append(
                (
                    f"- `{claim.id}` [{claim.status}] confidence={claim.confidence:.2f} "
                    f"tiers={', '.join(claim.source_tiers) or 'unknown'}: {claim.statement}"
                )
            )
    else:
        lines.append("- None yet.")

    lines.extend(
        [
            "",
            "## Non-Active Claims",
        ]
    )
    inactive_claims = [claim for claim in claim_register.claims if claim.status != "active"]
    if inactive_claims:
        for claim in inactive_claims:
            contradiction_text = f" contradicts={', '.join(claim.contradicts)}" if claim.contradicts else ""
            lines.append(
                (
                    f"- `{claim.id}` [{claim.status}] confidence={claim.confidence:.2f} "
                    f"tiers={', '.join(claim.source_tiers) or 'unknown'}{contradiction_text}: {claim.statement}"
                )
            )
    else:
        lines.append("- None yet.")

    if notes.strip():
        lines.extend(
            [
                "",
                "## Analyst Notes",
                notes.strip(),
            ]
        )

    lines.append("")
    return "\n".join(lines)


def _render_decision_log_header() -> str:
    return "\n".join(
        [
            "# Continuous Research Decision Log",
            "",
            "This append-only log records what each cycle checked, what changed, and why the next step was chosen.",
            "",
        ]
    )


def _safe_read_text(path: Path, *, description: str, default: str = "") -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.warning("Continuous research %s missing at %s; using default.", description, path)
    except Exception as exc:
        logger.warning("Continuous research failed to read %s at %s: %s", description, path, exc)
    return default


def _parse_markdown_sections(markdown_text: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for line in markdown_text.splitlines():
        if line.startswith("## "):
            if current_section is not None:
                sections[current_section] = "\n".join(current_lines).strip()
            current_section = line[3:].strip()
            current_lines = []
            continue

        if current_section is not None:
            current_lines.append(line)

    if current_section is not None:
        sections[current_section] = "\n".join(current_lines).strip()

    return sections


def _tail_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def _extract_domains_from_text(section_text: str) -> set[str]:
    domains: set[str] = set()
    for line in section_text.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\[\]xX\s]+", "", line).strip()
        if not cleaned or cleaned.lower() in {"none", "none recorded yet.", "none yet."}:
            continue
        tokens = re.findall(r"https?://[^\s]+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}", cleaned)
        for token in tokens:
            domain = urlparse(token).netloc.lower() if "://" in token else token.lower()
            domain = domain.strip().strip(".,)")
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domains.add(domain)
    return domains


def _parse_source_policy(markdown_text: str) -> SourcePolicyConfig:
    sections = _parse_markdown_sections(markdown_text)
    return SourcePolicyConfig(
        markdown=markdown_text.strip(),
        preferred_domains=_extract_domains_from_text(sections.get("Preferred Sources", "")),
        discouraged_domains=_extract_domains_from_text(sections.get("Discouraged Sources", "")),
        domain_notes=sections.get("Domain Notes", "").strip(),
    )


def _ensure_workspace_artifacts(
    task: ContinuousTask,
    task_dir: Path,
    *,
    event: TelegramMessageEvent | None = None,
) -> None:
    task_dir.mkdir(parents=True, exist_ok=True)
    _ensure_feedback_file(task_dir)
    _ensure_activity_log_file(task_dir)
    (task_dir / "updates").mkdir(exist_ok=True)
    _cycle_reports_dir_path(task_dir).mkdir(exist_ok=True)
    _evidence_dir_path(task_dir).mkdir(exist_ok=True)

    if not (task_dir / EVIDENCE_FILE_NAME).exists():
        _save_evidence_ledger(task_dir, EvidenceLedger())

    plan_path = _plan_path(task_dir)
    if not plan_path.exists():
        plan_path.write_text(task.plan or _default_plan(task.topic), encoding="utf-8")

    summary_path = _summary_path(task_dir)
    if not summary_path.exists():
        summary_path.write_text(task.last_summary or _default_summary(), encoding="utf-8")

    detailed_report_path = _detailed_report_path(task_dir)
    if not detailed_report_path.exists():
        detailed_report_path.write_text(_default_detailed_report(task), encoding="utf-8")

    user_request = _task_user_request(task, event)

    brief_path = _brief_path(task_dir)
    if not brief_path.exists():
        brief_path.write_text(_render_brief(task, user_request), encoding="utf-8")

    questions_path = _questions_path(task_dir)
    if not questions_path.exists():
        questions_path.write_text(_render_questions(task, user_request), encoding="utf-8")

    assessment_path = _assessment_path(task_dir)
    if not assessment_path.exists():
        assessment_path.write_text(_render_assessment(), encoding="utf-8")

    manifest_path = _workspace_manifest_path(task_dir)
    if not manifest_path.exists() or "Agent-Visible Working Memory" not in manifest_path.read_text(encoding="utf-8", errors="ignore"):
        manifest_path.write_text(_render_workspace_manifest(), encoding="utf-8")

    decision_log_path = _decision_log_path(task_dir)
    if not decision_log_path.exists():
        decision_log_path.write_text(_render_decision_log_header(), encoding="utf-8")

    claim_register_json_path = _claim_register_json_path(task_dir)
    if not claim_register_json_path.exists():
        claim_register_json_path.write_text(_default_claim_register().model_dump_json(indent=2), encoding="utf-8")

    claim_register_markdown_path = _claim_register_markdown_path(task_dir)
    if not claim_register_markdown_path.exists():
        claim_register = _load_claim_register(task_dir)
        claim_register_markdown_path.write_text(
            _render_claim_register_markdown(claim_register),
            encoding="utf-8",
        )

    cycle_state_path = _cycle_state_path(task_dir)
    if not cycle_state_path.exists():
        _save_cycle_runtime_state(task_dir, _default_cycle_runtime_state(task))

    candidate_pool_path = _candidate_pool_path(task_dir)
    if not candidate_pool_path.exists():
        _write_json_artifact(candidate_pool_path, _empty_candidate_pool_payload(task))

    fetched_sources_path = _fetched_sources_path(task_dir)
    if not fetched_sources_path.exists():
        _write_json_artifact(fetched_sources_path, _empty_fetched_sources_payload(task))

    analysis_result_path = _analysis_result_path(task_dir)
    if not analysis_result_path.exists():
        _write_json_artifact(analysis_result_path, _empty_analysis_result_payload(task))

    query_history_path = _query_history_path(task_dir)
    if not query_history_path.exists():
        query_history_path.write_text("", encoding="utf-8")


def _workspace_enabled(task: ContinuousTask, task_dir: Path) -> bool:
    manifest_path = _workspace_manifest_path(task_dir)
    if manifest_path.exists():
        logger.info("Using workspace-enabled continuous research path for task %s", task.task_id)
        return True

    if task.workspace_version == FOUNDATION_WORKSPACE_VERSION:
        logger.warning(
            "Workspace manifest missing for workspace-enabled task %s; regenerating defaults.",
            task.task_id,
        )
        _ensure_workspace_artifacts(task, task_dir)
        return True

    logger.info("Using legacy continuous research path for task %s", task.task_id)
    return False


def _load_workspace_artifacts(task: ContinuousTask, task_dir: Path) -> WorkspaceArtifacts:
    _ensure_workspace_artifacts(task, task_dir)
    claim_register = _load_claim_register(task_dir)

    brief_text = _safe_read_text(
        _brief_path(task_dir),
        description="brief",
        default=_render_brief(task, _task_user_request(task)),
    )
    sections = _parse_markdown_sections(brief_text)
    required_sections = {
        "User Request",
        "Explicit User Intent",
        "Assistant Working Interpretation",
    }
    if not required_sections.issubset(sections):
        logger.warning(
            "Continuous research brief for task %s is malformed; regenerating defaults.",
            task.task_id,
        )
        regenerated = _render_brief(task, _task_user_request(task))
        _brief_path(task_dir).write_text(regenerated, encoding="utf-8")
        brief_text = regenerated
        sections = _parse_markdown_sections(brief_text)

    feedback_raw = _safe_read_text(_ensure_feedback_file(task_dir), description="feedback")
    feedback = feedback_raw.strip()
    if feedback == FEEDBACK_PLACEHOLDER.strip():
        feedback = ""

    plan = _safe_read_text(
        _plan_path(task_dir),
        description="plan",
        default=task.plan or _default_plan(task.topic),
    ).strip() or task.plan or _default_plan(task.topic)
    questions = _safe_read_text(
        _questions_path(task_dir),
        description="questions",
        default=_render_questions(task, _task_user_request(task)),
    ).strip() or _render_questions(task, _task_user_request(task))
    summary = _safe_read_text(
        _summary_path(task_dir),
        description="summary",
        default=task.last_summary or _default_summary(),
    ).strip() or task.last_summary or _default_summary()
    detailed_report = _safe_read_text(
        _detailed_report_path(task_dir),
        description="detailed report",
        default=_default_detailed_report(task),
    ).strip() or _default_detailed_report(task)
    claim_register_markdown = _safe_read_text(
        _claim_register_markdown_path(task_dir),
        description="claim register markdown",
        default=_render_claim_register_markdown(claim_register),
    ).strip() or _render_claim_register_markdown(claim_register)
    query_history = _render_query_history_summary(_load_query_history(task_dir))

    return WorkspaceArtifacts(
        user_request=sections.get("User Request", _task_user_request(task)),
        explicit_user_intent=sections.get("Explicit User Intent", _task_user_request(task)),
        assistant_working_interpretation=sections.get(
            "Assistant Working Interpretation",
            f"- Topic: {task.topic}\n- Instructions: {task.instructions}",
        ),
        known_constraints=sections.get("Known Constraints", "- No additional confirmed constraints recorded yet."),
        inferred_assumptions=sections.get(
            "Inferred Assumptions",
            "- None recorded yet. Keep inferred assumptions labeled here until the user confirms them.",
        ),
        novelty_guidance=sections.get(
            "Novelty Guidance",
            "- Only treat materially new and user-relevant changes as novel.",
        ),
        cadence_and_stop_conditions=sections.get(
            "Cadence And Stop Conditions",
            "- Default cadence: run every 10 minutes while the task is active.",
        ),
        feedback=feedback,
        questions=questions,
        plan=plan,
        summary=summary,
        detailed_report=detailed_report,
        source_policy="",
        claim_register_markdown=claim_register_markdown,
        assessment="",
        recent_decision_log="",
        query_history=query_history,
    )


def _first_checklist_item(markdown_text: str) -> str:
    for line in markdown_text.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\[\]xX\s]+", "", line).strip()
        if cleaned:
            return cleaned
    return ""


def _render_decision_log_entry(
    *,
    task: ContinuousTask,
    completed_at: datetime,
    cycle_status: str,
    material: CycleResearchMaterial,
    found_new_info: bool,
    next_step_reason: str,
    update_sent: bool,
    notification_reason: str = "",
    review_mode: bool = False,
    claim_mutation_counts: dict[str, int] | None = None,
    error: str | None = None,
) -> str:
    selected_urls = [f"  - {candidate.url}" for candidate in material.selected_candidates[:10]]
    lines = [
        f"## Cycle {completed_at.isoformat()}",
        f"- Task: {task.task_id} | {task.topic}",
        f"- Status: {cycle_status}",
        f"- Review mode: {'yes' if review_mode else 'no'}",
        f"- New info found: {'yes' if found_new_info else 'no'}",
        f"- User-facing update sent: {'yes' if update_sent else 'no'}",
        f"- Queries run: {', '.join(material.queries) if material.queries else 'None'}",
        f"- Selected source count: {len(material.selected_candidates)}",
        "- Selected URLs:",
        *(selected_urls or ["  - None"]),
        f"- Changed source count: {len(material.new_or_changed_documents)}",
        f"- Claim mutations: {claim_mutation_counts or {}}",
        f"- Notification reason: {notification_reason or 'No notification rationale recorded.'}",
        f"- Next step reason: {next_step_reason or 'Continue monitoring based on the latest cycle output.'}",
    ]
    if error:
        lines.append(f"- Error: {error}")

    return "\n".join(lines).strip() + "\n\n"


def _append_decision_log_entry(task_dir: Path, entry: str) -> None:
    decision_log_path = _decision_log_path(task_dir)
    try:
        if not decision_log_path.exists():
            decision_log_path.write_text(_render_decision_log_header(), encoding="utf-8")
        with decision_log_path.open("a", encoding="utf-8") as f:
            f.write(entry)
        logger.info("Appended continuous research decision log entry to %s", decision_log_path)
    except Exception as exc:
        logger.warning("Failed to append continuous research decision log entry to %s: %s", decision_log_path, exc)


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


def _load_claim_register(task_dir: Path) -> ClaimRegister:
    claim_register_path = _claim_register_json_path(task_dir)
    if not claim_register_path.exists():
        return _default_claim_register()
    try:
        return ClaimRegister.model_validate_json(claim_register_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Continuous research claim register at %s is unreadable: %s", claim_register_path, exc)
        return _default_claim_register()


def _save_claim_register(
    task_dir: Path,
    claim_register: ClaimRegister,
    *,
    notes: str = "",
) -> None:
    claim_register_path = _claim_register_json_path(task_dir)
    claim_register_path.write_text(claim_register.model_dump_json(indent=2), encoding="utf-8")
    _claim_register_markdown_path(task_dir).write_text(
        _render_claim_register_markdown(claim_register, notes=notes),
        encoding="utf-8",
    )


def _domain_from_title(title: str) -> str | None:
    cleaned = title.strip().lower()
    cleaned = re.sub(r"^https?://", "", cleaned).split("/", 1)[0].strip()
    if cleaned.startswith("www."):
        cleaned = cleaned[4:]
    if re.fullmatch(r"(?:[a-z0-9-]+\.)+[a-z]{2,}", cleaned):
        return cleaned
    return None


def _candidate_source_from_metadata(title: str, source: str | None, url: str) -> str | None:
    normalized_source = (source or "").lower().strip()
    if normalized_source.startswith("www."):
        normalized_source = normalized_source[4:]

    title_domain = _domain_from_title(title)
    if normalized_source in {"", "vertexaisearch.cloud.google.com"} and title_domain is not None:
        return title_domain
    if normalized_source:
        return normalized_source

    parsed_domain = urlparse(url).netloc.lower().strip()
    if parsed_domain.startswith("www."):
        parsed_domain = parsed_domain[4:]
    return parsed_domain or title_domain


def _candidate_domain(candidate: SearchCandidate | SourceDocument) -> str:
    source = _candidate_source_from_metadata(candidate.title, candidate.source, candidate.url) or ""
    source = source.lower().strip()
    if source.startswith("www."):
        source = source[4:]
    return source


def _domain_matches(domain: str, configured_domains: set[str]) -> bool:
    return any(domain == configured or domain.endswith(f".{configured}") for configured in configured_domains)


def _classify_source_tier(domain: str, source_policy: SourcePolicyConfig | None = None) -> str:
    normalized = domain.lower().strip()
    if normalized.startswith("www."):
        normalized = normalized[4:]

    if source_policy is not None:
        if _domain_matches(normalized, source_policy.preferred_domains):
            return "official"
        if _domain_matches(normalized, source_policy.discouraged_domains):
            return "low_trust"

    if any(marker in normalized for marker in REGULATORY_DOMAIN_MARKERS):
        return "regulatory"
    if any(marker in normalized for marker in LOW_TRUST_DOMAIN_MARKERS):
        return "low_trust"
    if any(marker in normalized for marker in PRIMARY_REPORTING_DOMAIN_MARKERS):
        return "primary_reporting"
    if any(marker in normalized for marker in SECONDARY_REPORTING_DOMAIN_MARKERS):
        return "secondary_reporting"
    if normalized.endswith(".org") or normalized.endswith(".edu"):
        return "primary_reporting"
    if normalized.endswith(".com") or normalized.endswith(".in") or normalized.endswith(".net"):
        return "secondary_reporting"
    return "secondary_reporting"


def _tokenize_text(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]{3,}", text.lower())
        if token not in QUESTION_STOPWORDS
    }


def _active_question_keywords(questions_text: str) -> set[str]:
    sections = _parse_markdown_sections(questions_text)
    active_questions = sections.get("Active Questions", "")
    return _tokenize_text(active_questions)


def _active_question_lines(questions_text: str) -> list[str]:
    sections = _parse_markdown_sections(questions_text)
    active_questions = sections.get("Active Questions", "")
    lines: list[str] = []
    for line in active_questions.splitlines():
        cleaned = re.sub(r"^[\-\*\d\.\)\[\]xX\s]+", "", line).strip()
        if cleaned:
            lines.append(cleaned)
    return lines


def _keyword_query_chunks(text: str, *, chunk_size: int = 4, max_chunks: int = 3) -> list[str]:
    tokens: list[str] = []
    seen: set[str] = set()
    for token in re.findall(r"[a-z0-9]{3,}", text.lower()):
        if token in QUESTION_STOPWORDS or token in seen:
            continue
        seen.add(token)
        tokens.append(token)
    chunks: list[str] = []
    for index in range(0, len(tokens), chunk_size):
        chunk = " ".join(tokens[index : index + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if len(chunks) >= max_chunks:
            break
    return chunks


def _clean_search_focus_line(line: str) -> str:
    cleaned = re.sub(r"^(?:[\-\*]\s+|\d+[.)]\s+|\[[ xX]\]\s+)+", "", line).strip()
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    for prefix in (
        "what is the current status of",
        "what is the current pricing for",
        "what is the confirmed pricing for",
        "what is the specific historical",
        "what is the specific",
        "what is the confirmed",
        "what is the status of",
        "what is the current",
        "what is the",
        "what are the",
        "are there",
        "how far is",
        "execute targeted searches for",
        "attempt to retrieve",
        "search for",
        "investigate",
        "check",
        "look for",
        "find",
        "review",
        "verify",
        "compare",
        "monitor",
        "track",
        "assess",
    ):
        if lowered.startswith(prefix + " "):
            cleaned = cleaned[len(prefix) :].strip()
            lowered = cleaned.lower()
            break

    for marker in (
        "to address",
        "to refine",
        "to verify",
        "to confirm",
        "to compare",
        "to assess",
        "to understand",
        "to check",
        "to validate",
        "so that",
    ):
        marker_index = lowered.find(f" {marker} ")
        if marker_index != -1:
            cleaned = cleaned[:marker_index].strip()
            lowered = cleaned.lower()
            break

    return cleaned.rstrip("?.!:;,")


def _query_site_filters(query: str) -> set[str]:
    return {match.group(0).lower() for match in re.finditer(r"site:[^\s]+", query.lower())}


def _load_previous_search_candidates(task_dir: Path) -> list[SearchCandidate]:
    candidate_pool_path = _candidate_pool_path(task_dir)
    if not candidate_pool_path.exists():
        return []

    try:
        payload = json.loads(candidate_pool_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Continuous research candidate pool at %s is unreadable: %s", candidate_pool_path, exc)
        return []

    candidates: list[SearchCandidate] = []
    for item in payload.get("candidates", []):
        try:
            candidates.append(SearchCandidate.model_validate(item))
        except Exception:
            continue
    return candidates


def _search_noise_terms(previous_candidates: list[SearchCandidate]) -> set[str]:
    if not previous_candidates:
        return set()

    counts: Counter[str] = Counter()
    for candidate in previous_candidates:
        haystack = " ".join([candidate.title, candidate.snippet, candidate.url]).lower()
        if re.search(r"\brent(?:al)?\b", haystack):
            counts["rent"] += 1
        if re.search(r"\bpg\b|paying guest", haystack):
            counts["pg"] += 1
        if re.search(r"\boffice\b|\bcommercial\b", haystack):
            counts["office"] += 1
        if re.search(r"\bhotel\b|\bhostel\b|booking\.com", haystack):
            counts["hotel"] += 1

    return {label for label, count in counts.items() if count >= 1}


def _search_noise_terms_from_text(text: str) -> set[str]:
    lowered = text.lower()
    markers: set[str] = set()
    if re.search(r"\brent(?:al)?\b", lowered):
        markers.add("rent")
    if re.search(r"\bpg\b|paying guest", lowered):
        markers.add("pg")
    if re.search(r"\boffice\b|\bcommercial\b", lowered):
        markers.add("office")
    if re.search(r"\bhotel\b|\bhostel\b|booking\.com", lowered):
        markers.add("hotel")
    return markers


def _focus_tokens(text: str, *, task_topic: str = "") -> set[str]:
    normalized = _clean_search_focus_line(text).lower()
    normalized = re.sub(r"\b(?:what|which|current|status|specific|confirmed|execute|targeted|search(?:es)?|attempt|retrieve)\b", " ", normalized)
    tokens = _tokenize_text(normalized)
    if task_topic:
        tokens -= _tokenize_text(task_topic)

    synonym_groups = {
        "inventory": {"flat", "flats", "apartment", "apartments", "unit", "units", "bhk"},
        "price": {"price", "pricing", "budget", "cr", "crore"},
        "water": {"kaveri", "water", "supply", "bwssb"},
        "flood": {"flood", "flooding", "stormwater", "drainage", "lake"},
        "metro": {"metro", "station", "walking", "route", "distance", "commute", "bmrcl"},
    }
    normalized_tokens: set[str] = set()
    for token in tokens:
        replacement = next((group for group, variants in synonym_groups.items() if token in variants), token)
        normalized_tokens.add(replacement)
    return normalized_tokens


def _focus_lines_are_near_duplicates(candidate: str, existing: list[str], *, task_topic: str) -> bool:
    candidate_tokens = _focus_tokens(candidate, task_topic=task_topic)
    if not candidate_tokens:
        return False

    for other in existing:
        other_tokens = _focus_tokens(other, task_topic=task_topic)
        if not other_tokens:
            continue
        smaller, larger = sorted((candidate_tokens, other_tokens), key=len)
        if smaller.issubset(larger) and len(smaller) >= 3:
            return True
        overlap_ratio = len(candidate_tokens.intersection(other_tokens)) / max(
            1,
            len(candidate_tokens.union(other_tokens)),
        )
        if overlap_ratio >= 0.7:
            return True
    return False


def _queries_are_near_duplicates(candidate: str, existing: list[str]) -> bool:
    candidate_lower = candidate.lower()
    candidate_sites = _query_site_filters(candidate_lower)
    candidate_tokens = _tokenize_text(re.sub(r"site:[^\s]+", " ", candidate_lower))

    for other in existing:
        other_lower = other.lower()
        if candidate_lower == other_lower:
            return True

        other_sites = _query_site_filters(other_lower)
        if candidate_sites or other_sites:
            if candidate_sites != other_sites:
                continue

        other_tokens = _tokenize_text(re.sub(r"site:[^\s]+", " ", other_lower))
        if not candidate_tokens or not other_tokens:
            continue

        smaller, larger = sorted((candidate_tokens, other_tokens), key=len)
        if smaller.issubset(larger) and len(larger - smaller) <= 2:
            return True

        overlap_ratio = len(candidate_tokens.intersection(other_tokens)) / max(
            1,
            len(candidate_tokens.union(other_tokens)),
        )
        if overlap_ratio >= 0.85:
            return True

    return False


def _compose_search_query(topic: str, focus: str) -> str:
    normalized_topic = " ".join(topic.split()).strip()
    cleaned_focus = _clean_search_focus_line(focus)
    if not cleaned_focus:
        return normalized_topic

    topic_tokens = _tokenize_text(normalized_topic)
    focus_tokens = _tokenize_text(cleaned_focus)
    focus_has_specific_anchor = (
        bool(re.search(r"\b(for|in|near|vs|versus)\b", cleaned_focus.lower()))
        or any(character.isdigit() for character in cleaned_focus)
    )
    if focus_has_specific_anchor and len(focus_tokens - topic_tokens) >= 3:
        return cleaned_focus

    return f"{normalized_topic} {cleaned_focus}".strip()


def _focused_question_query(
    focus: str,
    *,
    task: ContinuousTask,
    search_noise_terms: set[str],
) -> str:
    cleaned_focus = _clean_search_focus_line(focus)
    if not cleaned_focus:
        return " ".join(task.topic.split()).strip()

    raw_focus = focus.lower()
    tokens = _focus_tokens(cleaned_focus, task_topic=task.topic)
    query = cleaned_focus

    if {"inventory", "price"}.intersection(tokens) or any(
        marker in raw_focus for marker in ("price", "pricing", "budget", "bhk", "flat", "apartment", "unit")
    ):
        query = cleaned_focus
        if any(marker in raw_focus for marker in ("price", "pricing")) and not re.search(r"\bprice|pricing\b", query.lower()):
            query = f"{query} price"
        if "sale" not in query.lower():
            query = f"{query} sale resale apartment"
        negative_terms: list[str] = []
        if "rent" in search_noise_terms:
            negative_terms.extend(["-rent", "-rental"])
        if "pg" in search_noise_terms:
            negative_terms.append("-pg")
        if "office" in search_noise_terms:
            negative_terms.extend(["-office", "-commercial"])
        if "hotel" in search_noise_terms:
            negative_terms.extend(["-hotel", "-hostel"])
        if negative_terms:
            query = f"{query} {' '.join(negative_terms)}"
        return query

    if "water" in tokens:
        additions = [term for term in ("BWSSB", "Kaveri water", "connection", "supply") if term.lower() not in query.lower()]
        return f"{query} {' '.join(additions[:2])}".strip()

    if "flood" in tokens:
        additions = [term for term in ("flood map", "history", "BBMP", "stormwater") if term.lower() not in query.lower()]
        return f"{query} {' '.join(additions[:2])}".strip()

    if "metro" in tokens:
        additions = [term for term in ("BMRCL", "metro station", "walking distance", "map") if term.lower() not in query.lower()]
        return f"{query} {' '.join(additions[:2])}".strip()

    return _compose_search_query(task.topic, cleaned_focus)


def _is_bootstrap_cycle(task: ContinuousTask) -> bool:
    return task.cycle_count < BOOTSTRAP_CYCLE_COUNT


def _cycle_search_limits(task: ContinuousTask) -> tuple[int, int, int]:
    if _is_bootstrap_cycle(task):
        return (
            BOOTSTRAP_MAX_SEARCH_QUERIES,
            BOOTSTRAP_MAX_RESULTS_PER_QUERY,
            BOOTSTRAP_MAX_FETCHED_SOURCES_PER_CYCLE,
        )
    return (
        MAX_SEARCH_QUERIES,
        MAX_RESULTS_PER_QUERY,
        MAX_FETCHED_SOURCES_PER_CYCLE,
    )


def _question_relevance_score(candidate: SearchCandidate, question_keywords: set[str]) -> int:
    if not question_keywords:
        return 0
    candidate_text = " ".join(
        [
            candidate.query,
            candidate.title,
            candidate.snippet,
            _candidate_domain(candidate),
        ]
    )
    return len(question_keywords.intersection(_tokenize_text(candidate_text)))


def _prioritize_candidates(
    candidates: list[SearchCandidate],
    ledger: EvidenceLedger,
    *,
    questions_text: str | None = None,
    source_policy: SourcePolicyConfig | None = None,
) -> tuple[list[SearchCandidate], dict[str, str], dict[str, int]]:
    del source_policy
    deduped: dict[str, SearchCandidate] = {}
    for candidate in candidates:
        deduped.setdefault(_normalize_url(candidate.url), candidate)

    question_keywords = _active_question_keywords(questions_text or "")
    domain_frequency: Counter[str] = Counter(_candidate_domain(candidate) for candidate in deduped.values())

    def sort_key(candidate: SearchCandidate) -> tuple[int, int, int, float, int]:
        normalized_url = _normalize_url(candidate.url)
        relevance_rank = -_question_relevance_score(candidate, question_keywords)
        is_seen = 1 if normalized_url in ledger.records else 0
        crowded_domain_rank = domain_frequency[_candidate_domain(candidate)]
        published_rank = 0.0
        if candidate.published_at is not None:
            published_rank = -candidate.published_at.timestamp()
        position = candidate.position or 999
        return (relevance_rank, is_seen, crowded_domain_rank, published_rank, position)

    prioritized = sorted(deduped.values(), key=sort_key)
    return prioritized, {}, {}


def _claim_confidence_for_tiers(source_tiers: list[str]) -> float:
    if "official" in source_tiers:
        return 0.95
    if "regulatory" in source_tiers:
        return 0.9
    if "primary_reporting" in source_tiers:
        return 0.75
    if "secondary_reporting" in source_tiers:
        return 0.55
    return 0.35


def _claim_id_from_text(statement: str) -> str:
    digest = hashlib.sha256(statement.lower().encode("utf-8", errors="ignore")).hexdigest()
    return f"claim_{digest[:12]}"


def _merge_claim_records(
    existing: ClaimRegister,
    incoming_claims: list[ClaimRecord],
) -> tuple[ClaimRegister, dict[str, int]]:
    merged: dict[str, ClaimRecord] = {claim.id: claim for claim in existing.claims}
    counters: Counter[str] = Counter()

    for claim in incoming_claims:
        counters["total"] += 1
        if claim.status == "contradicted":
            counters["contradicted"] += 1
        if claim.status == "superseded":
            counters["superseded"] += 1

        prior = merged.get(claim.id)
        if prior is None:
            counters["created"] += 1
        elif prior.model_dump(mode="json") != claim.model_dump(mode="json"):
            counters["updated"] += 1
        merged[claim.id] = claim

        for contradicted_id in claim.contradicts:
            contradicted = merged.get(contradicted_id)
            if contradicted is not None and contradicted.status == "active":
                contradicted.status = "contradicted"
                contradicted.updated_at = claim.updated_at
                counters["contradicted"] += 1

    return ClaimRegister(claims=sorted(merged.values(), key=lambda record: (record.updated_at, record.id), reverse=True)), dict(counters)


def _build_fallback_claim_records(
    documents: list[SourceDocument],
    document_confidences: dict[str, DocumentConfidence],
    now: datetime,
) -> list[ClaimRecord]:
    fallback_claims: list[ClaimRecord] = []
    for document in documents:
        normalized_url = _normalize_url(document.url)
        document_confidence = document_confidences.get(normalized_url)
        confidence = document_confidence.confidence if document_confidence is not None else 0.5
        confidence_reason = (
            document_confidence.confidence_reason
            if document_confidence is not None
            else "Fallback claim derived from a changed document without structured confidence."
        )
        statement = document.title.strip() or document.snippet.strip() or document.url
        fallback_claims.append(
            ClaimRecord(
                id=_claim_id_from_text(statement),
                statement=statement,
                status="active",
                confidence=confidence,
                confidence_reason=confidence_reason,
                supporting_urls=[document.url],
                source_tiers=[],
                contradicts=[],
                updated_at=now,
                notes=f"Derived automatically from changed source document: {document.url}",
            )
        )
    return fallback_claims


def _render_known_sources_for_planner(ledger: EvidenceLedger) -> str:
    if not ledger.records:
        return "No tracked sources yet."

    records = sorted(ledger.records.values(), key=lambda record: record.last_checked_at, reverse=True)
    lines: list[str] = []
    for record in records[:MAX_KNOWN_SOURCES_IN_PROMPT]:
        next_check = record.next_check_after.isoformat() if record.next_check_after else "not scheduled"
        lines.append(f"- {record.title} | {record.url} | last_checked={record.last_checked_at.isoformat()} | next_check={next_check}")
    return "\n".join(lines)


def _build_query_planning_prompt(
    *,
    task: ContinuousTask,
    explicit_user_intent: str,
    questions_text: str,
    plan_text: str,
    summary_text: str,
    recent_decision_log: str,
    query_history: list[QueryHistoryEntry],
    ledger: EvidenceLedger,
    max_queries: int,
) -> str:
    return "\n\n".join(
        [
            "Plan the next web search queries for a continuous research monitoring cycle.",
            f"Query budget: choose at most {max_queries} queries.",
            f"Task topic: {task.topic}",
            "Explicit user intent:",
            _trim_for_prompt(explicit_user_intent or task.instructions, BRIEF_PROMPT_CHAR_LIMIT),
            "Living questions:",
            _trim_for_prompt(questions_text or _render_questions(task, explicit_user_intent), QUESTIONS_PROMPT_CHAR_LIMIT),
            "Current plan:",
            _trim_for_prompt(plan_text or task.plan or _default_plan(task.topic), PLAN_PROMPT_CHAR_LIMIT),
            "Current summary:",
            _trim_for_prompt(summary_text or task.last_summary or _default_summary(), SUMMARY_PROMPT_CHAR_LIMIT),
            "Recent query history:",
            _render_query_history_summary(query_history),
            "Known tracked sources:",
            _render_known_sources_for_planner(ledger),
            "Instructions:",
            (
                "Choose the queries yourself from the evidence and prior attempts. "
                "You may repeat a query only when the history justifies it. "
                "You may broaden, narrow, corroborate, or check contradictions. "
                "Do not assume a fixed domain profile and do not rely on prebuilt source reliability policy."
            ),
        ]
    )


def _continuous_query_planner_model() -> str:
    return gemini_model.get_balanced_model()


def _continuous_analysis_model() -> str:
    return gemini_model.get_large_model()


def get_query_planner_agent() -> Agent:
    api_key = get_env("GEMINI_API_KEY", required=True)
    return Agent(
        model=GoogleModel(
            _continuous_query_planner_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=(
            "You are a search planning agent for continuous research. "
            "Use factual task history to decide the next queries. "
            "Do not follow hardcoded topic profiles. Return only the structured search plan."
        ),
        output_type=SearchPlan,
    )


def _sanitize_planned_queries(plan: SearchPlan, *, max_queries: int) -> list[SearchQueryDecision]:
    decisions: list[SearchQueryDecision] = []
    seen: set[str] = set()
    for decision in plan.queries:
        query = " ".join(decision.query.split()).strip()[:200]
        if not query:
            continue
        normalized = query.lower()
        if normalized in seen:
            continue
        seen.add(normalized)
        decisions.append(decision.model_copy(update={"query": query}))
        if len(decisions) >= max_queries:
            break
    return decisions


async def _plan_search_queries(
    task: ContinuousTask,
    *,
    task_dir: Path,
    ledger: EvidenceLedger,
    user_search_context: str | None = None,
    plan_text: str | None = None,
    questions_text: str | None = None,
    recent_decision_log: str | None = None,
    max_queries: int | None = None,
    trace: Callable[..., None] | None = None,
) -> tuple[list[str], list[SearchQueryDecision], list[QueryHistoryEntry]]:
    limit = max_queries or MAX_SEARCH_QUERIES
    query_history = _load_query_history(task_dir)
    prompt = _build_query_planning_prompt(
        task=task,
        explicit_user_intent=(user_search_context or task.instructions),
        questions_text=questions_text or "",
        plan_text=plan_text or task.plan,
        summary_text=task.last_summary,
        recent_decision_log=recent_decision_log or "",
        query_history=query_history,
        ledger=ledger,
        max_queries=limit,
    )
    if trace is not None:
        trace(
            "query_planning_started",
            prompt_chars=len(prompt),
            query_history_count=len(query_history),
            max_queries=limit,
        )

    try:
        result = await get_query_planner_agent().run(prompt)
        output: SearchPlan | None = getattr(result, "data", getattr(result, "output", None))
        if output is None:
            raise RuntimeError("query planner returned no structured output")
        decisions = _sanitize_planned_queries(output, max_queries=limit)
    except Exception as exc:
        logger.warning("Continuous research query planner failed for task %s: %s", task.task_id, exc)
        fallback_queries = _build_search_queries(task, max_queries=limit)
        decisions = [
            SearchQueryDecision(
                query=query,
                reason=f"Mechanical fallback after planner failure: {exc}",
                expected_signal="General update signal.",
                query_role="new",
            )
            for query in fallback_queries
        ]
        if trace is not None:
            trace("query_planning_failed", level=logging.WARNING, error=str(exc))

    if not decisions:
        decisions = [
            SearchQueryDecision(
                query=query,
                reason="Mechanical fallback because planner produced no usable queries.",
                expected_signal="General update signal.",
                query_role="new",
            )
            for query in _build_search_queries(task, max_queries=limit)
        ]

    queries = [decision.query for decision in decisions]
    if trace is not None:
        trace(
            "queries_built",
            query_count=len(queries),
            queries=queries,
            query_decisions=[decision.model_dump(mode="json") for decision in decisions],
        )
    return queries, decisions, query_history


def _build_search_queries(
    task: ContinuousTask,
    *,
    user_search_context: str | None = None,
    plan_text: str | None = None,
    questions_text: str | None = None,
    source_policy_text: str | None = None,
    recent_decision_log: str | None = None,
    previous_candidates: list[SearchCandidate] | None = None,
    max_queries: int | None = None,
) -> list[str]:
    del user_search_context, plan_text, questions_text, source_policy_text, recent_decision_log, previous_candidates
    limit = max_queries or MAX_SEARCH_QUERIES
    topic = " ".join(task.topic.split()).strip()
    if not topic:
        return []
    queries = [topic]
    if limit > 1:
        queries.append(f"{topic} latest update")
    return queries[:limit]


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


async def _search_candidates(
    query: str,
    *,
    max_results: int = MAX_RESULTS_PER_QUERY,
    trace: Callable[..., None] | None = None,
) -> tuple[list[SearchCandidate], list[str], SearchObservation]:
    if trace is not None:
        trace(
            "search_started",
            query=query,
            max_results=max_results,
            tool="search_web_candidates",
        )
    search_result = await search_web_candidates(
        query,
        max_results=max_results,
        use_fallback=True,
    )
    candidates = [
        SearchCandidate(
            query=query,
            title=item.title,
            url=item.url,
            snippet=item.snippet,
            source=_candidate_source_from_metadata(item.title, item.source, item.url),
            published_at=item.published_at,
            provider=item.provider,
            position=item.position,
        )
        for item in search_result.candidates
    ]
    observation = SearchObservation(
        query=query,
        answer=search_result.answer or "",
        candidate_count=len(candidates),
        top_candidates=candidates[:max_results],
        errors=list(search_result.warnings),
    )
    if candidates:
        if trace is not None:
            trace(
                "search_completed",
                query=query,
                tool="search_web_candidates",
                providers_tried=search_result.providers_tried,
                warnings=search_result.warnings,
                answer_preview=_trace_detail_preview(search_result.answer, limit=300),
                candidate_count=len(candidates),
                candidates=_search_candidates_trace_payload(candidates),
            )
        return candidates, [], observation

    search_errors = [
        f"Search failed for '{query}': {warning}"
        for warning in search_result.warnings
    ]
    if not search_errors:
        search_errors.append(f"Search failed for '{query}': no search results returned")
    observation.errors = search_errors
    if trace is not None:
        trace(
            "search_completed",
            query=query,
            tool="search_web_candidates",
            providers_tried=search_result.providers_tried,
            warnings=search_result.warnings,
            answer_preview=_trace_detail_preview(search_result.answer, limit=300),
            candidate_count=0,
            errors=search_errors,
        )
    return [], search_errors, observation


async def _search_candidates_for_queries(
    queries: list[str],
    *,
    max_results: int = MAX_RESULTS_PER_QUERY,
    trace: Callable[..., None] | None = None,
) -> tuple[list[SearchCandidate], list[str], list[SearchObservation]]:
    candidates: list[SearchCandidate] = []
    search_errors: list[str] = []
    search_observations: list[SearchObservation] = []
    semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_SEARCH_QUERIES))

    async def _run_query(query: str) -> tuple[list[SearchCandidate], list[str], SearchObservation]:
        async with semaphore:
            try:
                return await _search_candidates(query, max_results=max_results, trace=trace)
            except Exception as exc:
                logger.warning("Search failed for query %r: %s", query, exc)
                error = f"Search failed for '{query}': {exc}"
                if trace is not None:
                    trace(
                        "search_completed",
                        query=query,
                        tool="search_web_candidates",
                        candidate_count=0,
                        errors=[error],
                    )
                return [], [error], SearchObservation(query=query, errors=[error])

    results = await asyncio.gather(*(_run_query(query) for query in queries))
    for found_candidates, query_errors, observation in results:
        candidates.extend(found_candidates)
        search_errors.extend(query_errors)
        search_observations.append(observation)

    return candidates, search_errors, search_observations


async def _fetch_source_document(
    candidate: SearchCandidate,
    *,
    trace: Callable[..., None] | None = None,
) -> tuple[SourceDocument | None, str | None]:
    if trace is not None:
        trace(
            "fetch_started",
            url=candidate.url,
            title=candidate.title,
            query=candidate.query,
            tool="read_pdf" if candidate.url.lower().endswith(".pdf") else "web_fetch",
        )
    try:
        if candidate.url.lower().endswith(".pdf"):
            content = await read_pdf(candidate.url)
        else:
            content = await web_fetch(candidate.url)
    except Exception as exc:
        if trace is not None:
            trace(
                "fetch_completed",
                url=candidate.url,
                title=candidate.title,
                query=candidate.query,
                success=False,
                error=str(exc),
            )
        return None, f"Fetch failed for query {candidate.query!r} at {candidate.url}: {exc}"

    if not content:
        if trace is not None:
            trace(
                "fetch_completed",
                url=candidate.url,
                title=candidate.title,
                query=candidate.query,
                success=False,
                error="no content returned",
            )
        return None, f"Fetch failed for query {candidate.query!r} at {candidate.url}: no content returned"
    if content.startswith("Failed to fetch or parse"):
        if trace is not None:
            trace(
                "fetch_completed",
                url=candidate.url,
                title=candidate.title,
                query=candidate.query,
                success=False,
                error=content,
            )
        return None, f"Fetch failed for query {candidate.query!r} at {candidate.url}: {content}"

    excerpt = _trim_for_prompt(content, MAX_SOURCE_EXCERPT_CHARS)
    content_hash = hashlib.sha256(content.encode("utf-8", errors="ignore")).hexdigest()
    if trace is not None:
        trace(
            "fetch_completed",
            url=candidate.url,
            title=candidate.title,
            query=candidate.query,
            success=True,
            content_chars=len(content),
            excerpt_chars=len(excerpt),
            content_hash=content_hash[:12],
        )

    return (
        SourceDocument(
            query=candidate.query,
            title=candidate.title,
            url=candidate.url,
            snippet=candidate.snippet,
            source=_candidate_domain(candidate),
            published_at=candidate.published_at,
            content_excerpt=excerpt,
            content_hash=content_hash,
        ),
        None,
    )


async def _fetch_source_documents(
    candidates: list[SearchCandidate],
    *,
    trace: Callable[..., None] | None = None,
) -> tuple[list[SourceDocument], list[str]]:
    if not candidates:
        return [], []

    semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_FETCHES_PER_CYCLE))

    async def _run_fetch(candidate: SearchCandidate) -> tuple[SourceDocument | None, str | None]:
        async with semaphore:
            return await _fetch_source_document(candidate, trace=trace)

    fetched_documents: list[SourceDocument] = []
    fetch_errors: list[str] = []
    fetch_results = await asyncio.gather(*(_run_fetch(candidate) for candidate in candidates))
    for document, fetch_error in fetch_results:
        if fetch_error:
            fetch_errors.append(fetch_error)
        elif document is not None:
            fetched_documents.append(document)

    return fetched_documents, fetch_errors


def _select_candidates_for_fetch(
    prioritized_candidates: list[SearchCandidate],
    ledger: EvidenceLedger,
    *,
    queries: list[str] | None = None,
    max_candidates: int = MAX_FETCHED_SOURCES_PER_CYCLE,
    trace: Callable[..., None] | None = None,
) -> list[SearchCandidate]:
    if max_candidates <= 0:
        return []

    now = _utc_now()
    selected: list[SearchCandidate] = []
    selected_urls: set[str] = set()
    eligible: list[SearchCandidate] = []
    skipped_recent = 0

    for candidate in prioritized_candidates:
        record = ledger.records.get(_normalize_url(candidate.url))
        if record is not None:
            if record.next_check_after is not None:
                if now < record.next_check_after:
                    skipped_recent += 1
                    continue
            elif (now - record.last_checked_at) < SOURCE_RECHECK_COOLDOWN:
                skipped_recent += 1
                continue
        eligible.append(candidate)

    candidates_by_query: dict[str, list[SearchCandidate]] = {}
    for candidate in eligible:
        candidates_by_query.setdefault(candidate.query, []).append(candidate)

    query_order = list(dict.fromkeys(queries or [candidate.query for candidate in prioritized_candidates]))

    def append_candidate(candidate: SearchCandidate) -> None:
        normalized_url = _normalize_url(candidate.url)
        if normalized_url in selected_urls:
            return
        selected.append(candidate)
        selected_urls.add(normalized_url)

    for query in query_order:
        if len(selected) >= max_candidates:
            break
        for candidate in candidates_by_query.get(query, []):
            append_candidate(candidate)
            break

    for candidate in eligible:
        if len(selected) >= max_candidates:
            break
        append_candidate(candidate)

    if skipped_recent:
        logger.info(
            "Skipped %s recently checked sources because they are still inside the %s cooldown.",
            skipped_recent,
            SOURCE_RECHECK_COOLDOWN,
        )
        if trace is not None:
            trace(
                "candidate_recheck_skipped",
                skipped_recent=skipped_recent,
                cooldown=str(SOURCE_RECHECK_COOLDOWN),
            )
    if trace is not None:
        trace(
            "candidates_selected_for_fetch",
            selected_count=len(selected),
            max_candidates=max_candidates,
            selected_query_count=len({candidate.query for candidate in selected}),
            represented_queries=list(dict.fromkeys(candidate.query for candidate in selected)),
            selected_candidates=_search_candidates_trace_payload(selected),
        )
    return selected


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
                change_count=1,
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
            record.change_count += 1
            new_or_changed.append(document)
        else:
            record.unchanged_count += 1

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
    remaining_excerpt_budget = MAX_CHANGED_DOC_PROMPT_CHARS
    for index, document in enumerate(documents, start=1):
        if remaining_excerpt_budget <= 0:
            excerpt = "[Excerpt omitted because the cycle excerpt budget was exhausted.]"
        else:
            excerpt = _trim_for_prompt(document.content_excerpt, min(MAX_SOURCE_EXCERPT_CHARS, remaining_excerpt_budget))
            remaining_excerpt_budget -= len(excerpt)
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
                    excerpt,
                ]
            )
        )
    return "\n\n".join(blocks)


def _render_search_observations_for_prompt(material: CycleResearchMaterial) -> str:
    if not material.search_observations:
        return "No per-query search observations were captured this cycle."

    blocks: list[str] = []
    for observation in material.search_observations:
        candidates = observation.top_candidates or [
            candidate for candidate in material.candidates if candidate.query == observation.query
        ][:MAX_RESULTS_PER_QUERY]
        candidate_lines = [
            f"  - {candidate.title} | {candidate.url} | domain={_candidate_domain(candidate)}"
            for candidate in candidates[:MAX_RESULTS_PER_QUERY]
        ]
        status = (
            f"candidates={observation.candidate_count}, "
            f"selected={observation.selected_count}, "
            f"fetched={observation.fetched_count}, "
            f"changed={observation.changed_count}"
        )
        answer = _trim_for_prompt(
            observation.answer or "No grounded answer text returned.",
            MAX_SEARCH_OBSERVATION_ANSWER_CHARS,
        )
        lines = [
            f"### Query: {observation.query}",
            f"Status: {status}",
            "Search-grounded answer:",
            answer,
            "Top candidates:",
            "\n".join(candidate_lines) if candidate_lines else "  - No candidates returned.",
        ]
        if observation.errors:
            lines.extend(["Errors:", "\n".join(f"  - {error}" for error in observation.errors[:5])])
        blocks.append("\n".join(lines))

    return _trim_for_prompt("\n\n".join(blocks), MAX_SEARCH_OBSERVATIONS_PROMPT_CHARS)


def _render_agent_working_context(task: ContinuousTask, artifacts: WorkspaceArtifacts) -> str:
    sections = [
        "# Agent Working Context",
        "",
        "This is the single assembled context for the research agent. Diagnostic files such as activity logs, cycle state, candidate pools, fetched-source JSON, analysis JSON, cycle reports, prior assessments, and decision-log audit entries are excluded from ordinary reasoning context.",
        "",
        "## Authoritative User Intent",
        f"User Request:\n{_trim_for_prompt(artifacts.user_request or 'No original user request recorded.', BRIEF_PROMPT_CHAR_LIMIT)}",
        f"Explicit User Intent (authoritative):\n{_trim_for_prompt(artifacts.explicit_user_intent or 'No explicit user intent recorded.', BRIEF_PROMPT_CHAR_LIMIT)}",
        f"Assistant Working Interpretation (not authoritative):\n{_trim_for_prompt(artifacts.assistant_working_interpretation or 'No assistant interpretation recorded.', BRIEF_PROMPT_CHAR_LIMIT)}",
        f"Known Constraints:\n{_trim_for_prompt(artifacts.known_constraints, PLAN_PROMPT_CHAR_LIMIT)}",
        f"Inferred Assumptions:\n{_trim_for_prompt(artifacts.inferred_assumptions, PLAN_PROMPT_CHAR_LIMIT)}",
        "",
        "## Research Memory",
        f"Current Summary:\n{_trim_for_prompt(artifacts.summary or _default_summary(), SUMMARY_PROMPT_CHAR_LIMIT)}",
        f"Previous Detailed Report:\n{_trim_for_prompt(artifacts.detailed_report or _default_detailed_report(task), DETAILED_REPORT_PROMPT_CHAR_LIMIT)}",
        "",
        "## Open Work",
        f"Plan:\n{_trim_for_prompt(artifacts.plan or _default_plan(task.topic), PLAN_PROMPT_CHAR_LIMIT)}",
        f"Living Questions:\n{_trim_for_prompt(artifacts.questions or _render_questions(task, artifacts.user_request), QUESTIONS_PROMPT_CHAR_LIMIT)}",
        "",
        "## Continuity And Search Memory",
        f"Claim Register Snapshot:\n{_trim_for_prompt(artifacts.claim_register_markdown or _render_claim_register_markdown(_default_claim_register()), CLAIM_REGISTER_PROMPT_CHAR_LIMIT)}",
        f"Recent Query History:\n{_trim_for_prompt(artifacts.query_history or 'No prior query history recorded.', DECISION_LOG_TAIL_CHAR_LIMIT)}",
    ]
    if artifacts.feedback:
        sections.extend(
            [
                "",
                "## New User Feedback",
                artifacts.feedback,
            ]
        )
    return "\n\n".join(sections)


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
            "Search Observations By Query:",
            _render_search_observations_for_prompt(material),
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
            "- Put richer source-backed detail in updated_detailed_report rather than overloading updated_summary.",
            "- Keep updated_plan to a short markdown checklist with next checks informed by this cycle.",
            "- Use Search Observations By Query to understand branches that produced signals but were not fetched; label those signals as search-grounded/unfetched until corroborated by fetched sources.",
            "- Distinguish unselected branches from fetch failures in updated_assessment and next checks.",
            "- If there are no new or changed source documents, set found_new_info=false.",
        ]
    )

    return "\n\n".join(prompt_parts)


def _build_workspace_cycle_prompt(
    task: ContinuousTask,
    artifacts: WorkspaceArtifacts,
    material: CycleResearchMaterial,
    *,
    review_mode: bool = False,
) -> str:
    prompt_parts = [
        f"Topic: {task.topic}",
        f"Cycle Mode: {'periodic_review' if review_mode else 'monitoring'}",
        _render_agent_working_context(task, artifacts),
    ]

    prompt_parts.extend(
        [
            "Queries Run This Cycle:",
            "\n".join(f"- {query}" for query in material.queries),
            "Search Observations By Query:",
            _render_search_observations_for_prompt(material),
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
            "- Treat only the Explicit User Intent section as authoritative user requirements.",
            "- You may propose interpretations and open questions, but do not rewrite explicit user intent.",
            "- Do not promote assistant interpretation or inferred assumptions into explicit user requirements.",
            "- Keep updated_questions as concise markdown with sections: Active Questions, Recently Resolved, Deferred Questions.",
            "- Maintain claim_records as the current authoritative claim register. Mark contradictions and supersessions explicitly.",
            "- Use updated_claim_register_markdown only for concise analyst notes that complement the claim register.",
            "- Maintain updated_detailed_report as a comprehensive living report that preserves useful details from the previous report and incorporates this cycle's evidence.",
            "- The detailed report should include source-backed findings, comparisons, tradeoffs, confidence/caveats, rejected or weak signals, watchlist items, open questions, and useful citations.",
            "- Do not make updated_detailed_report a raw chronological log; keep it organized as a current best understanding.",
            "- Maintain updated_assessment with sections: What Changed, What Is Genuinely New, What Matters To The User, What Remains Unresolved, Notification Decision.",
            "- Preserve unresolved questions that still matter; move items to Recently Resolved only when the latest evidence genuinely answers them.",
            "- Only set found_new_info=true if the new or changed source documents contain materially new, user-relevant information.",
            "- Use Search Observations By Query to understand every branch the search planner opened, including branches that were not selected or could not be fetched.",
            "- Treat search-grounded answers as weaker than fetched documents: they can guide reasoning, open questions, and next-cycle fetch plans, but important claims need fetched/corroborated evidence before becoming canonical.",
            "- Distinguish between no candidate returned, candidate not selected, fetch failed, and fetched evidence was weak.",
            "- Evaluate document_confidences from each document's provenance, directness, recency, specificity, independence, retrieval completeness, contradictions, and user relevance.",
            "- Do not use a prebuilt source reliability policy; reason from the fetched document content and context.",
            f"- Avoid turning claims below confidence {CANONICAL_SUMMARY_CONFIDENCE_THRESHOLD:.2f} into canonical summary facts; label them as unverified instead.",
            f"- Set should_notify=true only when a materially relevant claim has confidence at least {NOTIFICATION_CONFIDENCE_THRESHOLD:.2f}.",
            "- Set should_notify=true only when the change is materially relevant to the explicit user intent.",
            "- notification_reason must briefly explain the notification decision.",
            "- Optionally provide source_monitoring recheck suggestions in hours for fetched sources.",
            "- Use markdown links when citing sources.",
            "- Keep updated_summary concise and canonical rather than a running log.",
            "- Use updated_detailed_report for the details that are too granular for updated_summary.",
            "- Keep updated_plan to a short markdown checklist with next checks informed by this cycle.",
            "- If there are no new or changed source documents, set found_new_info=false.",
        ]
    )

    if review_mode:
        prompt_parts.extend(
            [
                "Periodic Review Requirements:",
                "- This is a periodic review pass. Reassess the brief, questions, and claim register for drift or stale assumptions.",
                "- You may provide updated_brief during review, but preserve the Explicit User Intent section exactly as written unless the user explicitly changed it.",
                "- Prune stale questions and retire dead claims if the workspace evidence supports it.",
            ]
        )
    else:
        prompt_parts.extend(
            [
                "Monitoring Mode Requirement:",
                "- Leave updated_brief empty unless a periodic review is explicitly requested.",
            ]
        )

    return "\n\n".join(prompt_parts)


@dataclass
class NotificationDecision:
    should_notify: bool
    reason: str
    outcome: str = "no_change"
    max_confidence: float = 0.0


def _safe_cycle_report_name(cycle_id: str, completed_at: datetime, outcome: str) -> str:
    base = cycle_id or f"cycle_{completed_at.strftime('%Y%m%dT%H%M%SZ')}"
    safe_base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base).strip("_") or "cycle"
    safe_outcome = re.sub(r"[^A-Za-z0-9_.-]+", "_", outcome).strip("_") or "unknown"
    return f"{safe_base}_{safe_outcome}.md"


def _render_query_plan_for_cycle_report(material: CycleResearchMaterial) -> str:
    if not material.queries:
        return "- No queries were run."

    decisions_by_query = {decision.query: decision for decision in material.query_decisions}
    candidate_counts = Counter(candidate.query for candidate in material.candidates)
    selected_counts = Counter(candidate.query for candidate in material.selected_candidates)
    fetched_counts = Counter(document.query for document in material.fetched_documents)
    changed_counts = Counter(document.query for document in material.new_or_changed_documents)

    lines: list[str] = []
    for query in material.queries:
        decision = decisions_by_query.get(query)
        lines.append(f"- `{query}`")
        if decision is not None and decision.reason:
            lines.append(f"  - Reason: {decision.reason}")
        if decision is not None and decision.expected_signal:
            lines.append(f"  - Expected signal: {decision.expected_signal}")
        if decision is not None and decision.query_role:
            lines.append(f"  - Role: {decision.query_role}")
        lines.append(
            "  - Results: "
            f"{candidate_counts[query]} candidates, "
            f"{selected_counts[query]} selected, "
            f"{fetched_counts[query]} fetched, "
            f"{changed_counts[query]} changed"
        )
    return "\n".join(lines)


def _render_document_confidences_for_cycle_report(output: CycleResult | None) -> str:
    if output is None or not output.document_confidences:
        return "- No structured document confidence was produced."

    lines: list[str] = []
    for confidence in output.document_confidences:
        lines.extend(
            [
                f"- {confidence.url}",
                f"  - Confidence: {confidence.confidence:.2f}",
                f"  - Evidence type: {confidence.evidence_type}",
                f"  - Reason: {confidence.confidence_reason or 'No reason provided.'}",
            ]
        )
        if confidence.limitations:
            lines.append(f"  - Limitations: {confidence.limitations}")
    return "\n".join(lines)


def _render_sources_for_cycle_report(material: CycleResearchMaterial) -> str:
    documents = material.new_or_changed_documents or material.fetched_documents
    if not documents:
        return "- No fetched source documents were available."

    lines: list[str] = []
    for index, document in enumerate(documents[:MAX_CYCLE_REPORT_SOURCES], start=1):
        published = document.published_at.isoformat() if document.published_at else "unknown"
        excerpt = _trim_for_prompt(
            document.content_excerpt.strip() or "No fetched excerpt available.",
            MAX_CYCLE_REPORT_SOURCE_EXCERPT_CHARS,
        )
        change_marker = "changed" if any(_normalize_url(document.url) == _normalize_url(changed.url) for changed in material.new_or_changed_documents) else "fetched"
        lines.extend(
            [
                f"### Source {index}: {document.title or document.url}",
                "",
                f"- URL: {document.url}",
                f"- Source: {document.source or _candidate_domain(document)}",
                f"- Query: `{document.query}`",
                f"- Published: {published}",
                f"- Cycle status: {change_marker}",
                "",
                "Excerpt:",
                "",
                "```text",
                excerpt,
                "```",
                "",
            ]
        )
    if len(documents) > MAX_CYCLE_REPORT_SOURCES:
        lines.append(f"- {len(documents) - MAX_CYCLE_REPORT_SOURCES} additional fetched sources omitted from this report.")
    return "\n".join(lines).strip()


def _render_errors_for_cycle_report(material: CycleResearchMaterial, error: str = "") -> str:
    errors = [*material.search_errors, *material.fetch_errors]
    if error:
        errors.append(error)
    if not errors:
        return "- No search, fetch, or analysis errors recorded."
    return "\n".join(f"- {_trim_for_prompt(item, 1000)}" for item in errors)


def _write_cycle_report(
    task: ContinuousTask,
    task_dir: Path,
    *,
    cycle_id: str,
    completed_at: datetime,
    status: str,
    outcome: str,
    material: CycleResearchMaterial,
    output: CycleResult | None = None,
    notification_decision: NotificationDecision | None = None,
    error: str = "",
    review_mode: bool = False,
) -> Path:
    reports_dir = _cycle_reports_dir_path(task_dir)
    reports_dir.mkdir(exist_ok=True)
    notification = notification_decision or NotificationDecision(False, "No notification decision recorded.", outcome=outcome)
    file_path = reports_dir / _safe_cycle_report_name(cycle_id, completed_at, outcome)

    analysis_lines = [
        f"- Found new info: {'yes' if output and output.found_new_info else 'no'}",
        f"- Model requested notification: {'yes' if output and output.should_notify else 'no'}",
        f"- App notification decision: {'yes' if notification.should_notify else 'no'}",
        f"- Notification reason: {notification.reason}",
        f"- Max confidence: {notification.max_confidence:.2f}",
    ]
    if output is not None and output.new_findings.strip():
        analysis_lines.extend(["", "### New Findings", "", output.new_findings.strip()])
    elif error:
        analysis_lines.extend(["", "### New Findings", "", "Structured analysis was unavailable because the analysis stage failed. Review the fetched source excerpts below."])
    else:
        analysis_lines.extend(["", "### New Findings", "", "No structured new findings were produced."])

    if output is not None and output.updated_assessment.strip():
        analysis_lines.extend(["", "### Assessment", "", output.updated_assessment.strip()])

    report = "\n".join(
        [
            f"# Continuous Research Cycle Report: {cycle_id or 'unknown'}",
            "",
            f"- Task ID: {task.task_id}",
            f"- Topic: {task.topic}",
            f"- Completed: {completed_at.astimezone().isoformat()}",
            f"- Status: {status}",
            f"- Outcome: {outcome}",
            f"- Review mode: {'yes' if review_mode else 'no'}",
            "",
            "## Collection Summary",
            f"- Queries run: {len(material.queries)}",
            f"- Search candidates: {len(material.candidates)}",
            f"- Selected candidates: {len(material.selected_candidates)}",
            f"- Fetched documents: {len(material.fetched_documents)}",
            f"- New or changed documents: {len(material.new_or_changed_documents)}",
            "",
            "## Query Plan And Results",
            _render_query_plan_for_cycle_report(material),
            "",
            "## Search Observations",
            _render_search_observations_for_prompt(material),
            "",
            "## Analysis",
            "\n".join(analysis_lines),
            "",
            "## Document Confidence",
            _render_document_confidences_for_cycle_report(output),
            "",
            "## Source Excerpts",
            _render_sources_for_cycle_report(material),
            "",
            "## Errors",
            _render_errors_for_cycle_report(material, error=error),
            "",
        ]
    )
    file_path.write_text(report.strip() + "\n", encoding="utf-8")
    return file_path


def _is_review_cycle(task: ContinuousTask) -> bool:
    return task.cycle_count > 0 and (task.cycle_count + 1) % REVIEW_CYCLE_INTERVAL == 0


def _replace_markdown_section(markdown_text: str, section_name: str, new_body: str) -> str:
    pattern = re.compile(
        rf"(^## {re.escape(section_name)}\n)(.*?)(?=^## |\Z)",
        flags=re.MULTILINE | re.DOTALL,
    )
    if pattern.search(markdown_text):
        return pattern.sub(lambda match: f"{match.group(1)}{new_body.strip()}\n\n", markdown_text, count=1).rstrip() + "\n"
    return markdown_text.rstrip() + f"\n\n## {section_name}\n{new_body.strip()}\n"


def _preserve_explicit_intent_in_brief(existing_brief: str, proposed_brief: str) -> str:
    if not proposed_brief.strip():
        return existing_brief

    existing_sections = _parse_markdown_sections(existing_brief)
    proposed_sections = _parse_markdown_sections(proposed_brief)
    if "Explicit User Intent" not in proposed_sections:
        return _replace_markdown_section(
            proposed_brief,
            "Explicit User Intent",
            existing_sections.get("Explicit User Intent", ""),
        )

    return _replace_markdown_section(
        proposed_brief,
        "Explicit User Intent",
        existing_sections.get("Explicit User Intent", proposed_sections.get("Explicit User Intent", "")),
    )


def _changed_document_tiers(material: CycleResearchMaterial) -> dict[str, str]:
    changed_tiers: dict[str, str] = {}
    for document in material.new_or_changed_documents:
        normalized_url = _normalize_url(document.url)
        changed_tiers[normalized_url] = material.selected_candidate_tiers.get(
            normalized_url,
            _classify_source_tier(_candidate_domain(document)),
        )
    return changed_tiers


def _document_confidence_by_url(output: CycleResult) -> dict[str, DocumentConfidence]:
    confidences: dict[str, DocumentConfidence] = {}
    for document_confidence in output.document_confidences:
        confidences[_normalize_url(document_confidence.url)] = document_confidence
    return confidences


def _max_changed_document_confidence(output: CycleResult, material: CycleResearchMaterial) -> float:
    confidences = _document_confidence_by_url(output)
    max_confidence = 0.0
    for document in material.new_or_changed_documents:
        confidence = confidences.get(_normalize_url(document.url))
        if confidence is not None:
            max_confidence = max(max_confidence, confidence.confidence)
    for claim in output.claim_records:
        max_confidence = max(max_confidence, claim.confidence)
    return max_confidence


def _assessment_ties_to_explicit_intent(assessment: str, explicit_user_intent: str) -> bool:
    intent_tokens = _tokenize_text(explicit_user_intent)
    if not intent_tokens:
        return True
    assessment_tokens = _tokenize_text(assessment)
    return bool(intent_tokens.intersection(assessment_tokens))


def _notification_support_satisfies_policy(material: CycleResearchMaterial) -> bool:
    changed_tiers = _changed_document_tiers(material)
    if any(tier in {"official", "regulatory"} for tier in changed_tiers.values()):
        return True

    supported_urls = {
        normalized_url
        for normalized_url, tier in changed_tiers.items()
        if tier != "low_trust"
    }
    return len(supported_urls) >= 2


def _should_freeze_summary_for_support_gap(material: CycleResearchMaterial) -> bool:
    changed_tiers = _changed_document_tiers(material)
    return bool(changed_tiers) and all(tier == "low_trust" for tier in changed_tiers.values())


def _should_freeze_summary_for_confidence_gap(output: CycleResult, material: CycleResearchMaterial) -> bool:
    return bool(material.new_or_changed_documents) and _max_changed_document_confidence(output, material) < CANONICAL_SUMMARY_CONFIDENCE_THRESHOLD


def _clamp_recheck_hours(hours: int) -> int:
    return min(MAX_SOURCE_RECHECK_HOURS, max(MIN_SOURCE_RECHECK_HOURS, hours))


def _apply_source_monitoring_decisions(task_dir: Path, output: CycleResult) -> None:
    if not output.source_monitoring:
        return
    ledger = _load_evidence_ledger(task_dir)
    now = _utc_now()
    changed = False
    for decision in output.source_monitoring:
        record = ledger.records.get(_normalize_url(decision.url))
        if record is None:
            continue
        hours = _clamp_recheck_hours(decision.suggested_recheck_after_hours)
        record.next_check_after = now + timedelta(hours=hours)
        record.recheck_reason = decision.reason
        changed = True
    if changed:
        _save_evidence_ledger(task_dir, ledger)


def _evaluate_notification_gate(
    *,
    output: CycleResult,
    material: CycleResearchMaterial,
    explicit_user_intent: str,
) -> NotificationDecision:
    max_confidence = _max_changed_document_confidence(output, material)
    if not output.should_notify:
        if output.found_new_info and material.new_or_changed_documents and max_confidence >= NOTIFICATION_CONFIDENCE_THRESHOLD:
            return NotificationDecision(
                False,
                output.notification_reason.strip() or "Model did not request notification.",
                outcome="verified_update",
                max_confidence=max_confidence,
            )
        outcome = "unverified_signal" if output.found_new_info and material.new_or_changed_documents else "no_change"
        return NotificationDecision(
            False,
            output.notification_reason.strip() or "Model did not request notification.",
            outcome=outcome,
            max_confidence=max_confidence,
        )

    if not material.new_or_changed_documents:
        return NotificationDecision(False, "No changed source documents were available for notification.", outcome="no_change", max_confidence=max_confidence)

    if not output.found_new_info:
        return NotificationDecision(
            False,
            "Notification suppressed because the cycle did not mark the change as new.",
            outcome="no_change",
            max_confidence=max_confidence,
        )

    if max_confidence < NOTIFICATION_CONFIDENCE_THRESHOLD:
        return NotificationDecision(
            False,
            f"Notification suppressed because evidence confidence {max_confidence:.2f} is below {NOTIFICATION_CONFIDENCE_THRESHOLD:.2f}.",
            outcome="unverified_signal",
            max_confidence=max_confidence,
        )

    assessment_text = output.updated_assessment.strip()
    if not _assessment_ties_to_explicit_intent(assessment_text, explicit_user_intent):
        return NotificationDecision(
            False,
            "Notification suppressed because the assessment did not tie the change to explicit user intent.",
            outcome="verified_update",
            max_confidence=max_confidence,
        )

    return NotificationDecision(
        True,
        output.notification_reason.strip() or "Material change satisfied notification policy.",
        outcome="notified_update",
        max_confidence=max_confidence,
    )


def _normalize_output_claim_records(
    *,
    output: CycleResult,
    material: CycleResearchMaterial,
    now: datetime,
) -> list[ClaimRecord]:
    if output.claim_records:
        normalized_claims: list[ClaimRecord] = []
        for claim in output.claim_records:
            claim_copy = claim.model_copy(deep=True)
            if claim_copy.status not in CLAIM_STATUSES:
                claim_copy.status = "active"
            claim_copy.updated_at = claim_copy.updated_at or now
            normalized_claims.append(claim_copy)
        return normalized_claims

    return _build_fallback_claim_records(material.new_or_changed_documents, _document_confidence_by_url(output), now)


def _fallback_assessment_markdown(
    *,
    output: CycleResult,
    material: CycleResearchMaterial,
    notification_decision: NotificationDecision | None = None,
) -> str:
    changed_lines = (
        [f"- {document.title} ({document.url})" for document in material.new_or_changed_documents]
        if material.new_or_changed_documents
        else ["- No new or changed source documents were detected this cycle."]
    )
    decision = notification_decision or NotificationDecision(False, output.notification_reason.strip() or "No notification decision recorded.")
    genuine_newness = (
        "- The cycle identified materially new information."
        if output.found_new_info
        else "- The cycle did not confirm materially new information."
    )
    user_impact = (
        "- The change was material enough to notify the user."
        if decision.should_notify
        else "- The change was not strong enough to justify a user-facing notification."
    )
    unresolved = _first_checklist_item(output.updated_plan) or "Continue monitoring based on the current plan."

    return "\n".join(
        [
            "# Continuous Research Assessment",
            "",
            "## What Changed",
            *changed_lines,
            "",
            "## What Is Genuinely New",
            genuine_newness,
            "",
            "## What Matters To The User",
            user_impact,
            "",
            "## What Remains Unresolved",
            f"- {unresolved}",
            "",
            "## Notification Decision",
            f"- Notify: {'yes' if decision.should_notify else 'no'}",
            f"- Reason: {decision.reason}",
            "",
        ]
    )


def _should_skip_analysis_for_no_change(
    *,
    material: CycleResearchMaterial,
    user_feedback: str,
    review_mode: bool,
    task: ContinuousTask,
) -> bool:
    return (
        not material.new_or_changed_documents
        and not user_feedback
        and not review_mode
        and not material.search_errors
        and not material.fetch_errors
        and getattr(task, "unverified_signal_count", 0) == 0
    )


async def _collect_cycle_material(
    task: ContinuousTask,
    task_dir: Path,
    *,
    cycle_id: str = "",
    user_search_context: str | None = None,
    plan_text: str | None = None,
    questions_text: str | None = None,
    source_policy_text: str | None = None,
    recent_decision_log: str | None = None,
    trace: Callable[..., None] | None = None,
) -> CycleResearchMaterial:
    ledger = _load_evidence_ledger(task_dir)
    max_queries, max_results_per_query, max_fetched_sources = _cycle_search_limits(task)
    queries, query_decisions, query_history = await _plan_search_queries(
        task,
        task_dir=task_dir,
        ledger=ledger,
        user_search_context=user_search_context,
        plan_text=plan_text,
        questions_text=questions_text,
        recent_decision_log=recent_decision_log,
        max_queries=max_queries,
        trace=trace,
    )
    if trace is not None:
        trace(
            "search_budget_prepared",
            search_limits={
                "max_queries": max_queries,
                "max_results_per_query": max_results_per_query,
                "max_fetched_sources": max_fetched_sources,
            },
        )

    search_stage_timeout = _search_stage_timeout(len(queries))

    try:
        candidates, search_errors, search_observations = await asyncio.wait_for(
            _search_candidates_for_queries(
                queries,
                max_results=max_results_per_query,
                trace=trace,
            ),
            timeout=search_stage_timeout.total_seconds(),
        )
    except asyncio.TimeoutError as exc:
        if trace is not None:
            trace(
                "search_timeout",
                level=logging.ERROR,
                query_count=len(queries),
                timeout_seconds=int(search_stage_timeout.total_seconds()),
            )
        raise RuntimeError(f"Search stage timed out after {int(search_stage_timeout.total_seconds())} seconds.") from exc

    prioritized_candidates, candidate_tiers, tier_counts = _prioritize_candidates(
        candidates,
        ledger,
        questions_text=questions_text,
    )
    selected_candidates = _select_candidates_for_fetch(
        prioritized_candidates,
        ledger,
        queries=queries,
        max_candidates=max_fetched_sources,
        trace=trace,
    )
    selected_candidate_tiers: dict[str, str] = {}
    _persist_candidate_pool_artifact(
        task,
        task_dir,
        cycle_id=cycle_id,
        queries=queries,
        query_decisions=query_decisions,
        search_observations=search_observations,
        candidates=candidates,
        prioritized_candidates=prioritized_candidates,
        selected_candidates=selected_candidates,
        search_errors=search_errors,
    )
    if trace is not None:
        trace(
            "candidate_prioritization_completed",
            candidate_count=len(candidates),
            prioritized_count=len(prioritized_candidates),
            prioritized_candidates=_search_candidates_trace_payload(prioritized_candidates),
        )
    try:
        fetched_documents, fetch_errors = await asyncio.wait_for(
            _fetch_source_documents(selected_candidates, trace=trace),
            timeout=FETCH_STAGE_TIMEOUT.total_seconds(),
        )
    except asyncio.TimeoutError as exc:
        if trace is not None:
            trace(
                "fetch_timeout",
                level=logging.ERROR,
                selected_candidate_count=len(selected_candidates),
                timeout_seconds=int(FETCH_STAGE_TIMEOUT.total_seconds()),
            )
        raise RuntimeError(f"Fetch stage timed out after {int(FETCH_STAGE_TIMEOUT.total_seconds())} seconds.") from exc

    new_or_changed_documents = _merge_documents_into_ledger(ledger, fetched_documents)
    search_observations = _annotate_search_observations(
        search_observations,
        queries=queries,
        candidates=candidates,
        selected_candidates=selected_candidates,
        fetched_documents=fetched_documents,
        changed_documents=new_or_changed_documents,
        search_errors=search_errors,
        fetch_errors=fetch_errors,
    )
    _persist_candidate_pool_artifact(
        task,
        task_dir,
        cycle_id=cycle_id,
        queries=queries,
        query_decisions=query_decisions,
        search_observations=search_observations,
        candidates=candidates,
        prioritized_candidates=prioritized_candidates,
        selected_candidates=selected_candidates,
        search_errors=search_errors,
    )
    _save_evidence_ledger(task_dir, ledger)
    _persist_fetched_sources_artifact(
        task,
        task_dir,
        cycle_id=cycle_id,
        fetch_errors=fetch_errors,
        fetched_documents=fetched_documents,
        changed_documents=new_or_changed_documents,
    )
    if trace is not None:
        trace(
            "ledger_updated",
            fetched_document_count=len(fetched_documents),
            changed_document_count=len(new_or_changed_documents),
            changed_documents=_source_documents_trace_payload(new_or_changed_documents),
            search_error_count=len(search_errors),
            fetch_error_count=len(fetch_errors),
        )

    return CycleResearchMaterial(
        queries=queries,
        search_observations=search_observations,
        candidates=candidates,
        selected_candidates=selected_candidates,
        fetched_documents=fetched_documents,
        new_or_changed_documents=new_or_changed_documents,
        search_errors=search_errors,
        fetch_errors=fetch_errors,
        ledger=ledger,
        selected_candidate_tiers=selected_candidate_tiers,
        source_tier_counts={},
        query_decisions=query_decisions,
        query_history=query_history,
    )


def get_cycle_agent() -> Agent:
    api_key = get_env("GEMINI_API_KEY", required=True)
    return Agent(
        model=GoogleModel(
            _continuous_analysis_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=(
            "You are a Continuous Research Analyst. "
            "You review newly fetched source material for a monitoring task and decide whether anything is materially new. "
            "Be conservative about novelty. Repeated coverage, minor rewrites, or unchanged sources are not new information. "
            "When there is new information, summarize it crisply and cite the supporting URLs with markdown links. "
            "Always maintain a concise canonical summary, a comprehensive detailed report, a short markdown plan, a living question set, "
            "a current assessment, per-document confidence, and a claim register. "
            "Do not rewrite explicit user intent. "
            "Evaluate confidence from the actual fetched documents and mark contradictions or supersessions explicitly."
        ),
        output_type=CycleResult,
    )


def _is_transient_analysis_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "503",
            "unavailable",
            "high demand",
            "temporarily",
            "timeout",
            "timed out",
            "rate limit",
            "resource exhausted",
        )
    )


async def _run_cycle_analysis_with_retries(
    prompt: str,
    *,
    trace: Callable[..., None] | None = None,
) -> CycleResult:
    retry_delays = list(ANALYSIS_TRANSIENT_RETRY_DELAYS_SECONDS)
    max_attempts = len(retry_delays) + 1
    last_error: BaseException | None = None

    for attempt_number in range(1, max_attempts + 1):
        try:
            result = await asyncio.wait_for(
                get_cycle_agent().run(prompt),
                timeout=ANALYSIS_STAGE_TIMEOUT.total_seconds(),
            )
            output: CycleResult | None = getattr(result, "data", getattr(result, "output", None))
            if output is None:
                raise RuntimeError("Continuous research cycle returned no structured output.")
            return output
        except asyncio.TimeoutError as exc:
            last_error = RuntimeError(
                f"Analysis stage timed out after {int(ANALYSIS_STAGE_TIMEOUT.total_seconds())} seconds."
            )
            if trace is not None:
                trace(
                    "analysis_timeout",
                    level=logging.ERROR,
                    attempt=attempt_number,
                    timeout_seconds=int(ANALYSIS_STAGE_TIMEOUT.total_seconds()),
                )
        except Exception as exc:
            last_error = exc

        if (
            attempt_number < max_attempts
            and last_error is not None
            and _is_transient_analysis_error(last_error)
        ):
            delay_seconds = retry_delays[attempt_number - 1]
            if trace is not None:
                trace(
                    "analysis_retry_scheduled",
                    level=logging.WARNING,
                    attempt=attempt_number,
                    next_attempt=attempt_number + 1,
                    delay_seconds=delay_seconds,
                    error=str(last_error),
                )
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)
            continue

        break

    if last_error is None:
        raise RuntimeError("Analysis stage failed without an error.")
    raise last_error


async def _run_continuous_cycle(task: ContinuousTask) -> str:
    logger.info("Running continuous cycle for task %s (%s)", task.task_id, task.topic)
    task_dir = OUTPUT_DIR / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    feedback_path = _ensure_feedback_file(task_dir)
    _ensure_activity_log_file(task_dir)
    task_lock = _get_task_lock(task.task_id)
    workspace_enabled = _workspace_enabled(task, task_dir)
    if task.is_cycle_running and not task_lock.locked():
        _recover_orphaned_cycle_state(task, task_dir)

    cycle_state = _load_cycle_runtime_state(task, task_dir)
    cycle_id = ""

    def save_cycle_state() -> None:
        _save_cycle_runtime_state(task_dir, cycle_state)

    def trace(event: str, *, level: int = logging.INFO, **details: object) -> None:
        nonlocal cycle_state
        _trace_continuous_research(task, task_dir, event, level=level, **details)
        if cycle_id:
            cycle_state = _update_cycle_runtime_state_from_trace(
                cycle_state,
                event=event,
                details=details,
            )
            save_cycle_state()

    if task_lock.locked():
        logger.info("Skipping overlapping cycle for task %s", task.task_id)
        trace("cycle_skipped", reason="already_running", workspace_enabled=workspace_enabled)
        return "skipped_already_running"

    async with task_lock:
        if task.status != "running":
            logger.info("Skipping cycle for task %s because status is %s", task.task_id, task.status)
            trace("cycle_skipped", reason="status_not_running", current_status=task.status, workspace_enabled=workspace_enabled)
            return "skipped_not_running"

        started_at = _utc_now()
        cycle_id = f"cycle_{started_at.strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"
        cycle_state = CycleRuntimeState(
            task_id=task.task_id,
            cycle_id=cycle_id,
            status="running",
            stage="starting",
            started_at=started_at,
            last_heartbeat_at=started_at,
        )
        save_cycle_state()
        global_continuous_state.update_task(
            task.task_id,
            is_cycle_running=True,
            last_cycle_started_at=started_at,
        )
        trace(
            "cycle_started",
            cycle_number=task.cycle_count + 1,
            workspace_enabled=workspace_enabled,
            status=task.status,
        )
        feedback_snapshot = _safe_read_text(
            feedback_path,
            description="feedback",
            default=FEEDBACK_PLACEHOLDER,
        ).strip()
        user_feedback = ""
        if feedback_snapshot and feedback_snapshot != FEEDBACK_PLACEHOLDER.strip():
            user_feedback = feedback_snapshot

        workspace_artifacts: WorkspaceArtifacts | None = None
        existing_claim_register = _default_claim_register()
        review_mode = False
        if workspace_enabled:
            workspace_artifacts = _load_workspace_artifacts(task, task_dir)
            user_feedback = workspace_artifacts.feedback
            existing_claim_register = _load_claim_register(task_dir)
            review_mode = _is_review_cycle(task)
            logger.info(
                "Continuous research workspace cycle for task %s: review_mode=%s",
                task.task_id,
                review_mode,
            )
            trace(
                "workspace_context_loaded",
                review_mode=review_mode,
                feedback_present=bool(user_feedback),
                explicit_intent_preview=_trace_detail_preview(workspace_artifacts.explicit_user_intent, limit=240),
                current_plan_preview=_trace_detail_preview(workspace_artifacts.plan, limit=240),
                current_summary_preview=_trace_detail_preview(workspace_artifacts.summary, limit=240),
            )
        else:
            trace(
                "legacy_context_loaded",
                feedback_present=bool(user_feedback),
                current_plan_preview=_trace_detail_preview(task.plan or _default_plan(task.topic), limit=240),
                current_summary_preview=_trace_detail_preview(task.last_summary or _default_summary(), limit=240),
            )

        material = CycleResearchMaterial()
        claim_mutation_counts: dict[str, int] = {}
        notification_decision = NotificationDecision(False, "Legacy path does not use notification gating.")

        try:
            material = await _collect_cycle_material(
                task,
                task_dir,
                cycle_id=cycle_id,
                user_search_context=workspace_artifacts.explicit_user_intent if workspace_artifacts else None,
                plan_text=workspace_artifacts.plan if workspace_artifacts else None,
                questions_text=workspace_artifacts.questions if workspace_artifacts else None,
                recent_decision_log=workspace_artifacts.recent_decision_log if workspace_artifacts else None,
                trace=trace,
            )
            if not material.fetched_documents and (material.search_errors or material.fetch_errors):
                error_text = "\n".join(material.search_errors + material.fetch_errors)
                raise RuntimeError(f"Unable to collect usable source material this cycle.\n{error_text}")

            if _should_skip_analysis_for_no_change(
                material=material,
                user_feedback=user_feedback,
                review_mode=review_mode,
                task=task,
            ):
                completed_at = _utc_now()
                task.no_new_findings_count += 1
                status = "paused" if task.no_new_findings_count >= AUTO_PAUSE_AFTER_EMPTY_CYCLES else task.status
                _persist_analysis_result_artifact(
                    task,
                    task_dir,
                    cycle_id=cycle_id,
                    status="skipped_no_changes",
                    error="No new or changed source documents; analysis skipped.",
                )
                _append_query_history(
                    task_dir,
                    _query_history_entries_for_cycle(
                        cycle_id=cycle_id,
                        queries=material.queries,
                        candidates=material.candidates,
                        selected_candidates=material.selected_candidates,
                        fetched_documents=material.fetched_documents,
                        changed_documents=material.new_or_changed_documents,
                        search_errors=material.search_errors,
                        fetch_errors=material.fetch_errors,
                        outcome="no_change",
                    ),
                )
                global_continuous_state.update_task(
                    task.task_id,
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
                    trace(
                        "user_update_sent",
                        update_type="auto_paused_after_empty_cycles",
                        no_new_findings_count=task.no_new_findings_count,
                    )
                if workspace_enabled:
                    _append_decision_log_entry(
                        task_dir,
                        _render_decision_log_entry(
                            task=task,
                            completed_at=completed_at,
                            cycle_status="no_change",
                            material=material,
                            found_new_info=False,
                            next_step_reason="No new or changed source documents were detected.",
                            update_sent=False,
                            notification_reason="Analysis skipped because no source changed.",
                            review_mode=review_mode,
                            claim_mutation_counts=claim_mutation_counts,
                        ),
                    )
                report_path = _write_cycle_report(
                    task,
                    task_dir,
                    cycle_id=cycle_id,
                    completed_at=completed_at,
                    status="skipped_no_changes",
                    outcome="no_change",
                    material=material,
                    notification_decision=NotificationDecision(
                        False,
                        "Analysis skipped because no source changed.",
                        outcome="no_change",
                    ),
                    review_mode=review_mode,
                )
                cycle_state.status = "completed"
                cycle_state.stage = "completed"
                cycle_state.completed_at = completed_at
                cycle_state.last_heartbeat_at = completed_at
                cycle_state.result = "no_change"
                cycle_state.notification_reason = "Analysis skipped because no source changed."
                save_cycle_state()
                trace(
                    "cycle_completed",
                    result="no_change",
                    no_new_findings_count=task.no_new_findings_count,
                    status=status,
                    cycle_report=report_path.name,
                )
                return "no_change"

            prompt = (
                _build_workspace_cycle_prompt(task, workspace_artifacts, material, review_mode=review_mode)
                if workspace_artifacts is not None
                else _build_cycle_prompt(task, user_feedback, material)
            )
            trace(
                "analysis_prompt_prepared",
                mode="workspace" if workspace_artifacts is not None else "legacy",
                review_mode=review_mode,
                prompt_chars=len(prompt),
                selected_candidate_count=len(material.selected_candidates),
                changed_document_count=len(material.new_or_changed_documents),
                selected_candidates=_search_candidates_trace_payload(material.selected_candidates),
                changed_documents=_source_documents_trace_payload(material.new_or_changed_documents),
            )
            trace(
                "analysis_started",
                model=_continuous_analysis_model(),
            )
            output = await _run_cycle_analysis_with_retries(prompt, trace=trace)
            _persist_analysis_result_artifact(
                task,
                task_dir,
                cycle_id=cycle_id,
                status="completed",
                output=output,
            )
            trace(
                "analysis_completed",
                found_new_info=output.found_new_info,
                should_notify=output.should_notify,
                notification_reason=output.notification_reason,
                updated_summary_preview=_trace_detail_preview(output.updated_summary, limit=300),
                updated_plan_preview=_trace_detail_preview(output.updated_plan, limit=300),
                updated_assessment_preview=_trace_detail_preview(output.updated_assessment, limit=300),
                supporting_urls=output.supporting_urls,
            )

            completed_at = _utc_now()
            next_step_reason = _first_checklist_item(output.updated_plan)

            if workspace_enabled and workspace_artifacts is not None:
                current_brief = _safe_read_text(
                    _brief_path(task_dir),
                    description="brief",
                    default=_render_brief(task, _task_user_request(task)),
                )
                next_questions = (
                    output.updated_questions.strip()
                    if output.updated_questions.strip()
                    else workspace_artifacts.questions or _render_questions(task, _task_user_request(task))
                )

                notification_decision = _evaluate_notification_gate(
                    output=output,
                    material=material,
                    explicit_user_intent=workspace_artifacts.explicit_user_intent,
                )
                trace(
                    "notification_gate_evaluated",
                    should_notify=notification_decision.should_notify,
                    reason=notification_decision.reason,
                    outcome=notification_decision.outcome,
                    max_confidence=notification_decision.max_confidence,
                )
                next_assessment = (
                    output.updated_assessment.strip()
                    if output.updated_assessment.strip()
                    else _fallback_assessment_markdown(
                        output=output,
                        material=material,
                        notification_decision=notification_decision,
                    )
                )

                incoming_claims = (
                    _normalize_output_claim_records(output=output, material=material, now=completed_at)
                    if output.claim_records or material.new_or_changed_documents
                    else []
                )
                merged_claim_register = existing_claim_register
                if incoming_claims:
                    merged_claim_register, claim_mutation_counts = _merge_claim_records(existing_claim_register, incoming_claims)
                elif output.updated_claim_register_markdown.strip():
                    claim_mutation_counts = {"notes_only": 1}

                if incoming_claims or output.updated_claim_register_markdown.strip():
                    _save_claim_register(
                        task_dir,
                        merged_claim_register,
                        notes=output.updated_claim_register_markdown,
                    )

                next_brief = current_brief
                if review_mode and output.updated_brief.strip():
                    next_brief = _preserve_explicit_intent_in_brief(current_brief, output.updated_brief.strip())
                    _brief_path(task_dir).write_text(next_brief.strip() + "\n", encoding="utf-8")
                elif output.updated_brief.strip():
                    logger.info("Ignoring updated_brief outside review mode for task %s", task.task_id)

                task.plan = output.updated_plan
                if _should_freeze_summary_for_confidence_gap(output, material) and not notification_decision.should_notify:
                    task.last_summary = workspace_artifacts.summary or _default_summary()
                else:
                    task.last_summary = output.updated_summary

                _plan_path(task_dir).write_text(task.plan, encoding="utf-8")
                _summary_path(task_dir).write_text(task.last_summary, encoding="utf-8")
                next_detailed_report = (
                    output.updated_detailed_report.strip()
                    if output.updated_detailed_report.strip()
                    else workspace_artifacts.detailed_report or _default_detailed_report(task)
                )
                _detailed_report_path(task_dir).write_text(next_detailed_report.strip() + "\n", encoding="utf-8")
                _questions_path(task_dir).write_text(next_questions.strip() + "\n", encoding="utf-8")
                _assessment_path(task_dir).write_text(next_assessment.strip() + "\n", encoding="utf-8")
                _apply_source_monitoring_decisions(task_dir, output)

                logger.info(
                    "Continuous research decision for task %s: notify=%s reason=%s claim_mutations=%s",
                    task.task_id,
                    notification_decision.should_notify,
                    notification_decision.reason,
                    claim_mutation_counts,
                )
                cycle_state.stage = "commit"
                cycle_state.last_heartbeat_at = _utc_now()
                cycle_state.selected_candidate_count = len(material.selected_candidates)
                cycle_state.fetched_document_count = len(material.fetched_documents)
                cycle_state.changed_document_count = len(material.new_or_changed_documents)
                cycle_state.notification_reason = notification_decision.reason
                save_cycle_state()
                trace(
                    "workspace_artifacts_updated",
                    claim_mutation_counts=claim_mutation_counts,
                    summary_frozen=(
                        _should_freeze_summary_for_confidence_gap(output, material) and not notification_decision.should_notify
                    ),
                    updated_questions_preview=_trace_detail_preview(next_questions, limit=240),
                    updated_assessment_preview=_trace_detail_preview(next_assessment, limit=240),
                    updated_detailed_report_preview=_trace_detail_preview(next_detailed_report, limit=240),
                )
            else:
                task.last_summary = output.updated_summary
                task.plan = output.updated_plan
                _plan_path(task_dir).write_text(task.plan, encoding="utf-8")
                _summary_path(task_dir).write_text(task.last_summary, encoding="utf-8")
                if output.updated_detailed_report.strip():
                    _detailed_report_path(task_dir).write_text(output.updated_detailed_report.strip() + "\n", encoding="utf-8")
                _apply_source_monitoring_decisions(task_dir, output)
                trace(
                    "legacy_artifacts_updated",
                    updated_plan_preview=_trace_detail_preview(task.plan, limit=240),
                    updated_summary_preview=_trace_detail_preview(task.last_summary, limit=240),
                )

            cycle_report_outcome = (
                notification_decision.outcome
                if workspace_enabled
                else ("notified_update" if output.found_new_info else "no_change")
            )
            report_notification_decision = (
                notification_decision
                if workspace_enabled
                else NotificationDecision(
                    output.found_new_info,
                    output.notification_reason.strip() or "Legacy path sends updates when new info is found.",
                    outcome=cycle_report_outcome,
                    max_confidence=_max_changed_document_confidence(output, material),
                )
            )
            cycle_report_path = _write_cycle_report(
                task,
                task_dir,
                cycle_id=cycle_id,
                completed_at=completed_at,
                status="completed",
                outcome=cycle_report_outcome,
                material=material,
                output=output,
                notification_decision=report_notification_decision,
                review_mode=review_mode,
            )
            trace(
                "cycle_report_written",
                file_name=cycle_report_path.name,
                outcome=cycle_report_outcome,
                source_count=len(material.new_or_changed_documents or material.fetched_documents),
            )

            if user_feedback:
                current_feedback = _safe_read_text(
                    feedback_path,
                    description="feedback",
                    default=FEEDBACK_PLACEHOLDER,
                ).strip()
                if current_feedback == feedback_snapshot:
                    feedback_path.write_text(FEEDBACK_PLACEHOLDER, encoding="utf-8")
                    trace("feedback_cleared", reason="feedback_consumed_by_cycle")

            should_send_update = output.found_new_info
            if workspace_enabled:
                should_send_update = output.found_new_info and notification_decision.should_notify

            if should_send_update:
                task.no_new_findings_count = 0
                _append_query_history(
                    task_dir,
                    _query_history_entries_for_cycle(
                        cycle_id=cycle_id,
                        queries=material.queries,
                        candidates=material.candidates,
                        selected_candidates=material.selected_candidates,
                        fetched_documents=material.fetched_documents,
                        changed_documents=material.new_or_changed_documents,
                        search_errors=material.search_errors,
                        fetch_errors=material.fetch_errors,
                        outcome="notified_update",
                    ),
                )
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
                trace(
                    "update_written",
                    file_name=file_name,
                    supporting_urls=output.supporting_urls,
                    new_findings_preview=_trace_detail_preview(output.new_findings, limit=400),
                )

                global_continuous_state.update_task(
                    task.task_id,
                    last_summary=task.last_summary,
                    plan=task.plan,
                    no_new_findings_count=0,
                    unverified_signal_count=0,
                    suppressed_notification_count=0,
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
                trace(
                    "user_update_sent",
                    update_type="new_information",
                    next_step_reason=next_step_reason,
                    notification_reason=notification_decision.reason if workspace_enabled else output.notification_reason,
                )
                if workspace_enabled:
                    _append_decision_log_entry(
                        task_dir,
                        _render_decision_log_entry(
                            task=task,
                            completed_at=completed_at,
                            cycle_status="notified_update",
                            material=material,
                            found_new_info=output.found_new_info,
                            next_step_reason=next_step_reason,
                            update_sent=True,
                            notification_reason=notification_decision.reason if workspace_enabled else output.notification_reason,
                            review_mode=review_mode,
                            claim_mutation_counts=claim_mutation_counts,
                        ),
                    )
                cycle_state.status = "completed"
                cycle_state.stage = "completed"
                cycle_state.completed_at = completed_at
                cycle_state.last_heartbeat_at = completed_at
                cycle_state.result = "notified_update"
                cycle_state.notification_reason = notification_decision.reason if workspace_enabled else output.notification_reason
                save_cycle_state()
                trace(
                    "cycle_completed",
                    result="notified_update",
                    next_step_reason=next_step_reason,
                    no_new_findings_count=0,
                )
                return "notified_update"

            outcome = notification_decision.outcome if workspace_enabled else ("verified_update" if output.found_new_info else "no_change")
            if outcome == "no_change":
                task.no_new_findings_count += 1
            status = "paused" if task.no_new_findings_count >= AUTO_PAUSE_AFTER_EMPTY_CYCLES else task.status
            unverified_signal_count = task.unverified_signal_count + 1 if outcome == "unverified_signal" else task.unverified_signal_count
            suppressed_notification_count = (
                task.suppressed_notification_count + 1
                if output.should_notify and not notification_decision.should_notify
                else task.suppressed_notification_count
            )
            _append_query_history(
                task_dir,
                _query_history_entries_for_cycle(
                    cycle_id=cycle_id,
                    queries=material.queries,
                    candidates=material.candidates,
                    selected_candidates=material.selected_candidates,
                    fetched_documents=material.fetched_documents,
                    changed_documents=material.new_or_changed_documents,
                    search_errors=material.search_errors,
                    fetch_errors=material.fetch_errors,
                    outcome=outcome,
                ),
            )
            global_continuous_state.update_task(
                task.task_id,
                last_summary=task.last_summary,
                plan=task.plan,
                no_new_findings_count=task.no_new_findings_count,
                unverified_signal_count=unverified_signal_count,
                suppressed_notification_count=suppressed_notification_count,
                cycle_count=task.cycle_count + 1,
                failure_count=0,
                last_error="",
                is_cycle_running=False,
                last_cycle_completed_at=completed_at,
                last_new_info_at=completed_at if outcome == "verified_update" else task.last_new_info_at,
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
                trace(
                    "user_update_sent",
                    update_type="auto_paused_after_empty_cycles",
                    no_new_findings_count=task.no_new_findings_count,
                )
            if workspace_enabled:
                _append_decision_log_entry(
                    task_dir,
                    _render_decision_log_entry(
                        task=task,
                        completed_at=completed_at,
                        cycle_status=outcome,
                        material=material,
                        found_new_info=output.found_new_info,
                        next_step_reason=next_step_reason,
                        update_sent=(status == "paused"),
                        notification_reason=notification_decision.reason if workspace_enabled else output.notification_reason,
                        review_mode=review_mode,
                        claim_mutation_counts=claim_mutation_counts,
                        ),
                    )
            cycle_state.status = "completed"
            cycle_state.stage = "completed"
            cycle_state.completed_at = completed_at
            cycle_state.last_heartbeat_at = completed_at
            cycle_state.result = outcome
            cycle_state.notification_reason = notification_decision.reason if workspace_enabled else output.notification_reason
            save_cycle_state()
            trace(
                "cycle_completed",
                result=outcome,
                next_step_reason=next_step_reason,
                no_new_findings_count=task.no_new_findings_count,
                unverified_signal_count=unverified_signal_count,
                status=status,
                notification_reason=notification_decision.reason if workspace_enabled else output.notification_reason,
            )
            return outcome

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
            _persist_analysis_result_artifact(
                task,
                task_dir,
                cycle_id=cycle_id,
                status="failed",
                error=str(exc),
            )
            try:
                report_path = _write_cycle_report(
                    task,
                    task_dir,
                    cycle_id=cycle_id,
                    completed_at=completed_at,
                    status="failed",
                    outcome="failed",
                    material=material,
                    notification_decision=NotificationDecision(
                        False,
                        str(exc),
                        outcome="failed",
                    ),
                    error=str(exc),
                    review_mode=review_mode,
                )
                trace(
                    "cycle_report_written",
                    file_name=report_path.name,
                    outcome="failed",
                    source_count=len(material.new_or_changed_documents or material.fetched_documents),
                )
            except Exception as report_exc:
                logger.warning("Failed to write continuous research cycle report for task %s: %s", task.task_id, report_exc)
            cycle_state.status = "failed"
            cycle_state.stage = "failed"
            cycle_state.completed_at = completed_at
            cycle_state.last_heartbeat_at = completed_at
            cycle_state.error = str(exc)
            cycle_state.result = "failed"
            save_cycle_state()
            trace(
                "cycle_failed",
                level=logging.ERROR,
                error=str(exc),
                failure_count=failure_count,
                status=status,
            )

            event = TelegramMessageEvent(**task.event_dict)
            if status == "paused":
                await send_proactive_update(
                    event,
                    (
                        f"Continuous research for '{task.topic}' hit repeated errors and has been paused.\n"
                        f"Latest error: {exc}"
                    ),
                )
                trace(
                    "user_update_sent",
                    update_type="auto_paused_after_failures",
                    failure_count=failure_count,
                    error=str(exc),
                )
            if workspace_enabled:
                _append_decision_log_entry(
                    task_dir,
                    _render_decision_log_entry(
                        task=task,
                        completed_at=completed_at,
                        cycle_status="failed",
                        material=material,
                        found_new_info=False,
                        next_step_reason=(
                            "Pause the task after repeated failures."
                            if status == "paused"
                            else "Retry on the next cycle after the failure."
                        ),
                        update_sent=(status == "paused"),
                        notification_reason=str(exc),
                        review_mode=review_mode,
                        claim_mutation_counts=claim_mutation_counts,
                        error=str(exc),
                    ),
                )
            _append_query_history(
                task_dir,
                _query_history_entries_for_cycle(
                    cycle_id=cycle_id,
                    queries=material.queries,
                    candidates=material.candidates,
                    selected_candidates=material.selected_candidates,
                    fetched_documents=material.fetched_documents,
                    changed_documents=material.new_or_changed_documents,
                    search_errors=material.search_errors,
                    fetch_errors=material.fetch_errors + [str(exc)],
                    outcome="failed",
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
        workspace_version=FOUNDATION_WORKSPACE_VERSION,
        source_event_id=event.event_id,
        source_message_id=event.message_id,
        source_channel_id=event.channel_id,
        event_dict=event.model_dump(mode="json"),
        plan=_default_plan(topic),
    )
    global_continuous_state.add_task(task)

    task_dir = OUTPUT_DIR / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    _ensure_workspace_artifacts(task, task_dir, event=event)
    _trace_continuous_research(
        task,
        task_dir,
        "task_started",
        user_request=_trace_detail_preview(event.message, limit=300),
        instructions=_trace_detail_preview(instructions, limit=300),
        workspace_version=task.workspace_version,
    )

    asyncio.create_task(_run_continuous_cycle(task))
    return (
        f"Started continuous research on '{topic}'. Task ID is {task_id}. "
        f"It maintains `{task_id}/{DETAILED_REPORT_FILE_NAME}` and writes cycle notes in `{task_id}/{CYCLE_REPORTS_DIR_NAME}/`; "
        f"use `{task_id}/feedback.md` for direct feedback that should be folded into the next cycle. "
        "I will notify you only when I find something genuinely new."
    )


async def stop_continuous_research(ctx: RunContext[TelegramMessageEvent], task_id: str) -> str:
    """Stop a continuous research task permanently."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    previous_status = task.status
    global_continuous_state.update_task(task_id, status="stopped")
    _trace_continuous_research(task, OUTPUT_DIR / task.task_id, "task_stopped", previous_status=previous_status)
    return f"Stopped continuous research task {task_id} ('{task.topic}')."


async def pause_continuous_research(ctx: RunContext[TelegramMessageEvent], task_id: str) -> str:
    """Pause a continuous research task. It can be resumed later."""
    task = global_continuous_state.get_task(task_id)
    if not task:
        return f"Task {task_id} not found."

    previous_status = task.status
    global_continuous_state.update_task(task_id, status="paused")
    _trace_continuous_research(task, OUTPUT_DIR / task.task_id, "task_paused", previous_status=previous_status)
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
    _trace_continuous_research(
        task,
        OUTPUT_DIR / task.task_id,
        "task_resumed",
        had_new_instructions=bool(new_instructions),
        new_instructions=_trace_detail_preview(new_instructions or "", limit=300),
    )

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
    _trace_continuous_research(
        task,
        task_dir,
        "feedback_recorded",
        feedback_preview=_trace_detail_preview(feedback, limit=300),
        task_status=task.status,
    )

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

    _trace_continuous_research(task, OUTPUT_DIR / task.task_id, "manual_cycle_triggered")
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
        task_dir = OUTPUT_DIR / task.task_id
        detailed_report_path = _detailed_report_path(task_dir)
        if detailed_report_path.exists():
            response_lines.append(f"  **Detailed Report**: {detailed_report_path}")
        feedback_path = _ensure_feedback_file(task_dir)
        response_lines.append(f"  **Feedback Inbox**: {feedback_path}")
        reports_dir = _cycle_reports_dir_path(task_dir)
        if reports_dir.exists():
            latest_reports = sorted(reports_dir.glob("*.md"), key=lambda path: path.stat().st_mtime, reverse=True)
            if latest_reports:
                response_lines.append(f"  **Latest Cycle Report**: {latest_reports[0]}")
        if task.last_error:
            response_lines.append(f"  **Last Error**: {_trim_for_prompt(task.last_error, 200)}")
        response_lines.append("")

    return "\n".join(response_lines).rstrip()
