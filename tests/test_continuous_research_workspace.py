from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent.continuous_research as continuous_research
import agent.continuous_state as continuous_state_module
from agent.continuous_state import ContinuousResearchState, ContinuousTask
from models import TelegramMessageEvent


@dataclass
class WorkspaceTestEnv:
    output_dir: Path
    state: ContinuousResearchState


@pytest.fixture
def workspace_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> WorkspaceTestEnv:
    output_dir = tmp_path / "continuous_research_output"
    state_file = output_dir / ".state.json"

    monkeypatch.setattr(continuous_state_module, "STATE_FILE", state_file)
    monkeypatch.setattr(continuous_research, "OUTPUT_DIR", output_dir)

    state = ContinuousResearchState()
    monkeypatch.setattr(continuous_research, "global_continuous_state", state)

    return WorkspaceTestEnv(output_dir=output_dir, state=state)


def build_event(
    *,
    message: str = "Keep tracking apartments near Embassy Tech Village with good appreciation potential.",
    event_id: str = "evt-1",
) -> TelegramMessageEvent:
    return TelegramMessageEvent(
        event_id=event_id,
        source="telegram",
        message=message,
        channel_id=101,
        sender_id=202,
        message_id=303,
    )


def build_task(
    *,
    event: TelegramMessageEvent,
    task_id: str = "cr_test",
    topic: str = "Embassy Tech Village apartments",
    instructions: str = "Track 3BHK units with east-facing views and no servant room.",
    workspace_version: str = "",
    plan: str = "",
    last_summary: str = "",
) -> ContinuousTask:
    return ContinuousTask(
        task_id=task_id,
        topic=topic,
        instructions=instructions,
        status="running",
        workspace_version=workspace_version,
        source_event_id=event.event_id,
        source_message_id=event.message_id,
        source_channel_id=event.channel_id,
        event_dict=event.model_dump(mode="json"),
        plan=plan,
        last_summary=last_summary,
    )


def install_workspace_task(
    env: WorkspaceTestEnv,
    task: ContinuousTask,
    *,
    event: TelegramMessageEvent | None = None,
) -> Path:
    env.state.add_task(task)
    task_dir = env.output_dir / task.task_id
    continuous_research._ensure_workspace_artifacts(task, task_dir, event=event)
    return task_dir


class FakeCycleAgent:
    def __init__(self, output: continuous_research.CycleResult, prompt_log: list[str]) -> None:
        self.output = output
        self.prompt_log = prompt_log

    async def run(self, prompt: str) -> SimpleNamespace:
        self.prompt_log.append(prompt)
        return SimpleNamespace(output=self.output)


def patch_cycle_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    material: continuous_research.CycleResearchMaterial,
    output: continuous_research.CycleResult,
    prompt_log: list[str],
    sent_updates: list[str],
) -> None:
    async def fake_collect_cycle_material(*args, **kwargs):
        return material

    async def fake_send_proactive_update(event: TelegramMessageEvent, message: str) -> None:
        sent_updates.append(message)

    monkeypatch.setattr(continuous_research, "_collect_cycle_material", fake_collect_cycle_material)
    monkeypatch.setattr(continuous_research, "get_cycle_agent", lambda: FakeCycleAgent(output, prompt_log))
    monkeypatch.setattr(continuous_research, "send_proactive_update", fake_send_proactive_update)


@pytest.mark.asyncio
async def test_start_continuous_research_bootstraps_workspace_and_anchors_intent(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(
        message=(
            "I want to keep looking for flats near Embassy Tech Village in the 2-4 cr range "
            "that are good value and likely to appreciate."
        ),
        event_id="evt-bootstrap",
    )
    tool_instructions = (
        "Track luxury 3BHK units. Require east-facing layouts, no servant room, and 10th+ floor options."
    )

    def fake_create_task(coro):
        coro.close()
        return SimpleNamespace()

    monkeypatch.setattr(continuous_research.asyncio, "create_task", fake_create_task)

    response = await continuous_research.start_continuous_research(
        SimpleNamespace(deps=event),
        "Luxury apartments near Embassy Tech Village",
        tool_instructions,
    )

    tasks = workspace_env.state.get_all_tasks()
    assert len(tasks) == 1
    task = tasks[0]
    task_dir = workspace_env.output_dir / task.task_id

    assert "Started continuous research" in response
    assert task.workspace_version == continuous_research.FOUNDATION_WORKSPACE_VERSION

    for relative_path in (
        continuous_research.BRIEF_FILE_NAME,
        continuous_research.WORKSPACE_MANIFEST_FILE_NAME,
        continuous_research.DECISION_LOG_FILE_NAME,
        continuous_research.ACTIVITY_LOG_FILE_NAME,
        continuous_research.CYCLE_STATE_FILE_NAME,
        continuous_research.CANDIDATE_POOL_FILE_NAME,
        continuous_research.FETCHED_SOURCES_FILE_NAME,
        continuous_research.ANALYSIS_RESULT_FILE_NAME,
        continuous_research.QUESTIONS_FILE_NAME,
        continuous_research.PLAN_FILE_NAME,
        continuous_research.SUMMARY_FILE_NAME,
        continuous_research.DETAILED_REPORT_FILE_NAME,
        continuous_research.ASSESSMENT_FILE_NAME,
        continuous_research.QUERY_HISTORY_FILE_NAME,
        continuous_research.EVIDENCE_FILE_NAME,
        f"{continuous_research.EVIDENCE_DIR_NAME}/{continuous_research.CLAIM_REGISTER_JSON_FILE_NAME}",
        f"{continuous_research.EVIDENCE_DIR_NAME}/{continuous_research.CLAIM_REGISTER_MD_FILE_NAME}",
        "feedback.md",
    ):
        assert (task_dir / relative_path).exists(), relative_path
    assert (task_dir / "updates").is_dir()
    assert (task_dir / continuous_research.CYCLE_REPORTS_DIR_NAME).is_dir()

    brief_sections = continuous_research._parse_markdown_sections(
        (task_dir / continuous_research.BRIEF_FILE_NAME).read_text(encoding="utf-8")
    )
    assert event.message in brief_sections["User Request"]
    assert event.message in brief_sections["Explicit User Intent"]
    assert "east-facing" not in brief_sections["Explicit User Intent"].lower()
    assert "east-facing" in brief_sections["Assistant Working Interpretation"].lower()

    manifest = (task_dir / continuous_research.WORKSPACE_MANIFEST_FILE_NAME).read_text(encoding="utf-8")
    assert "`brief.md` is authoritative for user intent" in manifest
    assert "Agent-Visible Working Memory" in manifest
    assert "Diagnostic And Audit Artifacts" in manifest
    assert "assembled into a single Agent Working Context" in manifest
    assert "`feedback.md` is a transient feedback inbox" in manifest
    assert "assistant tools may append user feedback there" in manifest
    assert "`decision_log.md` is the append-only audit history" in manifest
    assert "`activity.log` is the live execution trace" in manifest
    assert "`cycle_state.json` is the live per-cycle execution state" in manifest
    assert "`questions.md`" in manifest
    assert "`source_policy.md`" not in manifest
    assert "`query_history.jsonl`" in manifest
    assert "`cycle_reports/`" in manifest
    assert "`detailed_report.md`" in manifest
    assert "`evidence/claim_register.json`" in manifest
    assert "`assessment.md`" in manifest

    questions_text = (task_dir / continuous_research.QUESTIONS_FILE_NAME).read_text(encoding="utf-8")
    assert "## Active Questions" in questions_text
    assert "## Recently Resolved" in questions_text
    assert "## Deferred Questions" in questions_text

    assert not (task_dir / continuous_research.SOURCE_POLICY_FILE_NAME).exists()

    cycle_state = continuous_research.CycleRuntimeState.model_validate_json(
        (task_dir / continuous_research.CYCLE_STATE_FILE_NAME).read_text(encoding="utf-8")
    )
    assert cycle_state.status == "idle"
    assert cycle_state.stage == "idle"

    claim_register_text = (
        task_dir / continuous_research.EVIDENCE_DIR_NAME / continuous_research.CLAIM_REGISTER_MD_FILE_NAME
    ).read_text(encoding="utf-8")
    assert "## Active Claims" in claim_register_text


@pytest.mark.asyncio
async def test_workspace_enabled_cycle_uses_bounded_prompt_context(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-workspace")
    task = build_task(
        event=event,
        task_id="cr_workspace",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)
    (task_dir / "rogue.md").write_text("THIS SHOULD NOT APPEAR", encoding="utf-8")

    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        selected_candidates=[
            continuous_research.SearchCandidate(
                query="embassy tech village apartments latest updates",
                title="Example Listing",
                url="https://example.com/listing",
            )
        ],
        new_or_changed_documents=[
            continuous_research.SourceDocument(
                query="embassy tech village apartments latest updates",
                title="Example Listing",
                url="https://example.com/listing",
                content_excerpt="Listing content changed but did not contain material updates.",
                content_hash="hash-workspace-context",
            )
        ],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="Nothing materially new.",
        updated_summary="Stable summary after bounded context review.",
        updated_plan="- [ ] Review official developer sites next cycle",
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    assert not sent_updates
    assert len(prompt_log) == 1
    prompt = prompt_log[0]
    assert "Explicit User Intent (authoritative):" in prompt
    assert "Assistant Working Interpretation (not authoritative):" in prompt
    assert "Living Questions:" in prompt
    assert "Source Policy:" not in prompt
    assert "Claim Register Snapshot:" in prompt
    assert "Recent Query History:" in prompt
    assert "Previous Detailed Report:" in prompt
    assert "Search Observations By Query:" in prompt
    assert "updated_detailed_report" in prompt
    assert "Evaluate document_confidences" in prompt
    assert "Do not promote assistant interpretation or inferred assumptions" in prompt
    assert "THIS SHOULD NOT APPEAR" not in prompt


@pytest.mark.asyncio
async def test_legacy_cycle_path_remains_unchanged_without_manifest(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-legacy")
    task = build_task(
        event=event,
        task_id="cr_legacy",
        instructions="Legacy instructions that should still appear in the legacy prompt.",
    )
    workspace_env.state.add_task(task)
    task_dir = workspace_env.output_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    continuous_research._ensure_feedback_file(task_dir)
    continuous_research._save_evidence_ledger(task_dir, continuous_research.EvidenceLedger())

    material = continuous_research.CycleResearchMaterial(
        queries=["legacy query"],
        new_or_changed_documents=[
            continuous_research.SourceDocument(
                query="legacy query",
                title="Legacy Source",
                url="https://example.com/legacy",
                content_excerpt="Legacy changed content.",
                content_hash="hash-legacy",
            )
        ],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No changes.",
        updated_summary="Legacy summary",
        updated_plan="- [ ] Legacy next step",
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    prompt = prompt_log[0]
    assert "Instructions: Legacy instructions that should still appear in the legacy prompt." in prompt
    assert "Explicit User Intent (authoritative):" not in prompt


def test_load_workspace_artifacts_uses_agent_visible_files_and_excludes_audit_context(
    workspace_env: WorkspaceTestEnv,
) -> None:
    event = build_event(event_id="evt-load")
    task = build_task(
        event=event,
        task_id="cr_load",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan="- [ ] Default workspace plan",
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)
    (task_dir / "rogue.md").write_text("rogue-content", encoding="utf-8")
    large_tail = "x" * (continuous_research.DECISION_LOG_TAIL_CHAR_LIMIT + 250)
    (task_dir / continuous_research.DECISION_LOG_FILE_NAME).write_text(
        continuous_research._render_decision_log_header() + large_tail,
        encoding="utf-8",
    )
    (task_dir / continuous_research.QUESTIONS_FILE_NAME).write_text(
        "# Continuous Research Questions\n\n## Active Questions\n- What changed?\n",
        encoding="utf-8",
    )

    artifacts = continuous_research._load_workspace_artifacts(task, task_dir)

    assert artifacts.recent_decision_log == ""
    assert artifacts.plan == "- [ ] Default workspace plan"
    assert "## Active Questions" in artifacts.questions
    assert artifacts.source_policy == ""
    assert "No prior query history recorded" in artifacts.query_history
    assert "## Active Claims" in artifacts.claim_register_markdown
    assert artifacts.assessment == ""


@pytest.mark.asyncio
async def test_no_update_cycle_appends_decision_log_entry(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-no-update")
    task = build_task(
        event=event,
        task_id="cr_no_update",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)

    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        selected_candidates=[
            continuous_research.SearchCandidate(
                query="embassy tech village apartments latest updates",
                title="Example Listing",
                url="https://example.com/listing",
            )
        ],
        new_or_changed_documents=[
            continuous_research.SourceDocument(
                query="embassy tech village apartments latest updates",
                title="Example Listing",
                url="https://example.com/listing",
                content_excerpt="Changed copy without material information.",
                content_hash="hash-no-update",
            )
        ],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No materially new findings.",
        updated_summary="Still waiting on new signals.",
        updated_detailed_report=(
            "# Continuous Research Detailed Report\n\n"
            "## Current Bottom Line\n"
            "- No material updates yet, but official developer sources remain the next best check.\n"
        ),
        updated_plan="- [ ] Check official developer sites next cycle",
        updated_questions=(
            "# Continuous Research Questions\n\n"
            "## Active Questions\n"
            "- What do official developer sites say about current availability?\n\n"
            "## Recently Resolved\n"
            "- None yet.\n\n"
            "## Deferred Questions\n"
            "- None yet.\n"
        ),
        updated_assessment=(
            "# Continuous Research Assessment\n\n"
            "## What Changed\n"
            "- No new or changed source documents were detected this cycle.\n\n"
            "## What Is Genuinely New\n"
            "- The cycle did not confirm materially new information.\n\n"
            "## What Matters To The User\n"
            "- No material change for the user.\n\n"
            "## What Remains Unresolved\n"
            "- Check official developer sites next cycle.\n\n"
            "## Notification Decision\n"
            "- Notify: no\n"
            "- Reason: No material change.\n"
        ),
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    log_text = (task_dir / continuous_research.DECISION_LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "- Status: no_change" in log_text
    assert "- Queries run: embassy tech village apartments latest updates" in log_text
    assert "- Selected source count: 1" in log_text
    assert "- Changed source count: 1" in log_text
    assert "- Next step reason: Check official developer sites next cycle" in log_text
    assert "- User-facing update sent: no" in log_text
    assert "- Notification reason: Model did not request notification." in log_text
    activity_log_text = (task_dir / continuous_research.ACTIVITY_LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "cycle_started" in activity_log_text
    assert "analysis_prompt_prepared" in activity_log_text
    assert "analysis_completed" in activity_log_text
    assert "cycle_completed" in activity_log_text
    questions_text = (task_dir / continuous_research.QUESTIONS_FILE_NAME).read_text(encoding="utf-8")
    assert "What do official developer sites say about current availability?" in questions_text
    assessment_text = (task_dir / continuous_research.ASSESSMENT_FILE_NAME).read_text(encoding="utf-8")
    assert "## Notification Decision" in assessment_text
    detailed_report_text = (task_dir / continuous_research.DETAILED_REPORT_FILE_NAME).read_text(encoding="utf-8")
    assert "official developer sources remain the next best check" in detailed_report_text
    cycle_state = continuous_research.CycleRuntimeState.model_validate_json(
        (task_dir / continuous_research.CYCLE_STATE_FILE_NAME).read_text(encoding="utf-8")
    )
    assert cycle_state.status == "completed"
    assert cycle_state.stage == "completed"
    analysis_result = (task_dir / continuous_research.ANALYSIS_RESULT_FILE_NAME).read_text(encoding="utf-8")
    assert "Still waiting on new signals." in analysis_result


@pytest.mark.asyncio
async def test_update_cycle_appends_decision_log_and_keeps_update_file_behavior(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-update")
    task = build_task(
        event=event,
        task_id="cr_update",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)
    changed_document = continuous_research.SourceDocument(
        query="embassy tech village apartments latest updates",
        title="Example Listing",
        url="https://rera.karnataka.gov.in/listing",
        source="rera.karnataka.gov.in",
        content_excerpt="Fresh pricing update",
        content_hash="hash-1",
    )
    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        selected_candidates=[
            continuous_research.SearchCandidate(
                query="embassy tech village apartments latest updates",
                title="Example Listing",
                url="https://rera.karnataka.gov.in/listing",
                source="rera.karnataka.gov.in",
            )
        ],
        new_or_changed_documents=[changed_document],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=True,
        new_findings="A new listing price was published.",
        updated_summary="Observed a new pricing signal.",
        updated_plan="- [ ] Re-check the builder website for confirmation",
        updated_questions=(
            "# Continuous Research Questions\n\n"
            "## Active Questions\n"
            "- Does the builder website confirm the new pricing signal?\n\n"
            "## Recently Resolved\n"
            "- Was there a new pricing signal? Yes, a new listing price was published.\n\n"
            "## Deferred Questions\n"
            "- None yet.\n"
        ),
        updated_assessment=(
            "# Continuous Research Assessment\n\n"
            "## What Changed\n"
            "- A regulatory listing published a new pricing signal.\n\n"
            "## What Is Genuinely New\n"
            "- A new pricing signal was published.\n\n"
            "## What Matters To The User\n"
            "- This affects the user's search for good-value apartments near Embassy Tech Village.\n\n"
            "## What Remains Unresolved\n"
            "- Does the builder website confirm the new pricing signal?\n\n"
            "## Notification Decision\n"
            "- Notify: yes\n"
            "- Reason: A regulatory source published a material price update.\n"
        ),
        should_notify=True,
        notification_reason="A regulatory source published a material price update.",
        supporting_urls=["https://rera.karnataka.gov.in/listing"],
        document_confidences=[
            continuous_research.DocumentConfidence(
                url="https://rera.karnataka.gov.in/listing",
                evidence_type="regulatory",
                confidence=0.9,
                confidence_reason="The fetched document directly supports the pricing signal.",
            )
        ],
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "notified_update"
    update_files = sorted((task_dir / "updates").glob("*.md"))
    assert len(update_files) == 1
    assert "A new listing price was published." in update_files[0].read_text(encoding="utf-8")
    assert sent_updates

    log_text = (task_dir / continuous_research.DECISION_LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "- Status: notified_update" in log_text
    assert "- Changed source count: 1" in log_text
    assert "- User-facing update sent: yes" in log_text
    assert "- Notification reason: A regulatory source published a material price update." in log_text
    questions_text = (task_dir / continuous_research.QUESTIONS_FILE_NAME).read_text(encoding="utf-8")
    assert "Does the builder website confirm the new pricing signal?" in questions_text
    claim_register_json = (
        task_dir / continuous_research.EVIDENCE_DIR_NAME / continuous_research.CLAIM_REGISTER_JSON_FILE_NAME
    ).read_text(encoding="utf-8")
    assert "Example Listing" in claim_register_json
    cycle_state = continuous_research.CycleRuntimeState.model_validate_json(
        (task_dir / continuous_research.CYCLE_STATE_FILE_NAME).read_text(encoding="utf-8")
    )
    assert cycle_state.status == "completed"
    assert cycle_state.result == "notified_update"


@pytest.mark.asyncio
async def test_workspace_cycle_regenerates_missing_brief_and_manifest_and_keeps_explicit_intent_stable(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(
        event_id="evt-recovery",
        message="Keep looking for good-value apartments near Embassy Tech Village.",
    )
    task = build_task(
        event=event,
        task_id="cr_recovery",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        instructions="Track east-facing luxury 3BHK units with no servant room.",
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)
    (task_dir / continuous_research.BRIEF_FILE_NAME).unlink()
    (task_dir / continuous_research.WORKSPACE_MANIFEST_FILE_NAME).unlink()

    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No update.",
        updated_summary="Still monitoring.",
        updated_plan="- [ ] Check infrastructure updates next cycle",
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    assert (task_dir / continuous_research.BRIEF_FILE_NAME).exists()
    assert (task_dir / continuous_research.WORKSPACE_MANIFEST_FILE_NAME).exists()

    brief_sections = continuous_research._parse_markdown_sections(
        (task_dir / continuous_research.BRIEF_FILE_NAME).read_text(encoding="utf-8")
    )
    assert event.message in brief_sections["Explicit User Intent"]
    assert "east-facing" not in brief_sections["Explicit User Intent"].lower()
    assert "east-facing" in brief_sections["Assistant Working Interpretation"].lower()


@pytest.mark.asyncio
async def test_workspace_cycle_recovers_missing_questions_file(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-questions-recovery")
    task = build_task(
        event=event,
        task_id="cr_questions_recovery",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)
    (task_dir / continuous_research.QUESTIONS_FILE_NAME).unlink()

    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No update.",
        updated_summary="Still monitoring.",
        updated_plan="- [ ] Check infrastructure updates next cycle",
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    regenerated_questions = (task_dir / continuous_research.QUESTIONS_FILE_NAME).read_text(encoding="utf-8")
    assert "## Active Questions" in regenerated_questions


def test_prioritize_candidates_does_not_apply_source_trust_policy() -> None:
    ledger = continuous_research.EvidenceLedger()
    questions_text = (
        "# Continuous Research Questions\n\n"
        "## Active Questions\n"
        "- What does the builder site say about pricing and possession?\n"
    )
    candidates = [
        continuous_research.SearchCandidate(
            query="pricing possession",
            title="Forum rumor on pricing",
            url="https://reddit.com/r/example/post",
            snippet="pricing possession rumor",
            source="reddit.com",
        ),
        continuous_research.SearchCandidate(
            query="pricing possession",
            title="Builder pricing update",
            url="https://builder.example.com/pricing",
            snippet="official pricing and possession",
            source="builder.example.com",
        ),
        continuous_research.SearchCandidate(
            query="pricing possession",
            title="Secondary listing",
            url="https://housing.com/listing",
            snippet="pricing listing",
            source="housing.com",
        ),
    ]

    prioritized, candidate_tiers, tier_counts = continuous_research._prioritize_candidates(
        candidates,
        ledger,
        questions_text=questions_text,
    )

    assert {candidate.url for candidate in prioritized} == {candidate.url for candidate in candidates}
    assert candidate_tiers == {}
    assert tier_counts == {}


@pytest.mark.asyncio
async def test_low_trust_only_change_updates_artifacts_but_suppresses_notification(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-low-trust")
    task = build_task(
        event=event,
        task_id="cr_low_trust",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)

    changed_document = continuous_research.SourceDocument(
        query="embassy tech village apartments latest updates",
        title="Forum rumor",
        url="https://reddit.com/r/example/post",
        source="reddit.com",
        content_excerpt="Rumor about a dramatic price drop",
        content_hash="hash-low-trust",
    )
    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        selected_candidates=[
            continuous_research.SearchCandidate(
                query="embassy tech village apartments latest updates",
                title="Forum rumor",
                url="https://reddit.com/r/example/post",
                source="reddit.com",
            )
        ],
        new_or_changed_documents=[changed_document],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=True,
        new_findings="A forum post claimed a dramatic price drop.",
        updated_summary="A forum post claimed a dramatic price drop.",
        updated_detailed_report=(
            "# Continuous Research Detailed Report\n\n"
            "## Weak Or Unverified Signals\n"
            "- A forum post claimed a dramatic price drop, but confidence is low and it needs official verification.\n"
        ),
        updated_plan="- [ ] Verify the claim using official sources",
        updated_assessment=(
            "# Continuous Research Assessment\n\n"
            "## What Changed\n"
            "- A low-trust forum post claimed a dramatic price drop.\n\n"
            "## What Is Genuinely New\n"
            "- The claim is new but unverified.\n\n"
            "## What Matters To The User\n"
            "- This might matter if verified, but it is not yet reliable.\n\n"
            "## What Remains Unresolved\n"
            "- Verify the claim using official sources.\n\n"
            "## Notification Decision\n"
            "- Notify: yes\n"
            "- Reason: The model requested notification.\n"
        ),
        should_notify=True,
        notification_reason="The model requested notification.",
        document_confidences=[
            continuous_research.DocumentConfidence(
                url="https://reddit.com/r/example/post",
                evidence_type="forum_social",
                confidence=0.35,
                confidence_reason="The source is a single unverified user post.",
            )
        ],
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "unverified_signal"
    assert not sent_updates
    assert not list((task_dir / "updates").glob("*.md"))
    cycle_reports = list((task_dir / continuous_research.CYCLE_REPORTS_DIR_NAME).glob("*.md"))
    assert len(cycle_reports) == 1
    cycle_report_text = cycle_reports[0].read_text(encoding="utf-8")
    assert "A forum post claimed a dramatic price drop." in cycle_report_text
    assert "Rumor about a dramatic price drop" in cycle_report_text
    assert "Notification suppressed because evidence confidence 0.35 is below 0.80" in cycle_report_text
    summary_text = (task_dir / continuous_research.SUMMARY_FILE_NAME).read_text(encoding="utf-8")
    assert summary_text == continuous_research._default_summary()
    detailed_report_text = (task_dir / continuous_research.DETAILED_REPORT_FILE_NAME).read_text(encoding="utf-8")
    assert "Weak Or Unverified Signals" in detailed_report_text
    assert "dramatic price drop" in detailed_report_text
    log_text = (task_dir / continuous_research.DECISION_LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "below 0.80" in log_text
    saved_task = workspace_env.state.get_task(task.task_id)
    assert saved_task.no_new_findings_count == 0
    assert saved_task.unverified_signal_count == 1


@pytest.mark.asyncio
async def test_periodic_review_preserves_explicit_intent_and_updates_brief_without_source_policy(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(
        event_id="evt-review",
        message="Track good-value apartments near Embassy Tech Village.",
    )
    task = build_task(
        event=event,
        task_id="cr_review",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task.cycle_count = continuous_research.REVIEW_CYCLE_INTERVAL - 1
    task_dir = install_workspace_task(workspace_env, task, event=event)

    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments review"],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No update.",
        updated_summary="Still monitoring.",
        updated_plan="- [ ] Focus on builder and regulatory sources next cycle",
        updated_brief=(
            "# Continuous Research Brief\n\n"
            "## User Request\n"
            "Track good-value apartments near Embassy Tech Village.\n\n"
            "## Explicit User Intent\n"
            "Track penthouses only.\n\n"
            "## Assistant Working Interpretation\n"
            "- Focus on value, appreciation, and builder credibility.\n\n"
            "## Known Constraints\n"
            "- No confirmed additional constraints.\n\n"
            "## Inferred Assumptions\n"
            "- None.\n\n"
            "## Novelty Guidance\n"
            "- Material changes only.\n\n"
            "## Cadence And Stop Conditions\n"
            "- Default cadence.\n"
        ),
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    assert "Cycle Mode: periodic_review" in prompt_log[0]
    assert "Source Policy:" not in prompt_log[0]
    assert "updated_source_policy" not in prompt_log[0]
    brief_sections = continuous_research._parse_markdown_sections(
        (task_dir / continuous_research.BRIEF_FILE_NAME).read_text(encoding="utf-8")
    )
    assert brief_sections["Explicit User Intent"] == event.message
    assert not (task_dir / continuous_research.SOURCE_POLICY_FILE_NAME).exists()


@pytest.mark.asyncio
async def test_collect_cycle_material_persists_candidate_and_fetch_stage_artifacts(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-stage-artifacts")
    task = build_task(
        event=event,
        task_id="cr_stage_artifacts",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)

    async def fake_search_candidates_for_queries(*args, **kwargs):
        return (
            [
                continuous_research.SearchCandidate(
                    query="builder pricing",
                    title="Builder pricing update",
                    url="https://builder.example.com/pricing",
                    source="builder.example.com",
                    snippet="pricing and possession",
                ),
                continuous_research.SearchCandidate(
                    query="builder pricing",
                    title="Forum rumor",
                    url="https://reddit.com/r/example/post",
                    source="reddit.com",
                    snippet="pricing rumor",
                ),
            ],
            [],
            [
                continuous_research.SearchObservation(
                    query="builder pricing",
                    answer="The builder published pricing evidence.",
                    candidate_count=2,
                )
            ],
        )

    async def fake_fetch_source_documents(candidates, *, trace=None):
        return (
            [
                continuous_research.SourceDocument(
                    query=candidates[0].query,
                    title=candidates[0].title,
                    url=candidates[0].url,
                    source=candidates[0].source,
                    snippet=candidates[0].snippet,
                    content_excerpt="Builder posted a pricing update.",
                    content_hash="hash-stage-artifact",
                )
            ],
            [],
        )

    fake_plan = continuous_research.SearchPlan(
        queries=[
            continuous_research.SearchQueryDecision(
                query="builder pricing",
                reason="Check builder evidence.",
                expected_signal="Pricing evidence.",
                query_role="refinement",
            )
        ]
    )
    fake_planner = SimpleNamespace(run=lambda prompt: asyncio.sleep(0, result=SimpleNamespace(output=fake_plan)))

    monkeypatch.setattr(continuous_research, "_search_candidates_for_queries", fake_search_candidates_for_queries)
    monkeypatch.setattr(continuous_research, "_fetch_source_documents", fake_fetch_source_documents)
    monkeypatch.setattr(continuous_research, "get_query_planner_agent", lambda: fake_planner)

    material = await continuous_research._collect_cycle_material(
        task,
        task_dir,
        cycle_id="cycle_test_stage",
        questions_text=(
            "# Continuous Research Questions\n\n"
            "## Active Questions\n"
            "- What does the builder say about pricing?\n"
        ),
    )

    candidate_pool = (task_dir / continuous_research.CANDIDATE_POOL_FILE_NAME).read_text(encoding="utf-8")
    fetched_sources = (task_dir / continuous_research.FETCHED_SOURCES_FILE_NAME).read_text(encoding="utf-8")

    assert material.selected_candidates[0].url == "https://builder.example.com/pricing"
    assert "builder.example.com/pricing" in candidate_pool
    assert "reddit.com/r/example/post" in candidate_pool
    assert "Check builder evidence." in candidate_pool
    assert "The builder published pricing evidence." in candidate_pool
    assert material.search_observations[0].selected_count == 2
    assert material.search_observations[0].fetched_count == 1
    assert "Builder pricing update" in fetched_sources
    assert "hash-stage-artifact" in fetched_sources


@pytest.mark.asyncio
async def test_cycle_recovers_orphaned_running_state_before_new_run(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-orphaned")
    task = build_task(
        event=event,
        task_id="cr_orphaned",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task.is_cycle_running = True
    task.last_cycle_started_at = datetime.now(timezone.utc) - timedelta(minutes=30)
    task_dir = install_workspace_task(workspace_env, task, event=event)

    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartments latest updates"],
        ledger=continuous_research.EvidenceLedger(),
    )
    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No update.",
        updated_summary="Still monitoring.",
        updated_plan="- [ ] Keep monitoring official sources",
    )
    prompt_log: list[str] = []
    sent_updates: list[str] = []
    patch_cycle_dependencies(
        monkeypatch,
        material=material,
        output=output,
        prompt_log=prompt_log,
        sent_updates=sent_updates,
    )

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "no_change"
    activity_log = (task_dir / continuous_research.ACTIVITY_LOG_FILE_NAME).read_text(encoding="utf-8")
    assert "stale_cycle_recovered" in activity_log
    cycle_state = continuous_research.CycleRuntimeState.model_validate_json(
        (task_dir / continuous_research.CYCLE_STATE_FILE_NAME).read_text(encoding="utf-8")
    )
    assert cycle_state.status == "completed"
    assert cycle_state.result == "no_change"


@pytest.mark.asyncio
async def test_search_timeout_marks_cycle_failed_and_persists_failed_cycle_state(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-timeout")
    task = build_task(
        event=event,
        task_id="cr_timeout",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)

    async def slow_search_candidates_for_queries(*args, **kwargs):
        await asyncio.sleep(0.05)
        return [], []

    async def fake_send_proactive_update(event: TelegramMessageEvent, message: str) -> None:
        return None

    monkeypatch.setattr(continuous_research, "_search_candidates_for_queries", slow_search_candidates_for_queries)
    monkeypatch.setattr(continuous_research, "send_proactive_update", fake_send_proactive_update)
    monkeypatch.setattr(continuous_research, "SEARCH_STAGE_TIMEOUT", timedelta(milliseconds=1))

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "failed"
    cycle_state = continuous_research.CycleRuntimeState.model_validate_json(
        (task_dir / continuous_research.CYCLE_STATE_FILE_NAME).read_text(encoding="utf-8")
    )
    assert cycle_state.status == "failed"
    assert cycle_state.stage == "failed"
    assert "timed out" in cycle_state.error
    analysis_result = (task_dir / continuous_research.ANALYSIS_RESULT_FILE_NAME).read_text(encoding="utf-8")
    assert '"status": "failed"' in analysis_result


@pytest.mark.asyncio
async def test_analysis_failure_writes_degraded_cycle_report_with_fetched_content(
    workspace_env: WorkspaceTestEnv,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event = build_event(event_id="evt-analysis-failure")
    task = build_task(
        event=event,
        task_id="cr_analysis_failure",
        workspace_version=continuous_research.FOUNDATION_WORKSPACE_VERSION,
        plan=continuous_research._default_plan("Embassy Tech Village apartments"),
    )
    task_dir = install_workspace_task(workspace_env, task, event=event)

    changed_document = continuous_research.SourceDocument(
        query="embassy tech village apartment pricing",
        title="Builder pricing update",
        url="https://builder.example.com/pricing",
        source="builder.example.com",
        content_excerpt="3BHK inventory now starts at 2.45 Cr with larger carpet area options.",
        content_hash="hash-analysis-failure",
    )
    material = continuous_research.CycleResearchMaterial(
        queries=["embassy tech village apartment pricing"],
        candidates=[
            continuous_research.SearchCandidate(
                query="embassy tech village apartment pricing",
                title="Builder pricing update",
                url="https://builder.example.com/pricing",
                source="builder.example.com",
            )
        ],
        selected_candidates=[
            continuous_research.SearchCandidate(
                query="embassy tech village apartment pricing",
                title="Builder pricing update",
                url="https://builder.example.com/pricing",
                source="builder.example.com",
            )
        ],
        fetched_documents=[changed_document],
        new_or_changed_documents=[changed_document],
        ledger=continuous_research.EvidenceLedger(),
    )

    async def fake_collect_cycle_material(*args, **kwargs):
        return material

    class FailingCycleAgent:
        async def run(self, prompt: str) -> SimpleNamespace:
            raise RuntimeError("status_code: 503, model unavailable")

    async def fake_send_proactive_update(event: TelegramMessageEvent, message: str) -> None:
        return None

    monkeypatch.setattr(continuous_research, "_collect_cycle_material", fake_collect_cycle_material)
    monkeypatch.setattr(continuous_research, "get_cycle_agent", lambda: FailingCycleAgent())
    monkeypatch.setattr(continuous_research, "send_proactive_update", fake_send_proactive_update)
    monkeypatch.setattr(continuous_research, "ANALYSIS_TRANSIENT_RETRY_DELAYS_SECONDS", ())

    result = await continuous_research._run_continuous_cycle(task)

    assert result == "failed"
    cycle_reports = list((task_dir / continuous_research.CYCLE_REPORTS_DIR_NAME).glob("*.md"))
    assert len(cycle_reports) == 1
    cycle_report_text = cycle_reports[0].read_text(encoding="utf-8")
    assert "Structured analysis was unavailable" in cycle_report_text
    assert "3BHK inventory now starts at 2.45 Cr" in cycle_report_text
    assert "status_code: 503" in cycle_report_text
