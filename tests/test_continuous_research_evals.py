from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import agent.continuous_research as continuous_research


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "continuous_research"


def _load_case(case_name: str) -> tuple[Path, dict]:
    case_dir = FIXTURES_DIR / case_name
    scenario = json.loads((case_dir / "scenario.json").read_text(encoding="utf-8"))
    return case_dir, scenario


def _build_material(scenario: dict) -> continuous_research.CycleResearchMaterial:
    selected_candidates: list[continuous_research.SearchCandidate] = []
    changed_documents: list[continuous_research.SourceDocument] = []

    for position, document in enumerate(scenario.get("documents", []), start=1):
        selected_candidates.append(
            continuous_research.SearchCandidate(
                query=document["query"],
                title=document["title"],
                url=document["url"],
                snippet=document.get("snippet", ""),
                source=document.get("source"),
                position=position,
            )
        )
        changed_documents.append(
            continuous_research.SourceDocument(
                query=document["query"],
                title=document["title"],
                url=document["url"],
                snippet=document.get("snippet", ""),
                source=document.get("source"),
                content_excerpt=document.get("content_excerpt", ""),
                content_hash=document.get("content_hash", f"hash-{position}"),
            )
        )
    return continuous_research.CycleResearchMaterial(
        queries=scenario.get("queries", []),
        selected_candidates=selected_candidates,
        new_or_changed_documents=changed_documents,
        ledger=continuous_research.EvidenceLedger(),
    )


def _document_confidences(scenario: dict) -> list[continuous_research.DocumentConfidence]:
    tier_confidence = {
        "official": 0.95,
        "regulatory": 0.9,
        "primary_reporting": 0.85,
        "secondary_reporting": 0.65,
        "low_trust": 0.35,
    }
    return [
        continuous_research.DocumentConfidence(
            url=document["url"],
            evidence_type=document.get("tier", "unknown"),
            confidence=tier_confidence.get(document.get("tier", "unknown"), 0.5),
            confidence_reason="Fixture confidence derived from scenario evidence type.",
        )
        for document in scenario.get("documents", [])
    ]


def test_eval_explicit_intent_drift_keeps_explicit_intent_stable() -> None:
    case_dir, scenario = _load_case("explicit_intent_drift")
    existing_brief = (case_dir / "brief.md").read_text(encoding="utf-8")
    proposed_brief = (case_dir / "proposed_brief.md").read_text(encoding="utf-8")

    merged_brief = continuous_research._preserve_explicit_intent_in_brief(existing_brief, proposed_brief)
    sections = continuous_research._parse_markdown_sections(merged_brief)

    assert sections["Explicit User Intent"] == scenario["expected_explicit_intent"]
    assert "builder credibility" in sections["Assistant Working Interpretation"].lower()


def test_eval_low_trust_false_positive_does_not_enter_canonical_summary_or_notify() -> None:
    case_dir, scenario = _load_case("low_trust_false_positive")
    brief_sections = continuous_research._parse_markdown_sections(
        (case_dir / "brief.md").read_text(encoding="utf-8")
    )
    prior_summary = (case_dir / "summary.md").read_text(encoding="utf-8").strip()
    assessment_text = (case_dir / "assessment.md").read_text(encoding="utf-8")
    material = _build_material(scenario)

    output = continuous_research.CycleResult(
        found_new_info=True,
        new_findings="A forum post claimed a sharp discount.",
        updated_summary=scenario["proposed_summary"],
        updated_plan="- [ ] Verify the claim using official sources",
        updated_assessment=assessment_text,
        should_notify=True,
        notification_reason="The model requested notification.",
        document_confidences=_document_confidences(scenario),
    )

    decision = continuous_research._evaluate_notification_gate(
        output=output,
        material=material,
        explicit_user_intent=brief_sections["Explicit User Intent"],
    )
    resulting_summary = (
        prior_summary
        if continuous_research._should_freeze_summary_for_confidence_gap(output, material) and not decision.should_notify
        else output.updated_summary
    )

    assert not decision.should_notify
    assert "below 0.80" in decision.reason
    assert resulting_summary == prior_summary


def test_eval_contradiction_and_supersession_are_reflected_in_claim_register() -> None:
    case_dir, scenario = _load_case("contradiction_supersession")
    existing_register = continuous_research.ClaimRegister.model_validate_json(
        (case_dir / "evidence" / "claim_register.json").read_text(encoding="utf-8")
    )
    incoming_claims = [
        continuous_research.ClaimRecord(
            id=claim["id"],
            statement=claim["statement"],
            status=claim["status"],
            confidence=claim["confidence"],
            supporting_urls=claim["supporting_urls"],
            source_tiers=claim["source_tiers"],
            contradicts=claim["contradicts"],
            updated_at=datetime.fromisoformat(claim["updated_at"]),
            notes=claim["notes"],
        )
        for claim in scenario["incoming_claims"]
    ]

    merged_register, counters = continuous_research._merge_claim_records(existing_register, incoming_claims)
    merged_statuses = {claim.id: claim.status for claim in merged_register.claims}

    assert merged_statuses["claim_completion_q4"] == "contradicted"
    assert merged_statuses["claim_completion_q1"] == "active"
    assert counters["contradicted"] >= 1


def test_eval_corroborated_high_trust_change_triggers_notification() -> None:
    case_dir, scenario = _load_case("corroborated_high_trust_update")
    brief_sections = continuous_research._parse_markdown_sections(
        (case_dir / "brief.md").read_text(encoding="utf-8")
    )
    assessment_text = (case_dir / "assessment.md").read_text(encoding="utf-8")
    material = _build_material(scenario)

    output = continuous_research.CycleResult(
        found_new_info=True,
        new_findings="Two independent high-trust sources confirmed a price update.",
        updated_summary="Two high-trust sources confirmed the price update.",
        updated_plan="- [ ] Re-check the builder release next cycle",
        updated_assessment=assessment_text,
        should_notify=True,
        notification_reason="Two independent high-trust sources confirmed a material change.",
        document_confidences=_document_confidences(scenario),
    )

    decision = continuous_research._evaluate_notification_gate(
        output=output,
        material=material,
        explicit_user_intent=brief_sections["Explicit User Intent"],
    )

    assert decision.should_notify


def test_eval_no_update_steady_state_does_not_trigger_notification() -> None:
    case_dir, scenario = _load_case("no_update_steady_state")
    brief_sections = continuous_research._parse_markdown_sections(
        (case_dir / "brief.md").read_text(encoding="utf-8")
    )
    assessment_text = (case_dir / "assessment.md").read_text(encoding="utf-8")
    material = _build_material(scenario)

    output = continuous_research.CycleResult(
        found_new_info=False,
        new_findings="No meaningful change.",
        updated_summary="Still monitoring with no material change.",
        updated_plan="- [ ] Check primary sources again next cycle",
        updated_assessment=assessment_text,
        should_notify=False,
        notification_reason="No material change.",
    )

    decision = continuous_research._evaluate_notification_gate(
        output=output,
        material=material,
        explicit_user_intent=brief_sections["Explicit User Intent"],
    )

    assert not decision.should_notify
    assert decision.reason == "No material change."
