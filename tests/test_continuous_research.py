import asyncio
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import agent.continuous_research as continuous_research
import agent.continuous_state as continuous_state_module
from agent.research_steps import WebSearchCandidate, WebSearchResult
from agent.continuous_state import ContinuousResearchState, ContinuousTask
from models import TelegramMessageEvent


class ContinuousResearchHelperTests(unittest.IsolatedAsyncioTestCase):
    def test_continuous_research_model_routing_uses_pro_for_analysis(self) -> None:
        self.assertEqual(
            continuous_research._continuous_query_planner_model(),
            continuous_research.gemini_model.get_balanced_model(),
        )
        self.assertEqual(
            continuous_research._continuous_analysis_model(),
            continuous_research.gemini_model.get_large_model(),
        )

    def test_search_stage_timeout_scales_with_gemini_retry_window(self) -> None:
        with patch.object(
            continuous_research,
            "_gemini_search_503_min_retry_seconds",
            return_value=300.0,
        ):
            one_batch_timeout = continuous_research._search_stage_timeout(4)
            three_batch_timeout = continuous_research._search_stage_timeout(12)

        self.assertEqual(one_batch_timeout, timedelta(minutes=6))
        self.assertEqual(three_batch_timeout, timedelta(minutes=16))

    def test_search_stage_timeout_honors_explicit_override(self) -> None:
        with patch.object(
            continuous_research,
            "SEARCH_STAGE_TIMEOUT",
            timedelta(seconds=1),
        ):
            timeout = continuous_research._search_stage_timeout(12)

        self.assertEqual(timeout, timedelta(seconds=1))

    def test_extract_source_candidates_parses_and_dedupes_urls(self) -> None:
        raw_result = """
Gemini Grounded Answer: latest summary here
Sources:
- Example Story | https://example.com/story?utm_source=newsletter
2. Another Story | https://another.example/path | Fresh snippet
- Duplicate Story | https://example.com/story?utm_medium=email
"""

        candidates = continuous_research._extract_source_candidates_from_text(
            raw_result,
            "mars mission",
        )

        self.assertEqual(len(candidates), 2)
        self.assertEqual(candidates[0].title, "Example Story")
        self.assertEqual(candidates[1].snippet, "Fresh snippet")
        self.assertEqual(
            continuous_research._normalize_url(candidates[0].url),
            "https://example.com/story",
        )

    def test_merge_documents_into_ledger_only_flags_new_or_changed_content(self) -> None:
        ledger = continuous_research.EvidenceLedger()
        document = continuous_research.SourceDocument(
            query="mars mission",
            title="Example Story",
            url="https://example.com/story",
            snippet="snippet",
            source="example.com",
            content_excerpt="first excerpt",
            content_hash="hash-1",
        )

        first_pass = continuous_research._merge_documents_into_ledger(ledger, [document])
        second_pass = continuous_research._merge_documents_into_ledger(ledger, [document])
        changed_document = document.model_copy(
            update={"content_hash": "hash-2", "content_excerpt": "updated excerpt"}
        )
        third_pass = continuous_research._merge_documents_into_ledger(
            ledger,
            [changed_document],
        )

        self.assertEqual(len(first_pass), 1)
        self.assertEqual(len(second_pass), 0)
        self.assertEqual(len(third_pass), 1)
        record = ledger.records[continuous_research._normalize_url(document.url)]
        self.assertEqual(record.content_hash, "hash-2")
        self.assertEqual(record.latest_excerpt, "updated excerpt")

    def test_build_search_queries_is_minimal_mechanical_fallback(self) -> None:
        task = ContinuousTask(
            task_id="cr_fallback_queries",
            topic="flats near Embassy Tech Village and my budget is 2-4 cr",
            instructions="Keep searching for good value options with space and amenities.",
            status="running",
            event_dict={},
        )

        queries = continuous_research._build_search_queries(
            task,
            max_queries=2,
        )

        self.assertEqual(queries[0], task.topic)
        self.assertEqual(queries[1], f"{task.topic} latest update")
        joined = " | ".join(queries).lower()
        self.assertNotIn("-rent", joined)
        self.assertNotIn("bwssb", joined)
        self.assertNotIn("flood map", joined)

    async def test_plan_search_queries_uses_agent_output_and_query_history(self) -> None:
        task = ContinuousTask(
            task_id="cr_agent_queries",
            topic="Mars mission",
            instructions="Track launch updates.",
            status="running",
            event_dict={},
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            task_dir = Path(temp_dir)
            continuous_research._append_query_history(
                task_dir,
                [
                    continuous_research.QueryHistoryEntry(
                        cycle_id="cycle_old",
                        query="Mars mission latest update",
                        candidate_count=5,
                        fetched_count=2,
                        changed_count=0,
                        top_domains=["nasa.gov"],
                        outcome="no_change",
                    )
                ],
            )
            fake_plan = continuous_research.SearchPlan(
                queries=[
                    continuous_research.SearchQueryDecision(
                        query="Mars mission launch delay official statement",
                        reason="Prior latest-update query was low-yield.",
                        expected_signal="Primary confirmation or contradiction.",
                        query_role="refinement",
                    )
                ]
            )
            fake_agent = MagicMock()
            fake_agent.run = AsyncMock(return_value=SimpleNamespace(output=fake_plan))

            with patch.object(continuous_research, "get_query_planner_agent", return_value=fake_agent):
                queries, decisions, history = await continuous_research._plan_search_queries(
                    task,
                    task_dir=task_dir,
                    ledger=continuous_research.EvidenceLedger(),
                    max_queries=2,
                )

        self.assertEqual(queries, ["Mars mission launch delay official statement"])
        self.assertEqual(decisions[0].query_role, "refinement")
        self.assertEqual(history[0].query, "Mars mission latest update")
        prompt = fake_agent.run.await_args.args[0]
        self.assertIn("Mars mission latest update", prompt)
        self.assertIn("Do not assume a fixed domain profile", prompt)

    def test_select_candidates_for_fetch_prefers_sources_outside_recheck_cooldown(self) -> None:
        now = continuous_research._utc_now()
        ledger = continuous_research.EvidenceLedger(
            records={
                continuous_research._normalize_url("https://example.com/recent"): continuous_research.EvidenceRecord(
                    url="https://example.com/recent",
                    title="Recent Source",
                    first_seen_at=now,
                    last_seen_at=now,
                    last_checked_at=now,
                    content_hash="recent-hash",
                ),
                continuous_research._normalize_url("https://example.com/stale"): continuous_research.EvidenceRecord(
                    url="https://example.com/stale",
                    title="Stale Source",
                    first_seen_at=now - continuous_research.SOURCE_RECHECK_COOLDOWN,
                    last_seen_at=now - continuous_research.SOURCE_RECHECK_COOLDOWN,
                    last_checked_at=now - continuous_research.SOURCE_RECHECK_COOLDOWN - timedelta(minutes=1),
                    content_hash="stale-hash",
                ),
            }
        )
        candidates = [
            continuous_research.SearchCandidate(
                query="mars mission",
                title="Recent Source",
                url="https://example.com/recent",
            ),
            continuous_research.SearchCandidate(
                query="mars mission",
                title="Stale Source",
                url="https://example.com/stale",
            ),
            continuous_research.SearchCandidate(
                query="mars mission",
                title="New Source",
                url="https://example.com/new",
            ),
        ]

        selected = continuous_research._select_candidates_for_fetch(ledger=ledger, prioritized_candidates=candidates, max_candidates=2)

        self.assertEqual(
            [candidate.url for candidate in selected],
            ["https://example.com/stale", "https://example.com/new"],
        )

    def test_select_candidates_for_fetch_skips_recent_sources_inside_cooldown(self) -> None:
        now = continuous_research._utc_now()
        ledger = continuous_research.EvidenceLedger(
            records={
                continuous_research._normalize_url("https://example.com/recent"): continuous_research.EvidenceRecord(
                    url="https://example.com/recent",
                    title="Recent Source",
                    first_seen_at=now,
                    last_seen_at=now,
                    last_checked_at=now,
                    content_hash="recent-hash",
                ),
            }
        )
        candidates = [
            continuous_research.SearchCandidate(
                query="mars mission",
                title="Recent Source",
                url="https://example.com/recent",
            ),
        ]

        selected = continuous_research._select_candidates_for_fetch(ledger=ledger, prioritized_candidates=candidates, max_candidates=1)

        self.assertEqual(selected, [])

    def test_select_candidates_for_fetch_preserves_query_branches_before_filling_remaining_slots(self) -> None:
        ledger = continuous_research.EvidenceLedger()
        candidates = [
            continuous_research.SearchCandidate(
                query="query one",
                title="Query One Result 1",
                url="https://example.com/q1-a",
            ),
            continuous_research.SearchCandidate(
                query="query one",
                title="Query One Result 2",
                url="https://example.com/q1-b",
            ),
            continuous_research.SearchCandidate(
                query="query one",
                title="Query One Result 3",
                url="https://example.com/q1-c",
            ),
            continuous_research.SearchCandidate(
                query="query two",
                title="Query Two Result",
                url="https://example.com/q2",
            ),
            continuous_research.SearchCandidate(
                query="query three",
                title="Query Three Result",
                url="https://example.com/q3",
            ),
            continuous_research.SearchCandidate(
                query="query four",
                title="Query Four Result",
                url="https://example.com/q4",
            ),
        ]

        selected = continuous_research._select_candidates_for_fetch(
            ledger=ledger,
            prioritized_candidates=candidates,
            queries=["query one", "query two", "query three", "query four"],
            max_candidates=5,
        )

        self.assertEqual(
            [candidate.query for candidate in selected],
            ["query one", "query two", "query three", "query four", "query one"],
        )
        self.assertEqual(
            [candidate.url for candidate in selected],
            [
                "https://example.com/q1-a",
                "https://example.com/q2",
                "https://example.com/q3",
                "https://example.com/q4",
                "https://example.com/q1-b",
            ],
        )

    def test_candidate_domain_uses_grounding_title_for_vertex_redirects(self) -> None:
        candidate = continuous_research.SearchCandidate(
            query="example query",
            title="prophunt.ai",
            url="https://vertexaisearch.cloud.google.com/grounding-api-redirect/example",
            source="vertexaisearch.cloud.google.com",
        )

        self.assertEqual(continuous_research._candidate_domain(candidate), "prophunt.ai")

    async def test_search_candidates_prefers_structured_search_results(self) -> None:
        structured_result = WebSearchResult(
            query="mars mission",
            answer="latest summary here",
            candidates=[
                WebSearchCandidate(
                    title="Gemini Story",
                    url="https://example.com/gemini-story",
                    provider="gemini",
                    position=1,
                )
            ],
            providers_tried=["gemini"],
        )

        with patch.object(
            continuous_research,
            "search_web_candidates",
            new=AsyncMock(return_value=structured_result),
        ):
            candidates, errors, observation = await continuous_research._search_candidates("mars mission")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].title, "Gemini Story")
        self.assertEqual(candidates[0].provider, "gemini")
        self.assertEqual(errors, [])
        self.assertEqual(observation.query, "mars mission")
        self.assertEqual(observation.answer, "latest summary here")
        self.assertEqual(observation.candidate_count, 1)

    async def test_search_candidates_enables_fallback_providers(self) -> None:
        structured_result = WebSearchResult(
            query="mars mission",
            candidates=[
                WebSearchCandidate(
                    title="Fallback Story",
                    url="https://fallback.example/story",
                    provider="searxng",
                    position=1,
                )
            ],
            providers_tried=["gemini", "searxng"],
        )

        with patch.object(
            continuous_research,
            "search_web_candidates",
            new=AsyncMock(return_value=structured_result),
        ) as search_web_candidates:
            candidates, errors, observation = await continuous_research._search_candidates("mars mission")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].title, "Fallback Story")
        self.assertEqual(errors, [])
        self.assertEqual(observation.candidate_count, 1)
        search_web_candidates.assert_awaited_once_with(
            "mars mission",
            max_results=continuous_research.MAX_RESULTS_PER_QUERY,
            use_fallback=True,
        )

    async def test_search_candidates_emits_trace_details(self) -> None:
        structured_result = WebSearchResult(
            query="mars mission",
            answer="latest summary here",
            candidates=[
                WebSearchCandidate(
                    title="Gemini Story",
                    url="https://example.com/gemini-story",
                    provider="gemini",
                    position=1,
                )
            ],
            providers_tried=["gemini"],
        )
        trace_events: list[tuple[str, dict[str, object]]] = []

        def trace(event: str, **details: object) -> None:
            trace_events.append((event, details))

        with patch.object(
            continuous_research,
            "search_web_candidates",
            new=AsyncMock(return_value=structured_result),
        ):
            candidates, errors, observation = await continuous_research._search_candidates("mars mission", trace=trace)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(errors, [])
        self.assertEqual(observation.answer, "latest summary here")
        self.assertEqual([event for event, _ in trace_events], ["search_started", "search_completed"])
        self.assertEqual(trace_events[1][1]["providers_tried"], ["gemini"])
        self.assertEqual(trace_events[1][1]["candidate_count"], 1)

    async def test_fetch_source_document_emits_trace_details(self) -> None:
        candidate = continuous_research.SearchCandidate(
            query="mars mission",
            title="Launch Update",
            url="https://example.com/launch-update",
        )
        trace_events: list[tuple[str, dict[str, object]]] = []

        def trace(event: str, **details: object) -> None:
            trace_events.append((event, details))

        with patch.object(
            continuous_research,
            "web_fetch",
            new=AsyncMock(return_value="Example body text"),
        ):
            document, error = await continuous_research._fetch_source_document(candidate, trace=trace)

        self.assertIsNotNone(document)
        self.assertIsNone(error)
        self.assertEqual([event for event, _ in trace_events], ["fetch_started", "fetch_completed"])
        self.assertTrue(trace_events[1][1]["success"])
        self.assertEqual(trace_events[1][1]["content_chars"], len("Example body text"))


class ContinuousResearchCycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        self.output_dir = Path(self.temp_dir.name) / "continuous_output"
        self.state_file = self.output_dir / ".state.json"

        self.state_file_patcher = patch.object(
            continuous_state_module,
            "STATE_FILE",
            self.state_file,
        )
        self.state_file_patcher.start()
        self.addCleanup(self.state_file_patcher.stop)

        self.output_dir_patcher = patch.object(
            continuous_research,
            "OUTPUT_DIR",
            self.output_dir,
        )
        self.output_dir_patcher.start()
        self.addCleanup(self.output_dir_patcher.stop)

        self.state = ContinuousResearchState()
        self.global_state_patcher = patch.object(
            continuous_research,
            "global_continuous_state",
            self.state,
        )
        self.global_state_patcher.start()
        self.addCleanup(self.global_state_patcher.stop)

        continuous_research._TASK_LOCKS.clear()
        self.addCleanup(continuous_research._TASK_LOCKS.clear)

    def _build_task(self, *, task_id: str, status: str = "running") -> ContinuousTask:
        event = TelegramMessageEvent(
            event_id=f"evt-{task_id}",
            source="telegram",
            message="track this topic",
            channel_id=100,
            sender_id=200,
            message_id=300,
        )
        task = ContinuousTask(
            task_id=task_id,
            topic="Mars mission",
            instructions="Track launch updates",
            status=status,
            event_dict=event.model_dump(mode="json"),
        )
        self.state.add_task(task)
        return task

    async def test_run_continuous_cycle_skips_overlap_and_updates_state_once(self) -> None:
        task = self._build_task(task_id="cr_overlap")
        gate = asyncio.Event()
        release = asyncio.Event()

        async def fake_collect_cycle_material(active_task, task_dir, **kwargs):
            gate.set()
            await release.wait()
            return continuous_research.CycleResearchMaterial(
                queries=["mars mission latest updates"],
                new_or_changed_documents=[
                    continuous_research.SourceDocument(
                        query="mars mission latest updates",
                        title="Mars Update",
                        url="https://example.com/mars",
                        content_excerpt="Changed source copy.",
                        content_hash="hash-overlap",
                    )
                ],
                ledger=continuous_research.EvidenceLedger(),
            )

        cycle_output = continuous_research.CycleResult(
            found_new_info=False,
            new_findings="Nothing materially new.",
            updated_summary="Stable summary",
            updated_plan="- [ ] Check next launch update",
        )
        fake_agent = MagicMock()
        fake_agent.run = AsyncMock(return_value=SimpleNamespace(output=cycle_output))

        with patch.object(
            continuous_research,
            "_collect_cycle_material",
            side_effect=fake_collect_cycle_material,
        ), patch.object(
            continuous_research,
            "get_cycle_agent",
            return_value=fake_agent,
        ), patch.object(
            continuous_research,
            "send_proactive_update",
            new=AsyncMock(),
        ):
            first_cycle = asyncio.create_task(continuous_research._run_continuous_cycle(task))
            await gate.wait()
            second_result = await continuous_research._run_continuous_cycle(task)
            release.set()
            first_result = await first_cycle

        saved_task = self.state.get_task(task.task_id)
        self.assertEqual(second_result, "skipped_already_running")
        self.assertEqual(first_result, "no_change")
        self.assertEqual(saved_task.cycle_count, 1)
        self.assertFalse(saved_task.is_cycle_running)
        self.assertEqual(saved_task.last_summary, "Stable summary")
        self.assertTrue((self.output_dir / task.task_id / "summary.md").exists())

    async def test_status_reports_paused_tasks_and_last_error(self) -> None:
        running_task = self._build_task(task_id="cr_running", status="running")
        paused_task = self._build_task(task_id="cr_paused", status="paused")
        self.state.update_task(
            paused_task.task_id,
            last_error="network timeout while fetching source",
            cycle_count=3,
        )
        self.state.update_task(
            running_task.task_id,
            plan="- [ ] Review NASA updates",
            last_summary="Latest tracked summary",
        )

        status_text = await continuous_research.get_continuous_research_status(MagicMock())

        self.assertIn("cr_running", status_text)
        self.assertIn("cr_paused", status_text)
        self.assertIn("paused", status_text)
        self.assertIn("network timeout while fetching source", status_text)

    async def test_start_continuous_research_is_idempotent_for_same_event(self) -> None:
        event = TelegramMessageEvent(
            event_id="evt-dedupe",
            source="telegram",
            message="track mars mission updates",
            channel_id=111,
            sender_id=222,
            message_id=333,
        )
        ctx = MagicMock()
        ctx.deps = event

        def fake_create_task(coro):
            coro.close()
            return MagicMock()

        with patch.object(
            continuous_research.asyncio,
            "create_task",
            side_effect=fake_create_task,
        ):
            first_response = await continuous_research.start_continuous_research(
                ctx,
                "Mars mission",
                "Track launch updates",
            )
            second_response = await continuous_research.start_continuous_research(
                ctx,
                "Mars mission",
                "Track launch updates",
            )

        tasks = self.state.get_all_tasks()
        self.assertEqual(len(tasks), 1)
        self.assertIn("Started continuous research", first_response)
        self.assertIn("feedback.md", first_response)
        self.assertIn("already tracking this request", second_response)
        self.assertEqual(tasks[0].source_event_id, event.event_id)
        self.assertEqual(tasks[0].source_message_id, event.message_id)


if __name__ == "__main__":
    unittest.main()
