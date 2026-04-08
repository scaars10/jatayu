import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import agent.continuous_research as continuous_research
import agent.continuous_state as continuous_state_module
from agent.continuous_state import ContinuousResearchState, ContinuousTask
from models import TelegramMessageEvent


class ContinuousResearchHelperTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_search_candidates_prefers_web_search_results_before_searxng(self) -> None:
        raw_result = "Sources:\n- Gemini Story | https://example.com/gemini-story"

        async def fake_web_search(query: str) -> str:
            return raw_result

        with patch.object(
            continuous_research,
            "web_search",
            new=fake_web_search,
        ), patch(
            "search.searxng.search_web",
            side_effect=AssertionError("SearXNG should not be called when Gemini returns candidates"),
        ):
            candidates, errors = await continuous_research._search_candidates("mars mission")

        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0].title, "Gemini Story")
        self.assertEqual(errors, [])


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

        async def fake_collect_cycle_material(active_task, task_dir):
            gate.set()
            await release.wait()
            return continuous_research.CycleResearchMaterial(
                queries=["mars mission latest updates"],
                ledger=continuous_research.EvidenceLedger(),
            )

        cycle_output = continuous_research.CycleResult(
            found_new_info=False,
            new_findings="Nothing materially new.",
            updated_summary="Stable summary",
            updated_plan="- [ ] Check next launch update",
        )
        fake_agent = MagicMock()
        fake_agent.run = AsyncMock(return_value=MagicMock(output=cycle_output))

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
        self.assertEqual(first_result, "completed_without_updates")
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
        self.assertIn("already tracking this request", second_response)
        self.assertEqual(tasks[0].source_event_id, event.event_id)
        self.assertEqual(tasks[0].source_message_id, event.message_id)


if __name__ == "__main__":
    unittest.main()
