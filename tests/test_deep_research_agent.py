import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import agent.deep_research_agent as deep_research_agent
from models import TelegramMessageEvent
from storage.models import ResearchTaskRecord


class DeepResearchWorkspaceTests(unittest.TestCase):
    def test_sync_research_workspace_writes_task_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "deep_research_output"
            task = ResearchTaskRecord(
                id=42,
                topic="AI safety",
                specific_questions="What changed?\nWhat matters?",
                status="paused",
                step="read_sources",
                sources_content="Source: https://example.com/article\nImportant findings",
                report="Sources:\nhttps://example.com/article\nhttps://example.com/notes",
                feedback="Focus on governance implications.",
                created_at=datetime(2026, 4, 8, 9, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 4, 8, 9, 5, tzinfo=timezone.utc),
            )
            event = TelegramMessageEvent(
                event_id="evt-deep-workspace",
                source="telegram",
                message="Do deep research on AI safety",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )

            with patch.object(
                deep_research_agent,
                "DEEP_RESEARCH_OUTPUT_DIR",
                output_dir,
            ):
                task_dir = deep_research_agent._sync_research_workspace(task, event)

            self.assertTrue((task_dir / "metadata.json").exists())
            self.assertTrue((task_dir / "specific_questions.md").exists())
            self.assertTrue((task_dir / "feedback.md").exists())
            self.assertTrue((task_dir / "sources.txt").exists())
            self.assertTrue((task_dir / "sources_content.md").exists())
            self.assertTrue((task_dir / "report.md").exists())

            metadata = json.loads((task_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["task_id"], 42)
            self.assertEqual(metadata["request_event"]["event_id"], event.event_id)


if __name__ == "__main__":
    unittest.main()
