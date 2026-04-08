import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from agent.long_term_memory import LongTermMemoryManager
from models import TelegramMessageEvent
from storage import ConversationTurn, LongTermMemoryRecord


class LongTermMemoryManagerTests(unittest.IsolatedAsyncioTestCase):
    async def test_remember_text_exchange_skips_research_related_exchanges(self) -> None:
        storage_service = AsyncMock()
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock()

        with patch("agent.long_term_memory.get_client", return_value=client):
            manager = LongTermMemoryManager(storage_service)
            event = TelegramMessageEvent(
                event_id="evt-research-memory",
                source="telegram",
                message="Keep searching for flats near Embassy Tech Village and notify me about good options.",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )

            stored = await manager.remember_text_exchange(
                event,
                "Started continuous research on 'flats near Embassy Tech Village'. Task ID is cr_1234abcd. I will keep looking in the background.",
                history=[],
            )

        self.assertEqual(stored, [])
        storage_service.get_long_term_memories.assert_not_called()
        storage_service.upsert_long_term_memory.assert_not_called()
        client.aio.models.generate_content.assert_not_called()

    async def test_remember_text_exchange_upserts_model_selected_memories(self) -> None:
        storage_service = AsyncMock()
        storage_service.get_long_term_memories.return_value = [
            LongTermMemoryRecord(
                id=1,
                scope_type="participant",
                scope_key="participant:200",
                channel_id=None,
                participant_id=200,
                memory_key="preferred_language",
                category="preference",
                summary="Prefers Python.",
                importance="high",
                confidence=0.9,
                source_message_id=10,
                status="active",
                last_observed_at=datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc),
                created_at=datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 3, 22, 9, 0, tzinfo=timezone.utc),
            )
        ]
        storage_service.upsert_long_term_memory.side_effect = [
            LongTermMemoryRecord(
                id=2,
                scope_type="participant",
                scope_key="participant:200",
                channel_id=None,
                participant_id=200,
                memory_key="favorite_editor",
                category="preference",
                summary="Uses Neovim as the primary editor.",
                importance="medium",
                confidence=0.83,
                source_message_id=11,
                status="active",
                last_observed_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
                created_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
            ),
            LongTermMemoryRecord(
                id=3,
                scope_type="channel",
                scope_key="channel:100",
                channel_id=100,
                participant_id=None,
                memory_key="project_name",
                category="project",
                summary="The channel is working on the Jatayu agent.",
                importance="high",
                confidence=0.88,
                source_message_id=11,
                status="active",
                last_observed_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
                created_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
                updated_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
            ),
        ]
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(
                text="""
                {
                  "memories": [
                    {
                      "scope": "participant",
                      "memory_key": "favorite_editor",
                      "category": "preference",
                      "summary": "Uses Neovim as the primary editor.",
                      "reasoning": "Explicit preference expressed by user.",
                      "importance": "medium",
                      "confidence": 0.90
                    },
                    {
                      "scope": "channel",
                      "memory_key": "project_name",
                      "category": "project",
                      "summary": "The channel is working on the Jatayu agent.",
                      "reasoning": "Fundamental project context for this channel.",
                      "importance": "high",
                      "confidence": 0.95
                    }
                  ]
                }
                """
            )
        )

        with patch("agent.long_term_memory.get_client", return_value=client):
            manager = LongTermMemoryManager(storage_service)
            event = TelegramMessageEvent(
                event_id="evt-memory",
                source="telegram",
                message="I use Neovim and we're building the Jatayu agent.",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )
            history = [
                ConversationTurn(
                    role="user",
                    text=event.message,
                    source="telegram",
                    message_type="message",
                    occurred_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
                )
            ]

            stored = await manager.remember_text_exchange(
                event,
                "I'll keep that in mind.",
                history=history,
            )

        self.assertEqual(len(stored), 2)
        storage_service.get_long_term_memories.assert_awaited_once_with(100, 200, limit=8)
        self.assertEqual(storage_service.upsert_long_term_memory.await_count, 2)
        first_call = storage_service.upsert_long_term_memory.await_args_list[0].kwargs
        self.assertEqual(first_call["scope_type"], "participant")
        self.assertEqual(first_call["memory_key"], "favorite_editor")
        second_call = storage_service.upsert_long_term_memory.await_args_list[1].kwargs
        self.assertEqual(second_call["scope_type"], "channel")
        self.assertEqual(second_call["memory_key"], "project_name")

    def test_parse_extraction_response_handles_invalid_json(self) -> None:
        parsed = LongTermMemoryManager._parse_extraction_response("not json")

        self.assertEqual(parsed, [])
