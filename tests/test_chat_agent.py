import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import types

from agent.chat_agent import AgentOutput, ChatAgent
from models import TelegramMessageEvent
from storage import ConversationTurn, LongTermMemoryRecord


class ChatAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_respond_includes_recent_context_from_storage(self) -> None:
        storage_service = AsyncMock()
        storage_service.get_conversation_context.return_value = [
            ConversationTurn(
                role="user",
                text="Earlier question",
                source="telegram",
                message_type="message",
                occurred_at=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
            ),
            ConversationTurn(
                role="assistant",
                text="Earlier answer",
                source="agent",
                message_type="agent_response",
                occurred_at=datetime(2026, 3, 22, 10, 1, tzinfo=timezone.utc),
            ),
            ConversationTurn(
                role="user",
                text="Latest question",
                source="telegram",
                message_type="message",
                occurred_at=datetime(2026, 3, 22, 10, 2, tzinfo=timezone.utc),
            ),
        ]
        storage_service.get_long_term_memories.return_value = [
            LongTermMemoryRecord(
                id=1,
                scope_type="participant",
                scope_key="participant:200",
                channel_id=None,
                participant_id=200,
                memory_key="preferred_language",
                category="preference",
                summary="Prefers concise technical explanations.",
                importance="high",
                confidence=0.9,
                source_message_id=10,
                status="active",
                last_observed_at=datetime(2026, 3, 22, 9, 30, tzinfo=timezone.utc),
                created_at=datetime(2026, 3, 22, 9, 30, tzinfo=timezone.utc),
                updated_at=datetime(2026, 3, 22, 9, 30, tzinfo=timezone.utc),
            )
        ]
        memory_manager = AsyncMock()
        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(output=AgentOutput(response="Contextual answer", requires_audio=False))

        with patch("agent.chat_agent.get_env", return_value="fake_key"), patch("pydantic_ai.Agent.run", mock_run):
            agent = ChatAgent(
                storage_service=storage_service,
                history_limit=6,
                memory_limit=4,
                memory_manager=memory_manager,
            )
            event = TelegramMessageEvent(
                event_id="evt-context",
                source="telegram",
                message="Latest question",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )

            response = await agent.respond(event)

        self.assertEqual(response.response, "Contextual answer")
        storage_service.get_conversation_context.assert_awaited_once_with(100, limit=6)
        storage_service.get_long_term_memories.assert_awaited_once_with(100, 200, limit=4)
        memory_manager.remember_text_exchange.assert_awaited_once()
        
        prompt_parts = mock_run.await_args.args[0]
        prompt = prompt_parts[0] if isinstance(prompt_parts, list) else prompt_parts
        self.assertIn("Long-term memory:", prompt)
        self.assertIn("Prefers concise technical explanations.", prompt)
        self.assertIn("user: Earlier question", prompt)
        self.assertIn("assistant: Earlier answer", prompt)
        self.assertIn("user: Latest question", prompt)
        self.assertTrue(prompt.endswith("assistant:"))

    def test_build_prompt_falls_back_to_current_message_when_history_is_empty(self) -> None:
        prompt = ChatAgent._build_prompt([], [], "hello")

        self.assertEqual(prompt, "hello")


if __name__ == "__main__":
    unittest.main()
