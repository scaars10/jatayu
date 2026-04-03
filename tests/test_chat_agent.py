import unittest
from datetime import datetime, timezone
from unittest import mock
from unittest.mock import AsyncMock, MagicMock, patch, ANY

from agent.chat_agent import AgentOutput, ChatAgent
from models import TelegramMessageEvent
from storage import ConversationTurn, LongTermMemoryRecord


class ChatAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_respond_includes_recent_context_from_storage(self) -> None:
        storage_service = AsyncMock()
        storage_service.knowledge_graph = MagicMock()
        storage_service.knowledge_graph.search_graph.return_value = {}
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
        mock_run.assert_awaited_once_with(mock.ANY, deps=event)
        
        prompt_parts = mock_run.await_args.args[0]
        prompt = prompt_parts[0] if isinstance(prompt_parts, list) else prompt_parts
        self.assertIn("Long-term memory:", prompt)
        self.assertIn("Prefers concise technical explanations.", prompt)
        self.assertIn("user: Earlier question", prompt)
        self.assertIn("assistant: Earlier answer", prompt)
        self.assertIn("user: Latest question", prompt)
        self.assertTrue(prompt.endswith("assistant:"))

    def test_build_prompt_falls_back_to_current_message_when_history_is_empty(self) -> None:
        prompt = ChatAgent._build_prompt([], [], {}, "hello")

        self.assertEqual(prompt, "hello")

    async def test_respond_triggers_deep_research_tool(self) -> None:
        storage_service = AsyncMock()
        storage_service.knowledge_graph = MagicMock()
        storage_service.knowledge_graph.search_graph.return_value = {}
        storage_service.get_conversation_context.return_value = []
        storage_service.get_long_term_memories.return_value = []
        memory_manager = AsyncMock()

        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(output=AgentOutput(
            response="I have started deep research on 'AI advancements'. I will keep working in the background and send you a detailed report once I've compiled my findings.",
            requires_audio=False
        ))

        with patch("agent.chat_agent.get_env", return_value="fake_key"), patch("pydantic_ai.Agent.run", mock_run):

            agent = ChatAgent(
                storage_service=storage_service,
                memory_manager=memory_manager,
            )
            event = TelegramMessageEvent(
                event_id="evt-research",
                source="telegram",
                message="Can you do a deep research on AI advancements?",
                channel_id=101,
                sender_id=201,
                message_id=301,
            )

            response = await agent.respond(event)

        self.assertIn("deep research", response.response)
        mock_run.assert_awaited_once_with(mock.ANY, deps=event)

    async def test_respond_triggers_get_research_task_status_tool(self) -> None:
        storage_service = AsyncMock()
        storage_service.knowledge_graph = MagicMock()
        storage_service.knowledge_graph.search_graph.return_value = {}
        storage_service.get_conversation_context.return_value = []
        storage_service.get_long_term_memories.return_value = []
        memory_manager = AsyncMock()

        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(output=AgentOutput(
            response="Research task 123 on 'AI safety' is currently in_progress.",
            requires_audio=False
        ))

        with patch("agent.chat_agent.get_env", return_value="fake_key"), \
             patch("pydantic_ai.Agent.run", mock_run):

            agent = ChatAgent(
                storage_service=storage_service,
                memory_manager=memory_manager,
            )
            event = TelegramMessageEvent(
                event_id="evt-status",
                source="telegram",
                message="what is the status of task 123",
                channel_id=102,
                sender_id=202,
                message_id=302,
            )

            response = await agent.respond(event)

        self.assertIn("task 123", response.response)
        mock_run.assert_awaited_once_with(mock.ANY, deps=event)

    async def test_respond_triggers_provide_feedback_to_research_task_tool(self) -> None:
        storage_service = AsyncMock()
        storage_service.knowledge_graph = MagicMock()
        storage_service.knowledge_graph.search_graph.return_value = {}
        storage_service.get_conversation_context.return_value = []
        storage_service.get_long_term_memories.return_value = []
        memory_manager = AsyncMock()

        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(output=AgentOutput(
            response="Feedback has been provided to research task 123.",
            requires_audio=False
        ))

        with patch("agent.chat_agent.get_env", return_value="fake_key"), \
             patch("pydantic_ai.Agent.run", mock_run):

            agent = ChatAgent(
                storage_service=storage_service,
                memory_manager=memory_manager,
            )
            event = TelegramMessageEvent(
                event_id="evt-feedback",
                source="telegram",
                message="provide feedback to task 123: focus on the ethics of AI",
                channel_id=103,
                sender_id=203,
                message_id=303,
            )

            response = await agent.respond(event)

        self.assertIn("Feedback has been provided", response.response)
        mock_run.assert_awaited_once_with(mock.ANY, deps=event)

    async def test_respond_triggers_continue_research_task_tool(self) -> None:
        storage_service = AsyncMock()
        storage_service.knowledge_graph = MagicMock()
        storage_service.knowledge_graph.search_graph.return_value = {}
        storage_service.get_conversation_context.return_value = []
        storage_service.get_long_term_memories.return_value = []
        memory_manager = AsyncMock()

        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(output=AgentOutput(
            response="Continuing research on 'AI safety'.\n\nNext step: read_sources.\n\nI will now proceed with the research. I will let you know when the next step is complete.",
            requires_audio=False
        ))

        with patch("agent.chat_agent.get_env", return_value="fake_key"), \
             patch("pydantic_ai.Agent.run", mock_run):

            agent = ChatAgent(
                storage_service=storage_service,
                memory_manager=memory_manager,
            )
            event = TelegramMessageEvent(
                event_id="evt-continue",
                source="telegram",
                message="continue research on task 123",
                channel_id=104,
                sender_id=204,
                message_id=304,
            )

            response = await agent.respond(event)

        self.assertIn("Continuing research on 'AI safety'", response.response)
        self.assertIn("Next step: read_sources", response.response)
        mock_run.assert_awaited_once_with(mock.ANY, deps=event)

    async def test_respond_triggers_read_pdf_tool(self) -> None:
        storage_service = AsyncMock()
        storage_service.knowledge_graph = MagicMock()
        storage_service.knowledge_graph.search_graph.return_value = {}
        storage_service.get_conversation_context.return_value = []
        storage_service.get_long_term_memories.return_value = []
        memory_manager = AsyncMock()

        mock_run = AsyncMock()
        mock_run.return_value = MagicMock(output=AgentOutput(
            response="The PDF contains information about AI.",
            requires_audio=False
        ))

        with patch("agent.chat_agent.get_env", return_value="fake_key"), \
             patch("pydantic_ai.Agent.run", mock_run):

            agent = ChatAgent(
                storage_service=storage_service,
                memory_manager=memory_manager,
            )
            event = TelegramMessageEvent(
                event_id="evt-pdf",
                source="telegram",
                message="read the pdf at https://example.com/ai.pdf",
                channel_id=105,
                sender_id=205,
                message_id=305,
            )

            response = await agent.respond(event)

        self.assertIn("The PDF contains information about AI", response.response)
        mock_run.assert_awaited_once_with(mock.ANY, deps=event)

if __name__ == "__main__":
    unittest.main()
