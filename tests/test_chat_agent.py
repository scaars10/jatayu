import unittest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from google.genai import types

from agent.chat_agent import ChatAgent
from models import TelegramMessageEvent
from search.searxng import SearchToolBriefResult, SearchToolResult
from storage import ConversationTurn, LongTermMemoryRecord


class _StubSearchTool:
    def __init__(self, brief_result: SearchToolBriefResult) -> None:
        self.config = MagicMock(max_results=8)
        self._brief_result = brief_result
        self.calls: list[dict[str, object]] = []

    def search_web_brief(
        self,
        query: str,
        *,
        max_results: int = 5,
        categories=None,
        engines=None,
        language=None,
        page: int = 1,
        use_cache: bool = True,
    ) -> SearchToolBriefResult:
        self.calls.append(
            {
                "query": query,
                "max_results": max_results,
                "categories": categories,
                "engines": engines,
                "language": language,
                "page": page,
                "use_cache": use_cache,
            }
        )
        return self._brief_result


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
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            return_value=MagicMock(text="Contextual answer")
        )

        with patch("agent.chat_agent.get_client", return_value=client):
            agent = ChatAgent(
                storage_service=storage_service,
                history_limit=6,
                memory_limit=4,
                memory_manager=memory_manager,
                search_enabled=False,
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

        self.assertEqual(response, "Contextual answer")
        storage_service.get_conversation_context.assert_awaited_once_with(100, limit=6)
        storage_service.get_long_term_memories.assert_awaited_once_with(100, 200, limit=4)
        memory_manager.remember_text_exchange.assert_awaited_once()
        prompt = client.aio.models.generate_content.await_args.kwargs["contents"]
        self.assertIn("Long-term memory:", prompt)
        self.assertIn("Prefers concise technical explanations.", prompt)
        self.assertIn("user: Earlier question", prompt)
        self.assertIn("assistant: Earlier answer", prompt)
        self.assertIn("user: Latest question", prompt)
        self.assertTrue(prompt.endswith("assistant:"))

    def test_build_prompt_falls_back_to_current_message_when_history_is_empty(self) -> None:
        prompt = ChatAgent._build_prompt([], [], "hello")

        self.assertEqual(prompt, "hello")

    async def test_respond_executes_web_search_tool_when_model_requests_it(self) -> None:
        initial_response = MagicMock(
            text=None,
            function_calls=[
                types.FunctionCall(
                    name="search_web",
                    args={"query": "latest mars mission", "max_results": 2},
                )
            ],
            candidates=[
                MagicMock(
                    content=types.Content(
                        role="model",
                        parts=[
                            types.Part.from_function_call(
                                name="search_web",
                                args={
                                    "query": "latest mars mission",
                                    "max_results": 2,
                                },
                            )
                        ],
                    )
                )
            ],
        )
        final_response = MagicMock(
            text="Latest Mars mission results: https://example.com/mars",
            function_calls=[],
            candidates=[],
        )
        client = MagicMock()
        client.aio.models.generate_content = AsyncMock(
            side_effect=[initial_response, final_response]
        )
        search_tool = _StubSearchTool(
            SearchToolBriefResult(
                summary="Top 1 SearXNG results for 'latest mars mission':",
                result=SearchToolResult(
                    query="latest mars mission",
                    total_results=1,
                    results=(),
                ),
            )
        )

        with patch("agent.chat_agent.get_client", return_value=client):
            agent = ChatAgent(
                memory_manager=AsyncMock(),
                search_tool=search_tool,
                search_enabled=True,
            )
            event = TelegramMessageEvent(
                event_id="evt-search",
                source="telegram",
                message="What is the latest Mars mission?",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )

            response = await agent.respond(event)

        self.assertEqual(
            response,
            "Latest Mars mission results: https://example.com/mars",
        )
        self.assertEqual(len(search_tool.calls), 1)
        self.assertEqual(search_tool.calls[0]["query"], "latest mars mission")
        self.assertEqual(search_tool.calls[0]["max_results"], 2)
        self.assertEqual(client.aio.models.generate_content.await_count, 2)
        second_call_contents = client.aio.models.generate_content.await_args_list[1].kwargs[
            "contents"
        ]
        self.assertEqual(len(second_call_contents), 3)
        self.assertEqual(second_call_contents[1].role, "model")
        self.assertEqual(second_call_contents[2].role, "tool")


if __name__ == "__main__":
    unittest.main()
