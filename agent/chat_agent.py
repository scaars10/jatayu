from __future__ import annotations

import logging

from google.genai import types

from agent.base_agent import BaseAgent
from agent.gemini_model import gemini_model, get_client
from agent.long_term_memory import LongTermMemoryManager
from agent.web_search import ChatAgentWebSearch, MAX_WEB_SEARCH_TOOL_TURNS
from config.env_config import get_env, get_env_map
from models import TelegramMessageEvent
from search.searxng import SearxngConfig, SearxngSearchTool
from storage import ConversationTurn, LongTermMemoryRecord, StorageService


class ChatAgent(BaseAgent):
    def __init__(
        self,
        storage_service: StorageService | None = None,
        history_limit: int = 12,
        memory_limit: int = 8,
        memory_manager: LongTermMemoryManager | None = None,
        search_tool: SearxngSearchTool | None = None,
        search_enabled: bool | None = None,
        search_max_results: int = 5,
    ) -> None:
        self.storage_service = storage_service
        self.history_limit = history_limit
        self.memory_limit = memory_limit
        self._logger = logging.getLogger(__name__)
        if memory_manager is not None:
            self.memory_manager = memory_manager
        elif storage_service is not None:
            self.memory_manager = LongTermMemoryManager(
                storage_service=storage_service,
                max_existing_memories=memory_limit,
            )
        else:
            self.memory_manager = None
        self.web_search = self._build_web_search(
            search_tool=search_tool,
            search_enabled=search_enabled,
            search_max_results=search_max_results,
        )

    async def stop(self) -> None:
        if self.web_search is not None:
            self.web_search.close()

    async def respond(self, event: TelegramMessageEvent) -> str | None:
        history: list[ConversationTurn] = []
        memories: list[LongTermMemoryRecord] = []
        contents = event.message
        if self.storage_service is not None:
            history = await self.storage_service.get_conversation_context(
                event.channel_id,
                limit=self.history_limit,
            )
            memories = await self.storage_service.get_long_term_memories(
                event.channel_id,
                event.sender_id,
                limit=self.memory_limit,
            )
            contents = self._build_prompt(history, memories, event.message)

        response_text = await self._generate_response(contents)
        if response_text and self.memory_manager is not None:
            try:
                await self.memory_manager.remember_text_exchange(
                    event,
                    response_text,
                    history=history,
                )
            except Exception as exc:
                print(f"[MEMORY][ERROR] {exc}")

        return response_text

    async def _generate_response(self, contents: str) -> str | None:
        client = get_client()
        model = gemini_model.get_balanced_model()

        if self.web_search is None:
            response = await client.aio.models.generate_content(
                model=model,
                contents=contents,
            )
            return response.text

        conversation = [self._build_user_content(contents)]
        config = self.web_search.generation_config
        response = await client.aio.models.generate_content(
            model=model,
            contents=conversation,
            config=config,
        )

        for _ in range(MAX_WEB_SEARCH_TOOL_TURNS):
            function_calls = list(getattr(response, "function_calls", None) or [])
            if not function_calls:
                return response.text

            conversation.append(self._extract_function_call_content(response))
            conversation.append(await self._build_tool_response_content(function_calls))
            response = await client.aio.models.generate_content(
                model=model,
                contents=conversation,
                config=config,
            )

        return response.text or "I couldn't complete that web lookup. Please try again."

    async def _build_tool_response_content(
        self,
        function_calls: list[types.FunctionCall],
    ) -> types.Content:
        if self.web_search is None:
            return types.Content(
                role="tool",
                parts=[
                    types.Part.from_function_response(
                        name="search_web",
                        response={"error": "Web search is unavailable."},
                    )
                ],
            )

        parts: list[types.Part] = []
        for function_call in function_calls:
            name = function_call.name or self.web_search.function_name
            if name != self.web_search.function_name:
                payload = {"error": f"Unsupported tool call: {name}"}
            else:
                payload = await self.web_search.execute(function_call.args)
            parts.append(
                types.Part.from_function_response(
                    name=name,
                    response=payload,
                )
            )

        return types.Content(role="tool", parts=parts)

    @staticmethod
    def _build_user_content(contents: str) -> types.Content:
        return types.Content(
            role="user",
            parts=[types.Part.from_text(text=contents)],
        )

    @staticmethod
    def _extract_function_call_content(response) -> types.Content:
        candidates = getattr(response, "candidates", None) or []
        if candidates:
            content = getattr(candidates[0], "content", None)
            if content is not None:
                parts = list(getattr(content, "parts", None) or [])
                if parts:
                    return types.Content(
                        role=getattr(content, "role", None) or "model",
                        parts=parts,
                    )

        function_calls = list(getattr(response, "function_calls", None) or [])
        parts = [
            types.Part.from_function_call(
                name=function_call.name or "search_web",
                args=dict(function_call.args or {}),
            )
            for function_call in function_calls
        ]
        return types.Content(role="model", parts=parts)

    def _build_web_search(
        self,
        *,
        search_tool: SearxngSearchTool | None,
        search_enabled: bool | None,
        search_max_results: int,
    ) -> ChatAgentWebSearch | None:
        if search_enabled is False:
            return None

        if search_tool is not None:
            return ChatAgentWebSearch(
                search_tool=search_tool,
                max_results=search_max_results,
                logger=self._logger,
            )

        if search_enabled is None and not self._is_web_search_enabled_from_env():
            return None

        config = SearxngConfig.from_env(get_env_map())
        if not config.enabled:
            raise ValueError(
                "Agent web search is enabled, but SearXNG is disabled in configuration."
            )

        return ChatAgentWebSearch(
            search_tool=SearxngSearchTool.from_config(config),
            max_results=search_max_results,
            owns_tool=True,
            logger=self._logger,
        )

    @staticmethod
    def _is_web_search_enabled_from_env() -> bool:
        raw_value = get_env("JATAYU_AGENT_WEB_SEARCH_ENABLED")
        if raw_value is None:
            raw_value = get_env("AGENT_WEB_SEARCH_ENABLED")
        if raw_value is None:
            return False

        normalized = raw_value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        raise ValueError(
            "JATAYU_AGENT_WEB_SEARCH_ENABLED must be a boolean value."
        )

    @staticmethod
    def _build_prompt(
        history: list[ConversationTurn],
        memories: list[LongTermMemoryRecord],
        current_message: str,
    ) -> str:
        if not history and not memories:
            return current_message

        lines = [
            "Use the long-term memory and recent conversation context below when relevant, then reply to the latest user message.",
        ]
        if memories:
            lines.append("Long-term memory:")
            for memory in memories:
                lines.append(
                    f"- [{memory.scope_type}/{memory.importance}] {memory.summary}"
                )

        if history:
            lines.append("Recent conversation:")
        for turn in history:
            lines.append(f"{turn.role}: {turn.text}")

        last_turn = history[-1] if history else None
        if last_turn is None or last_turn.role != "user" or last_turn.text != current_message:
            lines.append(f"user: {current_message}")

        lines.append("assistant:")
        return "\n".join(lines)
