from __future__ import annotations

import logging

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent, RunContext
from pydantic_ai.builtin_tools import WebSearchTool
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.base_agent import AgentReply, BaseAgent
from agent.gemini_model import gemini_model
from agent.long_term_memory import LongTermMemoryManager
from agent.tool_manager import SYSTEM_INSTRUCTION
from config.env_config import get_env, init_config
from models import TelegramMessageEvent
from storage import ConversationTurn, LongTermMemoryRecord, StorageService

class AgentOutput(BaseModel):
    response: str = Field(description="The text response to the user.")
    requires_audio: bool = Field(default=False, description="Set to true if the user explicitly asked for an audio response or voice note.")

class ChatAgent(BaseAgent):
    def __init__(
        self,
        storage_service: StorageService | None = None,
        history_limit: int = 12,
        memory_limit: int = 8,
        memory_manager: LongTermMemoryManager | None = None,
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
            
        init_config()
        api_key = get_env("GEMINI_API_KEY", required=True)
        self.agent = Agent(
            model=GoogleModel(
                gemini_model.get_balanced_model(),
                provider=GoogleProvider(api_key=api_key),
            ),
            system_prompt=SYSTEM_INSTRUCTION,
            builtin_tools=[WebSearchTool()],
            output_type=AgentOutput,
        )

    async def stop(self) -> None:
        pass

    async def respond(self, event: TelegramMessageEvent) -> AgentReply | None:
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

        result = await self._generate_response(contents, event.audio_bytes, event.audio_mime_type)
        if result and self.memory_manager is not None:
            try:
                await self.memory_manager.remember_text_exchange(
                    event,
                    result.response,
                    history=history,
                )
            except Exception as exc:
                print(f"[MEMORY][ERROR] {exc}")

        if not result:
            return None

        return AgentReply(
            response=result.response,
            requires_audio=result.requires_audio,
        )

    async def _generate_response(
        self, 
        contents: str, 
        audio_bytes: bytes | None, 
        audio_mime_type: str | None
    ) -> AgentOutput | None:
        
        prompt_parts = []
        if audio_bytes and audio_mime_type:
            prompt_parts.append(BinaryContent(data=audio_bytes, media_type=audio_mime_type))
            
        prompt_parts.append(contents)

        # Pydantic AI Agent Run
        # We can optionally pass tools or deps here if needed in the future
        result = await self.agent.run(prompt_parts)
        return result.output

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
