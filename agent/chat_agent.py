from __future__ import annotations

import logging
import re
from types import SimpleNamespace

from pydantic import BaseModel, Field
from pydantic_ai import Agent, BinaryContent
from agent.research_steps import web_search
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.base_agent import AgentReply, BaseAgent
from agent.continuous_state import ContinuousTask, global_continuous_state
from agent.deep_research_agent import start_deep_research_task, get_research_task_status, provide_feedback_to_research_task, continue_research_task, read_pdf
from agent.continuous_research import start_continuous_research, stop_continuous_research, pause_continuous_research, resume_continuous_research, get_continuous_research_status, update_continuous_research_plan, trigger_continuous_research_cycle
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
    _INTERNAL_TOOL_CALL_RE = re.compile(
        r'["\'}\]]*\s*call:[^:\s{}]+(?::[^:\s{}]+)*:final_result\{.*$',
        flags=re.DOTALL,
    )

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
            deps_type=TelegramMessageEvent,
            tools=[
                web_search,
                start_deep_research_task,
                get_research_task_status,
                provide_feedback_to_research_task,
                continue_research_task,
                read_pdf,
                start_continuous_research,
                stop_continuous_research,
                pause_continuous_research,
                resume_continuous_research,
                get_continuous_research_status,
                update_continuous_research_plan,
                trigger_continuous_research_cycle
            ],
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

        result = await self._generate_response(contents, event)
        if result:
            result.response = self._sanitize_response_text(result.response)
            if self.memory_manager is not None:
                try:
                    await self.memory_manager.remember_text_exchange(
                        event,
                        result.response,
                        history=history,
                    )
                except Exception as exc:
                    self._logger.error(f"[MEMORY][ERROR] {exc}")

        if not result:
            return None

        return AgentReply(
            response=result.response,
            requires_audio=result.requires_audio,
        )

    async def _generate_response(
        self, 
        contents: str, 
        event: TelegramMessageEvent
    ) -> AgentOutput | None:
        direct_research_response = await self._maybe_start_continuous_research(event)
        if direct_research_response is not None:
            return direct_research_response

        direct_feedback_response = await self._maybe_route_continuous_feedback(event)
        if direct_feedback_response is not None:
            return direct_feedback_response
        
        prompt_parts = []
        if event.audio_bytes and event.audio_mime_type:
            prompt_parts.append(BinaryContent(data=event.audio_bytes, media_type=event.audio_mime_type))
            
        prompt_parts.append(contents)

        # Pydantic AI Agent Run
        result = await self.agent.run(prompt_parts, deps=event)
        return result.output

    async def _maybe_start_continuous_research(
        self,
        event: TelegramMessageEvent,
    ) -> AgentOutput | None:
        message = event.message.strip()
        if not self._should_direct_start_continuous_research(message):
            return None

        topic = self._derive_continuous_research_topic(message)
        instructions = " ".join(message.split())
        response = await start_continuous_research(
            SimpleNamespace(deps=event),
            topic=topic,
            instructions=instructions,
        )
        return AgentOutput(response=response, requires_audio=False)

    async def _maybe_route_continuous_feedback(
        self,
        event: TelegramMessageEvent,
    ) -> AgentOutput | None:
        task = self._find_continuous_feedback_target(event)
        if task is None:
            return None

        response = await update_continuous_research_plan(
            SimpleNamespace(deps=event),
            task.task_id,
            event.message.strip(),
        )
        return AgentOutput(response=response, requires_audio=False)

    @classmethod
    def _sanitize_response_text(cls, text: str) -> str:
        if not text:
            return text

        cleaned = cls._INTERNAL_TOOL_CALL_RE.sub("", text).strip()
        if cleaned:
            return cleaned
        return text.strip()

    @staticmethod
    def _should_direct_start_continuous_research(message: str) -> bool:
        normalized = " ".join(message.lower().split())
        management_markers = (
            "status of task",
            "status for task",
            "continue research",
            "pause continuous research",
            "resume continuous research",
            "stop continuous research",
            "feedback to task",
        )
        if any(marker in normalized for marker in management_markers):
            return False

        ongoing_markers = (
            "keep searching",
            "keep on searching",
            "keep looking",
            "keep on looking",
            "keep checking",
            "monitor",
            "track",
            "watch for",
            "notify me",
            "alert me",
            "over time",
            "ongoing",
        )
        subject_markers = (
            "flat",
            "flats",
            "apartment",
            "apartments",
            "property",
            "properties",
            "listing",
            "listings",
            "house",
            "houses",
            "villa",
            "news",
            "updates",
            "price",
            "prices",
        )
        return any(marker in normalized for marker in ongoing_markers) and any(
            marker in normalized for marker in subject_markers
        )

    @staticmethod
    def _derive_continuous_research_topic(message: str) -> str:
        normalized = " ".join(message.split()).strip()
        lower = normalized.lower()

        cut_markers = (
            " and i want to ",
            ". i want to ",
            " basically ",
            " so that ",
        )
        cut_index = len(normalized)
        for marker in cut_markers:
            marker_index = lower.find(marker)
            if marker_index > 0:
                cut_index = min(cut_index, marker_index)

        topic = normalized[:cut_index].strip(" ,.;")
        topic = re.sub(r"^(i am|i'm|im)\s+looking for\s+", "", topic, flags=re.IGNORECASE)
        topic = re.sub(r"^looking for\s+", "", topic, flags=re.IGNORECASE)
        topic = topic.strip()

        if not topic:
            topic = normalized[:160].rstrip(" ,.;")

        return topic

    @classmethod
    def _find_continuous_feedback_target(
        cls,
        event: TelegramMessageEvent,
    ) -> ContinuousTask | None:
        if not cls._looks_like_continuous_feedback(event.message):
            return None

        candidates = [
            task
            for task in global_continuous_state.get_all_tasks()
            if task.status in {"running", "paused"}
            and task.source_channel_id == event.channel_id
        ]
        if len(candidates) != 1:
            return None

        return candidates[0]

    @staticmethod
    def _looks_like_continuous_feedback(message: str) -> bool:
        normalized = " ".join(message.lower().split())
        feedback_markers = (
            "keep it in plan",
            "remove it from plan",
            "i don't want",
            "i dont want",
            "i do not want",
            "no preference",
            "with or without",
            "exclude",
            "include",
            "focus on",
            "prioritize",
            "deprioritize",
            "servant room",
            "value for money",
            "good value",
            "my budget",
            "budget is",
            "makes sense",
        )
        return any(marker in normalized for marker in feedback_markers)

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
