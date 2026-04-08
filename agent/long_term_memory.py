from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.compressor import global_compression_state
from agent.gemini_model import gemini_model, get_client
from models import TelegramMessageEvent
from storage import ConversationTurn, LongTermMemoryRecord, StorageService


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["channel", "participant"]
    memory_key: str
    category: str
    summary: str
    reasoning: str = Field(description="Why this fact is durable and worth remembering.")
    importance: Literal["low", "medium", "high"] = "medium"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class MemoryExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    memories: list[MemoryCandidate] = Field(default_factory=list)


class LongTermMemoryManager:
    def __init__(
        self,
        storage_service: StorageService,
        max_existing_memories: int = 8,
    ) -> None:
        self.storage_service = storage_service
        self.max_existing_memories = max_existing_memories

    async def remember_text_exchange(
        self,
        event: TelegramMessageEvent,
        response_text: str,
        *,
        history: list[ConversationTurn] | None = None,
    ) -> list[LongTermMemoryRecord]:
        if not event.message.strip() or not response_text.strip():
            return []

        if self._is_research_related_exchange(event.message, response_text):
            return []

        existing_memories = await self.storage_service.get_long_term_memories(
            event.channel_id,
            event.sender_id,
            limit=self.max_existing_memories,
        )
        recent_history = history or await self.storage_service.get_conversation_context(
            event.channel_id,
            limit=8,
        )

        prompt = self._build_extraction_prompt(
            history=recent_history,
            existing_memories=existing_memories,
            user_message=event.message,
            assistant_response=response_text,
        )
        raw_response = await get_client().aio.models.generate_content(
            model=gemini_model.get_balanced_model(),
            contents=prompt,
        )
        candidates = self._parse_extraction_response(raw_response.text or "")

        stored: list[LongTermMemoryRecord] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in candidates:
            # Skip if confidence is too low or it's just a low-importance one-off
            if candidate.confidence < 0.85:
                continue
            
            normalized_key = self._normalize_key(candidate.memory_key)
            category = self._normalize_key(candidate.category) or "general"
            summary = candidate.summary.strip()
            if not normalized_key or not summary:
                continue

            dedupe_key = (candidate.scope, normalized_key)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)

            record = await self.storage_service.upsert_long_term_memory(
                scope_type=candidate.scope,
                memory_key=normalized_key,
                category=category,
                summary=summary,
                importance=candidate.importance,
                confidence=candidate.confidence,
                channel_external_id=event.channel_id,
                participant_external_id=event.sender_id,
                source_event_id=event.event_id,
            )
            stored.append(record)

        if stored:
            global_compression_state.increment_updates(len(stored))

        return stored

    @staticmethod
    def _is_research_related_exchange(user_message: str, assistant_response: str) -> bool:
        user_text = user_message.strip().lower()
        assistant_text = assistant_response.strip().lower()
        combined_text = f"{user_text}\n{assistant_text}"

        response_markers = (
            "started deep research",
            "started continuous research",
            "continuous research",
            "deep research",
            "research task",
            "gathering sources for your research",
            "reading sources for your research",
            "synthesizing the report for your research",
            "provide feedback or tell me to continue",
            "continue to the final report",
            "i will keep working in the background",
            "i will keep looking in the background",
        )
        if any(marker in combined_text for marker in response_markers):
            return True

        if "sources:" in assistant_text and ("http://" in assistant_text or "https://" in assistant_text):
            return True

        if assistant_text.count("http://") + assistant_text.count("https://") >= 2:
            return True

        task_setup_patterns = (
            r"\bkeep searching\b",
            r"\bkeep looking\b",
            r"\bmonitor\b",
            r"\btrack\b",
            r"\bwatch for updates\b",
            r"\bnotify me\b",
            r"\balert me\b",
        )
        domain_patterns = (
            r"\bflat\b",
            r"\bflats\b",
            r"\bapartment\b",
            r"\bapartments\b",
            r"\bproperty\b",
            r"\bproperties\b",
            r"\blisting\b",
            r"\blistings\b",
            r"\bnews\b",
            r"\bupdates\b",
        )
        if any(re.search(pattern, user_text) for pattern in task_setup_patterns) and any(
            re.search(pattern, user_text) for pattern in domain_patterns
        ):
            return True

        return False

    @staticmethod
    def _build_extraction_prompt(
        *,
        history: list[ConversationTurn],
        existing_memories: list[LongTermMemoryRecord],
        user_message: str,
        assistant_response: str,
    ) -> str:
        lines = [
            "You are a long-term memory maintenance agent for a chat assistant.",
            "Analyze the latest exchange and decide if it contains durable facts worth remembering.",
            "",
            "### CRITERIA FOR REMEMBERING",
            "- DURABILITY: Information that will remain true and relevant for weeks or months.",
            "- UTILITY: Facts that help the assistant provide more personalized or context-aware help.",
            "- PERSONAL FOCUS: ONLY extract information that the USER has shared about themselves, their preferences, their life, or their projects.",
            "- IGNORE RESEARCH: DO NOT extract information from research reports, web search results, or general knowledge that the assistant has provided (e.g., apartment listings, technical documentation, news summaries, recipes, general facts). Only remember these if the user explicitly adopts them as a personal plan or preference.",
            "- EXAMPLES TO REMEMBER: User preferences, long-running projects, names of people/places, recurring tasks, hard constraints, and significant personal/professional milestones.",
            "- EXAMPLES TO IGNORE: One-off questions ('How do I fix this syntax error?'), temporary requests ('Summarize this link', 'What is the weather in London?'), generic small talk ('How are you?', 'Thanks for help'), specific code snippets that are context-dependent, and details that will be irrelevant by tomorrow.",
            "",
            "### DEDUPLICATION & UPDATING",
            "1. Review the 'Existing memories' list below.",
            "2. If a new fact relates to an existing memory, REUSE the exact 'memory_key' to update it.",
            "3. If a new fact is already covered by an existing memory and provides no new information, DO NOT extract it.",
            "4. If a new fact contradicts an existing memory, extract it with the same 'memory_key' - the system will update the summary.",
            "",
            "### DURABILITY CHECK",
            "Before adding a memory, ask yourself: 'Will the assistant still need to know this in 2 weeks?' If the answer is 'No', do not add it.",
            "",
            "### OUTPUT FORMAT",
            "Return strict JSON with this shape:",
            "{",
            '  "memories": [',
            "    {",
            '      "scope": "participant|channel",',
            '      "memory_key": "snake_case_key",',
            '      "category": "snake_case_category",',
            '      "summary": "Clear, concise fact description.",',
            '      "reasoning": "Explain why this is durable and not just a one-off request.",',
            '      "importance": "low|medium|high",',
            '      "confidence": 0.0 to 1.0',
            "    }",
            "  ]",
            "}",
            "If nothing meets the durability criteria, return {\"memories\":[]}.",
            "",
            "### CONTEXT",
        ]

        if existing_memories:
            lines.append("Existing memories:")
            for memory in existing_memories:
                lines.append(
                    f"- [{memory.scope_type}/{memory.importance}] {memory.memory_key}: {memory.summary}"
                )
            lines.append("")

        if history:
            lines.append("Recent conversation context:")
            for turn in history[-8:]:
                lines.append(f"{turn.role}: {turn.text}")
            lines.append("")

        lines.append("Latest exchange (PRIORITIZE THIS):")
        lines.append(f"user: {user_message}")
        lines.append(f"assistant: {assistant_response}")
        return "\n".join(lines)

    @classmethod
    def _parse_extraction_response(
        cls,
        text: str,
    ) -> list[MemoryCandidate]:
        if not text.strip():
            return []

        try:
            payload = json.loads(cls._extract_json_payload(text))
            result = MemoryExtractionResult.model_validate(payload)
        except (json.JSONDecodeError, ValidationError):
            return []

        return result.memories

    @staticmethod
    def _extract_json_payload(text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
            stripped = re.sub(r"\s*```$", "", stripped)

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start == -1 or end == -1 or end < start:
            return stripped

        return stripped[start:end + 1]

    @staticmethod
    def _normalize_key(value: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
        return normalized.strip("_")
