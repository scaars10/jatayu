from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from agent.gemini_model import gemini_model, get_client
from models import TelegramMessageEvent
from storage import ConversationTurn, LongTermMemoryRecord, StorageService


class MemoryCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scope: Literal["channel", "participant"]
    memory_key: str
    category: str
    summary: str
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
            model=gemini_model.get_light_model(),
            contents=prompt,
        )
        candidates = self._parse_extraction_response(raw_response.text or "")

        stored: list[LongTermMemoryRecord] = []
        seen_keys: set[tuple[str, str]] = set()
        for candidate in candidates:
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

        return stored

    @staticmethod
    def _build_extraction_prompt(
        *,
        history: list[ConversationTurn],
        existing_memories: list[LongTermMemoryRecord],
        user_message: str,
        assistant_response: str,
    ) -> str:
        lines = [
            "You maintain long-term memory for a chat agent.",
            "Decide whether the exchange contains durable information worth remembering for future conversations.",
            "Store only facts that are likely to matter later: user identity, preferences, long-running projects, recurring tasks, constraints, commitments, important dates, and stable relationships.",
            "Do not store one-off requests, temporary context, generic small talk, or details that will expire quickly.",
            "Reuse an existing memory_key when updating the same fact.",
            'Return strict JSON with this shape: {"memories":[{"scope":"participant|channel","memory_key":"snake_case","category":"snake_case","summary":"text","importance":"low|medium|high","confidence":0.0}]}',
            "If nothing should be remembered, return {\"memories\":[]}.",
        ]

        if existing_memories:
            lines.append("Existing memories:")
            for memory in existing_memories:
                lines.append(
                    f"- [{memory.scope_type}/{memory.importance}] {memory.memory_key}: {memory.summary}"
                )

        if history:
            lines.append("Recent conversation:")
            for turn in history[-8:]:
                lines.append(f"{turn.role}: {turn.text}")

        lines.append("Latest exchange:")
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
