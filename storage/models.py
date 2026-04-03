from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class ChannelRecord:
    id: int
    provider: str
    external_id: str
    channel_type: str | None
    title: str | None
    metadata_json: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ParticipantRecord:
    id: int
    provider: str
    external_id: str
    display_name: str | None
    metadata_json: str | None
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    id: int
    channel_id: int
    status: str
    summary: str | None
    started_at: datetime
    last_message_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MessageRecord:
    id: int
    event_id: str
    conversation_id: int
    channel_id: int
    participant_id: int | None
    source: str
    direction: str
    role: str
    message_type: str
    provider_message_id: str | None
    reply_to_provider_message_id: str | None
    request_event_id: str | None
    text_content: str | None
    payload_json: str
    delivery_status: str
    occurred_at: datetime
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class MessageAttachmentRecord:
    id: int
    message_id: int
    media_type: str
    provider_file_id: str
    provider_file_unique_id: str
    mime_type: str | None
    file_name: str | None
    width: int | None
    height: int | None
    duration_seconds: int | None
    file_size: int | None
    performer: str | None
    title: str | None
    caption: str | None
    media_group_id: str | None
    payload_json: str | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class ConversationTurn:
    role: str
    text: str
    source: str
    message_type: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class LongTermMemoryRecord:
    id: int
    scope_type: str
    scope_key: str
    channel_id: int | None
    participant_id: int | None
    memory_key: str
    category: str
    summary: str
    importance: str
    confidence: float
    source_message_id: int | None
    status: str
    last_observed_at: datetime
    created_at: datetime
    updated_at: datetime

@dataclass(frozen=True, slots=True)
class ResearchTaskRecord:
    id: int
    topic: str
    specific_questions: str | None
    status: str
    step: str | None
    sources_content: str | None
    report: str | None
    feedback: str | None
    created_at: datetime
    updated_at: datetime
