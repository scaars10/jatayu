from __future__ import annotations

import asyncio
from datetime import datetime

from models import (
    AgentResponseEvent,
    BaseEvent,
    TelegramAudioEvent,
    TelegramMessageEvent,
    TelegramPhotoEvent,
)

from storage.db import Database
from storage.models import ConversationTurn, LongTermMemoryRecord, MessageRecord
from storage.repositories import (
    AttachmentRepository,
    ChannelRepository,
    ConversationRepository,
    LongTermMemoryRepository,
    MessageRepository,
    ParticipantRepository,
    ResearchTaskRepository,
    KnowledgeGraphRepository,
)


class StorageService:
    def __init__(self, database: Database | None = None) -> None:
        self.database = database or Database()
        self._lock = asyncio.Lock()
        self._started = False
        self.channels: ChannelRepository | None = None
        self.participants: ParticipantRepository | None = None
        self.conversations: ConversationRepository | None = None
        self.messages: MessageRepository | None = None
        self.attachments: AttachmentRepository | None = None
        self.long_term_memories: LongTermMemoryRepository | None = None
        self.research_tasks: ResearchTaskRepository | None = None
        self.knowledge_graph: KnowledgeGraphRepository | None = None

    async def start(self) -> None:
        async with self._lock:
            if self._started:
                return

            connection = self.database.connect()
            self.database.migrate()
            self.channels = ChannelRepository(connection)
            self.participants = ParticipantRepository(connection)
            self.conversations = ConversationRepository(connection)
            self.messages = MessageRepository(connection)
            self.attachments = AttachmentRepository(connection)
            self.long_term_memories = LongTermMemoryRepository(connection)
            self.research_tasks = ResearchTaskRepository(connection)
            self.knowledge_graph = KnowledgeGraphRepository(self.database)
            self._started = True

    async def close(self) -> None:
        async with self._lock:
            if not self._started:
                return

            self.database.close()
            self.channels = None
            self.participants = None
            self.conversations = None
            self.messages = None
            self.attachments = None
            self.long_term_memories = None
            self.research_tasks = None
            self.knowledge_graph = None
            self._started = False

    async def record_event(
        self,
        event: BaseEvent,
        *,
        channel_source: str | None = None,
    ) -> MessageRecord:
        await self.start()

        async with self._lock:
            repositories = self._require_repositories()
            messages = repositories["messages"]
            existing = messages.get_by_event_id(event.event_id)
            if existing is not None:
                return existing

            channel_provider = channel_source or self._get_channel_provider(event)
            participant_provider = "agent" if isinstance(event, AgentResponseEvent) else event.source

            with self.database.connection:
                channel = repositories["channels"].upsert(
                    provider=channel_provider,
                    external_id=str(event.channel_id),
                )
                participant = repositories["participants"].upsert(
                    provider=participant_provider,
                    external_id=str(event.sender_id),
                )
                conversation = repositories["conversations"].get_or_create_active(
                    channel_id=channel.id,
                    started_at=event.occurred_at,
                )
                repositories["conversations"].attach_participant(
                    conversation_id=conversation.id,
                    participant_id=participant.id,
                    joined_at=event.occurred_at,
                )
                message = messages.create(
                    event_id=event.event_id,
                    conversation_id=conversation.id,
                    channel_id=channel.id,
                    participant_id=participant.id,
                    source=event.source,
                    direction=self._get_direction(event),
                    role=self._get_role(event),
                    message_type=self._get_message_type(event),
                    provider_message_id=self._get_provider_message_id(event),
                    reply_to_provider_message_id=self._get_reply_to_provider_message_id(event),
                    request_event_id=getattr(event, "request_event_id", None),
                    text_content=self._get_text_content(event),
                    payload_json=event.model_dump_json(),
                    delivery_status=self._get_delivery_status(event),
                    occurred_at=event.occurred_at,
                )
                attachment = self._build_attachment_payload(message.id, event)
                if attachment is not None:
                    repositories["attachments"].create(**attachment)
                repositories["conversations"].touch(
                    conversation_id=conversation.id,
                    last_message_at=event.occurred_at,
                )

            return message

    async def mark_message_delivered(
        self,
        event_id: str,
        provider_message_id: int | str,
    ) -> MessageRecord | None:
        await self.start()

        async with self._lock:
            repositories = self._require_repositories()
            with self.database.connection:
                return repositories["messages"].mark_delivered(
                    event_id=event_id,
                    provider_message_id=str(provider_message_id),
                )

    async def get_conversation_context(
        self,
        channel_external_id: int | str,
        *,
        channel_provider: str = "telegram",
        limit: int = 12,
    ) -> list[ConversationTurn]:
        await self.start()

        async with self._lock:
            repositories = self._require_repositories()
            channel = repositories["channels"].get_by_provider_external_id(
                provider=channel_provider,
                external_id=str(channel_external_id),
            )
            if channel is None:
                return []

            return repositories["messages"].list_recent_turns(
                channel_id=channel.id,
                limit=limit,
            )

    async def get_long_term_memories(
        self,
        channel_external_id: int | str,
        participant_external_id: int | str | None,
        *,
        channel_provider: str = "telegram",
        participant_provider: str = "telegram",
        limit: int = 8,
    ) -> list[LongTermMemoryRecord]:
        await self.start()

        async with self._lock:
            repositories = self._require_repositories()
            channel = repositories["channels"].get_by_provider_external_id(
                provider=channel_provider,
                external_id=str(channel_external_id),
            )
            if channel is None:
                return []

            participant_id: int | None = None
            if participant_external_id is not None:
                participant = repositories["participants"].get_by_provider_external_id(
                    provider=participant_provider,
                    external_id=str(participant_external_id),
                )
                participant_id = participant.id if participant is not None else None

            return repositories["long_term_memories"].list_relevant(
                channel_id=channel.id,
                participant_id=participant_id,
                limit=limit,
            )

    async def upsert_long_term_memory(
        self,
        *,
        scope_type: str,
        memory_key: str,
        category: str,
        summary: str,
        importance: str,
        confidence: float,
        channel_external_id: int | str,
        participant_external_id: int | str | None = None,
        source_event_id: str | None = None,
        observed_at: datetime | None = None,
        channel_provider: str = "telegram",
        participant_provider: str = "telegram",
    ) -> LongTermMemoryRecord:
        await self.start()

        async with self._lock:
            repositories = self._require_repositories()
            with self.database.connection:
                channel = repositories["channels"].upsert(
                    provider=channel_provider,
                    external_id=str(channel_external_id),
                )
                participant = None
                if participant_external_id is not None:
                    participant = repositories["participants"].upsert(
                        provider=participant_provider,
                        external_id=str(participant_external_id),
                    )

                source_message_id = None
                source_message = None
                if source_event_id is not None:
                    source_message = repositories["messages"].get_by_event_id(source_event_id)
                    source_message_id = source_message.id if source_message is not None else None

                last_observed_at = observed_at
                if last_observed_at is None and source_message is not None:
                    last_observed_at = source_message.occurred_at
                if last_observed_at is None:
                    raise ValueError("observed_at or source_event_id with a stored message is required")

                scope_key, channel_id, participant_id = self._build_memory_scope(
                    scope_type=scope_type,
                    channel_id=channel.id,
                    participant_id=participant.id if participant is not None else None,
                )

                return repositories["long_term_memories"].upsert(
                    scope_type=scope_type,
                    scope_key=scope_key,
                    channel_id=channel_id,
                    participant_id=participant_id,
                    memory_key=memory_key,
                    category=category,
                    summary=summary,
                    importance=importance,
                    confidence=confidence,
                    source_message_id=source_message_id,
                    last_observed_at=last_observed_at,
                )

    async def archive_long_term_memory(
        self,
        *,
        scope_type: str,
        memory_key: str,
        channel_external_id: int | str,
        participant_external_id: int | str | None = None,
        channel_provider: str = "telegram",
        participant_provider: str = "telegram",
    ) -> LongTermMemoryRecord | None:
        await self.start()

        async with self._lock:
            repositories = self._require_repositories()
            channel = repositories["channels"].get_by_provider_external_id(
                provider=channel_provider,
                external_id=str(channel_external_id),
            )
            if channel is None:
                return None

            participant_id: int | None = None
            if participant_external_id is not None:
                participant = repositories["participants"].get_by_provider_external_id(
                    provider=participant_provider,
                    external_id=str(participant_external_id),
                )
                participant_id = participant.id if participant is not None else None

            scope_key, _, _ = self._build_memory_scope(
                scope_type=scope_type,
                channel_id=channel.id,
                participant_id=participant_id,
            )
            with self.database.connection:
                return repositories["long_term_memories"].archive(
                    scope_type=scope_type,
                    scope_key=scope_key,
                    memory_key=memory_key,
                )

    def _require_repositories(self) -> dict[str, object]:
        if (
            self.channels is None
            or self.participants is None
            or self.conversations is None
            or self.messages is None
            or self.attachments is None
            or self.long_term_memories is None
            or self.research_tasks is None
            or self.knowledge_graph is None
        ):
            raise RuntimeError("Storage service has not been started")

        return {
            "attachments": self.attachments,
            "channels": self.channels,
            "conversations": self.conversations,
            "long_term_memories": self.long_term_memories,
            "messages": self.messages,
            "participants": self.participants,
            "research_tasks": self.research_tasks,
            "knowledge_graph": self.knowledge_graph,
        }

    @staticmethod
    def _build_memory_scope(
        *,
        scope_type: str,
        channel_id: int,
        participant_id: int | None,
    ) -> tuple[str, int | None, int | None]:
        if scope_type == "channel":
            return f"channel:{channel_id}", channel_id, None

        if scope_type == "participant":
            if participant_id is None:
                raise ValueError("participant scope requires a participant id")
            return f"participant:{participant_id}", None, participant_id

        raise ValueError(f"Unsupported memory scope: {scope_type}")

    @staticmethod
    def _get_channel_provider(event: BaseEvent) -> str:
        if isinstance(event, AgentResponseEvent):
            return "telegram"

        return event.source

    @staticmethod
    def _get_direction(event: BaseEvent) -> str:
        if isinstance(event, AgentResponseEvent):
            return "outbound"

        return "inbound"

    @staticmethod
    def _get_role(event: BaseEvent) -> str:
        if isinstance(event, AgentResponseEvent):
            return "assistant"

        return "user"

    @staticmethod
    def _get_message_type(event: BaseEvent) -> str:
        return getattr(event, "message_type", "message")

    @staticmethod
    def _get_provider_message_id(event: BaseEvent) -> str | None:
        if isinstance(event, (TelegramMessageEvent, TelegramPhotoEvent, TelegramAudioEvent)):
            return str(event.message_id)

        return None

    @staticmethod
    def _get_reply_to_provider_message_id(event: BaseEvent) -> str | None:
        if isinstance(event, AgentResponseEvent):
            return str(event.reply_to_message_id)

        return None

    @staticmethod
    def _get_delivery_status(event: BaseEvent) -> str:
        if isinstance(event, AgentResponseEvent):
            return "pending"

        return "received"

    @staticmethod
    def _get_text_content(event: BaseEvent) -> str | None:
        if isinstance(event, TelegramMessageEvent):
            return event.message

        if isinstance(event, AgentResponseEvent):
            return event.response

        if isinstance(event, TelegramPhotoEvent):
            if event.caption:
                return f"[Photo] {event.caption}"
            return "[Photo]"

        if isinstance(event, TelegramAudioEvent):
            if event.transcript and event.transcript.strip():
                return event.transcript.strip()

            label = "Voice message" if event.media_type == "voice" else "Audio"
            details = event.caption or event.title or event.file_name
            if details:
                return f"[{label}] {details}"
            return f"[{label}]"

        return None

    @staticmethod
    def _build_attachment_payload(
        message_id: int,
        event: BaseEvent,
    ) -> dict[str, object] | None:
        if isinstance(event, TelegramPhotoEvent):
            return {
                "message_id": message_id,
                "media_type": "photo",
                "provider_file_id": event.file_id,
                "provider_file_unique_id": event.file_unique_id,
                "mime_type": None,
                "file_name": None,
                "width": event.width,
                "height": event.height,
                "duration_seconds": None,
                "file_size": event.file_size,
                "performer": None,
                "title": None,
                "caption": event.caption,
                "media_group_id": event.media_group_id,
                "payload_json": event.model_dump_json(),
            }

        if isinstance(event, TelegramAudioEvent):
            return {
                "message_id": message_id,
                "media_type": event.media_type,
                "provider_file_id": event.file_id,
                "provider_file_unique_id": event.file_unique_id,
                "mime_type": event.mime_type,
                "file_name": event.file_name,
                "width": None,
                "height": None,
                "duration_seconds": event.duration_seconds,
                "file_size": event.file_size,
                "performer": event.performer,
                "title": event.title,
                "caption": event.caption,
                "media_group_id": None,
                "payload_json": event.model_dump_json(),
            }

        return None
