from __future__ import annotations

from uuid import uuid4

from google.genai import errors as genai_errors
from pydantic_ai import exceptions as pydantic_ai_errors

from agent.audio_agent import AudioAgent
from agent.base_agent import AgentReply
from comms.base_runner import BaseRunner
from comms.nats import NatsClient, build_nats_client
from constants import AGENT_RESPONSE_SUBJECT, TELEGRAM_EVENT_SUBJECT
from models import (
    AgentResponseEvent,
    TelegramAudioEvent,
    TelegramMessageEvent,
    TelegramPhotoEvent,
)
from storage import AudioArtifactStore, StorageService

from agent.chat_agent import ChatAgent
from agent.image_agent import ImageAgent


AGENT_PARTICIPANT_ID = 0


class AgentReceiverRunner(BaseRunner):
    def __init__(
        self,
        nats_client: NatsClient | None = None,
        chat_agent: ChatAgent | None = None,
        image_agent: ImageAgent | None = None,
        audio_agent: AudioAgent | None = None,
        audio_artifact_store: AudioArtifactStore | None = None,
        storage_service: StorageService | None = None,
    ) -> None:
        self.nats_client = nats_client or build_nats_client()
        self.storage_service = storage_service or StorageService()
        self.chat_agent = chat_agent or ChatAgent(storage_service=self.storage_service)
        self.image_agent = image_agent or ImageAgent()
        self.audio_agent = audio_agent or AudioAgent(chat_agent=self.chat_agent)
        self.audio_artifact_store = audio_artifact_store or AudioArtifactStore()

    async def start(self) -> None:
        await self.storage_service.start()
        await self.nats_client.connect()
        await self.chat_agent.start()
        await self.image_agent.start()
        await self.audio_agent.start()
        await self.nats_client.subscribe_json(TELEGRAM_EVENT_SUBJECT, self.handle_payload)

    async def stop(self) -> None:
        await self.audio_agent.stop()
        await self.image_agent.stop()
        await self.chat_agent.stop()
        await self.nats_client.close()
        await self.storage_service.close()

    async def handle_payload(self, payload: dict[str, object]) -> None:
        event = self._build_event(payload)
        if event is None:
            return

        try:
            await self.storage_service.record_event(event)

            if isinstance(event, TelegramMessageEvent):
                response = await self.chat_agent.respond(event)
            elif isinstance(event, TelegramPhotoEvent):
                response = await self.image_agent.respond(event)
            elif isinstance(event, TelegramAudioEvent):
                response = await self.audio_agent.respond(event)
            else:
                return
        except Exception as exc:
            print(f"[AGENT][ERROR][{event.event_id}] {exc}")
            await self._publish_failure_response(event, exc)
            return

        reply = self._normalize_reply(response)
        if reply is None:
            return

        if reply.requires_audio and not reply.audio_bytes:
            try:
                audio_bytes, audio_mime_type = await self.audio_agent.synthesize(reply.response)
                if audio_bytes:
                    reply = AgentReply(
                        response=reply.response,
                        requires_audio=True,
                        audio_bytes=audio_bytes,
                        audio_mime_type=audio_mime_type,
                        audio_file_name=self.audio_agent._build_audio_file_name(audio_mime_type)
                    )
            except Exception as exc:
                print(f"[AGENT][TTS][ERROR] {exc}")

        response_event = self._build_response_event(event, reply)
        await self.storage_service.record_event(response_event, channel_source=event.source)
        await self.nats_client.publish_model(AGENT_RESPONSE_SUBJECT, response_event)

    @staticmethod
    def _build_event(payload: dict[str, object]):
        message_type = payload.get("message_type")

        if message_type == "message":
            return TelegramMessageEvent.model_validate(payload)

        if message_type == "photo":
            return TelegramPhotoEvent.model_validate(payload)

        if message_type == "audio":
            return TelegramAudioEvent.model_validate(payload)

        return None

    @staticmethod
    def _normalize_reply(
        response: str | AgentReply | None,
    ) -> AgentReply | None:
        if response is None:
            return None

        if isinstance(response, AgentReply):
            if not response.response:
                return None
            return response

        if not response:
            return None

        return AgentReply(response=response)

    def _build_response_event(
        self,
        event: TelegramMessageEvent | TelegramPhotoEvent | TelegramAudioEvent,
        reply: AgentReply,
    ) -> AgentResponseEvent:
        audio_file_path: str | None = None
        if reply.audio_bytes:
            audio_file_path = str(
                self.audio_artifact_store.create_audio_file(
                    reply.audio_bytes,
                    mime_type=reply.audio_mime_type,
                    file_name=reply.audio_file_name,
                    prefix="agent-response",
                )
            )

        return AgentResponseEvent(
            event_id=f"agent|{uuid4()}",
            source="agent",
            request_event_id=event.event_id,
            channel_id=event.channel_id,
            sender_id=AGENT_PARTICIPANT_ID,
            reply_to_message_id=event.message_id,
            response=reply.response,
            audio_file_path=audio_file_path,
            audio_mime_type=reply.audio_mime_type,
            audio_file_name=reply.audio_file_name,
        )

    async def _publish_failure_response(
        self,
        event: TelegramMessageEvent | TelegramPhotoEvent | TelegramAudioEvent,
        exc: Exception,
    ) -> None:
        try:
            await self.storage_service.record_event(event)
            response_event = AgentResponseEvent(
                event_id=f"agent|{uuid4()}",
                source="agent",
                request_event_id=event.event_id,
                channel_id=event.channel_id,
                sender_id=AGENT_PARTICIPANT_ID,
                reply_to_message_id=event.message_id,
                response=self._build_failure_message(event, exc),
            )
            await self.storage_service.record_event(response_event, channel_source=event.source)
            await self.nats_client.publish_model(AGENT_RESPONSE_SUBJECT, response_event)
        except Exception as fallback_exc:
            print(f"[AGENT][ERROR][FALLBACK][{event.event_id}] {fallback_exc}")

    @staticmethod
    def _build_failure_message(
        event: TelegramMessageEvent | TelegramPhotoEvent | TelegramAudioEvent,
        exc: Exception,
    ) -> str:
        if isinstance(exc, genai_errors.ServerError):
            detail = "a temporary model issue"
        elif isinstance(exc, pydantic_ai_errors.ModelHTTPError):
            if exc.status_code in {503, 429}:
                detail = "the model being temporarily overloaded or unavailable"
            else:
                detail = "a temporary model connection issue"
        elif isinstance(exc, pydantic_ai_errors.UnexpectedModelBehavior):
            detail = "the model acting up temporarily"
        else:
            detail = "an internal processing issue"

        if isinstance(event, TelegramAudioEvent):
            return f"I couldn't process that audio because of {detail}. Please try again."

        return f"I couldn't finish that reply because of {detail}. Please try again."
