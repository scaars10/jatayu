import mimetypes

from comms.base_runner import BaseRunner
from telegram import InputFile
from telegram.ext import ApplicationBuilder, MessageHandler, filters

from comms.nats import NatsClient, build_nats_client
from constants import AGENT_RESPONSE_SUBJECT
from comms.telegram.listener.message_listener import MessageListener
from config.env_config import get_env, get_env_int_list
from models import AgentResponseEvent
from storage import AudioArtifactStore, StorageService


class TelegramRunner(BaseRunner):
    def __init__(
        self,
        nats_client: NatsClient | None = None,
        bot_token: str | None = None,
        allowed_chat_ids: list[int] | None = None,
        audio_artifact_store: AudioArtifactStore | None = None,
        storage_service: StorageService | None = None,
    ) -> None:
        if nats_client is None:
            nats_client = build_nats_client()

        if bot_token is None:
            bot_token = get_env("JATAYU_TELEGRAM_TOKEN", required=True) or ""

        if allowed_chat_ids is None:
            allowed_chat_ids = get_env_int_list("TELEGRAM_LISTENER_CHAT_ID", required=True)

        self.nats_client = nats_client
        self.audio_artifact_store = audio_artifact_store or AudioArtifactStore()
        self.storage_service = storage_service or StorageService()
        self.application = ApplicationBuilder().token(bot_token).build()
        self.listener = MessageListener(allowed_chat_ids, nats_client)
        self.application.add_handler(MessageHandler(filters.ALL, self.listener.on_message))

    async def start(self) -> None:
        await self.storage_service.start()
        await self.nats_client.connect()
        await self.application.initialize()
        await self.nats_client.subscribe_json(AGENT_RESPONSE_SUBJECT, self.handle_agent_response)
        await self.application.start()

        if self.application.updater is None:
            raise RuntimeError("Telegram application is missing an updater")

        await self.application.updater.start_polling()

    async def stop(self) -> None:
        updater = self.application.updater

        if updater is not None and updater.running:
            await updater.stop()

        if self.application.running:
            await self.application.stop()

        await self.application.shutdown()
        await self.nats_client.close()
        await self.storage_service.close()

    async def handle_agent_response(self, payload: dict[str, object]) -> None:
        event = AgentResponseEvent.model_validate(payload)
        try:
            sent_message = await self._send_agent_response(event)
            await self.storage_service.mark_message_delivered(
                event_id=event.event_id,
                provider_message_id=sent_message.message_id,
            )
        except Exception as exc:
            print(f"[TELEGRAM][ERROR][{event.event_id}] {exc}")
            sent_message = await self.application.bot.send_message(
                chat_id=event.channel_id,
                text=self._build_delivery_error_notice(event),
                reply_to_message_id=event.reply_to_message_id,
            )
            await self.storage_service.mark_message_delivered(
                event_id=event.event_id,
                provider_message_id=sent_message.message_id,
            )

    async def _send_agent_response(self, event: AgentResponseEvent):
        if not event.audio_file_path:
            return await self.application.bot.send_message(
                chat_id=event.channel_id,
                text=event.response,
                reply_to_message_id=event.reply_to_message_id,
            )

        try:
            audio_file_path = self.audio_artifact_store.resolve_managed_path(event.audio_file_path)
        except ValueError:
            return await self._send_audio_fallback(
                event,
                "I couldn't find the audio reply, so I'm sending the text version instead.",
            )

        if not audio_file_path.exists() or not audio_file_path.is_file():
            return await self._send_audio_fallback(
                event,
                "The audio reply file was missing, so I'm sending the text version instead.",
            )

        file_name = event.audio_file_name or self._build_audio_file_name(event.audio_mime_type)
        caption = event.response if len(event.response) <= 1024 else None

        is_voice_note = self._is_voice_note_mime_type(event.audio_mime_type, file_name)

        try:
            with audio_file_path.open("rb") as audio_file:
                input_file = InputFile(audio_file, filename=file_name)

                if is_voice_note:
                    return await self.application.bot.send_voice(
                        chat_id=event.channel_id,
                        voice=input_file,
                        caption=caption,
                        reply_to_message_id=event.reply_to_message_id,
                    )

                return await self.application.bot.send_audio(
                    chat_id=event.channel_id,
                    audio=input_file,
                    caption=caption,
                    reply_to_message_id=event.reply_to_message_id,
                )
        except Exception:
            return await self._send_audio_fallback(
                event,
                "I couldn't send the audio reply, so I'm sending the text version instead.",
            )

    async def _send_audio_fallback(
        self,
        event: AgentResponseEvent,
        notice: str,
    ):
        return await self.application.bot.send_message(
            chat_id=event.channel_id,
            text=f"{notice}\n\n{event.response}",
            reply_to_message_id=event.reply_to_message_id,
        )

    @staticmethod
    def _build_delivery_error_notice(event: AgentResponseEvent) -> str:
        if event.audio_file_path:
            return "I couldn't deliver the audio reply. Please try again."

        return "I couldn't deliver the reply. Please try again."

    @staticmethod
    def _build_audio_file_name(mime_type: str | None) -> str:
        if not mime_type:
            return "reply.ogg"

        extension = mimetypes.guess_extension(mime_type, strict=False)
        if not extension:
            return "reply.ogg"

        if extension == ".oga":
            extension = ".ogg"

        return f"reply{extension}"

    @staticmethod
    def _is_voice_note_mime_type(mime_type: str | None, file_name: str | None) -> bool:
        normalized_mime = (mime_type or "").lower().strip()
        normalized_name = (file_name or "").lower().strip()

        if normalized_mime in {"audio/ogg", "audio/opus", "application/ogg"}:
            return True

        return normalized_name.endswith(".ogg") or normalized_name.endswith(".opus")
