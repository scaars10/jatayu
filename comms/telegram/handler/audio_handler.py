from telegram import Message, Update
from telegram.ext import ContextTypes

from models import TelegramAudioEvent

from comms.telegram.handler.base_handler import MessageHandlerBase


class AudioHandler(MessageHandlerBase):

    def can_handle(self, message: Message) -> bool:
        return bool(message.voice or message.audio)

    async def _handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        chat = update.effective_chat
        message = update.effective_message

        if not chat or not message:
            return

        if message.voice:
            print(f"[VOICE][{chat.title or chat.id}] duration={message.voice.duration}")

        if message.audio:
            print(f"[AUDIO][{chat.title or chat.id}] title={message.audio.title}")

        await context.bot.send_message(
            chat_id=chat.id,
            text="Audio received"
        )

    def build_event(self, update: Update) -> TelegramAudioEvent | None:
        chat = update.effective_chat
        message = update.effective_message
        sender_id = self.get_sender_id(update)

        if not chat or not message or sender_id is None:
            return None

        occurred_at = self.get_occurred_at()
        event_id = self.build_event_id(occurred_at, chat.id, message.message_id)

        if message.audio:
            return TelegramAudioEvent(
                event_id=event_id,
                source="telegram",
                occurred_at=occurred_at,
                channel_id=chat.id,
                sender_id=sender_id,
                message_id=message.message_id,
                media_type="audio",
                file_id=message.audio.file_id,
                file_unique_id=message.audio.file_unique_id,
                duration_seconds=message.audio.duration,
                caption=message.caption,
                mime_type=message.audio.mime_type,
                file_name=message.audio.file_name,
                performer=message.audio.performer,
                title=message.audio.title,
                file_size=message.audio.file_size,
            )

        if message.voice:
            return TelegramAudioEvent(
                event_id=event_id,
                source="telegram",
                occurred_at=occurred_at,
                channel_id=chat.id,
                sender_id=sender_id,
                message_id=message.message_id,
                media_type="voice",
                file_id=message.voice.file_id,
                file_unique_id=message.voice.file_unique_id,
                duration_seconds=message.voice.duration,
                caption=message.caption,
                mime_type=message.voice.mime_type,
                file_size=message.voice.file_size,
            )

        return None
