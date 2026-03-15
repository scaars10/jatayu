from telegram import Message, Update
from telegram.ext import ContextTypes

from models import TelegramPhotoEvent

from comms.telegram.handler.base_handler import MessageHandlerBase


class PhotoHandler(MessageHandlerBase):

    def can_handle(self, message: Message) -> bool:
        return bool(message.photo)

    async def _handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:

        chat = update.effective_chat
        message = update.effective_message

        if not chat or not message or not message.photo:
            return

        photo = message.photo[-1]
        caption = message.caption or ""

        print(
            f"[PHOTO][{chat.title or chat.id}] "
            f"id={photo.file_id} caption={caption}"
        )

        await context.bot.send_message(
            chat_id=chat.id,
            text="Photo received"
        )

    def build_event(self, update: Update) -> TelegramPhotoEvent | None:
        chat = update.effective_chat
        message = update.effective_message
        sender_id = self.get_sender_id(update)

        if not chat or not message or not message.photo or sender_id is None:
            return None

        photo = message.photo[-1]
        occurred_at = self.get_occurred_at()
        return TelegramPhotoEvent(
            event_id=self.build_event_id(occurred_at, chat.id, message.message_id),
            source="telegram",
            occurred_at=occurred_at,
            channel_id=chat.id,
            sender_id=sender_id,
            message_id=message.message_id,
            file_id=photo.file_id,
            file_unique_id=photo.file_unique_id,
            width=photo.width,
            height=photo.height,
            caption=message.caption,
            file_size=photo.file_size,
            media_group_id=message.media_group_id,
        )
