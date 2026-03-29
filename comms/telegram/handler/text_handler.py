from telegram import Message, Update
from telegram.ext import ContextTypes

from models import TelegramMessageEvent

from comms.telegram.handler.base_handler import MessageHandlerBase


class TextHandler(MessageHandlerBase):
    def can_handle(self, message: Message) -> bool:
        return message.text is not None

    async def _handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message

        if not chat or not message or message.text is None:
            return

        chat_id = chat.id
        text = message.text

        print(f"[TEXT][{chat.title or chat_id}] {text}")

    def build_event(self, update: Update) -> TelegramMessageEvent | None:
        chat = update.effective_chat
        message = update.effective_message
        sender_id = self.get_sender_id(update)

        if not chat or not message or message.text is None or sender_id is None:
            return None

        occurred_at = self.get_occurred_at()
        return TelegramMessageEvent(
            event_id=self.build_event_id(occurred_at, chat.id, message.message_id),
            source="telegram",
            occurred_at=occurred_at,
            message=message.text,
            channel_id=chat.id,
            sender_id=sender_id,
            message_id=message.message_id,
        )
