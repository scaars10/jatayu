from telegram import Message, Update
from telegram.ext import ContextTypes

from comms.telegram.handler.base_handler import MessageHandlerBase


class TextHandler(MessageHandlerBase):
    def can_handle(self, message: Message) -> bool:
        return message.text is not None

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat = update.effective_chat
        message = update.effective_message

        if not chat or not message or message.text is None:
            return

        chat_id = chat.id
        text = message.text

        print(f"[TEXT][{chat.title or chat_id}] {text}")

        if "hello" in text.lower():
            await context.bot.send_message(
                chat_id=chat.id,
                text="Hello from Jatayu",
            )
