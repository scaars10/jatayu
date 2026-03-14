from telegram import Message, Update
from telegram.ext import ContextTypes

from comms.telegram.handler.base_handler import MessageHandlerBase


class PhotoHandler(MessageHandlerBase):

    def can_handle(self, message: Message) -> bool:
        return bool(message.photo)

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

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
