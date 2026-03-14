from telegram import Message, Update
from telegram.ext import ContextTypes

from comms.telegram.handler.base_handler import MessageHandlerBase


class AudioHandler(MessageHandlerBase):

    def can_handle(self, message: Message) -> bool:
        return bool(message.voice or message.audio)

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

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
