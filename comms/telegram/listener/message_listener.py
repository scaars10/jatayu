from telegram import Update
from telegram.ext import ContextTypes

from comms.nats import NatsClient
from comms.telegram.handler.handler_resolver import HandlerResolver


class MessageListener:

    def __init__(self, allowed_chat_ids: list[int], nats_client: NatsClient):

        self.allowed_chat_ids = set(allowed_chat_ids)
        self.resolver = HandlerResolver(nats_client)

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):

        if not (update.message or update.channel_post):
            return

        chat = update.effective_chat
        message = update.effective_message

        if not chat or not message:
            return

        if chat.id not in self.allowed_chat_ids:
            return

        handler = self.resolver.resolve(message)

        if handler is None:
            print(f"[UNSUPPORTED][{chat.title or chat.id}]")
            return

        await handler.handle(update, context)
