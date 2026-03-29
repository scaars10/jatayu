from telegram import Message

from comms.nats import NatsClient
from comms.telegram.handler.audio_handler import AudioHandler
from comms.telegram.handler.base_handler import MessageHandlerBase
from comms.telegram.handler.photo_handler import PhotoHandler
from comms.telegram.handler.text_handler import TextHandler


class HandlerResolver:

    def __init__(self, nats_client: NatsClient):

        self.handlers: list[MessageHandlerBase] = [
            TextHandler(nats_client),
            PhotoHandler(nats_client),
            AudioHandler(nats_client),
        ]

    def resolve(self, message: Message) -> MessageHandlerBase | None:

        for handler in self.handlers:
            if handler.can_handle(message):
                return handler

        return None
