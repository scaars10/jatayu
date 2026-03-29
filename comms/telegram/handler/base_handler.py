from abc import ABC, abstractmethod

from datetime import datetime, timezone
from typing import TypeVar

from telegram import Message, Update
from telegram.ext import ContextTypes

from comms.nats import NatsClient
from constants import TELEGRAM_EVENT_SUBJECT
from models import BaseEvent

T = TypeVar("T")


class MessageHandlerBase(ABC):
    def __init__(self, nats_client: NatsClient) -> None:
        self.nats_client = nats_client

    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        """Return True when this handler can process the message."""

    @abstractmethod
    async def _handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> T:
        """Process the incoming update."""

    @abstractmethod
    def build_event(self, update: Update) -> BaseEvent | None:
        """Build the outgoing Telegram event for this handler."""

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> T | None:
        result = await self._handle(update, context)
        await self.post_handle(result, update, context)
        return result

    async def post_handle(self, result: T | None, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Perform any post-processing after the handler has completed."""
        telegram_event = self.build_event(update)

        if telegram_event is None:
            return

        await self.nats_client.publish_model(TELEGRAM_EVENT_SUBJECT, telegram_event)

    @staticmethod
    def build_event_id(occurred_at: datetime, chat_id: int, message_id: int) -> str:
        return f"telegram|{occurred_at.isoformat()}|{chat_id}|{message_id}"

    @staticmethod
    def get_occurred_at() -> datetime:
        return datetime.now(timezone.utc)

    @staticmethod
    def get_sender_id(update: Update) -> int | None:
        if update.effective_user is not None:
            return update.effective_user.id

        message = update.effective_message
        if message is not None and message.sender_chat is not None:
            return message.sender_chat.id

        return None
