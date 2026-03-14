from abc import ABC, abstractmethod

from telegram import Message, Update
from telegram.ext import ContextTypes


class MessageHandlerBase(ABC):
    @abstractmethod
    def can_handle(self, message: Message) -> bool:
        """Return True when this handler can process the message."""

    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process the incoming update."""
