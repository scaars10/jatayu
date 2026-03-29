from .base_event import BaseEvent


class TelegramMessageEvent(BaseEvent):
    message_type: str = "message"
    message: str
    channel_id: int
    sender_id: int
    message_id: int
