from .base_event import BaseEvent


class TelegramMessageEvent(BaseEvent):
    message_type: str = "message"
    message: str
    channel_id: int
    sender_id: int
    message_id: int
    audio_bytes: bytes | None = None
    audio_mime_type: str | None = None
