from typing import Literal

from .base_event import BaseEvent


class TelegramAudioEvent(BaseEvent):
    message_type: str = "audio"
    channel_id: int
    sender_id: int
    message_id: int
    media_type: Literal["audio", "voice"]
    file_id: str
    file_unique_id: str
    duration_seconds: int
    caption: str | None = None
    mime_type: str | None = None
    file_name: str | None = None
    performer: str | None = None
    title: str | None = None
    file_size: int | None = None
