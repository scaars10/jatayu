from .base_event import BaseEvent


class TelegramPhotoEvent(BaseEvent):
    message_type: str = "photo"
    channel_id: int
    sender_id: int
    message_id: int
    file_id: str
    file_unique_id: str
    width: int
    height: int
    caption: str | None = None
    file_size: int | None = None
    media_group_id: str | None = None
