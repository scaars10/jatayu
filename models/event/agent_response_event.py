from .base_event import BaseEvent


class AgentResponseEvent(BaseEvent):
    message_type: str = "agent_response"
    request_event_id: str
    channel_id: int
    sender_id: int
    reply_to_message_id: int
    response: str
    audio_file_path: str | None = None
    audio_mime_type: str | None = None
    audio_file_name: str | None = None
