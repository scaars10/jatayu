from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentReply:
    response: str
    requires_audio: bool = False
    audio_bytes: bytes | None = None
    audio_mime_type: str | None = None
    audio_file_name: str | None = None


class BaseAgent(ABC):
    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    @abstractmethod
    async def respond(self, event: Any):
        pass
