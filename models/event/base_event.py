from datetime import datetime, timezone

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field


class BaseEvent(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str
    source: str
    occurred_at: AwareDatetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
