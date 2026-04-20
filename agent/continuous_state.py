import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

STATE_FILE = Path("continuous_research_output/.state.json")


class ContinuousTask(BaseModel):
    task_id: str
    topic: str
    instructions: str
    status: str  # "running", "paused", "stopped"
    workspace_version: str = ""
    source_event_id: str = ""
    source_message_id: int | None = None
    source_channel_id: int | None = None
    event_dict: dict[str, Any]  # To reconstruct the TelegramMessageEvent for replies
    last_summary: str = ""
    plan: str = ""  # High-level research plan
    no_new_findings_count: int = 0
    unverified_signal_count: int = 0
    suppressed_notification_count: int = 0
    cycle_count: int = 0
    failure_count: int = 0
    last_error: str = ""
    is_cycle_running: bool = False
    last_cycle_started_at: datetime | None = None
    last_cycle_completed_at: datetime | None = None
    last_new_info_at: datetime | None = None


class ContinuousResearchState:
    def __init__(self):
        self.tasks: dict[str, ContinuousTask] = {}
        self.load()

    def load(self):
        if STATE_FILE.exists():
            try:
                with open(STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for k, v in data.items():
                        task = ContinuousTask(**v)
                        # This flag is only meaningful within the live process.
                        task.is_cycle_running = False
                        self.tasks[k] = task
            except Exception as e:
                logger.error(f"Failed to load continuous research state: {e}")

    def save(self):
        try:
            STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            payload = {k: v.model_dump(mode="json") for k, v in self.tasks.items()}
            temp_path = STATE_FILE.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            temp_path.replace(STATE_FILE)
        except Exception as e:
            logger.error(f"Failed to save continuous research state: {e}")

    def add_task(self, task: ContinuousTask):
        self.tasks[task.task_id] = task
        self.save()

    def get_task(self, task_id: str) -> ContinuousTask | None:
        return self.tasks.get(task_id)

    def update_task(self, task_id: str, **kwargs):
        if task_id in self.tasks:
            for k, v in kwargs.items():
                setattr(self.tasks[task_id], k, v)
            self.save()

    def get_all_tasks(self) -> list[ContinuousTask]:
        return list(self.tasks.values())

global_continuous_state = ContinuousResearchState()
