from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union

from models import TelegramAudioEvent, TelegramMessageEvent, TelegramPhotoEvent

EventTypes = Union[TelegramMessageEvent, TelegramPhotoEvent, TelegramAudioEvent]

class MessageQueue:
    def __init__(self, expiry_minutes: int = 30):
        self.queues: Dict[int, List[EventTypes]] = {}
        self.last_added: Dict[int, datetime] = {}
        self.expiry_minutes = expiry_minutes
        self.lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def add(self, event: EventTypes):
        async with self.lock:
            channel_id = event.channel_id
            if channel_id not in self.queues:
                self.queues[channel_id] = []
            
            # Deduplication/Clubbing logic
            # If it's a message, check if it's similar to the last one
            if isinstance(event, TelegramMessageEvent):
                is_duplicate = False
                for existing in self.queues[channel_id]:
                    if isinstance(existing, TelegramMessageEvent):
                        if existing.message.strip().lower() == event.message.strip().lower():
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    self._logger.info(f"[QUEUE] Skipping duplicate message for channel {channel_id}")
                    return

            self.queues[channel_id].append(event)
            self.last_added[channel_id] = datetime.now()
            self._logger.info(f"[QUEUE] Added event {event.event_id} for channel {channel_id}. Queue size: {len(self.queues[channel_id])}")

    async def get_all_channels(self) -> List[int]:
        async with self.lock:
            return list(self.queues.keys())

    async def pop_latest_for_channel(self, channel_id: int) -> Optional[EventTypes]:
        async with self.lock:
            if channel_id not in self.queues or not self.queues[channel_id]:
                return None
            
            events = self.queues.pop(channel_id)
            self.last_added.pop(channel_id, None)
            
            # Return the latest event to respond to
            return events[-1]

    async def cleanup_expired(self):
        async with self.lock:
            now = datetime.now()
            expired_channels = [
                channel_id for channel_id, last_time in self.last_added.items()
                if now - last_time > timedelta(minutes=self.expiry_minutes)
            ]
            for channel_id in expired_channels:
                self._logger.info(f"[QUEUE] Expiring queue for channel {channel_id}")
                self.queues.pop(channel_id, None)
                self.last_added.pop(channel_id, None)
