import json

import nats
from pydantic import BaseModel

from config.env_config import get_env

DEFAULT_NATS_URL = "nats://localhost:4222"


class NatsClient:
    def __init__(self, url: str = DEFAULT_NATS_URL) -> None:
        self.url = url
        self.nc = None

    async def connect(self) -> None:
        self.nc = await nats.connect(self.url)

    async def close(self) -> None:
        if self.nc is not None:
            await self.nc.drain()

    async def publish_json(self, subject: str, payload: dict) -> None:
        assert self.nc is not None
        await self.nc.publish(subject, json.dumps(payload).encode("utf-8"))

    async def publish_model(self, subject: str, model: BaseModel) -> None:
        assert self.nc is not None
        await self.nc.publish(subject, model.model_dump_json().encode("utf-8"))

    async def subscribe_json(self, subject: str, handler, queue: str | None = None) -> None:
        assert self.nc is not None

        async def _cb(msg) -> None:
            payload = json.loads(msg.data.decode("utf-8"))
            await handler(payload)

        await self.nc.subscribe(subject, queue=queue, cb=_cb)


def build_nats_client(url: str | None = None) -> NatsClient:
    if url is None:
        url = get_env("JATAYU_NATS_URL", DEFAULT_NATS_URL) or DEFAULT_NATS_URL

    return NatsClient(url)
