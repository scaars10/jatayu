import unittest
from unittest.mock import AsyncMock, patch

from comms.nats.client import DEFAULT_NATS_URL, NatsClient, build_nats_client
from constants import TELEGRAM_EVENT_SUBJECT
from models import TelegramMessageEvent


class NatsClientTests(unittest.IsolatedAsyncioTestCase):
    async def test_connect_uses_configured_url(self) -> None:
        nats_client = NatsClient("nats://example:4222")
        connection = AsyncMock()

        with patch("comms.nats.client.nats.connect", AsyncMock(return_value=connection)) as mock_connect:
            await nats_client.connect()

        mock_connect.assert_awaited_once_with("nats://example:4222")
        self.assertIs(nats_client.nc, connection)

    async def test_publish_model_serializes_event(self) -> None:
        client = AsyncMock()
        nats_client = NatsClient()
        nats_client.nc = client

        event = TelegramMessageEvent(
            event_id="evt-1",
            source="telegram",
            message="hello",
            channel_id=1,
            sender_id=2,
            message_id=3,
        )

        await nats_client.publish_model(TELEGRAM_EVENT_SUBJECT, event)

        client.publish.assert_awaited_once_with(
            TELEGRAM_EVENT_SUBJECT,
            event.model_dump_json().encode("utf-8"),
        )

    async def test_subscribe_json_decodes_payload(self) -> None:
        client = AsyncMock()
        nats_client = NatsClient()
        nats_client.nc = client

        event = TelegramMessageEvent(
            event_id="evt-2",
            source="telegram",
            message="hello",
            channel_id=10,
            sender_id=20,
            message_id=30,
        )
        callback = AsyncMock()

        await nats_client.subscribe_json(TELEGRAM_EVENT_SUBJECT, callback)

        subscribe_callback = client.subscribe.await_args.kwargs["cb"]

        message = type(
            "FakeMsg",
            (),
            {
                "data": event.model_dump_json().encode("utf-8"),
            },
        )()

        await subscribe_callback(message)

        callback.assert_awaited_once()
        self.assertEqual(callback.await_args.args[0], event.model_dump(mode="json"))

    async def test_close_drains_connection(self) -> None:
        client = AsyncMock()
        nats_client = NatsClient()
        nats_client.nc = client

        await nats_client.close()

        client.drain.assert_awaited_once()


class NatsClientFactoryTests(unittest.TestCase):
    def test_build_nats_client_uses_env_when_url_is_not_provided(self) -> None:
        with patch("comms.nats.client.get_env", return_value="nats://env:4222") as mock_get_env:
            client = build_nats_client()

        mock_get_env.assert_called_once_with("JATAYU_NATS_URL", DEFAULT_NATS_URL)
        self.assertEqual(client.url, "nats://env:4222")

    def test_build_nats_client_uses_explicit_url_when_provided(self) -> None:
        with patch("comms.nats.client.get_env") as mock_get_env:
            client = build_nats_client("nats://direct:4222")

        mock_get_env.assert_not_called()
        self.assertEqual(client.url, "nats://direct:4222")


if __name__ == "__main__":
    unittest.main()
