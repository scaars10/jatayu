import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from telegram.constants import ParseMode

from models import AgentResponseEvent
from comms.telegram.runner import TelegramRunner
from storage import AudioArtifactStore


class TelegramRunnerTests(unittest.TestCase):
    def test_runner_loads_its_own_config_and_nats_client_when_not_provided(self) -> None:
        builder = MagicMock()
        app = MagicMock()
        builder.token.return_value = builder
        builder.build.return_value = app
        nats_client = object()

        with patch("comms.telegram.runner.get_env", return_value="telegram-token") as mock_get_env:
            with patch("comms.telegram.runner.get_env_int_list", return_value=[1, 2]) as mock_get_env_int_list:
                with patch("comms.telegram.runner.ApplicationBuilder", return_value=builder) as mock_builder:
                    with patch("comms.telegram.runner.MessageListener") as mock_listener:
                        with patch("comms.telegram.runner.build_nats_client", return_value=nats_client) as mock_build_nats_client:
                            runner = TelegramRunner()

        mock_get_env.assert_called_once_with("JATAYU_TELEGRAM_TOKEN", required=True)
        mock_get_env_int_list.assert_called_once_with("TELEGRAM_LISTENER_CHAT_ID", required=True)
        mock_build_nats_client.assert_called_once_with()
        mock_builder.assert_called_once_with()
        builder.token.assert_called_once_with("telegram-token")
        mock_listener.assert_called_once_with([1, 2], nats_client)
        self.assertIs(runner.nats_client, nats_client)
        self.assertIs(runner.application, app)

    def test_format_for_telegram_sanitizes_artifacts_and_flattens_tables(self) -> None:
        text = """
## Watchlist

| Project | Livability Factor | Appreciation Potential |
| :--- | :--- | :--- |
| *DNR Arista* | *High* | *Extreme* |
| *Purva Weaves* | *High* | *High* |

Would you like me to keep tracking?"}call:default_api:final_result{requires_audio:false,response:
"""

        formatted = TelegramRunner._format_for_telegram(text)

        self.assertIn("*Watchlist*", formatted)
        self.assertIn("- *DNR Arista*: Livability Factor: *High*; Appreciation Potential: *Extreme*", formatted)
        self.assertIn("- *Purva Weaves*: Livability Factor: *High*; Appreciation Potential: *High*", formatted)
        self.assertNotIn("| Project |", formatted)
        self.assertNotIn("call:default_api:final_result", formatted)


class TelegramRunnerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_handle_agent_response_sends_reply_to_original_message(self) -> None:
        nats_client = AsyncMock()
        storage_service = AsyncMock()
        application = MagicMock()
        application.initialize = AsyncMock()
        application.start = AsyncMock()
        application.stop = AsyncMock()
        application.shutdown = AsyncMock()
        application.bot = AsyncMock()
        application.bot.send_message.return_value = MagicMock(message_id=444)

        with patch("comms.telegram.runner.ApplicationBuilder") as mock_builder:
            builder = MagicMock()
            builder.token.return_value = builder
            builder.build.return_value = application
            mock_builder.return_value = builder
            with patch("comms.telegram.runner.MessageListener"):
                runner = TelegramRunner(
                    nats_client=nats_client,
                    bot_token="telegram-token",
                    allowed_chat_ids=[1],
                    storage_service=storage_service,
                )

        event = AgentResponseEvent(
            event_id="evt-response",
            source="agent",
            request_event_id="evt-request",
            channel_id=100,
            sender_id=200,
            reply_to_message_id=300,
            response="hello back",
        )

        await runner.handle_agent_response(event.model_dump(mode="json"))

        application.bot.send_message.assert_awaited_once_with(
            chat_id=100,
            text="hello back",
            reply_to_message_id=300,
            parse_mode=ParseMode.MARKDOWN,
        )
        storage_service.mark_message_delivered.assert_awaited_once_with(
            event_id="evt-response",
            provider_message_id=444,
        )

    async def test_handle_agent_response_sends_audio_when_present(self) -> None:
        nats_client = AsyncMock()
        storage_service = AsyncMock()
        application = MagicMock()
        application.initialize = AsyncMock()
        application.start = AsyncMock()
        application.stop = AsyncMock()
        application.shutdown = AsyncMock()
        application.bot = AsyncMock()
        application.bot.send_audio.return_value = MagicMock(message_id=555)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_store = AudioArtifactStore(Path(temp_dir))
            audio_path = audio_store.create_audio_file(
                b"audio-reply",
                mime_type="audio/wav",
                file_name="reply.wav",
                prefix="telegram-test",
            )

            with patch("comms.telegram.runner.ApplicationBuilder") as mock_builder:
                builder = MagicMock()
                builder.token.return_value = builder
                builder.build.return_value = application
                mock_builder.return_value = builder
                with patch("comms.telegram.runner.MessageListener"):
                    runner = TelegramRunner(
                        nats_client=nats_client,
                        bot_token="telegram-token",
                        allowed_chat_ids=[1],
                        audio_artifact_store=audio_store,
                        storage_service=storage_service,
                    )

            event = AgentResponseEvent(
                event_id="evt-audio-response",
                source="agent",
                request_event_id="evt-request",
                channel_id=100,
                sender_id=200,
                reply_to_message_id=300,
                response="hello back",
                audio_file_path=str(audio_path),
                audio_mime_type="audio/wav",
                audio_file_name="reply.wav",
            )

            await runner.handle_agent_response(event.model_dump(mode="json"))

            application.bot.send_audio.assert_awaited_once()
            self.assertFalse(application.bot.send_message.await_count)
            storage_service.mark_message_delivered.assert_awaited_once_with(
                event_id="evt-audio-response",
                provider_message_id=555,
            )
            self.assertTrue(audio_path.exists())

    async def test_audio_send_failure_falls_back_to_text_notice(self) -> None:
        nats_client = AsyncMock()
        storage_service = AsyncMock()
        application = MagicMock()
        application.initialize = AsyncMock()
        application.start = AsyncMock()
        application.stop = AsyncMock()
        application.shutdown = AsyncMock()
        application.bot = AsyncMock()
        application.bot.send_audio.side_effect = RuntimeError("telegram send failed")
        application.bot.send_message.return_value = MagicMock(message_id=556)

        with tempfile.TemporaryDirectory() as temp_dir:
            audio_store = AudioArtifactStore(Path(temp_dir))
            audio_path = audio_store.create_audio_file(
                b"audio-reply",
                mime_type="audio/wav",
                file_name="reply.wav",
                prefix="telegram-test",
            )

            with patch("comms.telegram.runner.ApplicationBuilder") as mock_builder:
                builder = MagicMock()
                builder.token.return_value = builder
                builder.build.return_value = application
                mock_builder.return_value = builder
                with patch("comms.telegram.runner.MessageListener"):
                    runner = TelegramRunner(
                        nats_client=nats_client,
                        bot_token="telegram-token",
                        allowed_chat_ids=[1],
                        audio_artifact_store=audio_store,
                        storage_service=storage_service,
                    )

            event = AgentResponseEvent(
                event_id="evt-audio-fallback",
                source="agent",
                request_event_id="evt-request",
                channel_id=100,
                sender_id=200,
                reply_to_message_id=300,
                response="hello back",
                audio_file_path=str(audio_path),
                audio_mime_type="audio/wav",
                audio_file_name="reply.wav",
            )

            await runner.handle_agent_response(event.model_dump(mode="json"))

            application.bot.send_message.assert_awaited_once_with(
                chat_id=100,
                text="I couldn't send the audio reply, so I'm sending the text version instead.\n\nhello back",
                reply_to_message_id=300,
                parse_mode=ParseMode.MARKDOWN,
            )
            storage_service.mark_message_delivered.assert_awaited_once_with(
                event_id="evt-audio-fallback",
                provider_message_id=556,
            )

    async def test_start_subscribes_to_agent_responses(self) -> None:
        nats_client = AsyncMock()
        storage_service = AsyncMock()
        application = MagicMock()
        application.initialize = AsyncMock()
        application.start = AsyncMock()
        application.stop = AsyncMock()
        application.shutdown = AsyncMock()
        updater = AsyncMock()
        updater.running = True
        application.updater = updater

        with patch("comms.telegram.runner.ApplicationBuilder") as mock_builder:
            builder = MagicMock()
            builder.token.return_value = builder
            builder.build.return_value = application
            mock_builder.return_value = builder
            with patch("comms.telegram.runner.MessageListener"):
                runner = TelegramRunner(
                    nats_client=nats_client,
                    bot_token="telegram-token",
                    allowed_chat_ids=[1],
                    storage_service=storage_service,
                )

        await runner.start()

        storage_service.start.assert_awaited_once()
        nats_client.connect.assert_awaited_once()
        nats_client.subscribe_json.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
