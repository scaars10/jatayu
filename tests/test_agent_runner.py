import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock

from agent import AgentReply
from agent.runner import AgentReceiverRunner
from constants import AGENT_RESPONSE_SUBJECT
from models import (
    AgentResponseEvent,
    TelegramAudioEvent,
    TelegramMessageEvent,
    TelegramPhotoEvent,
)
from storage import AudioArtifactStore


class AgentReceiverRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_text_event_routes_to_chat_agent_and_publishes_response(self) -> None:
        nats_client = AsyncMock()
        chat_agent = AsyncMock()
        image_agent = AsyncMock()
        audio_agent = AsyncMock()
        storage_service = AsyncMock()
        chat_agent.respond.return_value = AgentReply(response="hello back")

        runner = AgentReceiverRunner(
            nats_client=nats_client,
            chat_agent=chat_agent,
            image_agent=image_agent,
            audio_agent=audio_agent,
            storage_service=storage_service,
        )
        payload = TelegramMessageEvent(
            event_id="evt-1",
            source="telegram",
            message="hello",
            channel_id=100,
            sender_id=200,
            message_id=300,
        ).model_dump(mode="json")

        await runner.handle_payload(payload)

        chat_agent.respond.assert_awaited_once()
        image_agent.respond.assert_not_called()
        audio_agent.respond.assert_not_called()
        self.assertEqual(storage_service.record_event.await_count, 2)
        nats_client.publish_model.assert_awaited_once()
        subject, response_event = nats_client.publish_model.await_args.args
        self.assertEqual(subject, AGENT_RESPONSE_SUBJECT)
        self.assertIsInstance(response_event, AgentResponseEvent)
        self.assertEqual(response_event.request_event_id, "evt-1")
        self.assertEqual(response_event.channel_id, 100)
        self.assertEqual(response_event.reply_to_message_id, 300)
        self.assertEqual(response_event.response, "hello back")
        self.assertEqual(response_event.sender_id, 0)

    async def test_photo_event_routes_to_image_agent_and_publishes_response(self) -> None:
        nats_client = AsyncMock()
        chat_agent = AsyncMock()
        image_agent = AsyncMock()
        audio_agent = AsyncMock()
        storage_service = AsyncMock()
        image_agent.respond.return_value = "this is a cat"

        runner = AgentReceiverRunner(
            nats_client=nats_client,
            chat_agent=chat_agent,
            image_agent=image_agent,
            audio_agent=audio_agent,
            storage_service=storage_service,
        )
        payload = TelegramPhotoEvent(
            event_id="evt-2",
            source="telegram",
            channel_id=100,
            sender_id=200,
            message_id=300,
            file_id="file-1",
            file_unique_id="unique-1",
            width=1280,
            height=720,
        ).model_dump(mode="json")

        await runner.handle_payload(payload)

        image_agent.respond.assert_awaited_once()
        chat_agent.respond.assert_not_called()
        audio_agent.respond.assert_not_called()
        self.assertEqual(storage_service.record_event.await_count, 2)
        nats_client.publish_model.assert_awaited_once()
        self.assertEqual(
            nats_client.publish_model.await_args.args[0],
            AGENT_RESPONSE_SUBJECT,
        )
        self.assertEqual(
            nats_client.publish_model.await_args.args[1].reply_to_message_id,
            300,
        )

    async def test_audio_event_routes_to_audio_agent_and_publishes_audio_response(self) -> None:
        nats_client = AsyncMock()
        chat_agent = AsyncMock()
        image_agent = AsyncMock()
        audio_agent = AsyncMock()
        storage_service = AsyncMock()
        audio_agent.transcribe.return_value = "Please reply in audio"
        audio_agent.respond.return_value = AgentReply(
            response="Audio reply",
            audio_bytes=b"voice-bytes",
            audio_mime_type="audio/wav",
            audio_file_name="reply.wav",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = AgentReceiverRunner(
                nats_client=nats_client,
                chat_agent=chat_agent,
                image_agent=image_agent,
                audio_agent=audio_agent,
                audio_artifact_store=AudioArtifactStore(Path(temp_dir)),
                storage_service=storage_service,
            )
            payload = TelegramAudioEvent(
                event_id="evt-3",
                source="telegram",
                channel_id=100,
                sender_id=200,
                message_id=300,
                media_type="audio",
                file_id="file-1",
                file_unique_id="unique-1",
                duration_seconds=42,
            ).model_dump(mode="json")

            await runner.handle_payload(payload)

            chat_agent.respond.assert_not_called()
            image_agent.respond.assert_not_called()
            audio_agent.respond.assert_awaited_once()
            self.assertEqual(storage_service.record_event.await_count, 2)
            nats_client.publish_model.assert_awaited_once()
            subject, response_event = nats_client.publish_model.await_args.args
            self.assertEqual(subject, AGENT_RESPONSE_SUBJECT)
            self.assertEqual(response_event.response, "Audio reply")
            self.assertEqual(response_event.audio_mime_type, "audio/wav")
            self.assertEqual(response_event.audio_file_name, "reply.wav")
            self.assertIsNotNone(response_event.audio_file_path)
            audio_path = Path(response_event.audio_file_path or "")
            self.assertTrue(audio_path.exists())
            self.assertEqual(audio_path.read_bytes(), b"voice-bytes")

    async def test_audio_spool_is_retained_when_publish_fails(self) -> None:
        nats_client = AsyncMock()
        nats_client.publish_model.side_effect = RuntimeError("publish failed")
        chat_agent = AsyncMock()
        image_agent = AsyncMock()
        audio_agent = AsyncMock()
        storage_service = AsyncMock()
        audio_agent.transcribe.return_value = "Please reply in audio"
        audio_agent.respond.return_value = AgentReply(
            response="Audio reply",
            audio_bytes=b"voice-bytes",
            audio_mime_type="audio/wav",
            audio_file_name="reply.wav",
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            runner = AgentReceiverRunner(
                nats_client=nats_client,
                chat_agent=chat_agent,
                image_agent=image_agent,
                audio_agent=audio_agent,
                audio_artifact_store=AudioArtifactStore(Path(temp_dir)),
                storage_service=storage_service,
            )
            payload = TelegramAudioEvent(
                event_id="evt-4",
                source="telegram",
                channel_id=100,
                sender_id=200,
                message_id=300,
                media_type="audio",
                file_id="file-1",
                file_unique_id="unique-1",
                duration_seconds=42,
            ).model_dump(mode="json")

            with self.assertRaises(RuntimeError):
                await runner.handle_payload(payload)

            files = list(Path(temp_dir).iterdir())
            self.assertEqual(len(files), 1)

    async def test_processing_error_publishes_concise_failure_message(self) -> None:
        nats_client = AsyncMock()
        chat_agent = AsyncMock()
        image_agent = AsyncMock()
        audio_agent = AsyncMock()
        storage_service = AsyncMock()
        audio_agent.respond = AsyncMock(side_effect=RuntimeError("processing blew up"))

        runner = AgentReceiverRunner(
            nats_client=nats_client,
            chat_agent=chat_agent,
            image_agent=image_agent,
            audio_agent=audio_agent,
            storage_service=storage_service,
        )
        payload = TelegramAudioEvent(
            event_id="evt-5",
            source="telegram",
            channel_id=100,
            sender_id=200,
            message_id=300,
            media_type="voice",
            file_id="file-1",
            file_unique_id="unique-1",
            duration_seconds=12,
        ).model_dump(mode="json")

        await runner.handle_payload(payload)

        nats_client.publish_model.assert_awaited_once()
        subject, response_event = nats_client.publish_model.await_args.args
        self.assertEqual(subject, AGENT_RESPONSE_SUBJECT)
        self.assertEqual(
            response_event.response,
            "I couldn't process that audio because of an internal processing issue. Please try again.",
        )

    async def test_start_and_stop_manage_dependencies(self) -> None:
        nats_client = AsyncMock()
        chat_agent = AsyncMock()
        image_agent = AsyncMock()
        audio_agent = AsyncMock()
        storage_service = AsyncMock()

        runner = AgentReceiverRunner(
            nats_client=nats_client,
            chat_agent=chat_agent,
            image_agent=image_agent,
            audio_agent=audio_agent,
            storage_service=storage_service,
        )

        await runner.start()
        await runner.stop()

        storage_service.start.assert_awaited_once()
        nats_client.connect.assert_awaited_once()
        chat_agent.start.assert_awaited_once()
        image_agent.start.assert_awaited_once()
        audio_agent.start.assert_awaited_once()
        nats_client.subscribe_json.assert_awaited_once()
        audio_agent.stop.assert_awaited_once()
        image_agent.stop.assert_awaited_once()
        chat_agent.stop.assert_awaited_once()
        nats_client.close.assert_awaited_once()
        storage_service.close.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
