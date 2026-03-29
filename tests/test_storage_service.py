import tempfile
import unittest
from pathlib import Path

from models import (
    AgentResponseEvent,
    TelegramAudioEvent,
    TelegramMessageEvent,
    TelegramPhotoEvent,
)
from storage import Database, StorageService


class StorageServiceTests(unittest.IsolatedAsyncioTestCase):
    async def test_storage_service_creates_schema_and_persists_context(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "messages.sqlite3"
            service = StorageService(Database(database_path))

            text_event = TelegramMessageEvent(
                event_id="telegram|evt-1",
                source="telegram",
                message="hello",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )
            photo_event = TelegramPhotoEvent(
                event_id="telegram|evt-2",
                source="telegram",
                channel_id=100,
                sender_id=200,
                message_id=301,
                file_id="photo-file",
                file_unique_id="photo-unique",
                width=1280,
                height=720,
                caption="see this",
                file_size=4096,
            )
            audio_event = TelegramAudioEvent(
                event_id="telegram|evt-3",
                source="telegram",
                channel_id=100,
                sender_id=200,
                message_id=302,
                media_type="voice",
                file_id="voice-file",
                file_unique_id="voice-unique",
                duration_seconds=7,
                transcript="Can you help me with this?",
            )
            response_event = AgentResponseEvent(
                event_id="agent|evt-4",
                source="agent",
                request_event_id=text_event.event_id,
                channel_id=100,
                sender_id=0,
                reply_to_message_id=300,
                response="hello back",
            )

            await service.record_event(text_event)
            await service.record_event(photo_event)
            await service.record_event(audio_event)
            await service.record_event(response_event, channel_source="telegram")
            delivered_message = await service.mark_message_delivered(
                event_id=response_event.event_id,
                provider_message_id=999,
            )
            context = await service.get_conversation_context(100, limit=10)

            self.assertTrue(database_path.exists())
            self.assertIsNotNone(delivered_message)
            self.assertEqual(delivered_message.delivery_status, "delivered")
            self.assertEqual(delivered_message.provider_message_id, "999")
            self.assertEqual(
                [turn.text for turn in context],
                ["hello", "[Photo] see this", "Can you help me with this?", "hello back"],
            )

            attachment_rows = service.database.connection.execute(
                """
                SELECT media_type, provider_file_id
                FROM message_attachments
                ORDER BY id
                """
            ).fetchall()
            self.assertEqual(
                [(row["media_type"], row["provider_file_id"]) for row in attachment_rows],
                [("photo", "photo-file"), ("voice", "voice-file")],
            )
            participant_memory = await service.upsert_long_term_memory(
                scope_type="participant",
                memory_key="favorite_editor",
                category="preference",
                summary="Uses Neovim as the main editor.",
                importance="high",
                confidence=0.92,
                channel_external_id=100,
                participant_external_id=200,
                source_event_id=text_event.event_id,
            )
            channel_memory = await service.upsert_long_term_memory(
                scope_type="channel",
                memory_key="project_name",
                category="project",
                summary="The channel is building Jatayu.",
                importance="medium",
                confidence=0.81,
                channel_external_id=100,
                participant_external_id=200,
                source_event_id=text_event.event_id,
            )
            memories = await service.get_long_term_memories(100, 200, limit=10)

            self.assertEqual(participant_memory.memory_key, "favorite_editor")
            self.assertEqual(channel_memory.memory_key, "project_name")
            self.assertEqual(
                [memory.memory_key for memory in memories],
                ["favorite_editor", "project_name"],
            )

            await service.close()

    async def test_repeated_event_id_is_idempotent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = StorageService(Database(Path(temp_dir) / "messages.sqlite3"))
            event = TelegramMessageEvent(
                event_id="telegram|evt-repeat",
                source="telegram",
                message="same message",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )

            first = await service.record_event(event)
            second = await service.record_event(event)

            self.assertEqual(first.id, second.id)
            row = service.database.connection.execute(
                "SELECT COUNT(*) AS count FROM messages WHERE event_id = ?",
                (event.event_id,),
            ).fetchone()
            self.assertEqual(row["count"], 1)

            await service.close()

    async def test_long_term_memory_upsert_updates_existing_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            service = StorageService(Database(Path(temp_dir) / "messages.sqlite3"))
            event = TelegramMessageEvent(
                event_id="telegram|evt-memory",
                source="telegram",
                message="Remember that I prefer concise answers.",
                channel_id=100,
                sender_id=200,
                message_id=300,
            )

            await service.record_event(event)
            first = await service.upsert_long_term_memory(
                scope_type="participant",
                memory_key="response_style",
                category="preference",
                summary="Prefers concise answers.",
                importance="medium",
                confidence=0.8,
                channel_external_id=100,
                participant_external_id=200,
                source_event_id=event.event_id,
            )
            second = await service.upsert_long_term_memory(
                scope_type="participant",
                memory_key="response_style",
                category="preference",
                summary="Prefers very concise answers.",
                importance="high",
                confidence=0.95,
                channel_external_id=100,
                participant_external_id=200,
                source_event_id=event.event_id,
            )

            self.assertEqual(first.id, second.id)
            self.assertEqual(second.summary, "Prefers very concise answers.")
            self.assertEqual(second.importance, "high")

            await service.close()
