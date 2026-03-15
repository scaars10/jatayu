import unittest

from pydantic import ValidationError

from agent.model_selector import StaticModelSelector
from models import (
    BaseEvent,
    TelegramAudioEvent,
    TelegramMessageEvent,
    TelegramPhotoEvent,
)


class PydanticModelTests(unittest.TestCase):
    def test_static_model_selector_returns_expected_models(self) -> None:
        selector = StaticModelSelector(
            balanced_model="balanced",
            large_model="large",
            light_model="light",
        )

        self.assertEqual(selector.get_balanced_model(), "balanced")
        self.assertEqual(selector.get_large_model(), "large")
        self.assertEqual(selector.get_light_model(), "light")

    def test_static_model_selector_validates_input_types(self) -> None:
        with self.assertRaises(ValidationError):
            StaticModelSelector(
                balanced_model="balanced",
                large_model="large",
                light_model=123,
            )

    def test_base_event_uses_timezone_aware_default_timestamp(self) -> None:
        event = BaseEvent(event_id="evt-1", source="telegram")

        self.assertIsNotNone(event.occurred_at.tzinfo)
        self.assertEqual(event.model_dump()["source"], "telegram")

    def test_base_event_parses_iso_timestamps(self) -> None:
        event = BaseEvent(
            event_id="evt-2",
            source="telegram",
            occurred_at="2026-03-14T10:15:00Z",
        )

        self.assertEqual(event.occurred_at.isoformat(), "2026-03-14T10:15:00+00:00")

    def test_models_are_frozen(self) -> None:
        selector = StaticModelSelector(
            balanced_model="balanced",
            large_model="large",
            light_model="light",
        )
        event = TelegramMessageEvent(
            event_id="evt-3",
            source="telegram",
            message="hello",
            channel_id=100,
            sender_id=200,
        )

        with self.assertRaises(ValidationError):
            selector.light_model = "other"

        with self.assertRaises(ValidationError):
            event.source = "other"

    def test_telegram_photo_event_includes_media_lookup_fields(self) -> None:
        event = TelegramPhotoEvent(
            event_id="evt-photo",
            source="telegram",
            channel_id=100,
            sender_id=200,
            message_id=300,
            file_id="file-1",
            file_unique_id="unique-1",
            width=1280,
            height=720,
            caption="caption",
            file_size=4096,
            media_group_id="group-1",
        )

        self.assertEqual(event.file_id, "file-1")
        self.assertEqual(event.media_group_id, "group-1")

    def test_telegram_audio_event_supports_audio_and_voice_metadata(self) -> None:
        audio_event = TelegramAudioEvent(
            event_id="evt-audio",
            source="telegram",
            channel_id=100,
            sender_id=200,
            message_id=300,
            media_type="audio",
            file_id="file-2",
            file_unique_id="unique-2",
            duration_seconds=42,
            mime_type="audio/mpeg",
            file_name="clip.mp3",
            performer="artist",
            title="track",
            file_size=8192,
        )
        voice_event = TelegramAudioEvent(
            event_id="evt-voice",
            source="telegram",
            channel_id=100,
            sender_id=200,
            message_id=301,
            media_type="voice",
            file_id="file-3",
            file_unique_id="unique-3",
            duration_seconds=7,
        )

        self.assertEqual(audio_event.media_type, "audio")
        self.assertEqual(audio_event.file_name, "clip.mp3")
        self.assertEqual(voice_event.media_type, "voice")


if __name__ == "__main__":
    unittest.main()
