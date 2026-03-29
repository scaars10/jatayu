import unittest
from unittest.mock import AsyncMock, patch

from agent.audio_agent import AudioAgent
from models import TelegramAudioEvent


class AudioAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_respond_falls_back_to_text_when_synthesis_fails(self) -> None:
        chat_agent = AsyncMock()
        chat_agent.respond.return_value = "Here is the answer."

        with patch("agent.audio_agent.Bot"):
            agent = AudioAgent(
                chat_agent=chat_agent,
                bot_token="telegram-token",
            )

        with patch.object(agent, "_synthesize", AsyncMock(side_effect=RuntimeError("tts failed"))):
            reply = await agent.respond(
                TelegramAudioEvent(
                    event_id="evt-audio",
                    source="telegram",
                    channel_id=100,
                    sender_id=200,
                    message_id=300,
                    media_type="voice",
                    file_id="file-1",
                    file_unique_id="unique-1",
                    duration_seconds=7,
                    transcript="hello",
                ),
                transcript="hello",
            )

        self.assertIsNotNone(reply)
        assert reply is not None
        self.assertEqual(
            reply.response,
            "I couldn't generate the audio reply, so I'm sending text instead.\n\nHere is the answer.",
        )
        self.assertIsNone(reply.audio_bytes)


if __name__ == "__main__":
    unittest.main()
