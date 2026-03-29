import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from agent.base_agent import AgentReply
from agent.audio_agent import AudioAgent
from models import TelegramAudioEvent


class AudioAgentTests(unittest.IsolatedAsyncioTestCase):
    async def test_respond_falls_back_to_text_when_synthesis_fails(self) -> None:
        chat_agent = AsyncMock()
        chat_agent.respond.return_value = AgentReply(response="Here is the answer.")

        with patch("agent.audio_agent.Bot") as MockBot:
            mock_bot = MagicMock()
            mock_bot.get_file = AsyncMock()
            mock_file = AsyncMock()
            mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b'audio'))
            mock_bot.get_file.return_value = mock_file
            MockBot.return_value = mock_bot
            
            agent = AudioAgent(
                chat_agent=chat_agent,
                bot_token="telegram-token",
            )

        with patch.object(agent, "synthesize", AsyncMock(side_effect=RuntimeError("tts failed"))):
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
                )
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
