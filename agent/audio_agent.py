from __future__ import annotations

import asyncio
import io
import mimetypes
import wave

from google.genai import errors as genai_errors
from google.genai import types
from telegram import Bot

from agent.base_agent import AgentReply, BaseAgent
from agent.chat_agent import ChatAgent
from agent.gemini_model import get_client
from config.env_config import get_env
from models import TelegramAudioEvent, TelegramMessageEvent

DEFAULT_AUDIO_TRANSCRIPTION_MODEL = "gemini-2.5-flash"
DEFAULT_AUDIO_TTS_MODEL = "gemini-2.5-flash-preview-tts"
DEFAULT_AUDIO_VOICE = "Sadaltager"

DEFAULT_AUDIO_FILE_NAME = "reply.ogg"
DEFAULT_AUDIO_MIME_TYPE = "audio/ogg"

DEFAULT_VOICE_MIME_TYPE = "audio/ogg"
DEFAULT_AUDIO_INPUT_MIME_TYPE = "audio/mpeg"

# Gemini TTS output should be treated as mono 24kHz 16-bit PCM before wrapping/transcoding.
TTS_PCM_SAMPLE_RATE = 24000
TTS_PCM_CHANNELS = 1
TTS_PCM_SAMPLE_WIDTH = 2  # 16-bit PCM

TRANSCRIPTION_PROMPT = (
    "Transcribe the user's audio message. Return only the user's spoken words. "
    "If the audio does not contain clear speech, briefly summarize the audio in one sentence."
)

TTS_PROMPT_TEMPLATE = (
    "Read the following assistant reply exactly as written. "
    "Voice style: calm, intelligent, polished, efficient, confident, warm but restrained. "
    "Pace: a bit faster (not too fast) human conversational speed, slightly brisk, smooth and fluent."
    "Tone: Think something like Jarvis from Ironman meets Morgan Freeman. "
    "Do not add any extra words.\n\n{response}"
)


class AudioAgent(BaseAgent):
    def __init__(
        self,
        chat_agent: ChatAgent | None = None,
        bot_token: str | None = None,
        transcription_model: str | None = None,
        tts_model: str | None = None,
        voice_name: str | None = None,
    ) -> None:
        if bot_token is None:
            bot_token = get_env("JATAYU_TELEGRAM_TOKEN", required=True) or ""

        self.bot = Bot(token=bot_token)
        self.chat_agent = chat_agent or ChatAgent()
        self.transcription_model = transcription_model or (
            get_env(
                "JATAYU_AUDIO_TRANSCRIPTION_MODEL",
                DEFAULT_AUDIO_TRANSCRIPTION_MODEL,
            )
            or DEFAULT_AUDIO_TRANSCRIPTION_MODEL
        )
        self.tts_model = tts_model or (
            get_env("JATAYU_AUDIO_TTS_MODEL", DEFAULT_AUDIO_TTS_MODEL)
            or DEFAULT_AUDIO_TTS_MODEL
        )
        self.voice_name = voice_name or (
            get_env("JATAYU_AUDIO_VOICE", DEFAULT_AUDIO_VOICE)
            or DEFAULT_AUDIO_VOICE
        )

    async def start(self) -> None:
        await self.bot.initialize()

    async def stop(self) -> None:
        await self.bot.shutdown()

    async def transcribe(self, event: TelegramAudioEvent) -> str:
        file = await self.bot.get_file(event.file_id)
        audio_bytes = bytes(await file.download_as_bytearray())

        response = await get_client().aio.models.generate_content(
            model=self.transcription_model,
            contents=[
                TRANSCRIPTION_PROMPT,
                types.Part.from_bytes(
                    data=audio_bytes,
                    mime_type=self._resolve_input_mime_type(event),
                ),
            ],
        )
        transcript = (response.text or "").strip()
        user_message = self._compose_user_message(event, transcript)
        print(f"[AUDIO][{event.channel_id}] transcript={user_message or '<empty>'}")
        return user_message

    async def respond(
        self,
        event: TelegramAudioEvent,
        transcript: str | None = None,
    ) -> AgentReply | None:
        user_message = (transcript or event.transcript or "").strip()
        if not user_message:
            return None

        message_event = TelegramMessageEvent(
            event_id=event.event_id,
            source=event.source,
            occurred_at=event.occurred_at,
            message=user_message,
            channel_id=event.channel_id,
            sender_id=event.sender_id,
            message_id=event.message_id,
        )

        response_text = await self.chat_agent.respond(message_event)
        if not response_text:
            return None

        try:
            audio_bytes, audio_mime_type = await self._synthesize(response_text)
        except Exception as exc:
            print(f"[AUDIO][TTS][ERROR] {exc}")
            return AgentReply(
                response=self._build_audio_fallback_response(response_text, exc)
            )

        if not audio_bytes:
            return AgentReply(
                response=self._build_audio_fallback_response(
                    response_text,
                    RuntimeError("Audio reply was empty"),
                )
            )

        return AgentReply(
            response=response_text,
            audio_bytes=audio_bytes,
            audio_mime_type=audio_mime_type,
            audio_file_name=self._build_audio_file_name(audio_mime_type),
        )

    async def _synthesize(self, response_text: str) -> tuple[bytes | None, str | None]:
        response = await get_client().aio.models.generate_content(
            model=self.tts_model,
            contents=TTS_PROMPT_TEMPLATE.format(response=response_text),
            config=types.GenerateContentConfig(
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self.voice_name,
                        )
                    )
                ),
            ),
        )

        pcm_bytes = self._extract_audio_payload(response)
        if not pcm_bytes:
            return None, None

        ogg_opus_bytes = await self._pcm_to_ogg_opus(
            pcm_bytes=pcm_bytes,
            sample_rate=TTS_PCM_SAMPLE_RATE,
            channels=TTS_PCM_CHANNELS,
            sample_width=TTS_PCM_SAMPLE_WIDTH,
        )

        return ogg_opus_bytes, DEFAULT_AUDIO_MIME_TYPE

    @staticmethod
    def _compose_user_message(
        event: TelegramAudioEvent,
        transcript: str,
    ) -> str:
        parts: list[str] = []
        cleaned_transcript = transcript.strip()
        if cleaned_transcript:
            parts.append(cleaned_transcript)

        caption = (event.caption or "").strip()
        if caption and caption not in parts:
            parts.append(caption)

        return "\n\n".join(parts)

    @staticmethod
    def _extract_audio_payload(
        response,
    ) -> bytes | None:
        chunks: list[bytes] = []

        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) or []
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is None:
                    continue
                data = getattr(inline_data, "data", None)
                if isinstance(data, bytes) and data:
                    chunks.append(data)

        if not chunks:
            return None

        return b"".join(chunks)

    @staticmethod
    def _build_audio_fallback_response(
        response_text: str,
        exc: Exception,
    ) -> str:
        if isinstance(exc, genai_errors.ServerError):
            notice = (
                "I hit a temporary voice-generation issue, so I'm sending the text reply instead."
            )
        elif isinstance(exc, FileNotFoundError):
            notice = "I couldn't render the audio reply locally, so I'm sending text instead."
        else:
            notice = "I couldn't generate the audio reply, so I'm sending text instead."

        return f"{notice}\n\n{response_text}"

    @staticmethod
    async def _pcm_to_ogg_opus(
        *,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        sample_width: int,
    ) -> bytes:
        """
        Convert raw PCM from Gemini TTS into OGG/Opus so Telegram can display it
        as a regular voice note via sendVoice.
        """
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM is supported for OGG/Opus conversion")

        process = await asyncio.create_subprocess_exec(
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(sample_rate),
            "-ac",
            str(channels),
            "-i",
            "pipe:0",
            "-c:a",
            "libopus",
            "-b:a",
            "32k",
            "-vbr",
            "on",
            "-application",
            "voip",
            "-f",
            "ogg",
            "pipe:1",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await process.communicate(input=pcm_bytes)

        if process.returncode != 0:
            raise RuntimeError(stderr.decode("utf-8", errors="ignore").strip())

        if not stdout:
            raise RuntimeError("ffmpeg produced empty ogg output")

        return stdout

    @staticmethod
    def _wrap_pcm_as_wav(
        *,
        pcm_bytes: bytes,
        sample_rate: int,
        channels: int,
        sample_width: int,
    ) -> bytes:
        """
        Optional helper for debugging only.
        """
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_bytes)
        return buffer.getvalue()

    @staticmethod
    def _build_audio_file_name(mime_type: str | None) -> str:
        if not mime_type:
            return DEFAULT_AUDIO_FILE_NAME

        extension = mimetypes.guess_extension(mime_type, strict=False)
        if not extension:
            return DEFAULT_AUDIO_FILE_NAME

        if extension == ".oga":
            return "reply.ogg"

        return f"reply{extension}"

    @staticmethod
    def _resolve_input_mime_type(event: TelegramAudioEvent) -> str:
        if event.mime_type:
            return event.mime_type

        if event.file_name:
            guessed_mime_type, _ = mimetypes.guess_type(event.file_name)
            if guessed_mime_type:
                return guessed_mime_type

        if event.media_type == "voice":
            return DEFAULT_VOICE_MIME_TYPE

        return DEFAULT_AUDIO_INPUT_MIME_TYPE
