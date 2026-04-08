import mimetypes
import re

from comms.base_runner import BaseRunner
from telegram import InputFile
from telegram.constants import ParseMode
from telegram.error import BadRequest
from telegram.ext import ApplicationBuilder, MessageHandler, filters

from comms.nats import NatsClient, build_nats_client
from constants import AGENT_RESPONSE_SUBJECT
from comms.telegram.listener.message_listener import MessageListener
from config.env_config import get_env, get_env_int_list
from models import AgentResponseEvent
from storage import AudioArtifactStore, StorageService


class TelegramRunner(BaseRunner):
    _TABLE_SEPARATOR_RE = re.compile(r"^:?-{3,}:?$")
    _INTERNAL_TOOL_CALL_RE = re.compile(
        r'["\'}\]]*\s*call:[^:\s{}]+(?::[^:\s{}]+)*:final_result\{.*$',
        flags=re.DOTALL,
    )

    def __init__(
        self,
        nats_client: NatsClient | None = None,
        bot_token: str | None = None,
        allowed_chat_ids: list[int] | None = None,
        audio_artifact_store: AudioArtifactStore | None = None,
        storage_service: StorageService | None = None,
    ) -> None:
        if nats_client is None:
            nats_client = build_nats_client()

        if bot_token is None:
            bot_token = get_env("JATAYU_TELEGRAM_TOKEN", required=True) or ""

        if allowed_chat_ids is None:
            allowed_chat_ids = get_env_int_list("TELEGRAM_LISTENER_CHAT_ID", required=True)

        self.nats_client = nats_client
        self.audio_artifact_store = audio_artifact_store or AudioArtifactStore()
        self.storage_service = storage_service or StorageService()
        self.application = ApplicationBuilder().token(bot_token).build()
        self.listener = MessageListener(allowed_chat_ids, nats_client)
        self.application.add_handler(MessageHandler(filters.ALL, self.listener.on_message))

    async def start(self) -> None:
        await self.storage_service.start()
        await self.nats_client.connect()
        await self.application.initialize()
        await self.nats_client.subscribe_json(AGENT_RESPONSE_SUBJECT, self.handle_agent_response)
        await self.application.start()

        if self.application.updater is None:
            raise RuntimeError("Telegram application is missing an updater")

        await self.application.updater.start_polling()

    async def stop(self) -> None:
        updater = self.application.updater

        if updater is not None and updater.running:
            await updater.stop()

        if self.application.running:
            await self.application.stop()

        await self.application.shutdown()
        await self.nats_client.close()
        await self.storage_service.close()

    async def handle_agent_response(self, payload: dict[str, object]) -> None:
        event = AgentResponseEvent.model_validate(payload)
        try:
            sent_message = await self._send_agent_response(event)
            await self.storage_service.mark_message_delivered(
                event_id=event.event_id,
                provider_message_id=sent_message.message_id,
            )
        except Exception as exc:
            print(f"[TELEGRAM][ERROR][{event.event_id}] {exc}")
            sent_message = await self.application.bot.send_message(
                chat_id=event.channel_id,
                text=self._build_delivery_error_notice(event),
                reply_to_message_id=event.reply_to_message_id,
            )
            await self.storage_service.mark_message_delivered(
                event_id=event.event_id,
                provider_message_id=sent_message.message_id,
            )

    async def _send_text_in_chunks(self, chat_id: int, text: str, reply_to_message_id: int | None):
        max_length = 4000
        if not text:
            return await self.application.bot.send_message(
                chat_id=chat_id,
                text="*",
                reply_to_message_id=reply_to_message_id,
            )

        formatted_text = self._format_for_telegram(text)
        sent_message = None
        for i in range(0, len(formatted_text), max_length):
            chunk = formatted_text[i:i + max_length]
            try:
                sent_message = await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    reply_to_message_id=reply_to_message_id if i == 0 else None,
                    parse_mode=ParseMode.MARKDOWN,
                )
            except BadRequest:
                # Fallback to plain text if markdown parsing fails
                sent_message = await self.application.bot.send_message(
                    chat_id=chat_id,
                    text=chunk,
                    reply_to_message_id=reply_to_message_id if i == 0 else None,
                )
        return sent_message

    async def _send_agent_response(self, event: AgentResponseEvent):
        if not event.audio_file_path:
            return await self._send_text_in_chunks(
                chat_id=event.channel_id,
                text=event.response,
                reply_to_message_id=event.reply_to_message_id,
            )

        try:
            audio_file_path = self.audio_artifact_store.resolve_managed_path(event.audio_file_path)
        except ValueError:
            return await self._send_audio_fallback(
                event,
                "I couldn't find the audio reply, so I'm sending the text version instead.",
            )

        if not audio_file_path.exists() or not audio_file_path.is_file():
            return await self._send_audio_fallback(
                event,
                "The audio reply file was missing, so I'm sending the text version instead.",
            )

        file_name = event.audio_file_name or self._build_audio_file_name(event.audio_mime_type)
        caption = event.response if len(event.response) <= 1024 else None

        is_voice_note = self._is_voice_note_mime_type(event.audio_mime_type, file_name)

        try:
            with audio_file_path.open("rb") as audio_file:
                input_file = InputFile(audio_file, filename=file_name)
                
                formatted_caption = self._format_for_telegram(caption) if caption else None

                kwargs = {
                    "chat_id": event.channel_id,
                    "caption": formatted_caption,
                    "reply_to_message_id": event.reply_to_message_id,
                    "parse_mode": ParseMode.MARKDOWN if formatted_caption else None,
                }

                try:
                    if is_voice_note:
                        msg = await self.application.bot.send_voice(voice=input_file, **kwargs)
                    else:
                        msg = await self.application.bot.send_audio(audio=input_file, **kwargs)
                except BadRequest:
                    audio_file.seek(0)
                    kwargs["caption"] = caption
                    kwargs["parse_mode"] = None
                    if is_voice_note:
                        msg = await self.application.bot.send_voice(voice=input_file, **kwargs)
                    else:
                        msg = await self.application.bot.send_audio(audio=input_file, **kwargs)
                
                # If we had to omit the caption because it was too long, send the full text separately
                if caption is None and event.response:
                    await self._send_text_in_chunks(
                        chat_id=event.channel_id,
                        text=event.response,
                        reply_to_message_id=msg.message_id
                    )
                return msg
        except Exception:
            return await self._send_audio_fallback(
                event,
                "I couldn't send the audio reply, so I'm sending the text version instead.",
            )

    async def _send_audio_fallback(
        self,
        event: AgentResponseEvent,
        notice: str,
    ):
        return await self._send_text_in_chunks(
            chat_id=event.channel_id,
            text=f"{notice}\n\n{event.response}",
            reply_to_message_id=event.reply_to_message_id,
        )

    @staticmethod
    def _build_delivery_error_notice(event: AgentResponseEvent) -> str:
        if event.audio_file_path:
            return "I couldn't deliver the audio reply. Please try again."

        return "I couldn't deliver the reply. Please try again."

    @staticmethod
    def _build_audio_file_name(mime_type: str | None) -> str:
        if not mime_type:
            return "reply.ogg"

        extension = mimetypes.guess_extension(mime_type, strict=False)
        if not extension:
            return "reply.ogg"

        if extension == ".oga":
            extension = ".ogg"

        return f"reply{extension}"

    @staticmethod
    def _is_voice_note_mime_type(mime_type: str | None, file_name: str | None) -> bool:
        normalized_mime = (mime_type or "").lower().strip()
        normalized_name = (file_name or "").lower().strip()

        if normalized_mime in {"audio/ogg", "audio/opus", "application/ogg"}:
            return True

        return normalized_name.endswith(".ogg") or normalized_name.endswith(".opus")

    @staticmethod
    def _format_for_telegram(text: str) -> str:
        text = TelegramRunner._sanitize_internal_artifacts(text)
        text = TelegramRunner._convert_markdown_tables(text)
        # Telegram Legacy Markdown uses * for bold, but LLMs often output **
        # Replace **bold** with *bold*
        text = re.sub(r'\*\*([^\*]+)\*\*', r'*\1*', text)
        # LLMs also sometimes output ### headers. Legacy markdown has no header support.
        # Replace ### Header with *Header*
        text = re.sub(r'^#+\s+(.+)$', r'*\1*', text, flags=re.MULTILINE)
        # Replace --- horizontal rules with a unicode line
        text = re.sub(r'^\s*[-_]{3,}\s*$', '──────────────', text, flags=re.MULTILINE)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def _sanitize_internal_artifacts(cls, text: str) -> str:
        if not text:
            return text

        cleaned = cls._INTERNAL_TOOL_CALL_RE.sub("", text).strip()
        if cleaned:
            return cleaned
        return text.strip()

    @classmethod
    def _convert_markdown_tables(cls, text: str) -> str:
        lines = text.splitlines()
        converted: list[str] = []
        i = 0

        while i < len(lines):
            if cls._looks_like_markdown_table(lines, i):
                headers = cls._split_table_row(lines[i])
                i += 2
                table_rows: list[str] = []

                while i < len(lines) and cls._looks_like_table_row(lines[i]):
                    row = cls._render_table_row(headers, cls._split_table_row(lines[i]))
                    if row:
                        table_rows.append(row)
                    i += 1

                if table_rows:
                    if converted and converted[-1].strip():
                        converted.append("")
                    converted.extend(table_rows)
                    if i < len(lines) and lines[i].strip():
                        converted.append("")
                    continue

            converted.append(lines[i])
            i += 1

        return "\n".join(converted)

    @classmethod
    def _looks_like_markdown_table(cls, lines: list[str], index: int) -> bool:
        if index + 1 >= len(lines):
            return False

        header = cls._split_table_row(lines[index])
        separator = cls._split_table_row(lines[index + 1])
        if len(header) < 2 or len(header) != len(separator):
            return False

        return all(
            cls._TABLE_SEPARATOR_RE.fullmatch(cell) is not None
            for cell in separator
        )

    @staticmethod
    def _looks_like_table_row(line: str) -> bool:
        stripped = line.strip()
        return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2

    @staticmethod
    def _split_table_row(line: str) -> list[str]:
        return [cell.strip() for cell in line.strip().strip("|").split("|")]

    @staticmethod
    def _render_table_row(headers: list[str], cells: list[str]) -> str:
        pairs = [
            (headers[index], cells[index])
            for index in range(min(len(headers), len(cells)))
            if cells[index]
        ]
        if not pairs:
            return ""

        first_header = headers[0].strip().lower()
        if first_header in {"project", "name", "item"}:
            lead = pairs[0][1]
            details = [f"{header}: {value}" for header, value in pairs[1:]]
            if details:
                return f"- *{lead}*: " + "; ".join(details)
            return f"- *{lead}*"

        return "- " + " | ".join(f"{header}: {value}" for header, value in pairs)
