from google.genai import types
from telegram import Bot

from agent.base_agent import BaseAgent
from agent.gemini_model import gemini_model, get_client
from config.env_config import get_env
from models import TelegramPhotoEvent

DEFAULT_IMAGE_PROMPT = "Describe this image briefly."
DEFAULT_IMAGE_MIME_TYPE = "image/jpeg"


class ImageAgent(BaseAgent):
    def __init__(self, bot_token: str | None = None) -> None:
        if bot_token is None:
            bot_token = get_env("JATAYU_TELEGRAM_TOKEN", required=True) or ""

        self.bot = Bot(token=bot_token)

    async def start(self) -> None:
        await self.bot.initialize()

    async def stop(self) -> None:
        await self.bot.shutdown()

    async def respond(self, event: TelegramPhotoEvent) -> str | None:
        file = await self.bot.get_file(event.file_id)
        image_bytes = bytes(await file.download_as_bytearray())
        prompt = event.caption or DEFAULT_IMAGE_PROMPT
        print(f"[IMAGE][{event.channel_id}] prompt={prompt}")
        response = await get_client().aio.models.generate_content(
            model=gemini_model.get_balanced_model(),
            contents=[
                types.Part.from_text(text=prompt),
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=DEFAULT_IMAGE_MIME_TYPE,
                ),
            ],
        )
        return response.text
