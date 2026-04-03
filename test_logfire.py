import logfire
import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from config.env_config import init_config, get_env
from agent.gemini_model import gemini_model

logfire.configure(
    send_to_logfire=False,
    console=logfire.ConsoleOptions(min_log_level='trace')
)

init_config()
api_key = get_env("GEMINI_API_KEY", required=True)

agent = Agent(
    model=GoogleModel(
        gemini_model.get_balanced_model(),
        provider=GoogleProvider(api_key=api_key),
    ),
    system_prompt="You are a test"
)

async def main():
    await agent.run("Hello")

if __name__ == "__main__":
    asyncio.run(main())
