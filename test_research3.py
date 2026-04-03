import asyncio
import json
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from agent.gemini_model import gemini_model
from config.env_config import get_env
from agent.research_steps import DEEP_RESEARCH_SYSTEM_PROMPT, web_search

async def main():
    api_key = get_env("GEMINI_API_KEY", required=True)
    research_agent = Agent(
        model=GoogleModel(
            gemini_model.get_large_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        system_prompt=DEEP_RESEARCH_SYSTEM_PROMPT,
        tools=[web_search],
    )
    prompt = "Gather a list of relevant URLs for a research on 'latest mars mission'\nReturn a JSON list of URLs."
    result = await research_agent.run(prompt, message_history=[])
    print("OUTPUT:")
    print(repr(result.output))

if __name__ == "__main__":
    asyncio.run(main())
