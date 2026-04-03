import asyncio
from pydantic_ai import Agent
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from agent.gemini_model import gemini_model
from config.env_config import init_config, get_env
from agent.research_steps import web_search, read_pdf
from agent.deep_research_agent import start_deep_research_task, get_research_task_status, provide_feedback_to_research_task, continue_research_task

async def main():
    init_config()
    api_key = get_env("GEMINI_API_KEY", required=True)
    agent = Agent(
        model=GoogleModel(
            gemini_model.get_balanced_model(),
            provider=GoogleProvider(api_key=api_key),
        ),
        tools=[
            web_search,
            start_deep_research_task,
            get_research_task_status,
            provide_feedback_to_research_task,
            continue_research_task,
            read_pdf
        ],
    )
    result = await agent.run("Hello, who are you?")
    print(dir(result))
    print("Response:", getattr(result, "data", getattr(result, "output", None)))

if __name__ == "__main__":
    asyncio.run(main())
