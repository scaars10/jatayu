import asyncio
from agent.research_steps import gather_sources

async def main():
    print("Testing gather_sources...")
    urls = await gather_sources("latest mars mission", [])
    print(f"URLs: {urls}")

if __name__ == "__main__":
    asyncio.run(main())
