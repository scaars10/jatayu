import asyncio
from agent.research_steps import gather_sources
import logging

logging.basicConfig(level=logging.DEBUG)

async def main():
    print("\nTesting gather_sources...")
    urls = await gather_sources("latest mars mission", [])
    print(f"URLs: {urls}")

if __name__ == "__main__":
    asyncio.run(main())
