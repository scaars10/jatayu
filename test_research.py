import asyncio
from agent.research_steps import gather_sources, web_search

async def main():
    print("Testing web_search...")
    search_res = await web_search("latest mars mission")
    print(search_res)
    
    print("\nTesting gather_sources...")
    urls = await gather_sources("latest mars mission", [])
    print(f"URLs: {urls}")

if __name__ == "__main__":
    asyncio.run(main())
