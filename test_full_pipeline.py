import asyncio
from agent.research_steps import gather_sources, read_sources, synthesize_report
import logging

logging.basicConfig(level=logging.WARNING)

async def main():
    topic = "latest mars mission"
    specific_questions = []

    print("1. Gathering sources...")
    urls = await gather_sources(topic, specific_questions)
    print(f"Gathered {len(urls)} URLs: {urls}")
    
    if not urls:
        print("No URLs found, exiting.")
        return

    # Take max 2 urls to speed up testing
    urls = urls[:2]

    print("\n2. Reading sources...")
    sources_content = await read_sources(urls)
    print(f"Sources Content (first 500 chars):\n{sources_content[:500]}")
    
    if not sources_content.strip():
        print("No content read, exiting.")
        return
        
    print("\n3. Synthesizing report...")
    report = await synthesize_report(topic, specific_questions, sources_content)
    print(f"Report length: {len(report)}")
    print("Report Preview:")
    print(report[:1000])

if __name__ == "__main__":
    asyncio.run(main())
