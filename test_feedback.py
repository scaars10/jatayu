import asyncio
from agent.research_steps import synthesize_report
import logging

logging.basicConfig(level=logging.WARNING)

async def main():
    topic = "latest mars mission"
    specific_questions = []
    sources_content = "Source: https://mars.nasa.gov/\nMars 2020 Perseverance Rover is doing great.\n\n"
    feedback = "Make sure the report sounds like a pirate."
    
    print("Synthesizing report with feedback...")
    report = await synthesize_report(topic, specific_questions, sources_content, feedback)
    print(f"Report length: {len(report)}")
    print("Report Preview:")
    print(report)

if __name__ == "__main__":
    asyncio.run(main())
