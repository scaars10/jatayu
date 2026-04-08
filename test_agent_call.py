import asyncio
from agent.chat_agent import ChatAgent
from models import TelegramMessageEvent
from uuid import uuid4

async def main():
    agent = ChatAgent()
    event = TelegramMessageEvent(
        event_id=f"telegram|{uuid4()}",
        source="telegram",
        channel_id=123,
        sender_id=456,
        message_id=789,
        message="Provide summary of new updates for continuous research",
        occurred_at="2026-04-07T18:14:01Z"
    )
    print("Sending message to agent...")
    reply = await agent.respond(event)
    print("Reply:", reply)

if __name__ == "__main__":
    asyncio.run(main())
