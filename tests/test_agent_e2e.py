import pytest
from datetime import datetime, timezone
import asyncio
from agent.chat_agent import ChatAgent
from models import TelegramMessageEvent
from storage.service import StorageService
from agent.gemini_model import gemini_model
from config.env_config import init_config

@pytest.mark.asyncio
async def test_agent_end_to_end_flow():
    # Setup test DB or use in-memory
    init_config()
    storage_service = StorageService()
    await storage_service.start()
    
    agent = ChatAgent(
        storage_service=storage_service,
        history_limit=2,
        memory_limit=2,
    )
    
    # We create a simple request
    event = TelegramMessageEvent(
        event_id="e2e-test-1",
        source="telegram",
        message="What is the capital of France?",
        channel_id=9999,
        sender_id=8888,
        message_id=7777,
        occurred_at=datetime.now(timezone.utc)
    )
    
    await storage_service.record_event(event)
    
    reply = await agent.respond(event)
    assert reply is not None
    assert "Paris" in reply.response
    
    from models import AgentResponseEvent
    from uuid import uuid4
    response_event = AgentResponseEvent(
        event_id=f"agent|{uuid4()}",
        source="agent",
        request_event_id=event.event_id,
        channel_id=event.channel_id,
        sender_id=0,
        reply_to_message_id=event.message_id,
        response=reply.response,
    )
    await storage_service.record_event(response_event, channel_source=event.source)
    
    # Check if history is stored
    history = await storage_service.get_conversation_context(9999, limit=10)
    assert len(history) >= 2 # One user turn, one agent turn
    
    await storage_service.close()

