import asyncio
from agent.knowledge_graph import KnowledgeGraphProcessor
from storage.db import Database
from storage.service import StorageService
from storage.models import ConversationTurn
from datetime import datetime

async def main():
    db = Database()
    service = StorageService(db)
    await service.start()
    
    processor = KnowledgeGraphProcessor(service)
    
    history = [
        ConversationTurn(role="user", text="I'm starting a new project called Jatayu. It's an AI assistant.", source="telegram", message_type="message", occurred_at=datetime.now()),
        ConversationTurn(role="assistant", text="That sounds great! What language are you building Jatayu in?", source="agent", message_type="message", occurred_at=datetime.now())
    ]
    user_message = "I am writing it in Python. My friend Sarah is helping me with the UI."
    
    print("Running extraction...")
    await processor.extract_and_store(user_message, history)
    
    graph = service.knowledge_graph.search_graph("Jatayu")
    print("\nExtraction Results for Jatayu:")
    print(f"Nodes: {[n['name'] for n in graph.get('query_nodes', [])]}")
    print(f"Related: {[n['name'] for n in graph.get('related_nodes', [])]}")
    print(f"Edges: {[e['relation'] for e in graph.get('edges', [])]}")

    await service.close()

asyncio.run(main())
