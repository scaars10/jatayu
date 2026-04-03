import asyncio
from storage.db import Database
from storage.service import StorageService

async def main():
    db = Database()
    service = StorageService(db)
    await service.start()
    
    # Test upsert node
    repo = service.knowledge_graph
    n1 = repo.upsert_node(name="Tony Stark", type="PERSON", attributes={"alias": "Iron Man"})
    n2 = repo.upsert_node(name="J.A.R.V.I.S.", type="AI", attributes={"role": "Assistant"})
    
    # Test upsert edge
    e1 = repo.upsert_edge(source_id=n2.id, target_id=n1.id, relation="ASSISTS", attributes={"since": 2008})
    
    # Test search
    graph = repo.search_graph("Tony Stark")
    print(f"Nodes: {[n['name'] for n in graph['query_nodes']]}")
    print(f"Related: {[n['name'] for n in graph['related_nodes']]}")
    print(f"Edges: {[e['relation'] for e in graph['edges']]}")

    await service.close()

asyncio.run(main())
