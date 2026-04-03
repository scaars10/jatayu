# Proactive Personal Knowledge Graph (Second Brain)

## Overview
A true friend remembers the details. Rather than relying on a flat database of chat history, this feature builds a dynamic, interconnected Knowledge Graph of your life, ideas, projects, and preferences. It acts as an active "Second Brain" that connects the dots between your thoughts over time.

## Key Capabilities
- **Implicit Memory Capture**: As you talk casually, the AI extracts entities (people, projects, concepts) and their relationships, storing them silently.
- **Serendipitous Connections**: "You know, this bug you're fixing in the payment gateway sounds exactly like the race condition you ranted about in the authentication service three months ago."
- **Idea Incubation**: You can dump fragmented ideas into the chat. The AI categorizes them and proactively brings them up later when relevant.
- **Deep Querying**: "What were the names of those three books on architecture that Sarah recommended to me last year?"

## Implementation Plan

### Phase 1: Entity Extraction Pipeline
1. **Graph Database**: Set up a lightweight graph database like Neo4j or an embedded vector-graph hybrid (like Kùzu).
2. **Background Processor**: Create an asynchronous pipeline that runs after every conversation to extract nodes (Entities) and edges (Relationships) using an LLM.

### Phase 2: Contextual Retrieval
1. **Graph RAG (Retrieval-Augmented Generation)**: When the user asks a question, query both the vector store and the graph database. Traverse the graph to find connected concepts.
2. **Tool Creation**: `query_knowledge_graph(entity)`, `add_graph_relation(entity_a, relation, entity_b)`.

### Phase 3: Proactive Synthesis
1. **Thought Collision**: Implement a daily background job where the AI looks at recent inputs and searches the graph for distant but relevant past ideas.
2. **Unprompted Insights**: The AI spontaneously starts a conversation: "I was reviewing your notes on the new app UI, and it conflicts with the accessibility goals you set in January. Want to discuss?"