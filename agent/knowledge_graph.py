from __future__ import annotations

import json
import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider

from agent.gemini_model import gemini_model
from agent.compressor import global_compression_state
from config.env_config import get_env
from storage import ConversationTurn, StorageService

logger = logging.getLogger(__name__)

# System prompt for the Knowledge Graph Extraction Agent
KG_EXTRACTION_PROMPT = """
You are a Proactive Personal Knowledge Graph builder.
Your task is to analyze the conversation and extract important ENTITIES and their RELATIONSHIPS.
This acts as a "Second Brain" for the user, allowing the AI to remember long-term context.

Guidelines:
1. ENTITIES: Extract people, places, organizations, projects, concepts, preferences, tools, technologies.
   - Types should be concise (e.g., PERSON, PROJECT, TECHNOLOGY, CONCEPT, PREFERENCE).
   - Use canonical names (e.g., if user says "my friend John", name="John", type="PERSON").
2. RELATIONSHIPS: Define how entities relate to each other (e.g., LIKES, WORKS_ON, IS_PART_OF, EXPERIENCED_ISSUE).
3. DURABILITY: Only extract things that are durable and matter over the long term. Do not extract trivial small talk or transient states.
4. PERSONAL FOCUS: ONLY extract information that the USER has shared about themselves, their preferences, their life, or their projects. 
5. IGNORE RESEARCH: DO NOT extract information from research reports, web search results, or general knowledge that the assistant has provided (e.g., apartment listings, technical documentation, news summaries) unless the user explicitly adopts it or it relates directly to the user's personal context.
6. Keep attributes minimal but useful (e.g., {"role": "developer"}).

Extract the data strictly according to the requested schema.
"""

class EntityCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(description="The canonical name of the entity.")
    type: str = Field(description="The type of the entity (e.g., PERSON, PROJECT, CONCEPT).")
    attributes: dict[str, str] = Field(default_factory=dict, description="Key-value pairs for extra details.")

class RelationCandidate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: str = Field(description="Name of the source entity.")
    target: str = Field(description="Name of the target entity.")
    relation: str = Field(description="The relationship (e.g., WORKS_ON, LIKES).")
    attributes: dict[str, str] = Field(default_factory=dict, description="Key-value pairs for extra context.")

class GraphExtractionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entities: list[EntityCandidate] = Field(default_factory=list)
    relationships: list[RelationCandidate] = Field(default_factory=list)

class KnowledgeGraphProcessor:
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        api_key = get_env("GEMINI_API_KEY", required=True)
        self.agent = Agent(
            model=GoogleModel(
                gemini_model.get_balanced_model(),
                provider=GoogleProvider(api_key=api_key),
            ),
            system_prompt=KG_EXTRACTION_PROMPT,
            output_type=GraphExtractionResult,
        )

    async def extract_and_store(self, user_message: str, history: list[ConversationTurn]):
        """Analyze the latest interaction and store entities/relations in the KG."""
        await self.storage_service.start()
        repo = self.storage_service.knowledge_graph
        
        if not repo:
            logger.error("Knowledge Graph Repository not initialized.")
            return

        # Try to find relevant existing context to avoid duplicates
        # We can do a simple keyword extraction or just pass a few recent nodes
        # For now, let's just fetch all nodes if the graph is small, or a subset.
        # Given it's a personal graph, it shouldn't be massive yet.
        existing_nodes = repo.list_all_nodes()
        
        # Build prompt
        lines = ["### EXISTING KNOWLEDGE GRAPH NODES (Use these names if they match) ###"]
        if existing_nodes:
            for node in existing_nodes[-20:]: # Last 20 for some context
                lines.append(f"- {node.name} ({node.type})")
        else:
            lines.append("No existing nodes.")
            
        lines.append("\n### RECENT CONVERSATION ###")
        if history:
            for turn in history[-5:]:
                lines.append(f"{turn.role.upper()}: {turn.text}")
        lines.append(f"USER: {user_message}")
        lines.append("\nAnalyze the above conversation and extract the knowledge graph entities and relationships.")
        lines.append("IMPORTANT: If an entity already exists in the list above, use its EXACT name. Do not create 'React' if 'ReactJS' already exists and refers to the same thing.")
        
        prompt = "\n".join(lines)
        
        try:
            result = await self.agent.run(prompt, message_history=[])
            extracted = getattr(result, 'data', getattr(result, 'output', None))

            if not extracted:
                return

            if not extracted.entities and not extracted.relationships:
                return

            # Insert nodes
            node_map = {}
            # Pre-populate node_map with existing nodes to ensure we have IDs for relationships
            # even if they weren't explicitly re-extracted as entities in this turn
            for node in existing_nodes:
                node_map[node.name] = node.id

            for entity in extracted.entities:
                record = repo.upsert_node(
                    name=entity.name, 
                    type=entity.type.upper(), 
                    attributes=entity.attributes
                )
                node_map[entity.name] = record.id
                
            # Insert edges
            for rel in extracted.relationships:
                source_id = node_map.get(rel.source)
                target_id = node_map.get(rel.target)
                
                # If we don't have the node explicitly created, we might skip or create on the fly.
                # Here we skip if it wasn't extracted as an entity or didn't exist.
                if source_id and target_id:
                    repo.upsert_edge(
                        source_id=source_id,
                        target_id=target_id,
                        relation=rel.relation.upper(),
                        attributes=rel.attributes
                    )
                    
            logger.info(f"Knowledge Graph Updated: {len(extracted.entities)} entities, {len(extracted.relationships)} edges.")
            global_compression_state.increment_updates(len(extracted.entities) + len(extracted.relationships))
            
        except Exception as e:
            logger.error(f"Failed to process knowledge graph extraction: {e}")
