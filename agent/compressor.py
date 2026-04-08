import json
import logging
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field

from agent.gemini_model import gemini_model
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.providers.google import GoogleProvider
from config.env_config import get_env
from storage import StorageService

logger = logging.getLogger(__name__)

COMPRESSION_STATE_FILE = Path("data/storage/compression_state.json")

class CompressionState:
    def __init__(self):
        self.updates_count = 0
        self.last_run_time = 0.0
        self.load()

    def load(self):
        if COMPRESSION_STATE_FILE.exists():
            try:
                with open(COMPRESSION_STATE_FILE, "r") as f:
                    data = json.load(f)
                    self.updates_count = data.get("updates_count", 0)
                    self.last_run_time = data.get("last_run_time", 0.0)
            except Exception as e:
                logger.error(f"Failed to load compression state: {e}")

    def save(self):
        try:
            COMPRESSION_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(COMPRESSION_STATE_FILE, "w") as f:
                json.dump({"updates_count": self.updates_count, "last_run_time": self.last_run_time}, f)
        except Exception as e:
            logger.error(f"Failed to save compression state: {e}")

    def increment_updates(self, count=1):
        self.updates_count += count
        self.save()

    def reset_after_run(self, time_now: float):
        self.updates_count = 0
        self.last_run_time = time_now
        self.save()

global_compression_state = CompressionState()

class NewMemory(BaseModel):
    model_config = ConfigDict(extra="forbid")
    memory_key: str
    category: str = "general"
    summary: str
    importance: Literal["low", "medium", "high"] = "medium"
    confidence: float = 0.9

class MemoryCompressionUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    scope_type: str = Field(description="participant or channel")
    scope_key: str = Field(description="The scope key")
    memories_to_archive: list[str] = Field(description="List of memory_keys to archive because they are superseded.")
    new_memories: list[NewMemory] = Field(description="List of new consolidated memories.")

class KGMergeAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    target_node_id: int = Field(description="The ID of the node to keep.")
    nodes_to_merge: list[int] = Field(description="List of node IDs that should be merged into the target node and then deleted.")

class CompressorResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    memory_updates: list[MemoryCompressionUpdate] = Field(default_factory=list)
    kg_merges: list[KGMergeAction] = Field(default_factory=list)
    kg_nodes_to_delete: list[int] = Field(default_factory=list, description="List of node IDs to delete because they are irrelevant or junk.")

COMPRESSOR_PROMPT = """
You are the System Optimizer Agent for a Personal AI Assistant.
Your task is to analyze the long-term memories and the knowledge graph to find redundancies, duplicates, and opportunities for compression and cleaning.

1. LONG-TERM MEMORIES:
- CONSOLIDATION: Look for memories that cover the same topic, are outdated, or can be consolidated into a single, richer memory. REUSE the best existing memory_key.
- PRUNING (CRITICAL): Identify and archive "junk" memories. 
  * If a memory is just a fact from a research report (e.g., apartment listings, technical docs, news, generic facts) and DOES NOT directly relate to the user's personal context, preferences, or explicitly discussed personal projects, ARCHIVE IT.
  * If the user just asked "What is the capital of France?" and the assistant answered, DO NOT keep a memory of that unless the user said "I'm moving to Paris".
  * If a memory is a duplicate or a slightly different version of another, archive the redundant one.

2. KNOWLEDGE GRAPH:
- MERGING: Look for duplicate entities (e.g., "Google" vs "Google Inc.", or "React" vs "ReactJS") that refer to the exact same concept. Choose the best canonical node to keep (target_node_id).
- PRUNING (CRITICAL): Identify and suggest DELETION of nodes that are "research noise".
  * If a node is a specific apartment address from a search result, or a generic technology name that was just mentioned once in a research report, and has no personal link to the user, DELETE IT.
  * Look for nodes with very few edges that seem to be one-off research details.
"""

class Compressor:
    def __init__(self, storage_service: StorageService):
        self.storage_service = storage_service
        api_key = get_env("GEMINI_API_KEY", required=True)
        self.agent = Agent(
            model=GoogleModel(
                gemini_model.get_large_model(),
                provider=GoogleProvider(api_key=api_key),
            ),
            system_prompt=COMPRESSOR_PROMPT,
            output_type=CompressorResult,
        )

    async def run_compression(self):
        """Analyze and compress memories and knowledge graph."""
        logger.info("Starting compression cycle...")
        
        await self.storage_service.start()
        
        # 1. Fetch data
        memories = self.storage_service.long_term_memories.list_all_active()
        kg_repo = self.storage_service.knowledge_graph
        kg_nodes = kg_repo.list_all_nodes() if kg_repo else []
        kg_edges = kg_repo.list_all_edges() if kg_repo else []
        
        if not memories and not kg_nodes:
            logger.info("Nothing to compress.")
            return

        # 2. Build Prompt
        prompt = "### CURRENT LONG-TERM MEMORIES ###\n"
        for mem in memories:
            prompt += f"ID: {mem.id} | Scope: {mem.scope_type}/{mem.scope_key} | Key: {mem.memory_key} | Category: {mem.category} | Summary: {mem.summary}\n"
            
        prompt += "\n### CURRENT KNOWLEDGE GRAPH NODES ###\n"
        for node in kg_nodes:
            prompt += f"ID: {node.id} | Name: {node.name} | Type: {node.type} | Attributes: {node.attributes}\n"

        prompt += "\n### CURRENT KNOWLEDGE GRAPH EDGES ###\n"
        for edge in kg_edges:
            prompt += f"Source ID: {edge.source_node_id} --({edge.relation})--> Target ID: {edge.target_node_id}\n"
            
        prompt += "\nPlease analyze the above and provide compression operations. Be AGGRESSIVE in pruning research data and redundant entries."
        
        # 3. Run Agent
        try:
            result = await self.agent.run(prompt)
            output: CompressorResult = getattr(result, "data", getattr(result, "output", None))
            
            if not output:
                logger.warning("No output from compression agent.")
                return
                
            # 4. Apply Memory Updates
            for update in output.memory_updates:
                for key_to_archive in update.memories_to_archive:
                    self.storage_service.long_term_memories.archive(
                        scope_type=update.scope_type,
                        scope_key=update.scope_key,
                        memory_key=key_to_archive
                    )
                    logger.info(f"Archived memory: {key_to_archive}")
                    
                for new_mem in update.new_memories:
                    self.storage_service.long_term_memories.upsert(
                        scope_type=update.scope_type,
                        scope_key=update.scope_key,
                        channel_id=None,
                        participant_id=None,
                        memory_key=new_mem.memory_key,
                        category=new_mem.category,
                        summary=new_mem.summary,
                        importance=new_mem.importance,
                        confidence=new_mem.confidence,
                        source_message_id=None,
                        last_observed_at=datetime.now(timezone.utc),
                    )
                    logger.info(f"Created consolidated memory: {new_mem.memory_key}")
                    
            # 5. Apply KG Updates
            if kg_repo:
                # Handle Merges
                for merge in output.kg_merges:
                    for old_node_id in merge.nodes_to_merge:
                        kg_repo.reassign_edges(old_node_id, merge.target_node_id)
                        kg_repo.delete_node(old_node_id)
                        logger.info(f"Merged KG node {old_node_id} into {merge.target_node_id}")
                
                # Handle Deletions
                for node_id in output.kg_nodes_to_delete:
                    kg_repo.delete_node(node_id)
                    logger.info(f"Deleted junk KG node {node_id}")
                        
        except Exception as e:
            logger.error(f"Compression failed: {e}")
                        
        except Exception as e:
            logger.error(f"Compression failed: {e}")

async def compression_loop(storage_service: StorageService):
    """Background loop that periodically runs the compressor when idle and needed."""
    logger.info("Starting compression background loop.")
    compressor = Compressor(storage_service)
    
    check_interval = 60 * 10 # 10 minutes
    updates_threshold = 10 # Only run if there are at least 10 updates
    
    # Run once on startup if it has never run before
    if global_compression_state.last_run_time == 0.0:
        logger.info("Triggering initial compression on first startup to process any pre-existing data.")
        try:
            await compressor.run_compression()
            global_compression_state.reset_after_run(datetime.now(timezone.utc).timestamp())
        except Exception as e:
            logger.error(f"Error in initial compression: {e}")

    while True:
        try:
            await asyncio.sleep(check_interval)
            
            now = datetime.now(timezone.utc).timestamp()
            
            if global_compression_state.updates_count >= updates_threshold:
                logger.info(f"Triggering compression: {global_compression_state.updates_count} updates since last run.")
                await compressor.run_compression()
                global_compression_state.reset_after_run(now)
            else:
                time_since_last_run = now - global_compression_state.last_run_time
                if time_since_last_run > (24 * 3600) and global_compression_state.updates_count > 0:
                    logger.info("Triggering compression: Daily idle run.")
                    await compressor.run_compression()
                    global_compression_state.reset_after_run(now)
                    
        except Exception as e:
            logger.error(f"Error in compression loop: {e}")
