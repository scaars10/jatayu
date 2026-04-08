from __future__ import annotations

import json
import sqlite3
from typing import TYPE_CHECKING

from storage.models import KGEdgeRecord, KGNodeRecord

if TYPE_CHECKING:
    from storage.db import Database


class KnowledgeGraphRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def upsert_node(self, name: str, type: str, attributes: dict | None = None) -> KGNodeRecord:
        """Create or update a node and return its record."""
        attr_json = json.dumps(attributes) if attributes else None
        
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO kg_nodes (name, type, attributes)
                VALUES (?, ?, ?)
                ON CONFLICT(name, type) DO UPDATE SET
                    attributes = excluded.attributes,
                    updated_at = datetime('now')
                RETURNING id, name, type, attributes, created_at, updated_at
                """,
                (name, type, attr_json),
            )
            row = cursor.fetchone()
            return KGNodeRecord(**dict(row))

    def get_node_by_name(self, name: str) -> list[KGNodeRecord]:
        """Get all nodes matching a specific name."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, type, attributes, created_at, updated_at
                FROM kg_nodes
                WHERE name = ?
                """,
                (name,),
            )
            return [KGNodeRecord(**dict(row)) for row in cursor.fetchall()]

    def get_node_by_id(self, node_id: int) -> KGNodeRecord | None:
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, type, attributes, created_at, updated_at
                FROM kg_nodes
                WHERE id = ?
                """,
                (node_id,),
            )
            row = cursor.fetchone()
            if row:
                return KGNodeRecord(**dict(row))
            return None

    def upsert_edge(
        self, source_id: int, target_id: int, relation: str, attributes: dict | None = None
    ) -> KGEdgeRecord:
        """Create or update a directed edge between two nodes."""
        attr_json = json.dumps(attributes) if attributes else None
        
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                INSERT INTO kg_edges (source_node_id, target_node_id, relation, attributes)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(source_node_id, target_node_id, relation) DO UPDATE SET
                    attributes = excluded.attributes,
                    updated_at = datetime('now')
                RETURNING id, source_node_id, target_node_id, relation, attributes, created_at, updated_at
                """,
                (source_id, target_id, relation, attr_json),
            )
            row = cursor.fetchone()
            return KGEdgeRecord(**dict(row))

    def get_edges_for_node(self, node_id: int) -> list[KGEdgeRecord]:
        """Get all edges connected to a specific node (either as source or target)."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, source_node_id, target_node_id, relation, attributes, created_at, updated_at
                FROM kg_edges
                WHERE source_node_id = ? OR target_node_id = ?
                """,
                (node_id, node_id),
            )
            return [KGEdgeRecord(**dict(row)) for row in cursor.fetchall()]
    
    def list_all_nodes(self) -> list[KGNodeRecord]:
        """Get all nodes in the knowledge graph."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, name, type, attributes, created_at, updated_at
                FROM kg_nodes
                ORDER BY id ASC
                """
            )
            return [KGNodeRecord(**dict(row)) for row in cursor.fetchall()]

    def list_all_edges(self) -> list[KGEdgeRecord]:
        """Get all edges in the knowledge graph."""
        with self.db.connect() as conn:
            cursor = conn.execute(
                """
                SELECT id, source_node_id, target_node_id, relation, attributes, created_at, updated_at
                FROM kg_edges
                ORDER BY id ASC
                """
            )
            return [KGEdgeRecord(**dict(row)) for row in cursor.fetchall()]

    def delete_node(self, node_id: int) -> None:
        """Delete a node by its ID (cascades to edges)."""
        with self.db.connect() as conn:
            conn.execute("DELETE FROM kg_nodes WHERE id = ?", (node_id,))

    def reassign_edges(self, old_node_id: int, new_node_id: int) -> None:
        """Reassign all edges from old_node_id to new_node_id."""
        with self.db.connect() as conn:
            # Update edges where old_node_id was the source
            conn.execute(
                """
                UPDATE OR IGNORE kg_edges
                SET source_node_id = ?
                WHERE source_node_id = ?
                """,
                (new_node_id, old_node_id)
            )
            # Update edges where old_node_id was the target
            conn.execute(
                """
                UPDATE OR IGNORE kg_edges
                SET target_node_id = ?
                WHERE target_node_id = ?
                """,
                (new_node_id, old_node_id)
            )

    def search_graph(self, query_entity: str) -> dict:
        """Search for a specific entity and return its localized graph structure."""
        from dataclasses import asdict
        
        nodes = self.get_node_by_name(query_entity)
        if not nodes:
            return {}
            
        result = {"query_nodes": [], "related_nodes": [], "edges": []}
        seen_node_ids = set()
        
        for node in nodes:
            result["query_nodes"].append(asdict(node))
            seen_node_ids.add(node.id)
            edges = self.get_edges_for_node(node.id)
            for edge in edges:
                result["edges"].append(asdict(edge))
                other_id = edge.target_node_id if edge.source_node_id == node.id else edge.source_node_id
                if other_id not in seen_node_ids:
                    seen_node_ids.add(other_id)
                    other_node = self.get_node_by_id(other_id)
                    if other_node:
                        result["related_nodes"].append(asdict(other_node))
                        
        return result
