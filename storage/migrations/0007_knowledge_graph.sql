-- Migration 0007: Knowledge Graph (Nodes and Edges)

CREATE TABLE IF NOT EXISTS kg_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    type TEXT NOT NULL, -- e.g., 'PERSON', 'PROJECT', 'CONCEPT'
    attributes TEXT, -- JSON blob for extra details
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    UNIQUE(name, type)
);

CREATE TABLE IF NOT EXISTS kg_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_node_id INTEGER NOT NULL,
    target_node_id INTEGER NOT NULL,
    relation TEXT NOT NULL, -- e.g., 'WORKS_ON', 'IS_RELATED_TO'
    attributes TEXT, -- JSON blob for extra details like context/confidence
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%S', 'now')),
    FOREIGN KEY(source_node_id) REFERENCES kg_nodes(id) ON DELETE CASCADE,
    FOREIGN KEY(target_node_id) REFERENCES kg_nodes(id) ON DELETE CASCADE,
    UNIQUE(source_node_id, target_node_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_kg_nodes_name ON kg_nodes(name);
CREATE INDEX IF NOT EXISTS idx_kg_edges_source ON kg_edges(source_node_id);
CREATE INDEX IF NOT EXISTS idx_kg_edges_target ON kg_edges(target_node_id);
