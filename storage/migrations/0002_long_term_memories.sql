CREATE TABLE IF NOT EXISTS long_term_memories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    scope_type TEXT NOT NULL CHECK(scope_type IN ('channel', 'participant')),
    scope_key TEXT NOT NULL,
    channel_id INTEGER,
    participant_id INTEGER,
    memory_key TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    importance TEXT NOT NULL CHECK(importance IN ('low', 'medium', 'high')),
    confidence REAL NOT NULL,
    source_message_id INTEGER,
    status TEXT NOT NULL CHECK(status IN ('active', 'archived')),
    last_observed_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(scope_type, scope_key, memory_key),
    FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE,
    FOREIGN KEY(participant_id) REFERENCES participants(id) ON DELETE CASCADE,
    FOREIGN KEY(source_message_id) REFERENCES messages(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_long_term_memories_scope
ON long_term_memories(scope_type, scope_key, status);

CREATE INDEX IF NOT EXISTS idx_long_term_memories_channel
ON long_term_memories(channel_id, status);

CREATE INDEX IF NOT EXISTS idx_long_term_memories_participant
ON long_term_memories(participant_id, status);
