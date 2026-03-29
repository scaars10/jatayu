CREATE TABLE IF NOT EXISTS channels (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    channel_type TEXT,
    title TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider, external_id)
);

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    provider TEXT NOT NULL,
    external_id TEXT NOT NULL,
    display_name TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(provider, external_id)
);

CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    channel_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active',
    summary TEXT,
    started_at TEXT NOT NULL,
    last_message_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_conversations_active_channel
ON conversations(channel_id)
WHERE status = 'active';

CREATE TABLE IF NOT EXISTS conversation_participants (
    conversation_id INTEGER NOT NULL,
    participant_id INTEGER NOT NULL,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (conversation_id, participant_id),
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY(participant_id) REFERENCES participants(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_id TEXT NOT NULL UNIQUE,
    conversation_id INTEGER NOT NULL,
    channel_id INTEGER NOT NULL,
    participant_id INTEGER,
    source TEXT NOT NULL,
    direction TEXT NOT NULL CHECK(direction IN ('inbound', 'outbound')),
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    message_type TEXT NOT NULL,
    provider_message_id TEXT,
    reply_to_provider_message_id TEXT,
    request_event_id TEXT,
    text_content TEXT,
    payload_json TEXT NOT NULL,
    delivery_status TEXT NOT NULL CHECK(delivery_status IN ('received', 'pending', 'delivered', 'failed')),
    occurred_at TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY(conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    FOREIGN KEY(channel_id) REFERENCES channels(id) ON DELETE CASCADE,
    FOREIGN KEY(participant_id) REFERENCES participants(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_occurred_at
ON messages(conversation_id, occurred_at, id);

CREATE INDEX IF NOT EXISTS idx_messages_channel_provider_message
ON messages(channel_id, provider_message_id);

CREATE INDEX IF NOT EXISTS idx_messages_request_event_id
ON messages(request_event_id);

CREATE TABLE IF NOT EXISTS message_attachments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id INTEGER NOT NULL,
    media_type TEXT NOT NULL,
    provider_file_id TEXT NOT NULL,
    provider_file_unique_id TEXT NOT NULL,
    mime_type TEXT,
    file_name TEXT,
    width INTEGER,
    height INTEGER,
    duration_seconds INTEGER,
    file_size INTEGER,
    performer TEXT,
    title TEXT,
    caption TEXT,
    media_group_id TEXT,
    payload_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY(message_id) REFERENCES messages(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_message_attachments_message_id
ON message_attachments(message_id);
