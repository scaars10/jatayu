CREATE TABLE IF NOT EXISTS research_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    topic TEXT NOT NULL,
    specific_questions TEXT,
    status TEXT NOT NULL DEFAULT 'pending',
    report TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
