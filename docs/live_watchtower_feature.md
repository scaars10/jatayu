# Live Data Stream Monitoring (Real-Time Watchtower)

## Overview
Jarvis doesn't just wait for questions; he monitors the world and alerts Tony Stark when something requires his attention. This feature allows the AI to subscribe to live data streams (RSS, Twitter/X lists, GitHub activity, server logs, crypto/stock tickers) and process them in real-time.

## Key Capabilities
- **Custom Alert Triggers**: "Watch the HackerNews feed. If an article about a new Rust web framework hits the front page, ping me."
- **Server/System Monitoring**: "Monitor my AWS error logs. If you see a spike in 500 errors, wake me up and summarize the stack trace."
- **Market/Trend Watcher**: The AI watches crypto or stock prices and correlates them with breaking news, providing a synthesized brief when significant movement happens.
- **Proactive Action**: If a server goes down, the AI doesn't just alert you; it can proactively run diagnostics (e.g., pinging the server, checking database loads) and present you with the findings.

## Implementation Plan

### Phase 1: Stream Ingestion Engine
1. **Webhooks & Polling**: Create an asynchronous worker that polls RSS feeds, REST APIs, or listens to webhooks.
2. **Data Normalization**: Convert incoming unstructured stream data into a standardized JSON format for the LLM to evaluate.

### Phase 2: LLM Evaluation Loop
1. **Filter Prompts**: Feed the incoming data through a lightweight, fast LLM (like Gemini Flash) with the user's active "Watch Directives".
2. **Relevance Scoring**: If the LLM scores the data's relevance above a threshold, it triggers an alert.

### Phase 3: Automated Diagnostics
1. **Action Chains**: Link specific alerts to predefined action chains. (e.g., Alert: GitHub PR opened -> Action: AI reads the diff, summarizes the impact, and sends a Telegram message).
2. **Tool Creation**: `add_watch_directive(source, condition)`, `remove_watch_directive(id)`.