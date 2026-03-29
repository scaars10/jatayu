from __future__ import annotations

from google.genai import types

# Add new tools to this list as you expand the agent's capabilities
TOOLS: list[types.Tool] = [
    types.Tool(google_search=types.GoogleSearch()),
]

# Update this static instruction when adding new tools
SYSTEM_INSTRUCTION = """\
You have access to Google Search to provide up-to-date and accurate information. Use it when the user asks for current, recent, changing, or externally verifiable information. When search is used, rely on the provided context and include the most relevant URLs when useful.\
"""

def get_generation_config() -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        tools=TOOLS,
        system_instruction=SYSTEM_INSTRUCTION,
    )
