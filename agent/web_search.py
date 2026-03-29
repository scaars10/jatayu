from __future__ import annotations

import asyncio
import logging
from typing import Any

from google.genai import types

from search.searxng import SearxngSearchTool


WEB_SEARCH_FUNCTION_NAME = "search_web"
DEFAULT_AGENT_WEB_SEARCH_RESULTS = 5
MAX_WEB_SEARCH_TOOL_TURNS = 3
WEB_SEARCH_SYSTEM_INSTRUCTION = (
    "You have access to a live web search tool backed by SearXNG. "
    "Use it when the user asks for current, recent, changing, or externally verifiable information. "
    "Do not invent search results. "
    "When search is used, rely on the tool output and include the most relevant URLs when useful."
)


class ChatAgentWebSearch:
    def __init__(
        self,
        search_tool: SearxngSearchTool,
        *,
        max_results: int = DEFAULT_AGENT_WEB_SEARCH_RESULTS,
        owns_tool: bool = False,
        logger: logging.Logger | None = None,
    ) -> None:
        self.search_tool = search_tool
        self.max_results = max(1, min(max_results, search_tool.config.max_results))
        self._owns_tool = owns_tool
        self._logger = logger or logging.getLogger(__name__)
        self._tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=WEB_SEARCH_FUNCTION_NAME,
                    description=(
                        "Search the live web using SearXNG and return structured results. "
                        "Use this for up-to-date facts, recent news, websites, releases, "
                        "schedules, prices, or when external verification is needed."
                    ),
                    parameters_json_schema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The web search query.",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": (
                                    "Maximum number of results to return. "
                                    f"Defaults to {self.max_results}."
                                ),
                                "minimum": 1,
                                "maximum": search_tool.config.max_results,
                            },
                            "language": {
                                "type": "string",
                                "description": "Optional result language, for example 'en'.",
                            },
                            "categories": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional SearXNG categories to search.",
                            },
                            "engines": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Optional SearXNG engines to search.",
                            },
                        },
                        "required": ["query"],
                    },
                )
            ]
        )

    @property
    def function_name(self) -> str:
        return WEB_SEARCH_FUNCTION_NAME

    @property
    def generation_config(self) -> types.GenerateContentConfig:
        return types.GenerateContentConfig(
            tools=[self._tool],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
            system_instruction=WEB_SEARCH_SYSTEM_INSTRUCTION,
        )

    async def execute(self, arguments: dict[str, Any] | None) -> dict[str, Any]:
        query = self._coerce_query(arguments)
        if not query:
            return {"error": "Missing required argument: query"}

        try:
            brief = await asyncio.to_thread(
                self.search_tool.search_web_brief,
                query,
                max_results=self._coerce_max_results(arguments),
                categories=self._coerce_string_list(arguments, "categories"),
                engines=self._coerce_string_list(arguments, "engines"),
                language=self._coerce_string(arguments, "language"),
                use_cache=True,
            )
        except Exception as exc:
            self._logger.warning("SearXNG tool call failed: %s", exc)
            return {"error": f"SearXNG search failed: {exc}"}

        payload = brief.result.model_dump(mode="json")
        payload["summary"] = brief.summary
        payload["results"] = payload.get("results", [])
        payload["warnings"] = payload.get("warnings", [])
        return payload

    def close(self) -> None:
        if self._owns_tool:
            self.search_tool.close()

    def _coerce_max_results(self, arguments: dict[str, Any] | None) -> int:
        raw_value = (arguments or {}).get("max_results", self.max_results)
        try:
            return max(1, min(int(raw_value), self.search_tool.config.max_results))
        except (TypeError, ValueError):
            return self.max_results

    @staticmethod
    def _coerce_query(arguments: dict[str, Any] | None) -> str | None:
        raw_value = (arguments or {}).get("query")
        if not isinstance(raw_value, str):
            return None
        normalized = " ".join(raw_value.split())
        return normalized or None

    @staticmethod
    def _coerce_string(
        arguments: dict[str, Any] | None,
        key: str,
    ) -> str | None:
        raw_value = (arguments or {}).get(key)
        if not isinstance(raw_value, str):
            return None
        normalized = raw_value.strip()
        return normalized or None

    @staticmethod
    def _coerce_string_list(
        arguments: dict[str, Any] | None,
        key: str,
    ) -> list[str] | None:
        raw_value = (arguments or {}).get(key)
        if raw_value is None:
            return None
        if not isinstance(raw_value, list):
            return None

        values = [item.strip() for item in raw_value if isinstance(item, str) and item.strip()]
        return values or None
