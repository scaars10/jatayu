import unittest

from search.searxng import InMemoryTTLCache, SearxngConfig, SearxngSearchResponse
from search.searxng.tool import SearxngSearchTool


class _StubClient:
    def __init__(self, response: SearxngSearchResponse) -> None:
        self.response = response
        self.calls = 0

    def search(self, request):  # noqa: ANN001
        self.calls += 1
        return self.response

    def close(self) -> None:
        return None


class SearxngSearchToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SearxngConfig(
            base_url="https://searx.example",
            max_results=10,
            cache_ttl_seconds=60,
        )

    def test_normalization_behavior(self) -> None:
        response = SearxngSearchResponse.model_validate(
            {
                "query": "jatayu",
                "number_of_results": 3,
                "results": [
                    {
                        "title": "  Example Result  ",
                        "url": "https://example.com/path/#fragment",
                        "content": "  useful    snippet  ",
                        "engines": ["duckduckgo"],
                        "category": "general",
                        "publishedDate": "2026-03-22",
                        "thumbnail_src": "https://example.com/thumb.jpg",
                    },
                    {
                        "title": "Duplicate",
                        "url": "https://example.com/path/",
                        "content": "duplicate entry should be removed",
                        "engine": "bing",
                    },
                    {
                        "title": "Second Result",
                        "url": "https://another.example/story",
                        "content": None,
                        "engine": "bing",
                        "favicon": "https://another.example/favicon.ico",
                    },
                ],
            }
        )
        tool = SearxngSearchTool(
            config=self.config,
            client=_StubClient(response),
            cache=InMemoryTTLCache(),
        )

        result = tool.search_web("jatayu", max_results=5, use_cache=False)

        self.assertEqual(result.total_results, 2)
        self.assertEqual(result.provider_total_results, 3)
        self.assertEqual(result.results[0].title, "Example Result")
        self.assertEqual(result.results[0].url, "https://example.com/path")
        self.assertEqual(result.results[0].snippet, "useful snippet")
        self.assertEqual(result.results[0].engine, "duckduckgo")
        self.assertEqual(result.results[0].source, "example.com")
        self.assertIsNotNone(result.results[0].published_date)
        self.assertEqual(result.results[1].favicon, "https://another.example/favicon.ico")

    def test_cache_hit_miss_behavior(self) -> None:
        response = SearxngSearchResponse.model_validate(
            {
                "query": "cache me",
                "number_of_results": 1,
                "results": [{"title": "Cached", "url": "https://example.com/cached"}],
            }
        )
        client = _StubClient(response)
        tool = SearxngSearchTool(
            config=self.config,
            client=client,
            cache=InMemoryTTLCache(),
        )

        first = tool.search_web("cache me", max_results=3, use_cache=True)
        second = tool.search_web("cache me", max_results=3, use_cache=True)
        third = tool.search_web("cache me", max_results=3, use_cache=False)

        self.assertFalse(first.cached)
        self.assertTrue(second.cached)
        self.assertFalse(third.cached)
        self.assertEqual(client.calls, 2)

    def test_search_web_brief_returns_summary_and_results(self) -> None:
        response = SearxngSearchResponse.model_validate(
            {
                "query": "brief me",
                "number_of_results": 1,
                "results": [
                    {
                        "title": "Brief Result",
                        "url": "https://example.com/brief",
                        "content": "A compact summary test result.",
                    }
                ],
            }
        )
        tool = SearxngSearchTool(
            config=self.config,
            client=_StubClient(response),
            cache=InMemoryTTLCache(),
        )

        brief = tool.search_web_brief("brief me", use_cache=False)

        self.assertIn("Top 1 SearXNG results", brief.summary)
        self.assertEqual(brief.result.results[0].title, "Brief Result")


if __name__ == "__main__":
    unittest.main()
