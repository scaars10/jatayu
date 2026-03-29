import unittest

import httpx

from search.searxng import (
    SearxngClient,
    SearxngConfig,
    SearxngParsingError,
)


class SearxngClientTests(unittest.TestCase):
    def _make_client(
        self,
        handler,
        *,
        retry_count: int = 0,
        sleep_calls: list[float] | None = None,
    ) -> tuple[SearxngClient, httpx.Client]:
        transport = httpx.MockTransport(handler)
        http_client = httpx.Client(
            transport=transport,
            base_url="https://searx.example",
            headers={"User-Agent": "tests"},
            timeout=1.0,
        )
        config = SearxngConfig(
            base_url="https://searx.example",
            retry_count=retry_count,
            backoff_seconds=0.25,
        )
        client = SearxngClient(
            config=config,
            http_client=http_client,
            sleep_fn=(sleep_calls.append if sleep_calls is not None else None),
        )
        return client, http_client

    def test_successful_json_response_parsing(self) -> None:
        captured: dict[str, str] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/search")
            captured.update(dict(request.url.params))
            return httpx.Response(
                200,
                json={
                    "query": "jatayu search",
                    "number_of_results": 1,
                    "results": [
                        {
                            "title": "Example result",
                            "url": "https://example.com/article",
                            "content": "Useful snippet",
                            "engine": "duckduckgo",
                            "category": "general",
                            "publishedDate": "2026-03-22T10:15:00Z",
                        }
                    ],
                },
            )

        client, http_client = self._make_client(handler)
        try:
            response = client.search(
                "jatayu search",
                page=2,
                max_results=3,
                language="en",
                categories=["general", "news"],
                engines=["duckduckgo"],
                safe_search=2,
            )
        finally:
            http_client.close()

        self.assertEqual(captured["q"], "jatayu search")
        self.assertEqual(captured["format"], "json")
        self.assertEqual(captured["language"], "en")
        self.assertEqual(captured["categories"], "general,news")
        self.assertEqual(captured["engines"], "duckduckgo")
        self.assertEqual(captured["safesearch"], "2")
        self.assertEqual(captured["pageno"], "2")
        self.assertEqual(response.query, "jatayu search")
        self.assertEqual(response.number_of_results, 1)
        self.assertEqual(response.results[0].title, "Example result")
        self.assertEqual(response.results[0].url, "https://example.com/article")

    def test_timeout_retry_behavior(self) -> None:
        attempts = {"count": 0}
        sleep_calls: list[float] = []

        def handler(request: httpx.Request) -> httpx.Response:
            attempts["count"] += 1
            if attempts["count"] == 1:
                raise httpx.ReadTimeout("timed out", request=request)
            return httpx.Response(
                200,
                json={
                    "query": "retry me",
                    "number_of_results": 1,
                    "results": [{"title": "Recovered", "url": "https://example.com/recovered"}],
                },
            )

        client, http_client = self._make_client(
            handler,
            retry_count=1,
            sleep_calls=sleep_calls,
        )
        try:
            response = client.search("retry me")
        finally:
            http_client.close()

        self.assertEqual(attempts["count"], 2)
        self.assertEqual(sleep_calls, [0.25])
        self.assertEqual(response.results[0].title, "Recovered")

    def test_malformed_response_handling(self) -> None:
        def handler(_: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                content=b"{this is not valid json",
                headers={"Content-Type": "application/json"},
            )

        client, http_client = self._make_client(handler)
        try:
            with self.assertRaises(SearxngParsingError):
                client.search("bad json")
        finally:
            http_client.close()


if __name__ == "__main__":
    unittest.main()
