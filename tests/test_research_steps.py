from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import agent.research_steps as research_steps


def _make_gemini_response(answer: str, sources: list[tuple[str, str]]) -> SimpleNamespace:
    chunks = [
        SimpleNamespace(web=SimpleNamespace(title=title, uri=url))
        for title, url in sources
    ]
    candidate = SimpleNamespace(
        content=SimpleNamespace(parts=[SimpleNamespace(text=answer)]),
        grounding_metadata=SimpleNamespace(grounding_chunks=chunks),
    )
    return SimpleNamespace(candidates=[candidate])


def _make_gemini_multi_candidate_response(
    candidate_payloads: list[tuple[str, list[tuple[str, str]]]],
) -> SimpleNamespace:
    candidates = []
    for answer, sources in candidate_payloads:
        chunks = [
            SimpleNamespace(web=SimpleNamespace(title=title, uri=url))
            for title, url in sources
        ]
        candidates.append(
            SimpleNamespace(
                content=SimpleNamespace(parts=[SimpleNamespace(text=answer)]),
                grounding_metadata=SimpleNamespace(grounding_chunks=chunks),
            )
        )
    return SimpleNamespace(candidates=candidates)


class ResearchStepsSearchTests(unittest.IsolatedAsyncioTestCase):
    def test_build_gather_sources_prompt_requires_search_quality_evaluation(self) -> None:
        prompt = research_steps._build_gather_sources_prompt(
            "AI safety",
            ["What changed recently?", "Which sources are primary?"],
        )

        self.assertIn("Evaluate each search result before selecting it.", prompt)
        self.assertIn("If the current search results do not satisfy the criteria", prompt)
        self.assertIn("Relevance:", prompt)
        self.assertIn("Credibility:", prompt)
        self.assertIn("Recency:", prompt)
        self.assertIn("Coverage:", prompt)
        self.assertIn("Diversity:", prompt)
        self.assertIn("What changed recently?", prompt)
        self.assertIn("Which sources are primary?", prompt)
        self.assertIn("Return ONLY a JSON list of strings", prompt)

    async def test_search_with_gemini_retries_retryable_errors_before_success(self) -> None:
        response = _make_gemini_response(
            "Latest mission update",
            [("Gemini Story", "https://example.com/article?utm_source=newsletter")],
        )
        generate_content = AsyncMock(
            side_effect=[
                RuntimeError("503 UNAVAILABLE"),
                RuntimeError("temporarily unavailable"),
                response,
            ]
        )
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content),
            )
        )
        sleep_mock = AsyncMock()

        with patch("agent.gemini_model.get_client", return_value=fake_client), patch.object(
            research_steps,
            "_gemini_search_retry_count",
            return_value=3,
        ), patch.object(
            research_steps,
            "_gemini_search_retry_delay",
            return_value=0.25,
        ), patch.object(
            research_steps.asyncio,
            "sleep",
            sleep_mock,
        ):
            result = await research_steps._search_with_gemini("mars mission", max_results=1)

        self.assertEqual(generate_content.await_count, 3)
        self.assertEqual(sleep_mock.await_count, 2)
        self.assertEqual(result.answer, "Latest mission update")
        self.assertEqual(result.candidates[0].title, "Gemini Story")
        self.assertEqual(result.candidates[0].url, "https://example.com/article")

    async def test_search_with_gemini_keeps_retrying_on_503_until_success(self) -> None:
        response = _make_gemini_response(
            "Recovered answer",
            [("Gemini Story", "https://example.com/article")],
        )
        generate_content = AsyncMock(
            side_effect=[
                RuntimeError("503 UNAVAILABLE"),
                RuntimeError("503 backend overloaded"),
                response,
            ]
        )
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content),
            )
        )
        sleep_mock = AsyncMock()

        with patch("agent.gemini_model.get_client", return_value=fake_client), patch.object(
            research_steps,
            "_gemini_search_retry_count",
            return_value=1,
        ), patch.object(
            research_steps,
            "_gemini_search_503_min_retry_seconds",
            return_value=300.0,
        ), patch.object(
            research_steps,
            "_gemini_search_retry_delay",
            return_value=0.25,
        ), patch.object(
            research_steps.asyncio,
            "sleep",
            sleep_mock,
        ), patch.object(
            research_steps.time,
            "monotonic",
            side_effect=[0.0, 1.0, 2.0],
        ):
            result = await research_steps._search_with_gemini("mars mission", max_results=1)

        self.assertEqual(generate_content.await_count, 3)
        self.assertEqual(sleep_mock.await_count, 2)
        self.assertEqual(result.answer, "Recovered answer")
        self.assertEqual(result.candidates[0].title, "Gemini Story")

    async def test_search_with_gemini_stops_retrying_503_after_minimum_window(self) -> None:
        generate_content = AsyncMock(side_effect=RuntimeError("503 UNAVAILABLE"))
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content),
            )
        )
        sleep_mock = AsyncMock()

        with patch("agent.gemini_model.get_client", return_value=fake_client), patch.object(
            research_steps,
            "_gemini_search_retry_count",
            return_value=1,
        ), patch.object(
            research_steps,
            "_gemini_search_503_min_retry_seconds",
            return_value=300.0,
        ), patch.object(
            research_steps,
            "_gemini_search_retry_delay",
            return_value=0.25,
        ), patch.object(
            research_steps.asyncio,
            "sleep",
            sleep_mock,
        ), patch.object(
            research_steps.time,
            "monotonic",
            side_effect=[0.0, 300.0],
        ):
            with self.assertRaisesRegex(RuntimeError, "503 UNAVAILABLE"):
                await research_steps._search_with_gemini("mars mission", max_results=1)

        self.assertEqual(generate_content.await_count, 1)
        sleep_mock.assert_not_awaited()

    async def test_search_with_gemini_reads_all_grounded_results(self) -> None:
        response = _make_gemini_multi_candidate_response(
            [
                (
                    "Latest mission update",
                    [
                        ("Gemini Story", "https://example.com/article?utm_source=newsletter"),
                        ("Mission Brief", "https://example.com/brief"),
                    ],
                ),
                (
                    "Ignored second answer",
                    [
                        ("Launch Update", "https://updates.example.com/launch"),
                        ("Gemini Story", "https://example.com/article"),
                    ],
                ),
            ]
        )
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=AsyncMock(return_value=response)),
            )
        )

        with patch("agent.gemini_model.get_client", return_value=fake_client):
            result = await research_steps._search_with_gemini("mars mission", max_results=None)

        self.assertEqual(result.answer, "Latest mission update")
        self.assertEqual(
            [candidate.title for candidate in result.candidates],
            ["Gemini Story", "Mission Brief", "Launch Update"],
        )
        self.assertEqual(
            [candidate.url for candidate in result.candidates],
            [
                "https://example.com/article",
                "https://example.com/brief",
                "https://updates.example.com/launch",
            ],
        )

    async def test_search_with_gemini_uses_large_model(self) -> None:
        response = _make_gemini_response(
            "Latest mission update",
            [("Gemini Story", "https://example.com/article")],
        )
        generate_content = AsyncMock(return_value=response)
        fake_client = SimpleNamespace(
            aio=SimpleNamespace(
                models=SimpleNamespace(generate_content=generate_content),
            )
        )

        with patch("agent.gemini_model.get_client", return_value=fake_client):
            await research_steps._search_with_gemini("mars mission", max_results=1)

        self.assertEqual(
            generate_content.await_args.kwargs["model"],
            research_steps.gemini_model.get_large_model(),
        )

    async def test_search_web_candidates_does_not_fall_back_by_default(self) -> None:
        with patch.object(
            research_steps,
            "_search_with_gemini",
            side_effect=RuntimeError("503 unavailable"),
        ), patch.object(
            research_steps,
            "_search_with_searxng",
            new=AsyncMock(),
        ) as searxng_search, patch.object(
            research_steps,
            "_search_with_duckduckgo",
            new=AsyncMock(),
        ) as duckduckgo_search:
            result = await research_steps.search_web_candidates("mars mission", max_results=1)

        self.assertEqual(result.candidates, [])
        self.assertIn("gemini", result.providers_tried)
        self.assertTrue(any("Gemini search failed" in warning for warning in result.warnings))
        searxng_search.assert_not_awaited()
        duckduckgo_search.assert_not_awaited()

    async def test_search_web_candidates_falls_back_when_enabled(self) -> None:
        searxng_result = research_steps.WebSearchResult(
            query="mars mission",
            candidates=[
                research_steps.WebSearchCandidate(
                    title="Fallback Story",
                    url="https://fallback.example/story",
                    provider="searxng",
                    position=1,
                )
            ],
            providers_tried=["searxng"],
        )

        with patch.object(
            research_steps,
            "_search_with_gemini",
            side_effect=RuntimeError("503 unavailable"),
        ), patch.object(
            research_steps,
            "_search_with_searxng",
            new=AsyncMock(return_value=searxng_result),
        ) as searxng_search, patch.object(
            research_steps,
            "_search_with_duckduckgo",
            new=AsyncMock(),
        ) as duckduckgo_search:
            result = await research_steps.search_web_candidates(
                "mars mission",
                max_results=1,
                use_fallback=True,
            )

        self.assertEqual(result.candidates[0].title, "Fallback Story")
        self.assertEqual(result.candidates[0].provider, "searxng")
        self.assertIn("gemini", result.providers_tried)
        self.assertTrue(any("Gemini search failed" in warning for warning in result.warnings))
        searxng_search.assert_awaited_once()
        duckduckgo_search.assert_not_awaited()

    async def test_web_search_reads_all_gemini_results_without_fallback(self) -> None:
        search_result = research_steps.WebSearchResult(
            query="mars mission",
            answer="Latest mission update",
            candidates=[
                research_steps.WebSearchCandidate(
                    title="Gemini Story",
                    url="https://example.com/article",
                    provider="gemini",
                    position=1,
                )
            ],
            providers_tried=["gemini"],
        )

        with patch.object(
            research_steps,
            "search_web_candidates",
            new=AsyncMock(return_value=search_result),
        ) as search_web_candidates:
            result_text = await research_steps.web_search("mars mission")

        search_web_candidates.assert_awaited_once_with(
            "mars mission",
            max_results=None,
            use_fallback=False,
        )
        self.assertIn("Gemini Grounded Answer: Latest mission update", result_text)
        self.assertIn("Gemini Story | https://example.com/article", result_text)


if __name__ == "__main__":
    unittest.main()
