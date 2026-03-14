import unittest

from agent.gemini_model import (
    BALANCED_MODEL,
    LARGE_MODEL,
    LIGHT_MODEL,
    get_balanced_model,
    get_large_model,
    get_light_model,
    get_model_for_complexity,
)


class GeminiModelTests(unittest.TestCase):
    def test_light_model_returns_flash_lite(self) -> None:
        self.assertEqual(get_light_model(), LIGHT_MODEL)
        self.assertIn("flash-lite", LIGHT_MODEL)

    def test_balanced_model_returns_flash(self) -> None:
        self.assertEqual(get_balanced_model(), BALANCED_MODEL)
        self.assertIn("flash", BALANCED_MODEL)

    def test_large_model_returns_pro(self) -> None:
        self.assertEqual(get_large_model(), LARGE_MODEL)
        self.assertIn("pro", LARGE_MODEL)

    def test_complexity_aliases_map_to_expected_models(self) -> None:
        for complexity in ("easy", "light", "simple", "small"):
            self.assertEqual(get_model_for_complexity(complexity), LIGHT_MODEL)

        for complexity in ("balanced", "default", "medium", "moderate"):
            self.assertEqual(get_model_for_complexity(complexity), BALANCED_MODEL)

        for complexity in ("large", "complex", "hard", "advanced"):
            self.assertEqual(get_model_for_complexity(complexity), LARGE_MODEL)

    def test_unknown_complexity_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            get_model_for_complexity("tiny")


if __name__ == "__main__":
    unittest.main()
