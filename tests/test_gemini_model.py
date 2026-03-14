import importlib
import unittest
from unittest.mock import patch

gemini_model_module = importlib.import_module("agent.gemini_model")
from agent.gemini_model import BALANCED_MODEL, LARGE_MODEL, LIGHT_MODEL, gemini_model


class GeminiModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self._original_client = gemini_model_module._client
        gemini_model_module._client = None

    def tearDown(self) -> None:
        gemini_model_module._client = self._original_client

    def test_selector_returns_expected_models(self) -> None:
        self.assertEqual(gemini_model.get_light_model(), LIGHT_MODEL)
        self.assertEqual(gemini_model.get_balanced_model(), BALANCED_MODEL)
        self.assertEqual(gemini_model.get_large_model(), LARGE_MODEL)

    def test_get_client_initializes_once_and_caches(self) -> None:
        fake_client = object()

        with patch("agent.gemini_model.init_config") as mock_init_config:
            with patch("agent.gemini_model.get_env", return_value="fake-api-key") as mock_get_env:
                with patch("agent.gemini_model.genai.Client", return_value=fake_client) as mock_client_ctor:
                    first = gemini_model_module.get_client()
                    second = gemini_model_module.get_client()

        self.assertIs(first, fake_client)
        self.assertIs(second, fake_client)
        self.assertEqual(mock_init_config.call_count, 1)
        self.assertEqual(mock_get_env.call_count, 1)
        self.assertEqual(mock_client_ctor.call_count, 1)
        mock_client_ctor.assert_called_once_with(api_key="fake-api-key")

    def test_get_client_reuses_existing_without_initialization(self) -> None:
        existing_client = object()
        gemini_model_module._client = existing_client

        with patch("agent.gemini_model.init_config") as mock_init_config:
            with patch("agent.gemini_model.get_env") as mock_get_env:
                with patch("agent.gemini_model.genai.Client") as mock_client_ctor:
                    client = gemini_model_module.get_client()

        self.assertIs(client, existing_client)
        mock_init_config.assert_not_called()
        mock_get_env.assert_not_called()
        mock_client_ctor.assert_not_called()


if __name__ == "__main__":
    unittest.main()
