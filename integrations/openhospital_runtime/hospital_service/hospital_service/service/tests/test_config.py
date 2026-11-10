from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from hospital_service.service.config import load_settings


class ModelConfigTest(unittest.TestCase):
    def test_loads_examination_channels_in_order(self) -> None:
        settings = self._load("""
model:
  model: general-model
exam_model:
  channels:
    - base_url: https://primary.example/v1
      model: primary-model
      api_key: primary-key
    - base_url: https://fallback.example/v1
      model: fallback-model
      api_key: fallback-key
""")

        self.assertEqual(settings.model.model, "general-model")
        self.assertIsNone(settings.model.api_key)
        primary, fallback = settings.exam_model.channels
        self.assertEqual(primary.model, "primary-model")
        self.assertEqual(primary.api_key, "primary-key")
        self.assertEqual(primary.base_url, "https://primary.example/v1")
        self.assertEqual(fallback.model, "fallback-model")
        self.assertEqual(fallback.api_key, "fallback-key")
        self.assertEqual(fallback.base_url, "https://fallback.example/v1")

    def test_accepts_single_model_name(self) -> None:
        settings = self._load("""
model:
  model: single-model
  api_key: single-key
""")

        self.assertEqual(settings.model.models, ("single-model",))
        self.assertEqual(settings.model.model, "single-model")
        self.assertEqual(settings.model.api_key, "single-key")
        self.assertEqual(settings.exam_model.channels, (settings.model,))

    def test_accepts_legacy_flat_examination_model(self) -> None:
        settings = self._load("""
exam_model:
  base_url: https://legacy.example/v1
  model:
    - primary-model
    - fallback-model
  api_key: legacy-key
""")

        channel = settings.exam_model.channels[0]
        self.assertEqual(channel.models, ("primary-model", "fallback-model"))
        self.assertEqual(channel.api_key, "legacy-key")

    def test_allows_same_model_name_across_channels(self) -> None:
        settings = self._load("""
exam_model:
  channels:
    - base_url: https://primary.example/v1
      model: shared-model
      api_key: primary-key
    - base_url: https://fallback.example/v1
      model: shared-model
      api_key: fallback-key
""")

        self.assertEqual(
            [channel.model for channel in settings.exam_model.channels],
            ["shared-model", "shared-model"],
        )

    @staticmethod
    def _load(content: str):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.yaml"
            path.write_text(content, encoding="utf-8")
            with patch.dict(os.environ, {
                "API_KEY": "",
                "EXAM_MODEL_API_KEY": "",
                "EXAM_MODEL_API_KEY_2": "",
            }, clear=False):
                return load_settings(path)


if __name__ == "__main__":
    unittest.main()
