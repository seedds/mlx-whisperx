import importlib
import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from mlx_whisperx._language import (
    is_english_only_model,
    normalize_language,
    normalize_language_settings,
)


def _write_config(directory: Path, n_vocab: int) -> None:
    (directory / "config.json").write_text(json.dumps({"n_vocab": n_vocab}), encoding="utf-8")


class LanguageTests(unittest.TestCase):
    def test_normalize_language_accepts_aliases_case_insensitively(self):
        self.assertEqual(normalize_language("ENGLISH"), "en")
        self.assertEqual(normalize_language("Portuguese"), "pt")

    def test_normalize_language_rejects_unknown_values(self):
        with self.assertRaisesRegex(ValueError, "Unsupported language"):
            normalize_language("Klingon")

    def test_english_only_name_warns_and_forces_english(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            language, task = normalize_language_settings("tiny.en", "German", "transcribe")

        self.assertEqual(language, "en")
        self.assertEqual(task, "transcribe")
        self.assertEqual(len(caught), 1)
        self.assertIn("English-only model", str(caught[0].message))

    def test_english_only_model_rejects_translate(self):
        with self.assertRaisesRegex(ValueError, "do not support task='translate'"):
            normalize_language_settings("tiny.en", "English", "translate")

    def test_local_model_config_detects_english_only_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_config(model_dir, 51864)
            self.assertTrue(is_english_only_model(str(model_dir)))

    def test_local_model_config_detects_multilingual_models(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            _write_config(model_dir, 51865)
            self.assertFalse(is_english_only_model(str(model_dir)))

    def test_api_normalizes_language_before_pipeline_construction(self):
        api = importlib.import_module("mlx_whisperx.transcribe")
        captured = []

        class FakePipeline:
            def __init__(self, options):
                captured.append(options)

            def transcribe(self, audio):
                return {"audio": audio, "language": captured[-1].language}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with mock.patch("mlx_whisperx.transcribe.MLXWhisperXPipeline", FakePipeline):
                result = api.transcribe("audio.wav", model="tiny.en", language="German")

        self.assertEqual(result["language"], "en")
        self.assertEqual(captured[0].language, "en")
        self.assertEqual(len(caught), 1)
