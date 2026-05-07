import sys
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

from mlx_whisperx import cli


class CLITests(unittest.TestCase):
    def test_language_aliases_are_normalized_case_insensitively(self):
        parser = cli.build_parser()
        args = parser.parse_args(["audio.wav", "--language", "ENGLISH"])
        self.assertEqual(args.language, "en")

    def test_clip_timestamps_requires_no_vad(self):
        argv = ["mlx-whisperx", "audio.wav", "--clip_timestamps", "0,5"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as exc:
                cli.main()
        self.assertEqual(exc.exception.code, 2)

    def test_translate_is_rejected_for_english_only_models(self):
        argv = ["mlx-whisperx", "audio.wav", "--model", "tiny.en", "--task", "translate"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as exc:
                cli.main()
        self.assertEqual(exc.exception.code, 2)

    def test_main_forwards_clip_timestamps_when_no_vad(self):
        calls: list[dict] = []

        def fake_transcribe(audio_path, **kwargs):
            calls.append({"audio_path": audio_path, **kwargs})
            return {"segments": [], "word_segments": [], "language": "en"}

        writer = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "mlx-whisperx",
                "audio.wav",
                "--no_vad",
                "--clip_timestamps",
                "0,5",
                "--model_dir",
                str(Path(tmpdir) / "models"),
                "--model_cache_only",
                "True",
                "--output_dir",
                tmpdir,
                "--output_format",
                "json",
            ]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("mlx_whisperx.cli.transcribe", side_effect=fake_transcribe),
                mock.patch("mlx_whisperx.cli.get_writer", return_value=writer),
            ):
                cli.main()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["clip_timestamps"], "0,5")
        self.assertTrue(calls[0]["no_vad"])
        self.assertTrue(calls[0]["model_cache_only"])
        writer.assert_called_once()

    def test_main_warns_and_forces_english_for_english_only_models(self):
        calls: list[dict] = []

        def fake_transcribe(audio_path, **kwargs):
            calls.append({"audio_path": audio_path, **kwargs})
            return {"segments": [], "word_segments": [], "language": kwargs["language"]}

        writer = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "mlx-whisperx",
                "audio.wav",
                "--model",
                "tiny.en",
                "--language",
                "German",
                "--output_dir",
                tmpdir,
                "--output_format",
                "json",
            ]
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with (
                    mock.patch.object(sys, "argv", argv),
                    mock.patch("mlx_whisperx.cli.transcribe", side_effect=fake_transcribe),
                    mock.patch("mlx_whisperx.cli.get_writer", return_value=writer),
                ):
                    cli.main()

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["language"], "en")
        self.assertEqual(len(caught), 1)
        self.assertIn("English-only model", str(caught[0].message))
