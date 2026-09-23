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

    def test_main_defaults_to_silero_vad_method(self):
        calls: list[dict] = []

        def fake_transcribe(audio_path, **kwargs):
            calls.append({"audio_path": audio_path, **kwargs})
            return {"segments": [], "word_segments": [], "language": "en"}

        writer = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "mlx-whisperx",
                "audio.wav",
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
        self.assertEqual(calls[0]["vad_method"], "silero")
        writer.assert_called_once()

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

    def test_main_forwards_allow_missing_alignment_deps(self):
        calls: list[dict] = []

        def fake_transcribe(audio_path, **kwargs):
            calls.append({"audio_path": audio_path, **kwargs})
            return {"segments": [], "word_segments": [], "language": "en"}

        writer = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = [
                "mlx-whisperx",
                "audio.wav",
                "--allow_missing_alignment_deps",
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
        self.assertTrue(calls[0]["allow_missing_alignment_deps"])
        writer.assert_called_once()

    def test_main_exits_nonzero_when_every_file_fails(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            argv = ["mlx-whisperx", "a.wav", "b.wav", "--output_dir", tmpdir, "--output_format", "json"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("mlx_whisperx.cli.transcribe", side_effect=RuntimeError("boom")),
                mock.patch("mlx_whisperx.cli.get_writer", return_value=mock.Mock()),
            ):
                with self.assertRaises(SystemExit) as exc:
                    cli.main()

        self.assertNotEqual(exc.exception.code, 0)

    def test_zero_temperature_increment_is_rejected(self):
        argv = ["mlx-whisperx", "audio.wav", "--temperature_increment_on_fallback", "0"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as exc:
                cli.main()

        self.assertEqual(exc.exception.code, 2)

    def test_temperature_above_ladder_top_is_still_decoded(self):
        calls: list[dict] = []

        def fake_transcribe(audio_path, **kwargs):
            calls.append(kwargs)
            return {"segments": [], "word_segments": [], "language": "en"}

        with tempfile.TemporaryDirectory() as tmpdir:
            argv = ["mlx-whisperx", "audio.wav", "--output_dir", tmpdir, "--temperature", "1.2"]
            with (
                mock.patch.object(sys, "argv", argv),
                mock.patch("mlx_whisperx.cli.transcribe", side_effect=fake_transcribe),
                mock.patch("mlx_whisperx.cli.get_writer", return_value=mock.Mock()),
            ):
                cli.main()

        self.assertEqual(calls[0]["temperature"], (1.2,))

    def test_every_cli_option_is_accepted_by_transcribe(self):
        """Guard the CLI -> transcribe() -> PipelineOptions chain.

        `main` forwards parsed arguments as keywords, so an option added to the
        parser but not to `transcribe` raises TypeError only at runtime, after a
        model has already been loaded.
        """
        import inspect

        from mlx_whisperx.pipeline import PipelineOptions
        from mlx_whisperx.transcribe import transcribe

        parser = cli.build_parser()
        options = {
            action.dest
            for action in parser._actions
            if action.dest not in {"help", "audio"}
        }
        # Options `main` consumes itself -- writer settings, logging, and flags
        # translated into a different keyword -- rather than forwarding.
        translated = {
            "compute_type",
            "temperature_increment_on_fallback",
            "log_level",
            "output_dir",
            "output_format",
            "output_name",
            "highlight_words",
            "max_line_count",
            "max_line_width",
            "max_words_per_line",
        }
        accepted = set(inspect.signature(transcribe).parameters)
        self.assertEqual(set(), options - translated - accepted)

        pipeline_fields = set(PipelineOptions.__dataclass_fields__)
        forwarded = accepted - {"audio", "output_dir", "output_format", "output_name"}
        self.assertEqual(set(), forwarded - pipeline_fields)

    def test_output_name_is_rejected_for_multiple_inputs(self):
        argv = ["mlx-whisperx", "a.wav", "b.wav", "--output_name", "combined"]
        with mock.patch.object(sys, "argv", argv):
            with self.assertRaises(SystemExit) as exc:
                cli.main()

        self.assertEqual(exc.exception.code, 2)
