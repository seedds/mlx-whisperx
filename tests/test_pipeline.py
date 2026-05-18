import json
import importlib
import sys
import tempfile
import types
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

from mlx_whisperx.alignment import AlignmentDependencyError
from mlx_whisperx.pipeline import MLXWhisperXPipeline, PipelineOptions


class PipelineTests(unittest.TestCase):
    def _write_config(self, directory: Path, n_vocab: int) -> None:
        (directory / "config.json").write_text(json.dumps({"n_vocab": n_vocab}), encoding="utf-8")

    def test_pipeline_rejects_clip_timestamps_with_vad(self):
        fake_backend = mock.Mock()
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions(clip_timestamps="0,5"))

        with self.assertRaisesRegex(ValueError, "clip_timestamps requires no_vad=True"):
            pipeline.transcribe(np.zeros(16000, dtype=np.float32))

    def test_asr_forwards_cache_and_clip_options_to_backend(self):
        fake_backend = mock.Mock()
        fake_backend.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "text": "hello",
        }
        options = PipelineOptions(
            no_vad=True,
            language="en",
            model_dir="/tmp/model-cache",
            model_cache_only=True,
            clip_timestamps="0,5",
        )
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(options)

        result = pipeline._asr(
            np.zeros(16000, dtype=np.float32),
            [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}],
        )

        self.assertEqual(result["language"], "en")
        kwargs = fake_backend.transcribe.call_args.kwargs
        self.assertEqual(kwargs["model_dir"], "/tmp/model-cache")
        self.assertTrue(kwargs["model_cache_only"])
        self.assertEqual(kwargs["clip_timestamps"], "0,5")

    def test_no_vad_enables_backend_progress_when_verbose(self):
        fake_backend = mock.Mock()
        fake_backend.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "text": "hello",
        }
        options = PipelineOptions(no_vad=True, language="en", verbose=True)
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(options)

        pipeline._asr(
            np.zeros(16000, dtype=np.float32),
            [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}],
        )

        self.assertFalse(fake_backend.transcribe.call_args.kwargs["verbose"])
        self.assertFalse(fake_backend.transcribe.call_args.kwargs["without_timestamps"])

    def test_vad_cut_only_uses_backend_transcribe_with_timestamps(self):
        fake_backend = mock.Mock()
        fake_backend.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "text": "hello",
        }
        fake_backend.transcribe_chunk.return_value = {"start": 0.0, "end": 1.0, "text": "ignored"}
        options = PipelineOptions(vad_cut_only=True, language="en")
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(options)

        result = pipeline._asr(
            np.zeros(16000, dtype=np.float32),
            [{"start": 0.0, "end": 1.0, "segments": []}],
        )

        self.assertEqual(result["segments"][0]["text"], "hello")
        self.assertFalse(fake_backend.transcribe.call_args.kwargs["without_timestamps"])
        fake_backend.transcribe.assert_called_once()
        fake_backend.transcribe_chunk.assert_not_called()

    def test_vad_cut_only_chunks_preserve_full_timeline(self):
        fake_backend = mock.Mock()
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions(vad_cut_only=True, chunk_size=5))

        chunks = pipeline._vad_cut_only_chunks(
            [
                {"start": 7.106, "end": 9.694, "segments": [(7.106, 8.062), (9.122, 9.694)]},
            ],
            13.0,
        )

        self.assertEqual(
            chunks,
            [
                {"start": 0.0, "end": 5.0, "segments": []},
                {"start": 5.0, "end": 7.106, "segments": []},
                {"start": 7.106, "end": 9.694, "segments": [(7.106, 8.062), (9.122, 9.694)]},
                {"start": 9.694, "end": 13.0, "segments": []},
            ],
        )

    def test_vad_cut_only_without_speech_returns_full_file_chunk(self):
        fake_backend = mock.Mock()
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions(vad_cut_only=True))

        self.assertEqual(
            pipeline._vad_cut_only_chunks([], 12.5),
            [{"start": 0.0, "end": 12.5, "segments": []}],
        )

    def test_asr_without_vad_handles_shadowed_backend_transcribe_module(self):
        transcribe_func = mock.Mock(
            return_value={
                "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
                "language": "en",
                "text": "hello",
            }
        )
        fake_backend = types.SimpleNamespace(
            transcribe=types.SimpleNamespace(transcribe=transcribe_func)
        )
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions(no_vad=True, language="en"))

        result = pipeline._asr(
            np.zeros(16000, dtype=np.float32),
            [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}],
        )

        self.assertEqual(result["segments"][0]["text"], "hello")
        transcribe_func.assert_called_once()

    def test_asr_uses_direct_chunk_decode_in_vad_mode(self):
        fake_backend = mock.Mock()
        fake_backend.detect_language.return_value = "en"
        fake_backend.transcribe_chunk.side_effect = [
            {"start": 0.0, "end": 1.0, "text": "hello", "avg_logprob": -0.1},
            {"start": 0.0, "end": 1.5, "text": "world", "avg_logprob": -0.2},
        ]
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions())

        result = pipeline._asr(
            np.zeros(16000 * 5, dtype=np.float32),
            [
                {"start": 1.0, "end": 2.0, "segments": [(1.0, 2.0)]},
                {"start": 3.0, "end": 4.5, "segments": [(3.0, 4.5)]},
            ],
        )

        self.assertEqual(result["language"], "en")
        self.assertEqual(
            result["segments"],
            [
                {"start": 1.0, "end": 2.0, "text": "hello", "avg_logprob": -0.1},
                {"start": 3.0, "end": 4.5, "text": "world", "avg_logprob": -0.2},
            ],
        )
        fake_backend.detect_language.assert_called_once()
        self.assertEqual(fake_backend.transcribe_chunk.call_count, 2)
        fake_backend.transcribe.assert_not_called()

    def test_asr_with_vad_compacts_inner_speech_segments_before_decoding(self):
        fake_backend = mock.Mock()
        fake_backend.detect_language.return_value = "en"
        fake_backend.transcribe_chunk.return_value = {
            "start": 0.0,
            "end": 2.5,
            "text": "hello world",
            "avg_logprob": -0.1,
        }
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions())

        audio = np.arange(16000 * 6, dtype=np.float32)
        pipeline._asr(
            audio,
            [
                {
                    "start": 1.0,
                    "end": 4.5,
                    "segments": [(1.0, 2.0), (3.0, 4.5)],
                }
            ],
        )

        chunk_audio = fake_backend.transcribe_chunk.call_args.args[0]
        expected = np.concatenate([audio[16000:32000], audio[48000:72000]])
        np.testing.assert_array_equal(chunk_audio, expected)

    def test_asr_with_vad_restores_timestamps_across_internal_silence(self):
        fake_backend = mock.Mock()
        fake_backend.detect_language.return_value = "en"
        fake_backend.transcribe_chunk.return_value = {
            "start": 0.8,
            "end": 1.2,
            "text": "hello",
            "avg_logprob": -0.1,
        }
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions())

        result = pipeline._asr(
            np.zeros(16000 * 15, dtype=np.float32),
            [
                {
                    "start": 10.0,
                    "end": 13.5,
                    "segments": [(10.0, 11.0), (12.0, 13.5)],
                }
            ],
        )

        self.assertEqual(
            result["segments"],
            [{"start": 10.8, "end": 12.2, "text": "hello", "avg_logprob": -0.1}],
        )

    def test_pipeline_normalizes_language_for_direct_usage(self):
        fake_backend = mock.Mock()
        with tempfile.TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir)
            self._write_config(model_dir, 51864)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
                    pipeline = MLXWhisperXPipeline(
                        PipelineOptions(model=str(model_dir), language="German", no_vad=True)
                    )

        self.assertEqual(pipeline.options.language, "en")
        self.assertEqual(len(caught), 1)

    def test_pipeline_can_continue_without_alignment_when_opted_in(self):
        fake_backend = mock.Mock()
        fake_backend.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "text": "hello",
        }
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(
                PipelineOptions(
                    no_vad=True,
                    language="en",
                    allow_missing_alignment_deps=True,
                )
            )

        with mock.patch.object(
            pipeline,
            "_align",
            side_effect=AlignmentDependencyError("missing alignment deps"),
        ):
            result = pipeline.transcribe(np.zeros(16000, dtype=np.float32))

        self.assertEqual(result["language"], "en")
        self.assertEqual(result["word_segments"], [])
        self.assertEqual(result["segments"][0]["text"], "hello")
        self.assertEqual(result["segments"][0]["words"], [])

    def test_pipeline_still_raises_alignment_dependency_errors_by_default(self):
        fake_backend = mock.Mock()
        fake_backend.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "text": "hello",
        }
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(PipelineOptions(no_vad=True, language="en"))

        with (
            mock.patch.object(
                pipeline,
                "_align",
                side_effect=AlignmentDependencyError("missing alignment deps"),
            ),
            self.assertRaisesRegex(AlignmentDependencyError, "missing alignment deps"),
        ):
            pipeline.transcribe(np.zeros(16000, dtype=np.float32))

    def test_pipeline_does_not_mask_other_alignment_failures(self):
        fake_backend = mock.Mock()
        fake_backend.transcribe.return_value = {
            "segments": [{"start": 0.0, "end": 1.0, "text": "hello"}],
            "language": "en",
            "text": "hello",
        }
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=fake_backend):
            pipeline = MLXWhisperXPipeline(
                PipelineOptions(
                    no_vad=True,
                    language="en",
                    allow_missing_alignment_deps=True,
                )
            )

        with (
            mock.patch.object(pipeline, "_align", side_effect=RuntimeError("alignment broke")),
            self.assertRaisesRegex(RuntimeError, "alignment broke"),
        ):
            pipeline.transcribe(np.zeros(16000, dtype=np.float32))

    def test_model_holder_cache_keys_include_cache_settings(self):
        tiktoken = sys.modules.setdefault("tiktoken", types.ModuleType("tiktoken"))
        if not hasattr(tiktoken, "Encoding"):
            tiktoken.Encoding = object
        huggingface_hub = sys.modules.setdefault(
            "huggingface_hub",
            types.ModuleType("huggingface_hub"),
        )
        if not hasattr(huggingface_hub, "snapshot_download"):
            huggingface_hub.snapshot_download = mock.Mock()
        sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))
        numba = sys.modules.setdefault("numba", types.ModuleType("numba"))
        if not hasattr(numba, "jit"):
            numba.jit = lambda *args, **kwargs: (lambda func: func)
        scipy = sys.modules.setdefault("scipy", types.ModuleType("scipy"))
        if not hasattr(scipy, "signal"):
            scipy.signal = types.SimpleNamespace(medfilt=lambda x, kernel_size=None: x)
        ModelHolder = importlib.import_module(
            "mlx_whisperx.backend.mlx_whisper.transcribe"
        ).ModelHolder

        ModelHolder.model = None
        ModelHolder.model_key = None
        with mock.patch("mlx_whisperx.backend.mlx_whisper.transcribe.load_model", side_effect=[object(), object()]) as load_model:
            first = ModelHolder.get_model("repo", "dtype", model_dir="cache-a", model_cache_only=False)
            second = ModelHolder.get_model("repo", "dtype", model_dir="cache-b", model_cache_only=False)

        self.assertIsNot(first, second)
        self.assertEqual(load_model.call_count, 2)
