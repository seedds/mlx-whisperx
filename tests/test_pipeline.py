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
