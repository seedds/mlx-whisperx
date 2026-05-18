import importlib
import sys
import types
import unittest
import warnings
from unittest import mock


class BackendTranscribeTests(unittest.TestCase):
    def _import_transcribe(self):
        mx_module = sys.modules.setdefault("mlx", types.ModuleType("mlx"))
        mx_core = sys.modules.setdefault("mlx.core", types.ModuleType("mlx.core"))
        if not hasattr(mx_core, "array"):
            mx_core.array = object
        if not hasattr(mx_core, "Dtype"):
            mx_core.Dtype = object
        if not hasattr(mx_core, "float16"):
            mx_core.float16 = "float16"
        if not hasattr(mx_core, "float32"):
            mx_core.float32 = "float32"
        mx_module.core = mx_core
        sys.modules.setdefault("tqdm", types.ModuleType("tqdm"))

        audio = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.audio",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.audio"),
        )
        audio.FRAMES_PER_SECOND = 50
        audio.HOP_LENGTH = 320
        audio.N_FRAMES = 3000
        audio.N_SAMPLES = 480000
        audio.SAMPLE_RATE = 16000
        audio.load_audio = mock.Mock()
        audio.log_mel_spectrogram = mock.Mock()
        audio.pad_or_trim = mock.Mock()

        decoding = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.decoding",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.decoding"),
        )

        class FakeDecodingOptions:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakeDecodingResult:
            def __init__(self, compression_ratio=0.0, avg_logprob=0.0):
                self.compression_ratio = compression_ratio
                self.avg_logprob = avg_logprob

        decoding.DecodingOptions = FakeDecodingOptions
        decoding.DecodingResult = FakeDecodingResult

        languages = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.languages",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.languages"),
        )
        languages.LANGUAGES = {"en": "English"}

        load_models = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.load_models",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.load_models"),
        )
        load_models.load_model = mock.Mock()

        timing = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.timing",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.timing"),
        )
        timing.add_word_timestamps = mock.Mock()

        tokenizer = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.tokenizer",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.tokenizer"),
        )
        tokenizer.get_tokenizer = mock.Mock()

        sys.modules.pop("mlx_whisperx.backend.mlx_whisper.transcribe", None)
        return importlib.import_module("mlx_whisperx.backend.mlx_whisper.transcribe")

    def test_decode_with_fallback_retries_beam_failure_with_greedy(self):
        transcribe = self._import_transcribe()
        result = transcribe.DecodingResult(compression_ratio=0.0, avg_logprob=0.0)

        def fake_decode(_segment, options):
            if getattr(options, "beam_size", None) is not None:
                raise RuntimeError("beam search produced 0 active beams, expected 5")
            return result

        model = types.SimpleNamespace(decode=mock.Mock(side_effect=fake_decode))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            returned = transcribe._decode_with_fallback(
                model,
                segment=object(),
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                decode_options={"beam_size": 5, "patience": 1.0, "language": "en"},
            )

        self.assertIs(returned, result)
        self.assertEqual(model.decode.call_count, 2)
        first_options = model.decode.call_args_list[0].args[1]
        second_options = model.decode.call_args_list[1].args[1]
        self.assertEqual(first_options.beam_size, 5)
        self.assertFalse(hasattr(second_options, "beam_size"))
        self.assertEqual(len(caught), 1)
        self.assertIn("retrying with greedy decoding", str(caught[0].message))

    def test_decode_with_fallback_reraises_other_runtime_errors(self):
        transcribe = self._import_transcribe()
        model = types.SimpleNamespace(decode=mock.Mock(side_effect=RuntimeError("other decode failure")))

        with self.assertRaisesRegex(RuntimeError, "other decode failure"):
            transcribe._decode_with_fallback(
                model,
                segment=object(),
                temperature=0.0,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                decode_options={"beam_size": 5, "patience": 1.0, "language": "en"},
            )
