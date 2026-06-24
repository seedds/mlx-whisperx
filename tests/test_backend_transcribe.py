import importlib
import sys
import types
import unittest
import warnings
from unittest import mock


class _FakeMels:
    """Minimal stand-in for an MLX array of stacked Mel windows."""

    def __init__(self, items):
        self._items = list(items)
        self.shape = (len(self._items),)

    def __getitem__(self, index):
        return self._items[index]


def mx_array(items):
    return _FakeMels(items)


class BackendTranscribeTests(unittest.TestCase):
    STUBBED_MODULES = [
        "mlx",
        "mlx.core",
        "mlx_whisperx.backend.mlx_whisper.audio",
        "mlx_whisperx.backend.mlx_whisper.decoding",
        "mlx_whisperx.backend.mlx_whisper.languages",
        "mlx_whisperx.backend.mlx_whisper.load_models",
        "mlx_whisperx.backend.mlx_whisper.timing",
        "mlx_whisperx.backend.mlx_whisper.tokenizer",
        "mlx_whisperx.backend.mlx_whisper.transcribe",
        "tqdm",
    ]

    def setUp(self):
        self._original_modules = {
            name: sys.modules.get(name)
            for name in self.STUBBED_MODULES
            if name in sys.modules
        }

    def tearDown(self):
        for name in self.STUBBED_MODULES:
            if name in self._original_modules:
                sys.modules[name] = self._original_modules[name]
                parent_name, _, attr = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None:
                    setattr(parent, attr, self._original_modules[name])
            else:
                sys.modules.pop(name, None)
                parent_name, _, attr = name.rpartition(".")
                parent = sys.modules.get(parent_name)
                if parent is not None and attr in parent.__dict__:
                    delattr(parent, attr)

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
        if not hasattr(mx_core, "stack"):
            mx_core.stack = lambda items: list(items)
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
            def __init__(self, compression_ratio=0.0, avg_logprob=0.0, no_speech_prob=0.0):
                self.compression_ratio = compression_ratio
                self.avg_logprob = avg_logprob
                self.no_speech_prob = no_speech_prob

        decoding.DecodingOptions = FakeDecodingOptions
        decoding.DecodingResult = FakeDecodingResult

        languages = sys.modules.setdefault(
            "mlx_whisperx.backend.mlx_whisper.languages",
            types.ModuleType("mlx_whisperx.backend.mlx_whisper.languages"),
        )
        languages.LANGUAGES = {"en": "English"}
        languages.TO_LANGUAGE_CODE = {"english": "en"}

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

    def test_batch_fallback_only_rebatches_failing_windows(self):
        transcribe = self._import_transcribe()

        good = transcribe.DecodingResult(compression_ratio=0.0, avg_logprob=0.0)
        # First item fails logprob threshold at T=0, then passes at the next temperature.
        bad_then_good = [
            transcribe.DecodingResult(compression_ratio=0.0, avg_logprob=-5.0),
            transcribe.DecodingResult(compression_ratio=0.0, avg_logprob=0.0),
        ]
        calls = []

        def fake_decode(mels, options):
            calls.append((list(mels), options.temperature))
            if options.temperature == 0.0:
                return [bad_then_good[0], good]
            # Only the failing window is re-decoded.
            return [bad_then_good[1]]

        model = types.SimpleNamespace(decode=mock.Mock(side_effect=fake_decode))

        results = transcribe._decode_batch_with_fallback(
            model,
            mx_array([10, 20]),
            temperature=(0.0, 0.2),
            compression_ratio_threshold=2.4,
            logprob_threshold=-1.0,
            no_speech_threshold=0.6,
            decode_options={"beam_size": 5, "patience": 1.0, "language": "en"},
        )

        self.assertEqual(results, [bad_then_good[1], good])
        # Second decode call re-batches only the single failing window.
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], 0.0)
        self.assertEqual(calls[1][1], 0.2)
        self.assertEqual(len(calls[1][0]), 1)

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
