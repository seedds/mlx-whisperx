import importlib.util
import os
import re
import unittest
from pathlib import Path

from mlx_whisperx import transcribe


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "ben10_s01e02_04m10s_04m33s.wav"


def _normalize_text(text: str) -> str:
    """Collapse case and punctuation differences that are irrelevant to this regression."""
    text = text.lower().replace("’", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@unittest.skipUnless(importlib.util.find_spec("mlx") is not None, "requires mlx runtime")
@unittest.skipUnless(importlib.util.find_spec("torch") is not None, "requires torch for silero VAD")
@unittest.skipUnless(
    os.environ.get("MLX_WHISPERX_RUN_SAMPLE_TEST") == "1",
    "set MLX_WHISPERX_RUN_SAMPLE_TEST=1 to run model-output regression",
)
class SampleRegressionTests(unittest.TestCase):
    def _assert_keeps_brakes_phrase(self, **options):
        model = os.environ.get("MLX_WHISPERX_TEST_MODEL", "mlx-community/whisper-turbo")

        result = transcribe(
            str(FIXTURE_PATH),
            model=model,
            vad_method="silero",
            no_align=True,
            verbose=False,
            print_progress=False,
            **options,
        )

        text = _normalize_text(" ".join(segment["text"] for segment in result["segments"]))

        self.assertIn("don't have brakes", text)
        self.assertIn("it'll come to him", text)
        self.assertIn("look around", text)

    def test_vad_transcription_keeps_brakes_phrase(self):
        self._assert_keeps_brakes_phrase()

    def test_interleaved_context_transcription_keeps_brakes_phrase(self):
        # 12-second chunks split this clip in two with the shouted line inside the first
        # (shorter chunks would leave it between chunks, where nothing is decoded), so
        # two streams exercise both the per-stream prompts and the wrap-around redo.
        self._assert_keeps_brakes_phrase(interleaved_context=True, batch_size=2, chunk_size=12)


if __name__ == "__main__":
    unittest.main()
