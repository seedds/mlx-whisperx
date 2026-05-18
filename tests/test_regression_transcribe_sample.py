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
class SampleRegressionTests(unittest.TestCase):
    def test_vad_transcription_keeps_brakes_phrase(self):
        model = os.environ.get("MLX_WHISPERX_TEST_MODEL", "mlx-community/whisper-turbo")

        result = transcribe(
            str(FIXTURE_PATH),
            model=model,
            vad_method="silero",
            no_align=True,
            verbose=False,
            print_progress=False,
        )

        text = _normalize_text(" ".join(segment["text"] for segment in result["segments"]))

        self.assertIn("don't have brakes", text)
        self.assertIn("it'll come to him", text)
        self.assertIn("look around", text)


if __name__ == "__main__":
    unittest.main()
