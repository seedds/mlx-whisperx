import io
import unittest

from mlx_whisperx.writers import WriteSRT


class WriterTests(unittest.TestCase):
    def test_srt_falls_back_to_segment_text_when_words_are_missing(self):
        writer = WriteSRT(".")
        output = io.StringIO()
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hello there",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5},
                        {"word": "there", "start": 0.5, "end": 1.0},
                    ],
                },
                {
                    "start": 1.2,
                    "end": 2.0,
                    "text": "missing line",
                    "words": [],
                },
            ],
        }

        writer.write_result(result, output, {})

        self.assertEqual(
            output.getvalue(),
            "1\n"
            "00:00:00,000 --> 00:00:01,000\n"
            "hello there\n\n"
            "2\n"
            "00:00:01,200 --> 00:00:02,000\n"
            "missing line\n\n",
        )

    def test_srt_splits_unaligned_multisentence_segments(self):
        writer = WriteSRT(".")
        output = io.StringIO()
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 240.0,
                    "end": 268.0,
                    "text": "Next stop, Gaseous Gardens. Hey! You don't have brakes! It'll come to him. Huh?",
                    "words": [],
                }
            ],
        }

        writer.write_result(result, output, {})

        self.assertIn("Next stop, Gaseous Gardens.", output.getvalue())
        self.assertIn("Hey!", output.getvalue())
        self.assertIn("You don't have brakes!", output.getvalue())
        self.assertIn("It'll come to him.", output.getvalue())
        self.assertIn("Huh?", output.getvalue())
