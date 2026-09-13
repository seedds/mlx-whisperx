import io
import tempfile
import unittest
from pathlib import Path

from mlx_whisperx.writers import WriteAUD, WriteSRT


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

    def test_output_name_with_dots_is_not_truncated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = WriteSRT(tmpdir)
            result = {"language": "en", "segments": [{"start": 0.0, "end": 1.0, "text": "one", "words": []}]}

            writer(result, "meeting.part1")
            writer(result, "meeting.part2")

            names = sorted(p.name for p in Path(tmpdir).iterdir())
            self.assertEqual(names, ["meeting.part1.srt", "meeting.part2.srt"])

    def test_cue_breaks_when_the_speaker_changes(self):
        writer = WriteSRT(".")
        output = io.StringIO()
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hi there friend",
                    "words": [
                        {"word": "hi", "start": 0.0, "end": 0.5, "speaker": "SPEAKER_00"},
                        {"word": "there", "start": 0.5, "end": 1.0, "speaker": "SPEAKER_01"},
                        {"word": "friend", "start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
                    ],
                }
            ],
        }

        writer.write_result(result, output, {})
        text = output.getvalue()

        self.assertIn("[SPEAKER_00]: hi", text)
        self.assertIn("[SPEAKER_01]: there friend", text)
        self.assertNotIn("[SPEAKER_00]: hi there friend", text)

    def test_highlighting_does_not_emit_backward_gap_cues(self):
        writer = WriteSRT(".")
        output = io.StringIO()
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "one two",
                    "words": [
                        {"word": "one", "start": 0.0, "end": 1.0},
                        {"word": "two", "start": 0.8, "end": 1.4},
                    ],
                }
            ],
        }

        writer.write_result(result, output, {"highlight_words": True})

        for line in output.getvalue().splitlines():
            if "-->" not in line:
                continue
            start, end = (part.strip() for part in line.split("-->"))
            self.assertLessEqual(start, end, msg=line)

    def test_unaligned_fallback_applies_max_words_per_line(self):
        writer = WriteSRT(".")
        output = io.StringIO()
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 4.0, "text": "one two three four five", "words": []}],
        }

        writer.write_result(result, output, {"max_words_per_line": 2})
        cues = [line for line in output.getvalue().splitlines() if line and "-->" not in line and not line.isdigit()]

        self.assertEqual(cues, ["one two", "three four", "five"])

    def test_audacity_writer_emits_label_track(self):
        writer = WriteAUD(".")
        output = io.StringIO()
        result = {
            "language": "en",
            "segments": [{"start": 0.0, "end": 1.5, "text": "hello", "speaker": "SPEAKER_00"}],
        }

        writer.write_result(result, output, {})

        self.assertEqual(output.getvalue(), "0.0\t1.5\t[[SPEAKER_00]]hello\n")
