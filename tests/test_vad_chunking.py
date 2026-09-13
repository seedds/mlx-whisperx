import unittest

from mlx_whisperx.vads.vad import Segment, Vad


class VadChunkingTests(unittest.TestCase):
    def test_long_speech_turn_is_split_to_chunk_size(self):
        chunks = Vad.merge_chunks([Segment(0.0, 90.0, "UNKNOWN")], 30, 0.5, None)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 30 + 1e-9)
        self.assertAlmostEqual(chunks[0]["start"], 0.0)
        self.assertAlmostEqual(chunks[-1]["end"], 90.0)

    def test_splitting_preserves_full_coverage_without_gaps(self):
        segments = Vad.split_long_segments([Segment(1.0, 76.0, None)], 30)

        self.assertEqual(segments[0].start, 1.0)
        self.assertEqual(segments[-1].end, 76.0)
        for previous, current in zip(segments, segments[1:]):
            self.assertEqual(previous.end, current.start)

    def test_short_segments_are_left_alone(self):
        segments = [Segment(0.0, 5.0, None), Segment(6.0, 9.0, None)]

        self.assertEqual(Vad.split_long_segments(segments, 30), segments)
