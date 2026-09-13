import unittest

import pandas as pd

from mlx_whisperx.diarize import IntervalTree, assign_word_speakers


class DiarizeAssignmentTests(unittest.TestCase):
    def test_find_nearest_uses_interval_distance_not_midpoint(self):
        tree = IntervalTree([(0.0, 100.0, "A"), (102.0, 103.0, "B")])

        self.assertEqual(tree.find_nearest(100.1), "A")

    def test_words_without_overlap_are_left_unlabelled_by_default(self):
        diarize_df = pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "A"}])
        result = {"segments": [{"start": 5.0, "end": 6.0, "text": "hi", "words": [{"word": "hi", "start": 5.0, "end": 6.0}]}]}

        assign_word_speakers(diarize_df, result)

        self.assertNotIn("speaker", result["segments"][0]["words"][0])
