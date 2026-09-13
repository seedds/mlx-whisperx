import unittest

import pandas as pd

from mlx_whisperx.alignment import _sentence_spans, interpolate_nans


class InterpolateNansTests(unittest.TestCase):
    def test_ignore_method_leaves_gaps_untouched(self):
        values = pd.Series([1.0, float("nan"), 3.0])

        result = interpolate_nans(values, method="ignore")

        self.assertTrue(pd.isna(result.iloc[1]))

    def test_ignore_method_does_not_raise_with_two_known_values(self):
        interpolate_nans(pd.Series([1.0, 2.0]), method="ignore")

    def test_nearest_method_still_fills_gaps(self):
        result = interpolate_nans(pd.Series([1.0, float("nan"), 3.0]), method="nearest")

        self.assertFalse(result.isna().any())


class SentenceSpanTests(unittest.TestCase):
    def test_falls_back_to_punctuation_split_without_a_splitter(self):
        text = "One. Two."

        spans = _sentence_spans(text, None)

        self.assertEqual([text[start:end] for start, end in spans], ["One.", " Two."])

    def test_empty_text_yields_one_span(self):
        self.assertEqual(_sentence_spans("", None), [(0, 0)])
