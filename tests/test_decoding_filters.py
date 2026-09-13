import unittest

import mlx.core as mx
import numpy as np

from mlx_whisperx.backend.mlx_whisper.decoding import ApplyTimestampRules


class FakeTokenizer:
    """Small stand-in with the vocabulary layout ApplyTimestampRules relies on."""

    eot = 10
    no_timestamps = 11
    timestamp_begin = 12


class ApplyTimestampRulesTests(unittest.TestCase):
    N_VOCAB = 20

    def _apply(self, tokens, logits=None, sample_begin=0):
        rule = ApplyTimestampRules(FakeTokenizer(), sample_begin, None)
        if logits is None:
            logits = mx.zeros((len(tokens), self.N_VOCAB))
        return np.array(rule.apply(logits, mx.array(tokens)))

    def test_timestamps_cannot_decrease(self):
        # An opened-and-closed pair at t=16 must not be followed by an earlier timestamp.
        out = self._apply([[15, 1, 16]])

        self.assertTrue(np.all(np.isneginf(out[0, FakeTokenizer.timestamp_begin:16])))
        self.assertFalse(np.isneginf(out[0, 17]))

    def test_open_segment_may_close_at_the_same_timestamp(self):
        out = self._apply([[1, 16]])

        self.assertFalse(np.isneginf(out[0, 16]))

    def test_some_token_always_remains_available(self):
        # Timestamp mass dominates, but after an opening timestamp all timestamps are
        # forbidden, so text must stay selectable rather than everything being masked.
        logits = np.full((1, self.N_VOCAB), -10.0, dtype=np.float32)
        logits[0, FakeTokenizer.timestamp_begin:] = 5.0
        logits[0, 1] = 0.0

        out = self._apply([[1, 16]], logits=mx.array(logits))

        self.assertFalse(np.all(np.isneginf(out[0])))

    def test_no_timestamps_token_is_always_suppressed(self):
        out = self._apply([[1, 2]])

        self.assertTrue(np.isneginf(out[0, FakeTokenizer.no_timestamps]))
