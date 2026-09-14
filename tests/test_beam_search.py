import unittest

import mlx.core as mx
import numpy as np

from mlx_whisperx.backend.mlx_whisper.decoding import BeamSearchDecoder

EOT = 10


class FakeInference:
    """Records beam permutations; beam search calls this from `update`."""

    def __init__(self):
        self.rearranged = []

    def rearrange_kv_cache(self, source_indices):
        self.rearranged.append(list(source_indices))


def make_decoder(beam_size=2, patience=None):
    return BeamSearchDecoder(beam_size, EOT, FakeInference(), patience)


def finalize(decoder, tokens, sum_logprobs):
    out_tokens, out_scores = decoder.finalize(mx.array(tokens), mx.array(sum_logprobs))
    return np.array(out_tokens).tolist(), np.array(out_scores).tolist()


def strip_padding(sequence):
    """Drop the trailing EOT padding `finalize` adds to rectangularise the batch."""
    sequence = list(sequence)
    return sequence[: sequence.index(EOT) + 1] if EOT in sequence else sequence


class BeamSearchFinalizeTests(unittest.TestCase):
    def test_finished_candidates_are_not_displaced_by_unfinished_beams(self):
        # Beams still mid-sentence always outscore completed ones, because a shorter
        # sequence has accumulated fewer negative logprobs. They must not evict
        # genuinely finished transcriptions from the candidate list.
        decoder = make_decoder(beam_size=2)
        decoder.finished_sequences = [{(1, 2, EOT): -1.0, (3, 4, EOT): -2.0}]

        tokens, scores = finalize(decoder, [[[5, 6, 7], [8, 9, 1]]], [[-0.1, -0.2]])

        self.assertEqual(
            [strip_padding(sequence) for sequence in tokens[0]],
            [[1, 2, EOT], [3, 4, EOT]],
        )
        self.assertEqual(scores[0], [-1.0, -2.0])

    def test_unfinished_beams_only_fill_a_shortfall(self):
        # One finished sequence is not enough to fill the beam, so the best unfinished
        # beam is forced to EOT to top it up -- but only up to beam_size.
        decoder = make_decoder(beam_size=2)
        decoder.finished_sequences = [{(1, 2, EOT): -5.0}]

        tokens, scores = finalize(decoder, [[[5, 6, 7], [8, 9, 1]]], [[-0.1, -0.2]])

        self.assertEqual(
            [strip_padding(sequence) for sequence in tokens[0]],
            [[1, 2, EOT], [5, 6, 7, EOT]],
        )
        # Scores round-trip through a float32 array, so compare approximately.
        self.assertEqual(len(scores[0]), 2)
        self.assertAlmostEqual(scores[0][0], -5.0, places=5)
        self.assertAlmostEqual(scores[0][1], -0.1, places=5)

    def test_all_beams_unfinished_falls_back_to_active_beams(self):
        # A window that hits the token limit without any beam emitting EOT must still
        # produce candidates rather than an empty result.
        decoder = make_decoder(beam_size=2)
        decoder.finished_sequences = [{}]

        tokens, scores = finalize(decoder, [[[5, 6, 7], [8, 9, 1]]], [[-0.2, -0.1]])

        self.assertEqual(
            [strip_padding(sequence) for sequence in tokens[0]],
            [[8, 9, 1, EOT], [5, 6, 7, EOT]],
        )
        self.assertEqual(len(scores[0]), 2)
        self.assertAlmostEqual(scores[0][0], -0.1, places=5)
        self.assertAlmostEqual(scores[0][1], -0.2, places=5)

    def test_patience_keeps_extra_finished_candidates(self):
        # patience=2.0 asks beam search to collect 4 candidates for a beam of 2, and
        # all of them should survive finalization for the ranker to choose between.
        decoder = make_decoder(beam_size=2, patience=2.0)
        decoder.finished_sequences = [
            {(1, EOT): -1.0, (2, EOT): -2.0, (3, EOT): -3.0, (4, EOT): -4.0}
        ]

        tokens, scores = finalize(decoder, [[[5, 6, 7], [8, 9, 1]]], [[-0.1, -0.2]])

        self.assertEqual(len(tokens[0]), 4)
        self.assertEqual(scores[0], [-1.0, -2.0, -3.0, -4.0])

    def test_candidates_do_not_leak_between_audio_items(self):
        decoder = make_decoder(beam_size=2)
        decoder.finished_sequences = [
            {(1, 2, EOT): -1.0, (3, 4, EOT): -2.0},
            {(5, 6, EOT): -3.0, (7, 8, EOT): -4.0},
        ]

        tokens, scores = finalize(
            decoder,
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]],
            [[-0.1, -0.2], [-0.3, -0.4]],
        )

        self.assertEqual(
            [strip_padding(sequence) for sequence in tokens[0]],
            [[1, 2, EOT], [3, 4, EOT]],
        )
        self.assertEqual(
            [strip_padding(sequence) for sequence in tokens[1]],
            [[5, 6, EOT], [7, 8, EOT]],
        )
        self.assertEqual(scores, [[-1.0, -2.0], [-3.0, -4.0]])

    def test_padding_never_introduces_an_empty_candidate(self):
        # Audio items can finish with different candidate counts, and the shorter group
        # is padded to keep the array rectangular. A bare-EOT pad would decode to empty
        # text and divide by a zero length in the ranker, so pads repeat a real candidate.
        decoder = make_decoder(beam_size=2, patience=2.0)
        decoder.finished_sequences = [
            {(1, EOT): -1.0, (2, EOT): -2.0, (3, EOT): -3.0, (4, EOT): -4.0},
            {(5, EOT): -1.5, (6, EOT): -2.5},
        ]

        tokens, scores = finalize(
            decoder,
            [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]]],
            [[-0.1, -0.2], [-0.3, -0.4]],
        )

        self.assertEqual(len(tokens[0]), len(tokens[1]))
        for group in tokens:
            for sequence in group:
                self.assertGreater(len(strip_padding(sequence)), 1)
        for group in scores:
            self.assertTrue(all(np.isfinite(score) for score in group))


class BeamSearchUpdateTests(unittest.TestCase):
    N_VOCAB = 16

    def _logits(self, rows):
        logits = np.full((len(rows), self.N_VOCAB), -20.0, dtype=np.float32)
        for row_index, preferences in enumerate(rows):
            for token, value in preferences.items():
                logits[row_index, token] = value
        return mx.array(logits)

    def test_finished_candidates_are_capped_by_patience(self):
        # Every step here finishes both beams. Without a cap the finished dict grows
        # without bound and finalize has to sort an ever-longer candidate list.
        decoder = make_decoder(beam_size=2, patience=1.0)
        tokens = mx.array([[1, 2], [1, 3]])
        sum_logprobs = mx.array([0.0, 0.0])

        for _ in range(4):
            logits = self._logits([{EOT: 5.0, 7: 4.0}, {EOT: 5.0, 8: 4.0}])
            tokens, _, sum_logprobs = decoder.update(tokens, logits, sum_logprobs)

        self.assertLessEqual(len(decoder.finished_sequences[0]), decoder.max_candidates)

    def test_reset_clears_candidates_between_windows(self):
        decoder = make_decoder(beam_size=2)
        decoder.finished_sequences = [{(1, EOT): -1.0}]

        decoder.reset()

        self.assertIsNone(decoder.finished_sequences)


if __name__ == "__main__":
    unittest.main()
