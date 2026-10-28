import types
import unittest

import numpy as np

from mlx_whisperx.backend.mlx_whisper.decoding import DecodingTask, MaximumLikelihoodRanker

TEXTS = {
    1: "Do" + "-o" * 100,
    2: "Doctor! Save your questions until... I have none, sir.",
    3: "DOCTOR! Save your questions until- I have none, sir!",
    4: "Doctoooo" + "r" * 100,
}


def fake_task(threshold, n_group):
    tokenizer = types.SimpleNamespace(decode=lambda tokens: " " + " ".join(TEXTS[t] for t in tokens))
    return types.SimpleNamespace(
        options=types.SimpleNamespace(compression_ratio_threshold=threshold),
        n_group=n_group,
        tokenizer=tokenizer,
    )


def select(threshold, tokens, sum_logprobs):
    task = fake_task(threshold, n_group=len(tokens[0]))
    scores = DecodingTask._demote_repetition_loops(task, tokens, sum_logprobs)
    return MaximumLikelihoodRanker(None).rank(tokens, scores)


class CandidateRankingTests(unittest.TestCase):
    def test_a_loop_does_not_outrank_real_text(self):
        # Loops repeat tokens the decoder is sure of, so they carry the best score.
        tokens = [[[1], [2], [4], [3]]]
        sum_logprobs = [[-0.12, -0.38, -0.12, -0.40]]

        self.assertEqual(select(2.4, tokens, sum_logprobs), [1])

    def test_without_a_threshold_ranking_is_on_likelihood_alone(self):
        tokens = [[[1], [2]]]
        sum_logprobs = [[-0.12, -0.38]]

        self.assertEqual(select(None, tokens, sum_logprobs), [0])

    def test_the_best_loop_wins_when_every_candidate_loops(self):
        tokens = [[[4], [1]]]
        sum_logprobs = [[-0.5, -0.12]]

        self.assertEqual(select(2.4, tokens, sum_logprobs), [1])

    def test_each_audio_item_is_ranked_on_its_own_candidates(self):
        tokens = [[[1], [2]], [[1], [4]]]
        sum_logprobs = [[-0.12, -0.38], [-0.12, -0.5]]

        self.assertEqual(select(2.4, tokens, sum_logprobs), [1, 0])

    def test_scores_returned_for_non_looping_candidates_are_unchanged(self):
        task = fake_task(2.4, n_group=2)

        scores = DecodingTask._demote_repetition_loops(task, [[[1], [2]]], [[-0.12, -0.38]])

        self.assertEqual(scores, [[-np.inf, -0.38]])

    def test_a_single_candidate_is_never_rescored(self):
        task = fake_task(2.4, n_group=1)
        sum_logprobs = [[-0.12]]

        self.assertIs(DecodingTask._demote_repetition_loops(task, [[[1]]], sum_logprobs), sum_logprobs)


if __name__ == "__main__":
    unittest.main()
