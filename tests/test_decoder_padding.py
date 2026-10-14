import unittest
from dataclasses import replace

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from mlx_whisperx.backend.mlx_whisper.decoding import DecodingOptions, DecodingTask, decode
from mlx_whisperx.backend.mlx_whisper.whisper import ModelDimensions, Whisper

EOT = 50256

# The English-only vocabulary keeps the real tokenizer's special token ids valid while
# the rest of the model stays tiny enough to run in milliseconds.
DIMS = ModelDimensions(
    n_mels=80,
    n_audio_ctx=24,
    n_audio_state=32,
    n_audio_head=2,
    n_audio_layer=1,
    n_vocab=51864,
    n_text_ctx=64,
    n_text_state=32,
    n_text_head=2,
    # Two layers are needed to expose NaNs leaking from padded rows: with one layer
    # only the unused padded outputs would be affected.
    n_text_layer=2,
)


def make_model(dtype=mx.float32) -> Whisper:
    mx.random.seed(0)
    model = Whisper(DIMS, dtype)
    # The decoder starts with zero positional embeddings, which would hide position bugs.
    model.decoder.positional_embedding = mx.random.normal((DIMS.n_text_ctx, DIMS.n_text_state))
    if dtype != mx.float32:
        model.update(tree_map(lambda p: p.astype(dtype), model.parameters()))
    return model


def audio_features(n: int, dtype=mx.float32) -> mx.array:
    # Pre-encoded features let `decode` skip the audio encoder.
    return mx.random.normal((n, DIMS.n_audio_ctx, DIMS.n_audio_state), key=mx.random.key(1)).astype(dtype)


class PaddedDecoderTests(unittest.TestCase):
    ROWS = [[50257, 11, 12, 13, 14, 15], [50257, 21, 22], [50257, 31, 32, 33, 34]]

    def _padded_vs_single(self, dtype, atol):
        model = make_model(dtype)
        features = audio_features(len(self.ROWS), dtype)
        width = max(len(row) for row in self.ROWS)
        pads = [width - len(row) for row in self.ROWS]
        batch = mx.array([[EOT] * pad + row for pad, row in zip(pads, self.ROWS)])
        pad_lengths = mx.array(pads)

        logits, cache, _ = model.decoder(batch, features, pad_lengths=pad_lengths)
        steps = [logits[:, -1]]
        for token in (101, 102):
            step = mx.full((len(self.ROWS), 1), token)
            logits, cache, _ = model.decoder(step, features, kv_cache=cache, pad_lengths=pad_lengths)
            steps.append(logits[:, -1])

        for index, row in enumerate(self.ROWS):
            # Fresh caches per row: the decoder fills the passed cache list in place.
            logits, cache, _ = model.decoder(mx.array([row]), features[index : index + 1])
            expected = [logits[:, -1]]
            for token in (101, 102):
                logits, cache, _ = model.decoder(mx.full((1, 1), token), features[index : index + 1], kv_cache=cache)
                expected.append(logits[:, -1])

            for got, want in zip(steps, expected):
                got_row = np.array(got[index].astype(mx.float32))
                want_row = np.array(want[0].astype(mx.float32))
                self.assertTrue(np.isfinite(got_row).all())
                np.testing.assert_allclose(got_row, want_row, atol=atol, rtol=0)

    def test_padded_rows_match_single_row_decoding_fp32(self):
        self._padded_vs_single(mx.float32, atol=1e-4)

    def test_padded_rows_match_single_row_decoding_fp16(self):
        self._padded_vs_single(mx.float16, atol=2e-2)


class PerItemPromptDecodeTests(unittest.TestCase):
    PROMPTS = [[], [1, 2, 3, 4, 5], [7, 8]]

    def _options(self, **kwargs) -> DecodingOptions:
        return DecodingOptions(
            # An explicit language skips detection, which English-only vocabularies lack.
            language="en",
            fp16=False,
            without_timestamps=True,
            sample_len=6,
            # Suppressing EOT forces every row through all sampled steps.
            suppress_tokens=[EOT],
            **kwargs,
        )

    def _assert_batched_matches_single(self, options: DecodingOptions):
        model = make_model()
        features = audio_features(len(self.PROMPTS))
        batched = decode(model, features, replace(options, prompts=self.PROMPTS))
        for index, prompt in enumerate(self.PROMPTS):
            single = decode(model, features[index], replace(options, prompt=prompt))
            self.assertEqual(batched[index].tokens, single.tokens)
            self.assertAlmostEqual(batched[index].no_speech_prob, single.no_speech_prob, places=4)

    def test_greedy_per_item_prompts_match_single_decodes(self):
        self._assert_batched_matches_single(self._options())

    def test_beam_search_per_item_prompts_match_single_decodes(self):
        self._assert_batched_matches_single(self._options(beam_size=2))

    def test_equal_length_prompts_take_the_unpadded_path(self):
        task = DecodingTask(make_model(), self._options(prompts=[[1, 2], [3, 4]]))
        self.assertEqual(task.pad_lengths, [0, 0])

    def test_prompt_and_prompts_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            DecodingTask(make_model(), self._options(prompt=[1], prompts=[[2]]))

    def test_prompt_count_must_match_the_batch(self):
        model = make_model()
        with self.assertRaises(ValueError):
            decode(model, audio_features(2), self._options(prompts=[[1], [2], [3]]))


if __name__ == "__main__":
    unittest.main()
