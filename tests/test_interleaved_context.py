import unittest
from unittest import mock

import numpy as np

from mlx_whisperx.pipeline import (
    MLXWhisperXPipeline,
    PipelineOptions,
    _extend_context,
    _interleave_streams,
    _rebase_progress,
    _split_streams,
)

SAMPLE_RATE = 16000
SILENT = {"no_speech_prob": 0.9, "avg_logprob": -2.0}


class StreamPlanningTests(unittest.TestCase):
    def test_streams_are_contiguous_and_earlier_streams_take_the_remainder(self):
        self.assertEqual(_split_streams(5, 2), [[0, 1, 2], [3, 4]])
        self.assertEqual(_split_streams(3, 3), [[0], [1], [2]])
        self.assertEqual(_split_streams(4, 1), [[0, 1, 2, 3]])

    def test_each_batch_holds_the_next_chunk_of_every_stream(self):
        self.assertEqual(
            _interleave_streams([[0, 1, 2], [3, 4]]),
            [[(0, 0), (1, 3)], [(0, 1), (1, 4)], [(0, 2)]],
        )

    def test_context_keeps_only_the_newest_tokens(self):
        self.assertEqual(_extend_context([1, 2], [3, 4], limit=3), [2, 3, 4])
        self.assertEqual(_extend_context([1, 2], [3], limit=0), [])

    def test_progress_is_reported_against_the_whole_pass(self):
        received = []
        callback = _rebase_progress(
            received.append, batch_index=2, total_batches=5, windows_before=6, total_windows=10
        )

        callback({"batch_index": 0, "total_batches": 1, "windows_done": 3, "total_windows": 3, "decode_seconds": 1.0})

        self.assertEqual(
            received,
            [{"batch_index": 2, "total_batches": 5, "windows_done": 9, "total_windows": 10, "decode_seconds": 1.0}],
        )
        self.assertIsNone(
            _rebase_progress(None, batch_index=0, total_batches=1, windows_before=0, total_windows=1)
        )


class FakeBackend:
    """Batched backend that names each chunk by the second its audio starts at.

    `overrides` maps `(chunk, call_index)` to extra result fields, where `call_index`
    counts backend calls from 0, so a test can shape one specific decode. Calls from
    `redo_call` on are the wrap-around pass and tag their text with "-redo".
    """

    def __init__(self, overrides=None, redo_call=None):
        self.overrides = overrides or {}
        self.redo_call = redo_call
        self.calls = []

    def detect_language(self, *args, **kwargs):
        return "en"

    def transcribe_chunks_batched(self, audio_list, *, batch_size, context_tokens, progress_callback=None, **kwargs):
        call_index = len(self.calls)
        chunks = [int(audio[0]) // SAMPLE_RATE for audio in audio_list]
        self.calls.append(
            {"chunks": chunks, "contexts": [list(context) for context in context_tokens], "batch_size": batch_size}
        )
        results = []
        for chunk in chunks:
            result = {
                "start": 0.0,
                "end": 0.5,
                "text": f"c{chunk}-redo" if self.redo_call is not None and call_index >= self.redo_call else f"c{chunk}",
                "tokens": [chunk + 1],
                "avg_logprob": -0.1,
                "no_speech_prob": 0.0,
                "temperature": 0.0,
            }
            result.update(self.overrides.get((chunk, call_index), {}))
            results.append(result)
        return results


def run_interleaved(n_chunks, batch_size, backend):
    audio = np.arange(SAMPLE_RATE * (n_chunks + 1), dtype=np.float32)
    chunks = [{"start": float(i), "end": i + 0.5, "segments": [(float(i), i + 0.5)]} for i in range(n_chunks)]
    with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=backend):
        pipeline = MLXWhisperXPipeline(
            PipelineOptions(batch_size=batch_size, interleaved_context=True, language="en")
        )
    return pipeline._asr(audio, chunks)


def texts(result):
    return [segment["text"] for segment in result["segments"]]


class InterleavedContextTests(unittest.TestCase):
    def test_streams_carry_their_own_history_and_later_streams_wrap_around(self):
        backend = FakeBackend(redo_call=3)

        result = run_interleaved(5, 2, backend)

        self.assertEqual(
            backend.calls,
            [
                {"chunks": [0, 3], "contexts": [[], []], "batch_size": 2},
                {"chunks": [1, 4], "contexts": [[1], [4]], "batch_size": 2},
                {"chunks": [2], "contexts": [[1, 2]], "batch_size": 1},
                # Stream 1's first chunk is re-decoded with the tail of stream 0.
                {"chunks": [3], "contexts": [[1, 2, 3]], "batch_size": 1},
            ],
        )
        self.assertEqual(texts(result), ["c0", "c1", "c2", "c3-redo", "c4"])
        self.assertEqual([segment["start"] for segment in result["segments"]], [0.0, 1.0, 2.0, 3.0, 4.0])

    def test_rejected_redo_keeps_the_first_pass_text(self):
        backend = FakeBackend(overrides={(3, 3): SILENT}, redo_call=3)

        result = run_interleaved(5, 2, backend)

        self.assertEqual(texts(result), ["c0", "c1", "c2", "c3", "c4"])

    def test_redo_cannot_bring_back_a_chunk_the_first_pass_dropped(self):
        backend = FakeBackend(overrides={(3, 0): SILENT}, redo_call=3)

        result = run_interleaved(5, 2, backend)

        self.assertEqual(texts(result), ["c0", "c1", "c2", "c4"])

    def test_hot_retries_and_loops_reset_history(self):
        backend = FakeBackend(
            overrides={
                (0, 0): {"temperature": 0.8},
                (3, 0): {"text": "la " * 20},
            }
        )

        run_interleaved(6, 2, backend)

        self.assertEqual(backend.calls[1]["contexts"], [[], []])
        self.assertEqual(backend.calls[2]["contexts"], [[2], [5]])

    def test_silent_chunk_leaves_the_stream_history_unchanged(self):
        backend = FakeBackend(overrides={(1, 1): SILENT})

        run_interleaved(6, 2, backend)

        self.assertEqual(backend.calls[1]["contexts"], [[1], [4]])
        self.assertEqual(backend.calls[2]["contexts"], [[1], [4, 5]])

    def test_batch_size_one_is_a_single_serial_stream(self):
        backend = FakeBackend()

        result = run_interleaved(3, 1, backend)

        self.assertEqual(
            [call["contexts"] for call in backend.calls],
            [[[]], [[1]], [[1, 2]]],
        )
        self.assertEqual(texts(result), ["c0", "c1", "c2"])

    def test_more_streams_than_chunk_pairs_warns(self):
        with self.assertLogs("mlx_whisperx.pipeline", level="WARNING") as logs:
            run_interleaved(3, 8, FakeBackend())

        self.assertTrue(any("smaller batch_size" in line for line in logs.output))

    def test_no_vad_warns_that_interleaved_context_has_no_effect(self):
        backend = mock.Mock()
        backend.transcribe.return_value = {"segments": [], "language": "en"}
        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=backend):
            pipeline = MLXWhisperXPipeline(
                PipelineOptions(no_vad=True, interleaved_context=True, language="en")
            )

        with self.assertLogs("mlx_whisperx.pipeline", level="WARNING") as logs:
            pipeline._asr(np.zeros(SAMPLE_RATE, dtype=np.float32), [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}])

        self.assertTrue(any("no effect with no_vad" in line for line in logs.output))
        backend.transcribe_chunks_batched.assert_not_called()


if __name__ == "__main__":
    unittest.main()
