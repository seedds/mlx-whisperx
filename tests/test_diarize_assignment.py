import types
import unittest

import numpy as np
import pandas as pd

from mlx_whisperx.diarize import DiarizationPipeline, IntervalTree, assign_word_speakers


class DiarizeAssignmentTests(unittest.TestCase):
    def test_find_nearest_uses_interval_distance_not_midpoint(self):
        tree = IntervalTree([(0.0, 100.0, "A"), (102.0, 103.0, "B")])

        self.assertEqual(tree.find_nearest(100.1), "A")

    def test_words_without_overlap_are_left_unlabelled_by_default(self):
        diarize_df = pd.DataFrame([{"start": 0.0, "end": 1.0, "speaker": "A"}])
        result = {"segments": [{"start": 5.0, "end": 6.0, "text": "hi", "words": [{"word": "hi", "start": 5.0, "end": 6.0}]}]}

        assign_word_speakers(diarize_df, result)

        self.assertNotIn("speaker", result["segments"][0]["words"][0])


class DiarizationProgressTests(unittest.TestCase):
    def _pipeline(self, hook_calls):
        class FakeModel:
            def __call__(self, audio_data, hook=None, **kwargs):
                self.hook = hook
                for call in hook_calls:
                    if hook is not None:
                        hook(*call[:2], **call[2])
                annotation = types.SimpleNamespace(itertracks=lambda yield_label: iter([]), labels=lambda: [])
                return types.SimpleNamespace(speaker_diarization=annotation)

        pipeline = DiarizationPipeline.__new__(DiarizationPipeline)
        pipeline._torch = types.SimpleNamespace(from_numpy=lambda array: array)
        pipeline.model = FakeModel()
        return pipeline

    def test_progress_spans_both_pyannote_steps_and_never_goes_back(self):
        pipeline = self._pipeline(
            [
                ("segmentation", None, {"total": 4, "completed": 2}),
                ("segmentation", None, {"total": 4, "completed": 4}),
                ("embeddings", None, {"total": 10, "completed": 0}),
                ("embeddings", None, {"total": 10, "completed": 5}),
                ("speaker_counting", None, {}),
            ]
        )
        received = []

        pipeline(np.zeros(16000, dtype=np.float32), progress_callback=received.append)

        self.assertEqual(received, [25.0, 50.0, 74.5, 100.0])

    def test_no_hook_is_passed_without_a_callback(self):
        pipeline = self._pipeline([])

        pipeline(np.zeros(16000, dtype=np.float32))

        self.assertIsNone(pipeline.model.hook)
