import unittest

import numpy as np

from mlx_whisperx.vads.coverage import (
    SAMPLE_RATE,
    audible_regions,
    covered_seconds,
    energy_pauses,
    is_repetition_loop,
    speech_level,
)
from mlx_whisperx.vads.vad import Segment, Vad


def tone(seconds: float, amplitude: float = 0.1) -> np.ndarray:
    """A constant-energy signal of the requested duration."""
    return np.full(int(seconds * SAMPLE_RATE), amplitude, dtype=np.float32)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(seconds * SAMPLE_RATE), dtype=np.float32)


class AudibleRegionTests(unittest.TestCase):
    def test_sustained_silence_splits_a_region(self):
        audio = np.concatenate([tone(3), silence(2), tone(3)])

        regions = audible_regions(audio, floor=0.01, min_silent_run=1.0)

        self.assertEqual(len(regions), 2)
        self.assertLess(regions[0][1], regions[1][0])

    def test_brief_dip_does_not_split_a_region(self):
        # A pause between words is shorter than min_silent_run and must not end
        # coverage, or every inter-word gap would become a chunk boundary.
        audio = np.concatenate([tone(3), silence(0.4), tone(3)])

        regions = audible_regions(audio, floor=0.01, min_silent_run=1.0)

        self.assertEqual(len(regions), 1)

    def test_continuous_audio_is_one_region_covering_the_file(self):
        audio = tone(10)

        regions = audible_regions(audio, floor=0.01, min_silent_run=1.0)

        self.assertEqual(len(regions), 1)
        self.assertAlmostEqual(covered_seconds(regions), 10.0, places=1)

    def test_pure_silence_covers_nothing(self):
        self.assertEqual(audible_regions(silence(10), floor=0.01), [])

    def test_quiet_speech_survives_a_floor_scaled_to_it(self):
        # The floor is a fraction of measured speech level precisely so that a
        # quietly mastered file is not discarded wholesale.
        audio = tone(5, amplitude=0.002)

        regions = audible_regions(audio, floor=0.002 * 0.05, min_silent_run=1.0)

        self.assertAlmostEqual(covered_seconds(regions), 5.0, places=1)

    def test_audio_shorter_than_one_frame_is_kept(self):
        regions = audible_regions(tone(0.1), floor=0.01)

        self.assertEqual(len(regions), 1)


class SpeechLevelTests(unittest.TestCase):
    def test_level_is_measured_only_inside_speech_regions(self):
        audio = np.concatenate([silence(5), tone(5, amplitude=0.2)])

        level = speech_level(audio, [Segment(5.0, 10.0)])

        self.assertAlmostEqual(level, 0.2, places=2)

    def test_level_is_zero_without_regions(self):
        self.assertEqual(speech_level(tone(5), []), 0.0)

    def test_tuples_are_accepted_as_regions(self):
        level = speech_level(tone(5, amplitude=0.3), [(0.0, 5.0)])

        self.assertAlmostEqual(level, 0.3, places=2)


class PlanChunkTests(unittest.TestCase):
    def test_chunks_never_exceed_chunk_size(self):
        chunks = Vad.plan_chunks([(0.0, 95.0)], chunk_size=30)

        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 30 + 1e-9)

    def test_chunks_tile_a_region_without_gaps_or_overlaps(self):
        # Downstream assembly concatenates chunk results verbatim, so an overlap
        # duplicates text and a gap drops it.
        chunks = Vad.plan_chunks([(0.0, 95.0)], chunk_size=30)

        self.assertAlmostEqual(chunks[0]["start"], 0.0)
        self.assertAlmostEqual(chunks[-1]["end"], 95.0)
        for previous, current in zip(chunks, chunks[1:]):
            self.assertAlmostEqual(previous["end"], current["start"])

    def test_seam_lands_in_a_pause_when_one_is_available(self):
        hints = [Segment(0.0, 24.0), Segment(26.0, 50.0)]

        chunks = Vad.plan_chunks([(0.0, 50.0)], chunk_size=30, cut_hints=hints)

        self.assertAlmostEqual(chunks[0]["end"], 25.0, places=3)

    def test_region_without_any_pause_is_still_split(self):
        # Continuous speech has no good cut point, but an oversized chunk would
        # overflow the decoder window, so it must be split regardless.
        chunks = Vad.plan_chunks([(0.0, 90.0)], chunk_size=30, cut_hints=[])

        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(chunk["end"] - chunk["start"], 30 + 1e-9)

    def test_silence_between_regions_is_excluded(self):
        chunks = Vad.plan_chunks([(0.0, 10.0), (60.0, 70.0)], chunk_size=30)

        self.assertEqual(len(chunks), 2)
        self.assertAlmostEqual(chunks[0]["end"], 10.0)
        self.assertAlmostEqual(chunks[1]["start"], 60.0)

    def test_chunk_spans_are_contiguous_not_speech_only(self):
        # `segments` drives intra-chunk audio assembly; one span per chunk keeps
        # the audio between speech turns rather than splicing it out.
        chunks = Vad.plan_chunks([(0.0, 20.0)], chunk_size=30)

        self.assertEqual(chunks[0]["segments"], [(0.0, 20.0)])

    def test_no_regions_yields_no_chunks(self):
        self.assertEqual(Vad.plan_chunks([], chunk_size=30), [])

    def test_explicit_pauses_take_priority_over_vad_hints(self):
        # VAD hints would put the seam at 25.0; the energy pause at 28.0 is the
        # better cut and must win.
        hints = [Segment(0.0, 24.0), Segment(26.0, 50.0)]

        chunks = Vad.plan_chunks([(0.0, 50.0)], chunk_size=30, cut_hints=hints, pauses=[28.0])

        self.assertAlmostEqual(chunks[0]["end"], 28.0, places=3)

    def test_vad_hints_are_used_when_no_pauses_are_supplied(self):
        hints = [Segment(0.0, 24.0), Segment(26.0, 50.0)]

        chunks = Vad.plan_chunks([(0.0, 50.0)], chunk_size=30, cut_hints=hints, pauses=[])

        self.assertAlmostEqual(chunks[0]["end"], 25.0, places=3)

    def test_pause_midpoints_ignore_gaps_that_are_too_short(self):
        hints = [Segment(0.0, 10.0), Segment(10.05, 20.0), Segment(25.0, 30.0)]

        self.assertEqual(Vad.pause_midpoints(hints, min_pause=0.15), [22.5])


class EnergyPauseTests(unittest.TestCase):
    def test_pause_is_found_at_its_midpoint(self):
        audio = np.concatenate([tone(5), silence(1.0), tone(5)])

        pauses = energy_pauses(audio, floor=0.01, min_pause=0.3)

        self.assertEqual(len(pauses), 1)
        self.assertAlmostEqual(pauses[0], 5.5, delta=0.1)

    def test_gap_shorter_than_min_pause_is_ignored(self):
        audio = np.concatenate([tone(5), silence(0.1), tone(5)])

        self.assertEqual(energy_pauses(audio, floor=0.01, min_pause=0.3), [])

    def test_every_pause_in_a_sequence_is_reported(self):
        audio = np.concatenate([tone(2), silence(0.5), tone(2), silence(0.5), tone(2)])

        self.assertEqual(len(energy_pauses(audio, floor=0.01, min_pause=0.3)), 2)

    def test_continuous_speech_offers_no_seam(self):
        self.assertEqual(energy_pauses(tone(10), floor=0.01), [])

    def test_pauses_are_found_where_a_vad_reports_nothing(self):
        # The whole point: seam candidates must exist inside passages a VAD
        # rejected, since it cannot nominate positions it never detected.
        quiet_speech = tone(4, amplitude=0.004)
        audio = np.concatenate([quiet_speech, silence(0.8), quiet_speech])

        pauses = energy_pauses(audio, floor=0.004 * 0.2, min_pause=0.3)

        self.assertEqual(len(pauses), 1)


class RepetitionLoopTests(unittest.TestCase):
    def test_single_token_repeated_is_a_loop(self):
        self.assertTrue(is_repetition_loop(" ".join(["Humanity"] * 112)))
        self.assertTrue(is_repetition_loop(" ".join(["Ten"] * 224)))

    def test_ordinary_dialogue_is_not_a_loop(self):
        self.assertFalse(
            is_repetition_loop("What are you doing? Impossible. How are you still alive?")
        )

    def test_emphatic_repetition_in_real_speech_is_kept(self):
        # Speech does repeat; the bar has to sit above what dialogue actually does.
        self.assertFalse(is_repetition_loop("No, no, no, I told you that is not what happened"))

    def test_short_text_is_never_a_loop(self):
        # "Yeah, yeah, yeah" is three of three tokens but plainly not a decoder loop.
        self.assertFalse(is_repetition_loop("yeah yeah yeah"))

    def test_empty_text_is_not_a_loop(self):
        self.assertFalse(is_repetition_loop(""))

    def test_punctuation_and_case_do_not_hide_a_loop(self):
        self.assertTrue(is_repetition_loop("Ten, ten. TEN! ten ten ten, ten ten ten ten."))


class ChunkTextGateTests(unittest.TestCase):
    def pipeline(self, **overrides):
        from unittest import mock

        from mlx_whisperx.pipeline import MLXWhisperXPipeline, PipelineOptions

        with mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=mock.Mock()):
            return MLXWhisperXPipeline(PipelineOptions(**overrides))

    def test_ordinary_speech_is_kept(self):
        pipeline = self.pipeline()

        self.assertTrue(
            pipeline._keep_chunk_text({"text": "Hello there, how are you?", "no_speech_prob": 0.1}, 0.0)
        )

    def test_repetition_loop_is_dropped_even_when_scored_as_speech(self):
        # The decoder's own retry never fires here, so the gate is the only thing
        # standing between a music-driven loop and the output file.
        pipeline = self.pipeline()

        self.assertFalse(
            pipeline._keep_chunk_text(
                {"text": " ".join(["Ten"] * 224), "no_speech_prob": 0.486}, 1391.25
            )
        )

    def test_text_over_detected_non_speech_is_dropped(self):
        pipeline = self.pipeline()

        self.assertFalse(
            pipeline._keep_chunk_text({"text": "A line of dialogue", "no_speech_prob": 0.9}, 0.0)
        )

    def test_empty_text_is_dropped(self):
        pipeline = self.pipeline()

        self.assertFalse(pipeline._keep_chunk_text({"text": "   ", "no_speech_prob": 0.1}, 0.0))

    def test_disabling_the_no_speech_threshold_keeps_the_loop_guard(self):
        pipeline = self.pipeline(no_speech_threshold=None)

        self.assertTrue(
            pipeline._keep_chunk_text({"text": "A line of dialogue", "no_speech_prob": 0.9}, 0.0)
        )
        self.assertFalse(
            pipeline._keep_chunk_text({"text": " ".join(["Ten"] * 50), "no_speech_prob": 0.9}, 0.0)
        )


class CoverageIsOptInTests(unittest.TestCase):
    """Coverage roughly doubles decode work for a small recall gain, so it is a
    deliberate choice rather than the default. The loop guard is not: it repairs
    output the decoder gets wrong, and costs nothing, so it runs on both paths.
    """

    def chunks(self, audio: np.ndarray, **overrides) -> list[dict]:
        from unittest import mock

        from mlx_whisperx.pipeline import MLXWhisperXPipeline, PipelineOptions

        vad_chunk = {"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}

        class FakeSilero:
            def __init__(self, **kwargs):
                pass

            @staticmethod
            def preprocess_audio(audio):
                return audio

            @staticmethod
            def merge_chunks(segments, chunk_size, onset=0.5, offset=None):
                return [dict(vad_chunk)]

            def __call__(self, audio, **kwargs):
                return [Segment(0.0, 1.0)]

        with (
            mock.patch("mlx_whisperx.pipeline.import_mlx_whisper", return_value=mock.Mock()),
            mock.patch("mlx_whisperx.pipeline.get_vad_class", return_value=FakeSilero),
        ):
            pipeline = MLXWhisperXPipeline(PipelineOptions(**overrides))
            return pipeline._vad_chunks(audio)

    def test_coverage_is_off_by_default(self):
        from mlx_whisperx.pipeline import PipelineOptions

        self.assertFalse(PipelineOptions().coverage)

    def test_default_run_chunks_from_the_vad_alone(self):
        # Ten seconds audible but only the first second flagged by the VAD. The
        # default must hand back the VAD's answer, untouched.
        audio = tone(10.0)

        chunks = self.chunks(audio)

        self.assertEqual(chunks, [{"start": 0.0, "end": 1.0, "segments": [(0.0, 1.0)]}])

    def test_opting_in_covers_audio_the_vad_missed(self):
        audio = tone(10.0)

        chunks = self.chunks(audio, coverage=True)

        self.assertGreater(covered_seconds((c["start"], c["end"]) for c in chunks), 9.0)

    def test_loop_guard_runs_on_both_paths(self):
        from unittest import mock

        from mlx_whisperx.pipeline import MLXWhisperXPipeline, PipelineOptions

        loop = {"text": " ".join(["Poetry"] * 40), "no_speech_prob": 0.7}
        for coverage in (False, True):
            with mock.patch(
                "mlx_whisperx.pipeline.import_mlx_whisper", return_value=mock.Mock()
            ):
                pipeline = MLXWhisperXPipeline(PipelineOptions(coverage=coverage))
            with self.subTest(coverage=coverage):
                self.assertFalse(pipeline._keep_chunk_text(loop, 0.0))


if __name__ == "__main__":
    unittest.main()
