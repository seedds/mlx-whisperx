"""Shared VAD primitives used by Silero and pyannote backends."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    """Single speech activity interval produced by a VAD backend."""

    start: float
    end: float
    speaker: Optional[str] = None


class Vad:
    """Base class containing validation and chunk-merging behavior."""

    def __init__(self, vad_onset: float):
        if not 0 < vad_onset < 1:
            raise ValueError("vad_onset must be between 0 and 1")

    @staticmethod
    def preprocess_audio(audio):
        """Convert input audio to the representation required by the backend."""
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size: int, onset: float, offset: Optional[float]):
        """Merge adjacent speech segments into ASR chunks capped by `chunk_size`.

        The ASR backend performs best on bounded windows. This method groups VAD turns
        into chunks while retaining the original turn boundaries in `segments` for
        debugging and VAD dump output.
        """
        if len(segments) == 0:
            return []

        curr_end = 0.0
        curr_start = segments[0].start
        seg_idxs: list[tuple[float, float]] = []
        merged_segments: list[dict] = []

        for segment in segments:
            if segment.end - curr_start > chunk_size and curr_end - curr_start > 0:
                # Close the current chunk before adding a VAD turn that would make the
                # chunk exceed the requested duration.
                merged_segments.append(
                    {"start": curr_start, "end": curr_end, "segments": seg_idxs}
                )
                curr_start = segment.start
                seg_idxs = []
            curr_end = segment.end
            seg_idxs.append((segment.start, segment.end))

        merged_segments.append({"start": curr_start, "end": curr_end, "segments": seg_idxs})
        return merged_segments
