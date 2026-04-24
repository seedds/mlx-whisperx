from dataclasses import dataclass
from typing import Optional


@dataclass
class Segment:
    start: float
    end: float
    speaker: Optional[str] = None


class Vad:
    def __init__(self, vad_onset: float):
        if not 0 < vad_onset < 1:
            raise ValueError("vad_onset must be between 0 and 1")

    @staticmethod
    def preprocess_audio(audio):
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size: int, onset: float, offset: Optional[float]):
        if len(segments) == 0:
            return []

        curr_end = 0.0
        curr_start = segments[0].start
        seg_idxs: list[tuple[float, float]] = []
        merged_segments: list[dict] = []

        for segment in segments:
            if segment.end - curr_start > chunk_size and curr_end - curr_start > 0:
                merged_segments.append(
                    {"start": curr_start, "end": curr_end, "segments": seg_idxs}
                )
                curr_start = segment.start
                seg_idxs = []
            curr_end = segment.end
            seg_idxs.append((segment.start, segment.end))

        merged_segments.append({"start": curr_start, "end": curr_end, "segments": seg_idxs})
        return merged_segments
