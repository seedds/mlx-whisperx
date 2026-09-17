"""Shared VAD primitives used by Silero and pyannote backends."""

from dataclasses import dataclass
from typing import Optional

import numpy as np


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
    def split_long_segments(segments, chunk_size: int) -> list["Segment"]:
        """Split speech turns longer than `chunk_size` into decoder-sized pieces.

        A single VAD turn can exceed the ASR window when a backend returns its own
        speech timeline instead of raw scores. Merging alone never splits one turn, so
        the oversized turn would be trimmed to the decoder window and lose its tail.
        """
        if chunk_size <= 0:
            return list(segments)

        split: list[Segment] = []
        for segment in segments:
            start = segment.start
            while segment.end - start > chunk_size:
                split.append(Segment(start, start + chunk_size, segment.speaker))
                start += chunk_size
            split.append(Segment(start, segment.end, segment.speaker))
        return split

    @staticmethod
    def pause_midpoints(segments, min_pause: float = 0.15) -> list[float]:
        """Midpoint of each pause between consecutive speech turns.

        Snapping a seam to a speech *edge* still cuts flush against an
        utterance; the middle of a pause leaves air on both sides.
        """
        turns = sorted(
            (float(s.start), float(s.end)) if hasattr(s, "start") else (float(s[0]), float(s[1]))
            for s in segments
        )
        return [
            round((previous[1] + current[0]) / 2, 3)
            for previous, current in zip(turns, turns[1:])
            if current[0] - previous[1] >= min_pause
        ]

    @staticmethod
    def plan_chunks(
        regions,
        chunk_size: int,
        cut_hints=None,
        min_chunk: float = 10.0,
        pauses=None,
    ) -> list[dict]:
        """Group coverage regions into decoder-sized chunks, cutting at pauses.

        `regions` is what must be transcribed. `pauses` are candidate seam
        positions measured from audio energy, and `cut_hints` are VAD speech
        turns used as a fallback source of the same. Neither decides what gets
        transcribed, only where the boundaries sit.

        Energy pauses are preferred because a VAD that missed a passage cannot
        nominate seam positions inside it -- exactly where a blind cut then
        lands, corrupting the words on both sides.

        Unlike `merge_chunks`, every chunk is a single contiguous span: the
        audio between speech turns is kept rather than spliced out. Chunks are
        emitted without overlap because downstream assembly concatenates chunk
        results verbatim, so any overlap would duplicate text.
        """
        if chunk_size <= 0:
            return []

        hint_edges = sorted(pauses) if pauses else Vad.pause_midpoints(cut_hints or [])
        chunks: list[dict] = []

        def emit(start: float, end: float) -> None:
            """Append [start, end), splitting anything longer than chunk_size."""
            if end - start <= chunk_size:
                if end > start:
                    chunks.append({"start": start, "end": end, "segments": [(start, end)]})
                return

            # Prefer to cut where speech pauses: a seam inside an utterance
            # corrupts the words on both sides of it. Take the latest pause that
            # still leaves a workable chunk, falling back to an arbitrary cut
            # only for a stretch that contains no pause at all.
            floor = min(min_chunk, chunk_size * 0.6)
            edges, position = [start], start
            while end - position > chunk_size:
                window = [
                    edge for edge in hint_edges if position + floor <= edge <= position + chunk_size
                ]
                position = max(window) if window else position + chunk_size
                edges.append(position)

            # Divide the tail evenly rather than leaving a runt behind it.
            tail = end - edges[-1]
            if tail > chunk_size:
                parts = int(np.ceil(tail / chunk_size))
                step = tail / parts
                edges += [edges[-1] + step * k for k in range(1, parts)]
            edges.append(end)

            for low, high in zip(edges, edges[1:]):
                if high > low:
                    chunks.append({"start": low, "end": high, "segments": [(low, high)]})

        current_start: Optional[float] = None
        current_end: Optional[float] = None
        for region_start, region_end in regions:
            if current_start is None:
                current_start, current_end = region_start, region_end
                continue
            if region_end - current_start > chunk_size:
                # Close the current chunk if it can stand alone; otherwise absorb
                # this region and let `emit` split the combined span.
                if current_end - current_start >= min_chunk:
                    emit(current_start, current_end)
                    current_start, current_end = region_start, region_end
                else:
                    emit(current_start, region_end)
                    current_start, current_end = None, None
            else:
                current_end = region_end

        if current_start is not None:
            emit(current_start, current_end)
        return chunks

    @staticmethod
    def merge_chunks(segments, chunk_size: int, onset: float, offset: Optional[float]):
        """Merge adjacent speech segments into ASR chunks capped by `chunk_size`.

        The ASR backend performs best on bounded windows. This method groups VAD turns
        into chunks while retaining the original turn boundaries in `segments` for
        debugging and VAD dump output.
        """
        if len(segments) == 0:
            return []

        segments = Vad.split_long_segments(segments, chunk_size)
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
