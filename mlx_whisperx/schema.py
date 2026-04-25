"""TypedDict definitions documenting the WhisperX-style JSON result shape."""

from typing import Callable, List, Optional, TypedDict


ProgressCallback = Optional[Callable[[float], None]]


try:
    from typing import NotRequired
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired


class SingleWordSegment(TypedDict):
    """Word-level alignment result.

    `start`, `end`, and `score` are absent when alignment could not confidently map
    a word to audio. `speaker` is only present after diarization.
    """

    word: str
    start: NotRequired[float]
    end: NotRequired[float]
    score: NotRequired[float]
    speaker: NotRequired[str]


class SingleCharSegment(TypedDict):
    """Optional character-level alignment returned when requested by the caller."""

    char: str
    start: NotRequired[float]
    end: NotRequired[float]
    score: NotRequired[float]


class SingleSegment(TypedDict):
    """Transcript segment with optional word, character, and speaker metadata."""

    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]
    speaker: NotRequired[str]
    words: NotRequired[List[SingleWordSegment]]
    chars: NotRequired[List[SingleCharSegment]]


class TranscriptionResult(TypedDict):
    """Top-level result emitted by the Python API and JSON writer."""

    segments: List[SingleSegment]
    language: str
    text: NotRequired[str]
    word_segments: NotRequired[List[SingleWordSegment]]
