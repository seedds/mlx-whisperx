from typing import Callable, List, Optional, TypedDict


ProgressCallback = Optional[Callable[[float], None]]


try:
    from typing import NotRequired
except ImportError:  # pragma: no cover
    from typing_extensions import NotRequired


class SingleWordSegment(TypedDict):
    word: str
    start: NotRequired[float]
    end: NotRequired[float]
    score: NotRequired[float]
    speaker: NotRequired[str]


class SingleCharSegment(TypedDict):
    char: str
    start: NotRequired[float]
    end: NotRequired[float]
    score: NotRequired[float]


class SingleSegment(TypedDict):
    start: float
    end: float
    text: str
    avg_logprob: NotRequired[float]
    speaker: NotRequired[str]
    words: NotRequired[List[SingleWordSegment]]
    chars: NotRequired[List[SingleCharSegment]]


class TranscriptionResult(TypedDict):
    segments: List[SingleSegment]
    language: str
    text: NotRequired[str]
    word_segments: NotRequired[List[SingleWordSegment]]
