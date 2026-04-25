"""Public package API for the MLX-backed WhisperX-style pipeline.

The package intentionally exposes a small surface area: high-level transcription,
audio loading, forced alignment helpers, and diarization helpers. Most users should
call :func:`transcribe`; the lower-level helpers are exported for advanced workflows
that want to run individual stages manually.
"""

from .alignment import align, load_align_model
from .audio import SAMPLE_RATE, load_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .transcribe import transcribe

__all__ = [
    "SAMPLE_RATE",
    "DiarizationPipeline",
    "align",
    "assign_word_speakers",
    "load_align_model",
    "load_audio",
    "transcribe",
]
