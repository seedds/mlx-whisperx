"""Silero voice activity detection backend."""

import inspect
import os
from pathlib import Path
from typing import Optional

import torch

from ..log_utils import get_logger
from .vad import Segment, Vad


logger = get_logger(__name__)
SILERO_VAD_ENV_PATH = "MLX_WHISPERX_SILERO_VAD_PATH"
SILERO_VAD_REMOTE_REPO = "snakers4/silero-vad:master"


def _silero_cache_candidates() -> list[Path]:
    """Return local paths that may contain a Silero Torch Hub checkout."""
    hub_dir = Path(torch.hub.get_dir()).expanduser()
    return [
        hub_dir / "snakers4_silero-vad_master",
        hub_dir / "snakers4_silero-vad_main",
    ]


def _load_silero_checkout(path: Path):
    """Load Silero from one local Torch Hub checkout."""
    return torch.hub.load(
        repo_or_dir=str(path),
        model="silero_vad",
        source="local",
        force_reload=False,
        onnx=False,
        trust_repo=True,
        verbose=False,
    )


def _load_silero_from_cache():
    """Load Silero from a local checkout when possible.

    Prefer local cache to avoid network access during normal CLI runs. An explicit
    environment path is authoritative: it is the only candidate tried, and its failure
    is raised rather than hidden behind another checkout or a silent download.
    """
    env_path = os.environ.get(SILERO_VAD_ENV_PATH)
    if env_path:
        path = Path(env_path).expanduser()
        if not (path / "hubconf.py").exists():
            raise RuntimeError(
                f"${SILERO_VAD_ENV_PATH} is set to {path}, which has no hubconf.py."
            )
        try:
            return _load_silero_checkout(path)
        except Exception as exc:
            raise RuntimeError(f"Failed to load Silero VAD from ${SILERO_VAD_ENV_PATH}.") from exc

    for path in _silero_cache_candidates():
        if not (path / "hubconf.py").exists():
            continue
        try:
            return _load_silero_checkout(path)
        except Exception:
            continue
    return None


class Silero(Vad):
    """Silero-backed VAD adapter with the same interface as pyannote VAD."""

    def __init__(self, **kwargs):
        """Load Silero from local cache or Torch Hub."""
        super().__init__(kwargs["vad_onset"])
        self.vad_onset = kwargs["vad_onset"]
        self.vad_offset = kwargs.get("vad_offset")
        self.chunk_size = kwargs["chunk_size"]
        silero = _load_silero_from_cache() or torch.hub.load(
            repo_or_dir=SILERO_VAD_REMOTE_REPO,
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
            skip_validation=True,
            verbose=False,
        )
        self.vad_pipeline, vad_utils = silero
        self.get_speech_timestamps = vad_utils[0]

    def _offset_kwargs(self) -> dict:
        """Return Silero's offset threshold argument when this version supports it.

        Silero exposes the offset as `neg_threshold`, which older releases lack. The
        option is skipped rather than passed blindly so an older checkout keeps working.
        """
        if self.vad_offset is None:
            return {}
        try:
            parameters = inspect.signature(self.get_speech_timestamps).parameters
        except (TypeError, ValueError):
            return {}
        if "neg_threshold" not in parameters:
            logger.warning(
                "Installed Silero VAD does not support an offset threshold; "
                "ignoring vad_offset=%s.",
                self.vad_offset,
            )
            return {}
        return {"neg_threshold": self.vad_offset}

    def __call__(self, audio: dict, **kwargs):
        """Return speech intervals in seconds for a 16 kHz waveform."""
        sample_rate = audio["sample_rate"]
        if sample_rate != 16000:
            raise ValueError("Only 16000 Hz sample rate is supported")
        timestamps = self.get_speech_timestamps(
            audio["waveform"],
            model=self.vad_pipeline,
            sampling_rate=sample_rate,
            max_speech_duration_s=self.chunk_size,
            threshold=self.vad_onset,
            **self._offset_kwargs(),
        )
        return [
            Segment(item["start"] / sample_rate, item["end"] / sample_rate, "UNKNOWN")
            for item in timestamps
        ]

    @staticmethod
    def preprocess_audio(audio):
        """Silero accepts the project-standard NumPy waveform directly."""
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size: int, onset: float = 0.5, offset: Optional[float] = None):
        """Delegate chunk merging to the shared VAD base implementation."""
        return Vad.merge_chunks(segments, chunk_size, onset, offset)
