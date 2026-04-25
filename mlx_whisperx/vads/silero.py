"""Silero voice activity detection backend."""

import os
from pathlib import Path
from typing import Optional

import torch

from .vad import Segment, Vad


SILERO_VAD_ENV_PATH = "MLX_WHISPERX_SILERO_VAD_PATH"
SILERO_VAD_REMOTE_REPO = "snakers4/silero-vad:master"


def _silero_cache_candidates() -> list[Path]:
    """Return local paths that may contain a Silero Torch Hub checkout."""
    candidates: list[Path] = []
    env_path = os.environ.get(SILERO_VAD_ENV_PATH)
    if env_path:
        candidates.append(Path(env_path).expanduser())

    hub_dir = Path(torch.hub.get_dir()).expanduser()
    candidates.extend(
        [
            hub_dir / "snakers4_silero-vad_master",
            hub_dir / "snakers4_silero-vad_main",
        ]
    )
    return candidates


def _load_silero_from_cache():
    """Load Silero from a local checkout when possible.

    Prefer local cache to avoid network access during normal CLI runs. If an explicit
    environment path is configured, failures are surfaced because the user asked for
    that exact checkout.
    """
    last_error = None
    for path in _silero_cache_candidates():
        if not (path / "hubconf.py").exists():
            continue
        try:
            return torch.hub.load(
                repo_or_dir=str(path),
                model="silero_vad",
                source="local",
                force_reload=False,
                onnx=False,
                trust_repo=True,
                verbose=False,
            )
        except Exception as exc:
            last_error = exc

    if os.environ.get(SILERO_VAD_ENV_PATH) and last_error is not None:
        raise RuntimeError(
            f"Failed to load Silero VAD from ${SILERO_VAD_ENV_PATH}."
        ) from last_error
    return None


class Silero(Vad):
    """Silero-backed VAD adapter with the same interface as pyannote VAD."""

    def __init__(self, **kwargs):
        """Load Silero from local cache or Torch Hub."""
        super().__init__(kwargs["vad_onset"])
        self.vad_onset = kwargs["vad_onset"]
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
