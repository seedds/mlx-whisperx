"""Audio loading utilities shared by VAD, ASR, alignment, and diarization.

All stages expect a mono 16 kHz float32 waveform. The helpers in this module keep
that representation consistent whether callers pass a file path, a NumPy array, or
another array-like object.
"""

import subprocess
from typing import Optional

import numpy as np


SAMPLE_RATE = 16000


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 NumPy waveform at the requested sample rate.

    ffmpeg handles demuxing, decoding, channel down-mixing, and resampling. The raw
    signed 16-bit PCM stream is then normalized to Whisper-style float samples in
    the range roughly [-1.0, 1.0].
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"Failed to load audio: {exc.stderr.decode()}") from exc
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def audio_to_numpy(audio: str | np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Normalize supported audio inputs into a flat float32 NumPy array."""
    if isinstance(audio, str):
        return load_audio(audio, sr=sr)
    if hasattr(audio, "tolist") and not isinstance(audio, np.ndarray):
        # Accept MLX/Torch-like arrays without importing those libraries here.
        audio = np.array(audio.tolist())
    return np.asarray(audio, dtype=np.float32).flatten()


def slice_audio(audio: np.ndarray, start: float, end: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Return a time slice of an already-normalized waveform."""
    start_sample = max(0, int(round(start * sr)))
    end_sample = min(len(audio), int(round(end * sr)))
    return audio[start_sample:end_sample]
