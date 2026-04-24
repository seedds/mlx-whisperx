import subprocess
from typing import Optional

import numpy as np


SAMPLE_RATE = 16000


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio as mono float32 NumPy waveform at 16 kHz."""
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
    if isinstance(audio, str):
        return load_audio(audio, sr=sr)
    if hasattr(audio, "tolist") and not isinstance(audio, np.ndarray):
        audio = np.array(audio.tolist())
    return np.asarray(audio, dtype=np.float32).flatten()


def slice_audio(audio: np.ndarray, start: float, end: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    start_sample = max(0, int(round(start * sr)))
    end_sample = min(len(audio), int(round(end * sr)))
    return audio[start_sample:end_sample]
