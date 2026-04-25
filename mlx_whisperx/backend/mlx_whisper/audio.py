# Copyright © 2023 Apple Inc.

"""Audio and spectrogram utilities for the vendored MLX Whisper backend.

Whisper models consume 30-second log-Mel windows at 16 kHz. This module contains the
fixed audio constants, ffmpeg loading path, and a small MLX STFT implementation used
to generate those model inputs.
"""

import os
from functools import lru_cache
from subprocess import CalledProcessError, run
from typing import Optional, Union

import mlx.core as mx
import numpy as np

# Hard-coded Whisper audio hyperparameters. These values are part of the model
# contract: changing them would make positional embeddings and timestamp math wrong.
SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = SAMPLE_RATE // HOP_LENGTH  # 10ms per audio frame
TOKENS_PER_SECOND = SAMPLE_RATE // N_SAMPLES_PER_TOKEN  # 20ms per audio token


def load_audio(file: str = Optional[str], sr: int = SAMPLE_RATE, from_stdin=False):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary. Requires the ffmpeg CLI in PATH.
    if from_stdin:
        cmd = ["ffmpeg", "-i", "pipe:0"]
    else:
        cmd = ["ffmpeg", "-nostdin", "-i", file]

    # fmt: off
    cmd.extend([
        "-threads", "0",
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sr),
        "-"
    ])
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return mx.array(np.frombuffer(out, np.int16)).flatten().astype(mx.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if array.shape[axis] > length:
        # Trim from the right; Whisper windows are always anchored at the current seek
        # position, so excess samples belong to a future window.
        sl = [slice(None)] * array.ndim
        sl[axis] = slice(0, length)
        array = array[tuple(sl)]

    if array.shape[axis] < length:
        # Right-pad with silence to satisfy the fixed encoder input shape.
        pad_widths = [(0, 0)] * array.ndim
        pad_widths[axis] = (0, length - array.shape[axis])
        array = mx.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(n_mels: int) -> mx.array:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
            mel_128=librosa.filters.mel(sr=16000, n_fft=400, n_mels=128),
        )
    """
    assert n_mels in {80, 128}, f"Unsupported n_mels: {n_mels}"

    filename = os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    return mx.load(filename)[f"mel_{n_mels}"]


@lru_cache(maxsize=None)
def hanning(size):
    """Return a cached periodic Hann window as an MLX array."""
    return mx.array(np.hanning(size + 1)[:-1])


def stft(x, window, nperseg=256, noverlap=None, nfft=None, axis=-1, pad_mode="reflect"):
    """Compute a lightweight one-dimensional STFT using MLX operations."""
    if nfft is None:
        nfft = nperseg
    if noverlap is None:
        noverlap = nfft // 4

    def _pad(x, padding, pad_mode="constant"):
        """Pad the signal before framing, matching Whisper's reflect padding."""
        if pad_mode == "constant":
            return mx.pad(x, [(padding, padding)])
        elif pad_mode == "reflect":
            prefix = x[1 : padding + 1][::-1]
            suffix = x[-(padding + 1) : -1][::-1]
            return mx.concatenate([prefix, x, suffix])
        else:
            raise ValueError(f"Invalid pad_mode {pad_mode}")

    padding = nperseg // 2
    x = _pad(x, padding, pad_mode)

    # Frame the waveform with overlapping strided views, then run an RFFT per frame.
    strides = [noverlap, 1]
    t = (x.size - nperseg + noverlap) // noverlap
    shape = [t, nfft]
    x = mx.as_strided(x, shape=shape, strides=strides)
    return mx.fft.rfft(x * window)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray],
    n_mels: int = 80,
    padding: int = 0,
):
    """
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, mx.array], shape = (*)
        The path to audio or either a NumPy or mlx array containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    Returns
    -------
    mx.array, shape = (80, n_frames)
        An  array that contains the Mel spectrogram
    """
    if isinstance(audio, str):
        # File paths are decoded through ffmpeg; in-memory arrays are assumed to already
        # be mono 16 kHz waveforms.
        audio = load_audio(audio)
    elif not isinstance(audio, mx.array):
        audio = mx.array(audio)

    if padding > 0:
        # Transcription pads by one full chunk so the final real window can be sliced
        # without special-casing short trailing audio.
        audio = mx.pad(audio, (0, padding))
    window = hanning(N_FFT)
    freqs = stft(audio, window, nperseg=N_FFT, noverlap=HOP_LENGTH)
    magnitudes = freqs[:-1, :].abs().square()

    filters = mel_filters(n_mels)
    mel_spec = magnitudes @ filters.T

    log_spec = mx.maximum(mel_spec, 1e-10).log10()
    # Match Whisper preprocessing: clamp dynamic range to 8 dB below the peak and map
    # values into the approximate range expected by the model.
    log_spec = mx.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec
