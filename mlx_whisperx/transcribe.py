from typing import Optional, Sequence

import numpy as np

from .pipeline import MLXWhisperXPipeline, PipelineOptions


def transcribe(
    audio: str | np.ndarray,
    *,
    model: str = "mlx-community/whisper-turbo",
    batch_size: int = 8,
    task: str = "transcribe",
    language: Optional[str] = None,
    temperature: float | Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    best_of: Optional[int] = 5,
    beam_size: Optional[int] = 5,
    patience: Optional[float] = 1.0,
    length_penalty: Optional[float] = 1.0,
    suppress_tokens: str = "-1",
    suppress_numerals: bool = False,
    initial_prompt: Optional[str] = None,
    hotwords: Optional[str] = None,
    condition_on_previous_text: bool = False,
    fp16: bool = True,
    compression_ratio_threshold: Optional[float] = 2.4,
    logprob_threshold: Optional[float] = -1.0,
    no_speech_threshold: Optional[float] = 0.6,
    vad_method: str = "silero",
    vad_onset: float = 0.500,
    vad_offset: float = 0.363,
    chunk_size: int = 30,
    no_vad: bool = False,
    vad_dump_path: Optional[str] = None,
    align_model: Optional[str] = None,
    no_align: bool = False,
    interpolate_method: str = "nearest",
    return_char_alignments: bool = False,
    diarize: bool = False,
    diarize_model: str = "pyannote/speaker-diarization-community-1",
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    speaker_embeddings: bool = False,
    hf_token: Optional[str] = None,
    model_dir: Optional[str] = None,
    model_cache_only: bool = False,
    device: str = "cpu",
    verbose: bool = False,
    print_progress: bool = False,
) -> dict:
    kwargs = locals().copy()
    audio_obj = kwargs.pop("audio")
    options = PipelineOptions(**kwargs)
    return MLXWhisperXPipeline(options).transcribe(audio_obj)
