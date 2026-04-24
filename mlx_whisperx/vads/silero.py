from typing import Optional

import torch

from .vad import Segment, Vad


class Silero(Vad):
    def __init__(self, **kwargs):
        super().__init__(kwargs["vad_onset"])
        self.vad_onset = kwargs["vad_onset"]
        self.chunk_size = kwargs["chunk_size"]
        self.vad_pipeline, vad_utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
            trust_repo=True,
            verbose=False,
        )
        self.get_speech_timestamps = vad_utils[0]

    def __call__(self, audio: dict, **kwargs):
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
        return audio

    @staticmethod
    def merge_chunks(segments, chunk_size: int, onset: float = 0.5, offset: Optional[float] = None):
        return Vad.merge_chunks(segments, chunk_size, onset, offset)
