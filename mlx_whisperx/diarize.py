from typing import Optional, Union

import numpy as np
import pandas as pd

from .audio import SAMPLE_RATE, audio_to_numpy


class DiarizationPipeline:
    def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
        try:
            import torch
            from pyannote.audio import Pipeline
        except Exception as exc:
            raise RuntimeError(
                "Pyannote diarization could not be imported in this environment. "
                "Fix the pyannote/torch/torchaudio install before using --diarize."
            ) from exc
        if isinstance(device, str):
            device = torch.device(device)
        self._torch = torch
        model_config = model_name or "pyannote/speaker-diarization-community-1"
        self.model = Pipeline.from_pretrained(model_config, token=token, cache_dir=cache_dir).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        return_embeddings: bool = False,
        progress_callback=None,
    ):
        audio_np = audio_to_numpy(audio)
        audio_data = {
            "waveform": self._torch.from_numpy(audio_np[None, :]),
            "sample_rate": SAMPLE_RATE,
        }
        output = self.model(
            audio_data,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        diarization = output.speaker_diarization
        diarize_df = pd.DataFrame(
            diarization.itertracks(yield_label=True),
            columns=["segment", "label", "speaker"],
        )
        diarize_df["start"] = diarize_df["segment"].apply(lambda segment: segment.start)
        diarize_df["end"] = diarize_df["segment"].apply(lambda segment: segment.end)

        if return_embeddings:
            embeddings = getattr(output, "speaker_embeddings", None)
            if embeddings is None:
                return diarize_df, None
            speaker_embeddings = {
                speaker: embeddings[idx].tolist()
                for idx, speaker in enumerate(diarization.labels())
            }
            return diarize_df, speaker_embeddings
        return diarize_df


class IntervalTree:
    def __init__(self, intervals: list[tuple[float, float, str]]):
        if not intervals:
            self.starts = np.array([])
            self.ends = np.array([])
            self.speakers: list[str] = []
            return
        sorted_intervals = sorted(intervals, key=lambda item: item[0])
        self.starts = np.array([item[0] for item in sorted_intervals], dtype=np.float64)
        self.ends = np.array([item[1] for item in sorted_intervals], dtype=np.float64)
        self.speakers = [item[2] for item in sorted_intervals]

    def query(self, start: float, end: float) -> list[tuple[str, float]]:
        if len(self.starts) == 0:
            return []
        right_idx = np.searchsorted(self.starts, end, side="left")
        if right_idx == 0:
            return []
        candidates = slice(0, right_idx)
        overlaps = (self.starts[candidates] < end) & (self.ends[candidates] > start)
        results = []
        for idx in np.where(overlaps)[0]:
            intersection = min(self.ends[idx], end) - max(self.starts[idx], start)
            if intersection > 0:
                results.append((self.speakers[idx], intersection))
        return results

    def find_nearest(self, time: float) -> Optional[str]:
        if len(self.starts) == 0:
            return None
        mids = (self.starts + self.ends) / 2
        return self.speakers[int(np.argmin(np.abs(mids - time)))]


def _dominant_speaker(overlaps: list[tuple[str, float]]) -> Optional[str]:
    if not overlaps:
        return None
    intersections: dict[str, float] = {}
    for speaker, duration in overlaps:
        intersections[speaker] = intersections.get(speaker, 0.0) + duration
    return max(intersections.items(), key=lambda item: item[1])[0]


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: dict,
    speaker_embeddings: Optional[dict[str, list[float]]] = None,
    fill_nearest: bool = False,
) -> dict:
    segments = transcript_result.get("segments", [])
    if not segments or diarize_df is None or len(diarize_df) == 0:
        return transcript_result

    tree = IntervalTree(
        [(row["start"], row["end"], row["speaker"]) for _, row in diarize_df.iterrows()]
    )
    for segment in segments:
        seg_start = segment.get("start", 0.0)
        seg_end = segment.get("end", seg_start)
        speaker = _dominant_speaker(tree.query(seg_start, seg_end))
        if speaker is None and fill_nearest:
            speaker = tree.find_nearest((seg_start + seg_end) / 2)
        if speaker is not None:
            segment["speaker"] = speaker

        for word in segment.get("words", []):
            if "start" not in word:
                continue
            word_start = word["start"]
            word_end = word.get("end", word_start)
            word_speaker = _dominant_speaker(tree.query(word_start, word_end))
            if word_speaker is None and fill_nearest:
                word_speaker = tree.find_nearest((word_start + word_end) / 2)
            if word_speaker is not None:
                word["speaker"] = word_speaker

    if speaker_embeddings is not None:
        transcript_result["speaker_embeddings"] = speaker_embeddings
    transcript_result["word_segments"] = [
        word for segment in segments for word in segment.get("words", [])
    ]
    return transcript_result
