"""Pyannote voice activity detection backend implementation."""

from .vad import Segment, Vad


DEFAULT_PYANNOTE_VAD_MODEL = "pyannote/segmentation-3.0"


class Pyannote(Vad):
    """Pyannote segmentation pipeline adapted to the shared VAD interface."""

    def __init__(self, device, token=None, model_name=None, cache_dir=None, **kwargs):
        """Load and configure pyannote voice activity detection."""
        super().__init__(kwargs["vad_onset"])
        try:
            import torch
            from pyannote.audio import Model
            from pyannote.audio.pipelines import VoiceActivityDetection
            from pyannote.core import Annotation, Segment as PyannoteSegment
        except Exception as exc:
            raise RuntimeError(
                "Pyannote VAD could not be imported in this environment. "
                "Use --vad_method silero, or fix the pyannote/torch/torchaudio install."
            ) from exc

        self._torch = torch
        self._Annotation = Annotation
        self._PyannoteSegment = PyannoteSegment
        self.vad_onset = kwargs["vad_onset"]
        self.vad_offset = kwargs.get("vad_offset", self.vad_onset)
        self.chunk_size = kwargs["chunk_size"]

        model_name = model_name or DEFAULT_PYANNOTE_VAD_MODEL
        try:
            model = Model.from_pretrained(model_name, token=token, cache_dir=cache_dir)
            if model is None:
                raise RuntimeError("model loader returned None")
        except Exception as exc:
            raise RuntimeError(
                f"Could not load pyannote VAD model {model_name!r}. If the model is gated, "
                "accept its Hugging Face terms and pass --hf_token. Otherwise use the default "
                "Silero VAD with --vad_method silero."
            ) from exc
        pipeline = VoiceActivityDetection(segmentation=model, device=torch.device(device))
        pipeline.instantiate(
            {
                "onset": self.vad_onset,
                "offset": self.vad_offset,
                "min_duration_on": 0.1,
                "min_duration_off": 0.1,
            }
        )
        self.vad_pipeline = pipeline

    def __call__(self, audio: dict, **kwargs):
        """Run pyannote VAD and return pyannote's timeline/score object."""
        return self.vad_pipeline(audio)

    @staticmethod
    def preprocess_audio(audio):
        """Convert the project-standard NumPy waveform into pyannote's tensor shape."""
        import torch

        return torch.from_numpy(audio).unsqueeze(0)

    def merge_chunks(self, segments, chunk_size: int, onset: float = 0.5, offset=None):
        """Binarize pyannote scores if necessary, then use shared chunk merging."""
        segments_list = []
        # pyannote returns scores; binarize using pyannote's native pipeline output API.
        if hasattr(segments, "get_timeline"):
            timeline = segments.get_timeline()
        else:
            binarize = _Binarize(
                self._Annotation,
                self._PyannoteSegment,
                max_duration=chunk_size,
                onset=onset,
                offset=offset,
            )
            timeline = binarize(segments).get_timeline()
        for speech_turn in timeline:
            segments_list.append(Segment(speech_turn.start, speech_turn.end, "UNKNOWN"))
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset)


class _Binarize:
    """Minimal pyannote score binarizer used when raw segmentation scores are returned."""

    def __init__(self, annotation_cls, segment_cls, onset=0.5, offset=None, max_duration=float("inf")):
        self.annotation_cls = annotation_cls
        self.segment_cls = segment_cls
        self.onset = onset
        self.offset = offset or onset
        self.max_duration = max_duration

    def __call__(self, scores):
        """Convert frame-level speech probabilities into active speech intervals."""
        import numpy as np

        active = self.annotation_cls()
        frames = scores.sliding_window
        timestamps = [frames[i].middle for i in range(scores.data.shape[0])]
        for label_idx, label_scores in enumerate(scores.data.T):
            start = timestamps[0]
            is_active = label_scores[0] > self.onset
            curr_scores = [label_scores[0]]
            curr_timestamps = [start]
            t = start
            for t, score in zip(timestamps[1:], label_scores[1:]):
                if is_active:
                    if t - start > self.max_duration:
                        # Split overlong regions at the lowest-confidence point after
                        # the midpoint to keep ASR chunks bounded without arbitrary cuts.
                        search_after = len(curr_scores) // 2
                        split_idx = search_after + np.argmin(curr_scores[search_after:])
                        split_t = curr_timestamps[split_idx]
                        active[self.segment_cls(start, split_t), label_idx] = label_idx
                        start = split_t
                        curr_scores = curr_scores[split_idx + 1 :]
                        curr_timestamps = curr_timestamps[split_idx + 1 :]
                    elif score < self.offset:
                        active[self.segment_cls(start, t), label_idx] = label_idx
                        start = t
                        is_active = False
                        curr_scores = []
                        curr_timestamps = []
                    curr_scores.append(score)
                    curr_timestamps.append(t)
                elif score > self.onset:
                    start = t
                    is_active = True
            if is_active:
                active[self.segment_cls(start, t), label_idx] = label_idx
        return active
