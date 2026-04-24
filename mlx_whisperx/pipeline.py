import gc
import warnings
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from ._compat import import_mlx_whisper
from .alignment import align, load_align_model
from .audio import SAMPLE_RATE, audio_to_numpy, slice_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .log_utils import get_logger
from .vads import get_vad_class


logger = get_logger(__name__)


@dataclass
class PipelineOptions:
    model: str = "mlx-community/whisper-turbo"
    batch_size: int = 8
    task: str = "transcribe"
    language: Optional[str] = None
    temperature: float | Sequence[float] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    best_of: Optional[int] = 5
    beam_size: Optional[int] = 5
    patience: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    suppress_tokens: str = "-1"
    suppress_numerals: bool = False
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None
    condition_on_previous_text: bool = False
    fp16: bool = True
    compression_ratio_threshold: Optional[float] = 2.4
    logprob_threshold: Optional[float] = -1.0
    no_speech_threshold: Optional[float] = 0.6
    vad_method: str = "silero"
    vad_onset: float = 0.500
    vad_offset: float = 0.363
    chunk_size: int = 30
    no_vad: bool = False
    align_model: Optional[str] = None
    no_align: bool = False
    interpolate_method: str = "nearest"
    return_char_alignments: bool = False
    diarize: bool = False
    diarize_model: str = "pyannote/speaker-diarization-community-1"
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    speaker_embeddings: bool = False
    hf_token: Optional[str] = None
    model_dir: Optional[str] = None
    model_cache_only: bool = False
    device: str = "cpu"
    verbose: bool = False
    print_progress: bool = False


class MLXWhisperXPipeline:
    def __init__(self, options: PipelineOptions):
        self.options = options
        self._mlx_whisper = import_mlx_whisper()

    def transcribe(self, audio: str | np.ndarray) -> dict:
        audio_np = audio_to_numpy(audio)
        audio_path = audio if isinstance(audio, str) else None

        vad_chunks = self._vad_chunks(audio_np)
        if not vad_chunks:
            return {"segments": [], "word_segments": [], "language": self.options.language or "en", "text": ""}

        result = self._asr(audio_np, vad_chunks)

        if self.options.task == "translate":
            no_align = True
        else:
            no_align = self.options.no_align

        if not no_align and result["segments"]:
            result = self._align(result, audio_np)

        if self.options.diarize:
            result = self._diarize(result, audio_path or audio_np)

        return self._normalize_result(result)

    def _normalize_result(self, result: dict) -> dict:
        """Return WhisperX-compatible JSON shape and strip internal metadata."""
        include_speakers = self.options.diarize
        include_chars = self.options.return_char_alignments
        normalized_segments: list[dict] = []

        for segment in result.get("segments", []):
            normalized_words: list[dict] = []
            for word in segment.get("words", []):
                normalized_word = {"word": word.get("word", "")}
                if "start" in word:
                    normalized_word["start"] = word["start"]
                if "end" in word:
                    normalized_word["end"] = word["end"]
                if "score" in word:
                    normalized_word["score"] = word["score"]
                if include_speakers and "speaker" in word:
                    normalized_word["speaker"] = word["speaker"]
                normalized_words.append(normalized_word)

            normalized_segment = {
                "start": segment.get("start", 0.0),
                "end": segment.get("end", 0.0),
                "text": segment.get("text", ""),
                "words": normalized_words,
            }
            if include_speakers and "speaker" in segment:
                normalized_segment["speaker"] = segment["speaker"]
            if include_chars and "chars" in segment:
                normalized_segment["chars"] = segment["chars"]
            normalized_segments.append(normalized_segment)

        normalized = {
            "segments": normalized_segments,
            "language": result.get("language") or self.options.language or "en",
            "word_segments": [
                word for segment in normalized_segments for word in segment.get("words", [])
            ],
        }
        normalized = {
            "segments": normalized["segments"],
            "word_segments": normalized["word_segments"],
            "language": normalized["language"],
        }
        if include_speakers and "speaker_embeddings" in result:
            normalized["speaker_embeddings"] = result["speaker_embeddings"]
        return normalized

    def _vad_chunks(self, audio: np.ndarray) -> list[dict]:
        duration = len(audio) / SAMPLE_RATE
        if self.options.no_vad:
            return [{"start": 0.0, "end": duration, "segments": [(0.0, duration)]}]

        VadClass = get_vad_class(self.options.vad_method)
        if self.options.vad_method == "pyannote":
            vad_model = VadClass(
                self.options.device,
                token=self.options.hf_token,
                vad_onset=self.options.vad_onset,
                vad_offset=self.options.vad_offset,
                chunk_size=self.options.chunk_size,
            )
        else:
            vad_model = VadClass(
                vad_onset=self.options.vad_onset,
                vad_offset=self.options.vad_offset,
                chunk_size=self.options.chunk_size,
            )

        if hasattr(vad_model, "preprocess_audio"):
            waveform = vad_model.preprocess_audio(audio)
            merge_chunks = vad_model.merge_chunks
        else:
            waveform = audio
            merge_chunks = None

        vad_segments = vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})
        if merge_chunks is None:
            raise TypeError("VAD model must provide merge_chunks")
        chunks = merge_chunks(
            vad_segments,
            self.options.chunk_size,
            onset=self.options.vad_onset,
            offset=self.options.vad_offset,
        )
        return [chunk for chunk in chunks if chunk["end"] > chunk["start"]]

    def _asr(self, audio: np.ndarray, vad_chunks: list[dict]) -> dict:
        if self.options.batch_size not in (0, 1, None):
            logger.info("batch_size is accepted for API parity; current ASR wrapper decodes VAD chunks serially")
        if self.options.beam_size is not None:
            logger.info("beam_size is accepted for API parity but ignored because mlx-whisper beam search is not implemented")
        if self.options.suppress_numerals:
            warnings.warn("--suppress_numerals is accepted but not implemented for mlx-whisper decoding yet")

        language = self.options.language
        all_segments: list[dict] = []
        detected_language = language

        prompt = self.options.initial_prompt
        if self.options.hotwords:
            prompt = f"{prompt or ''} {self.options.hotwords}".strip()

        total = len(vad_chunks)
        for idx, chunk in enumerate(vad_chunks):
            start = float(chunk["start"])
            end = float(chunk["end"])
            chunk_audio = slice_audio(audio, start, end)
            if chunk_audio.size == 0:
                continue

            decode_kwargs = {
                "language": language,
                "task": self.options.task,
                "best_of": self.options.best_of,
                "length_penalty": self.options.length_penalty,
                "suppress_tokens": self.options.suppress_tokens,
                "fp16": self.options.fp16,
                "without_timestamps": True,
            }
            decode_kwargs = {key: value for key, value in decode_kwargs.items() if value is not None}

            chunk_result = self._mlx_whisper.transcribe(
                chunk_audio,
                path_or_hf_repo=self.options.model,
                verbose=False,
                temperature=self.options.temperature,
                compression_ratio_threshold=self.options.compression_ratio_threshold,
                logprob_threshold=self.options.logprob_threshold,
                no_speech_threshold=self.options.no_speech_threshold,
                condition_on_previous_text=self.options.condition_on_previous_text,
                initial_prompt=prompt,
                word_timestamps=False,
                **decode_kwargs,
            )

            detected_language = detected_language or chunk_result.get("language")
            language = language or detected_language

            chunk_segments = chunk_result.get("segments", [])
            if not chunk_segments and chunk_result.get("text", "").strip():
                chunk_segments = [{"start": 0.0, "end": end - start, "text": chunk_result["text"]}]

            for segment in chunk_segments:
                text = segment.get("text", "")
                if not text.strip():
                    continue
                all_segments.append(
                    {
                        "start": round(start + float(segment.get("start", 0.0)), 3),
                        "end": round(start + float(segment.get("end", end - start)), 3),
                        "text": text,
                        **({"avg_logprob": segment["avg_logprob"]} if "avg_logprob" in segment else {}),
                    }
                )

            if self.options.print_progress:
                print(f"Progress: {((idx + 1) / total) * 50:.2f}%...")

        return {"segments": all_segments, "language": detected_language or language or "en"}

    def _align(self, result: dict, audio: np.ndarray) -> dict:
        language = result.get("language") or self.options.language or "en"
        logger.info("Performing alignment...")
        align_model, metadata = load_align_model(
            language,
            self.options.device,
            model_name=self.options.align_model,
            model_dir=self.options.model_dir,
            model_cache_only=self.options.model_cache_only,
        )
        try:
            aligned = align(
                result["segments"],
                align_model,
                metadata,
                audio,
                self.options.device,
                interpolate_method=self.options.interpolate_method,
                return_char_alignments=self.options.return_char_alignments,
                print_progress=self.options.print_progress,
            )
        finally:
            del align_model
            gc.collect()

        aligned["language"] = language
        return aligned

    def _diarize(self, result: dict, audio: str | np.ndarray) -> dict:
        logger.info("Performing diarization...")
        if self.options.hf_token is None:
            logger.warning("No hf_token provided; pyannote diarization may fail for gated models")
        diarize_model = DiarizationPipeline(
            model_name=self.options.diarize_model,
            token=self.options.hf_token,
            device=self.options.device,
            cache_dir=self.options.model_dir,
        )
        diarize_result = diarize_model(
            audio,
            min_speakers=self.options.min_speakers,
            max_speakers=self.options.max_speakers,
            return_embeddings=self.options.speaker_embeddings,
        )
        if self.options.speaker_embeddings:
            diarize_segments, speaker_embeddings = diarize_result
        else:
            diarize_segments = diarize_result
            speaker_embeddings = None
        return assign_word_speakers(diarize_segments, result, speaker_embeddings)
