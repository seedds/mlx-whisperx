"""High-level orchestration for the WhisperX-style transcription pipeline.

This module glues together four independent stages:

1. Voice activity detection splits long audio into speech-heavy chunks.
2. The vendored MLX Whisper backend transcribes each chunk.
3. Forced alignment optionally refines segment timestamps to word timestamps.
4. Pyannote diarization optionally assigns speaker labels.

The code keeps the ASR backend isolated from WhisperX-style output shaping so the
public JSON schema remains stable even if backend internals change.
"""

import gc
import importlib
import json
import pathlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Sequence

import numpy as np

from ._compat import import_mlx_whisper
from .alignment import align, load_align_model
from .audio import SAMPLE_RATE, audio_to_numpy, slice_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .log_utils import get_logger
from .vads import get_vad_class


logger = get_logger(__name__)
NUMERAL_SYMBOLS = "0123456789%$£"


@lru_cache(maxsize=32)
def _find_numeral_symbol_tokens(language: Optional[str], task: str) -> tuple[int, ...]:
    """Return tokenizer IDs whose decoded text contains numeric/currency symbols.

    This supports the `suppress_numerals` option without hard-coding vocabulary IDs,
    which differ between tokenizer variants and language/task settings.
    """
    tokenizer_module = importlib.import_module("mlx_whisperx.backend.mlx_whisper.tokenizer")
    tokenizer = tokenizer_module.get_tokenizer(True, language=language or "en", task=task)
    numeral_symbol_tokens: list[int] = []
    for token_id in range(tokenizer.eot):
        token = tokenizer.decode([token_id]).removeprefix(" ")
        if any(char in NUMERAL_SYMBOLS for char in token):
            numeral_symbol_tokens.append(token_id)
    return tuple(numeral_symbol_tokens)


def _merge_suppress_tokens(suppress_tokens: str | Sequence[int] | None, extra_tokens: Sequence[int]) -> list[int]:
    """Parse user-supplied suppression tokens and add extra backend token IDs."""
    if suppress_tokens is None:
        parsed_tokens: list[int] = []
    elif isinstance(suppress_tokens, str):
        parsed_tokens = [int(token) for token in suppress_tokens.split(",") if token]
    else:
        parsed_tokens = [int(token) for token in suppress_tokens]
    return sorted(set(parsed_tokens).union(extra_tokens))


@dataclass
class PipelineOptions:
    """Configuration for every stage of `MLXWhisperXPipeline`.

    The dataclass mostly mirrors CLI flags. Defaults are chosen to produce useful
    WhisperX-like output on Apple Silicon while keeping diarization disabled unless
    explicitly requested because it may require gated pyannote models.
    """

    model: str = "mlx-community/whisper-turbo"
    task: str = "transcribe"
    language: Optional[str] = None
    temperature: float | Sequence[float] = 0.0
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
    vad_model: Optional[str] = None
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
    vad_dump_path: Optional[str] = None


class MLXWhisperXPipeline:
    """Run VAD, MLX Whisper ASR, alignment, and optional diarization."""

    def __init__(self, options: PipelineOptions):
        """Store pipeline options and lazily import the vendored ASR backend."""
        self.options = options
        self._mlx_whisper = import_mlx_whisper()

    def transcribe(self, audio: str | np.ndarray) -> dict:
        """Execute the full configured pipeline for a path or waveform."""
        audio_np = audio_to_numpy(audio)
        audio_path = audio if isinstance(audio, str) else None

        vad_chunks = self._vad_chunks(audio_np)
        if not vad_chunks:
            # Preserve the public result shape even when VAD finds no speech.
            return {"segments": [], "word_segments": [], "language": self.options.language or "en", "text": ""}

        result = self._asr(audio_np, vad_chunks)

        if self.options.task == "translate":
            # Forced alignment requires transcript text in the source language. Whisper
            # translations are English text over non-English audio, so the CTC aligner
            # cannot reliably match them back to the original waveform.
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
                # Copy only public fields. Backend artifacts such as probabilities,
                # token IDs, and temporary alignment metadata are intentionally omitted.
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
        # Reconstruct the dict in canonical output order for stable JSON snapshots.
        normalized = {
            "segments": normalized["segments"],
            "word_segments": normalized["word_segments"],
            "language": normalized["language"],
        }
        if include_speakers and "speaker_embeddings" in result:
            normalized["speaker_embeddings"] = result["speaker_embeddings"]
        return normalized

    def _vad_chunks(self, audio: np.ndarray) -> list[dict]:
        """Run the selected VAD backend and return merged speech chunks."""
        duration = len(audio) / SAMPLE_RATE
        if self.options.no_vad:
            # Keep the downstream ASR path identical by representing full-file ASR as
            # a single VAD chunk instead of adding separate no-VAD transcription code.
            chunks = [{"start": 0.0, "end": duration, "segments": [(0.0, duration)]}]
            self._dump_vad_chunks(chunks, duration)
            return chunks

        VadClass = get_vad_class(self.options.vad_method)
        if self.options.vad_method == "pyannote":
            # Pyannote has a different constructor contract because it needs a Torch
            # device, optional auth token, and model cache settings.
            vad_model = VadClass(
                self.options.device,
                token=self.options.hf_token,
                model_name=self.options.vad_model,
                cache_dir=self.options.model_dir,
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
            # Silero accepts NumPy directly; pyannote expects a Torch waveform. The
            # backend owns conversion so this pipeline can stay backend-agnostic.
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
        chunks = [chunk for chunk in chunks if chunk["end"] > chunk["start"]]
        self._dump_vad_chunks(chunks, duration)
        return chunks

    def _dump_vad_chunks(self, chunks: list[dict], audio_duration: float) -> None:
        """Write VAD diagnostics to JSON when requested by the caller."""
        if not self.options.vad_dump_path:
            return

        payload = {
            "vad_method": self.options.vad_method,
            "vad_onset": self.options.vad_onset,
            "vad_offset": self.options.vad_offset,
            "vad_model": self.options.vad_model,
            "chunk_size": self.options.chunk_size,
            "no_vad": self.options.no_vad,
            "audio_duration": round(audio_duration, 3),
            "chunk_count": len(chunks),
            "chunks": [
                {
                    "index": index,
                    "start": round(float(chunk["start"]), 3),
                    "end": round(float(chunk["end"]), 3),
                    "duration": round(float(chunk["end"]) - float(chunk["start"]), 3),
                    "segments": [
                        {"start": round(float(start), 3), "end": round(float(end), 3)}
                        for start, end in chunk.get("segments", [])
                    ],
                }
                for index, chunk in enumerate(chunks)
            ],
        }
        output_path = pathlib.Path(self.options.vad_dump_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def _asr(self, audio: np.ndarray, vad_chunks: list[dict]) -> dict:
        """Transcribe each VAD chunk and restore chunk-local timestamps to file time."""
        suppress_tokens: str | list[int] = self.options.suppress_tokens
        if self.options.suppress_numerals:
            logger.info("Suppressing numeral and symbol tokens")
            suppress_tokens = _merge_suppress_tokens(
                self.options.suppress_tokens,
                _find_numeral_symbol_tokens(self.options.language, self.options.task),
            )

        language = self.options.language
        all_segments: list[dict] = []
        detected_language = language

        prompt = self.options.initial_prompt
        if self.options.hotwords:
            # The vendored backend accepts a single initial prompt string, so hotwords
            # are appended rather than passed through a separate API.
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
                "beam_size": self.options.beam_size,
                "patience": self.options.patience,
                "length_penalty": self.options.length_penalty,
                "suppress_tokens": suppress_tokens,
                "fp16": self.options.fp16,
                "without_timestamps": True,
            }
            # None-valued options should fall back to backend defaults rather than
            # overriding them with null values.
            decode_kwargs = {key: value for key, value in decode_kwargs.items() if value is not None}

            chunk_result = self._mlx_whisper.transcribe(
                chunk_audio,
                path_or_hf_repo=self.options.model,
                verbose=None,
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
                # Some backend paths can return only aggregate text. Synthesize a
                # chunk-local segment so the downstream output schema remains uniform.
                chunk_segments = [{"start": 0.0, "end": end - start, "text": chunk_result["text"]}]

            for segment in chunk_segments:
                text = segment.get("text", "")
                if not text.strip():
                    continue
                asr_segment = {
                    # Backend segment times are relative to the sliced chunk. Add the
                    # VAD chunk start so alignment/diarization see absolute file time.
                    "start": round(start + float(segment.get("start", 0.0)), 3),
                    "end": round(start + float(segment.get("end", end - start)), 3),
                    "text": text,
                    **({"avg_logprob": segment["avg_logprob"]} if "avg_logprob" in segment else {}),
                }
                all_segments.append(asr_segment)
                if self.options.verbose:
                    print(
                        "Transcript: "
                        f"[{asr_segment['start']} --> {asr_segment['end']}] "
                        f"{text.strip()}"
                    )

            if self.options.print_progress:
                print(f"Progress: {((idx + 1) / total) * 50:.2f}%...")

        return {"segments": all_segments, "language": detected_language or language or "en"}

    def _align(self, result: dict, audio: np.ndarray) -> dict:
        """Load a CTC alignment model and refine ASR segments to word timings."""
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
            # Alignment models can be large; release promptly before optional diarization.
            del align_model
            gc.collect()

        aligned["language"] = language
        return aligned

    def _diarize(self, result: dict, audio: str | np.ndarray) -> dict:
        """Run pyannote diarization and attach dominant speaker labels to words."""
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
