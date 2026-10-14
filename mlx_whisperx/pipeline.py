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

from ._language import is_english_only_model, normalize_language_settings
from ._compat import import_mlx_whisper
from .alignment import AlignmentDependencyError, align, load_align_model
from .audio import SAMPLE_RATE, audio_to_numpy, slice_audio
from .diarize import DiarizationPipeline, assign_word_speakers
from .log_utils import get_logger
from .vads import AUTO_VAD_METHOD, get_vad_class
from .vads.coverage import (
    audible_regions,
    covered_seconds,
    energy_pauses,
    is_repetition_loop,
    speech_level,
)
from .vads.vad import Vad


logger = get_logger(__name__)
NUMERAL_SYMBOLS = "0123456789%$£"
# One Whisper encoder window; VAD chunks are decoded whole, so they cannot exceed it.
MAX_CHUNK_SIZE = 30

# Below this fraction of the file reaching the decoder, something is wrong with
# speech detection rather than with the content.
LOW_COVERAGE_WARNING = 0.5

# Whisper keeps at most n_text_ctx // 2 - 1 prompt tokens. The backend makes the exact
# cut once the initial prompt is known; this only bounds the history each stream stores.
CONTEXT_TOKEN_LIMIT = 223
# Whisper stops carrying text forward after a window needed a hot retry to decode.
CONTEXT_RESET_TEMPERATURE = 0.5


@lru_cache(maxsize=32)
def _find_numeral_symbol_tokens(
    language: Optional[str],
    task: str,
    multilingual: bool = True,
) -> tuple[int, ...]:
    """Return tokenizer IDs whose decoded text contains numeric/currency symbols.

    This supports the `suppress_numerals` option without hard-coding vocabulary IDs,
    which differ between tokenizer variants and language/task settings. English-only
    checkpoints use a different vocabulary, so suppression must be computed against the
    same tokenizer the model will decode with.
    """
    tokenizer_module = importlib.import_module("mlx_whisperx.backend.mlx_whisper.tokenizer")
    # `get_tokenizer` drops language/task itself for English-only vocabularies.
    tokenizer = tokenizer_module.get_tokenizer(
        multilingual,
        language=language or "en",
        task=task,
    )
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


def _split_streams(n_chunks: int, n_streams: int) -> list[list[int]]:
    """Split chunk indices into contiguous streams; earlier streams take any remainder."""
    size, extra = divmod(n_chunks, n_streams)
    streams: list[list[int]] = []
    start = 0
    for stream in range(n_streams):
        length = size + (1 if stream < extra else 0)
        streams.append(list(range(start, start + length)))
        start += length
    return streams


def _interleave_streams(streams: list[list[int]]) -> list[list[tuple[int, int]]]:
    """Return batches holding the k-th chunk of every stream as `(stream, chunk)` pairs."""
    depth = max((len(stream) for stream in streams), default=0)
    return [
        [(index, stream[step]) for index, stream in enumerate(streams) if step < len(stream)]
        for step in range(depth)
    ]


def _extend_context(context: Sequence[int], tokens: Sequence[int], limit: int) -> list[int]:
    """Append decoded tokens to a stream's history, keeping only the newest `limit`."""
    if limit <= 0:
        # `[-0:]` would keep everything.
        return []
    return [*context, *tokens][-limit:]


def _rebase_progress(callback, *, batch_index: int, total_batches: int, windows_before: int, total_windows: int):
    """Report a single-batch backend call as one batch of the whole decode pass."""
    if callback is None:
        return None

    def _callback(info: dict) -> None:
        callback(
            {
                **info,
                "batch_index": batch_index,
                "total_batches": total_batches,
                "windows_done": windows_before + info["windows_done"],
                "total_windows": total_windows,
            }
        )

    return _callback


@dataclass
class PipelineOptions:
    """Configuration for every stage of `MLXWhisperXPipeline`.

    The dataclass mostly mirrors CLI flags. Defaults are chosen to produce useful
    WhisperX-like output on Apple Silicon while using Silero VAD by default and
    keeping diarization disabled unless explicitly requested because it may
    require gated pyannote models.
    """

    model: str = "mlx-community/whisper-turbo"
    task: str = "transcribe"
    language: Optional[str] = None
    temperature: float | Sequence[float] = 0.0
    best_of: Optional[int] = 5
    beam_size: Optional[int] = 1
    patience: Optional[float] = 1.0
    length_penalty: Optional[float] = 1.0
    suppress_tokens: str = "-1"
    suppress_numerals: bool = False
    initial_prompt: Optional[str] = None
    hotwords: Optional[str] = None
    condition_on_previous_text: bool = False
    interleaved_context: bool = False
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
    coverage_floor: float = 0.20
    min_silent_run: float = 1.0
    coverage: bool = False
    batch_size: int = 8
    align_model: Optional[str] = None
    no_align: bool = False
    allow_missing_alignment_deps: bool = False
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
    clip_timestamps: str | Sequence[float] | None = None
    device: str = "cpu"
    verbose: bool = False
    print_progress: bool = False
    vad_dump_path: Optional[str] = None


class MLXWhisperXPipeline:
    """Run VAD, MLX Whisper ASR, alignment, and optional diarization."""

    def __init__(self, options: PipelineOptions):
        """Store pipeline options and lazily import the vendored ASR backend."""
        options.language, options.task = normalize_language_settings(
            options.model,
            options.language,
            options.task,
        )
        self.options = options
        self._mlx_whisper = import_mlx_whisper()

    def transcribe(self, audio: str | np.ndarray) -> dict:
        """Execute the full configured pipeline for a path or waveform."""
        if self.options.clip_timestamps is not None and not self.options.no_vad:
            raise ValueError("clip_timestamps requires no_vad=True")
        if not self.options.no_vad and not 0 < self.options.chunk_size <= MAX_CHUNK_SIZE:
            # Chunks are decoded as one encoder window, so a larger chunk would lose
            # everything past the window instead of being transcribed, and a
            # non-positive one would produce no chunks at all.
            raise ValueError(
                f"chunk_size must be between 1 and {MAX_CHUNK_SIZE} seconds, got {self.options.chunk_size}"
            )
        if self.options.batch_size < 1:
            raise ValueError(f"batch_size must be at least 1, got {self.options.batch_size}")
        if (
            self.options.min_speakers is not None
            and self.options.max_speakers is not None
            and self.options.min_speakers > self.options.max_speakers
        ):
            raise ValueError("min_speakers must not exceed max_speakers")

        audio_np = audio_to_numpy(audio)

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
            try:
                result = self._align(result, audio_np)
            except AlignmentDependencyError:
                if not self.options.allow_missing_alignment_deps:
                    raise
                logger.warning(
                    "Alignment dependencies are unavailable; continuing without forced alignment "
                    "because allow_missing_alignment_deps=True."
                )

        if self.options.diarize:
            result = self._diarize(result, audio_np)

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
            "word_segments": [
                word for segment in normalized_segments for word in segment.get("words", [])
            ],
            "language": result.get("language") or self.options.language or "en",
        }
        if include_speakers and "speaker_embeddings" in result:
            normalized["speaker_embeddings"] = result["speaker_embeddings"]
        return normalized

    def _vad_chunks(self, audio: np.ndarray) -> list[dict]:
        """Run VAD and return either speech-only or boundary-preserving chunks."""
        duration = len(audio) / SAMPLE_RATE
        if self.options.no_vad:
            # Keep the downstream ASR path identical by representing full-file ASR as
            # a single VAD chunk instead of adding separate no-VAD transcription code.
            chunks = [{"start": 0.0, "end": duration, "segments": [(0.0, duration)]}]
            self._dump_vad_chunks(chunks, duration)
            return chunks

        resolved_vad_method, vad_model = self._load_vad_model()

        # Silero accepts NumPy directly; pyannote expects a Torch waveform. The
        # backend owns conversion so this pipeline can stay backend-agnostic.
        waveform = vad_model.preprocess_audio(audio)
        merge_chunks = vad_model.merge_chunks
        vad_segments = vad_model({"waveform": waveform, "sample_rate": SAMPLE_RATE})

        def vad_only_chunks() -> list[dict]:
            return merge_chunks(
                vad_segments,
                self.options.chunk_size,
                onset=self.options.vad_onset,
                offset=self.options.vad_offset,
            )

        if self.options.coverage:
            chunks = self._coverage_chunks(audio, vad_segments, vad_only_chunks)
        else:
            chunks = vad_only_chunks()

        chunks = [chunk for chunk in chunks if chunk["end"] > chunk["start"]]
        self._report_coverage(chunks, duration)
        self._dump_vad_chunks(chunks, duration, resolved_vad_method=resolved_vad_method)
        return chunks

    def _coverage_chunks(self, audio: np.ndarray, vad_segments, vad_only_chunks) -> list[dict]:
        """Plan chunks from audible energy, placing seams in energy dips.

        Speech the VAD misses is otherwise dropped silently and unrecoverably,
        so energy decides what is covered and sustained silence is the only
        thing allowed to skip audio. Seams come from the same signal, since a
        VAD that missed a passage cannot say where its pauses are.
        """
        level = speech_level(audio, vad_segments)
        if level <= 0:
            # No detected speech to measure against, so there is no scale for an
            # energy floor. This is silence or a total VAD failure; either way
            # the VAD's own answer is the only one available.
            return vad_only_chunks()

        floor = level * self.options.coverage_floor
        regions = audible_regions(
            audio,
            floor=floor,
            min_silent_run=self.options.min_silent_run,
        )
        return Vad.plan_chunks(
            regions,
            self.options.chunk_size,
            cut_hints=vad_segments,
            pauses=energy_pauses(audio, floor),
        )

    def _report_coverage(self, chunks: list[dict], duration: float) -> float:
        """Log the fraction of the file that will reach the decoder.

        Dropped audio is otherwise invisible: it produces no output, no warning,
        and no diagnostic, so a VAD regression looks identical to silence.
        """
        covered = covered_seconds((chunk["start"], chunk["end"]) for chunk in chunks)
        coverage = covered / duration if duration > 0 else 0.0
        message = "Coverage: %.1f%% of %.0fs in %d chunks"
        if coverage < LOW_COVERAGE_WARNING:
            logger.warning(message + " -- most of the audio will not be transcribed",
                           coverage * 100, duration, len(chunks))
        else:
            logger.info(message, coverage * 100, duration, len(chunks))
        return coverage

    def _load_vad_model(self) -> tuple[str, object]:
        """Instantiate the configured VAD backend, applying default fallback once."""
        if self.options.vad_method != AUTO_VAD_METHOD:
            return self.options.vad_method, self._build_vad_model(self.options.vad_method)

        try:
            return "pyannote", self._build_vad_model("pyannote")
        except RuntimeError as exc:
            logger.warning("Pyannote VAD unavailable; falling back to silero: %s", exc)
            return "silero", self._build_vad_model("silero")

    def _build_vad_model(self, vad_method: str):
        """Create a concrete VAD backend instance for the requested backend name."""
        VadClass = get_vad_class(vad_method)
        if vad_method == "pyannote":
            # Pyannote has a different constructor contract because it needs a Torch
            # device, optional auth token, and model cache settings.
            return VadClass(
                self.options.device,
                token=self.options.hf_token,
                model_name=self.options.vad_model,
                cache_dir=self.options.model_dir,
                model_cache_only=self.options.model_cache_only,
                vad_onset=self.options.vad_onset,
                vad_offset=self.options.vad_offset,
                chunk_size=self.options.chunk_size,
            )
        return VadClass(
            vad_onset=self.options.vad_onset,
            vad_offset=self.options.vad_offset,
            chunk_size=self.options.chunk_size,
        )

    def _dump_vad_chunks(
        self,
        chunks: list[dict],
        audio_duration: float,
        resolved_vad_method: Optional[str] = None,
    ) -> None:
        """Write VAD diagnostics to JSON when requested by the caller."""
        if not self.options.vad_dump_path:
            return

        covered = covered_seconds((chunk["start"], chunk["end"]) for chunk in chunks)
        payload = {
            "vad_method": resolved_vad_method or self.options.vad_method,
            "requested_vad_method": self.options.vad_method,
            "resolved_vad_method": resolved_vad_method,
            "vad_onset": self.options.vad_onset,
            "vad_offset": self.options.vad_offset,
            "vad_model": self.options.vad_model,
            "chunk_size": self.options.chunk_size,
            "no_vad": self.options.no_vad,
            "audio_duration": round(audio_duration, 3),
            "chunk_count": len(chunks),
            "covered_seconds": round(covered, 3),
            "coverage": round(covered / audio_duration, 4) if audio_duration > 0 else 0.0,
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

    def _build_asr_kwargs(self, *, without_timestamps: bool) -> dict:
        """Assemble backend decode kwargs shared by the VAD and window ASR paths."""
        suppress_tokens: str | list[int] = self.options.suppress_tokens
        if self.options.suppress_numerals:
            logger.info("Suppressing numeral and symbol tokens")
            suppress_tokens = _merge_suppress_tokens(
                self.options.suppress_tokens,
                _find_numeral_symbol_tokens(
                    self.options.language,
                    self.options.task,
                    not is_english_only_model(self.options.model),
                ),
            )

        prompt = self.options.initial_prompt
        if self.options.hotwords:
            # The vendored backend accepts a single initial prompt string, so hotwords
            # are appended rather than passed through a separate API.
            prompt = f"{prompt or ''} {self.options.hotwords}".strip()
        language = self.options.language
        decode_kwargs = {
            "language": language,
            "task": self.options.task,
            "best_of": self.options.best_of,
            "beam_size": self.options.beam_size,
            "patience": self.options.patience,
            "length_penalty": self.options.length_penalty,
            "suppress_tokens": suppress_tokens,
            "fp16": self.options.fp16,
            "without_timestamps": without_timestamps,
        }
        # None-valued options should fall back to backend defaults rather than
        # overriding them with null values.
        decode_kwargs = {key: value for key, value in decode_kwargs.items() if value is not None}
        backend_verbose = None
        if self.options.no_vad and (self.options.print_progress or self.options.verbose):
            # The full-file backend seek loop can run for a long time with no outer
            # chunk boundaries, so enable the backend progress bar in no-VAD mode.
            backend_verbose = False
        return {
            "path_or_hf_repo": self.options.model,
            "model_dir": self.options.model_dir,
            "model_cache_only": self.options.model_cache_only,
            "verbose": backend_verbose,
            "temperature": self.options.temperature,
            "compression_ratio_threshold": self.options.compression_ratio_threshold,
            "logprob_threshold": self.options.logprob_threshold,
            # Forward explicitly so serial and batched chunk decoding apply the same
            # silence rule instead of each falling back to a different default.
            "no_speech_threshold": self.options.no_speech_threshold,
            "initial_prompt": prompt,
            **decode_kwargs,
        }

    def _asr(self, audio: np.ndarray, vad_chunks: list[dict]) -> dict:
        """Transcribe each VAD chunk and restore chunk-local timestamps to file time."""
        # VAD chunk decoding does not need backend timestamp tokens because forced
        # alignment will refine the chunk text later. Full-file no-VAD decoding needs
        # backend segment timestamps so alignment/subtitle timing is anchored to tighter
        # spans instead of whole 30-second windows.
        asr_kwargs = self._build_asr_kwargs(without_timestamps=not self.options.no_vad)

        if self.options.no_vad:
            if self.options.interleaved_context:
                logger.warning(
                    "interleaved_context has no effect with no_vad; "
                    "use condition_on_previous_text to carry text between windows"
                )
            return self._asr_without_vad(audio, vad_chunks, asr_kwargs)
        return self._asr_with_vad(audio, vad_chunks, asr_kwargs)

    def _asr_without_vad(self, audio: np.ndarray, vad_chunks: list[dict], asr_kwargs: dict) -> dict:
        """Keep the existing backend seek-loop path for explicit no-VAD transcription."""
        all_segments: list[dict] = []
        language = self.options.language
        detected_language = language
        total = len(vad_chunks)
        backend_transcribe = self._mlx_whisper.transcribe
        if not callable(backend_transcribe):
            # Importing the vendored backend submodule can shadow the package-level
            # wrapper with the module object; resolve the real callable here.
            backend_transcribe = backend_transcribe.transcribe

        for idx, chunk in enumerate(vad_chunks):
            start = float(chunk["start"])
            end = float(chunk["end"])
            chunk_audio = slice_audio(audio, start, end)
            if chunk_audio.size == 0:
                continue

            backend_kwargs = {
                "condition_on_previous_text": self.options.condition_on_previous_text,
                "word_timestamps": False,
                **asr_kwargs,
            }
            if self.options.clip_timestamps is not None:
                backend_kwargs["clip_timestamps"] = self.options.clip_timestamps

            chunk_result = backend_transcribe(chunk_audio, **backend_kwargs)

            detected_language = detected_language or chunk_result.get("language")
            language = language or detected_language

            chunk_segments = chunk_result.get("segments", [])
            if not chunk_segments and chunk_result.get("text", "").strip():
                # Some backend paths can return only aggregate text. Synthesize a
                # chunk-local segment so the downstream output schema remains uniform.
                chunk_segments = [{"start": 0.0, "end": end - start, "text": chunk_result["text"]}]

            self._append_asr_segments(all_segments, chunk_segments, start, end)

            if self.options.print_progress:
                print(f"Progress: {((idx + 1) / total) * 50:.2f}%...")

        return {"segments": all_segments, "language": detected_language or language or "en"}

    def _asr_with_vad(self, audio: np.ndarray, vad_chunks: list[dict], asr_kwargs: dict) -> dict:
        """Decode each VAD chunk directly once, matching WhisperX's ASR flow."""
        all_segments: list[dict] = []
        language = self.options.language
        if language is None:
            language = self._mlx_whisper.detect_language(
                self._language_detection_audio(audio, vad_chunks),
                path_or_hf_repo=self.options.model,
                model_dir=self.options.model_dir,
                model_cache_only=self.options.model_cache_only,
                verbose=None,
                fp16=self.options.fp16,
            )

        chunk_kwargs = {**asr_kwargs, "language": language}

        if self.options.condition_on_previous_text:
            # Each VAD chunk is decoded as one window, so there is no earlier window
            # inside it to condition on; turning batching off for it would only be slower.
            logger.warning(
                "condition_on_previous_text has no effect with VAD; "
                "use interleaved_context to carry text across VAD chunks"
            )
        if self.options.interleaved_context:
            return self._asr_with_vad_interleaved(audio, vad_chunks, chunk_kwargs, language)
        # Batching decodes multiple independent VAD chunks in one pass.
        if self.options.batch_size > 1:
            return self._asr_with_vad_batched(audio, vad_chunks, chunk_kwargs, language)

        total = len(vad_chunks)
        for idx, chunk in enumerate(vad_chunks):
            start = float(chunk["start"])
            end = float(chunk["end"])
            speech_segments = [
                (float(seg_start) - start, float(seg_end) - start)
                for seg_start, seg_end in chunk.get("segments", [])
            ]
            chunk_audio = self._speech_only_chunk_audio(audio, chunk)
            if chunk_audio.size == 0:
                continue

            chunk_result = self._mlx_whisper.transcribe_chunk(chunk_audio, **chunk_kwargs)
            chunk_segments = []
            if self._keep_chunk_text(chunk_result, start):
                chunk_segments = [
                    self._restore_vad_chunk_segment(chunk_result, speech_segments, end - start)
                ]
            self._append_asr_segments(all_segments, chunk_segments, start, end)

            if self.options.print_progress:
                print(f"Progress: {((idx + 1) / total) * 50:.2f}%...")

        return {"segments": all_segments, "language": language or "en"}

    def _language_detection_audio(self, audio: np.ndarray, vad_chunks: list[dict]) -> np.ndarray:
        """Return up to one decoder window of detected speech for language detection.

        Detection only sees the first 30 seconds of what it is given. Feeding it the
        raw file lets an intro of music or silence decide the language; feeding it the
        VAD speech keeps the decision on actual dialogue.
        """
        window = MAX_CHUNK_SIZE * SAMPLE_RATE
        pieces: list[np.ndarray] = []
        collected = 0
        for chunk in vad_chunks:
            piece = self._speech_only_chunk_audio(audio, chunk)
            pieces.append(piece)
            collected += piece.size
            if collected >= window:
                break
        speech = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
        return speech[:window] if speech.size else audio[:window]

    def _asr_with_vad_batched(
        self,
        audio: np.ndarray,
        vad_chunks: list[dict],
        chunk_kwargs: dict,
        language: Optional[str],
    ) -> dict:
        """Decode VAD chunks in batches and restore chunk-local timestamps to file time."""
        all_segments: list[dict] = []
        prepared, audio_list = self._prepare_vad_chunks(audio, vad_chunks)
        if not audio_list:
            return {"segments": all_segments, "language": language or "en"}

        logger.info(
            "VAD ASR: %d chunks, batch_size=%d", len(audio_list), self.options.batch_size
        )
        # transcribe_chunk accepts `verbose`, the batched entry point does not.
        batched_kwargs = {key: value for key, value in chunk_kwargs.items() if key != "verbose"}
        chunk_results = self._mlx_whisper.transcribe_chunks_batched(
            audio_list,
            batch_size=self.options.batch_size,
            progress_callback=self._make_batch_progress_callback("VAD ASR"),
            **batched_kwargs,
        )
        self._emit_vad_results(all_segments, prepared, chunk_results)
        return {"segments": all_segments, "language": language or "en"}

    def _asr_with_vad_interleaved(
        self,
        audio: np.ndarray,
        vad_chunks: list[dict],
        chunk_kwargs: dict,
        language: Optional[str],
    ) -> dict:
        """Decode VAD chunks in batches while each chunk is prompted with the text before it.

        Chunks are split into `batch_size` contiguous streams and batch k holds chunk k of
        every stream, so each chunk can be prompted with what its own stream decoded so far
        while the batch still decodes in one pass. The first chunk of each later stream has
        no history in that pass; it is decoded again at the end, prompted with the tail of
        the stream before it, which is the audio immediately preceding it.
        """
        all_segments: list[dict] = []
        prepared, audio_list = self._prepare_vad_chunks(audio, vad_chunks)
        if not audio_list:
            return {"segments": all_segments, "language": language or "en"}

        n_chunks = len(audio_list)
        n_streams = min(self.options.batch_size, n_chunks)
        if n_streams > 1 and n_chunks < 2 * n_streams:
            logger.warning(
                "Interleaved context: %d chunks over %d streams gives some streams a single "
                "chunk, so little text carries over; a smaller batch_size gives longer histories",
                n_chunks,
                n_streams,
            )
        streams = _split_streams(n_chunks, n_streams)
        batches = _interleave_streams(streams)
        logger.info("VAD ASR: %d chunks, %d interleaved streams", n_chunks, n_streams)

        batched_kwargs = {key: value for key, value in chunk_kwargs.items() if key != "verbose"}
        progress = self._make_batch_progress_callback("VAD ASR")
        contexts: list[list[int]] = [[] for _ in streams]
        chunk_results: list[Optional[dict]] = [None] * n_chunks
        windows_done = 0
        for batch_index, batch in enumerate(batches):
            batch_results = self._mlx_whisper.transcribe_chunks_batched(
                [audio_list[chunk] for _, chunk in batch],
                batch_size=len(batch),
                context_tokens=[contexts[stream] for stream, _ in batch],
                progress_callback=_rebase_progress(
                    progress,
                    batch_index=batch_index,
                    total_batches=len(batches),
                    windows_before=windows_done,
                    total_windows=n_chunks,
                ),
                **batched_kwargs,
            )
            windows_done += len(batch)
            for (stream, chunk), chunk_result in zip(batch, batch_results):
                chunk_results[chunk] = chunk_result
                contexts[stream] = self._next_stream_context(contexts[stream], chunk_result)

        # Stream 0 starts the file, so its first chunk has no earlier text to gain.
        redo = [
            (streams[stream][0], contexts[stream - 1])
            for stream in range(1, n_streams)
            if contexts[stream - 1]
        ]
        if redo:
            redo_results = self._mlx_whisper.transcribe_chunks_batched(
                [audio_list[chunk] for chunk, _ in redo],
                batch_size=len(redo),
                context_tokens=[context for _, context in redo],
                progress_callback=self._make_batch_progress_callback("VAD ASR wrap-around"),
                **batched_kwargs,
            )
            for (chunk, _), chunk_result in zip(redo, redo_results):
                start = prepared[chunk]["start"]
                # The redo only refines text. Prompting can lower no_speech_prob, so a
                # chunk the first pass judged silent must stay that way.
                if self._keep_chunk_text(chunk_results[chunk], start, quiet=True) and self._keep_chunk_text(
                    chunk_result, start, quiet=True
                ):
                    chunk_results[chunk] = chunk_result

        self._emit_vad_results(all_segments, prepared, chunk_results)
        return {"segments": all_segments, "language": language or "en"}

    def _next_stream_context(self, context: list[int], chunk_result: dict) -> list[int]:
        """Update one stream's history after a chunk decodes, following Whisper's rules.

        Text from a hot-temperature retry or a repetition loop is not trusted as history,
        so the stream starts over. A chunk judged silent adds nothing but keeps what came
        before, as Whisper's own seek loop does for silent windows.
        """
        temperature = chunk_result.get("temperature") or 0.0
        if temperature > CONTEXT_RESET_TEMPERATURE or is_repetition_loop(chunk_result.get("text", "")):
            return []
        if not self._keep_chunk_text(chunk_result, 0.0, quiet=True):
            return context
        return _extend_context(context, chunk_result.get("tokens", []), CONTEXT_TOKEN_LIMIT)

    def _prepare_vad_chunks(self, audio: np.ndarray, vad_chunks: list[dict]) -> tuple[list[dict], list[np.ndarray]]:
        """Compact each VAD chunk to its speech and keep what is needed to map times back."""
        prepared: list[dict] = []
        audio_list: list[np.ndarray] = []
        for chunk in vad_chunks:
            start = float(chunk["start"])
            end = float(chunk["end"])
            chunk_audio = self._speech_only_chunk_audio(audio, chunk)
            if chunk_audio.size == 0:
                continue
            speech_segments = [
                (float(seg_start) - start, float(seg_end) - start)
                for seg_start, seg_end in chunk.get("segments", [])
            ]
            prepared.append({"start": start, "end": end, "speech_segments": speech_segments})
            audio_list.append(chunk_audio)
        return prepared, audio_list

    def _emit_vad_results(self, all_segments: list[dict], prepared: list[dict], chunk_results: list[dict]) -> None:
        """Append decoded chunks as file-time segments, skipping text that is rejected."""
        total = len(chunk_results)
        for idx, (meta, chunk_result) in enumerate(zip(prepared, chunk_results)):
            chunk_segments = []
            if self._keep_chunk_text(chunk_result, meta["start"]):
                chunk_segments = [
                    self._restore_vad_chunk_segment(
                        chunk_result,
                        meta["speech_segments"],
                        meta["end"] - meta["start"],
                    )
                ]
            self._append_asr_segments(all_segments, chunk_segments, meta["start"], meta["end"])

            if self.options.print_progress:
                print(f"Progress: {((idx + 1) / total) * 50:.2f}%...")

    def _make_batch_progress_callback(self, stage: str):
        """Build a callback that logs per-batch decode timing and throughput."""
        if not (self.options.print_progress or self.options.verbose):
            return None

        def _callback(info: dict) -> None:
            pct = (info["windows_done"] / info["total_windows"] * 100) if info["total_windows"] else 0.0
            logger.info(
                "%s: batch %d/%d | %d/%d windows (%.1f%%) | decode %.2fs, words %.2fs | %.1fx realtime",
                stage,
                info["batch_index"] + 1,
                info["total_batches"],
                info["windows_done"],
                info["total_windows"],
                pct,
                info["decode_seconds"],
                info["words_seconds"],
                info["realtime_factor"],
            )

        return _callback

    def _keep_chunk_text(self, chunk_result: dict, start: float, *, quiet: bool = False) -> bool:
        """Decide whether a decoded chunk's text should be emitted at all.

        The decoder's own quality signals gate only the temperature retry, never
        the output, so a chunk can be recognised as non-speech and still have its
        text emitted. Worse, a high `no_speech_prob` actively *suppresses* the
        retry (silence being a valid outcome), which is exactly the case that
        needs rejecting: music scored as non-speech, decoded into a repetition
        loop, and published because the text was non-empty.
        """
        text = chunk_result.get("text", "").strip()
        if not text:
            return False

        reason = None
        no_speech = chunk_result.get("no_speech_prob")
        threshold = self.options.no_speech_threshold
        avg_logprob = chunk_result.get("avg_logprob")
        logprob_threshold = self.options.logprob_threshold
        # Match Whisper's silence rule: a high no-speech score alone is not enough,
        # because quiet or stylised speech that decoded confidently is still speech.
        confident = (
            logprob_threshold is not None
            and avg_logprob is not None
            and avg_logprob > logprob_threshold
        )
        if threshold is not None and no_speech is not None and no_speech > threshold and not confident:
            reason = f"no_speech_prob={no_speech:.2f} > {threshold}"
        elif is_repetition_loop(text):
            reason = "repetition loop"

        if reason is None:
            return True
        if quiet:
            # Bookkeeping checks run before the final emit, which logs the drop once.
            return False

        logger.warning(
            "Dropping chunk at %02d:%05.2f (%s): %.60s",
            int(start) // 60,
            start % 60,
            reason,
            text,
        )
        return False

    def _speech_only_chunk_audio(self, audio: np.ndarray, chunk: dict) -> np.ndarray:
        """Concatenate only the voiced regions inside one VAD chunk."""
        speech_segments = chunk.get("segments", [])
        if not speech_segments:
            return slice_audio(audio, float(chunk["start"]), float(chunk["end"]))

        pieces = [
            slice_audio(audio, float(seg_start), float(seg_end))
            for seg_start, seg_end in speech_segments
        ]
        pieces = [piece for piece in pieces if piece.size > 0]
        if not pieces:
            return np.array([], dtype=np.float32)
        return np.concatenate(pieces)

    def _restore_vad_chunk_segment(
        self,
        segment: dict,
        speech_segments: list[tuple[float, float]],
        chunk_duration: float,
    ) -> dict:
        """Map compacted speech-only timestamps back onto the original chunk timeline."""
        if not speech_segments:
            return segment

        restored = dict(segment)
        restored["start"] = round(
            self._restore_speech_time(float(segment.get("start", 0.0)), speech_segments, is_end=False),
            3,
        )
        restored["end"] = round(
            self._restore_speech_time(
                float(segment.get("end", chunk_duration)),
                speech_segments,
                is_end=True,
            ),
            3,
        )
        return restored

    def _restore_speech_time(
        self,
        time_in_speech: float,
        speech_segments: list[tuple[float, float]],
        *,
        is_end: bool,
    ) -> float:
        """Convert compacted speech time to the equivalent time in the chunk timeline."""
        if not speech_segments:
            return time_in_speech

        remaining = max(0.0, time_in_speech)
        last_end = float(speech_segments[-1][1])
        for index, (seg_start, seg_end) in enumerate(speech_segments):
            seg_start = float(seg_start)
            seg_end = float(seg_end)
            duration = max(0.0, seg_end - seg_start)
            if remaining < duration:
                return seg_start + remaining
            if abs(remaining - duration) <= 1e-9:
                if is_end or index == len(speech_segments) - 1:
                    return seg_end
                return float(speech_segments[index + 1][0])
            remaining -= duration

        return last_end

    def _append_asr_segments(
        self,
        all_segments: list[dict],
        chunk_segments: list[dict],
        chunk_start: float,
        chunk_end: float,
    ) -> None:
        """Copy chunk-local ASR segments into absolute file time."""
        for segment in chunk_segments:
            text = segment.get("text", "")
            if not text.strip():
                continue
            asr_segment = {
                # Backend segment times are relative to the sliced chunk. Add the VAD
                # chunk start so alignment/diarization see absolute file time.
                "start": round(chunk_start + float(segment.get("start", 0.0)), 3),
                "end": round(chunk_start + float(segment.get("end", chunk_end - chunk_start)), 3),
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

    def _diarize(self, result: dict, audio: np.ndarray) -> dict:
        """Run pyannote diarization and attach dominant speaker labels to words."""
        logger.info("Performing diarization...")
        if self.options.hf_token is None:
            logger.warning("No hf_token provided; pyannote diarization may fail for gated models")
        diarize_model = DiarizationPipeline(
            model_name=self.options.diarize_model,
            token=self.options.hf_token,
            device=self.options.device,
            cache_dir=self.options.model_dir,
            model_cache_only=self.options.model_cache_only,
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
