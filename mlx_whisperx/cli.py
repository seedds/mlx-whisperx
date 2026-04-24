import argparse
import os
import pathlib
import traceback
import warnings

import numpy as np

from .log_utils import setup_logging
from .transcribe import transcribe
from .writers import get_writer


def optional_int(value: str):
    return None if value == "None" else int(value)


def optional_float(value: str):
    return None if value == "None" else float(value)


def str2bool(value: str) -> bool:
    values = {"True": True, "False": False, "true": True, "false": False}
    if value not in values:
        raise ValueError(f"Expected one of {set(values)}, got {value}")
    return values[value]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("audio", nargs="+", type=str, help="Audio file(s) to transcribe")
    parser.add_argument("--model", default="mlx-community/whisper-turbo", help="mlx-whisper model directory or Hugging Face repo")
    parser.add_argument("--model_dir", default=None, help="Directory for alignment/diarization model cache")
    parser.add_argument("--model_cache_only", type=str2bool, default=False, help="Use cached alignment models only")
    parser.add_argument("--device", default="cpu", help="Torch device for VAD/alignment/diarization stages")
    parser.add_argument("--batch_size", default=8, type=int, help="Accepted for WhisperX CLI parity; ASR currently runs chunks serially")
    parser.add_argument("--compute_type", default="default", choices=["default", "float16", "float32"], help="Controls fp16 passed to mlx-whisper")

    parser.add_argument("--output_dir", "-o", default=".", help="Directory to save outputs")
    parser.add_argument("--output_name", default=None, help="Output basename. Defaults to input file stem")
    parser.add_argument("--output_format", "-f", default="all", choices=["all", "srt", "vtt", "txt", "tsv", "json", "aud"], help="Output format")
    parser.add_argument("--verbose", type=str2bool, default=True, help="Print transcript/logging")
    parser.add_argument("--log-level", default=None, choices=["debug", "info", "warning", "error", "critical"], help="Logging level")

    parser.add_argument("--task", default="transcribe", choices=["transcribe", "translate"], help="Speech recognition or translation")
    parser.add_argument("--language", default=None, help="Language code. Defaults to auto-detect")

    parser.add_argument("--align_model", default=None, help="Alignment model name")
    parser.add_argument("--interpolate_method", default="nearest", choices=["nearest", "linear", "ignore"], help="Timestamp interpolation for unaligned words")
    parser.add_argument("--no_align", action="store_true", help="Skip forced alignment")
    parser.add_argument("--return_char_alignments", action="store_true", help="Return character alignments in JSON")

    parser.add_argument("--vad_method", default="silero", choices=["pyannote", "silero"], help="VAD backend")
    parser.add_argument("--vad_onset", type=float, default=0.500, help="VAD onset threshold")
    parser.add_argument("--vad_offset", type=float, default=0.363, help="VAD offset threshold")
    parser.add_argument("--chunk_size", type=int, default=30, help="Merged VAD chunk size in seconds")
    parser.add_argument("--no_vad", action="store_true", help="Skip VAD and transcribe the full file as one chunk")
    parser.add_argument("--vad_dump_path", default=None, help="Write VAD chunks and settings to this JSON path")

    parser.add_argument("--diarize", action="store_true", help="Assign speaker labels")
    parser.add_argument("--min_speakers", default=None, type=int, help="Minimum number of speakers")
    parser.add_argument("--max_speakers", default=None, type=int, help="Maximum number of speakers")
    parser.add_argument("--diarize_model", default="pyannote/speaker-diarization-community-1", help="Pyannote diarization model")
    parser.add_argument("--speaker_embeddings", action="store_true", help="Include speaker embeddings in JSON output")
    parser.add_argument("--hf_token", default=None, help="Hugging Face token for gated pyannote models")

    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--temperature_increment_on_fallback", type=optional_float, default=None, help="Temperature fallback increment")
    parser.add_argument("--best_of", type=optional_int, default=5, help="Number of candidates when sampling")
    parser.add_argument("--beam_size", type=optional_int, default=5, help="Beam size when temperature is zero")
    parser.add_argument("--patience", type=optional_float, default=1.0, help="Beam-search patience")
    parser.add_argument("--length_penalty", type=optional_float, default=1.0, help="Length penalty")
    parser.add_argument("--suppress_tokens", default="-1", help="Comma-separated token IDs to suppress")
    parser.add_argument("--suppress_numerals", action="store_true", help="Suppress numeric and currency-symbol tokens during decoding")
    parser.add_argument("--initial_prompt", default=None, help="Initial prompt")
    parser.add_argument("--hotwords", default=None, help="Hint phrases appended to the initial prompt")
    parser.add_argument("--condition_on_previous_text", type=str2bool, default=False, help="Prompt each chunk with previous text")
    parser.add_argument("--fp16", type=str2bool, default=True, help="Use fp16 in mlx-whisper")
    parser.add_argument("--compression_ratio_threshold", type=optional_float, default=2.4, help="Repetition failure threshold")
    parser.add_argument("--logprob_threshold", type=optional_float, default=-1.0, help="Low confidence failure threshold")
    parser.add_argument("--no_speech_threshold", type=optional_float, default=0.6, help="No-speech threshold")

    parser.add_argument("--max_line_width", type=optional_int, default=None, help="Subtitle line width option")
    parser.add_argument("--max_line_count", type=optional_int, default=None, help="Subtitle line count option")
    parser.add_argument("--max_words_per_line", type=optional_int, default=None, help="Subtitle words per cue")
    parser.add_argument("--highlight_words", type=str2bool, default=False, help="Highlight words in SRT/VTT")
    parser.add_argument("--segment_resolution", default="sentence", choices=["sentence", "chunk"], help="Accepted for parity; alignment currently returns sentence segments")
    parser.add_argument("--print_progress", type=str2bool, default=False, help="Print stage progress")
    return parser


def main() -> None:
    parser = build_parser()
    args = vars(parser.parse_args())

    log_level = args.pop("log_level")
    verbose = args.get("verbose")
    setup_logging(log_level or ("info" if verbose else "warning"))

    output_dir = args.pop("output_dir")
    output_format = args.pop("output_format")
    output_name = args.pop("output_name")
    os.makedirs(output_dir, exist_ok=True)
    writer = get_writer(output_format, output_dir)

    writer_args = {
        "highlight_words": args.pop("highlight_words"),
        "max_line_count": args.pop("max_line_count"),
        "max_line_width": args.pop("max_line_width"),
        "max_words_per_line": args.pop("max_words_per_line"),
    }
    args.pop("segment_resolution")

    if writer_args["max_line_count"] and not writer_args["max_line_width"]:
        warnings.warn("--max_line_count has no effect without --max_line_width")

    compute_type = args.pop("compute_type")
    if compute_type == "float32":
        args["fp16"] = False

    increment = args.pop("temperature_increment_on_fallback")
    if increment is not None:
        args["temperature"] = tuple(np.arange(args["temperature"], 1.0 + 1e-6, increment))

    audio_files = args.pop("audio")
    for audio_path in audio_files:
        name = output_name or pathlib.Path(audio_path).stem
        try:
            result = transcribe(audio_path, **args)
            writer(result, name, writer_args)
        except Exception as exc:
            traceback.print_exc()
            print(f"Skipping {audio_path} due to {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
